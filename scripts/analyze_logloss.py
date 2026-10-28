"""Razčlenitev logaritemske izgube: koliko prispevajo napovedi z verjetnostjo 0 za pravi razred.

Logaritemska izguba enega testnega primera je -log p, kjer je p verjetnost, ki jo
model da pravemu razredu. sklearn.metrics.log_loss (1.9) p pred logaritmom omeji
na [eps, 1 - eps] (eps = 2,2e-16 pri float64) in ne normira ponovno, zato
primer s p < eps prispeva natanko -log(eps) = 36,04, vsak drug pa -log p.
Izguba učenja je povprečje teh prispevkov, torej se razdeli na dva seštevka:

  izguba = (n_omejenih * 36,04) / n  +  (vsota -log p ostalih primerov) / n.

Skript to razčlenitev izračuna za vsako učenje iz shranjenih napovedi
(predictions/*.npz, isto branje kot scripts/compute_metrics.py: float32 ->
float64, vrstica normirana na 1) in jo povpreči tako kot povzetki diplome:
najprej čez 15 učenj nabora, nato čez nabore. Poleg tega izračuna, kolikšno
verjetnost model v povprečju da pravemu razredu, kadar ima ta največjo
verjetnost (pravilno razvrščeni primeri) - mera samozavesti pri pravilnih
napovedih.

Vprašanje iz diplome: zakaj ima naključni gozd pri podobni točnosti precej
višjo logaritemsko izgubo od ostalih ansamblov? Njegova verjetnost je povprečje
verjetnosti dreves, verjetnost drevesa pa delež razreda v listu (dokumentacija
scikit-learn), zato je pravi razred lahko dobil točno 0.

Kontrola: izguba in točnost vsakega učenja morata biti enaki stolpcema v
metrics.csv (do 1e-9); sicer skript konča z izhodno kodo 1.

Zagon (iz korena repozitorija; napovedi morajo biti na tem računalniku):
    python scripts/analyze_logloss.py cc18_v4/combined
    python scripts/analyze_logloss.py medic3_160_v2/combined
Izhod: <zagon>/summary/log_loss/logloss_split.csv (po algoritmih),
logloss_split_per_dataset.csv, logloss_split_report.txt.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from scripts.compute_metrics import prediction_path  # noqa: E402
from src.config import load_config  # noqa: E402
from src.metrics import full_proba  # noqa: E402
from src.summary import load_results, metric_dir, resolve_results  # noqa: E402

EPS = np.finfo(np.float64).eps
MAX_LOSS = -np.log(EPS)
TOLERANCE = 1e-9
NAMES = {"random_forest": "Naključni gozd", "xgboost": "XGBoost", "lightgbm": "LightGBM",
         "catboost": "CatBoost", "tabpfn": "TabPFN-3.5", "tabicl": "TabICL"}


class Report:
    """Izpis na zaslon in hkrati v logloss_split_report.txt."""

    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text)
        self.lines.append(text)


def split_fit(path, n_classes):
    """Razčlenitev izgube enega učenja iz njegove datoteke .npz."""
    z = np.load(path)
    proba, y_test = z["proba"], z["y_test"].astype(int)
    classes = z["proba_classes"] if "proba_classes" in z.files else np.arange(proba.shape[1])
    full, _ = full_proba(proba, classes, n_classes)
    p_true = full[np.arange(len(y_test)), y_test]
    clipped = p_true < EPS
    row_loss = -np.log(np.clip(p_true, EPS, 1 - EPS))
    correct = np.argmax(full, axis=1) == y_test
    return {
        "n_test": int(len(y_test)),
        "n_clipped": int(clipped.sum()),
        "log_loss": float(row_loss.mean()),
        "loss_from_clipped": float(clipped.sum() * MAX_LOSS / len(y_test)),
        "accuracy": float(correct.mean()),
        "p_true_when_correct": float(p_true[correct].mean()) if correct.any() else np.nan,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", help="run_id, mapa zagona ali pot do results.csv (npr. cc18_v4/combined)")
    args = parser.parse_args()

    config = load_config()
    results_csv, base_dir = resolve_results(args.results, config)
    df = load_results(results_csv)
    df["dataset_id"] = pd.read_csv(results_csv, dtype={"dataset_id": str})["dataset_id"]
    metrics = pd.read_csv(os.path.join(base_dir, "metrics.csv"))
    keys = ["run_id", "dataset", "algorithm", "repeat", "fold"]

    rows = []
    for i, row in df[~df["failed"]].iterrows():
        path = prediction_path(config["results_dir"], row)
        if not os.path.isfile(path):
            raise SystemExit(f"Manjka {os.path.relpath(path, REPO_ROOT)} - priklopi USB disk (CLAUDE.md).")
        rows.append({**{k: row[k] for k in keys}, **split_fit(path, int(row["n_classes"]))})
    fits = pd.DataFrame(rows)

    check = fits.merge(metrics[keys + ["log_loss", "accuracy"]], on=keys, suffixes=("", "_metrics"),
                       validate="one_to_one")
    max_diff = float(max((check["log_loss"] - check["log_loss_metrics"]).abs().max(),
                         (check["accuracy"] - check["accuracy_metrics"]).abs().max()))

    # Povprečje kot v povzetkih: najprej čez učenja nabora, nato čez nabore.
    per_dataset = fits.groupby(["dataset", "algorithm"]).agg(
        n_test=("n_test", "sum"), n_clipped=("n_clipped", "sum"), log_loss=("log_loss", "mean"),
        loss_from_clipped=("loss_from_clipped", "mean"), accuracy=("accuracy", "mean"),
        p_true_when_correct=("p_true_when_correct", "mean")).reset_index()
    per_dataset["loss_rest"] = per_dataset["log_loss"] - per_dataset["loss_from_clipped"]
    overall = per_dataset.groupby("algorithm").agg(
        n_datasets=("dataset", "nunique"), n_test=("n_test", "sum"), n_clipped=("n_clipped", "sum"),
        datasets_with_clipped=("n_clipped", lambda s: int((s > 0).sum())),
        log_loss=("log_loss", "mean"), loss_from_clipped=("loss_from_clipped", "mean"),
        loss_rest=("loss_rest", "mean"), accuracy=("accuracy", "mean"),
        p_true_when_correct=("p_true_when_correct", "mean")).reset_index()
    overall["clipped_row_share"] = overall["n_clipped"] / overall["n_test"]
    overall["clipped_loss_share"] = overall["loss_from_clipped"] / overall["log_loss"]
    overall = overall.sort_values("log_loss").reset_index(drop=True)

    report = Report()
    report(f"Razčlenitev logaritemske izgube: {os.path.relpath(results_csv, REPO_ROOT)}")
    report(f"  učenj {len(fits)}, testnih primerov skupaj {int(fits['n_test'].sum())}; "
           f"-log(eps) = {MAX_LOSS:.2f}")
    report(f"  kontrola proti metrics.csv: največja razlika {max_diff:.2e}")
    report()
    report(f"{'algoritem':15} {'izguba':>7} {'od p=0':>7} {'delež':>6} {'ostalo':>7} "
           f"{'vrstic p=0':>10} {'naborov':>7} {'točnost':>7} {'p prav.':>7}")
    for _, r in overall.iterrows():
        report(f"{NAMES.get(r['algorithm'], r['algorithm']):15} {r['log_loss']:7.4f} "
               f"{r['loss_from_clipped']:7.4f} {100 * r['clipped_loss_share']:5.1f}% {r['loss_rest']:7.4f} "
               f"{100 * r['clipped_row_share']:9.3f}% {r['datasets_with_clipped']:3d}/{r['n_datasets']:<3d} "
               f"{r['accuracy']:7.4f} {r['p_true_when_correct']:7.4f}")
    report()
    report("izguba = povprečna logaritemska izguba čez nabore; od p=0 = njen del iz primerov, ki jim je")
    report("model dal verjetnost pravega razreda pod eps (vsak prispeva -log eps); delež = od p=0 / izguba;")
    report("ostalo = izguba brez teh primerov; vrstic p=0 = delež takih testnih primerov (vsi skupaj);")
    report("naborov = število naborov z vsaj enim takim primerom; p prav. = povprečna verjetnost pravega")
    report("razreda pri pravilno razvrščenih primerih.")

    out_dir = metric_dir(base_dir, "log_loss")
    os.makedirs(out_dir, exist_ok=True)
    overall.to_csv(os.path.join(out_dir, "logloss_split.csv"), index=False)
    per_dataset.to_csv(os.path.join(out_dir, "logloss_split_per_dataset.csv"), index=False)
    with open(os.path.join(out_dir, "logloss_split_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report.lines) + "\n")
    print(f"\nZapisano v {os.path.relpath(out_dir, REPO_ROOT)}/logloss_split*")
    if max_diff > TOLERANCE:
        print(f"NAPAKA: razčlenitev se od metrics.csv razlikuje za več kot {TOLERANCE}.")
        sys.exit(1)


if __name__ == "__main__":
    main()
