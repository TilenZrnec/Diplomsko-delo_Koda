"""Točnost in logaritemska izguba iz shranjenih napovedi - brez ponovnega učenja.

Vsako uspelo učenje je shranilo predict_proba v
results/runs/<run_id>/predictions/<nabor>/<algoritem>_r<ponovitev>_f<fold>.npz.
Ta skript za vsako vrstico results.csv odpre pripadajočo datoteko in izračuna
dodatni metriki (src/metrics.py). Pri sestavljenem naboru (combined/) pove
stolpec run_id vsake vrstice, iz katerega zagona so napovedi.

Kot kontrolo iz iste datoteke ponovno izračuna še ROC-AUC (roc_auc_from_file) in
ga primerja z vrednostjo v results.csv: datoteke hranijo float32, zato se
razlikujeta za največ ~3e-4 (izmerjeno 2026-10-08). Skript konča z izhodno kodo 1,
če kakšnemu uspelemu učenju manjka datoteka ali je razlika večja od TOLERANCE.

Izhod ob results.csv:
    metrics.csv       - ena vrstica na učenje (neuspela učenja: prazne metrike)
    metrics_info.json - commit, različica scikit-learn, kontrole

Nato povzetek in testi za posamezno metriko (v summary/<metrika>/):
    python -m src.summary <zagon> --metric accuracy
    python -m src.stats   <zagon> --metric log_loss [--no-saturated]

Zagon (iz korena repozitorija; napovedi morajo biti na tem računalniku):
    python scripts/compute_metrics.py results/runs/cc18_v4/combined
"""

import argparse
import datetime
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

from src.config import load_config  # noqa: E402
from src.metrics import compute_accuracy, compute_log_loss  # noqa: E402
from src.summary import load_results, resolve_results  # noqa: E402
from src.utils import compute_roc_auc  # noqa: E402

TOLERANCE = 1e-3
KEYS = ["run_id", "dataset", "dataset_id", "algorithm", "repeat", "fold"]


def prediction_path(results_dir, row):
    return os.path.join(results_dir, row["run_id"], "predictions", str(row["dataset_id"]),
                        f"{row['algorithm']}_r{int(row['repeat'])}_f{int(row['fold'])}.npz")


def score_fit(path, n_classes):
    """Metrike enega učenja iz njegove datoteke .npz."""
    z = np.load(path)
    proba, y_test = z["proba"], z["y_test"]
    # Datoteke pred uvedbo proba_classes (cc18_v2) imajo stolpce v vrstnem redu razredov.
    classes = z["proba_classes"] if "proba_classes" in z.files else np.arange(proba.shape[1])
    loss, row_sum_error = compute_log_loss(y_test, proba, classes, n_classes)
    return {
        "roc_auc_from_file": compute_roc_auc(y_test, proba, classes),
        "accuracy": compute_accuracy(y_test, proba, classes),
        "log_loss": loss,
        "row_sum_error": row_sum_error,
    }


def git_state():
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True).stdout.strip()
    return git("rev-parse", "--short", "HEAD"), bool(git("status", "--porcelain", "--", "src", "scripts"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", help="run_id, mapa zagona ali pot do results.csv")
    args = parser.parse_args()

    config = load_config()
    results_csv, base_dir = resolve_results(args.results, config)
    print(f"Vir: {os.path.relpath(results_csv, REPO_ROOT)}")
    df = load_results(results_csv)
    df["dataset_id"] = pd.read_csv(results_csv, dtype={"dataset_id": str})["dataset_id"]

    rows, missing = [], []
    for i, row in df.iterrows():
        out = {k: row[k] for k in KEYS}
        out["n_classes"] = int(row["n_classes"])
        out["roc_auc"] = row["roc_auc"]
        if not row["failed"]:
            path = prediction_path(config["results_dir"], row)
            if os.path.isfile(path):
                out.update(score_fit(path, int(row["n_classes"])))
            else:
                missing.append(os.path.relpath(path, REPO_ROOT))
        rows.append(out)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(df)} učenj")

    metrics = pd.DataFrame(rows)
    for col in ["roc_auc_from_file", "accuracy", "log_loss", "row_sum_error"]:
        if col not in metrics:
            metrics[col] = np.nan
    diff = (metrics["roc_auc"] - metrics["roc_auc_from_file"]).abs()
    n_ok = int((~df["failed"]).sum())
    commit, dirty = git_state()
    info = {
        "results_csv": os.path.relpath(results_csv, REPO_ROOT),
        "computed_by": {"script": "scripts/compute_metrics.py", "git_commit": commit,
                        "git_dirty": dirty, "scikit_learn": sklearn.__version__},
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "fits": int(len(df)),
        "fits_failed": int(df["failed"].sum()),
        "fits_scored": int(metrics["accuracy"].notna().sum()),
        "missing_prediction_files": missing,
        "max_abs_roc_auc_diff": float(diff.max()) if diff.notna().any() else None,
        "max_row_sum_error": (float(metrics["row_sum_error"].max())
                              if metrics["row_sum_error"].notna().any() else None),
        "tolerance": TOLERANCE,
    }

    tmp = os.path.join(base_dir, "metrics.csv.tmp")
    metrics.to_csv(tmp, index=False)
    os.replace(tmp, os.path.join(base_dir, "metrics.csv"))
    with open(os.path.join(base_dir, "metrics_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\nUčenj: {len(df)}, uspelih {n_ok}, ocenjenih iz datotek {info['fits_scored']}")
    print(f"Največja razlika ROC-AUC (results.csv proti datoteki): {info['max_abs_roc_auc_diff']}")
    print(f"Največje odstopanje vsote vrstice proba od 1: {info['max_row_sum_error']}")
    print(f"Zapisano: {os.path.relpath(base_dir, REPO_ROOT)}/metrics.csv, metrics_info.json")

    ok = True
    if missing:
        ok = False
        print(f"\nNAPAKA: {len(missing)} uspelim učenjem manjka datoteka z napovedmi, npr. "
              f"{missing[0]} - priklopi USB disk in skopiraj mapo predictions/ (CLAUDE.md).")
    if info["max_abs_roc_auc_diff"] is not None and info["max_abs_roc_auc_diff"] > TOLERANCE:
        ok = False
        print(f"\nNAPAKA: ROC-AUC iz datoteke se od results.csv razlikuje za več kot {TOLERANCE}.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
