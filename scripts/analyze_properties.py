"""Ali je prednost temeljnih modelov odvisna od lastnosti podatkovnih množic?

Raziskovalna analiza za diplomo (razdelek »Odvisnost od lastnosti podatkovnih
množic«), narejena po zagonu - ni vnaprej določen test. Za vsak nabor CC18:

  prednost = max(TabPFN-3.5, TabICL) - max(štirje drevesni ansambli)

na povprečnem ROC-AUC čez 15 učenj (summary/pivot.csv), kot v razdelku z
rezultati po naborih. Prednost poveže z lastnostmi nabora iz
results/datasets/cc18.csv (scripts/describe_datasets.py): število primerov,
atributov in razredov, delež kategoričnih atributov in delež manjkajočih celic.

Za vsako lastnost Spearmanov koeficient korelacije (rangi, zato logaritem
velikosti ne spremeni ničesar) z dvostransko p-vrednostjo, Holmov popravek čez
lastnosti, posebej na vseh naborih in na nenasičenih (nasičenost iz pivot.csv,
najboljši ROC-AUC >= saturation_threshold). Na nasičenih je prednost omejena z
zgornjo mejo metrike, zato je glavna podmnožica nenasičenih. Dodatno mediana
prednosti pod in nad mejo LARGE_N primerov (do 10 000 primerov je bil
ovrednoten TabPFN v2, Hollmann in sod. 2025).

Zagon (iz korena repozitorija):
    python scripts/analyze_properties.py
    python scripts/analyze_properties.py --run cc18_v4/combined --datasets results/datasets/cc18.csv
Izhod: <zagon>/summary/properties_advantage.csv (po naborih), properties_spearman.csv,
properties_report.txt.
"""

import argparse
import os
import sys

import pandas as pd
from scipy import stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.config import load_config  # noqa: E402
from src.stats import holm  # noqa: E402
from src.summary import resolve_results  # noqa: E402

FOUNDATION = ["tabpfn", "tabicl"]
TREES = ["catboost", "lightgbm", "xgboost", "random_forest"]
PROPERTIES = {
    "n_rows": "število primerov",
    "n_features": "število atributov",
    "n_classes": "število razredov",
    "categorical_share": "delež kategoričnih atributov",
    "missing_share": "delež manjkajočih vrednosti",
}
LARGE_N = 10_000


class Report:
    """Izpis na zaslon in hkrati v properties_report.txt."""

    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text)
        self.lines.append(text)


def advantage_table(pivot, datasets):
    """Prednost temeljnih modelov in lastnosti, ena vrstica na nabor."""
    out = pd.DataFrame({
        "best_foundation": pivot[FOUNDATION].max(axis=1),
        "best_foundation_algorithm": pivot[FOUNDATION].idxmax(axis=1),
        "best_tree": pivot[TREES].max(axis=1),
        "best_tree_algorithm": pivot[TREES].idxmax(axis=1),
        "saturated": pivot["saturated"].astype(bool),
    })
    out["advantage"] = out["best_foundation"] - out["best_tree"]
    out["tabpfn_minus_catboost"] = pivot["tabpfn"] - pivot["catboost"]
    props = datasets.set_index("dataset")
    props["categorical_share"] = props["n_categorical"] / props["n_features"]
    missing = sorted(set(out.index) - set(props.index))
    if missing:
        raise SystemExit(f"V tabeli lastnosti manjkajo nabori: {missing}")
    out = out.join(props[["dataset_id"] + list(PROPERTIES)])
    return out.reset_index().rename(columns={"index": "dataset"})


def spearman(table, alpha):
    rows = []
    for subset, mask in (("vsi", pd.Series(True, index=table.index)), ("nenasičeni", ~table["saturated"])):
        sub = table[mask]
        block = []
        for prop in PROPERTIES:
            res = stats.spearmanr(sub[prop], sub["advantage"])
            block.append({"subset": subset, "n": int(len(sub)), "property": prop,
                          "rho": float(res.statistic), "p": float(res.pvalue)})
        for r, ph in zip(block, holm([r["p"] for r in block])):
            r["p_holm"] = float(ph)
            r["significant"] = bool(ph < alpha)
        rows += block
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="cc18_v4/combined", help="Zagon z rezultati CC18")
    parser.add_argument("--datasets", default="results/datasets/cc18.csv",
                        help="Lastnosti naborov (scripts/describe_datasets.py)")
    args = parser.parse_args()

    config = load_config()
    _, run_dir = resolve_results(args.run, config)
    pivot = pd.read_csv(os.path.join(run_dir, "summary", "pivot.csv"),
                        float_precision="round_trip").set_index("dataset")
    datasets = pd.read_csv(os.path.join(REPO_ROOT, args.datasets), dtype={"dataset_id": str})
    table = advantage_table(pivot, datasets)
    corr = spearman(table, config["alpha"])

    report = Report()
    report(f"Prednost temeljnih modelov: {os.path.relpath(run_dir, REPO_ROOT)} + {args.datasets}")
    report(f"  naborov {len(table)}, nenasičenih {int((~table['saturated']).sum())}; "
           f"prednost > 0 na {int((table['advantage'] > 0).sum())}")
    for subset, sub in (("vsi", table), ("nenasičeni", table[~table["saturated"]])):
        report(f"  {subset}: mediana prednosti {sub['advantage'].median():+.4f}, "
               f"povprečje {sub['advantage'].mean():+.4f}")
        small, large = sub[sub["n_rows"] <= LARGE_N], sub[sub["n_rows"] > LARGE_N]
        for label, part in ((f"<= {LARGE_N}", small), (f"> {LARGE_N}", large)):
            if len(part):
                report(f"    {label} primerov: {len(part)} naborov, mediana prednosti "
                       f"{part['advantage'].median():+.4f}, prednost > 0 na "
                       f"{int((part['advantage'] > 0).sum())}, TabPFN-3.5 > CatBoost na "
                       f"{int((part['tabpfn_minus_catboost'] > 0).sum())}")
    # Nasičenost stisne prednost proti 0; če je pogostejša pri večrazrednih naborih,
    # se to pokaže kot korelacija s številom razredov na vseh naborih.
    binary = table["n_classes"] == 2
    report(f"  nasičeni: med binarnimi {int(table.loc[binary, 'saturated'].sum())} od {int(binary.sum())}, "
           f"med večrazrednimi {int(table.loc[~binary, 'saturated'].sum())} od {int((~binary).sum())}")
    report()
    report("Spearmanov koeficient (prednost proti lastnosti), Holm čez lastnosti znotraj podmnožice:")
    for _, r in corr.iterrows():
        report(f"  {r['subset']:10} n={r['n']:2d}  {PROPERTIES[r['property']]:30} rho={r['rho']:+.3f}  "
               f"p={r['p']:.3g}  p_Holm={r['p_holm']:.3g}{'  *' if r['significant'] else ''}")
    report()
    report("Največja in najmanjša prednost (nenasičeni):")
    ns = table[~table["saturated"]].sort_values("advantage")
    for _, r in pd.concat([ns.head(3), ns.tail(5)]).iterrows():
        report(f"  {r['dataset']:35} {r['advantage']:+.4f}  primerov {r['n_rows']:6d}  "
               f"atributov {r['n_features']:5d}  razredov {r['n_classes']:3d}")

    out_dir = os.path.join(run_dir, "summary")
    table.to_csv(os.path.join(out_dir, "properties_advantage.csv"), index=False)
    corr.to_csv(os.path.join(out_dir, "properties_spearman.csv"), index=False)
    with open(os.path.join(out_dir, "properties_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report.lines) + "\n")
    print(f"\nZapisano v {os.path.relpath(out_dir, REPO_ROOT)}/properties_*")


if __name__ == "__main__":
    main()
