"""Izpiše povprečni ROC-AUC ± std po (dataset, algoritem) in povprečen
rang po algoritmu čez vse datasete.

Zagon: python -m src.summary [results_csv]

Brez argumenta povzame results/results.csv (lokalna pilotna izhodiščna
meritev, 3 dataseti). Za zagon na gruči je pot treba podati eksplicitno:
    python -m src.summary results/results_arnes_cc18.csv
"""

import argparse
import os

import pandas as pd
from tabulate import tabulate

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_CSV = os.path.join(REPO_ROOT, "results", "results.csv")


def summarize(results_csv=RESULTS_CSV):
    # Katera datoteka je povzeta, je izpisano izrecno: privzetek je pilot na
    # 3 datasetih, zato bi tiho povzemanje napačne datoteke pomenilo napačne
    # številke v diplomi brez vsakega opozorila.
    print(f"Vir: {os.path.relpath(results_csv, REPO_ROOT)}\n")
    df = pd.read_csv(results_csv)
    ok = df.dropna(subset=["roc_auc"])
    failed = df[df["roc_auc"].isna()]

    per_ds = (
        ok.groupby(["dataset", "algorithm"])["roc_auc"]
        .agg(mean="mean", std="std")
        .reset_index()
        .sort_values(["dataset", "mean"], ascending=[True, False])
    )

    ok = ok.assign(rank=ok.groupby(["dataset", "fold"])["roc_auc"].rank(ascending=False))
    overall = (
        ok.groupby("algorithm")
        .agg(mean_roc_auc=("roc_auc", "mean"), std_roc_auc=("roc_auc", "std"), mean_rank=("rank", "mean"))
        .sort_values("mean_rank")
        .reset_index()
    )

    print("=== ROC-AUC po (dataset, algoritem): mean ± std ===")
    print(tabulate(per_ds, headers="keys", floatfmt=".4f", showindex=False))

    print("\n=== Povprečje čez vse datasete: mean ROC-AUC ± std in povprečen rang ===")
    print(tabulate(overall, headers="keys", floatfmt=".4f", showindex=False))

    if not failed.empty:
        print("\n=== Spodleteli tek-i (roc_auc manjka) ===")
        print(tabulate(
            failed[["dataset", "algorithm", "fold"]],
            headers="keys", showindex=False,
        ))

    return per_ds, overall


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Povzetek rezultatov benchmarka.")
    parser.add_argument(
        "results_csv",
        nargs="?",
        default=RESULTS_CSV,
        help="Pot do CSV-ja z rezultati (privzeto results/results.csv)",
    )
    args = parser.parse_args()
    summarize(args.results_csv)
