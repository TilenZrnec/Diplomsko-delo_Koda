"""Primerjava dveh zagonov po (nabor, algoritem, fold) - sanity check prenosa
na gručo ali učinka nadgradnje knjižnic.

Zagon: python scripts/compare_results.py <A> <B>

A in B sta lahko run_id, mapa zagona (uporabi se <mapa>/results.csv) ali pot do
CSV-ja. Izpiše povprečno in največjo absolutno razliko ROC-AUC po algoritmih
ter 5 najslabših vrstic.
"""

import argparse
import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.summary import resolve_results  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Primerjava dveh zagonov.")
    parser.add_argument("a", help="run_id, mapa zagona ali CSV (npr. lokalno)")
    parser.add_argument("b", help="run_id, mapa zagona ali CSV (npr. Arnes)")
    args = parser.parse_args()

    csv_a, _ = resolve_results(args.a)
    csv_b, _ = resolve_results(args.b)
    print(f"A: {os.path.relpath(csv_a, REPO_ROOT)}\nB: {os.path.relpath(csv_b, REPO_ROOT)}\n")
    a = pd.read_csv(csv_a)
    b = pd.read_csv(csv_b)

    merged = a.merge(b, on=["dataset", "algorithm", "fold"], suffixes=("_a", "_b"))
    if len(merged) == 0:
        print(
            "OPOZORILO: združevanje ni dalo nobene vrstice - imena naborov se "
            "med datotekama morda razlikujejo."
        )
        return

    merged["abs_delta"] = (merged["roc_auc_a"] - merged["roc_auc_b"]).abs()

    print(f"Skupnih vrstic: {len(merged)}\n")
    print("|delta roc_auc| po algoritmih (NaN = neuspeh na eni strani):")
    print(merged.groupby("algorithm")["abs_delta"].agg(["mean", "max"]).to_string())

    worst = merged.nlargest(5, "abs_delta")
    print("\n5 najslabših vrstic:")
    cols = ["dataset", "algorithm", "fold", "roc_auc_a", "roc_auc_b", "abs_delta"]
    print(worst[cols].to_string(index=False))


if __name__ == "__main__":
    main()
