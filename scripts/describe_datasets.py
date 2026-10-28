"""Opisna statistika podatkovnih množic za poglavje Materiali v diplomi.

Za vsak nabor iz izbranega seznama (config.yaml, dataset_sets) naloži podatke z
load_dataset - isto pot kot benchmark, torej iz predpomnilnika OpenML oz. iz
lokalne datoteke, z istimi izločenimi stolpci in omejitvijo razredov - in
izračuna lastnosti, ki jih diploma navaja pri opisu zbirke OpenML-CC18 in nabora
Medic3 ter jih razdelek o odvisnosti od lastnosti naborov (scripts/
analyze_properties.py) poveže z rezultati:

  n_rows, n_features, n_categorical      velikost in vrste atributov (kategorični =
                                         stolpci, ki jih loader označi kot kategorične)
  n_classes, min/median/max_class_size   razredi in njihove velikosti
  majority_share, minority_share         delež največjega in najmanjšega razreda
  imbalance_ratio                        najmanjši / največji razred (merilo zbirke
                                         OpenML-CC18 zahteva več kot 0,05)
  missing_share                          delež manjkajočih celic (vrstice x atributi)
  rows_with_missing_share                delež vrstic z vsaj eno manjkajočo vrednostjo
  median_observed_per_row                mediana števila znanih atributov v vrstici
  rows_dropped                           vrstice, ki jih je odstranila omejitev razredov
                                         (max_classes, le pri medic3_160)

Nič se ne uči; nabori se le naložijo. Pri zaupnem naboru Medic3 zapiše le te
zbirne številke (nobene vrstice podatkov), ki jih diploma objavi.

Zagon (iz korena repozitorija; nabori morajo biti v predpomnilniku oz. na disku):
    python scripts/describe_datasets.py --dataset-set cc18
    python scripts/describe_datasets.py --dataset-set medic3
    python scripts/describe_datasets.py --dataset-set medic3_160
Izhod: results/datasets/<dataset-set>.csv in izpis povzetka (najmanj, mediana, največ).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.config import dataset_specs, load_config, spec_id  # noqa: E402
from src.data import load_dataset  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, "results", "datasets")


def describe(spec, config):
    """Ena vrstica lastnosti za en nabor (podatki naloženi tako kot v benchmarku)."""
    data = load_dataset(spec, n_splits=config["n_splits"], random_state=config["random_state"],
                        cache_dir=config["cache_dir"], n_repeats=config["n_repeats"])
    X, y = data["X"], pd.Series(np.asarray(data["y"]))
    counts = y.value_counts()
    observed = X.notna().sum(axis=1)
    limit = data["class_limit"] or {}
    return {
        "dataset_id": spec_id(spec),
        "dataset": data["name"],
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_categorical": int(len(data["categorical_cols"])),
        "n_classes": int(len(counts)),
        "min_class_size": int(counts.min()),
        "median_class_size": float(counts.median()),
        "max_class_size": int(counts.max()),
        "majority_share": float(counts.max() / len(y)),
        "minority_share": float(counts.min() / len(y)),
        "imbalance_ratio": float(counts.min() / counts.max()),
        "missing_share": float(X.isna().to_numpy().mean()),
        "rows_with_missing_share": float((observed < X.shape[1]).mean()),
        "median_observed_per_row": float(observed.median()),
        "rows_dropped": int(limit.get("n_rows_dropped", 0)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-set", required=True, help="Ime iz config.yaml (cc18, medic3, medic3_160)")
    args = parser.parse_args()

    config = load_config()
    specs = dataset_specs(config, args.dataset_set)
    rows = []
    for i, spec in enumerate(specs, 1):
        row = describe(spec, config)
        rows.append(row)
        print(f"  {i:2d}/{len(specs)} {row['dataset']}: {row['n_rows']} x {row['n_features']}, "
              f"{row['n_classes']} razredov, manjkajočih {100 * row['missing_share']:.2f} %")
    table = pd.DataFrame(rows)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{args.dataset_set}.csv")
    table.to_csv(out, index=False)

    print(f"\n{args.dataset_set}: {len(table)} naborov")
    table["categorical_share"] = table["n_categorical"] / table["n_features"]
    cols = ["n_rows", "n_features", "categorical_share", "n_classes", "imbalance_ratio",
            "missing_share", "rows_with_missing_share"]
    print(table[cols].describe().loc[["min", "50%", "max"]].to_string())
    if len(table) > 1:
        print(f"z vsaj enim kategoričnim atributom: {int((table['n_categorical'] > 0).sum())}, "
              f"samo s kategoričnimi: {int((table['n_categorical'] == table['n_features']).sum())}, "
              f"z manjkajočimi vrednostmi: {int((table['missing_share'] > 0).sum())}, "
              f"binarnih: {int((table['n_classes'] == 2).sum())}")
    print(f"Zapisano: {os.path.relpath(out, REPO_ROOT)}")


if __name__ == "__main__":
    main()
