"""Združi vse per-dataset CSV-je iz mape v en skupni CSV.

Zagon: python scripts/merge_results.py <izhodni.csv> [--input-dir MAPA]

Privzeta vhodna mapa je results/per_dataset/ (scratch izhod SLURM polja);
z --input-dir se združi kurirana mapa, npr. results/arnes/subset/per_dataset/.
Izpiše število vrstic in datasetov ter koliko vrstic ima neprazno napako.
"""

import argparse
import glob
import os

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Združevanje per-dataset CSV-jev.")
    parser.add_argument("output_csv", help="Pot do izhodnega skupnega CSV-ja")
    parser.add_argument(
        "--input-dir",
        default=os.path.join(REPO_ROOT, "results", "per_dataset"),
        help="Mapa s per-dataset CSV-ji (privzeto results/per_dataset/)",
    )
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
    if not paths:
        print(f"Ni najdenih CSV-jev v {args.input_dir}")
        return

    # *.partial so nedokončani dataseti (task je npr. presegel --time). V merge
    # namenoma ne gredo, a moraš zanje vedeti - zato glasno opozorilo.
    partials = sorted(glob.glob(os.path.join(args.input_dir, "*.partial")))
    if partials:
        print(f"OPOZORILO: {len(partials)} nedokončanih datasetov (*.partial) - NISO v merge:")
        for path in partials:
            n_done = len(pd.read_csv(path))
            print(f"  {os.path.basename(path)}: {n_done} od 30 učenj; oddaj znova za dokončanje")
        print()

    merged = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    merged.to_csv(args.output_csv, index=False)

    n_errors = merged["error"].notna().sum() if "error" in merged.columns else 0
    print(f"Združenih {len(paths)} datotek -> {args.output_csv}")
    print(f"Vrstic: {len(merged)}, datasetov: {merged['dataset'].nunique()}")
    print(f"Vrstic z neprazno napako: {n_errors}")


if __name__ == "__main__":
    main()
