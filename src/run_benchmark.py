"""Lokalni zagon: zaporedno čez vse nabore izbranega nabora, nato združi.

    python -m src.run_benchmark                          # pilotna trojica (subset)
    python -m src.run_benchmark --dataset-set cc18       # vseh 72 naborov, zaporedno
    python -m src.run_benchmark --run-id pilot_v2        # lastna oznaka zagona

Vsak zagon dobi svojo mapo results/runs/<run_id>/ (privzeto
<datum-čas>_<nabor>_<stroj>) z manifest.json, per_dataset/, predictions/ in
results.csv. Nič se ne prepiše: ponovitev z istim --run-id nadaljuje
nedokončane nabore, nov --run-id začne od začetka.

Zanka učenja je v src/runner.py - ista, kot jo uporablja Arnes.
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.config import dataset_specs, load_config  # noqa: E402
from src.runner import make_run_id, merge_run, run_dataset, run_dir_for, write_manifest  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Lokalni zagon benchmarka.")
    parser.add_argument("--dataset-set", default="subset", help="Ime nabora iz config.yaml (privzeto subset)")
    parser.add_argument("--run-id", default=None, help="Oznaka zagona (privzeto <datum-čas>_<nabor>_<stroj>)")
    parser.add_argument(
        "--algorithms", nargs="+", default=None,
        help="Podmnožica algoritmov (privzeto vsi iz config.yaml)",
    )
    args = parser.parse_args()

    config = load_config()
    specs = dataset_specs(config, args.dataset_set)
    run_id = args.run_id or make_run_id(args.dataset_set)
    run_dir = run_dir_for(config, run_id)
    print(f"Zagon: {run_id}\nMapa:  {os.path.relpath(run_dir, REPO_ROOT)}")
    write_manifest(run_dir, config, args.dataset_set, specs, run_id)

    for i, spec in enumerate(specs, start=1):
        print(f"\n--- nabor {i}/{len(specs)} ---")
        run_dataset(spec, config, run_dir, run_id, algorithms=args.algorithms)

    print()
    results_csv = merge_run(run_dir)
    if results_csv:
        print(f"\nPovzetek: python -m src.summary {os.path.relpath(run_dir, REPO_ROOT)}")


if __name__ == "__main__":
    main()
