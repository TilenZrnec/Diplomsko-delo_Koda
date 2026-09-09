"""Združi per_dataset/*.csv enega zagona v results.csv.

Zagon: python scripts/merge_results.py <run_id ali pot do mape zagona> [--output POT]

Privzeto piše v <mapa zagona>/results.csv. Nedokončane nabore (*.partial)
glasno našteje, a jih ne vključi - nepopoln zagon se tako vidi, namesto da bi
se skril. Lokalni zagon (src/run_benchmark.py) to naredi sam; skripta je za
Arnes, kjer taski tečejo ločeno in združevanje sproži uporabnik.
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.config import load_config  # noqa: E402
from src.runner import merge_run  # noqa: E402


def resolve_run_dir(run, config=None):
    """Sprejme run_id ali pot; vrne absolutno pot do mape zagona."""
    if os.path.isdir(run):
        return os.path.abspath(run)
    config = config or load_config()
    candidate = os.path.join(config["results_dir"], run)
    if os.path.isdir(candidate):
        return candidate
    raise SystemExit(f"Mapa zagona ne obstaja: {run} (niti {candidate})")


def main():
    parser = argparse.ArgumentParser(description="Združevanje per-dataset CSV-jev enega zagona.")
    parser.add_argument("run", help="run_id (pod results_dir) ali pot do mape zagona")
    parser.add_argument("--output", default=None, help="Izhodni CSV (privzeto <mapa>/results.csv)")
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run)
    merge_run(run_dir, output_path=args.output)


if __name__ == "__main__":
    main()
