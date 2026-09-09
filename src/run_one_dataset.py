"""Požene vse algoritme na enem samem naboru (SLURM array task na Arnesu).

Vsak array task obdela en nabor, tako da se nabori računajo vzporedno:

    python -m src.run_one_dataset --dataset-set cc18 --index N --run-id cc18_v2

Iz nabora <dataset-set> (config.yaml: dataset_sets) vzame N-ti opis nabora in
ga preda src/runner.py, ki zapiše results/runs/<run_id>/per_dataset/<id>.csv.
Vsa logika (kontrolne točke, nadaljevanje, zapis napovedi, manifest) je v
runnerju - tu je samo razčlenitev argumentov.

--run-id je obvezen: vsi taski enega polja morajo pisati v ISTO mapo, in
ponovna oddaja po prekinitvi mora uporabiti isti run_id, da nadaljuje iz
partiala namesto da začne znova. Batch skripta ga poda iz spremenljivke RUN_ID.
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.config import dataset_specs, load_config  # noqa: E402
from src.runner import run_dataset, run_dir_for, write_manifest  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Zagon vseh algoritmov na enem naboru.")
    parser.add_argument("--index", type=int, required=True, help="Indeks nabora v izbranem naboru")
    parser.add_argument("--dataset-set", default="cc18", help="Ime nabora iz config.yaml (privzeto cc18)")
    parser.add_argument("--run-id", required=True, help="Oznaka zagona (mapa results/runs/<run_id>/)")
    parser.add_argument(
        "--algorithms", nargs="+", default=None,
        help="Podmnožica algoritmov (privzeto vsi iz config.yaml)",
    )
    args = parser.parse_args()

    config = load_config()
    specs = dataset_specs(config, args.dataset_set)
    if not 0 <= args.index < len(specs):
        raise SystemExit(f"--index {args.index} je izven obsega 0..{len(specs) - 1}")

    run_dir = run_dir_for(config, args.run_id)
    write_manifest(run_dir, config, args.dataset_set, specs, args.run_id)
    run_dataset(specs[args.index], config, run_dir, args.run_id, algorithms=args.algorithms)


if __name__ == "__main__":
    main()
