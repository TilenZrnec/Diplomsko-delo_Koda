"""Sestavi celoten nabor rezultatov iz delnega ponovnega zagona in osnovnega zagona.

Ko popravimo le nekatere algoritme (2026-10-14: RandomForest in XGBoost "kot iz
škatle"), ponovno poženemo samo njih (npr. cc18_v4), ostali ostanejo iz
osnovnega zagona (cc18_v3), ker se njihova koda ni spremenila. Ta skript iz
obeh sestavi en results.csv za povzetek, statistiko in diplomo:

    vse vrstice NOVEGA zagona
  + vrstice OSNOVNEGA zagona za algoritme, ki jih novi zagon nima.

Izhod gre v results/runs/<nov>/combined/ (results.csv + sources.json), nikoli v
osnovni zagon. Pred sestavljanjem preveri, da sta zagona primerljiva: enaki
nastavitve delitev (random_state, n_splits, n_repeats), enaki nabori in enake
velikosti učnih/testnih foldov. Nato:
    python -m src.summary results/runs/<nov>/combined/results.csv
    python -m src.stats   results/runs/<nov>/combined/results.csv [--no-saturated]

Zagon (iz korena repozitorija):
    python scripts/combine_runs.py cc18_v4 --base cc18_v3
    python scripts/combine_runs.py medic3_raw_v2 --base medic3_raw
"""

import argparse
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402

SPLIT_KEYS = ("random_state", "n_splits", "n_repeats")


def git_state():
    """(commit, dirty) kode, ki je sestavila nabor - za sources.json."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                              text=True).stdout.strip()
    return git("rev-parse", "--short", "HEAD"), bool(git("status", "--porcelain", "--", "src", "scripts"))


def load_run(results_dir, run):
    run_dir = os.path.join(results_dir, run)
    df = pd.read_csv(os.path.join(run_dir, "results.csv"), dtype={"dataset_id": str})
    with open(os.path.join(run_dir, "manifest.json")) as f:
        manifest = json.load(f)
    return df, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("new", help="delni (novejši) zagon, npr. cc18_v4")
    parser.add_argument("--base", required=True, help="osnovni zagon, npr. cc18_v3")
    args = parser.parse_args()

    results_dir = load_config()["results_dir"]
    new, new_manifest = load_run(results_dir, args.new)
    base, base_manifest = load_run(results_dir, args.base)

    # 1. Enake nastavitve delitev - sicer foldi niso isti in primerjava ne velja.
    for key in SPLIT_KEYS:
        a, b = new_manifest["config"].get(key), base_manifest["config"].get(key)
        if a != b:
            raise SystemExit(f"{key} se razlikuje: {args.new}={a}, {args.base}={b}")

    # 2. Enaki nabori in enake velikosti foldov.
    folds = ["dataset_id", "fold"]
    new_folds = new[folds + ["n_train", "n_test"]].drop_duplicates()
    base_folds = base[folds + ["n_train", "n_test"]].drop_duplicates()
    joined = new_folds.merge(base_folds, on=folds, how="outer", suffixes=("_new", "_base"),
                             indicator=True)
    if (joined["_merge"] != "both").any():
        missing = joined[joined["_merge"] != "both"][folds].drop_duplicates()
        raise SystemExit(f"Nabori/foldi se ne ujemajo ({len(missing)} parov), npr.:\n{missing.head()}")
    if ((joined.n_train_new != joined.n_train_base) | (joined.n_test_new != joined.n_test_base)).any():
        raise SystemExit("Velikosti foldov se razlikujejo - zagona nista na istih delitvah.")

    # 3. Sestavljanje: novi zagon + manjkajoči algoritmi iz osnovnega.
    new_algos = sorted(new.algorithm.unique())
    kept = base[~base.algorithm.isin(new_algos)]
    combined = pd.concat([new, kept], ignore_index=True)
    combined = combined.sort_values(["dataset_id", "algorithm", "fold"], kind="stable")

    out_dir = os.path.join(results_dir, args.new, "combined")
    os.makedirs(out_dir, exist_ok=True)
    combined.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    commit, dirty = git_state()
    sources = {
        "note": "results.csv = all rows of 'new' + rows of 'base' for the algorithms 'new' lacks",
        "combined_by": {"script": "scripts/combine_runs.py", "git_commit": commit, "git_dirty": dirty},
        "new": {"run": args.new, "git_commit": new_manifest.get("git_commit"),
                "algorithms": new_algos, "rows": int(len(new))},
        "base": {"run": args.base, "git_commit": base_manifest.get("git_commit"),
                 "algorithms": sorted(kept.algorithm.unique()), "rows": int(len(kept))},
        "split_config": {k: new_manifest["config"].get(k) for k in SPLIT_KEYS},
        "rows": int(len(combined)), "datasets": int(combined.dataset_id.nunique()),
    }
    with open(os.path.join(out_dir, "sources.json"), "w") as f:
        json.dump(sources, f, indent=2)
    print(f"{len(combined)} vrstic ({combined.dataset_id.nunique()} naborov): {args.new} "
          f"{new_algos} + {args.base} {sources['base']['algorithms']} -> "
          f"{os.path.relpath(out_dir, REPO_ROOT)}/")


if __name__ == "__main__":
    main()
