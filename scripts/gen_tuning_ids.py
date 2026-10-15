"""Zapiše scripts/cc18_tuning_ids.json: nabori za poskus uglaševanja XGBoost.

Odločitev uporabnika 2026-10-14 (CLAUDE.md, What is next): vseh 72 naborov
CC18 RAZEN štirih velikih slikovnih (mnist_784, Devnagari-Script, CIFAR_10,
Fashion-MNIST - izločeni vnaprej in le zaradi računske cene), torej 68.
Vrstni red: najprej NENASIČENI (najboljši algoritem < saturation_threshold iz
config.yaml), nato nasičeni; znotraj skupine v vrstnem redu cc18_ids.json. Tako
sta obe oddaji na Arnes le razpona indeksov: 0..(n_nenasičenih-1), nato ostalo.

Nasičenost se določi iz rezultatov diplome (privzeto results/runs/cc18_v4/combined),
z istim pravilom kot src.summary. Datoteka se zapiše enkrat in je nato pripeta
(kot cc18_ids.json); skript jo le ponovno izračuna za preverbo.

Zagon (iz korena repozitorija):
    python scripts/gen_tuning_ids.py [--results results/runs/cc18_v4/combined/results.csv]
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.config import load_config  # noqa: E402
from src.summary import dataset_pivot, load_results, saturated_mask  # noqa: E402

# OpenML ID-ji štirih velikih slikovnih naborov (v cc18_v3 so imeli svoje polje SLURM).
EXCLUDED = {554: "mnist_784", 40923: "Devnagari-Script", 40927: "CIFAR_10", 40996: "Fashion-MNIST"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default=os.path.join(
        REPO_ROOT, "results", "runs", "cc18_v4", "combined", "results.csv"))
    args = parser.parse_args()

    config = load_config()
    df = load_results(args.results)
    pivot = dataset_pivot(df)
    saturated = saturated_mask(pivot, config["saturation_threshold"])
    name_to_id = dict(zip(df.dataset, df.dataset_id.astype(int)))
    sat_ids = {name_to_id[name] for name, s in saturated.items() if s}

    with open(os.path.join(REPO_ROOT, "scripts", "cc18_ids.json")) as f:
        cc18 = json.load(f)
    kept = [i for i in cc18 if i not in EXCLUDED]
    non_sat = [i for i in kept if i not in sat_ids]
    sat = [i for i in kept if i in sat_ids]
    ids = non_sat + sat

    out = os.path.join(REPO_ROOT, "scripts", "cc18_tuning_ids.json")
    with open(out, "w") as f:
        json.dump(ids, f)
    id_to_name = {v: k for k, v in name_to_id.items()}
    print(f"{len(ids)} naborov -> {os.path.relpath(out, REPO_ROOT)}: indeksi 0-{len(non_sat) - 1} "
          f"nenasičeni ({len(non_sat)}), {len(non_sat)}-{len(ids) - 1} nasičeni ({len(sat)}); "
          f"izločeni: {sorted(EXCLUDED.values())}")
    for k, i in enumerate(ids):
        print(f"  {k:2d}  {i:6d}  {id_to_name[i]}{'' if i in non_sat else '  (nasičen)'}")


if __name__ == "__main__":
    main()
