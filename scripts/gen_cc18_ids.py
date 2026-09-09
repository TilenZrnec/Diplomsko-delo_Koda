"""Pripne (pin) seznam OpenML ID-jev zbirke OpenML-CC18 v scripts/cc18_ids.json.

Zbirko prebere prek openml.study.get_suite(99) (alias "OpenML-CC18"), vzame
ID-je datasetov iz suite.data, jih odstrani podvojene in uredi naraščajoče.
Zapisana datoteka ima isto shemo kot scripts/subset_ids.json - navaden JSON
seznam celih števil.

Skript ima varovalko: če OpenML ne vrne natanko 72 datasetov, se konča z
napako. Tiha sprememba zbirke na strani OpenML torej ne more neopazno
spremeniti obsega eksperimenta. Zato se cc18_ids.json commita v repo in se
ob oddaji SLURM polja ne pridobiva znova.

Zagon (potrebuje internet - lokalno ali na prijavnem vozlišču):

    python scripts/gen_cc18_ids.py
"""

import json
import os
import sys

import openml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SUITE_ID = 99  # OpenML-CC18
EXPECTED_N = 72
OUT_PATH = os.path.join(REPO_ROOT, "scripts", "cc18_ids.json")


def main():
    # Isti predpomnilnik kot src/data.py, da metapodatki zbirke pristanejo v repu.
    openml.config.set_root_cache_directory(load_config()["cache_dir"])

    suite = openml.study.get_suite(SUITE_ID)
    ids = sorted({int(data_id) for data_id in suite.data})

    print(f"Zbirka: {suite.name} (alias {suite.alias}, ID {SUITE_ID})")
    print(f"Število datasetov: {len(ids)}")

    if len(ids) != EXPECTED_N:
        raise SystemExit(
            f"NAPAKA: pričakovanih {EXPECTED_N} datasetov, OpenML jih je vrnil {len(ids)}. "
            "Zbirka se je spremenila - preveri in po potrebi popravi EXPECTED_N."
        )

    with open(OUT_PATH, "w") as f:
        f.write(json.dumps(ids) + "\n")

    print(f"Zapisano v {OUT_PATH}")


if __name__ == "__main__":
    main()
