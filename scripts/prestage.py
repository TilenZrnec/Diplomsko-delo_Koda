"""Predpriprava na prijavnem vozlišču Arnes (login node nima GPU-ja).

Prenese OpenML datasete v data/openml_cache ter enkrat fitta TabPFN in
TabICL na majhnem naključnem vzorcu, da se uteži modelov preneseta v
lokalni predpomnilnik - računska vozlišča nato tečejo brez dostopa do
interneta (HF_HUB_OFFLINE=1).

Zagon (samo na prijavnem vozlišču, potrebuje internet):

    python scripts/prestage.py                                   # pilotna podmnožica
    python scripts/prestage.py --ids-file scripts/cc18_ids.json  # cel CC18
    python scripts/prestage.py --ids 31 37 38                    # izbrani ID-ji
    python scripts/prestage.py --ids 37 --skip-weights           # samo dataset, brez utež

--skip-weights preskoči TabPFN/TabICL - uporabno za preverjanje samega
predpomnilnika datasetov na računalniku brez poverilnice TabPFN.

Prenos posameznega dataseta ne prekine celotne predpriprave - napake se
zberejo in izpišejo na koncu, izhodna koda pa je 1, če kateri dataset
manjka. Na koncu izpiše skupno velikost predpomnilnika, da jo je mogoče
primerjati s kvoto domačega imenika (100 GB) pred oddajo SLURM polja.
"""

import argparse
import json
import os
import sys

import numpy as np
import openml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.data import load_dataset  # noqa: E402

CACHE_DIR = os.path.join(REPO_ROOT, "data", "openml_cache")


def effective_cache_dir():
    """Vrne mapo, v katero openml dejansko piše predpomnilnik.

    src/data.py nastavi koren predpomnilnika s set_root_cache_directory(),
    openml pa pod njim sam doda org/openml/www. Za preverjanje kvote
    domačega imenika šteje ta, dejanska pot - zato jo vprašamo knjižnico.
    """
    return openml.config.get_cache_directory()


def dir_size_bytes(path):
    """Vrne skupno velikost mape v bajtih (ekvivalent du -s, v Pythonu)."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if not os.path.islink(file_path):
                total += os.path.getsize(file_path)
    return total


def human_size(n_bytes):
    """Pretvori bajte v človeku berljiv zapis (kot du -sh)."""
    size = float(n_bytes)
    for unit in ("B", "K", "M", "G", "T"):
        if size < 1024 or unit == "T":
            return f"{size:.1f}{unit}"
        size /= 1024


def prestage_datasets(ids):
    """Prenese in predpomni vse datasete; vrne seznam (id, napaka) za neuspele."""
    failures = []
    for i, openml_id in enumerate(ids, start=1):
        try:
            dataset = load_dataset(openml_id, n_splits=5, random_state=42, cache_dir=CACHE_DIR)
            X = dataset["X"]
            print(
                f"[{i}/{len(ids)}] OpenML {openml_id} ({dataset['name']}): "
                f"{X.shape[0]} x {X.shape[1]}, preneseno in predpomnjeno"
            )
        except Exception as exc:  # noqa: BLE001 - predpriprava ne sme pasti zaradi enega dataseta
            print(f"[{i}/{len(ids)}] OpenML {openml_id}: NAPAKA - {type(exc).__name__}: {exc}")
            failures.append((openml_id, f"{type(exc).__name__}: {exc}"))
    return failures


def prestage_weights():
    """Enkraten prenos utež TabPFN in TabICL (idempotentno - če so že lokalno, ne naredi nič)."""
    # Majhen naključen binarni problem - dovolj, da se prenesejo uteži modelov.
    rng = np.random.default_rng(42)
    X = rng.standard_normal((40, 5))
    y = np.tile([0, 1], 20)

    from tabpfn import TabPFNClassifier

    clf = TabPFNClassifier()
    clf.fit(X, y)
    clf.predict_proba(X[:4])
    print("TabPFN: uteži prenesene in predpomnjene")

    from tabicl import TabICLClassifier

    clf = TabICLClassifier(device="cpu")
    clf.fit(X, y)
    clf.predict_proba(X[:4])
    print("TabICL: uteži prenesene in predpomnjene")


def main():
    parser = argparse.ArgumentParser(description="Predpriprava datasetov in utež na prijavnem vozlišču.")
    parser.add_argument(
        "--ids-file",
        default=os.path.join(REPO_ROOT, "scripts", "subset_ids.json"),
        help="JSON datoteka s seznamom OpenML ID-jev (privzeto scripts/subset_ids.json)",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        type=int,
        help="Neposreden seznam OpenML ID-jev (npr. --ids 31 37); prevlada nad --ids-file.",
    )
    parser.add_argument(
        "--skip-weights",
        action="store_true",
        help="Preskoči prenos utež TabPFN/TabICL - predpripravi samo datasete.",
    )
    args = parser.parse_args()

    if args.ids:
        ids = args.ids
        source = "--ids"
    else:
        with open(args.ids_file) as f:
            ids = [int(i) for i in json.load(f)]
        source = args.ids_file
    print(f"Predpriprava {len(ids)} datasetov iz {source}\n")

    failures = prestage_datasets(ids)
    if args.skip_weights:
        print("\nUteži TabPFN/TabICL preskočene (--skip-weights).")
    else:
        print()
        prestage_weights()

    cache_dir = effective_cache_dir()
    print(f"\nVelikost predpomnilnika OpenML ({cache_dir}): {human_size(dir_size_bytes(cache_dir))}")
    print("(Kvota domačega imenika na Arnesu je 100 GB - preveri z 'du -sh ~' pred oddajo.)")

    if failures:
        print(f"\nNEUSPELI dataseti ({len(failures)}):")
        for openml_id, error in failures:
            print(f"  {openml_id}: {error}")
        print("Računska vozlišča so brez interneta - te datasete pred oddajo predpripravi ročno.")
        sys.exit(1)


if __name__ == "__main__":
    main()
