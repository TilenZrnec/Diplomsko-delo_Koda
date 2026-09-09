"""Predpriprava na prijavnem vozlišču Arnes (login node nima GPU-ja).

Prenese nabore v cache_dir (config.yaml) ter enkrat fitta TabPFN in TabICL na
majhnem naključnem vzorcu, da se uteži modelov preneseta v lokalni
predpomnilnik - računska vozlišča nato tečejo brez dostopa do interneta
(HF_HUB_OFFLINE=1).

Zagon (samo na prijavnem vozlišču, potrebuje internet):

    python scripts/prestage.py                          # pilotna trojica (subset)
    python scripts/prestage.py --dataset-set cc18       # cel CC18
    python scripts/prestage.py --ids 31 37 38           # izbrani OpenML ID-ji
    python scripts/prestage.py --ids 37 --skip-weights  # samo nabor, brez utež

--skip-weights preskoči TabPFN/TabICL - uporabno za preverjanje samega
predpomnilnika naborov na računalniku brez poverilnice TabPFN.

Prenos posameznega nabora ne prekine celotne predpriprave - napake se zberejo
in izpišejo na koncu, izhodna koda pa je 1, če kateri nabor manjka. Na koncu
izpiše skupno velikost predpomnilnika, da jo je mogoče primerjati s kvoto
domačega imenika (100 GB) pred oddajo SLURM polja.
"""

import argparse
import os
import sys

import numpy as np
import openml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.config import dataset_specs, load_config, spec_id  # noqa: E402
from src.data import load_dataset  # noqa: E402


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


def prestage_datasets(specs, config):
    """Prenese in predpomni vse nabore; vrne seznam (id, napaka) za neuspele."""
    failures = []
    for i, spec in enumerate(specs, start=1):
        ds_id = spec_id(spec)
        try:
            dataset = load_dataset(
                spec, n_splits=config["n_splits"], random_state=config["random_state"],
                cache_dir=config["cache_dir"], n_repeats=config["n_repeats"],
            )
            X = dataset["X"]
            print(
                f"[{i}/{len(specs)}] {ds_id} ({dataset['name']}): "
                f"{X.shape[0]} x {X.shape[1]}, preneseno in predpomnjeno"
            )
        except Exception as exc:  # noqa: BLE001 - predpriprava ne sme pasti zaradi enega nabora
            print(f"[{i}/{len(specs)}] {ds_id}: NAPAKA - {type(exc).__name__}: {exc}")
            failures.append((ds_id, f"{type(exc).__name__}: {exc}"))
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
    parser = argparse.ArgumentParser(description="Predpriprava naborov in utež na prijavnem vozlišču.")
    parser.add_argument("--dataset-set", default="subset", help="Ime nabora iz config.yaml (privzeto subset)")
    parser.add_argument(
        "--ids", nargs="+", type=int,
        help="Neposreden seznam OpenML ID-jev (npr. --ids 31 37); prevlada nad --dataset-set.",
    )
    parser.add_argument(
        "--skip-weights", action="store_true",
        help="Preskoči prenos utež TabPFN/TabICL - predpripravi samo nabore.",
    )
    args = parser.parse_args()

    config = load_config()
    if args.ids:
        specs, source = list(args.ids), "--ids"
    else:
        specs, source = dataset_specs(config, args.dataset_set), args.dataset_set
    print(f"Predpriprava {len(specs)} naborov iz {source}\n")

    failures = prestage_datasets(specs, config)
    if args.skip_weights:
        print("\nUteži TabPFN/TabICL preskočene (--skip-weights).")
    else:
        print()
        prestage_weights()

    cache_dir = effective_cache_dir()
    print(f"\nVelikost predpomnilnika OpenML ({cache_dir}): {human_size(dir_size_bytes(cache_dir))}")
    print("(Kvota domačega imenika na Arnesu je 100 GB - preveri z 'du -sh ~' pred oddajo.)")

    if failures:
        print(f"\nNEUSPELI nabori ({len(failures)}):")
        for ds_id, error in failures:
            print(f"  {ds_id}: {error}")
        print("Računska vozlišča so brez interneta - te nabore pred oddajo predpripravi ročno.")
        sys.exit(1)


if __name__ == "__main__":
    main()
