"""Izpiše velikosti naborov iz izbranega nabora, urejene padajoče.

Namen je podatkovno podprta izbira SLURM parametrov --time in --mem: največji
nabori v zbirki določajo, koliko časa in pomnilnika potrebuje posamezen
array task. Rangira po številu celic (vrstice x atributi).

Privzeto bere metapodatke iz OpenML (hitro, brez prenosa celih naborov -
potrebuje internet). Z --from-cache namesto tega naloži že predpomnjene
nabore prek load_dataset in izpiše dejanske oblike DataFrameov; to je tudi
edina pot za lokalne CSV nabore (npr. medic3), ki na OpenML niso.

Zagon:

    python scripts/profile_datasets.py --dataset-set cc18
    python scripts/profile_datasets.py --dataset-set cc18 --from-cache
    python scripts/profile_datasets.py --dataset-set medic3 --from-cache
"""

import argparse
import os
import sys

import openml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.config import dataset_specs, load_config, spec_id  # noqa: E402
from src.data import load_dataset  # noqa: E402


def profile_from_metadata(specs):
    """Vrne vrstice (id, ime, vrstice, atributi, kategorični, razredi) iz OpenML metapodatkov."""
    ids = [int(s) if isinstance(s, int) else int(s["id"]) for s in specs
           if isinstance(s, int) or s.get("source", "openml") == "openml"]
    listing = openml.datasets.list_datasets(data_id=ids, output_format="dataframe")
    listing = listing.set_index("did") if "did" in listing.columns else listing
    rows = []
    for spec in specs:
        ds_id = spec_id(spec)
        if not ds_id.isdigit() or int(ds_id) not in listing.index:
            rows.append((ds_id, "? (ni na OpenML - uporabi --from-cache)", None, None, None, None))
            continue
        meta = listing.loc[int(ds_id)]
        rows.append((
            ds_id,
            meta["name"],
            int(meta["NumberOfInstances"]),
            # NumberOfFeatures šteje tudi ciljno spremenljivko, X ima en stolpec manj.
            int(meta["NumberOfFeatures"]) - 1,
            int(meta["NumberOfSymbolicFeatures"]) - 1,
            int(meta["NumberOfClasses"]),
        ))
    return rows


def profile_from_cache(specs, config):
    """Vrne iste vrstice, a iz dejansko naloženih (predpomnjenih) naborov."""
    rows = []
    for spec in specs:
        dataset = load_dataset(
            spec, n_splits=config["n_splits"], random_state=config["random_state"],
            cache_dir=config["cache_dir"], n_repeats=config["n_repeats"],
        )
        X = dataset["X"]
        rows.append((
            spec_id(spec),
            dataset["name"],
            X.shape[0],
            X.shape[1],
            len(dataset["categorical_cols"]),
            int(dataset["y"].nunique()),
        ))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Velikosti naborov za dimenzioniranje SLURM virov.")
    parser.add_argument("--dataset-set", default="cc18", help="Ime nabora iz config.yaml (privzeto cc18)")
    parser.add_argument(
        "--from-cache", action="store_true",
        help="Naloži predpomnjene nabore namesto branja metapodatkov z OpenML",
    )
    parser.add_argument("--top", type=int, default=0, help="Izpiši le N največjih (0 = vse)")
    args = parser.parse_args()

    config = load_config()
    specs = dataset_specs(config, args.dataset_set)

    rows = profile_from_cache(specs, config) if args.from_cache else profile_from_metadata(specs)
    # Rangiranje po celicah; nabori brez metapodatkov gredo na konec.
    rows.sort(key=lambda r: (r[2] or 0) * (r[3] or 0), reverse=True)
    if args.top:
        rows = rows[: args.top]

    print(f"{'id':>8}  {'ime':<26} {'vrstice':>9} {'atributi':>9} {'kateg.':>7} {'razredi':>8} {'celice':>12}")
    total_cells = 0
    for ds_id, name, n_rows, n_feat, n_cat, n_cls in rows:
        cells = (n_rows or 0) * (n_feat or 0)
        total_cells += cells
        print(
            f"{ds_id:>8}  {name[:26]:<26} {n_rows if n_rows is not None else '?':>9} "
            f"{n_feat if n_feat is not None else '?':>9} {n_cat if n_cat is not None else '?':>7} "
            f"{n_cls if n_cls is not None else '?':>8} {cells:>12,}"
        )
    print(f"\nSkupaj {len(rows)} naborov, {total_cells:,} celic")


if __name__ == "__main__":
    main()
