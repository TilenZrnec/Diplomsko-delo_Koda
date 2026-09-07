"""Požene vse algoritme na enem samem OpenML datasetu (SLURM array job).

Namenjeno gruči Arnes: vsak array task obdela en dataset, tako da se
posamezni dataseti računajo vzporedno. Zagon:

    python -m src.run_one_dataset --index N --ids-file scripts/subset_ids.json

Iz JSON datoteke s seznamom OpenML ID-jev vzame N-ti ID, naloži dataset z
obstoječim load_dataset (isti StratifiedKFold foldi kot lokalno) in zapiše
results/per_dataset/<openml_id>.csv. Če izhodna datoteka že obstaja, izpiše
"already done" in se konča z 0 - ponovni zagon arraya torej preskoči že
narejene datasete.

Kontrolne točke (checkpointing) na nivoju posameznega učenja
------------------------------------------------------------
Vsak (algoritem, fold) par je ena vrstica rezultata. Po vsakem takem učenju
se celoten dosedanji rezultat zapiše v <openml_id>.csv.partial. Če task
prekine SLURM (prekoračen --time) ali preemption, ponovni zagon prebere
partial in preskoči samo tiste (algoritem, fold) pare, ki so že narejeni -
delo se torej nikoli ne izgubi in dolg algoritem (npr. CatBoost na velikem
datasetu) lahko svojih 5 foldov razporedi čez več taskov.

Ključno: končni <openml_id>.csv nastane šele z atomarnim os.replace() iz
partiala, ko so vse vrstice zbrane. Nepopoln rezultat se torej NIKOLI ne
more pretvarjati, da je popoln - "already done" pomeni res dokončano.
Tudi partial se piše atomarno (prek .tmp), da ga prekinitev sredi pisanja
ne pusti okrnjenega in s tem ne pokvari nadaljevanja.
"""

import argparse
import json
import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.data import load_dataset  # noqa: E402
from src.models import REGISTRY  # noqa: E402


def write_partial(rows, partial_path):
    """Atomarno zapiše trenutne vrstice v partial (prek .tmp + os.replace).

    Zapis prek začasne datoteke pomeni, da je partial vedno veljaven CSV -
    prekinitev sredi pisanja ga ne more pustiti okrnjenega in s tem pokvariti
    nadaljevanja ob naslednjem zagonu.
    """
    tmp_path = partial_path + ".tmp"
    pd.DataFrame(rows).to_csv(tmp_path, index=False)
    os.replace(tmp_path, partial_path)


def main():
    parser = argparse.ArgumentParser(description="Zagon vseh algoritmov na enem OpenML datasetu.")
    parser.add_argument("--index", type=int, required=True, help="Indeks ID-ja v datoteki z ID-ji")
    parser.add_argument(
        "--ids-file",
        default="scripts/cc18_ids.json",
        help="JSON datoteka s seznamom OpenML ID-jev (privzeto scripts/cc18_ids.json)",
    )
    args = parser.parse_args()

    with open(args.ids_file) as f:
        ids = json.load(f)
    openml_id = ids[args.index]

    out_dir = os.path.join(REPO_ROOT, "results", "per_dataset")
    out_path = os.path.join(out_dir, f"{openml_id}.csv")
    partial_path = out_path + ".partial"
    if os.path.exists(out_path):
        print("already done")
        return
    os.makedirs(out_dir, exist_ok=True)

    # Nadaljevanje po prekinitvi: kaj je iz prejšnjega taska že narejeno.
    rows = []
    done = set()
    if os.path.exists(partial_path):
        previous = pd.read_csv(partial_path)
        rows = previous.to_dict("records")
        done = {(str(a), int(f)) for a, f in zip(previous["algorithm"], previous["fold"])}
        print(f"Najden partial: {len(done)} učenj že narejenih, nadaljujem.")

    cache_dir = os.path.join(REPO_ROOT, "data", "openml_cache")
    dataset = load_dataset(openml_id, n_splits=5, random_state=42, cache_dir=cache_dir)
    X, y = dataset["X"], dataset["y"]
    categorical_cols = dataset["categorical_cols"]
    ds_name = dataset["name"]
    print(
        f"=== {ds_name} (OpenML ID {openml_id}): {X.shape[0]} vzorcev, "
        f"{X.shape[1]} atributov, {len(categorical_cols)} kategoričnih ==="
    )

    for algo_name, run_fn in REGISTRY.items():
        for fold_idx, (train_idx, test_idx) in enumerate(dataset["folds"]):
            if (algo_name, fold_idx) in done:
                continue

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            result = run_fn(X_train, y_train, X_test, y_test, categorical_cols)

            status = "OK" if result["error"] is None else f"ERROR: {result['error']}"
            auc_str = f"{result['roc_auc']:.4f}" if result["roc_auc"] is not None else "n/a"
            print(f"  [{ds_name}] {algo_name} fold {fold_idx}: roc_auc={auc_str}, {status}")

            rows.append({
                "dataset": ds_name,
                "algorithm": algo_name,
                "fold": fold_idx,
                "roc_auc": result["roc_auc"],
                "train_time_s": result["train_time_s"],
                "inference_time_s": result["inference_time_s"],
                "error": result["error"],
            })

            write_partial(rows, partial_path)

    write_partial(rows, partial_path)
    # Atomarno: <id>.csv se pojavi šele zdaj, ko so vse vrstice zbrane.
    os.replace(partial_path, out_path)
    print(f"Rezultati shranjeni v {out_path} ({len(rows)} vrstic)")


if __name__ == "__main__":
    main()
