"""Skupni motor benchmarka: en nabor, vsi algoritmi, vsi foldi, s kontrolnimi točkami.

Ta modul je EDINO mesto, kjer je zapisana zanka "za vsak algoritem, za vsak
fold: uči, napovej, zapiši". Uporabljata ga oba vstopna skripta:

  * src/run_benchmark.py   - lokalni zagon: zaporedno čez vse nabore iz nabora
  * src/run_one_dataset.py - Arnes: en nabor na SLURM array task

Ker je zanka ena sama, lokalni in gručni zagon ne moreta razhajati (isti
stolpci, ista pravila nadaljevanja, isti zapis napovedi).

Mapa zagona (results/runs/<run_id>/)
------------------------------------
  manifest.json          - kdo/kdaj/s čim: git commit, stroj, GPU, konfiguracija,
                           pip freeze; zapisan ob prvem učenju, nato nedotaknjen
  per_dataset/<id>.csv   - dokončan nabor (nastane šele z atomarnim os.replace)
  per_dataset/<id>.csv.partial - nedokončan nabor (kontrolna točka po vsakem učenju)
  predictions/<id>/<algoritem>_r<ponovitev>_f<fold>.npz - predict_proba vsakega
                           učenja, da se lahko pozneje izračuna poljubna metrika
  results.csv            - združeni per_dataset/*.csv (merge_run)

Kontrolne točke na nivoju posameznega učenja
--------------------------------------------
Vsak (algoritem, fold) par je ena vrstica. Po vsakem učenju se celoten dosedanji
rezultat zapiše v <id>.csv.partial (atomarno prek .tmp). Prekinjen zagon
(prekoračen --time, preemption) ob ponovnem zagonu z ISTIM run_id prebere
partial in preskoči že narejene pare. Končni <id>.csv nastane šele z atomarnim
os.replace(), ko so vse vrstice zbrane - nepopoln rezultat se torej nikoli ne
more pretvarjati, da je popoln, in "already done" pomeni res dokončano.
"""

import datetime as _dt
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import time

import numpy as np
import pandas as pd

from src.config import REPO_ROOT, spec_id
from src.data import load_dataset
from src.models import REGISTRY

# Vrstni red stolpcev v vseh CSV-jih z rezultati. Nove stolpce dodajaj na konec,
# da starejši zagoni ostanejo berljivi z istimi orodji.
RESULT_COLUMNS = [
    "run_id", "dataset", "dataset_id", "algorithm", "repeat", "fold",
    "n_train", "n_test", "n_features", "n_categorical", "n_classes",
    "roc_auc", "train_time_s", "inference_time_s", "warmup_s", "device",
    "preprocessing", "raw_error", "error",
    "git_commit", "hostname", "timestamp",
]

# Algoritmi, ki so v tem procesu že bili ogreti (nalaganje utež, CUDA).
_WARMED = {}


# --------------------------------------------------------------------------- #
# Identiteta zagona                                                            #
# --------------------------------------------------------------------------- #

def make_run_id(dataset_set):
    """Sistematično ime zagona: <datum-čas>_<nabor>_<stroj>, brez ročnega poimenovanja."""
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}_{dataset_set}_{socket.gethostname()}"


def run_dir_for(config, run_id):
    """Mapa zagona pod results_dir; ustvari jo, če je še ni."""
    run_dir = os.path.join(config["results_dir"], run_id)
    os.makedirs(os.path.join(run_dir, "per_dataset"), exist_ok=True)
    return run_dir


def git_info():
    """(kratek commit hash, ali je delovno drevo umazano) - ali ('unknown', None) brez gita."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        )
        return commit, bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None


def pip_freeze():
    """Seznam 'paket==različica' iz nameščenih distribucij (brez klica pip)."""
    seen = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name and name not in seen:
            seen[name] = dist.version
    return [f"{name}=={version}" for name, version in sorted(seen.items(), key=lambda kv: kv[0].lower())]


def _gpu_info():
    try:
        import torch

        info = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu"] = torch.cuda.get_device_name(0)
        return info
    except ImportError:
        return {"torch": None, "cuda_available": False}


def _atomic_write_json(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def write_manifest(run_dir, config, dataset_set, specs, run_id):
    """Zapiše manifest.json, če ga še ni; vrne pot.

    Manifest nastane ob prvem učenju v zagonu (na Arnesu ga zapiše prvi array
    task, ki pride do tega mesta) in se potem ne spreminja. Vsaka vrstica
    rezultatov vseeno nosi svoj git_commit in hostname, ker lahko posamezni
    nabori tečejo na drugih vozliščih ali po popravku kode.
    """
    path = os.path.join(run_dir, "manifest.json")
    if os.path.exists(path):
        return path
    commit, dirty = git_info()
    ids_file = config["dataset_sets"].get(dataset_set)
    payload = {
        "run_id": run_id,
        "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_set": dataset_set,
        "ids_file": os.path.relpath(ids_file, REPO_ROOT) if ids_file else None,
        "n_datasets": len(specs),
        "datasets": specs,
        "git_commit": commit,
        "git_dirty": dirty,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID"),
        "cpu_threads": os.environ.get("OMP_NUM_THREADS") or os.cpu_count(),
        **_gpu_info(),
        "config": {k: v for k, v in config.items() if k != "config_path"},
        "pip_freeze": pip_freeze(),
    }
    _atomic_write_json(path, payload)
    return path


# --------------------------------------------------------------------------- #
# Zapis rezultatov                                                             #
# --------------------------------------------------------------------------- #

def write_partial(rows, partial_path):
    """Atomarno zapiše trenutne vrstice v partial (prek .tmp + os.replace).

    Zapis prek začasne datoteke pomeni, da je partial vedno veljaven CSV -
    prekinitev sredi pisanja ga ne more pustiti okrnjenega in s tem pokvariti
    nadaljevanja ob naslednjem zagonu.
    """
    tmp_path = partial_path + ".tmp"
    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(tmp_path, index=False)
    os.replace(tmp_path, partial_path)


def save_predictions(run_dir, ds_id, algo_name, repeat, fold, result, y_test, test_idx, classes):
    """Shrani predict_proba enega učenja v .npz (float32, da je datoteka pol manjša)."""
    if result.get("proba") is None:
        return
    pred_dir = os.path.join(run_dir, "predictions", ds_id)
    os.makedirs(pred_dir, exist_ok=True)
    path = os.path.join(pred_dir, f"{algo_name}_r{repeat}_f{fold}.npz")
    tmp = path + ".tmp.npz"
    np.savez_compressed(
        tmp,
        proba=np.asarray(result["proba"], dtype=np.float32),
        y_test=np.asarray(y_test, dtype=np.int32),
        test_idx=np.asarray(test_idx, dtype=np.int64),
        classes=np.asarray(classes),
    )
    os.replace(tmp, path)


def merge_run(run_dir, output_path=None, log=print):
    """Združi per_dataset/*.csv v results.csv; *.partial glasno našteje, a jih ne vključi."""
    per_dataset = os.path.join(run_dir, "per_dataset")
    paths = sorted(p for p in _glob_csv(per_dataset))
    partials = sorted(
        os.path.join(per_dataset, f) for f in os.listdir(per_dataset) if f.endswith(".partial")
    ) if os.path.isdir(per_dataset) else []
    if partials:
        log(f"OPOZORILO: {len(partials)} nedokončanih naborov (*.partial) - NISO v merge:")
        for path in partials:
            n_done = len(pd.read_csv(path))
            log(f"  {os.path.basename(path)}: {n_done} učenj narejenih; poženi znova z istim run_id")
    if not paths:
        log(f"Ni dokončanih naborov v {per_dataset}")
        return None
    merged = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    output_path = output_path or os.path.join(run_dir, "results.csv")
    merged.to_csv(output_path, index=False)
    n_errors = merged["error"].notna().sum() if "error" in merged.columns else 0
    log(f"Združenih {len(paths)} naborov -> {output_path}")
    log(f"Vrstic: {len(merged)}, naborov: {merged['dataset'].nunique()}, vrstic z napako: {n_errors}")
    return output_path


def _glob_csv(directory):
    if not os.path.isdir(directory):
        return []
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".csv")]


# --------------------------------------------------------------------------- #
# Ogrevanje                                                                    #
# --------------------------------------------------------------------------- #

def warmup(algo_name, module, random_state, log=print):
    """Enkrat na proces: majhen sintetični fit, da prvi merjeni fold ne vsebuje
    nalaganja utež na GPU, inicializacije CUDA in prevajanja jeder.

    Vrne porabljeni čas v sekundah (gre v stolpec warmup_s), ali None, če
    algoritem ogrevanja ne potrebuje (USES_GPU = False).
    """
    if not getattr(module, "USES_GPU", False):
        return None
    if algo_name in _WARMED:
        return _WARMED[algo_name]
    rng = np.random.default_rng(random_state)
    X = pd.DataFrame(rng.standard_normal((64, 4)), columns=[f"f{i}" for i in range(4)])
    y = pd.Series(np.tile([0, 1], 32))
    t0 = time.perf_counter()
    result = module.run(X.iloc[:48], y.iloc[:48], X.iloc[48:], y.iloc[48:], [], random_state)
    elapsed = time.perf_counter() - t0
    status = "OK" if result["error"] is None else f"NAPAKA: {result['error']}"
    log(f"  ogrevanje {algo_name}: {elapsed:.1f} s, {status}")
    _WARMED[algo_name] = elapsed
    return elapsed


# --------------------------------------------------------------------------- #
# Glavna zanka                                                                 #
# --------------------------------------------------------------------------- #

def run_dataset(spec, config, run_dir, run_id, algorithms=None, log=print):
    """Požene vse algoritme na vseh foldih enega nabora; vrne pot do <id>.csv.

    spec        - opis nabora (OpenML ID ali slovar za CSV), glej src.data
    config      - slovar iz src.config.load_config()
    run_dir     - mapa zagona (run_dir_for)
    algorithms  - seznam imen; privzeto config["algorithms"]
    """
    algorithms = algorithms or config["algorithms"]
    unknown = [a for a in algorithms if a not in REGISTRY]
    if unknown:
        raise KeyError(f"Neznani algoritmi v config.yaml: {unknown}; znani: {sorted(REGISTRY)}")

    ds_id = spec_id(spec)
    per_dataset = os.path.join(run_dir, "per_dataset")
    os.makedirs(per_dataset, exist_ok=True)
    out_path = os.path.join(per_dataset, f"{ds_id}.csv")
    partial_path = out_path + ".partial"
    if os.path.exists(out_path):
        log(f"[{ds_id}] already done ({out_path})")
        return out_path

    # Nadaljevanje po prekinitvi: kaj je iz prejšnjega zagona že narejeno.
    rows = []
    done = set()
    if os.path.exists(partial_path):
        previous = pd.read_csv(partial_path)
        rows = previous.to_dict("records")
        done = {(str(a), int(f)) for a, f in zip(previous["algorithm"], previous["fold"])}
        log(f"[{ds_id}] najden partial: {len(done)} učenj že narejenih, nadaljujem.")

    n_splits = config["n_splits"]
    n_repeats = config["n_repeats"]
    random_state = config["random_state"]
    dataset = load_dataset(
        spec, n_splits=n_splits, random_state=random_state,
        cache_dir=config["cache_dir"], n_repeats=n_repeats,
    )
    X, y = dataset["X"], dataset["y"]
    categorical_cols = dataset["categorical_cols"]
    ds_name = dataset["name"]
    n_classes = int(y.nunique())
    log(
        f"=== {ds_name} ({ds_id}): {X.shape[0]} vzorcev, {X.shape[1]} atributov, "
        f"{len(categorical_cols)} kategoričnih, {n_classes} razredov, "
        f"{len(dataset['folds'])} foldov ==="
    )

    commit, _dirty = git_info()
    hostname = socket.gethostname()

    for algo_name in algorithms:
        module = REGISTRY[algo_name]
        pending = [k for k in range(len(dataset["folds"])) if (algo_name, k) not in done]
        if not pending:
            continue
        warmup_s = warmup(algo_name, module, random_state, log=log)

        for fold_idx, (train_idx, test_idx) in enumerate(dataset["folds"]):
            if (algo_name, fold_idx) in done:
                continue
            repeat = fold_idx // n_splits
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            result = module.run(X_train, y_train, X_test, y_test, categorical_cols, random_state)

            status = "OK" if result["error"] is None else f"ERROR: {result['error']}"
            auc_str = f"{result['roc_auc']:.4f}" if result["roc_auc"] is not None else "n/a"
            log(f"  [{ds_name}] {algo_name} fold {fold_idx}: roc_auc={auc_str}, {status}")

            if config.get("save_predictions", True):
                save_predictions(
                    run_dir, ds_id, algo_name, repeat, fold_idx, result,
                    y_test, test_idx, dataset["classes"],
                )

            rows.append({
                "run_id": run_id,
                "dataset": ds_name,
                "dataset_id": ds_id,
                "algorithm": algo_name,
                "repeat": repeat,
                "fold": fold_idx,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_features": int(X.shape[1]),
                "n_categorical": int(len(categorical_cols)),
                "n_classes": n_classes,
                "roc_auc": result["roc_auc"],
                "train_time_s": result["train_time_s"],
                "inference_time_s": result["inference_time_s"],
                "warmup_s": warmup_s,
                "device": result.get("device"),
                "preprocessing": result.get("preprocessing"),
                "raw_error": result.get("raw_error"),
                "error": result["error"],
                "git_commit": commit,
                "hostname": hostname,
                "timestamp": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            })
            write_partial(rows, partial_path)

    write_partial(rows, partial_path)
    # Atomarno: <id>.csv se pojavi šele zdaj, ko so vse vrstice zbrane.
    os.replace(partial_path, out_path)
    log(f"[{ds_id}] rezultati shranjeni v {out_path} ({len(rows)} vrstic)")
    return out_path
