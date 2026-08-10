# Diplomsko-delo_Koda — Diploma Project (FRI)
#  Style
Explain things to me as if i am a compleete beginner.

## Purpose
Benchmarking 6 classical/foundation-model tabular ML algorithms on OpenML
datasets as part of a diploma thesis at FRI (Faculty of Computer and
Information Science, Ljubljana), with default hyperparameters throughout.
Comments and docstrings in the codebase are written in Slovenian. Currently
3 datasets; designed to extend to ~72 datasets later without restructuring.

## Structure
- `config.yaml` — single source of config: dataset OpenML IDs, `n_splits`,
  `random_state`, algorithm list, output paths.
- `src/data.py` — `load_dataset(openml_id, ...)` downloads/caches a dataset
  from OpenML and builds the 5-fold `StratifiedKFold` splits **once**; every
  algorithm reuses the same fold indices for a given dataset.
- `src/utils.py` — `compute_roc_auc()`, binary vs. multiclass aware.
- `src/models/` — one file per algorithm, each exposing a uniform
  `run(X_train, y_train, X_test, y_test, categorical_cols) -> dict` returning
  `{"model", "roc_auc", "train_time_s", "inference_time_s", "error",
  "preprocessing", "raw_error"}`. `src/models/__init__.py` exposes `REGISTRY`
  mapping config algorithm names to `run` functions.
  Implemented: `random_forest.py`, `xgboost_model.py`, `lightgbm_model.py`,
  `catboost_model.py`, `tabpfn_model.py`, `tabicl_model.py`.
- `src/run_benchmark.py` — orchestrates: load each dataset, run every
  algorithm on every fold, write `results/results.csv` and
  `results/preprocessing_log.md`. Run via `python -m src.run_benchmark`.
- `src/summary.py` — prints mean ROC-AUC ± std per (dataset, algorithm) and
  mean ROC-AUC ± std / mean rank per algorithm across datasets. Run via
  `python -m src.summary [results_csv]`; the optional path defaults to
  `results/results.csv` (the local 3-dataset pilot), so pass
  `results/results_arnes_cc18.csv` for the full sweep. It echoes `Vir: <path>`
  first so the summarised file is never ambiguous.
- `results/` — `results.csv` (columns: dataset, algorithm, fold, roc_auc,
  train_time_s, inference_time_s) and `preprocessing_log.md` (what
  preprocessing each algorithm × dataset combo required, and any raw-input
  errors). Both are generated, not hand-edited.
- `data/openml_cache/` — OpenML's local dataset cache (gitignored).

## Preprocessing policy (per algorithm — this is a deliberate experimental variable)
- **RandomForest**: no native NaN/categorical support → median imputation
  (numeric) + most-frequent imputation & ordinal encoding (categorical),
  fit on train fold only.
- **XGBoost**: native NaN handling; categorical columns ordinal-encoded to
  numeric codes with NaN preserved (`OrdinalEncoder(encoded_missing_value=np.nan)`).
- **LightGBM**: native NaN handling; categorical columns cast to pandas
  `category` dtype for native categorical support. Non-categorical columns
  are explicitly cast to float, since some nominally-numeric OpenML columns
  (e.g. an all-missing column) load as `object` dtype and LightGBM rejects
  non-numeric/category dtypes.
- **CatBoost**: native NaN handling (numeric) + native categorical support
  via `cat_features`; categorical columns cast to `str` first (CatBoost
  disallows float NaN inside categorical columns — casting makes `'nan'` its
  own category, not an imputation).
- **TabPFN / TabICL**: pass data as raw as possible by design — the
  experiment is to see what each model handles natively. Each model's
  `run()` tries the raw fold first; only on an exception does it apply a
  logged minimal fix and retry:
  - TabPFN (v3, `tabpfn` 8.x): raw input works on all current datasets,
    including `object`-dtype columns with unparsable/all-missing values
    (v2 needed an ordinal-encoding fallback there — kept as a safety net
    for future datasets, NaN preserved).
  - TabICL: raw input works. Under torch 2.4.1, `device='auto'` failed to
    resolve a torch device index in this WSL2/CUDA setup and a fallback to
    explicit `device='cuda:0'` was needed; fixed by the torch 2.13 upgrade,
    fallback kept as a safety net.
  Never silently preprocess — every raw failure + the fix applied is
  recorded in `results/preprocessing_log.md`.

## Environment
- Runs under WSL2 (Ubuntu) on Windows 11; the repo lives on the WSL **Linux**
  filesystem at `/home/tilen/fri/diplomska/Diplomsko-delo_Koda` (moved off the
  Windows mount `/mnt/d/fri/Diplomska/prvo_testiranje` on 2026-08-10, and
  renamed from `prvo_testiranje`). Any older reference to `/mnt/d/...` or to
  "prvo_testiranje" is stale.
- **Two machines** (desktop PC and laptop) use the *same* path
  `/home/tilen/fri/diplomska/Diplomsko-delo_Koda`, kept in sync through GitHub:
  `origin = https://github.com/TilenZrnec/Diplomsko-delo_Koda.git`, branch
  `main`. The parent `~/fri/diplomska/` also holds the thesis text
  (`Diplomsko_delo_Tabelarični_temeljni_modeli/`) and the CRISP-DM documents,
  which are *not* part of this repo. Sync discipline: commit + push before
  switching machines, pull on arrival — nothing here is shared live.
- Not carried by git, therefore per-machine and to be set up on each:
  the conda env `tabular`, the TabPFN credential (`~/.cache/tabpfn/`), the
  OpenML cache (`~/.cache/openml/`, see the caveat below), and
  `.claude/settings.local.json` (globally gitignored).
- Conda env: `tabular` (Python 3.10). GPU: NVIDIA RTX 3060, `torch` 2.13.0+cu130,
  `torch.cuda.is_available()` is `True`. Run project scripts with
  `conda run -n tabular python -m src.<module>` (or activate the env first).
- TabPFN v3 (`tabpfn` 8.x) requires a one-time license acceptance via a
  PriorLabs account; the credential is cached locally on this machine. On a
  fresh machine: interactive first `fit()` opens a browser login (needs a
  real TTY — `conda activate`, not `conda run`), or set `TABPFN_TOKEN` from
  https://ux.priorlabs.ai/account for headless use.
- `src/data.py` resolves the OpenML cache path relative to the repo root
  (`data/openml_cache`), not hardcoded — safe if the repo is moved.
  **Caveat (verified 2026-07-22):** under `openml` 0.14.2 the assignment
  `openml.config.cache_directory = ...` is a no-op — the library reads
  `_root_cache_directory` (settable only via
  `openml.config.set_root_cache_directory()`), so the cache actually lands in
  `~/.cache/openml/org/openml/www` and `data/openml_cache` stays empty. Left
  as-is deliberately (changing it would move the cache mid-project); it only
  matters for **disk-quota accounting on Arnes**, where the cache counts
  against the 100 GB home quota. `scripts/prestage.py` therefore measures the
  *effective* directory via `openml.config.get_cache_directory()`.

## Datasets (OpenML IDs, set in `config.yaml`)
- 31 — credit-g (1000 rows, 20 attrs, 13 categorical, no missing values)
- 37 — diabetes / Pima (768 rows, 8 attrs, all numeric, no missing values)
- 38 — sick (3772 rows, 29 attrs, 22 categorical, has missing values).
  **Note**: the originally-specified ID 3021 does not exist on OpenML
  ("Unknown dataset"); 38 is the standard "sick" thyroid dataset and was
  confirmed with the user as the intended substitute.

## Arnes HPC
- The benchmark also runs on the Arnes SLURM cluster (GPU partition `gpu`,
  H100 nodes). Env: micromamba prefix at `~/envs/tabular` (micromamba binary
  at `~/bin/micromamba`, no shell activation hooks in batch scripts).
- TabPFN token lives in `~/.tabpfn_token` (sourced by the batch script);
  compute nodes run offline (`HF_HUB_OFFLINE=1`), so run
  `python scripts/prestage.py [--ids-file FILE]` on the login node first — it
  caches every dataset in the ids-file (default `scripts/subset_ids.json` =
  31/37/38) plus the TabPFN/TabICL weights (TabICL on CPU), and prints the
  resulting cache size for the 100 GB home-quota check. Per-dataset download
  failures don't abort it; they're listed at the end and it exits 1.
- `src/run_one_dataset.py --index N --ids-file scripts/subset_ids.json` runs
  all REGISTRY algorithms on the Nth OpenML ID and writes
  `results/per_dataset/<openml_id>.csv`; exits early ("already done") if the
  file exists, so re-submitted arrays skip finished datasets.
- **Per-fit checkpointing.** After every single (algorithm, fold) fit,
  `run_one_dataset.py` rewrites `<openml_id>.csv.partial`; the final
  `<openml_id>.csv` is produced only at the end via **atomic `os.replace()`**.
  Two consequences: a killed task (`--time` overrun, preemption) loses no
  work — the next submission reads the partial and skips only the fits
  already recorded — and a partial can never masquerade as a complete result,
  so "already done" always means genuinely done. Granularity is per *fit*,
  not per algorithm, so one slow algorithm (CatBoost on Devnagari-Script)
  can spread its 5 folds across several tasks instead of restarting from
  fold 0 forever. The partial itself is also written atomically (via `.tmp`),
  so a kill mid-write can't corrupt it. Verified 2026-07-22 with a kill/resume
  test: killed at 9/30 fits → no `38.csv` present, only the partial; resumed
  → 30 rows, identical row order, **max Δ 0.0** vs. a clean single-shot run;
  re-run after completion → "already done".
- Submit from the repo root: `sbatch scripts/run_subset.sh` (array 0-2; fill
  `--account`/`--reservation` from `sacctmgr show assoc user=$USER`).
- `scripts/merge_results.py <out.csv> [--input-dir DIR]` concatenates
  per-dataset CSVs (default `results/per_dataset/`);
  `scripts/compare_results.py <local.csv> <arnes.csv>` diffs ROC-AUC.
  The merge globs `*.csv` only, so unfinished `*.partial` datasets are
  excluded — and it prints a loud warning listing each one with its
  fits-done count, so an incomplete sweep surfaces instead of hiding.
- `scripts/profile_datasets.py --ids-file FILE [--top N] [--from-cache]`
  prints n_rows × n_features per dataset, sorted descending — the data-driven
  input for sizing `--time`/`--mem`. Default reads OpenML metadata (needs
  internet, no full download); `--from-cache` loads the cached datasets.

### Full CC18 sweep (72 datasets) — **DONE**, do not re-run
**Status: complete.** The thesis result set is `results/results_arnes_cc18.csv`
— 2160 rows = 72 datasets × 6 algorithms × 5 folds, verified 2026-08-10.
Ten fits failed, all of them CIFAR_10 (OpenML 40927) × {TabPFN, TabICL} × 5
folds: TabICL asked for ~378 GB against 256 GB/node. Those rows exist with the
reason in `error` and an empty `roc_auc` — no subsampling and no
`disk_offload_dir` were used, so the default-hyperparameter protocol is intact.
Full provenance (4 SLURM job IDs, the mid-run script fixes, the caveat that
`train_time_s`/`inference_time_s` for datasets 554/40923/40927/40996 come from
re-runs on different nodes and are therefore *not* cross-dataset comparable) is
in `results/arnes_cc18/PROVENANCE.md`.

Headline numbers (mean ROC-AUC / mean rank across all 72): tabicl 0.9396/1.75,
tabpfn 0.9384/2.05, catboost 0.9280/3.43, lightgbm 0.9218/4.37, xgboost
0.9208/4.42, random_forest 0.9180/4.88. Reproduce with
`python -m src.summary results/results_arnes_cc18.csv` — **the path argument is
required**; bare `python -m src.summary` summarises the 3-dataset local pilot
instead, which is the wrong table for the thesis.

The sizing TODOs still written in `scripts/run_cc18.sh` are answered by the run
itself: `--mem=64G` was too small for indices 27/60/61/70 (mnist_784,
Devnagari-Script, CIFAR_10, Fashion-MNIST) and those needed 120–240G;
`--time=12:00:00` sufficed everywhere.

The procedure below is retained for reproducibility and for any future
re-run — **it is a record of what was done, not pending work.**
1. **Pin the ID set** (once, already done and committed):
   `python scripts/gen_cc18_ids.py` → `scripts/cc18_ids.json`, 72 IDs from
   `openml.study.get_suite(99)`, deduped and sorted; hard-fails if OpenML
   returns ≠ 72 so a silent suite change can't slip through. The file is
   committed, so the sweep does not re-fetch the suite at submit time.
2. **Prestage on the login node** (internet):
   `python scripts/prestage.py --ids-file scripts/cc18_ids.json`, then check
   the printed cache size and `du -sh ~` against the 100 GB quota.
3. **Clear scratch, then submit**: `rm -rf results/per_dataset/*` (**all** of
   it — `.csv`, `.partial` and `.partial.tmp`; a stale partial from an older
   code version would otherwise poison a resume) and
   `sbatch scripts/run_cc18.sh` (array `0-71%4`, now `--time=12:00:00`,
   `--mem=64G`, otherwise identical plumbing to `run_subset.sh`). Re-submit the
   four big indices separately with more memory:
   `ALLOW_SPARSE_ARRAY=1 sbatch --array=27,60,61,70 --mem=240G scripts/run_cc18.sh`.
4. **Merge and summarise**:
   `python scripts/merge_results.py results/results_arnes_cc18.csv` then
   `python -m src.summary results/results_arnes_cc18.csv` — pass the path, or
   you silently get the pilot. Never merge into `results/results.csv`.
5. **Completeness check**: expect **72 × 6 × 5 = 2160** rows. Fewer rows, or
   any `*.partial` left in `results/per_dataset/` (the merge lists them), means
   those datasets hit the wall — raise `--time` and re-submit; they resume
   from where they stopped. A fit that *failed* still produces its row, with
   the reason in `error`, so missing rows always mean "unfinished", never
   "failed".
- **Resume / re-submit**: the array is resume-safe at two levels — whole
  datasets with a `<id>.csv` print "already done" and cost seconds, and a
  dataset interrupted mid-way resumes from its partial at the first unfinished
  (algorithm, fold) fit. So a task killed by a too-short `--time` is fixed by
  raising `--time` and submitting again, with no lost compute. The script also
  verifies that the `--array` upper bound matches the ID count and aborts
  loudly if it doesn't.
- **CC18 contains the pilot datasets 31/37/38**, so the resume logic *would*
  skip them if their CSVs are still in `results/per_dataset/`. Two options:
  (a) knowingly reuse the validated pilot CSVs, or (b) `rm -rf
  results/per_dataset/*` first, so all 72 datasets come from one code
  version and one environment. **(b) is the recommendation for the thesis
  run** — single-version provenance; the pilot results stay archived in
  `results/arnes_subset/`.
- TabPFN/TabICL were expected to fail-soft on the four largest CC18 datasets
  (CIFAR_10 60000×3072, Devnagari-Script 92000×1024, mnist_784 and
  Fashion-MNIST 70000×784). **Outcome: only CIFAR_10 actually broke** — the
  other three completed once given 120–240G. No subsampling is implemented and
  none was added; the 10 CIFAR_10 rows carry their reason in `error`.

### Operational lessons (learned 2026-07-22, validation run)
- **The cluster remote uses SSH**, not HTTPS: `git@github.com:...`. The
  cluster has no credential helper for HTTPS, so an `https://` remote
  prompts for a password and fails in a batch context.
- **Scratch vs. curated results.** `results/per_dataset/` is *scratch*: it is
  gitignored, and `run_one_dataset.py` treats an existing
  `<openml_id>.csv` there as "already done" and skips the dataset. Therefore
  **`rm -rf results/per_dataset/*` before any real run** — otherwise stale
  files silently suppress recomputation. This actually happened: a
  git-committed local CSV made array task 0 skip, contaminating the first
  attempt with local numbers presented as cluster numbers. Curated,
  publishable results are committed under their own directory (e.g.
  `results/arnes_subset/`) with a `PROVENANCE.md`.
- **Merges always write to a new file.** Never merge into or overwrite
  `results/results.csv` — it is the immutable local RTX 3060 baseline.
  Cluster merges go to `results/results_arnes_subset.csv` and similar.
- **Job receipts via `sacct`.** After every cluster run, record the receipt:
  `sacct -j <jobid> --format=JobID,JobName%20,Elapsed,MaxRSS,State,NodeList`.
  Elapsed/MaxRSS per task go into the run's `PROVENANCE.md` — they are the
  input for sizing `--time` and `--mem` on the larger CC18 array.
- Validation outcome: cluster vs. local agreement is exact (bit-identical)
  for RF/XGBoost/LightGBM/CatBoost and ~1e-4 for TabPFN/TabICL (GPU
  nondeterminism, SM86 vs SM90). See `results/arnes_subset/PROVENANCE.md`.

## Conventions
- Each model module fails soft: exceptions are caught and stored in
  `result["error"]` rather than raised, so one failing (dataset, algorithm,
  fold) combo doesn't crash the full benchmark run.
- `random_state=42` used consistently for reproducibility (data split, CV
  folds, and every model's `random_state`).
- All algorithms use **default hyperparameters** — this is intentional per
  the thesis protocol, not an oversight to "fix".
- **One deliberate non-default: `RandomForestClassifier(n_jobs=-1)`.** This is
  a compute setting, not a hyperparameter — trees are independent, so it
  changes only wall-clock, never the fitted model. Verified 2026-07-22:
  `predict_proba` bit-identical serial vs. parallel on synthetic 20000×300 and
  20000×800, and all 30 `sick` rows reproduce `results/results.csv` exactly
  (max Δ 0.0 across all six algorithms) with the flag in place — so the pilot's
  ALL PASS verdict is unaffected. Without it RF was the only one of the four
  ensembles running single-core: XGBoost, LightGBM and CatBoost all default to
  every available core (capped by the batch script's `OMP_NUM_THREADS=8`).
  Measured speedup 8.1× (20000×300) and 6.9× (20000×800) on 12 cores; on tiny
  datasets it costs ~0.1–1 s of thread-pool overhead per dataset, which is the
  right trade against hours on the giant CC18 sets. `-1` resolves through
  joblib, which honours the SLURM cpuset/cgroup allocation, so it means 8 cores
  on the cluster, not the whole node.
