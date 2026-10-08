# Diplomsko-delo_Koda — Diploma Project (FRI)
#  Style
Explain things to me as if i am a compleete beginner.

## Purpose
Benchmarking 6 classical/foundation-model tabular ML algorithms on OpenML
datasets (and the private Medic3 dataset) as part of a diploma thesis at FRI
(Faculty of Computer and Information Science, Ljubljana), with default
hyperparameters throughout. Comments and docstrings in the codebase are
written in Slovenian.

**State on 2026-10-05:** new library freeze (dated 2026-10-05, conda env
`tabular3.5`): `tabpfn` 8.5.0 → 9.1.0, i.e. **TabPFN-3 → TabPFN-3.5**, plus patch
releases of torch, pandas, scikit-learn, matplotlib, tqdm and the new pin
`skrub`. Still **six algorithms**: the config key `tabpfn` now means TabPFN-3.5,
TabPFN-3 is gone. `results/runs/cc18_v2/` (6480 rows, previous freeze, TabPFN-3)
remains the latest complete result set until **`cc18_v3`** — the CC18 re-run on
the new freeze, with predictions kept — replaces it (**submitted to Arnes
2026-10-06**, jobs `20138834` + `20138836`, see What is next); then Medic3, then tuning
and the leakage test. The TabPFN-3.5 licence is accepted and the weights are
cached on both local machines (Kremen since 2026-10-06).
**Clean-up 2026-10-05:** all older runs (`check_refactor_oldenv`,
`subset_v2_local`, `subset_v2`) and the 2026-08 archive `results/arnes/` were
deleted; they live in git history, last present in commit `b1b9120` — e.g.
`git show b1b9120:results/runs/subset_v2/results.csv` or
`git checkout b1b9120 -- results/arnes`. Of the old-freeze runs only `cc18_v2`
is kept, until `cc18_v3` replaces it.

## Structure
- `config.yaml` — **the single source of every experiment parameter**:
  `random_state`, `n_splits`, `n_repeats`, `algorithms`, `dataset_sets`
  (name → JSON file of dataset specs), `cache_dir`, `results_dir`,
  `save_predictions`, `saturation_threshold`, `alpha`. `dataset_sets` also has
  `medic3_160` (Medic3 capped at 160 classes). Nothing is hardcoded
  elsewhere; every script reads it through `src/config.py`.
- `src/config.py` — `load_config()` (resolves paths to absolute),
  `dataset_specs(config, set_name)`, `spec_id(spec)`.
- `src/data.py` — `load_dataset(spec, n_splits, random_state, cache_dir,
  n_repeats)`. A spec is an OpenML ID (int) or a dict; `{"source": "csv",
  "path", "target", "drop_cols", ...}` loads a local file (also a `.zip`), which
  is how Medic3 enters the benchmark (`scripts/medic3.json`). Same output dict
  regardless of source. Builds the `(Repeated)StratifiedKFold` splits **once**;
  every algorithm reuses the same fold indices. The optional spec key
  `max_classes: N` (`_limit_classes`) keeps only the rows of the N most frequent
  classes, applied **before** label encoding so the codes stay 0..N-1; it exists
  because TabPFN-3 and TabPFN-3.5 both cap at 160 classes and Medic3 has 220
  (measured, see Datasets). The returned `class_limit` records how many
  rows that dropped.
- `src/runner.py` — **the only training loop.** `run_dataset()` runs every
  algorithm on every fold of one dataset with per-fit checkpointing
  (`<id>.csv.partial`, atomic finalisation to `<id>.csv`; with an explicit
  `--algorithms` subset the file is `<id>__<algos>.csv`, so parallel SLURM tasks
  on the **same** dataset do not overwrite each other's checkpoints — this is
  how Medic3 is split by algorithm), GPU warm-up, saved
  predictions, and a row per fit with `preprocessing`, `raw_error`, `error`,
  `device`, `git_commit`, `hostname`, `timestamp`. Also `write_manifest()`
  (git commit, host, GPU, config snapshot, pip freeze, and `model_checkpoints` —
  the weights file of each foundation model, because `pip freeze` only gives the
  package version) and `merge_run()`. The manifest is written through a temp
  file unique per host+PID (fixed 2026-10-05): with the old shared `.tmp`, 8
  processes × 150 concurrent writes raised 73 exceptions, now 0 — this matters
  for Medic3, where 6 tasks start on the same run directory at once.
- `src/run_benchmark.py` — local CLI: `python -m src.run_benchmark
  [--dataset-set subset] [--run-id X] [--algorithms ...]`; loops datasets in
  process, then merges to `results.csv`.
- `src/run_one_dataset.py` — Arnes CLI: `python -m src.run_one_dataset
  --dataset-set cc18 --index N --run-id X`; one dataset per SLURM array task.
- `src/models/` — one file per algorithm, each exposing
  `run(X_train, y_train, X_test, y_test, categorical_cols, random_state) -> dict`
  with keys `model, roc_auc, train_time_s, inference_time_s, error,
  preprocessing, raw_error, device, proba, classes` and a module flag
  `USES_GPU`. `classes` is `model.classes_` (label per `proba` column).
  `src/models/__init__.py` exposes `REGISTRY` mapping config names to modules.
  The two foundation-model modules also expose `checkpoint_name()` (weights file,
  read without loading the model). `tabpfn_model.make_classifier(random_state)`
  is the **only** place a TabPFN classifier is built:
  `TabPFNClassifier.create_default_for_version(ModelVersion.V3_5, ...)`, i.e. the
  version is pinned explicitly even though it is tabpfn 9's default — tabpfn
  already switched its default once (V3 in 8.5.0 → V3_5 in 9.0.0) and also reads
  it from `TABPFN_MODEL_VERSION`. `scripts/prestage.py` uses the same function, so
  it caches exactly the weights the benchmark loads.
- `src/utils.py` — `compute_roc_auc(y_true, proba, classes)` (binary vs.
  macro-OvR), `scored_class_indices()`, `describe_device()`, `cpu_threads()`.
  `classes` is `model.classes_`, i.e. the label of each `proba` column — needed
  because a class absent from the **train** fold has no column at all.
  **Macro-OvR averages only over classes present in the test fold**
  (decided 2026-09-22): otherwise `roc_auc_score(..., multi_class="ovr")` raises
  `ValueError` whenever a class is absent from a fold. **This is a defensive
  fix, not a Medic3 blocker** — the data card (2026-09-23) shows Medic3's
  smallest class has 50 rows, so no class is ever missing from a fold (verified
  on all 15 folds of both variants). It matters for any future dataset with
  classes smaller than `n_splits`. Verified bit-identical to sklearn's own macro-OvR
  when all classes are present, so CC18 numbers are unaffected. Regression
  against `cc18_v2` on OpenML 11 (multiclass) and 37 (binary), 90 fits each:
  **delta 0.0 for five of six algorithms**, TabPFN 1.2e-3 / 1.5e-3 which is the
  known GPU-hardware difference (RTX 3060 here vs. H100 on Arnes), not the
  metric. Regenerating `cc18_v2/summary/` changed no file. The per-fit column
  `n_classes_scored` says how many classes the average covered.
- `src/summary.py` — `python -m src.summary <run_id|dir|csv>` (path is
  **required**). Prints and writes to `summary/` (CSV + LaTeX): per-dataset
  table, dataset×algorithm pivot, ranks, overall table, and the same without
  saturated datasets. Rules: rank **per dataset** on the fold mean; a
  (dataset, algorithm) with any failed fold is a failure and gets the **worst
  rank**; mean ROC-AUC is reported over datasets where **all** algorithms
  succeeded; "saturated" = best algorithm ≥ `saturation_threshold`.
- `src/stats.py` — `python -m src.stats <run> [--no-saturated]`. Several
  datasets (Demšar 2006): Friedman test — chi2_F by **Demšar's formula from the
  mean ranks** (no tie correction, so every number can be recomputed by hand;
  scipy's `friedmanchisquare` adds one: 230.37 instead of 229.82 on `cc18_v2`) —
  plus the **Iman–Davenport F_F**, which Demšar recommends and which decides
  `reject_H0`; Nemenyi critical difference + CD diagram
  (`summary/cd_diagram.png/pdf`); pairwise Wilcoxon signed-rank with Holm
  correction (the combination recommended by Benavoli, Corani & Mangili 2016).
  **One dataset** (Medic3, detected automatically): Friedman/Nemenyi/Wilcoxon do
  not apply (N = 1), so each pair gets the **corrected repeated k-fold cv t-test**
  (Bouckaert & Frank 2004, variance correction of Nadeau & Bengio 2003:
  `t = mean(d) / sqrt((1/n + n_test/n_train) var(d))`, n − 1 df) on the per-fold
  ROC-AUC differences, Holm over all pairs → `summary/corrected_ttest_holm.csv/.tex`;
  an algorithm that failed on any fold is excluded. `holm()` is shared.
  Verified 2026-10-05: Demšar's worked example gives chi2_F 9.28, F_F 3.69 on
  (3, 39) df and his Holm example rejects the same two hypotheses; the Holm
  refactor reproduces the stored `cc18_v2` p_holm to 1e-16; the t-test equals
  `scipy.stats.ttest_rel` when n_test/n_train = 0. `cc18_v2` with the new code:
  F_F = 125.35 on (5, 355) df (39 non-saturated: 63.68 on (5, 190)).
  q(6, 0.05) from scipy = 2.850 (Demšar).
- `scripts/` — `gen_cc18_ids.py` (pins the 72 CC18 IDs), `profile_datasets.py
  --dataset-set X [--from-cache]`, `prestage.py --dataset-set X` (Arnes login
  node), `run_cc18.sh` / `run_subset.sh` / `run_medic3.sh` (SLURM, take
  `RUN_ID`),
  `merge_results.py <run>`, `compare_results.py <run A> <run B>`,
  `gen_version_table.py <run>` (LaTeX table of versions from the manifest; adds
  the weights file from `model_checkpoints` to the TabPFN/TabICL rows),
  `profile_medic3.py` (data card, stdlib only), `medic3.json` /
  `medic3_160.json` (Medic3 specs, raw and capped at 160 classes),
  `subset_ids.json`, `cc18_ids.json`.
- `results/` — see `results/README.md`. `results/runs/<run_id>/` per run
  (`manifest.json`, `per_dataset/`, `predictions/` [gitignored],
  `results.csv`, `summary/`). Kept: **`cc18_v2`** (until `cc18_v3` replaces
  it) and the two validation runs of the new freeze, `subset_v3_local`
  (Kremen) and `subset_v3` (Arnes); everything older is in git history (see the state
  note at the top). Rule: keep only the runs the thesis currently uses, delete
  superseded ones in a commit of their own.
- `data/openml_cache/` — OpenML's local dataset cache (gitignored); the
  library appends `org/openml/www`.
- `razlaga_repozitorija/` — explanatory material, **not** part of the
  experiment: `zaporedje.puml` (4 PlantUML sequence diagrams),
  `preveri_diagram.py` (consistency checker), `hooks/pre-push`.

## Sequence diagram — keep it in sync (mandatory)
`razlaga_repozitorija/zaporedje.puml` holds four PlantUML sequence diagrams
(`01_priprava`, `02_lokalni_pilot`, `03_arnes`, `04_analiza`) showing which file
calls which. **Rule: any change to the set of source files updates the diagram
in the same commit** (adding, renaming, deleting a file in `src/` or
`scripts/`, or changing what the diagram states about it). Every file in `src/`
and `scripts/` must appear with its **full path**.

Enforcement at `git push`: `razlaga_repozitorija/preveri_diagram.py` checks
both directions (stdlib only, any `python3`); `hooks/pre-push` blocks the push
if stale. Escape hatch: `git push --no-verify`. Per-machine setup (hooks are
not carried by git): `git config core.hooksPath razlaga_repozitorija/hooks`
(done on the laptop `DESKTOP-0EPDCR8` and on the desktop `Kremen`).

PlantUML creole eats some characters, so the diagram escapes them with `~`:
`~__init~__.py`, `~--mem`, `~#SBATCH`. The checker strips `~` before comparing.
Render with `java -jar plantuml.jar -tpng razlaga_repozitorija/zaporedje.puml`
or the VS Code PlantUML extension (`Alt+D`).

## Preprocessing policy (per algorithm — a deliberate experimental variable)
The protocol is "the *minimal* preprocessing an algorithm needs to accept the
input", not the best preprocessing. Each row of `results.csv` records what was
applied (`preprocessing`) and, for the foundation models, whether the raw input
failed first (`raw_error`).
- **RandomForest**: no native NaN/categorical support → median imputation
  (numeric) + most-frequent imputation & ordinal encoding (categorical), fit on
  the train fold only.
- **XGBoost**: native NaN handling; categorical columns ordinal-encoded to
  numeric codes with NaN preserved. Ordinal codes impose an artificial order
  on nominal categories — that is part of what is measured.
- **LightGBM**: native NaN handling; categorical columns cast to pandas
  `category` dtype; non-categorical columns explicitly cast to float.
- **CatBoost**: native NaN (numeric) + native categorical via `cat_features`;
  categorical columns mapped element-wise to strings with NaN → the string
  `'nan'` (its own category, not an imputation). **pandas 3.0 note:**
  `astype(str)` no longer turns NaN into `'nan'`, it keeps it missing and
  CatBoost rejects it; hence the explicit `map` in `catboost_model.py`. This
  bit the first `tabular3` pilot (all 5 `sick` folds failed) and is fixed.
- **TabPFN / TabICL**: raw input first; only on an exception apply a logged
  minimal fix and retry. With TabPFN-3 (tabpfn 8.5.0) raw input worked on every
  CC18 dataset; for TabPFN-3.5 `cc18_v3` will show it (the `raw_error` column).
  The fallbacks stay as safety nets.

## Timing
`train_time_s` and `inference_time_s` are wall-clock around `fit` and
`predict_proba` (including the algorithm's own preprocessing). Foundation
models do no training, so their cost sits in inference. GPU algorithms get one
warm-up fit per process (`warmup_s` column) so fold 0 does not carry weight
loading and CUDA initialisation. The `device` column says where each fit ran
(`cpu x8` vs. `cuda: NVIDIA H100 ...`); times across the two families compare
hardware as much as algorithms and the thesis must say so. Summaries report the
**median** over folds.

## Environment
- WSL2 (Ubuntu) on Windows 11; repo at
  `/home/tilen/fri/diplomska/Diplomsko-delo_Koda` on both machines (desktop
  `Kremen`, laptop `DESKTOP-0EPDCR8`), synced through GitHub
  (`origin = https://github.com/TilenZrnec/Diplomsko-delo_Koda.git`, `main`).
  Commit + push before switching machines, pull on arrival.
- Not carried by git: the conda envs, the TabPFN credential
  (`~/.cache/tabpfn/`), the OpenML cache, `.claude/settings.local.json`.
- **Conda envs** (renamed 2026-10-05 so the name says which TabPFN they carry):
  `tabular3.5` (Python 3.12.13, freeze 2026-10-05, TabPFN-3.5) is the current
  one; `tabular3` (Python 3.12.13, freeze 2026-09-09, TabPFN-3 — formerly
  `tabular2`) is the reference environment of `cc18_v2`; `tabularOriginal`
  (Python 3.10, the original 2026-07/08 stack — formerly `tabular`) is kept only
  to reproduce the deleted early runs. Both 3.12 envs were rebuilt from the exact
  `pip freeze` of their predecessors (identical package lists, verified),
  because `conda create --clone` mixed two pip versions. On `Kremen` (2026-10-06)
  both `tabular3` and `tabularOriginal` were rebuilt the same way, freezes
  identical, and the originals removed. **Trap in `tabularOriginal`:** it has
  both `nvidia-nccl-cu12` and `nvidia-nccl-cu13`, which write the **same file**
  `nvidia/nccl/lib/libnccl.so.2`; a freeze rebuild installed cu12 last and torch
  failed with `undefined symbol: ncclCommResume`. Fix: `pip install
  --force-reinstall --no-deps nvidia-nccl-cu13==2.29.7` (then the file is
  byte-identical to the original's). An identical freeze is therefore not
  enough — also import torch. Run project scripts with
  `conda run -n tabular3.5 python -m src.<module>`.
- **Laptop = Lenovo IdeaPad 3 (`DESKTOP-0EPDCR8`): no usable GPU in WSL and
  only 7.6 GB RAM for WSL.** `torch.cuda.is_available()` is False (no
  `/usr/lib/wsl/lib/libcuda*`), so foundation models run on CPU there. On
  2026-10-05 a TabPFN-3.5 probe on CPU (220 classes / 3072 features) crashed
  WSL twice, most likely out of memory. **Do not run TabPFN-3.5 fits on the
  laptop**: foundation-model runs, probes and timings belong on the desktop
  `Kremen` (RTX 3060) or Arnes. The laptop is fine for code, docs, analysis and
  tree-model checks (the CPU smoke run of four ensembles + TabICL on the subset
  ran without problems).
- **Version policy (write this in the thesis):** all libraries pinned to the
  newest stable release co-installable as one environment on the freeze date
  (**2026-10-05**; the previous freeze 2026-09-09 produced `cc18_v2`), identical
  locally and on Arnes, so no algorithm family is disadvantaged by library age.
  `requirements.txt` lists the top-level pins and the diff to the previous
  freeze; each run's `manifest.json` carries the full `pip freeze` and the
  foundation-model weights files; `scripts/gen_version_table.py` produces the
  LaTeX table from it (never type versions by hand into the thesis). `skrub` is
  pinned although our code does not import it: tabpfn 9 uses it in its own
  preprocessing (text/datetime encoders) and requires `skrub<0.11`, so 0.10.1 is
  the newest co-installable release (0.11.0 came out 2026-10-01).
- **tabpfn 9 facts (verified in the 9.0.0/9.1.0 source):** default model
  `ModelVersion.V3_5` (8.5.0: V3); `softmax_temperature` default `"auto"`
  (checkpoint-declared; 8.5.0: fixed 0.9); TabPFN-3.5 accepts up to 20 000
  features (TabPFN-3: 2000) — the class limit is read from the checkpoint and
  is not documented; measured 2026-10-06 it is **160, as in TabPFN-3** (see
  Datasets); weights `tabpfn-v3.5-20260909.safetensors`, licence
  `tabpfn-3-5-license-v1.0` (non-commercial, allows "testing, evaluation, and
  internal benchmarking"). TabPFN-3.5-Plus/-Thinking are API-only and are not
  used (offline nodes; Medic3 must never go to an API). Model card and the
  TabPFN-3 report both state synthetic-only pretraining.
- **Measured effects of library changes.** 2026-10-05 freeze vs 2026-09-09
  (local CPU smoke run, 75 shared fits): RandomForest, XGBoost, LightGBM,
  CatBoost bit-identical, TabICL ≤ 1.5e-5 — so in `cc18_v3` only TabPFN changes
  (different model); confirm on Arnes with `subset_v3` vs `cc18_v2`. **Local GPU
  pilot `subset_v3_local`** (2026-10-06, Kremen RTX 3060, 270 fits, 0 errors,
  0 `raw_error`) vs `cc18_v2` (Arnes H100), 270 shared fits: RandomForest,
  XGBoost, LightGBM, CatBoost **Δ = 0.0**; TabICL mean 6e-6, max 1.5e-4 (only on
  sick; credit-g and diabetes Δ 0) — GPU hardware, not the library; TabPFN-3.5
  vs TabPFN-3 fold-mean ROC-AUC **higher on all three**: credit-g 0.7937 →
  0.8022 (+0.0086, 14/15 folds up), diabetes 0.8398 → 0.8428 (+0.0030),
  sick 0.9981 → 0.9989 (+0.0009); max |Δ| 0.0214. `src.stats` and
  `gen_version_table.py` run on it (weights `tabpfn-v3.5-20260909`). **Arnes
  `subset_v3`** (2026-10-06, `gwn08`, H100 80GB HBM3, commit `c503f48`, 270 fits,
  0 errors) **confirms it**: vs `cc18_v2` trees Δ 0.0, TabICL mean 7e-6 / max
  6.1e-5, TabPFN the same per-dataset gains to 4 decimals; vs `subset_v3_local`
  trees Δ 0.0, TabICL max 1.2e-4, TabPFN max 2.4e-4 (GPU noise, two orders below
  the 0.023 between-fold sd). Packages identical to Kremen except pip/setuptools/
  wheel/packaging. Details in `results/runs/subset_v3/PROVENANCE.md`. History (git
  `b1b9120`): the 2026-09-09 upgrade left RF/LightGBM/CatBoost bit-identical and
  moved **XGBoost 2.1.1 → 3.4.1 by up to 0.0175** (its defaults changed); the
  2026-09-09 refactor reproduced the July baseline with Δ = 0.0 on 90/90 rows.
- TabPFN needs a one-time licence acceptance via a PriorLabs account, **per
  model version**: the account accepted TabPFN-3's licence, and on 2026-10-05
  the TabPFN-3.5 download still failed with `TabPFNLicenseError` ("one-time
  license acceptance … no interactive terminal"). Fix (the account owner, once):
  log in at https://ux.priorlabs.ai → Licenses tab → accept the TabPFN-3.5
  licence. tabpfn reads the token from `TABPFN_TOKEN` or
  `~/.cache/tabpfn/auth_token` (Arnes sources `~/.tabpfn_token`), so no token change is needed after accepting. Fresh
  machine without a cached token: interactive first `fit()` opens a browser
  login (needs a real TTY — `conda activate`, not `conda run`), or set
  `TABPFN_TOKEN` from https://ux.priorlabs.ai/account. **Accepted 2026-10-05**:
  afterwards the TabPFN-3.5 weights downloaded on the laptop with the existing
  cached token (`~/.cache/tabpfn/tabpfn-v3.5-20260909.safetensors`, 876 MB).
  **Kremen had no `auth_token` at all** (only the July v2/v3 weights), so on
  2026-10-06 the first fit raised `TabPFNLicenseError`; fixed by one interactive
  fit in a real terminal (`conda activate tabular3.5`, then `make_classifier(0)
  .fit(...)`): it prints a `ux.priorlabs.ai/login?callback=localhost…` link
  (in WSL open it in the Windows browser), caches the key, downloads the
  weights. Both machines now have the token and the 3.5 weights.
- `src/data.py` sets the OpenML cache with
  `openml.config.set_root_cache_directory()`. **Do not use
  `openml.config.cache_directory = ...`** — removed after `openml` 0.10 and,
  because `openml.config` is a plain module, the assignment silently creates an
  unread attribute. `openml` 0.15 still accepts
  `list_datasets(..., output_format="dataframe")` without a warning.

## Datasets
- `subset` = OpenML 31 credit-g (1000×20, 13 categorical), 37 diabetes
  (768×8, numeric), 38 sick (3772×29, 22 categorical, missing values; the
  originally specified ID 3021 does not exist on OpenML).
- `cc18` = the 72 IDs in `scripts/cc18_ids.json`, pinned from
  `openml.study.get_suite(99)`.
- `medic3` = `scripts/medic3.json` → `data/medic3/Medic3.csv` (148 MB,
  gitignored via `data/`, confidential; the path was corrected from
  `../Medic3.csv.zip` on 2026-09-23 — keep the file at this same relative path
  on every machine and on Arnes). Data card
  (`python3 scripts/profile_medic3.py`, 2026-09-23):
  - 122 093 × 275, 220 classes, **smallest class 50 rows** (median 182, max
    7462), so 5-fold stratified CV leaves no class out of any fold;
  - majority class 6.1 % (random guess over 220 classes = 0.45 %);
  - **82.2 % missing**, median 50 of 275 attributes observed per row;
  - **no categorical columns** — all 275 attributes are numeric, so CatBoost's
    native categorical handling gives it no advantage here;
  - `Field` alone predicts the class with 25 % accuracy → correctly dropped;
  - **missingness itself is a strong predictor** (e.g. A177 observed in 99.5 %
    of class C152's rows vs. 0.0 % of C202's). RandomForest is the only
    algorithm that imputes and therefore destroys this signal — that is exactly
    the experimental variable the thesis measures, so expect it to fall behind
    on Medic3 for preprocessing reasons, not model reasons. Say this in the
    thesis; do not "fix" it. TabPFN-3's hard
  cap was 160 classes, so it would fail-soft on Medic3 as agreed with the
  supervisor; TabICL handles 220 classes. Both verified empirically on
  2026-09-22 with synthetic data in `tabular3`: at 220 classes TabPFN-3 raises
  `Number of classes 220 exceeds the maximum number of classes 160 officially
  supported` while TabICL succeeds; at 160 classes both succeed. Sample count is
  not a problem — TabPFN-3 handled 73 600 training rows on Devnagari-Script.
  **TabPFN-3.5 keeps the 160-class cap** — measured 2026-10-06 on Kremen
  (RTX 3060, `tabular3.5`, classifiers from `make_classifier(0)`, synthetic data,
  10 rows per class, 5 features): 160 classes OK (1.1 s, 1.5 GiB GPU); 161 and
  220 raise `TabPFNValidationError: Number of classes `161` exceeds the maximum
  number of classes `160` officially supported by TabPFN.` The fitted
  `inference_config_` says `MAX_NUMBER_OF_CLASSES = 160`,
  `MAX_NUMBER_OF_FEATURES = 20000`, `MAX_NUMBER_OF_SAMPLES = 1 000 000`;
  defaults resolve to `softmax_temperature_ = 1.0`, `n_estimators_ = 8`. Binary
  200 × 3072 features (CIFAR_10's width) OK in 4.0 s, 6.8 GiB peak GPU. So on raw
  `medic3` TabPFN fails soft (15 rows with the reason in `error`, a result, not
  a bug) and **`medic3_160` is the only variant with all six algorithms**.
- `medic3_160` = `scripts/medic3_160.json` → the same file with
  `"max_classes": 160`, i.e. the rows of the 160 most frequent classes (ties
  broken by label, so the selection is deterministic). Measured 2026-09-23:
  keeps 117 825 of 122 093 rows (96.50 %), drops 4268 (3.50 %); smallest kept
  class has 102 rows. Decided 2026-09-22: the
  rest of the rows are **dropped**, not merged into an "other" class — that
  would be 161 classes and TabPFN would fail again. Report the dropped-row
  percentage in the thesis (`run_dataset` logs it from `class_limit`).

## Arnes HPC
- SLURM cluster, GPU partition `gpu`, H100 nodes; partition time limit
  **4-00:00:00** (`sinfo -p gpu -o "%P %l"`, 2026-10-06). `--constraint=h100`
  matches **two H100 variants**: `cc18_v2` ran entirely on H100 PCIe
  (`gwn01`, `gwn03`–`gwn06`), `subset_v3` on H100 80GB HBM3 (`gwn08`), so
  `cc18_v3` may mix them. One dataset = one task = one node, so all six
  algorithms of a dataset share hardware; the `device` column records the
  variant per fit. Say so in the thesis next to GPU times. Node memory
  (`sinfo -p gpu -N -o "%N %m %f"`, 2026-10-07): `gwn01`–`gwn06` H100 PCIe,
  **256 GB**; `gwn08`–`gwn10` H100 SXM (feature `sxm`), **512 GB**; `wn2xx`
  V100S, 128 GB (excluded by `--constraint=h100`). Env: micromamba prefix
  `~/envs/tabular3.5` (Python 3.12, from `requirements.txt`; `~/bin/micromamba`,
  no shell hooks in batch scripts). The batch scripts default to it; override
  with `TABULAR_ENV=/path sbatch ...`. The cluster's older prefixes were not
  renamed: `~/envs/tabular2` (= local `tabular3`, ran `cc18_v2`) and
  `~/envs/tabular` are no longer used and can be deleted with `rm -rf`.
- TabPFN token in `~/.tabpfn_token` (sourced by the batch script); compute
  nodes run offline (`HF_HUB_OFFLINE=1`), so on the login node first:
  `python scripts/prestage.py --dataset-set cc18` (caches datasets + the
  TabPFN-3.5 and TabICL weights the benchmark uses, prints the cache size for
  the 100 GB home-quota check). Prestage fails with `TabPFNLicenseError` until
  the account has accepted the TabPFN-3.5 licence.
- Submit from the repo root as **two arrays with the same RUN_ID** (decided
  2026-09-11): the 68 ordinary datasets at the script defaults (`--mem=64G`,
  `--time=12:00:00`, `%4`), and the four big ones (27 mnist_784, 60
  Devnagari-Script, 61 CIFAR_10, 70 Fashion-MNIST) with `--mem=240G
  --time=2-00:00:00` (was 36 h until `cc18_v2`, where Devnagari-Script used
  34 h 54 min of it). Both need `ALLOW_SPARSE_ARRAY=1` because neither array
  spans all 72 indices. Exact commands in `scripts/run_cc18.sh` header and
  `razlaga_repozitorija/razlaga.md`. A task that still hits `--time` is
  re-submitted with the same command; it resumes from its partial.
- Per-fit checkpointing: a killed task loses no work; re-submitting with the
  same `RUN_ID` resumes at the first unfinished (algorithm, fold). A new
  `RUN_ID` is a clean slate, so the old "rm -rf results/per_dataset/*" hazard
  is gone. Verified 2026-07-22 (kill/resume, max Δ 0.0).
- **Medic3** is submitted differently: one **algorithm** per array task, not
  one dataset, because it is a single dataset and all six algorithms in series
  would exceed `--time` (estimate from CC18: ~60 h, of which CatBoost ~48 h — at
  220 classes it dominates, just as it did on Devnagari-Script's 46 classes).
  CatBoost therefore gets its own job with a longer `--time` and more memory.
  Its header still says `--time=1-12:00:00` and expects one or two
  re-submissions; since the partition allows 4 days, decide before Medic3
  whether to request more instead.
  Exact commands in the header of `scripts/run_medic3.sh`. `prestage.py` is not
  needed (the dataset is not from OpenML), but the TabPFN/TabICL weights must
  already be cached because compute nodes are offline.
- Merge and analyse: `python scripts/merge_results.py cc18_v3`, then
  `python -m src.summary cc18_v3`, `python -m src.stats cc18_v3` (and
  `--no-saturated`), `python scripts/gen_version_table.py cc18_v3`. Expect
  72 × 6 × `n_splits` × `n_repeats` rows; fewer rows means unfinished (the merge
  lists the partials), a failed fit still has its row with the reason in
  `error`.
- **Keep the predictions.** `predictions/` is gitignored and stays on Arnes;
  copy it back after the run so any further metric can be computed without
  re-running, e.g. from the laptop:
  `rsync -av <user>@<arnes-login>:<repo>/results/runs/cc18_v3/predictions/
  results/runs/cc18_v3/predictions/` (`cc18_v2`'s copy is ~0.5 GB, 6450 files).
  **Archive = the user's USB drive** (the user's decision, 2026-10-08) — not
  GitHub: this repo is public and Medic3 predictions are confidential. `cc18_v2`
  (6450 files) and `cc18_v3` (6480) were copied to Kremen and verified on
  2026-10-08: every successful fit has its file, and ROC-AUC recomputed from the
  files matches `results.csv` to ≤ 3.2e-4 (they store float32). `cc18_v2` files
  predate the `proba_classes` field (columns = classes in order). When
  predictions are needed and not on the machine, ask the user to plug in the USB
  drive; never re-run just to regain them. After each new run, remind the user
  to copy its `predictions/` there.
- Record the job receipt after every run:
  `sacct -j <jobid> --format=JobID,JobName%20,Elapsed,MaxRSS,State,NodeList`
  into a `PROVENANCE.md` inside the run directory.
- Sizing knowledge from the 2026-08 sweep: `--mem=64G` sufficed for 68/72;
  indices 27/60/61/70 (mnist_784, Devnagari-Script, CIFAR_10, Fashion-MNIST)
  needed 120–240G; `--time=12:00:00` sufficed everywhere; CatBoost (1000
  default iterations) took 13 of the 16 total CPU hours. CIFAR_10 × {TabPFN,
  TabICL} failed (3072 features > TabPFN-3's 2000 cap; TabICL asked ~378 GB) —
  those rows carry the reason in `error`, no subsampling was or will be added.
  In `cc18_v3` TabPFN-3.5 (20 000-feature limit) ran on CIFAR_10 for the
  first time (15/15 OK, ~77 s per fit on H100 SXM). TabICL (tabicl 2.2.0)
  still needs ~378 GB there and **killed the task on a 512 GB node** instead
  of failing soft — see What is next, step 4. For any task that may contain a
  huge TabICL fit, either request (nearly) the whole node or keep it on a
  256 GB node.
- The cluster remote uses SSH (`git@github.com:...`), not HTTPS.

## What is next (agreed 2026-10-05)
Done before: Arnes validated against the laptop (2026-09-11, trees Δ 0, TabPFN
max 3.7e-4); `cc18_v2` finished 2026-09-11/12 (`n_repeats = 3` decided
2026-09-11: 15 fits per dataset × algorithm, model seed 42 + repeat).
Order from here (the user's decision): CC18 re-run → Medic3 → tuning → leakage.
1. **Done 2026-10-06 on `Kremen`** (env `tabular3.5` built, CUDA OK on the
   RTX 3060; Kremen env names fixed to match the laptop; probe and pilot results
   are under Datasets and Environment → Measured effects). Original plan: `git pull` in both repos; build `tabular3.5` from
   `requirements.txt` (`conda create -n tabular3.5 python=3.12.13`, then
   `pip install -r requirements.txt`; check `torch.cuda.is_available()`). Kremen
   still has the old env names (`tabular`, `tabular2`) — rename them as on the
   laptop if wanted (rebuild from `pip list --format=freeze`, never
   `conda create --clone`, see Environment). Then:
   - TabPFN-3.5 probe on synthetic data — 160, 161 and 220 classes (10 rows
     each, 5 features) and 200 × 3072 features — and record the class/feature
     limits under Datasets (decides whether TabPFN runs on raw Medic3);
   - local GPU pilot `python -m src.run_benchmark --dataset-set subset --run-id
     subset_v3_local` and `python scripts/compare_results.py cc18_v2
     subset_v3_local` (expect trees Δ 0, TabICL ~1e-5, TabPFN different).
2. Commit the probe findings and `subset_v3_local`, push.
3. **Done 2026-10-06** (`subset_v3`, see Environment → Measured effects and its
   `PROVENANCE.md`; all checks passed). Original plan: Arnes: build `~/envs/tabular3.5` from `requirements.txt` (exact commands in
   `razlaga_repozitorija/razlaga.md`, part 3), prestage `cc18`, run
   `RUN_ID=subset_v3 sbatch scripts/run_subset.sh`, then
   `compare_results.py cc18_v2 subset_v3` — `cc18_v2` contains the same three
   datasets with the same 15 folds and seeds, so it is the old-freeze baseline
   (270 shared fits). Expect trees Δ 0 and TabICL ~1e-5 (as in the local smoke
   run); TabPFN will differ because it is a different model. Write the measured
   effect into this file and the thesis.
4. `cc18_v3`: **submitted 2026-10-06** from commit `69cc51e` — job `20138834`
   (68 ordinary datasets, `%4`) and `20138836` (27/60/61/70, `--mem=240G
   --time=2-00:00:00`); prestage found all 72 datasets, home 30G of 100G.
   **CIFAR_10 incident (task 61):** on `gwn08` (512 GB node) it was
   OOM-killed after 11 h 03 min (`--mem=240G`, MaxRSS 256 GiB, 2026-10-07
   04:52 UTC), 2.5 min after the last TabPFN fit — i.e. at **TabICL** fold 0.
   The partial kept 75 fits: the four trees and **all 15 TabPFN-3.5 fits OK**
   (first time TabPFN runs on CIFAR_10; ROC-AUC 0.913–0.919, mean ≈ 0.916, vs
   CatBoost 0.910 in `cc18_v2`; ~77 s inference per fit). In `cc18_v2` TabICL's
   ~378 GB request was refused at once on a 256 GB node → soft error row; on a
   512 GB node it was evidently not refused, so the cgroup killed the whole
   job and no row was written. Resubmitted (option A, the user's choice) as
   job **`20176087`**: `PYTHONUNBUFFERED=1 ALLOW_SPARSE_ARRAY=1 RUN_ID=cc18_v3
   sbatch --array=61 --constraint=sxm --mem=480G --time=2-00:00:00
   scripts/run_cc18.sh` — TabICL gets a near-full 512 GB node. If it is still
   queued after ~a day, fallback B: `scancel` it and resubmit with
   `--exclude=gwn08,gwn09,gwn10 --mem=240G` (256 GB node, reproduces
   `cc18_v2`'s soft failure). Record all of this in `cc18_v3/PROVENANCE.md`.
   Logs looked empty because Python block-buffers stdout to a file (the kill
   discarded the buffer); `PYTHONUNBUFFERED=1` on the resubmit fixes it —
   consider adding it to the batch scripts after the run. Devnagari-Script
   (task 60, `gwn04`) is exactly on `cc18_v2`'s pace (CatBoost fold 8/15 at
   20.3 h), expected to finish ~35 h after start.
   **Do not `git pull` or commit on Arnes until it finishes** (every row
   records the git commit); at the end commit there, then `git pull --rebase`
   and push. Two arrays with the same `RUN_ID` (commands in the
   `scripts/run_cc18.sh` header). Expect ~70 h compute (cc18_v2: 69.4 h, CatBoost
   55.1 h) plus TabPFN-3.5 on CIFAR_10. Then merge, summary, stats (both
   variants), version table, `sacct` → `PROVENANCE.md`, and **rsync the
   predictions back** (see Arnes HPC). The thesis results chapter is then
   regenerated from `cc18_v3`: in the thesis repo's
   `rezultati/generiraj_iz_cc18_v2.py` change the hard-coded
   `R = 'results/runs/cc18_v2'` and the display name of `tabpfn` (TabPFN →
   TabPFN-3.5). Once `cc18_v3` is verified, delete `results/runs/cc18_v2` in its
   own commit (copy its `predictions/` somewhere first if you want to keep them —
   they are not in git).
5. **Medic3 (prepared 2026-09-22/23).** Code is ready: the ROC-AUC fix for
   absent classes, `max_classes`, per-algorithm result files,
   `scripts/run_medic3.sh`, the race-free manifest and the single-dataset
   statistics. Still to do: copy `data/medic3/Medic3.csv` to the same relative
   path on Arnes and record its `sha256` in the run's `PROVENANCE.md`; a local
   one-fold timing probe for the CatBoost sizing. Two runs: `medic3_raw` (220
   classes) and `medic3_160`; `python -m src.stats medic3_raw` then runs the
   corrected t-tests automatically.
6. Last: tuning one booster (nested CV, see the 2026-09-29 discussion: XGBoost
   or LightGBM, not CatBoost) and the CC18 leakage test (design to be agreed
   with the supervisor first); CC18 with injected NULLs is still on the
   supervisor's list.

## FRI methodology requirements (official diploma guidelines)
Writing/citation rules live in the thesis repo's `CLAUDE.md`
(`../Diplomsko-delo_Tabelaricni-temeljni-modeli/CLAUDE.md`).
- **Baseline — deliberately not used. Decided 2026-08-16, do not re-raise.**
  The four tree ensembles are the reference point; a `DummyClassifier` sits at
  ROC-AUC 0.5 by construction. No such entry in `REGISTRY`, none to be added.
- **Reproducibility.** Fixed seeds (`random_state` from `config.yaml` for the
  folds; model seed is `random_state + repeat`, i.e. 42/43/44 with three
  repeats, so repeats also cover model randomness), every parameter in `config.yaml`,
  versions pinned in `requirements.txt`, full environment in each run's
  `manifest.json`.
- **Systematic experiment logging**, never hand-named result folders: each run
  directory records configuration, metrics, timestamps, git commit and host
  automatically. `PROVENANCE.md` is only for what a script cannot know (SLURM
  receipts, what went wrong and was re-submitted).
- **Link results to code:** `git_commit` is in `manifest.json` and in every
  result row.
- **Version control and data versioning.** Code pushed to the remote; the
  dataset set is pinned in `scripts/cc18_ids.json`; the Medic3 file is
  identified by path and, once used, should get a checksum in its run's
  `PROVENANCE.md`.
- **Statistics — sources (all verified 2026-09-29; DOIs via Crossref, JMLR via
  jmlr.org).** Demšar 2006, JMLR 7(1):1–30 (Friedman §3.2.2 p. 11, Iman–Davenport
  F_F p. 11, Nemenyi + CD pp. 11–12, CD diagram §3.2.4 p. 15, Wilcoxon §3.1.3
  pp. 7–8, Holm pp. 12–13 incl. "can be used generally … family-wise error"
  p. 13); Benavoli, Corani & Mangili 2016, JMLR 17(5):1–10 (pairwise Wilcoxon +
  Holm instead of mean-rank post-hoc); Iman & Davenport 1980,
  doi:10.1080/03610928008827904; Holm 1979, Scand. J. Stat. 6(2):65–70, JSTOR
  4615733; Wilcoxon 1945, doi:10.2307/3001968; Friedman 1937,
  doi:10.1080/01621459.1937.10503522; Nemenyi 1963, PhD thesis, Princeton.
  Single dataset: Bouckaert & Frank 2004, doi:10.1007/978-3-540-24775-3_3
  (formula checked in the paper, §3.3; for best replicability they recommend
  10 × 10-fold CV — our 3 × 5 is valid but less replicable, say so for Medic3);
  Nadeau & Bengio 2003, Machine Learning 52:239–281, doi:10.1023/A:1024068626366.

## Conventions
- Each model module fails soft: exceptions go to `result["error"]`, never
  raised, so one failing (dataset, algorithm, fold) doesn't crash a run.
- All algorithms use **default hyperparameters** — intentional per the thesis
  protocol, not an oversight to "fix". The only non-defaults are compute
  settings: `RandomForestClassifier(n_jobs=-1)` (bit-identical predictions,
  verified), `LGBMClassifier(verbosity=-1)`,
  `CatBoostClassifier(verbose=False, allow_writing_files=False)`.
- Never merge into or overwrite an existing run's `results.csv` from another
  run; every run has its own directory.
