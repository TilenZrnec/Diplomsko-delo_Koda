# Diplomsko-delo_Koda — Diploma Project (FRI)
#  Style
Explain things to me as if i am a compleete beginner.

## Purpose
Benchmarking 6 classical/foundation-model tabular ML algorithms on OpenML
datasets (and the private Medic3 dataset) as part of a diploma thesis at FRI
(Faculty of Computer and Information Science, Ljubljana), with default
hyperparameters throughout. Comments and docstrings in the codebase are
written in Slovenian.

**State on 2026-09-09:** the code was restructured (one runner, config-driven
parameters, per-run result directories, saved predictions, statistics) and the
library stack was upgraded to the newest stable versions (Python 3.12). The
2026-08 CC18 sweep in `results/arnes/cc18/` was produced by the *old* code and
*old* versions; it is an archive, and the thesis result set is the **re-run
still to be done on Arnes** with the new stack (see "What is next").

## Structure
- `config.yaml` — **the single source of every experiment parameter**:
  `random_state`, `n_splits`, `n_repeats`, `algorithms`, `dataset_sets`
  (name → JSON file of dataset specs), `cache_dir`, `results_dir`,
  `save_predictions`, `saturation_threshold`, `alpha`. Nothing is hardcoded
  elsewhere; every script reads it through `src/config.py`.
- `src/config.py` — `load_config()` (resolves paths to absolute),
  `dataset_specs(config, set_name)`, `spec_id(spec)`.
- `src/data.py` — `load_dataset(spec, n_splits, random_state, cache_dir,
  n_repeats)`. A spec is an OpenML ID (int) or a dict; `{"source": "csv",
  "path", "target", "drop_cols", ...}` loads a local file (also a `.zip`), which
  is how Medic3 enters the benchmark (`scripts/medic3.json`). Same output dict
  regardless of source. Builds the `(Repeated)StratifiedKFold` splits **once**;
  every algorithm reuses the same fold indices.
- `src/runner.py` — **the only training loop.** `run_dataset()` runs every
  algorithm on every fold of one dataset with per-fit checkpointing
  (`<id>.csv.partial`, atomic finalisation to `<id>.csv`), GPU warm-up, saved
  predictions, and a row per fit with `preprocessing`, `raw_error`, `error`,
  `device`, `git_commit`, `hostname`, `timestamp`. Also `write_manifest()`
  (git commit, host, GPU, config snapshot, pip freeze) and `merge_run()`.
- `src/run_benchmark.py` — local CLI: `python -m src.run_benchmark
  [--dataset-set subset] [--run-id X] [--algorithms ...]`; loops datasets in
  process, then merges to `results.csv`.
- `src/run_one_dataset.py` — Arnes CLI: `python -m src.run_one_dataset
  --dataset-set cc18 --index N --run-id X`; one dataset per SLURM array task.
- `src/models/` — one file per algorithm, each exposing
  `run(X_train, y_train, X_test, y_test, categorical_cols, random_state) -> dict`
  with keys `model, roc_auc, train_time_s, inference_time_s, error,
  preprocessing, raw_error, device, proba` and a module flag `USES_GPU`.
  `src/models/__init__.py` exposes `REGISTRY` mapping config names to modules.
- `src/utils.py` — `compute_roc_auc()` (binary vs. macro-OvR),
  `describe_device()`, `cpu_threads()`.
- `src/summary.py` — `python -m src.summary <run_id|dir|csv>` (path is
  **required**). Prints and writes to `summary/` (CSV + LaTeX): per-dataset
  table, dataset×algorithm pivot, ranks, overall table, and the same without
  saturated datasets. Rules: rank **per dataset** on the fold mean; a
  (dataset, algorithm) with any failed fold is a failure and gets the **worst
  rank**; mean ROC-AUC is reported over datasets where **all** algorithms
  succeeded; "saturated" = best algorithm ≥ `saturation_threshold`.
- `src/stats.py` — `python -m src.stats <run> [--no-saturated]`: Friedman
  test, Nemenyi critical difference + CD diagram (`summary/cd_diagram.png/pdf`),
  pairwise Wilcoxon signed-rank with Holm correction. Uses the same pivot/rank
  functions as `summary.py`. Verified: q(6, 0.05) from scipy = 2.850 (Demšar).
- `scripts/` — `gen_cc18_ids.py` (pins the 72 CC18 IDs), `profile_datasets.py
  --dataset-set X [--from-cache]`, `prestage.py --dataset-set X` (Arnes login
  node), `run_cc18.sh` / `run_subset.sh` (SLURM, take `RUN_ID`),
  `merge_results.py <run>`, `compare_results.py <run A> <run B>`,
  `gen_version_table.py <run>` (LaTeX table of versions from the manifest),
  `profile_medic3.py` (data card, stdlib only), `medic3.json` (Medic3 spec),
  `subset_ids.json`, `cc18_ids.json`.
- `results/` — see `results/README.md`. `results/runs/<run_id>/` per run
  (`manifest.json`, `per_dataset/`, `predictions/` [gitignored],
  `results.csv`, `summary/`). `results/arnes/cc18/` is the only archive left
  (2026-08 sweep, old code/versions), kept until `cc18_v2` replaces it. The
  old `results/local/` and `results/arnes/subset/` were deleted on 2026-09-09
  (git history has them); `runs/check_refactor_oldenv` is the old-env
  reference now.
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
  bit the first `tabular2` pilot (all 5 `sick` folds failed) and is fixed.
- **TabPFN / TabICL**: raw input first; only on an exception apply a logged
  minimal fix and retry. Under the current versions raw input works on every
  CC18 dataset; the fallbacks stay as safety nets.

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
- **Conda envs.** `tabular2` (Python 3.12) is the current one, built
  2026-09-09 from `requirements.txt`; GPU RTX 3060, `torch 2.14.0+cu130`,
  CUDA works. `tabular` (Python 3.10, old pinned stack) is kept untouched as
  the reference environment that produced `results/runs/check_refactor_oldenv`
  and `results/arnes/cc18`. Run project scripts with
  `conda run -n tabular2 python -m src.<module>`.
- **Version policy (write this in the thesis):** all libraries pinned to the
  newest stable release co-installable as one environment on the freeze date
  2026-09-09, identical locally and on Arnes, so no algorithm family is
  disadvantaged by library age. `requirements.txt` lists the top-level pins;
  each run's `manifest.json` carries the full `pip freeze`;
  `scripts/gen_version_table.py` produces the LaTeX table from it (never type
  versions by hand into the thesis).
- **Measured effect of the upgrade** (pilot 3 datasets, same machine,
  `compare_results.py check_refactor_oldenv subset_v2_local`): RandomForest,
  LightGBM, CatBoost bit-identical; TabPFN/TabICL ≤ 2e-3; **XGBoost 2.1.1 →
  3.4.1 changes ROC-AUC by up to 0.0175** on a fold (its defaults moved) —
  worth a sentence in the thesis.
- **Refactor verified:** the restructured code run in the *old* env
  (`results/runs/check_refactor_oldenv`) reproduced the July 2026 local
  baseline (`results/local/results_local_subset.csv`, since deleted, in git
  history) with Δ = 0.0 on all 90 rows, including TabPFN/TabICL.
- TabPFN v3 needs a one-time licence acceptance via a PriorLabs account; the
  credential is cached locally. Fresh machine: interactive first `fit()` opens
  a browser login (needs a real TTY — `conda activate`, not `conda run`), or set
  `TABPFN_TOKEN` from https://ux.priorlabs.ai/account.
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
- `medic3` = `scripts/medic3.json` → `../Medic3.csv.zip` (outside the repo,
  confidential): 122 093 × 275, 220 classes, 82 % missing, `Index`/`Field`
  dropped, `Class` target. Loader verified 2026-09-09
  (`profile_datasets.py --dataset-set medic3 --from-cache`). TabPFN v3's hard
  cap is 160 classes, so it will fail-soft on Medic3 as agreed with the
  supervisor; TabICL handles 220 classes.

## Arnes HPC
- SLURM cluster, GPU partition `gpu`, H100 nodes. Env: micromamba prefix
  `~/envs/tabular2` (Python 3.12, from `requirements.txt`; `~/bin/micromamba`,
  no shell hooks in batch scripts). The batch scripts default to that prefix;
  override with `TABULAR_ENV=/path sbatch ...`. The old `~/envs/tabular` on
  the cluster can be deleted once `subset_v2` matches `subset_v2_local`.
- TabPFN token in `~/.tabpfn_token` (sourced by the batch script); compute
  nodes run offline (`HF_HUB_OFFLINE=1`), so on the login node first:
  `python scripts/prestage.py --dataset-set cc18` (caches datasets + TabPFN/
  TabICL weights, prints the cache size for the 100 GB home-quota check).
- Submit from the repo root with an explicit run id:
  `RUN_ID=cc18_v2 sbatch scripts/run_cc18.sh` (array `0-71%4`, `--mem=64G`,
  `--time=12:00:00`). The script checks that the array bound matches the
  dataset count and aborts loudly if not. Re-submit the four big indices with
  more memory, **same RUN_ID**:
  `ALLOW_SPARSE_ARRAY=1 RUN_ID=cc18_v2 sbatch --array=27,60,61,70 --mem=240G scripts/run_cc18.sh`.
- Per-fit checkpointing: a killed task loses no work; re-submitting with the
  same `RUN_ID` resumes at the first unfinished (algorithm, fold). A new
  `RUN_ID` is a clean slate, so the old "rm -rf results/per_dataset/*" hazard
  is gone. Verified 2026-07-22 (kill/resume, max Δ 0.0).
- Merge and analyse: `python scripts/merge_results.py cc18_v2`, then
  `python -m src.summary cc18_v2`, `python -m src.stats cc18_v2` (and
  `--no-saturated`), `python scripts/gen_version_table.py cc18_v2`. Expect
  72 × 6 × `n_splits` × `n_repeats` rows; fewer rows means unfinished (the merge
  lists the partials), a failed fit still has its row with the reason in
  `error`.
- Record the job receipt after every run:
  `sacct -j <jobid> --format=JobID,JobName%20,Elapsed,MaxRSS,State,NodeList`
  into a `PROVENANCE.md` inside the run directory.
- Sizing knowledge from the 2026-08 sweep: `--mem=64G` sufficed for 68/72;
  indices 27/60/61/70 (mnist_784, Devnagari-Script, CIFAR_10, Fashion-MNIST)
  needed 120–240G; `--time=12:00:00` sufficed everywhere; CatBoost (1000
  default iterations) took 13 of the 16 total CPU hours. CIFAR_10 × {TabPFN,
  TabICL} failed (3072 features > TabPFN's 2000 cap; TabICL asked ~378 GB) —
  those rows carry the reason in `error`, no subsampling was or will be added.
- The cluster remote uses SSH (`git@github.com:...`), not HTTPS.

### 2026-08 CC18 sweep (archive, old code + old versions)
`results/arnes/cc18/results_arnes_cc18.csv`, 2160 rows, `PROVENANCE.md` with
4 SLURM job IDs. Headline (mean rank, failure = worst rank, per-dataset ranks):
tabicl 1.58, tabpfn 1.88, catboost 3.38, lightgbm 4.50, xgboost 4.59,
random_forest 5.07; Friedman p = 5e-47, Nemenyi CD = 0.89; 33 of 72 datasets
are saturated (best ROC-AUC ≥ 0.995). Reproduce with
`python -m src.summary results/arnes/cc18/results_arnes_cc18.csv` and
`python -m src.stats results/arnes/cc18/results_arnes_cc18.csv`.

## What is next (agreed 2026-09-09)
1. Build `~/envs/tabular2` on Arnes from `requirements.txt`, prestage, run
   `RUN_ID=subset_v2 sbatch scripts/run_subset.sh`, compare with
   `subset_v2_local` (expect trees Δ 0, foundation models ~1e-4). Exact
   commands are in `razlaga_repozitorija/razlaga.md`, part 3.
2. Run the full CC18 with `RUN_ID=cc18_v2`; decide `n_repeats` in
   `config.yaml` first (1 = as before; 3 ≈ 50 CPU-hours mostly CatBoost).
3. Then the supervisor's plan (2026-08-12): Medic3 raw, Medic3 restricted to
   160 classes, CC18 leakage check, tuning one booster, CC18 with injected NULLs.

## FRI methodology requirements (official diploma guidelines)
Writing/citation rules live in the thesis repo's `CLAUDE.md`
(`../Diplomsko-delo_Tabelaricni-temeljni-modeli/CLAUDE.md`).
- **Baseline — deliberately not used. Decided 2026-08-16, do not re-raise.**
  The four tree ensembles are the reference point; a `DummyClassifier` sits at
  ROC-AUC 0.5 by construction. No such entry in `REGISTRY`, none to be added.
- **Reproducibility.** Fixed seeds (`random_state` from `config.yaml`
  everywhere, including every model), every parameter in `config.yaml`,
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
