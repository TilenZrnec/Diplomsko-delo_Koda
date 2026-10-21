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
TabPFN-3 is gone. **`results/runs/cc18_v3/` is the result set of the thesis**
(finished 2026-10-08: 6480 rows, 0 errors — all six algorithms succeeded on all
72 datasets; TabPFN-3.5 now ranks first, TabICL second, the reverse of
`cc18_v2`; see its `PROVENANCE.md`). The previous run `cc18_v2` (TabPFN-3) was
deleted from the working tree on 2026-10-08 and is read from git history
(commit `5a14045`) where needed. **Medic3 finished 2026-10-11**: runs
`medic3_raw` and `medic3_160` (see Datasets → Medic3 results); LightGBM collapses
there at its default settings (ROC-AUC ~0.50), verified not to be our code.
**Out-of-the-box fix 2026-10-14/15:** RandomForest and XGBoost preprocessed more
than their libraries need; fixed and rerun as `cc18_v4` / `medic3_raw_v2` /
`medic3_160_v2` (only those two algorithms). **The thesis result sets are now
the `combined/` folders of those runs** (RF + XGBoost from the rerun, the other
four from `cc18_v3` / `medic3_raw` / `medic3_160`; see Preprocessing policy →
Audit and What is next, step 6). Conclusions unchanged. Next: thesis update,
tuning, then the leakage test. The TabPFN-3.5 licence is accepted and the weights are
cached on both local machines (Kremen since 2026-10-06).
**Clean-up 2026-10-05:** all older runs (`check_refactor_oldenv`,
`subset_v2_local`, `subset_v2`) and the 2026-08 archive `results/arnes/` were
deleted; they live in git history, last present in commit `b1b9120` — e.g.
`git show b1b9120:results/runs/subset_v2/results.csv` or
`git checkout b1b9120 -- results/arnes`.

## Run registry — what every run IS (read this before talking about runs)
**The user asked for this explicitly (2026-10-15): there are many run names and
they get confusing.** Whenever you mention a run to the user, say what it is in
plain words, e.g. "`cc18_v4` (the RandomForest + XGBoost rerun)", never the bare
name. Update this table in the same commit as any new, deleted or re-purposed
run, and keep `results/README.md` (Slovenian, same table) in sync.

Name parts: `cc18` = the 72 OpenML-CC18 datasets; `medic3_raw` = the
confidential hospital dataset Medic3 with all 220 classes; `medic3_160` =
Medic3 limited to its 160 most frequent classes (TabPFN's class limit);
`subset` = three small CC18 datasets (credit-g, diabetes, sick) for quick
checks; `_v2`/`_v3`/`_v4` = version of that result set (new library freeze or
protocol fix); `_local` = run on Kremen instead of Arnes; **`combined/`** = the
full six-algorithm table assembled by `scripts/combine_runs.py` from a partial
rerun plus the earlier run.

| Run | What it is | Role now |
|---|---|---|
| **`cc18_v4/combined/`** | CC18, all six algorithms, 15 folds each: RF + XGBoost from `cc18_v4`, the other four from `cc18_v3` | **THE CC18 thesis results** |
| `cc18_v4` | CC18, **only RandomForest + XGBoost**, rerun after the out-of-the-box fix (Arnes, 2026-10-14/15) | source of RF and XGBoost |
| `cc18_v3` | CC18, all six algorithms, library freeze 2026-10-05 with TabPFN-3.5 (Arnes, 2026-10-06/08) | source of LightGBM, CatBoost, TabPFN-3.5, TabICL; **its RF and XGBoost rows are superseded** (old preprocessing) — never cite them |
| **`medic3_raw_v2/combined/`** | Medic3 (220 classes), all six algorithms | **THE Medic3-220 thesis results** (TabPFN fails on the class limit, LightGBM collapses) |
| `medic3_raw_v2` | Medic3 (220), **RF (fixed) + XGBoost (check, identical to before)** (Arnes, 2026-10-15) | source of RF and XGBoost |
| `medic3_raw` | Medic3 (220), all six algorithms (Arnes, 2026-10-10/11) | source of LightGBM, CatBoost, TabPFN, TabICL; RF row superseded; `diagnostics/` = LightGBM-collapse analysis |
| **`medic3_160_v2/combined/`** | Medic3 (160 classes), all six algorithms | **THE Medic3-160 thesis results** (the only Medic3 variant where all six run) |
| `medic3_160_v2` | Medic3 (160), **RF (fixed) + XGBoost (check, identical)** (Arnes, 2026-10-15) | source of RF and XGBoost |
| `medic3_160` | Medic3 (160), all six algorithms (Arnes, 2026-10-10/11) | source of LightGBM, CatBoost, TabPFN-3.5, TabICL; RF row superseded; `diagnostics/` = LightGBM-collapse analysis |
| `cc18_xgb_tuned` (Arnes, CPU only) | **tuning experiment**: only `xgboost_tuned` (XGBoost tuned by grid search with nested CV) on `cc18_tuning` — part 1 = the 36 non-saturated CC18 sets (done 2026-10-15, job `20494570`, 540 rows), part 2 = the other 32 (done 2026-10-20, jobs `20511472`, `20511665`, `20652997`); 1020 rows, checked 2026-10-21; `tuning/` = all 128 inner scores per fold; `metrics.csv` = accuracy and log loss | thesis results of experiment 4 (compared with `cc18_v4/combined`) |
| `subset_v4_local` | 3 datasets, RF + XGBoost, Kremen: check of the out-of-the-box fix; `diagnostics/audit_defaults.txt` = audit of all six | evidence only, not thesis results |
| `cc18_leakage` (planned, Arnes H100) | **leakage test**: 34 non-saturated CC18 sets × 3 arms (original, monotone warp, warp + noise) × CatBoost, TabPFN-3.5, TabICL × folds 0–4 = 1530 rows; `DESIGN.md` = the pre-registered design with sources; `diagnostics/perturbation_check.csv` = data-only check of the arms | thesis results of experiment 3 (with `medic3_160_leakage`) |
| `medic3_160_leakage` (planned, Arnes) | leakage-test reference: Medic3-160 × the same 3 arms, 45 rows (FMs on GPU, CatBoost on the CPU partition) | reference for `cc18_leakage` |
| `subset_xgb_tuned_local` | 3 datasets (credit-g, diabetes, sick), `xgboost_tuned`, Kremen: pilot of the tuning experiment before Arnes | evidence only |
| `leakage_smoke_local` | sick × 3 arms × CatBoost, TabPFN-3.5, TabICL, Kremen RTX 3060, from `6444529` (clean tree), 45 rows, 0 errors: technical check of the leakage code (sick is saturated, so not in the test) | evidence only |
| `subset_v3_local` | 3 datasets, all six, Kremen: check that the 2026-10-05 freeze works | evidence only |
| `subset_v3` | the same check on Arnes | evidence only |
| `cc18_v2` (deleted; git `5a14045`) | CC18, all six, older freeze 2026-09-09 with **TabPFN-3** | read from git only for the thesis table "TabPFN-3 vs TabPFN-3.5" |
| older runs (git `b1b9120`) | pre-September runs and the August archive | history only |

**Thesis status (2026-10-15, thesis commit `76bf380`):** the Results chapter
reads `cc18_v4/combined/` and the two Medic3 `combined/` folders through
`rezultati/generiraj_rezultate.py` (thesis repo); the TabPFN-3 comparison
still uses `cc18_v3` vs `cc18_v2` on purpose (same code, only TabPFN differs).
**Secondary metrics (2026-10-15):** accuracy and log loss of all three thesis
sets were computed from the saved predictions (`scripts/compute_metrics.py`;
every successful fit had its file; ROC-AUC from the files within 3.2e-4 on CC18
and 7.5e-4 on Medic3 — the latter only the collapsed LightGBM, whose
near-0/1 probabilities tie after float32 rounding). Findings: on CC18 the top
three (TabPFN-3.5, TabICL, CatBoost) are the same under all three metrics and
both foundation models beat every tree ensemble (Wilcoxon–Holm) under all of
them; only the bottom three swap places, and RandomForest is clearly last by
log loss. **On Medic3-160 CatBoost beats TabPFN-3.5 on accuracy (n.s.,
p_Holm 0.084) and log loss (significant, 15/15 folds, p_Holm 0.0085)**, the
reverse of ROC-AUC. Thesis: tables `tab_metrike`, `tab_medic3_metrike`, section
`sec:rez-metrike`, subsection `sec:metrike`; new chapter "Sorodni pristopi".
After every future run (incl. `cc18_xgb_tuned`) run `compute_metrics.py` on it
while its predictions are on the machine.

## Structure
- `config.yaml` — **the single source of every experiment parameter**:
  `random_state`, `n_splits`, `n_repeats`, `algorithms`, `dataset_sets`
  (name → JSON file of dataset specs), `cache_dir`, `results_dir`,
  `save_predictions`, `saturation_threshold`, `alpha`, and `tuning` (inner
  folds, `n_estimators` list, grid — only for `xgboost_tuned`), and `leakage`
  (the leakage test's fixed parameters: arms, noise, seed, models, control).
  `dataset_sets` also has `medic3_160` (Medic3 capped at 160 classes),
  `cc18_tuning` (the 68 sets of the tuning experiment) and the leakage sets
  `cc18_leakage`, `medic3_160_leakage`, `leakage_smoke` (written by
  `scripts/gen_leakage_ids.py`, never by hand). Nothing is hardcoded
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
  rows that dropped. Leakage-test keys (2026-10-21, What is next step 8):
  `perturbation` (`original` | `monotone` | `monotone_noise`, `_perturb`;
  with `noise_sd`, `perturbation_seed`) changes only the numeric attributes
  (signed cube root of the standardised value, noise before the root; NaN,
  categorical, constant and all-missing columns untouched), adds `_<arm>` to
  the dataset name and `spec_id`, and returns `perturbation` (what was done);
  `repeats: r` keeps only the first r repeats' folds, cut from the full
  `n_repeats` split so they are exactly the thesis folds.
- `src/runner.py` — **the only training loop.** `run_dataset()` runs every
  algorithm on every fold of one dataset with per-fit checkpointing
  (`<id>.csv.partial`, atomic finalisation to `<id>.csv`; with an explicit
  `--algorithms` subset the file is `<id>__<algos>.csv`, so parallel SLURM tasks
  on the **same** dataset do not overwrite each other's checkpoints — this is
  how Medic3 is split by algorithm), GPU warm-up, saved
  predictions, and a row per fit with `preprocessing`, `raw_error`, `error`,
  `device`, `git_commit`, `hostname`, `timestamp`, `hyperparameters` (JSON, only
  for tuned algorithms; empty otherwise). A result with a `tuning_table` is also
  saved as `tuning/<id>/<algo>_r<r>_f<f>.csv` (metrics only, committed). Also `write_manifest()`
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
  `xgboost_tuned` (`src/models/xgboost_tuned.py`) is registered but **not** in
  `config.yaml` `algorithms`: the tuning experiment, run with `--algorithms
  xgboost_tuned` (see What is next, step 7). It uses `xgboost_model.prepare()`,
  so its input is identical to the default XGBoost's.
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
- `src/metrics.py` — **secondary metrics** (decided 2026-10-15, user):
  `compute_accuracy` (argmax of `proba`, the same rule as `predict()`),
  `compute_log_loss` (sklearn `log_loss`, which clips to [eps, 1-eps] with eps =
  2.2e-16 in float64, so one row costs at most 36.04; the stored float32 `proba`
  is cast to float64 and each row renormalised to sum 1 — max row-sum error seen
  3.6e-7), and `HIGHER_IS_BETTER` (log loss is the only lower-is-better metric).
  **Deliberately not on the training path** (no model or runner imports it), so
  adding metrics never changes the code that trains. ROC-AUC stays the primary
  metric (computed during training, `src/utils.py`); why these two were added
  and the literature behind it: thesis `sec:sor-zbirke` (table `tab:sorodni`) and
  `sec:metrike`.
- `src/summary.py` — `python -m src.summary <run_id|dir|csv> [--metric M]` (path is
  **required**; `--metric accuracy|log_loss` reads `metrics.csv` and writes to
  `summary/<metric>/`, ROC-AUC output in `summary/` is unchanged — regenerating
  all three thesis sets after the change gave byte-identical files; saturation
  is always decided by ROC-AUC so the non-saturated subset is the same for every
  metric). Prints and writes to `summary/` (CSV + LaTeX): per-dataset
  table, dataset×algorithm pivot, ranks, overall table, and the same without
  saturated datasets. Rules: rank **per dataset** on the fold mean; a
  (dataset, algorithm) with any failed fold is a failure and gets the **worst
  rank**; mean ROC-AUC is reported over datasets where **all** algorithms
  succeeded; "saturated" = best algorithm ≥ `saturation_threshold`.
- `src/stats.py` — `python -m src.stats <run> [--no-saturated] [--metric M]`
  (`--metric` as in `src.summary`; for log loss ranks and wins count the lower
  value as better, `mean_diff` stays raw a − b). Several
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
  `RUN_ID`; `run_cc18.sh` also `ALGORITHMS="a b"` for a subset of algorithms),
  `audit_defaults.py` (proves every algorithm runs out of the box, see
  Preprocessing policy; GPU for TabPFN/TabICL, a few minutes on Kremen),
  `gen_tuning_ids.py` (rebuilds `cc18_tuning_ids.json` from `cc18_v4/combined`:
  68 sets, the 36 non-saturated first), `run_tuning.sh` (SLURM, tuning
  experiment, CPU only, one dataset per task, takes `RUN_ID`),
  leakage test (What is next, step 8): `gen_leakage_ids.py [--check]` (writes
  `cc18_leakage.json`, `medic3_160_leakage.json`, `leakage_smoke.json` from
  `config.yaml` `leakage`; `--check` = data-only verification of the arms,
  `results/runs/cc18_leakage/diagnostics/perturbation_check.csv`),
  `run_leakage.sh` (SLURM GPU, one dataset per task, its arms in a row on one
  node, takes `RUN_ID`, optional `DATASET_SET`, `ALGORITHMS`),
  `run_leakage_cpu.sh` (SLURM CPU partition, one spec = one arm per task; for
  Medic3-160 CatBoost), `analyze_leakage.py <run> [--reference <run>]` (the
  pre-registered analysis, writes `summary/leakage_*`),
  `compute_metrics.py <run|dir>` (accuracy and log loss of every fit from its
  saved `predictions/*.npz` — no re-run; for `combined/` the row's `run_id`
  says which run's predictions to open; also recomputes ROC-AUC from the file as
  a check, exit 1 on a missing file or |Δ| > 1e-3; writes `metrics.csv` and
  `metrics_info.json` next to `results.csv`),
  `combine_runs.py <new> --base <old>` (full result set from a partial rerun:
  all rows of the new run + the old run's rows for the algorithms the new one
  lacks, into `<new>/combined/` with `sources.json`; refuses if seeds, folds or
  datasets differ; then `src.summary` / `src.stats` on `combined/results.csv`),
  `merge_results.py <run>`, `compare_results.py <run A> <run B>`,
  `gen_version_table.py <run>` (LaTeX table of versions from the manifest; adds
  the weights file from `model_checkpoints` to the TabPFN/TabICL rows),
  `profile_medic3.py` (data card, stdlib only), `diagnose_lightgbm_medic3.py`
  (why LightGBM collapses on Medic3: fold 0 through our code and through plain
  pandas/sklearn/LightGBM, defaults vs `reg_lambda=1` vs `min_child_weight=1`,
  training loss and largest leaf per iteration; writes
  `results/runs/<run>/diagnostics/`; CPU only, ~35 min on the laptop, the
  independent part also runs in `tabularOriginal`), `medic3.json` /
  `medic3_160.json` (Medic3 specs, raw and capped at 160 classes),
  `subset_ids.json`, `cc18_ids.json`.
- `results/` — see `results/README.md`. `results/runs/<run_id>/` per run
  (`manifest.json`, `per_dataset/`, `predictions/` [gitignored],
  `results.csv`, `summary/`, `diagnostics/` [checks made by a script after the
  run], `metrics.csv` + `metrics_info.json` [`scripts/compute_metrics.py`],
  `summary/<metric>/` [secondary metrics]). Kept: **`cc18_v4/combined/`** (the thesis results: RF + XGBoost from
  `cc18_v4`, LightGBM, CatBoost, TabPFN, TabICL from **`cc18_v3`**, which stays
  as their source), the Medic3 runs **`medic3_raw`** / **`medic3_160`** and
  their partial reruns **`medic3_raw_v2`** / **`medic3_160_v2`** (thesis sets:
  their `combined/`), and the two validation runs of the
  freeze, `subset_v3_local` (Kremen) and `subset_v3` (Arnes), plus
  `subset_v4_local` (Kremen, RF + XGBoost only: local validation of the
  2026-10-14 out-of-the-box fix before `cc18_v4`); everything older
  is in git history — `cc18_v2` e.g.
  `git show 5a14045:results/runs/cc18_v2/results.csv` (the thesis generator
  reads it that way), the rest in `b1b9120` (see the state note at the top). Rule: keep only the runs the thesis currently uses, delete
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
**Rule (user, 2026-10-14): every algorithm runs out of the box** — library
default hyperparameters and only the preprocessing without which the library
rejects the input. Not the best preprocessing. Each row of `results.csv`
records what was applied (`preprocessing`) and, for the foundation models,
whether the raw input failed first (`raw_error`). **`scripts/audit_defaults.py`
proves it** (run it after every library upgrade; exit code 1 on any breach):
it intercepts each library's `fit` while running our module on `sick` fold 0,
compares the parameters with the class defaults, repeats with unordered
categories, and tests on synthetic data that every preprocessing step we do is
one the library needs. Result 2026-10-14: all six pass (see Audit below).
- **RandomForest** (sklearn 1.9.1): **native NaN** (since sklearn 1.4); it
  rejects strings and has no categorical parameter, so categorical columns are
  ordinal-encoded (NaN stays NaN, unseen category → -1), fit on the train fold
  only; numeric columns cast to float (object columns). Column order numeric
  then categorical, as in `cc18_v3`, so sets without NaN reproduce `cc18_v3`.
  **Until 2026-10-14 (`cc18_v3`, `medic3_raw`, `medic3_160`) it imputed
  (median / most frequent)** on the wrong belief that RF has no NaN support.
- **XGBoost** (3.4.1): **native NaN and native categories** —
  `enable_categorical=True` is the 3.x default; categorical columns go in as
  pandas `category` (test fold aligned to the train fold's categories, a no-op
  for OpenML), numeric columns cast to float (it rejects object dtype).
  `xgboost_model.prepare()` is the shared step (also for tuning). **Until
  2026-10-14 (`cc18_v3`) it got ordinal codes**, which were minimal only in
  xgboost 2.1.1 (default False).
- **LightGBM**: native NaN; categorical columns cast to pandas `category` and
  named in `categorical_feature`; non-categorical columns cast to float (it
  rejects object dtype). Why the explicit list: the default `'auto'` takes only
  *unordered* categoricals, and openml-python marks every nominal attribute
  `ordered=True`; a plain `fit(X, y)` on the loader's frame would silently
  treat them as integer codes. Our list equals plain `fit(X, y)` on correctly
  typed (unordered) columns — max |Δproba| 0.0 on credit-g and sick, checked by
  the audit. So `cc18_v3`'s LightGBM is out of the box and stays.
- **CatBoost**: native NaN (numeric) + native categorical via `cat_features`;
  categorical columns mapped element-wise to strings with NaN → the string
  `'nan'` (its own category, not an imputation). Both steps are required: it
  rejects a `category` column not in `cat_features` and NaN inside a
  categorical feature. **pandas 3.0 note:** `astype(str)` no longer turns NaN
  into `'nan'`, it keeps it missing and CatBoost rejects it; hence the explicit
  `map` in `catboost_model.py`. This bit the first `tabular3` pilot (all 5
  `sick` folds failed) and is fixed.
- **TabPFN / TabICL**: raw input first; only on an exception apply a logged
  minimal fix and retry. With TabPFN-3 (tabpfn 8.5.0) raw input worked on every
  CC18 dataset, and so it did with TabPFN-3.5 in `cc18_v3` (0 `raw_error` rows
  for both foundation models). The fallbacks stay as safety nets. "Raw" means
  no preprocessing *by us*: internally TabICL mean-imputes numeric NaN without
  an indicator, TabPFN-3.5 adds NaN indicators before imputing (see Datasets →
  Medic3, where this matters). TabPFN's explicit `model_path` (version pinning,
  see Structure) is the very file the default `'auto'` resolves to
  (`tabpfn-v3.5-20260909.safetensors`; the audit recomputes it).
- **The `ordered=True` flag** that openml-python puts on all categorical
  columns changes nothing in any of our six modules (max |Δproba| 0.0, all six,
  credit-g and sick; the audit repeats it every run).
- **Audit 2026-10-14 (the finding behind the rule).** Two modules preprocessed
  more than their library needs: XGBoost (ordinal codes; 3.x handles
  categories by default) and RandomForest (imputation; sklearn ≥ 1.4 handles
  NaN). Hyperparameters were defaults everywhere; the only non-defaults are
  the seed and the compute/logging settings listed under Conventions. Affected
  results: XGBoost on the 23 CC18 sets with categorical columns (Medic3 is all
  numeric, so XGBoost there is unaffected); RandomForest on the 9 CC18 sets
  with missing values (adult, sick, MiceProtein, cylinder-bands,
  dresses-sales, eucalyptus, credit-approval, jm1, breast-w; OpenML counts)
  and on both Medic3 runs (82 % missing). **Partial rerun (user, 2026-10-14):**
  XGBoost + RandomForest on all 72 CC18 sets as `cc18_v4` (the unaffected sets
  double as a check against `cc18_v3`) and RandomForest on Medic3 as
  `medic3_raw_v2` / `medic3_160_v2`; LightGBM, CatBoost, TabPFN, TabICL stay
  from `cc18_v3` / `medic3_raw` / `medic3_160` (code unchanged). Local
  validation `subset_v4_local` (Kremen, 90 fits, 0 errors) vs `cc18_v3`:
  diabetes RF and XGBoost Δ 0; credit-g RF Δ 0, XGBoost changed (mean 0.7723 →
  0.7610); sick (the only one with NaN) both changed (RF 0.9966 → 0.9968,
  XGBoost 0.9965 → 0.9971).

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
  wheel/packaging. Details in `results/runs/subset_v3/PROVENANCE.md`. **Full
  run `cc18_v3` vs `cc18_v2`** (6480 fits): CatBoost, LightGBM, XGBoost Δ 0;
  RF 31/1080 fits ≤ 2.3e-5 (parallel-prediction nondeterminism, see
  Conventions); TabICL 147/1065 ≤ 3.7e-3 (largest on adult, which moved from
  H100 PCIe to HBM3; cause not investigated); TabPFN-3.5 beats TabPFN-3 on
  56/71 datasets and runs on CIFAR_10. Ranks: TabPFN 1.90 → 1.43, TabICL
  1.58 → 1.88 (TabICL was significantly better in `cc18_v2`, 47:20, p_Holm
  0.0046; now TabPFN-3.5 is, 45:23, p_Holm 0.0018). History (git
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
    `Field` is the department ("oddelek", 11 values; dropped together with
    `Index` on the supervisor's instruction). **Column check 2026-10-19:**
    header names are exact (`Index` col 0, `Field` 276, `Class` 277 of 278);
    `drop_cols` is `["Index", "Field"]` at `420bdac` and `8b601d1`; all four
    Medic3 runs record `n_features` 275 for every algorithm and fold; file
    sha256 = PROVENANCE; no attribute is a 1:1 relabelling of `Field`, the best
    single one (A177) predicts it at 0.418 held-out vs 0.323 majority;
    `_load_csv` would skip a missing `drop_cols` name silently, left as is
    because the Medic3 file is never changed (user, 2026-10-21);
  - **missingness itself is a strong predictor** (e.g. A177 observed in 99.5 %
    of class C152's rows vs. 0.0 % of C202's). **Only TabICL loses this signal,
    inside its own library** (until 2026-10-14 RandomForest did too, through
    *our* median imputation, which was not minimal — removed, RF now takes NaN
    natively; `medic3_raw` / `medic3_160` RF rows are superseded by
    `medic3_raw_v2` / `medic3_160_v2`) — tabicl 2.2.0 `TransformToNumerical`
    (`tabicl/_sklearn/preprocessing.py`) runs `SimpleImputer(keep_empty_features
    =True)`, i.e. the column mean with no missing indicator, on every numeric
    column before the model sees it. TabPFN-3.5 keeps the signal:
    `tabpfn/architectures/tabpfn_v3_5.py` builds NaN indicators *before* its own
    mean imputation ("Indicators need to be computed before imputation") and
    feeds both to the model; XGBoost, LightGBM and CatBoost handle NaN natively.
    That is exactly the experimental variable the thesis measures, so expect
    TabICL to fall behind on Medic3 partly for preprocessing reasons, not
    model reasons. Say this in the thesis; do not "fix" it (TabICL's imputation
    is the library default, not our preprocessing). TabPFN-3's hard
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
- **Medic3 results (2026-10-11, commit `420bdac`; SLURM receipts, sha256 and
  details in each run's `PROVENANCE.md`).** `medic3_160` (all six): TabPFN-3.5
  0.9614, CatBoost 0.9599, TabICL 0.9543, XGBoost 0.9400, RF 0.8530, LightGBM
  0.5057; all 15 pairs significant (corrected t-test + Holm), the weakest
  TabPFN-3.5 vs CatBoost (−0.0015, TabPFN better on 15/15 folds, p_Holm 0.005).
  `medic3_raw` (TabPFN fails soft on the class limit, as expected): CatBoost
  0.9551, TabICL 0.9499, XGBoost 0.9306, RF 0.8149, LightGBM 0.5028; all 10
  pairs significant.
  **LightGBM collapses on all 30 folds (0.499–0.519) at its defaults, and it is
  not our code** (`scripts/diagnose_lightgbm_medic3.py`, outputs in each run's
  `diagnostics/`): plain LightGBM with none of our code collapses the same way;
  its own training loss rises (4.9 → 32.6–33.0 raw, 3.9 → 32.7–32.9 at 160) while the
  largest leaf value grows from ~10² to ~10⁶ and predictions become ~100 %
  confident and wrong; our pipeline equals the plain path until the leaves
  explode (8–10 iterations in the first laptop runs, only 2 in the reruns from the
  commit; the largest leaf is already ~2000 at iteration 2); one brake fixes it on the same fold (`reg_lambda=1` → 0.938 raw /
  0.948 at 160, `min_child_weight=1` → 0.938 / 0.945). Those are XGBoost 3.4.1's
  defaults (`lambda` 1, `min_child_weight` 1; CatBoost 1.2.10 `l2_leaf_reg` 3),
  while LightGBM 4.7.0 has `reg_lambda` 0.0 and `min_child_weight` 1e-3;
  LightGBM 4.6.0 collapses the same way. Mechanism (our reading): leaf value
  −G/(H+λ) with per-row hessians p(1−p) ≈ 0 for rare classes. LightGBM raises no
  error and only warns "Stopped training because there are no more leaves…"
  (hidden by our `verbosity=-1`, which does not change the model). **Decision
  2026-10-12 (user): report the default result as is (option A)**; the braked
  numbers are diagnostics, never LightGBM's result. The exact digits of a
  collapsed LightGBM are **not reproducible** (two identical calls on one
  machine differ; fold 0 raw: Arnes 0.5000 vs local 0.500–0.510; at 160: Arnes
  0.5168 vs local 0.500–0.515); the collapse is. The braked numbers and the
  first-iteration losses DO repeat, also across machines (`medic3_160`, laptop
  vs Kremen: accuracy and largest leaf identical, ROC-AUC within 3.7e-8, losses
  within 2e-13) — identical to every digit we report, not bit-identical.
  Diagnostics provenance (each file's `*_info.json` has commit, host, versions):
  `medic3_raw/diagnostics/lightgbm_4.7.0_*` on the laptop from `a5a2197`
  (2026-10-12); `medic3_160/diagnostics/*` and `medic3_raw/diagnostics/lightgbm_4.6.0_*`
  on Kremen from `79e1d26` (2026-10-13; 4.7.0 in `tabular3.5`, 4.6.0 in
  `tabularOriginal`). All from a clean tree with the same script (sha256
  `e438f529…`); the earlier pre-commit laptop files were replaced.

## Arnes HPC
- SLURM cluster, GPU partition `gpu`, H100 nodes; partition time limit
  **4-00:00:00** (`sinfo -p gpu -o "%P %l"`, 2026-10-06). All partitions
  (`sinfo -o "%P %l"`, 2026-10-15): **`all` (default, CPU nodes `wn…`) 2 days**,
  `long` 7 days, `gpu` 4 days, `mig-preempt` / `mig-priority` 1 day; sbatch
  refuses a longer `--time` at submission ("Requested time limit is invalid"). `--constraint=h100`
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
  **Sizing measured 2026-10-09** → three submissions per run, same `RUN_ID`:
  - RF/XGBoost/LightGBM (0,1,2) at default memory, `--time=1-00:00:00`.
  - TabPFN/TabICL (4,5) with **`--mem=160G`**: TabICL (default `offload="auto"`)
    keeps a column-embedding output of rows × features × 2048 B (8 ensemble
    members × 128 dims × 2 B — reproduces CIFAR_10's "~378 GB" exactly), i.e.
    ~69 GB for Medic3. That is over half the H100, so tabicl puts it in RAM if it
    fits 0.85 × the cgroup headroom; otherwise (no disk dir set) it allocates
    anyway and the cgroup kills the task with no row — at the old 64G default it
    would have died at fold 0. TabPFN-3.5 joins it because it never ran on ~118k
    rows.
  - CatBoost (3) with a long `--time` (3 days raw, 2 days `medic3_160`) and
    default memory. Local probe on Kremen (Ryzen 5 5600, 12 threads, fold 0,
    `catboost_model` code path): **6.72 s/iteration at 220 classes (1000
    iterations ≈ 1.9 h/fold), 4.58 s at 160 classes, peak RSS ≤ 4.1 GiB**.
    Calibration on fold 0 of two CC18 sets (ROC-AUC bit-identical to Arnes, so
    same folds/seed/model): isolet 443.9 s Kremen vs 627.6 s `gwn03` (PCIe),
    Fashion-MNIST 594.7 s vs 802.2 s `gwn05` (PCIe, `cc18_v2`) → **PCIe ≈ 1.35–1.41
    × Kremen**. From `cc18_v2` (all PCIe) vs `cc18_v3` (CatBoost bit-identical):
    **PCIe ≈ 1.6 × SXM** on 15 datasets from 0.7 s to 3400 s per fold (1.5–1.87);
    the outlier Fashion-MNIST (1.15) shared `gwn08` with mnist and CIFAR_10 in
    `cc18_v3`, so a shared node can cost ~35 %. Estimate for 15 folds: raw ~40 h
    PCIe / ~25 h SXM, `medic3_160` ~27 h / ~17 h. Probe script and raw numbers:
    `logs/medic3_probe/` on Kremen (gitignored).
  **Measured in the run (2026-10-10/11, no resubmissions):** TabICL MaxRSS
  99.4 GiB (raw) / 92.3 GiB (160), so the 160G request was needed; CatBoost
  2.26 h per fold on `gwn08` (SXM, shared with most other tasks; 33 h 48 min in
  all) and 1.54 h on `gwn09` (23 h 08 min), ~40 % slower than the probe-based
  SXM estimate; XGBoost only ~2.5 min per fold; TabPFN-3.5 on ~94k training
  rows 74 s per fold, MaxRSS 5.9 GiB.
  Exact commands in the header of `scripts/run_medic3.sh`. `prestage.py
  --dataset-set medic3` is optional: it loads the CSV (checks the path) and
  re-checks the cached TabPFN/TabICL weights, which must be there because
  compute nodes are offline.
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
  to copy its `predictions/` there. **Policy (user, 2026-10-13):** archive = USB
  drive, working copies = the WSL partitions of the laptop and Kremen; the `D:`
  copy below was a one-off for the older runs, not part of the policy.
  **One-off backup:** on 2026-10-09 the user
  copied the whole `results/runs/` of Kremen to its Windows drive
  `D:\fri\diplomska_win` (`/mnt/d/fri/diplomska_win` in WSL, **Kremen only**):
  `cc18_v2`, `cc18_v3`, `check_refactor_oldenv`, `subset_v2_local`,
  `subset_v3`, `subset_v3_local`. The non-TabPFN-3.5 folders (`cc18_v2`,
  `check_refactor_oldenv`, `subset_v2_local` — only their gitignored
  `predictions/` were left) were then deleted from Kremen after checking the
  backup was byte-for-byte complete; their tracked files are in git history.
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
- **Progress checker (not in the repo):** `~/preveri_medic3.py` on Arnes (copy on
  the laptop at `~/fri/diplomska/preveri_medic3.py`), stdlib only, read-only:
  folds done per algorithm, errors (the expected TabPFN class limit is tagged),
  time per fold and ETA vs the job's time left, current RAM, alarms in the logs,
  `sacct` states. Run it in the repo root with the env's Python:
  `~/bin/micromamba run -p ~/envs/tabular3.5 python ~/preveri_medic3.py`.
- **Copying from Arnes:** `scp`/`rsync` run on the machine that RECEIVES the
  files (laptop or Kremen); run on Arnes they try to log in to Arnes itself and
  fail with "Permission denied (publickey)".

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
4. **Done 2026-10-08: `cc18_v3`** (commit `69cc51e`; jobs `20138834`,
   `20138836`, and `20176087` for CIFAR_10). Full story — the CIFAR_10 OOM
   kill at TabICL on a 512 GB node with `--mem=240G`, the resubmission with
   `--constraint=sxm --mem=480G` (TabICL then used 394 GiB and succeeded),
   times, memory, node mix — is in `results/runs/cc18_v3/PROVENANCE.md`.
   Merged, summarised, stats (both variants), version table, `sacct.txt`,
   committed from Arnes (`ab56ff3`); predictions copied to Kremen, verified, and
   archived by the user on USB (see Arnes HPC → Keep the predictions).
   **Thesis updated the same day** (thesis repo): the generator is now
   `rezultati/generiraj_iz_cc18_v3.py` (reads `cc18_v3`, makes the dataset
   counts automatic, copies `summary/razlicice.tex` into the thesis as
   `tab_razlicice.tex`, and builds `tab_zagona.tex`, the cc18_v2-vs-cc18_v3
   comparison, reading `cc18_v2` via `git show 5a14045:...`); the Results
   chapter text was rewritten (TabPFN-3.5 first; no failures; new sections
   "Neuspela in pomnilniško zahtevna učenja" and "Primerjava s prejšnjim
   zagonom z modelom TabPFN-3"), and Methods got the new versions, the RF
   correction and the TabPFN-3.5 paragraph. **Open items left as `\todo` /
   `% PREVERI` in the thesis:** the XGBoost `enable_categorical` decision (see
   Preprocessing policy); a verifiable source for TabPFN-3.5 (none in
   `literatura.bib`, so no properties are claimed); the TabPFN-3 report's
   "200 atributov" vs the 2000-feature limit tabpfn 8.5.0 enforced; the
   documentation bib entries still point to the old versions (sklearn 1.5,
   xgboost 2.1.0, lightgbm 4.6.0) — the default values cited from them were
   re-checked against the new versions and still hold.
   `results/runs/cc18_v2` was then deleted from the working tree in its own
   commit (history: `5a14045`).
5. **Done 2026-10-11: Medic3.** Submitted 2026-10-10 ~10:45 from `hpc-login4`
   at `420bdac` as three submissions per run (jobs `20318167`–`20318172`), all
   COMPLETED, no resubmissions; merged, summarised and tested on Arnes and
   committed from there (`4df56e5`). Predictions copied to the laptop and the
   USB drive (2026-10-12) and from USB to Kremen (2026-10-13), verified on both
   (every successful fit has its file, ROC-AUC from the files matches
   `results.csv`; see each run's `PROVENANCE.md`). Results and the LightGBM
   collapse: Datasets → Medic3 results; receipts in each run's `PROVENANCE.md`.
6. **Done 2026-10-15: partial out-of-the-box rerun** (Preprocessing policy →
   Audit). Code fixed and validated on Kremen: `scripts/audit_defaults.py`
   passes for all six; `subset_v4_local` behaves as expected; Medic3 fold 0:
   XGBoost old and new module identical on Kremen
   (Δproba 0 — so the change cannot touch all-numeric data; Kremen vs the
   stored Arnes value differs by 6.2e-4 raw / 4.6e-4 at 160, hardware rounding,
   see Conventions), RandomForest with native NaN raw 0.8138 → 0.8348 and 160
   0.8489 → 0.8717, still ~11–14 s per fold. On Arnes, after `git pull` (no new packages; Medic3.csv
   and the CC18 OpenML cache are already there), from the repo root:
       ALGORITHMS="random_forest xgboost" ALLOW_SPARSE_ARRAY=1 RUN_ID=cc18_v4 sbatch --array=0-26,28-59,62-69,71%4 scripts/run_cc18.sh
       ALGORITHMS="random_forest xgboost" ALLOW_SPARSE_ARRAY=1 RUN_ID=cc18_v4 sbatch --array=27,60,61,70 --mem=120G --time=1-00:00:00 scripts/run_cc18.sh
       PYTHONUNBUFFERED=1 ALLOW_SPARSE_ARRAY=1 RUN_ID=medic3_raw_v2 sbatch --array=0,1 scripts/run_medic3.sh
       DATASET_SET=medic3_160 PYTHONUNBUFFERED=1 ALLOW_SPARSE_ARRAY=1 RUN_ID=medic3_160_v2 sbatch --array=0,1 scripts/run_medic3.sh
   (`--array=0,1` = RandomForest and XGBoost; XGBoost on Medic3 is rerun only
   as a check — identical input, expected Δ 0 to `medic3_raw` / `medic3_160` on
   Arnes — and so that RF and XGBoost come from the reruns everywhere.)
   The scripts still request one H100 per task although RF and XGBoost do not
   use it: same node pool as `cc18_v3`, so their train times stay comparable
   with the other four algorithms'. Afterwards: `merge_results.py` for each
   run, `sacct` into each run's PROVENANCE.md, commit from Arnes; copy the three
   `predictions/` folders to the laptop, Kremen and the USB drive.
   Checks after the run: `compare_results.py cc18_v3 cc18_v4` → XGBoost Δ 0 on
   the 49 sets without categorical columns, RF Δ ≤ ~2e-5 on the 63 without
   NaN, changes only on the 23 / 9 affected sets; Medic3 XGBoost is not rerun
   (all numeric, identical input) beyond that check. Then the thesis result set = `cc18_v3`'s
   LightGBM, CatBoost, TabPFN, TabICL + `cc18_v4`'s RF and XGBoost (a combine
   script writes it into a subfolder of `cc18_v4`, never into `cc18_v3`);
   regenerate summary/stats and the thesis tables; update Methods (RF and
   XGBoost preprocessing, the `\todo` in `sec:xgboost`) and the Medic3
   discussion (only TabICL imputes now).
   **Outcome (checked 2026-10-15 on Kremen; receipts in each run's
   PROVENANCE.md):** jobs `20470896`/`20470897` (`cc18_v4`, 2160 rows) and
   `20470898`/`20470899` (Medic3, 30 rows each), all COMPLETED, 0 errors, all
   rows from `8b601d1`, clean tree. XGBoost Δ 0 on the 49 CC18 sets without
   categorical columns and on both Medic3 runs (Arnes reproduces them exactly);
   RF on the 63 sets without NaN: 22/945 fits differ, max 1.3e-4 — the known
   parallel-prediction noise (cc18_v2 vs v3: 24/945), the largest being one
   flipped tie on blood-transfusion = 1/(2·35·114). XGBoost on the 23
   categorical sets: 12 better, 8 worse, 3 identical (binary-only categorical
   columns), mean −0.0003; RF on the 9 NaN sets: mean −0.0010; RF on Medic3
   better on 15/15 folds (raw 0.8149 → 0.8394, 160 0.8530 → 0.8726).
   Combined sets (`scripts/combine_runs.py`): rank order unchanged (TabPFN-3.5 >
   TabICL > CatBoost > LightGBM > XGBoost > RF on all 72), Friedman still
   rejects; Wilcoxon–Holm changes one pair per analysis: LightGBM vs RF (72)
   no longer significant (p_Holm 0.042 → 0.057), LightGBM vs XGBoost (37
   non-saturated) now significant (0.082 → 0.026); on the 37 RF and XGBoost
   swap mean ranks (4.95 vs 4.97). Medic3: all 10 / 15 pairs still significant,
   order unchanged. Predictions (2160 + 30 + 30 files) copied to Kremen, the
   laptop and the USB drive (2026-10-15) and verified on Kremen (every fit has
   its file, ROC-AUC from the files matches; see each PROVENANCE.md).
   **Thesis updated 2026-10-15** (thesis commits `76bf380`, `2ef9c62`): Results
   chapter on the combined sets as one result set — **the rerun is not
   mentioned in the thesis (user's decision)**, provenance only in LaTeX
   comments; new section "Podatkovni nabor Medic3"; tree-ensemble Methods
   sections cut to ~half a page each (supervisor: ~50 pages, known methods
   brief); RF / XGBoost / LightGBM preprocessing text corrected, Bouckaert & Frank 2004 and Nadeau & Bengio 2003 added to
   `literatura.bib` (Crossref). Still open in the thesis: the Discussion
   (LightGBM mechanism, only TabICL imputes on Medic3), Materiali (CC18 and
   Medic3 descriptions), Protokol / Predobdelava / Okolje / Ponovljivost.
7. Then: tuning one booster by grid search with nested CV, then the CC18
   leakage test (design approved by the supervisor 2026-10-20, step 8); CC18 with
   injected NULLs is still on the supervisor's list. **Tuning decisions (user,
   2026-10-14):**
   - **XGBoost**, with the same out-of-the-box preprocessing as the default
     XGBoost after the 2026-10-14 correction (native categories, see
     Preprocessing policy → Audit), so the rerun's default XGBoost rows are the
     default arm and only the hyperparameters differ.
   - **Extra arm, native categories (user, 2026-10-14) — superseded the same
     day** by the out-of-the-box decision (Preprocessing policy → Audit): the
     default XGBoost itself now takes native categories, so the tuned arm does
     too, and ordinal vs native at defaults is simply `cc18_v3` vs the rerun.
     Measured at defaults, 15 folds: credit-g native 0.761 vs ordinal 0.772
     (3/15 folds better), cmc 0.700 vs 0.705 (2/15).
   - **Nested CV:** outer = the 15 folds of `cc18_v3`; inner = 3-fold
     stratified CV on the outer training part only; best mean inner ROC-AUC
     wins, is refit on the whole training part, scored once on the test part.
   - **Grid** (Probst, Boulesteix & Bischl 2019, JMLR 20(53):1–32, Table 3,
     verified 2026-10-13): `learning_rate` {0.3, 0.1, 0.03, 0.01} ×
     `n_estimators` {100, 200, 500, 1000} × `min_child_weight` {1, 4} ×
     `colsample_bylevel` {1, 0.5} × `subsample` {1, 0.8} = **128 configs**
     (0.01 added 2026-10-15 after the pilot, see below), defaults included;
     `max_depth` not tuned (lowest tunability). One fit to 1000 rounds per
     config; the tree counts are read with `iteration_range` (verified
     2026-10-13 identical to separate fits, also with subsampling). Non-default
     values lie inside Probst's tuning ranges except 200/500 trees (their
     range ~920–4550); in the pilot, check how often winners sit at 1000 trees.
   - **Datasets:** one pinned list of 68 (CC18 minus Devnagari-Script,
     CIFAR_10, Fashion-MNIST, mnist_784, excluded on cost). Run the 36
     non-saturated first, the other 32 later under the same `RUN_ID` and the
     same commit (finished datasets are skipped). Upper bound 26 h / 124 h of
     8-core compute; pilot on Kremen first.
   - Worked example (cmc, fold 0): the default config reproduces `cc18_v3`
     exactly; winner lr 0.03 / 100 trees / `colsample_bylevel` 0.5 /
     `subsample` 0.8, test 0.734 vs default 0.701. Explainer page for the user:
     https://claude.ai/artifact/GF5XYzKoUrpT6srp7AmTND.
   - **Implemented 2026-10-15** (`9e94dbc`): `src/models/xgboost_tuned.py`
     (registry name `xgboost_tuned`), grid in `config.yaml` (`tuning`),
     `scripts/cc18_tuning_ids.json` (68 sets: indices 0-35 = the 36
     non-saturated, 36-67 = the 32 saturated; rebuilt by
     `scripts/gen_tuning_ids.py` from `cc18_v4/combined`), `scripts/run_tuning.sh`
     (CPU only, no partition = cluster default; fallback `--partition=gpu
     --gres=gpu:1 --constraint=h100`). Ties go to the earlier grid point
     (defaults first, fewer trees first). Every outer fold also fits the
     default XGBoost in the same process: `default_test_roc_auc` in the
     `hyperparameters` JSON must equal `cc18_v4` (check) and
     `default_train_time_s` is the same-node time baseline (cost ratio).
     Unit check (Kremen, cmc fold 0): default equals `cc18_v4` exactly, winner
     lr 0.03 / 100 trees / `colsample_bylevel` 0.5 / `subsample` 0.8, test 0.732
     vs default 0.695, search 55 s (~790 × one default fit).
   - **Pilot `subset_xgb_tuned_local`** (Kremen, credit-g / diabetes / sick,
     first with the 96-config grid, from `9e94dbc`): 45 folds, 0 errors;
     default inside the module = `cc18_v4` to 1.1e-16 on every fold; tuned
     better than default on credit-g (+0.023, 13/15 folds), diabetes (+0.040,
     15/15), sick (+0.0004, 8/15); tuning time 425–520 × one default fit on the
     same machine; winners at 1000 trees 1/45 (upper edge fine) but **32/45 in
     the low corner** (lr 0.03, 100 trees) — on diabetes the inner score fell
     with every step of more learning, so the optimum lay below the grid.
     Probe on inner scores only (no test folds), credit-g/diabetes/cmc folds
     0–4: with lr 0.01 the best tree count is 200–500 in 12/15 folds (inside
     the grid), gain over the lr-0.03 best −0.003…+0.006. Hence lr 0.01 (inside
     Probst's range 0.002–0.355, brackets their optimal default 0.018 with
     0.03). **Final pilot** with the 128-config grid, from `f7d8366` (the
     Arnes commit), same name (the 96-config one was discarded uncommitted):
     45 folds, 0 errors, default = `cc18_v4` to 1.1e-16; tuned − default:
     credit-g +0.0234 (13/15 folds better), diabetes +0.0414 (15/15), sick
     +0.0005 (8/15); winners by (lr, trees): low corner (0.01, 100) 6/45, at
     1000 trees 4/45, lr 0.01 wins 29/45 — the grid now brackets the optimum;
     tuning time 626–713 × one default fit.
   - **Arnes run `cc18_xgb_tuned` — first 36 submitted 2026-10-15** from
     `hpc-login4` at `f7d8366`, job `20494570` (`--array=0-35%8`); with no
     partition given it ran on the default CPU partition `all` (nodes `wn…`),
     no GPU reserved. Commands (from the repo root, after `git pull`):
         RUN_ID=cc18_xgb_tuned sbatch --array=0-35%8 scripts/run_tuning.sh
     Estimate (default XGBoost time in `cc18_v4` × ratio 450–800 × 4/3 for lr
     0.01): ~28–49 h of 8-core compute, longest task Bioresponse ≤ ~5 h, ~4–7 h
     wall with 8 at a time. The other 32 later, same `RUN_ID` (isolet ≤ ~62 h,
     har ≤ ~21 h):
         RUN_ID=cc18_xgb_tuned sbatch --array=36-48,50-53,55-67%8 scripts/run_tuning.sh
         RUN_ID=cc18_xgb_tuned sbatch --array=49,54 --partition=long --time=4-00:00:00 scripts/run_tuning.sh
     Rows record the commit of each submission; the **training path** must not
     change between the two halves. Check with `git diff f7d8366 HEAD --
     src/config.py src/data.py src/runner.py src/run_one_dataset.py src/utils.py
     src/models config.yaml scripts/run_tuning.sh` (must be empty). Narrowed on
     2026-10-15: the secondary-metric work changed `src/summary.py` and
     `src/stats.py` and added `src/metrics.py` and `scripts/compute_metrics.py`,
     none of which the training path imports (checked: models import only
     `src.utils`/`src.config`, the runner `config`, `data`, `models`, `utils`), so
     a plain `git diff -- src scripts` is no longer empty but training is
     identical. **Do not `git pull` on Arnes while an array of this run is still
     queued or running** — tasks that have not started yet run whatever code is in
     the directory when they start. After each half: merge,
     `sacct` → `sacct.txt`, `git pull`, commit `results/runs/cc18_xgb_tuned`
     (incl. `tuning/`; `predictions/` stays out), push; copy `predictions/` to
     Kremen, laptop, USB.
   - **Part 1 checked 2026-10-15 on Kremen** (receipts and every number in
     `results/runs/cc18_xgb_tuned/PROVENANCE.md`): job `20494570`, 36 tasks
     COMPLETED on 13 CPU nodes `wn…`, 540 rows, 0 errors, all from `f7d8366`,
     clean tree, committed from Arnes as `785257b`; 4 h 50 min wall, 26.9 h of
     8-core compute (longest madelon 4 h 04 min), MaxRSS ≤ 0.97 GiB. Checks:
     in-run default = `cc18_v4` on 415/540 folds, the rest within 1.1e-16
     (rounding, same prediction order); `tuning/` tables match `results.csv`
     on 540/540 (read with `float_precision="round_trip"` — pandas' default
     parser can be 1 ulp off); Arnes = the Kremen pilot on credit-g and
     diabetes (same winner and ROC-AUC, 30/30). On the 36: tuned beats default
     on 33 (mean +0.0141, Wilcoxon p 1.9e-8), ties default CatBoost (15:21,
     p_Holm 0.67), loses to TabPFN-3.5 on 36/36 and TabICL on 33/36, closes
     39 % of the mean gap to TabPFN-3.5. The low corner (0.01, 100) won
     73/540 folds, almost only on small noisy sets (kc2, analcatdata_dmft,
     dresses-sales, cmc, blood-transfusion, numerai28.6), by a median inner
     margin of 0.0016, while TabPFN-3.5 leads there by 0.009–0.030. **The grid
     stays unchanged for part 2**: it was fixed before the run, and changing it
     now would choose the grid after seeing results; the thesis names it as a
     limitation. Part 2 estimate (`cc18_v4` default time × measured ratio
     340–1194): isolet 20–69 h, har 7–24 h, the other 30 together 14–50 h
     (letter ≤ 6.3 h). **Part 2 submitted 2026-10-15** without `git pull`
     (Arnes at `785257b`, training path = `f7d8366`): job `20511472`
     (`--array=36-48,50-53,55-67%8`, default partition `all`, 12 h). For
     49/54 sbatch rejected `--time=3-00:00:00` (`all` allows 2 days, see Arnes
     HPC), so they went to the `long` partition (7 days) with the command above
     (`--partition=long --time=4-00:00:00`): job `20511665`. Had `long` refused,
     the fallback was `all` with `--time=2-00:00:00` and a resubmission of
     `--array=49` after a TIMEOUT (it resumes at the last finished fold). The
     `3-00:00:00` in the
     `scripts/run_tuning.sh` header gets corrected after part 2 (left alone so
     the training-path diff stays empty while the run is in progress).
     **State 2026-10-19 (Monday, ~11:05):** job `20511472` COMPLETED, all 30
     tasks (longest index 37, 5 h 00 min); har (`20511665_54`) COMPLETED in
     31 h 09 min on `wn119` — over its 7–24 h estimate. **isolet
     (`20511665_49`) cannot finish within its 4 days:** at 84.8 h it had 10/15
     folds in `per_dataset/300__xgboost_tuned.csv.partial` (fold times 5.8–9.4 h;
     repeat 0 mean 6.4 h, repeat 1 mean 8.5 h; fold 9 ended 2026-10-18 23:37,
     fold 10 then already ran > 10 h), ~37–43 h still needed, TIMEOUT expected
     ~2026-10-19 21:05. The user submitted the continuation from
     `hpc-login3`, Arnes still at `785257b`, no `git pull`:
         RUN_ID=cc18_xgb_tuned sbatch --array=49 --partition=long --time=4-00:00:00 \
             --dependency=afternotok:20511665_49 scripts/run_tuning.sh
     → job **`20652997`** (`20652997_49`), starts only if `20511665_49` ends not
     OK (TIMEOUT) and resumes from the partial; only the fold in progress at the
     kill is lost. If `20511665_49` were to finish (e.g. time limit extended by
     Arnes support), `20652997` stays `DependencyNeverSatisfied` → `scancel` it.
     Expected end with 4–5 folds left: ~2026-10-21. Both isolet and har ran far
     over the part-1-based estimate on `wn119`; quantify with the in-run
     `default_train_time_s` in the analysis (same-node ratio) and note it in
     PROVENANCE part 2 together with the TIMEOUT and resubmission.
     **2026-10-20:** the user reports `20652997_49` started (so `20511665_49`
     ended not OK — confirm TIMEOUT with `sacct` for `sacct.txt`); its log
     should contain `najden partial: 10` (or 11) `učenj že narejenih`. Expected
     end ~2026-10-21/22. The user now works on **Kremen** again; after the
     run they copy the predictions to Kremen and the laptop.
     **Part 2 done and checked 2026-10-21 on Kremen** (receipts and every
     number in `results/runs/cc18_xgb_tuned/PROVENANCE.md`, part 2):
     `20511472` 30/30 COMPLETED, `20511665_54` COMPLETED, `20511665_49`
     TIMEOUT with 11/15 folds, `20652997_49` COMPLETED (resumed at 11, ~11.5 h
     of fold 11 lost); merged on Arnes (1020 rows, 0 errors), committed from
     Arnes as `0ad1a88`; predictions (1020, 32 MB) on Kremen, laptop, USB;
     `compute_metrics.py` done. All checks of part 1 hold on all 68. Results:
     tuned beats default on 61/68 (+0.0083; non-saturated +0.0141, saturated
     +0.0018), ties default CatBoost (30:38, p_Holm 0.52), loses to TabPFN-3.5
     on 66/68 and TabICL on 62/68, closes 38 % of the gap to TabPFN-3.5; log
     loss better on 65/68; worse than default notably only on ilpd (−0.0088,
     selection noise: inner gain > 0 on 15/15 folds, test gain < 0 on 11).
     Saturated sets prefer more learning (1000 trees in 253/480 folds, plateau
     there), non-saturated the least (low corner 73/540). Probst et al. 2019
     Table 2: XGBoost tunability vs package defaults 0.043 ± 0.006 (oracle, an
     upper bound) — read from the PDF 2026-10-21. isolet/har ran over the
     estimate because `wn116`/`wn119` ran the same 8-core default fit ~2.6×
     slower than `gwn09` (the ratio itself was normal). Checklist items 1–3
     done; item 4 done except the thesis tuning section.
   - **After part 2 — checklist (in this order):**
     1. On Arnes (the user has the commands): `sacct` says COMPLETED for
        `20511472`, `20511665_54` and `20652997_49` (`20511665_49` = TIMEOUT,
        expected); `merge_results.py cc18_xgb_tuned` → 1020 rows, no `.partial`;
        `sacct -j 20494570,20511472,20511665,20652997 ... > sacct.txt`; only then
        `git pull`, commit `results/runs/cc18_xgb_tuned`, push.
     2. On the laptop/Kremen: `git pull`; copy the predictions from Arnes (run
        on the receiving machine, repo root):
        `rsync -av tz43558@hpc-login.arnes.si:Diplomsko-delo_Koda/results/runs/cc18_xgb_tuned/predictions/ results/runs/cc18_xgb_tuned/predictions/`
        (1020 files); then onto the USB drive, later to the other machine.
     3. `python scripts/compute_metrics.py cc18_xgb_tuned` (env `tabular3.5`):
        accuracy and log loss of the tuned XGBoost, with the ROC-AUC check
        from the files (the secondary-metrics session asked for this).
     4. Analysis of all 68 as for part 1 (default = `cc18_v4`, tuned vs
        default, vs `cc18_v4/combined`, winners and grid edges, time), the
        part-2 section of `PROVENANCE.md`, the registry rows here and in
        `results/README.md`, and the `run_tuning.sh` header (2 days on `all`,
        isolet/har on `long`). Then the thesis tuning section.
   For the supervisor: leakage design, injected NULLs, the LightGBM finding,
   and the out-of-the-box correction of XGBoost and RandomForest (2026-10-14).
8. **Leakage test — state 2026-10-21: design APPROVED by the supervisor,
   parameters FIXED with verified sources, code implemented, checked on
   Kremen and pushed; next: the user runs it on Arnes.** The pre-registration is
   `results/runs/cc18_leakage/DESIGN.md` (Slovenian; every decision with its
   source and page) — read it before touching the test.
   - **Supervisor's answer (email, 2026-10-20):** "Test na 36 množicah kar
     naredite, tako kot ste si zamislili je čisto OK, ne glede na izid bova sum
     na uhajanje navedla kot omejitev." → run the test as designed below; the
     thesis states the suspicion of leakage as a **limitation whatever the
     outcome** (Discussion, `ch:zakljucek`; noted in its outline comment).
   - **Purpose (user, after a call with the supervisor):** the supervisor was
     surprised how well TabPFN does on CC18 and wants to know whether TabPFN
     was trained on CC18 data — i.e. contamination / memorisation of the
     foundation model, **not** a leak in our pipeline. A label-shuffle check of
     our pipeline answers a different question; at most a small add-on.
   - **TabPFN-3.5 report exists:** arXiv 2609.17895, "TabPFN-3.5: Technical
     Report" (Jäger, Erickson, Grinsztajn, … 47 authors; v1 2026-09-15, v2
     2026-09-22). Whole text read via WebFetch 2026-10-19: §3.3 "Our synthetic
     data generation pipeline closely follows TabPFN-3"; no mention of
     training, fine-tuning or validating on real data, OpenML (only C.7
     ScoringBench as an evaluation benchmark), CC18, contamination or overlap;
     §C.5 (TALENT, borrowing the TabICLv2 protocol and baselines) uses "the 300
     base datasets minus the 26 that were used during the development of
     TabPFN-2 and TabICLv2". The thesis `\todo` "Poišči preverljiv vir za
     TabPFN-3.5" and the sentence "tehničnega poročila v času pisanja nismo
     imeli na voljo" can now be fixed (bib entry from the arXiv record; quote
     only what the report says).
   - **What TabPFN-3.5 does itself** (`inference_config` in the header of
     `tabpfn-v3.5-20260909.safetensors`, read without loading the model):
     `N_ESTIMATORS` 8, `PREPROCESS_TRANSFORMS` one config `name: "none"`
     (numbers enter untransformed), `FEATURE_SHIFT_METHOD` and
     `CLASS_SHIFT_METHOD` `"shuffle"`, categoricals `"ordinal_shuffled"`.
     So permuting columns, classes or category codes tests nothing — the
     perturbation must change the values. The architecture config has
     `cell_ecdf_num_buckets` 8192 (probably a rank/ECDF cell embedding;
     unverified), so an order-preserving transform may leave part of the input
     unchanged → also an arm that changes ranks (noise).
   - **Proposed design (in the email to the supervisor, drafted 2026-10-19):**
     the 36 non-saturated sets of the tuning run (= the 37 minus CIFAR_10);
     TabPFN-3.5, TabICL, CatBoost (control: trees react to strictly monotone
     transforms only through split points between values); arms: original,
     (a) each numeric attribute through a strictly increasing function, (b) (a)
     plus slight noise (strength to fix); the same arms on `medic3_160` (private,
     all numeric — separates memorisation from TabPFN's general sensitivity to
     the transform); 5 folds of repeat 0, original arm rerun in the same job;
     rule fixed before running: per dataset drop(TabPFN) − drop(CatBoost),
     Wilcoxon over datasets. Cost per repeat on all 72 (from `cc18_v4/combined`):
     TabPFN 0.8 GPU-h, TabICL 1.3 GPU-h, CatBoost 15.9 CPU-h. Limitation: it
     cannot detect indirect influence (developers selecting the model on public
     benchmarks) → named as a limitation. Optional extras not in the email:
     overlap analysis if the list of the 26 is published; label-shuffle check.
   - **Asked the supervisor** whether the test is needed at all, or whether the
     report + Medic3 (TabPFN-3.5 first there too, 0.9614 vs CatBoost 0.9599,
     a dataset it cannot have seen) + a stated limitation suffice; and if
     needed, whether the design is fine. **Answer: run it as designed** (top of
     this step). Runs on Arnes/Kremen only (laptop: no GPU), after the tuning run.
   - **Checked 2026-10-20 on Kremen** (library source and synthetic data only;
     no CC18 result of any arm was looked at, so the design is still fixed blind):
     - **TabPFN-3.5 cell input** (`tabpfn/architectures/tabpfn_v3_5.py`,
       `_preprocess_raw` and `FourierPlusMetadataFeatureGroupEmbedder`): every
       cell enters as (1) its value standardised on the train rows (clipped at
       ±100), Fourier-embedded; (2) a NaN/inf indicator; (3) its midrank ECDF
       against the train rows (exact up to 8192 distinct values per column,
       interpolated above), sin/cos-embedded. A strictly increasing transform
       changes only (1); (2) and (3) stay identical. This verifies the ECDF
       guess of 2026-10-19: the noise arm is the one that reaches the rank channel.
     - **TabICL** (tabicl 2.2.0 defaults): 8 members, `norm_methods` None →
       `["none", "power"]`, i.e. 4 members get standard scaling only and 4 also
       Yeo-Johnson (which partly undoes a monotone distortion); outliers clipped
       at 4 sd.
     - **CatBoost** (synthetic, 2400 train / 600 test rows, 10 features, a
       signed cube): the training bins are the same row sets (borders compared
       as ranks of the train values), ROC-AUC identical to 6 decimals,
       predictions differ by ≤ 1.3e-3 (test values between two train values next
       to a border). Not exactly invariant when the original values tie in
       float32 (CatBoost's input type) and the transformed ones do not: with a +7
       offset and 0.01 scale the same check gave max |Δproba| 0.097, ΔAUC 5e-5.
       So "trees hardly react" holds; "they use only the order" is approximate.
     - **Two of the 36 sets have no numeric attribute** (analcatdata_dmft 469,
       connect-4 40668): both arms leave them unchanged, so they carry no
       information → exclude them, **34 remain** (tell the supervisor/thesis why).
       Few numeric attributes: dresses-sales 1 of 12, Internet-Advertisements 3
       of 1558, cmc 2 of 9. Largest standardised value |z| = 119
       (bank-marketing): exp(z) would overflow float32 (above 88.7), a power
       does not.
     - **Cost** (`cc18_v4/combined`, repeat-0 times on Arnes): 34 sets × 3 arms ×
       5 folds = CatBoost 0.73 CPU-h, TabPFN-3.5 0.36 GPU-h, TabICL 0.31 GPU-h;
       largest task numerai28.6 ~0.2 h. Medic3-160 × 3 arms × 5 folds: CatBoost
       ~23 h (1.54 h per fold), TabPFN ~0.3 h, TabICL ~0.5 h but ~92 GiB RAM →
       Arnes (Kremen has 31 GB).
   - **Proposal of 2026-10-20 (superseded in one point):** arm (a) was the
     cube `((x − mean)/sd)³`. The user (2026-10-21) accepted the proposals "as
     long as you can cite the exact sources"; no source used a cube, so arm (a)
     became the **signed cube root** of the standardised value, the published
     warp of Biloš et al. 2026 (below). Everything else stayed as proposed.
   - **Final design (user's condition met 2026-10-21; details, quotes and pages
     in `DESIGN.md`):** 34 sets (the 36 non-saturated minus analcatdata_dmft
     and connect-4) + Medic3-160 as reference; CatBoost (control), TabPFN-3.5
     (primary), TabICL (secondary); arms `original`, `monotone` = cbrt(z),
     `monotone_noise` = cbrt(z + ε), z = (x − mean)/sd over all rows (ddof 0,
     NaN kept), ε ~ N(0, 0.1²) once per dataset, seed 42; folds 0–4, model
     seed 42, all arms of a dataset in one task; per dataset Δ = drop(model) −
     drop(CatBoost) on the 5-fold mean ROC-AUC; one-sided Wilcoxon (H1: Δ > 0,
     `zsplit`), Holm within each model over the two arms, α 0.05; a significant
     CC18 result counts as memorisation-consistent only if Medic3's Δ is
     **significantly** below the CC18 median Δ (one-sided corrected repeated-CV
     t-test on Medic3's 5 fold Δs, the thesis's Bouckaert & Frank 2004 /
     Nadeau & Bengio 2003 test; changed 2026-10-21 before any real run — the
     first rule, "Δ below the median" without a test, gave the wrong verdict on
     synthetic data with the same planted drop on both sides, a coin flip under
     fold noise). **Verified sources** (PDFs read with pdftotext, pages
     are printed page numbers): Bordt et al. 2024 (COLM, arXiv 2404.06209v3:
     p. 2, §4.1 p. 6, §4.2 p. 7, App. E p. 23, Tables 9–10 p. 30 — the test
     principle, ~1 % perturbation, novel datasets as reference, LR/GBT check);
     TabArena App. E.6 pp. 55–56 (no contamination test exists for TFMs, calls
     for tools "akin to the work by Bordt et al."); Galili & Meilijson 2016
     (arXiv 1611.04561, p. 2, quotes CART p. 57; exception near split points,
     p. 1); Biloš et al. 2026 (arXiv 2605.21288v1, preprint: Table 7 p. 16
     signed cube-root warp exponent 1/3 fixed before evaluation and noise in
     feature-std units; §H.1 p. 42; Table 3 p. 9 TabPFNv2 −1.1 pp n.s.,
     TabICLv2 −0.3 pp; §H.5 p. 44 XGBoost 0.0 under the warp; §H.6 p. 44
     pre-specified one-sided Wilcoxon on per-dataset differences, Holm within
     each model); TabPFN-3.5 report (arXiv 2609.17895v2: §3.1 p. 9 value +
     ECDF channels, "midranks are unchanged by any strictly increasing
     transformation"; Fig. 8 p. 10, K = 4 ECDF harmonics; §3.2 p. 11 no
     quantile transforms; §C.5 p. 29 "the 26 that were used during the
     development"); Qu et al. 2025 p. 7 (TabICL z-normalisation with or without
     power transform); Demšar 2006 §3.1.3 pp. 7–8, pp. 12–13. **Not from a
     source (own derivation, stated as such):** centring before the root, the
     noise level 0.1 sd (variance 1 %; E|ΔECDF| ≈ 0.0225 for a normal attribute,
     measured 0.0224–0.0228 on climate/numerai, up to 0.20 where noise breaks
     ties: Bioresponse, spambase), the Medic3 rule (its test is sourced, the
     rule itself is ours). **Caution:** the
     TabPFN-3.5 report's "without using LLMs, real data, internet search" is
     about the Thinking mode (§3.4), not pretraining — never cite it for
     pretraining.
   - **Implemented 2026-10-21** (committed together with `DESIGN.md`, before
     any run on the perturbed data): spec keys
     `perturbation`, `noise_sd`, `perturbation_seed`, `repeats` in
     `src/data.py` (`_perturb`; folds are cut from the `n_repeats` 3 splits, so
     folds 0–4 equal the thesis folds; dataset name gets the suffix `_<arm>` so
     arms never merge in `src.summary`); `spec_id` suffix `_<arm>`; runner logs
     the arm and writes a new last column `perturbation`; `config.yaml`
     section `leakage` + sets `cc18_leakage` (102 specs), `medic3_160_leakage`
     (3), `leakage_smoke` (sick × 3); `scripts/gen_leakage_ids.py [--check]`,
     `scripts/run_leakage.sh` (GPU, one task = one dataset's arms in a row),
     `scripts/run_leakage_cpu.sh` (Medic3 CatBoost, one arm per task),
     `scripts/analyze_leakage.py` (the pre-registered analysis; checks: rows,
     errors, CatBoost's monotone drop, original arm vs the thesis run).
   - **Checked 2026-10-21 on Kremen:** `gen_leakage_ids.py --check` (data only,
     no model): all 34 sets + Medic3-160 pass — folds and labels equal the
     thesis folds 0–4 in every arm, categorical columns and NaN pattern
     unchanged, monotone arm keeps order and ties; float32 rounding changes the
     number of distinct values by ≤ ~1 % in a few columns of 4 sets
     (mfeat-fourier, mfeat-zernike, Bioresponse, numerai28.6). Dry runs of both
     SLURM scripts (task i → specs 3i..3i+2; bad `--array` and missing RUN_ID
     rejected). Smoke test `leakage_smoke_local` (sick, 45 fits, 0 errors):
     found and fixed an all-missing `object` column (TBG) that the dtype guard
     rejected — all-missing columns are now left unchanged; the 34 sets were
     unaffected (check report byte-identical before/after). CatBoost's monotone
     drop on sick exactly 0; original arm = `cc18_v4/combined` (CatBoost 5/5
     identical, TabPFN ≤ 3.1e-5, TabICL ≤ 1.5e-5). Implementation committed as
     **`6444529`**; the smoke run was then **rerun from that commit** (clean
     tree, `git_dirty` false) and is committed as `leakage_smoke_local`: 45/45
     fits identical (Δ 0.0) to the first, uncommitted run. **Synthetic test of
     the analysis** (scratch only): planted extra TabPFN drop 0.01 found (p
     0.002, Holm 0.004, 9/9), unplanted arms/models not rejected, a dataset
     with a failed fit excluded; it exposed the weak Medic3 rule (above) — the
     revised rule says "not distinguishable" when the reference has the same
     drop (p 0.43) and "memorisation-consistent" when it has none (p 0.016).
   - **On Arnes** (after commit + push; `git pull` is safe, nothing queued):
         RUN_ID=cc18_leakage sbatch --array=0-33%4 scripts/run_leakage.sh
         DATASET_SET=medic3_160_leakage ALGORITHMS="tabpfn tabicl" RUN_ID=medic3_160_leakage sbatch --array=0 --mem=160G scripts/run_leakage.sh
         DATASET_SET=medic3_160_leakage RUN_ID=medic3_160_leakage sbatch --array=0-2 scripts/run_leakage_cpu.sh
     Expect 1530 and 45 rows. Then `merge_results.py` for both, `sacct` into a
     PROVENANCE.md per run, commit from Arnes; copy predictions; then
     `python scripts/analyze_leakage.py cc18_leakage --reference medic3_160_leakage`
     and `compute_metrics.py` on both. Do NOT change any parameter after
     seeing results; report whatever comes out (the limitation is stated
     regardless). The tuning-run training-path check is `git diff f7d8366
     0ad1a88 -- <training path>` (the part-2 commit), no longer against HEAD.

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
- **Tuning — sources (verified 2026-10-14 from the full texts).** Probst,
  Boulesteix & Bischl 2019, JMLR 20(53):1–32 (Tables 1–3: XGBoost tunability
  per HP, tuning ranges q0.05–q0.95; 38 binary OpenML100 sets; package default
  nrounds 500). Nested CV: Cawley & Talbot 2010, JMLR 11(70):2079–2107 —
  p. 2080 "model selection must be treated as an integral part of the model
  fitting process and performed afresh every time a model is fitted to a new
  sample of data", compare "learning algorithm/model selection procedure
  combinations", "nested cross-validation or 'double cross' (Stone, 1974)";
  pp. 2102–2103 tuning outside the CV ("external protocol") is optimistically
  biased on all 13 benchmarks, "should not be used in practice"; p. 2103
  model selection "conducted independently in each trial". Varma & Simon 2006,
  BMC Bioinformatics 7:91, doi:10.1186/1471-2105-7-91 (abstract: the CV
  estimate of a CV-tuned classifier is substantially biased — on null data
  error < 30 % for 18.5 % / 38 % of sets while true error is chance; nested CV
  "provides an almost unbiased estimate"). Bischl et al. 2023, WIREs Data
  Mining Knowl. Discov. 13(2):e1484, doi:10.1002/widm.1484 (Crossref; quotes
  from arXiv 2107.05847v3): §4.4 pp. 22–24 reporting the tuned score is
  "optimistically biased"; nested resampling; per-fold winners are "temporary
  objects", for a final model tune once more on the complete data; §4.2.1
  p. 9 random search often beats grid search (Bergstra & Bengio 2012, not yet
  verified) — grid search is the supervisor's choice; say so as a limitation.

## Conventions
- Each model module fails soft: exceptions go to `result["error"]`, never
  raised, so one failing (dataset, algorithm, fold) doesn't crash a run.
- All algorithms use **default hyperparameters** — intentional per the thesis
  protocol, not an oversight to "fix". The only non-defaults are the seed
  (`random_state`) and compute/logging settings:
  `RandomForestClassifier(n_jobs=-1)`, `LGBMClassifier(verbosity=-1)`,
  `CatBoostClassifier(verbose=False, allow_writing_files=False)`; TabPFN's
  explicit `model_path` is the file the default `'auto'` picks. Verified by
  `scripts/audit_defaults.py` (2026-10-14: all six pass).
- **RandomForest with `n_jobs=-1` is not bit-reproducible at prediction
  time** (found 2026-10-08, correcting an earlier "bit-identical, verified"
  note): parallel *fits* give bit-identical trees (= serial), but parallel
  `predict_proba` sums the tree probabilities in thread order, so probabilities
  differ at rounding level and ROC-AUC by up to ~2e-5 between identical runs
  (cmc: 5 runs, 3–4 distinct AUCs, in sklearn 1.9.0 and 1.9.1). This, not the
  library upgrade, explains RF's 31/1080 changed fits between `cc18_v2` and
  `cc18_v3`. Left as is (negligible, documented in the thesis, `sec:rf`); a
  fully deterministic variant would predict with `n_jobs=1` — a protocol
  change, ask the user first.
- Never merge into or overwrite an existing run's `results.csv` from another
  run; every run has its own directory.
- **Fail-soft catches exceptions only.** A model that trains "successfully" but
  is broken (LightGBM on Medic3) leaves ordinary rows with no `error`. After
  every run, check per-fold ROC-AUC: anything near 0.5 on a dataset where the
  others score high is a red flag to investigate before writing.
- **XGBoost is deterministic on one machine and across Arnes nodes**
  (`cc18_v2` vs `cc18_v3`: Δ 0 on all 72 sets, PCIe and SXM nodes), **but not
  between Kremen and Arnes on large data**: Medic3 fold 0 differs by 6.2e-4
  (raw) / 4.6e-4 (160) although the old and the new module give identical
  predictions on Kremen (2026-10-14); small CC18 sets showed Δ 0 Kremen vs
  Arnes. Hardware floating point, not the code: compare XGBoost numbers within
  one machine pool, and treat Kremen pilots as approximate.
- **LightGBM is not bit-reproducible by default** (`deterministic=false`):
  invisible in stable training (CC18: Δ 0 across runs and machines), visible
  once training diverges (Medic3). Changing it would be a protocol change; ask
  the user first.
