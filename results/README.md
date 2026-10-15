# Rezultati

Vsak zagon živi v svoji mapi pod `runs/`. Hranimo samo zagone, ki jih diploma
trenutno uporablja; nadomeščene zbrišemo v ločenem commitu, ostanejo pa v git
zgodovini. Nič v tej mapi se ne ureja na roko, vse nastane iz zagona.

## Kaj je zdaj tukaj

| `run_id` | Kaj | Okolje |
|---|---|---|
| `cc18_v3` | celoten CC18, 6480 vrstic, zamrznitev 2026-10-05 (TabPFN-3.5); rezultat diplome za LightGBM, CatBoost, TabPFN, TabICL (RandomForest in XGBoost nadomesti `cc18_v4`) | Arnes `~/envs/tabular3.5`, H100 |
| `medic3_raw` | zaupni Medic3, vseh 220 razredov, 90 vrstic (TabPFN-3.5: 15 × meja 160 razredov); RandomForest nadomesti `medic3_raw_v2` | Arnes `~/envs/tabular3.5`, H100 |
| `medic3_160` | Medic3, 160 najpogostejših razredov, 90 vrstic, vseh šest algoritmov; RandomForest nadomesti `medic3_160_v2` | Arnes `~/envs/tabular3.5`, H100 |
| `subset_v3_local` | validacija zamrznitve 2026-10-05 na trojici credit-g, diabetes, sick | Kremen `tabular3.5`, RTX 3060 |
| `subset_v3` | ista validacija na gruči | Arnes `~/envs/tabular3.5`, H100 |
| `subset_v4_local` | RandomForest in XGBoost po popravku "kot iz škatle" (2026-10-14) na trojici credit-g, diabetes, sick; `diagnostics/audit_defaults.txt` = izpis revizije vseh šestih algoritmov | Kremen `tabular3.5`, RTX 3060 |

| `cc18_v4` | RandomForest + XGBoost "kot iz škatle" na vseh 72 naborih, 2160 vrstic (2026-10-14/15); **`combined/` = rezultat diplome** (ostali štirje iz `cc18_v3`) | Arnes `~/envs/tabular3.5`, H100 (le CPU) |
| `medic3_raw_v2` | RandomForest + XGBoost (preverba) na Medic3, 30 vrstic; `combined/` = rezultat za diplomo | Arnes `~/envs/tabular3.5`, H100 (le CPU) |
| `medic3_160_v2` | isto za 160 razredov; `combined/` = rezultat za diplomo | Arnes `~/envs/tabular3.5`, H100 (le CPU) |

`combined/` sestavi `scripts/combine_runs.py <nov> --base <star>`: vse vrstice
novega zagona + vrstice starega za algoritme, ki jih novi nima (LightGBM,
CatBoost, TabPFN, TabICL, katerih koda se ni spremenila); `sources.json` pove vir.

`cc18_v2` (zamrznitev 2026-09-09, TabPFN-3) je bil po `cc18_v3` zbrisan
2026-10-08; v gitu je zadnjič v commitu `5a14045`, npr.
`git show 5a14045:results/runs/cc18_v2/results.csv`.

Starejši zagoni (`check_refactor_oldenv`, `subset_v2_local`, `subset_v2`) in arhiv
`arnes/cc18/` iz avgusta 2026 so bili zbrisani 2026-10-05. Zadnjič so v commitu
`b1b9120`; vrneš jih npr. z `git checkout b1b9120 -- results/arnes` ali pogledaš z
`git show b1b9120:results/runs/subset_v2/results.csv`.

## `runs/<run_id>/` — vsak zagon svoja mapa

| Datoteka | Kaj je | V gitu? |
|---|---|---|
| `manifest.json` | git commit, stroj, GPU, Python, torch/CUDA, celoten `config.yaml`, `pip freeze` in od 2026-10-05 tudi `model_checkpoints` (datoteka z utežmi vsakega temeljnega modela, ki je `pip freeze` ne pove) | da |
| `per_dataset/<id>.csv` | dokončan nabor, ena vrstica na (algoritem, fold); pri Medic3, ki teče po algoritmih, `<id>__<algoritem>.csv` | da |
| `per_dataset/<id>.csv.partial` | nedokončan nabor (kontrolna točka); ob ponovnem zagonu z istim `run_id` se nadaljuje | ne |
| `predictions/<id>/<algoritem>_r<rep>_f<fold>.npz` | `predict_proba`, `y_test`, `test_idx`, `classes`, `proba_classes` vsakega učenja — za poljubno metriko brez ponovnega zagona; po zagonu na Arnesu jih prekopiraj domov (`rsync`, glej `razlaga_repozitorija/razlaga.md`) | ne (velike) |
| `results.csv` | združeni `per_dataset/*.csv` (`scripts/merge_results.py` ali samodejno lokalno) | da |
| `summary/` | tabele (CSV + LaTeX), diagram kritične razdalje, `friedman.csv` (chi2_F in Iman-Davenportova F_F), `razlicice.tex`; pri zagonu z enim naborom namesto testov čez nabore `corrected_ttest_holm.csv/.tex` — iz `src/summary.py`, `src/stats.py`, `scripts/gen_version_table.py` | da |
| `diagnostics/` | dodatne preverbe posameznega zagona, ki jih naredi skript (ne zagon sam), npr. `lightgbm_<različica>_*.csv/.json` iz `scripts/diagnose_lightgbm_medic3.py` — le metrike, nobenih podatkov | da |
| `PROVENANCE.md`, `sacct.txt` | ročni zapis tega, česar skript ne ve: SLURM opravila, ponovne oddaje, težave | da |

Stolpci `results.csv`: `run_id, dataset, dataset_id, algorithm, repeat, fold,
n_train, n_test, n_features, n_categorical, n_classes, roc_auc, train_time_s,
inference_time_s, warmup_s, device, preprocessing, raw_error, error,
git_commit, hostname, timestamp, n_classes_scored`. Dnevnik predobdelave je torej
del vsake vrstice (`preprocessing`, `raw_error`), ne ločena datoteka.
`n_classes_scored` (od 2026-09-22) pove, po koliko razredih je bilo povprečeno
makro ROC-AUC; `cc18_v2` tega stolpca še nima.

Oznaka zagona (`run_id`) je privzeto `<datum-čas>_<nabor>_<stroj>`; na Arnesu jo
poda `RUN_ID=... sbatch ...`. Nov `run_id` = nov zagon od začetka, stari se ne
prepiše. Zagoni, ki se ne obdržijo (poskusi, preverjanja), se preprosto
zbrišejo, preden gredo v commit.
