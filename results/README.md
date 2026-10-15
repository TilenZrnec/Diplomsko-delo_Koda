# Rezultati

Vsak zagon živi v svoji mapi pod `runs/`. Hranimo samo zagone, ki jih diploma
trenutno uporablja; nadomeščene zbrišemo v ločenem commitu, ostanejo pa v git
zgodovini. Nič v tej mapi se ne ureja na roko, vse nastane iz zagona.

## Kaj je zdaj tukaj

Deli imen: `cc18` = 72 naborov OpenML-CC18; `medic3_raw` = zaupni Medic3 z vsemi
220 razredi; `medic3_160` = Medic3, omejen na 160 najpogostejših razredov (meja
TabPFN); `subset` = trije majhni nabori (credit-g, diabetes, sick) za hitre
preverbe; `_v2`/`_v3`/`_v4` = različica tega nabora rezultatov (nova zamrznitev
knjižnic ali popravek protokola); `_local` = zagon na Kremnu namesto na Arnesu;
**`combined/`** = celotna tabela vseh šestih algoritmov, ki jo
`scripts/combine_runs.py` sestavi iz delnega ponovnega zagona in prejšnjega
zagona (`sources.json` pove, od kod je kateri algoritem).

**Rezultati diplome**

| Mapa | Kaj je |
|---|---|
| `cc18_v4/combined/` | CC18, vseh šest algoritmov: RandomForest in XGBoost iz `cc18_v4`, ostali štirje iz `cc18_v3` |
| `medic3_raw_v2/combined/` | Medic3 (220 razredov), vseh šest: RandomForest in XGBoost iz `medic3_raw_v2`, ostali iz `medic3_raw` |
| `medic3_160_v2/combined/` | Medic3 (160 razredov), vseh šest: RandomForest in XGBoost iz `medic3_160_v2`, ostali iz `medic3_160` |

**Zagoni, iz katerih so sestavljeni**

| `run_id` | Kaj je | Vloga | Okolje |
|---|---|---|---|
| `cc18_v4` | CC18, **le RandomForest + XGBoost**, ponovljena po popravku "kot iz škatle" (2026-10-14/15), 2160 vrstic | vir RF in XGBoost | Arnes `~/envs/tabular3.5`, H100 (le CPU) |
| `cc18_v3` | CC18, vseh šest, zamrznitev 2026-10-05 (TabPFN-3.5), 6480 vrstic | vir LightGBM, CatBoost, TabPFN, TabICL; **njegove vrstice RF in XGBoost so nadomeščene** (stara predobdelava) | Arnes `~/envs/tabular3.5`, H100 |
| `medic3_raw_v2` | Medic3 (220), RandomForest (popravljen) + XGBoost (preverba, enak kot prej), 30 vrstic | vir RF in XGBoost | Arnes, H100 (le CPU) |
| `medic3_raw` | Medic3 (220), vseh šest, 90 vrstic (TabPFN-3.5: 15 × meja 160 razredov; LightGBM propade) | vir LightGBM, CatBoost, TabPFN, TabICL; RF nadomeščen; `diagnostics/` = analiza propada LightGBM | Arnes, H100 |
| `medic3_160_v2` | Medic3 (160), RandomForest (popravljen) + XGBoost (preverba), 30 vrstic | vir RF in XGBoost | Arnes, H100 (le CPU) |
| `medic3_160` | Medic3 (160), vseh šest, 90 vrstic | vir LightGBM, CatBoost, TabPFN-3.5, TabICL; RF nadomeščen; `diagnostics/` = analiza propada LightGBM | Arnes, H100 |

**Poskus uglaševanja (poskus 4)**

| `run_id` | Kaj je | Okolje |
|---|---|---|
| `cc18_xgb_tuned` | le `xgboost_tuned` (XGBoost, uglašen z iskanjem po mreži in gnezdenim prečnim preverjanjem) na 68 naborih `cc18_tuning`: del 1 = 36 nenasičenih (končan 2026-10-15, job `20494570`, 540 vrstic), del 2 = 32 nasičenih (sledi); `tuning/` = vseh 128 notranjih rezultatov na fold; primerja se s `cc18_v4/combined` | Arnes `~/envs/tabular3.5`, le CPU |

**Le preverbe (niso rezultati diplome)**

| `run_id` | Kaj je | Okolje |
|---|---|---|
| `subset_v4_local` | RandomForest in XGBoost po popravku "kot iz škatle" na trojici; `diagnostics/audit_defaults.txt` = izpis revizije vseh šestih algoritmov | Kremen `tabular3.5`, RTX 3060 |
| `subset_xgb_tuned_local` | pilot poskusa uglaševanja (`xgboost_tuned`) na trojici pred zagonom na Arnesu | Kremen `tabular3.5` |
| `subset_v3_local` | vseh šest na trojici: deluje zamrznitev 2026-10-05 | Kremen `tabular3.5`, RTX 3060 |
| `subset_v3` | ista preverba na gruči | Arnes `~/envs/tabular3.5`, H100 |

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
