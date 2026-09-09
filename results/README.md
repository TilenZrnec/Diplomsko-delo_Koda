# Rezultati

Dve generaciji rezultatov. **Nova** (od 2026-09-09) živi v `runs/`, vsak zagon v
svoji mapi; **stara** (`local/`, `arnes/`) je arhiv zagonov s prejšnjo kodo in
prejšnjimi različicami knjižnic in se ne spreminja več. Nič v tej mapi se ne
ureja na roko, vse nastane iz zagona.

## `runs/<run_id>/` — vsak zagon svoja mapa

| Datoteka | Kaj je | V gitu? |
|---|---|---|
| `manifest.json` | git commit, stroj, GPU, Python, torch/CUDA, celoten `config.yaml`, `pip freeze` | da |
| `per_dataset/<id>.csv` | dokončan nabor, ena vrstica na (algoritem, fold) | da |
| `per_dataset/<id>.csv.partial` | nedokončan nabor (kontrolna točka); ob ponovnem zagonu z istim `run_id` se nadaljuje | ne |
| `predictions/<id>/<algoritem>_r<rep>_f<fold>.npz` | `predict_proba`, `y_test`, `test_idx` vsakega učenja — za poljubno metriko brez ponovnega zagona | ne (velike) |
| `results.csv` | združeni `per_dataset/*.csv` (`scripts/merge_results.py` ali samodejno lokalno) | da |
| `summary/` | tabele (CSV + LaTeX), diagram kritične razdalje, `razlicice.tex` — iz `src/summary.py`, `src/stats.py`, `scripts/gen_version_table.py` | da, za kurirane zagone |

Stolpci `results.csv`: `run_id, dataset, dataset_id, algorithm, repeat, fold,
n_train, n_test, n_features, n_categorical, n_classes, roc_auc, train_time_s,
inference_time_s, warmup_s, device, preprocessing, raw_error, error,
git_commit, hostname, timestamp`. Dnevnik predobdelave je torej del vsake
vrstice (`preprocessing`, `raw_error`), ne ločena datoteka.

Oznaka zagona (`run_id`) je privzeto `<datum-čas>_<nabor>_<stroj>`; na Arnesu jo
poda `RUN_ID=... sbatch ...`. Nov `run_id` = nov zagon od začetka, stari se ne
prepiše. Zagoni, ki se ne obdržijo (poskusi, preverjanja), se preprosto
zbrišejo, preden gredo v commit.

Zagoni v `runs/`:

| `run_id` | Kaj | Okolje |
|---|---|---|
| `check_refactor_oldenv` | preverjanje, da refaktorirana koda vrne iste številke kot `local/results_local_subset.csv` — vseh 90 vrstic Δ = 0,0 | staro (`tabular`, Python 3.10, sklearn 1.5.1, torch 2.13) |
| `subset_v2_local` | pilotna trojica z nadgrajenimi knjižnicami (`requirements.txt`, zamrznitev 2026-09-09) | novo (`tabular2`, Python 3.12, sklearn 1.9.0, torch 2.14) |

## Arhiv (stara koda, stare različice)

| Mapa | Zagon | Vsebina |
|---|---|---|
| `local/` | lokalni pilot, NVIDIA RTX 3060 (WSL2), 3 nabori | `results_local_subset.csv` (90 vrstic) in `preprocessing_log.md` |
| `arnes/subset/` | validacija prenosa na gručo Arnes (H100), isti 3 nabori | `results_arnes_subset.csv`, `per_dataset/` vhodi, oba `pip freeze`, `PROVENANCE.md` |
| `arnes/cc18/` | poln OpenML-CC18, 72 naborov, 2026-08 | `results_arnes_cc18.csv` (2160 vrstic) in `PROVENANCE.md` |

Stari CSV-ji imajo le stolpce `dataset, algorithm, fold, roc_auc, train_time_s,
inference_time_s[, error]`; `src/summary.py` in `src/stats.py` jih še vedno
znata prebrati (`python -m src.summary results/arnes/cc18/results_arnes_cc18.csv`).
Ti rezultati so nastali z `scikit-learn 1.5.1`, `xgboost 2.1.1`, `tabpfn 8.1.0`
itd. (glej `arnes/subset/arnes_pip_freeze.txt`) in **niso** rezultati diplome
po nadgradnji; služijo kot referenca za primerjavo učinka različic.
