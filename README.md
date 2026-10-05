# Diplomsko delo — koda

Primerjava šestih algoritmov za tabelarične podatke (naključni gozd, XGBoost,
LightGBM, CatBoost, TabPFN-3.5, TabICL) na zbirki OpenML-CC18 (72 podatkovnih
množic) in na zaupnem naboru Medic3, povsod s privzetimi hiperparametri.

**Za začetek preberi [`razlaga_repozitorija/razlaga.md`](razlaga_repozitorija/razlaga.md)**:
razlaga po korakih za začetnika, s točnimi ukazi za lokalni zagon in za Arnes.

## Kaj je kje

| Pot | Kaj |
|---|---|
| `config.yaml` | vse nastavitve eksperimenta: seme, foldi, ponovitve, algoritmi, nabori |
| `requirements.txt` | pripete različice knjižnic (zamrznitev 2026-10-05, okolje `tabular3.5`) |
| `src/` | koda benchmarka (spodaj) |
| `src/models/` | en modul na algoritem, vsak s funkcijo `run()` |
| `scripts/` | pomožni programi in seznami naborov (spodaj) |
| `results/runs/<zagon>/` | rezultati zagonov; trenutno le `cc18_v2` (glej [`results/README.md`](results/README.md)) |
| `razlaga_repozitorija/` | razlaga, diagram zaporedja `zaporedje.puml` in njegov preverjalnik |
| `data/` (ni v gitu) | predpomnilnik OpenML in zaupni `medic3/Medic3.csv` |
| `CLAUDE.md` | delovne opombe za Claude (v angleščini): stanje, odločitve, naslednji koraki |

### `src/`

| Datoteka | Kaj naredi |
|---|---|
| `config.py` | prebere `config.yaml` |
| `data.py` | naloži nabor (OpenML ali CSV) in enkrat za vse algoritme naredi folde |
| `runner.py` | edina zanka učenja: vsak algoritem na vsakem foldu, kontrolne točke, `manifest.json` |
| `run_benchmark.py` | lokalni zagon (vsi nabori zaporedno) |
| `run_one_dataset.py` | zagon enega nabora (en SLURM task na Arnesu) |
| `summary.py` | tabele: ROC-AUC, rangi, zmage, časi, nasičeni nabori |
| `stats.py` | statistični testi: Friedman z Iman-Davenportovo F_F, Nemenyi, Wilcoxon s Holmom; pri enem naboru popravljeni t-test |
| `utils.py` | izračun ROC-AUC, opis naprave (CPU/GPU) |

### `scripts/` po namenu

| Namen | Datoteke |
|---|---|
| seznami naborov | `subset_ids.json`, `cc18_ids.json` (naredi ga `gen_cc18_ids.py`), `medic3.json`, `medic3_160.json` |
| spoznavanje podatkov | `profile_datasets.py`, `profile_medic3.py` |
| Arnes | `prestage.py` (vnaprejšnji prenos naborov in utež), `run_subset.sh`, `run_cc18.sh`, `run_medic3.sh` (SLURM) |
| po zagonu | `merge_results.py` (združi), `compare_results.py` (primerja dva zagona), `gen_version_table.py` (tabela različic za diplomo) |

## Okolja conda

| Ime | Kaj |
|---|---|
| `tabular3.5` | trenutno: zamrznitev 2026-10-05, TabPFN-3.5 |
| `tabular3` | prejšnja zamrznitev 2026-09-09, TabPFN-3; z njim je nastal `cc18_v2` |
| `tabularOriginal` | prvotni sklad iz julija in avgusta 2026 |

## Najpogostejši ukazi

```bash
conda activate tabular3.5
python -m src.run_benchmark --dataset-set subset   # lokalni pilot na treh naborih
python -m src.summary cc18_v2                      # tabele rezultatov
python -m src.stats cc18_v2                        # statistični testi
```

Starejši zagoni in arhiv iz avgusta 2026 so bili zbrisani; ostanejo v git
zgodovini (commit `b1b9120`).
