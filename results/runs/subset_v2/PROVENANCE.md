# subset_v2 — validacija prenosa na Arnes z nadgrajenim okoljem

Pilotna trojica (OpenML 31, 37, 38) na gruči Arnes, isti commit in iste
različice knjižnic kot lokalni zagon `subset_v2_local`. Namen: potrditi, da
gruča vrne enake številke kot lokalni računalnik, preden se odda polni CC18.

## Identifikacija (iz manifest.json)

| Postavka | Vrednost |
|---|---|
| Repo commit | `57e029a`, čisto delovno drevo |
| SLURM job ID | `19096168` (array, 3 taski) |
| Vozlišče | `gwn06.arnes.si`, NVIDIA H100 PCIe, 8 jeder |
| Okolje | micromamba `~/envs/tabular2`, Python 3.12.14, torch 2.14.0+cu130, CUDA 13.0 |
| Datum | 2026-09-11 |
| Lokalna referenca | `results/runs/subset_v2_local` (RTX 3060, WSL2, Python 3.12.13) |

Ključni paketi (numpy, pandas, scikit-learn, scipy, xgboost, lightgbm,
catboost, tabpfn, tabicl, torch, openml) so **identični** lokalnemu okolju;
razlikuje se 14 tranzitivnih paketov brez vpliva na numeriko.

## Sodba

`python scripts/compare_results.py subset_v2_local subset_v2`, 90/90 vrstic:

| Algoritem | mean \|Δ\| | max \|Δ\| | Sodba |
|---|---|---|---|
| random_forest | 0 | 0 | bitno identično |
| xgboost | 0 | 0 | bitno identično |
| lightgbm | 0 | 0 | bitno identično |
| catboost | 0 | 0 | bitno identično |
| tabicl | 1e-6 | 1.5e-5 | PASS (nedeterminizem GPU, SM86 proti SM90) |
| tabpfn | 6.3e-5 | 3.7e-4 | PASS (nedeterminizem GPU) |

Odstopanja TabPFN/TabICL so za red velikosti pod odklonom med foldi (~2e-3)
in enakega reda kot pri validaciji stare kode julija 2026. **ALL PASS**, gruča
je pripravljena za `cc18_v2`.

## Potrdilo opravila (sacct)

Izpolni na prijavnem vozlišču:

```bash
sacct -j 19096168 --format=JobID,JobName%20,Elapsed,MaxRSS,State,NodeList
```

```
(prilepi izpis sem)
```
