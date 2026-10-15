# medic3_160_v2 — RandomForest "kot iz škatle" na Medic3 (160 razredov), 2026-10-14

Delni ponovni zagon po popravku protokola (CLAUDE.md, Preprocessing policy →
Audit; commit `e3140ce`): RandomForest manjkajočih vrednosti ne nadomešča več,
ker jih scikit-learn 1.9.1 obravnava sam. Medic3 nima kategoričnih stolpcev, zato
ima XGBoost enak vhod kot prej; teče le kot **preverba**, da gruča ponovi
`medic3_160`. Ostali štirje algoritmi ostanejo iz `medic3_160`.

## Identifikacija (iz manifest.json in results.csv)

| Postavka | Vrednost |
|---|---|
| Repo commit | `8b601d1` (vseh 30 vrstic), čisto delovno drevo; modeli enaki kot v `e3140ce` |
| SLURM job ID | `20470899` (task 0 RandomForest, task 1 XGBoost) |
| Vozlišče | `gwn09` (H100 SXM) za oba; le CPU, 8 jeder |
| Okolje | micromamba `~/envs/tabular3.5`, Python 3.12.13 |
| Podatki | `data/medic3/Medic3.csv` z `scripts/medic3_160.json` (`max_classes: 160`), enako kot `medic3_160`; XGBoost spodaj potrjuje enak vhod in enake delitve |
| Oddaja | 2026-10-14 s `hpc-login4`; prvo učenje 2026-10-15 01:30, zadnje 02:06 |

## Izid

`python scripts/merge_results.py medic3_160_v2`: **30 vrstic (2 × 15), 0 z napako.**

| Algoritem | ROC-AUC `medic3_160` | ROC-AUC `medic3_160_v2` | razlika | mediana učenja |
|---|---|---|---|---|
| random_forest | 0.8530 | **0.8726** | boljši v 15/15 foldih, največ 0,025 | 11,7 s → 11,1 s |
| xgboost | 0.9400 | 0.9400 | **0 v vseh 15 foldih** | 153 s → 130 s |

Nativna obravnava NaN RandomForestu vrne informacijo o tem, katere vrednosti
manjkajo, ki mu jo je naša imputacija jemala; na Medic3 vrednosti zdaj nadomešča
le še TabICL, znotraj svoje knjižnice. XGBoost je bitno enak `medic3_160`.

## Celoten nabor rezultatov: `combined/`

`python scripts/combine_runs.py medic3_160_v2 --base medic3_160` →
`combined/results.csv` (90 vrstic: RandomForest in XGBoost iz tega zagona,
LightGBM, CatBoost, TabPFN-3.5, TabICL iz `medic3_160`). `python -m src.stats`
na njem: **vseh 15 parov ostane značilnih**, vrstni red TabPFN-3.5 > CatBoost >
TabICL > XGBoost > RandomForest > LightGBM nespremenjen (RandomForest proti
XGBoost: −0,0674, prej −0,0870; najšibkejši par ostane CatBoost proti
TabPFN-3.5, p_Holm 5,0e-03).

## Napovedi

`predictions/` (30 datotek `.npz`) ni v gitu in ne sme biti (zaupno). Kopije:
Arnes; prenesti na prenosnik, Kremen in USB-ključ.

## Opravila SLURM

| Task | Algoritem | Trajanje | MaxRSS | Vozlišče | Izid |
|---|---|---|---|---|---|
| 20470899_0 | random_forest | 0:03:19 | 10,7 GiB | gwn09 | COMPLETED |
| 20470899_1 | xgboost | 0:33:17 | 2,1 GiB | gwn09 | COMPLETED |

Celotni izpis: `sacct.txt`. Brez ponovnih oddaj.
