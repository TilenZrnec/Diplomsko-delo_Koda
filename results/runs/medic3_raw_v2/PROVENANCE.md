# medic3_raw_v2 — RandomForest "kot iz škatle" na Medic3 (vseh 220 razredov), 2026-10-14

Delni ponovni zagon po popravku protokola (CLAUDE.md, Preprocessing policy →
Audit; commit `e3140ce`): RandomForest manjkajočih vrednosti ne nadomešča več,
ker jih scikit-learn 1.9.1 obravnava sam. Medic3 nima kategoričnih stolpcev, zato
ima XGBoost enak vhod kot prej; teče le kot **preverba**, da gruča ponovi
`medic3_raw`. Ostali štirje algoritmi ostanejo iz `medic3_raw`.

## Identifikacija (iz manifest.json in results.csv)

| Postavka | Vrednost |
|---|---|
| Repo commit | `8b601d1` (vseh 30 vrstic), čisto delovno drevo; modeli enaki kot v `e3140ce` |
| SLURM job ID | `20470898` (task 0 RandomForest, task 1 XGBoost) |
| Vozlišči | RandomForest `gwn09` (H100 SXM), XGBoost `gwn01` (H100 PCIe); oba le CPU, 8 jeder |
| Okolje | micromamba `~/envs/tabular3.5`, Python 3.12.13 |
| Podatki | `data/medic3/Medic3.csv`, ista datoteka kot `medic3_raw` (sha256 v njegovem PROVENANCE.md); XGBoost spodaj jo potrjuje: enak vhod, enake delitve, enak rezultat |
| Oddaja | 2026-10-14 s `hpc-login4`; prvo učenje 2026-10-15 01:26, zadnje 02:44 |

## Izid

`python scripts/merge_results.py medic3_raw_v2`: **30 vrstic (2 × 15), 0 z napako.**

| Algoritem | ROC-AUC `medic3_raw` | ROC-AUC `medic3_raw_v2` | razlika | mediana učenja |
|---|---|---|---|---|
| random_forest | 0.8149 | **0.8394** | boljši v 15/15 foldih, največ 0,029 | 13,1 s → 13,6 s |
| xgboost | 0.9306 | 0.9306 | **0 v vseh 15 foldih** | 152 s → 304 s (gwn08 SXM → gwn01 PCIe) |

Nativna obravnava NaN RandomForestu vrne informacijo o tem, katere vrednosti
manjkajo (pri 82 % manjkajočih na Medic3 močan signal), ki mu jo je naša
imputacija jemala. Na Medic3 zdaj vrednosti nadomešča le še TabICL, znotraj
svoje knjižnice. XGBoost je bitno enak `medic3_raw` (na Kremnu se je zaradi
drugačnega procesorja razlikoval za 6,2e-4, glej CLAUDE.md, Conventions);
podaljšani čas je posledica počasnejšega vozlišča, ne kode.

## Celoten nabor rezultatov: `combined/`

`python scripts/combine_runs.py medic3_raw_v2 --base medic3_raw` →
`combined/results.csv` (90 vrstic: RandomForest in XGBoost iz tega zagona,
LightGBM, CatBoost, TabPFN, TabICL iz `medic3_raw`). `python -m src.stats` na njem:
TabPFN izključen (15 × meja 160 razredov, kot prej); **vseh 10 parov ostane
značilnih**, vrstni red CatBoost > TabICL > XGBoost > RandomForest > LightGBM
nespremenjen (RandomForest proti XGBoost: −0,0911, prej −0,1157).

## Napovedi

`predictions/` (30 datotek `.npz`) ni v gitu in ne sme biti (zaupno). Kopije: Arnes,
Kremen, prenosnik in USB-ključ (2026-10-15). Preverjeno na Kremnu: vsako učenje
ima datoteko, nobene odvečne, foldi vsake ponovitve razdelijo vse vrstice,
vsi algoritmi imajo na istem foldu iste vrstice, ROC-AUC iz datotek se z
`results.csv` ujema (XGBoost ≤ 1e-16, RandomForest natanko).

## Opravila SLURM

| Task | Algoritem | Trajanje | MaxRSS | Vozlišče | Izid |
|---|---|---|---|---|---|
| 20470898_0 | random_forest | 0:04:04 | 14,7 GiB | gwn09 | COMPLETED |
| 20470898_1 | xgboost | 1:17:59 | 2,2 GiB | gwn01 | COMPLETED |

Celotni izpis: `sacct.txt`. Brez ponovnih oddaj.
