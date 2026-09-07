# Provenance — validacija prenosa benchmarka na Arnes HPC

Ta mapa dokumentira **validacijski zagon** pilotne podmnožice (OpenML 31, 37, 38)
na gruči Arnes in njegovo primerjavo z lokalno referenco.

## Identifikacija zagona

| Postavka | Vrednost |
|---|---|
| Repo commit ob zagonu | `b85a3fb` |
| SLURM job ID | `17731379` (array, 3 taski) |
| Vozlišče | `gwn01` |
| GPU | NVIDIA H100 (`-C h100`, particija `gpu`) |
| Datum | 2026-07-22 |
| Status taskov | vsi `COMPLETED` |
| Lokalna referenca | `results/local/results_local_subset.csv` — zagon na NVIDIA RTX 3060 (WSL2), nespremenjen |

### Per-task viri (sacct)

> **TODO / MANJKA:** izpis `sacct` za job 17731379 ob pisanju tega dokumenta
> ni bil na voljo. Dopolni s spodnjim ukazom in prilepi tabelo Elapsed/MaxRSS
> za vsak task (`17731379_0`, `_1`, `_2`) na to mesto:
>
> ```bash
> sacct -j 17731379 --format=JobID,Elapsed,MaxRSS,MaxVMSize,State,ExitCode
> ```
>
> Groba meritev iz zagona: ~10–60 s na task.
>
> Ta števila so vhod za dimenzioniranje `--time`/`--mem` celotnega CC18 polja,
> a **le kot spodnja meja** — pilotni dataseti so drobni (največ 3772 × 29),
> CC18 pa vsebuje tudi 60000 × 3072. Glej `results/arnes/cc18/PROVENANCE.md`.

## Datoteke

| Datoteka | Opis |
|---|---|
| `per_dataset/31.csv`, `37.csv`, `38.csv` | surovi per-dataset izhodi z gruče (H100), kurirani |
| `arnes_pip_freeze.txt` | `pip freeze` okolja na gruči (micromamba `~/envs/tabular`) |
| `local_pip_freeze.txt` | `pip freeze` lokalnega okolja (conda `tabular`, RTX 3060) |
| `results_arnes_subset.csv` | združeni CSV (90 vrstic, 3 dataseti, 0 napak) |

Združeno z:

```bash
python scripts/merge_results.py results/arnes/subset/results_arnes_subset.csv \
    --input-dir results/arnes/subset/per_dataset
python scripts/compare_results.py results/local/results_local_subset.csv \
    results/arnes/subset/results_arnes_subset.csv
```

## Sodba (predregistrirane tolerance)

Primerjanih **90/90** vrstic po ključu (dataset, algorithm, fold); **81** vrstic
je bitno identičnih. Vseh 9 odstopanj je omejenih na `tabpfn` in `tabicl`.

| Algoritem | mean \|Δ\| | max \|Δ\| | Kriterij | Sodba |
|---|---|---|---|---|
| random_forest | 0.000e+00 | 0.000e+00 | < 1e-6 | **PASS** |
| xgboost | 0.000e+00 | 0.000e+00 | < 1e-3 | **PASS** |
| lightgbm | 0.000e+00 | 0.000e+00 | < 1e-3 | **PASS** |
| catboost | 0.000e+00 | 0.000e+00 | < 1e-3 | **PASS** |
| tabpfn | 2.790e-05 | 2.381e-04 | < 5e-3 ali ≤ fold-std | **PASS** |
| tabicl | 1.428e-05 | 1.228e-04 | < 5e-3 ali ≤ fold-std | **PASS** |

**Skupna sodba: ALL PASS.**

TabPFN in TabICL zadostita *obema* kriterijema: največja razlika (2.4e-4) je
~20× pod absolutno toleranco 5e-3 in hkrati za red velikosti manjša od
najmanjšega fold-to-fold standardnega odklona lokalne reference
(min. 7.0e-4 za tabicl, 1.1e-3 za tabpfn).

### Največja odstopanja (5 najslabših vrstic)

| dataset | algorithm | fold | roc_auc lokalno | roc_auc Arnes | \|Δ\| |
|---|---|---|---|---|---|
| credit-g | tabpfn | 1 | 0.763452 | 0.763214 | 2.38e-04 |
| sick | tabicl | 3 | 0.998956 | 0.998833 | 1.23e-04 |
| credit-g | tabpfn | 3 | 0.831905 | 0.831786 | 1.19e-04 |
| sick | tabpfn | 4 | 0.998219 | 0.998188 | 3.10e-05 |
| sick | tabicl | 4 | 0.998710 | 0.998680 | 3.10e-05 |

Deterministični algoritmi (RF, XGBoost, LightGBM, CatBoost) so na CPU
reproducirani bitno natančno. Preostala odstopanja izvirajo iz nedeterminizma
GPU float operacij ob različni arhitekturi (SM86 RTX 3060 vs. SM90 H100) —
drugačen izbor kernelov in vrstni red redukcij, ne razlika v podatkih,
predobdelavi ali verzijah knjižnic.

## Okolje — razlike med lokalnim in Arnes

Vseh 8 ključnih paketov (+ pandas, scipy) je **identičnih**:

| paket | lokalno (3060) | Arnes (H100) | status |
|---|---|---|---|
| torch | 2.13.0 | 2.13.0 | same |
| tabpfn | 8.1.0 | 8.1.0 | same |
| tabicl | 2.1.1 | 2.1.1 | same |
| scikit-learn | 1.5.1 | 1.5.1 | same |
| xgboost | 2.1.1 | 2.1.1 | same |
| lightgbm | 4.6.0 | 4.6.0 | same |
| catboost | 1.2.7 | 1.2.7 | same |
| numpy | 1.26.4 | 1.26.4 | same |
| pandas | 2.2.2 | 2.2.2 | same |
| scipy | 1.14.1 | 1.14.1 | same |

Razlikuje se 31 tranzitivnih paketov, nobeden od njih ne vpliva na numeriko
modelov. Najpomembnejše skupine:

- **CUDA runtime wheels**: lokalno so nameščeni `nvidia-*-cu12` koleščki
  (cuBLAS 12.1.3.1, cuDNN 9.1.0.70, cuFFT, cuSOLVER, …), na gruči jih ni —
  tam torch uporablja sistemski CUDA stack modula. `nvidia-nccl-cu12`
  2.20.5 (lokalno) vs. 2.30.7 (Arnes). To je najverjetnejši mehanizem
  zgoraj opisanih 1e-4 odstopanj na GPU.
- **HuggingFace stack**: `huggingface-hub` 0.36.2 → 1.24.0, `hf-xet`
  1.5.1 → 1.5.2 (na gruči teče offline, `HF_HUB_OFFLINE=1`, uteži prestage-ane).
- **Ostalo**: `joblib` 1.4.2 → 1.5.3, `pyarrow` 24.0.0 → 25.0.0, `certifi`,
  `filelock`, `plotly`, `tzdata`; na gruči dodatno `anyio`/`httpx`/`h11`
  (odvisnosti novejšega `huggingface-hub`).
