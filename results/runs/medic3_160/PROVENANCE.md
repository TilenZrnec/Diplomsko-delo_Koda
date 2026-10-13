# medic3_160 — zaupni nabor Medic3, 160 najpogostejših razredov (zamrznitev 2026-10-05)

Različica Medic3, na kateri teče **vseh šest algoritmov** (TabPFN-3.5 ima mejo
160 razredov). Druga različica je `medic3_raw` (vseh 220 razredov); tam je tudi
podrobna razlaga propada LightGBM, ki velja tudi tu.

## Identifikacija (iz manifest.json in results.csv)

| Postavka | Vrednost |
|---|---|
| Repo commit | `420bdac` (vseh 90 vrstic), čisto delovno drevo |
| SLURM job ID | `20318170` (RF, XGBoost, LightGBM), `20318171` (TabPFN, TabICL), `20318172` (CatBoost) |
| Vozlišča | `gwn08` (RF, TabPFN, TabICL; H100 80GB HBM3), `gwn09` (LightGBM, CatBoost; H100 SXM), `gwn05` (XGBoost; H100 PCIe) — drevesni ansambli tečejo le na CPU, 8 jeder na opravilo |
| Okolje | micromamba `~/envs/tabular3.5`, Python 3.12.13, torch 2.14.1+cu130 |
| Uteži | `tabpfn-v3.5-20260909.safetensors`, `tabicl-classifier-v2-20260212.ckpt` (kot `cc18_v3`) |
| Podatki | `data/medic3/Medic3.csv`, sha256 `27b23cbfa0511850c1b030482a0cc08d3c7ea3d7862eee64f18f6b40d2a5c005` (enak na Kremnu in Arnesu); `scripts/medic3_160.json` (`max_classes: 160`) |
| Omejitev razredov | 220 → 160 razredov, obdržanih 117 825 od 122 093 vrstic, zavrženih 4268 (3.50 %); najmanjši obdržani razred ima 102 vrstici |
| Oddaja | 2026-10-10 ~10:45 s `hpc-login4` (manifest 11:08:44 UTC), konec 2026-10-11 13:00 UTC (zadnje učenje CatBoost) |
| Protokol | 117 825 vrstic × 275 atributov, 160 razredov; 5 delitev × 3 ponovitve; seme delitev 42, seme modela 42 + ponovitev; en algoritem na SLURM task |

## Izid

`python scripts/merge_results.py medic3_160`: **90 vrstic, 0 z napako**, 0 vrstic
z `raw_error` (oba temeljna modela sta sprejela surov vhod). Makro-OvR je v vsakem
učenju zajel vseh 160 razredov.

| Algoritem | ROC-AUC (15 učenj) | mediana učenja | mediana napovedovanja | naprava |
|---|---|---|---|---|
| tabpfn (TabPFN-3.5) | 0.9614 ± 0.0007 | 2.3 s | 74.4 s | H100 HBM3 |
| catboost | 0.9599 ± 0.0009 | 5550 s | 0.21 s | cpu x8 |
| tabicl | 0.9543 ± 0.0011 | 10.0 s | 76.6 s | H100 HBM3 |
| xgboost | 0.9400 ± 0.0019 | 153 s | 2.0 s | cpu x8 |
| random_forest | 0.8530 ± 0.0025 | 11.7 s | 0.56 s | cpu x8 |
| lightgbm | **0.5057 ± 0.0063** | 50 s | 1.3 s | cpu x8 |

`python -m src.stats medic3_160` (popravljeni t-test za ponovljeno prečno
preverjanje, Bouckaert in Frank 2004, Holm čez 15 parov): **vseh 15 parov je
značilnih**. Najšibkejši je CatBoost proti TabPFN-3.5: razlika le −0.0015, a
TabPFN-3.5 je boljši v vseh 15 učenjih (t = −3.33, p_Holm = 5.0e-03). Vrstni red:
TabPFN-3.5 > CatBoost > TabICL > XGBoost > naključni gozd > LightGBM. Razlike med
prvimi tremi so majhne (≤ 0.0071); za en nabor s 3 × 5 delitvami Bouckaert in
Frank priporočata 10 × 10 za boljšo ponovljivost — to v nalogi navedemo.

## LightGBM: privzete nastavitve divergirajo (ni napaka naše kode)

Propad v **vseh 15 učenjih (ROC-AUC 0.4999–0.5193)**, brez izjeme. Diagnostika
na foldu 0 (`scripts/diagnose_lightgbm_medic3.py`, `diagnostics/`, 2026-10-12,
z delovno različico skripta pred commitom `a5a2197`; ponovna izdelava iz commita je
bila prekinjena, del brez naše kode je v obeh enak — glej `*_info.json`) pokaže
isto kot pri `medic3_raw`:

- neodvisna pot brez naše kode: učna izguba raste od 3.91 po prvi iteraciji do
  ~32.5, največji list s 73 na red velikosti 10^6, točnost ~0.059 ≈ delež
  največjega razreda 0.063, ROC-AUC ~0.50;
- naša pot je neodvisni enaka v prvih iteracijah (razlika ≤ 1e-12), nato se zaradi
  neponovljivosti LightGBM (`deterministic = false`) in ojačanja divergence
  razideta; Arnes je na foldu 0 dal 0.5168, lokalni poskusi ~0.50;
- `reg_lambda=1` → ROC-AUC 0.948, `min_child_weight=1` → 0.945 (privzeti vrednosti
  XGBoost), učna izguba pada, največji list pod 20;
- LightGBM 4.6.0 se obnaša enako.

Diploma poroča privzeti rezultat; vrednosti z zavoro so le diagnostika.

## Opravila SLURM

| Task | Algoritem | --mem | --time | Trajanje | MaxRSS | Izid |
|---|---|---|---|---|---|---|
| 20318170_0 | random_forest | 64G | 1-00:00:00 | 0:03:31 | 11.1 GiB | COMPLETED |
| 20318170_1 | xgboost | 64G | 1-00:00:00 | 0:39:27 | 2.1 GiB | COMPLETED |
| 20318170_2 | lightgbm | 64G | 1-00:00:00 | 0:13:16 | 2.0 GiB | COMPLETED |
| 20318171_4 | tabpfn | 160G | 1-00:00:00 | 0:19:41 | 5.9 GiB | COMPLETED |
| 20318171_5 | tabicl | 160G | 1-00:00:00 | 0:22:44 | 92.3 GiB | COMPLETED |
| 20318172_3 | catboost | 64G | 2-00:00:00 | 23:08:17 | 3.9 GiB | COMPLETED |

Celotni izpis: `sacct.txt`. Brez ponovnih oddaj.

## Napovedi

`predictions/` (90 datotek `.npz`, ~802 MiB) **ni v gitu in ne sme biti**
(zaupno). Kopije: Arnes, prenosnik (preverjeno 2026-10-12: vsako učenje ima
datoteko; ROC-AUC iz datotek se z `results.csv` ujema na ≤ 3.8e-9 pri vseh
algoritmih razen LightGBM, ≤ 7.1e-4, ker so njegove verjetnosti v float32 skoraj
vse točno 0) in uporabnikov USB-ključ (2026-10-12). Kremen: še ne.

## Opombe za ponovitev

- **TabPFN-3.5 je prvič tekel na ~94 000 učnih vrsticah** (160 razredov): brez
  težav, mediana napovedovanja 74 s na učenje, MaxRSS 5.9 GiB.
- **TabICL potrebuje ~92 GiB RAM**: `--mem=160G` je bil nujen.
- **CatBoost** 1.54 h na učenje (mediana 5550 s) na `gwn09` (SXM) proti oceni
  ~1.1 h; skupaj 23 h 08 min od 48 h.
