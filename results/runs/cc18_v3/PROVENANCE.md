# cc18_v3 — polni zagon OpenML-CC18 z zamrznitvijo 2026-10-05 (TabPFN-3.5)

Rezultat, na katerem temelji diploma. Nadomešča `cc18_v2` (zamrznitev
2026-09-09, TabPFN-3), ki je po tem zagonu ostal le v zgodovini gita.

## Identifikacija (iz manifest.json in results.csv)

| Postavka | Vrednost |
|---|---|
| Repo commit | `69cc51e` (vseh 6480 vrstic), čisto delovno drevo |
| SLURM job ID | `20138834` (68 običajnih naborov), `20138836` (4 veliki nabori), `20176087` (ponovna oddaja CIFAR_10, glej spodaj) |
| Vozlišča | `gwn01`–`gwn04`, `gwn06` (H100 PCIe, 256 GB) in `gwn08`, `gwn10` (H100 80GB HBM3, 512 GB); 8 jeder na opravilo |
| Okolje | micromamba `~/envs/tabular3.5`, Python 3.12.13, torch 2.14.1+cu130, CUDA 13.0 |
| Uteži | `tabpfn-v3.5-20260909.safetensors`, `tabicl-classifier-v2-20260212.ckpt` |
| Oddaja | 2026-10-06 (manifest ustvarjen 15:03:41+00:00), konec 2026-10-08 |
| Protokol | 72 naborov × 6 algoritmov × 5 delitev × 3 ponovitve; seme delitev 42, seme modela 42 + ponovitev |

Različice knjižnic: `summary/razlicice.tex` (samodejno iz manifest.json).

## Izid

- `python scripts/merge_results.py cc18_v3`: **6480 vrstic, 72 naborov, 0 vrstic
  z napako**, brez `.partial`; 0 vrstic z `raw_error` (oba temeljna modela sta
  povsod sprejela surov vhod). Prvič so **vsi algoritmi uspeli na vseh 72
  naborih**; v `cc18_v2` je bilo 30 neuspelih učenj (CIFAR_10 × TabPFN-3, TabICL).
- Vsak algoritem ima 1080 vrstic. Naprava: `cpu x8` 4320 vrstic (drevesni
  ansambli), `cuda: NVIDIA H100 PCIe` 1680, `cuda: NVIDIA H100 80GB HBM3` 480
  (TabPFN-3.5, TabICL). Makro-OvR je povsod zajel vse razrede
  (`n_classes_scored = n_classes`).
- Nasičenih naborov (najboljši ROC-AUC ≥ 0,995) je 35, nenasičenih 37
  (`cc18_v2`: 33 in 39) — TabPFN-3.5 je čez mejo potisnil breast-w (0.9949 →
  0.9951) in electricity (0.9853 → 0.9968).

| Algoritem | povprečni rang | zmage | povprečni ROC-AUC (72) |
|---|---|---|---|
| tabpfn (TabPFN-3.5) | 1.431 | 48 | 0.9416 |
| tabicl | 1.875 | 26 | 0.9391 |
| catboost | 3.368 | 1 | 0.9283 |
| lightgbm | 4.458 | 0 | 0.9222 |
| xgboost | 4.660 | 2 | 0.9209 |
| random_forest | 5.208 | 0 | 0.9185 |

Friedman: chi2_F = 249.375, F_F = 160.051 na (5, 355) df, p = 1.14e-88;
Nemenyi CD = 0.889. Nenasičeni (37): F_F = 93.151 na (5, 180) df, CD = 1.240.
TabPFN-3.5 je proti TabICL boljši na 45, slabši na 23 naborih,
p_Holm = 1.77e-03 (nenasičeni: 29:8, p_Holm = 2.57e-03).

### Primerjava s `cc18_v2` (`compare_results.py cc18_v2 cc18_v3`, 6480 skupnih vrstic)

| Algoritem | mean \|Δ\| | max \|Δ\| | Razlaga |
|---|---|---|---|
| catboost, lightgbm, xgboost | 0 | 0 | bitno identično |
| random_forest | 1.5e-7 | 2.3e-5 | 31 od 1080 učenj (cmc, analcatdata_dmft, first-order-theorem-proving, jm1, mfeat-morphological, PhishingWebsites); najverjetneje vrstni red seštevanja glasov dreves v vzporednih nitih (`n_jobs=-1`), ni preverjeno |
| tabicl | 1.4e-5 | 3.7e-3 | ista knjižnica; največ na adult, ki je tokrat tekel na drugi izvedbi H100 (PCIe → HBM3); CIFAR_10 v `cc18_v2` ni uspel |
| tabpfn | 4.5e-3 | 6.4e-2 | drug model (TabPFN-3 → TabPFN-3.5); boljši na 56 od 71 naborov (CIFAR_10 izvzet) |

**Vrstni red prvih dveh se je zamenjal:** v `cc18_v2` je bil TabICL prvi
(rang 1.576) in TabPFN-3 drugi (1.896), TabICL boljši na 47 : 20 naborih
(p_Holm = 0.0046); v `cc18_v3` je TabPFN-3.5 prvi (1.431) in TabICL drugi
(1.875). Vrstni red drevesnih ansamblov je nespremenjen.

## Opravila SLURM

| Job ID | Array | --mem | --time | Izid |
|---|---|---|---|---|
| 20138834 | 0-26,28-59,62-69,71%4 | 64G | 12:00:00 | 68/68 COMPLETED |
| 20138836 | 27,60,61,70 | 240G | 2-00:00:00 | 3/4 COMPLETED, 61 OUT_OF_MEMORY |
| 20176087 | 61 | 480G, `--constraint=sxm` | 2-00:00:00 | 1/1 COMPLETED |

**Incident CIFAR_10 (task 61).** Prvi poskus je tekel na `gwn08` (512 GB) in je
bil po 11 h 03 min (2026-10-07 04:52 UTC) ubit zaradi pomnilnika (MaxRSS
256 GiB pri 240G). Delni rezultat je ohranil 75 učenj: štiri drevesne ansamble
in **vseh 15 učenj TabPFN-3.5** (prvič na CIFAR_10, zadnje ob 04:49:25 UTC).
Ubit je bil torej TabICL pri prvem učenju: v `cc18_v2` je na vozlišču z 256 GB
zahtevek ~378 GB takoj zavrnil alokator (mehka napaka z zapisano vrstico), na
vozlišču z 512 GB pa zahtevek ni bil zavrnjen, zato je cgroup ubil celotno
opravilo in vrstica ni nastala. Ponovna oddaja (uporabnikova izbira) z istim
`RUN_ID` je nadaljevala pri TabICL, učenje 0:
`PYTHONUNBUFFERED=1 ALLOW_SPARSE_ARRAY=1 RUN_ID=cc18_v3 sbatch --array=61
--constraint=sxm --mem=480G --time=2-00:00:00 scripts/run_cc18.sh`. Na `gwn10`
je TabICL uspel v vseh 15 učenjih (ROC-AUC 0.9025–0.9079, povprečje 0.9050,
mediana napovedovanja 290 s) pri MaxRSS 394 GiB. Učenja CIFAR_10 so zato z dveh
vozlišč (drevesa in TabPFN-3.5 z `gwn08`, TabICL z `gwn10`), obe H100 HBM3.

Poraba (MaxRSS iz `sacct`):

| Task | Nabor | Trajanje | MaxRSS | Vozlišče |
|---|---|---|---|---|
| 20138834 (68 taskov) | običajni | 0:34 do 2:57:30 | največ 5.1 GiB | gwn01/02/03/06/08 |
| 20138836_27 | mnist_784 (554) | 2:45:25 | 132.9 GiB | gwn08 |
| 20138836_60 | Devnagari-Script (40923) | 1-11:13:49 | 172.6 GiB | gwn04 |
| 20138836_61 | CIFAR_10 (40927), 1. poskus | 11:03:06 | 256.1 GiB (OOM) | gwn08 |
| 20176087_61 | CIFAR_10 (40927), nadaljevanje | 1:34:35 | 394.1 GiB | gwn10 |
| 20138836_70 | Fashion-MNIST (40996) | 4:04:27 | 138.9 GiB | gwn08 |

Celotni izpis: `sacct.txt` v tej mapi
(`sacct -j 20138834,20138836,20176087 --format=JobID,JobName%20,Elapsed,MaxRSS,State,NodeList`).

## Napovedi

`predictions/` (6480 datotek `.npz`, ~474 MB) ni v gitu. Kopije: Arnes (mapa
zagona), Kremen (preverjeno 2026-10-08: vsako uspelo učenje ima datoteko, ROC-AUC,
izračunan iz datotek, se z `results.csv` ujema na ≤ 3.2e-4, ker so verjetnosti
shranjene kot float32) in uporabnikov USB-ključ (arhiv).

## Opombe za ponovitev

- **Skupni računski čas 62.7 h** (vsota `train_time_s` + `inference_time_s`;
  `cc18_v2`: 69.4 h), od tega CatBoost 47.8 h, XGBoost 5.5 h, TabICL 4.0 h,
  LightGBM 2.5 h, TabPFN-3.5 2.5 h, naključni gozd 0.5 h.
- **Devnagari-Script je spet najdaljši:** 35 h 14 min od 2 dni (CatBoost
  28.3 h), na `gwn04` v istem tempu kot v `cc18_v2`.
- **Časi so odvisni od vozlišča.** CatBoost je na CIFAR_10 z bitno identičnimi
  rezultati porabil 8.75 h na `gwn08` proti 14.24 h na `gwn04` v `cc18_v2`.
  Časi med nabori in med zagonoma zato niso strogo primerljivi; v nalogi to
  navedemo ob tabeli časov.
- **TabICL na CIFAR_10 potrebuje ~400 GB** (MaxRSS 394 GiB): pri ponovitvi
  task 61 takoj oddaj na vozlišče z 512 GB z `--constraint=sxm --mem=480G`.
  Na vozlišču z 256 GB TabICL mehko odpove kot v `cc18_v2`, na vozlišču z
  512 GB in `--mem=240G` pa ubije opravilo.
- Izpis Pythona v dnevnik je med tekom prazen (medpomnjenje); ponovna oddaja s
  `PYTHONUNBUFFERED=1` ga izpisuje sproti.
