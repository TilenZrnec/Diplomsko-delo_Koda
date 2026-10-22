# cc18_leakage — test uhajanja podatkov na CC18, 2026-10-21

Tretji poskus diplome (CLAUDE.md, What is next, korak 8). Zasnova, utemeljitve in
viri so v `DESIGN.md`, ki je bil skupaj s kodo commitan **pred** zagonom
(`6444529`, popravek referenčnega pravila in čisti tehnični preizkus `1186a57`).
Referenca na naboru Medic3-160 je zagon `medic3_160_leakage` (svoj PROVENANCE.md).

## Identifikacija (iz manifest.json, results.csv in sacct.txt)

| Postavka | Vrednost |
|---|---|
| Repo commit | `1186a57` (vseh 1530 vrstic), čisto delovno drevo (`git_dirty: false`) |
| Oddaja | 2026-10-21 s `hpc-login3`; Arnes tik pred tem `0ad1a88` → `1186a57` (fast-forward) |
| SLURM job ID | `20724058`, `--array=0-33%4`, particija `gpu`, `--constraint=h100`, 8 jeder, 64G |
| Vozlišča | `gwn03` (H100 PCIe, 17 naborov), `gwn08` (H100 80GB HBM3, 16), `gwn10` (H100 80GB HBM3, numerai28.6); vse tri različice posameznega nabora na istem vozlišču in istem GPU (preverjeno) |
| Okolje | micromamba `~/envs/tabular3.5`, Python 3.12.13, CUDA 13.0; uteži `tabpfn-v3.5-20260909.safetensors`, `tabicl-classifier-v2-20260212.ckpt` |
| Čas | prvo učenje 2026-10-21 14:14, zadnje 15:09; vseh 34 taskov COMPLETED, skupaj 1,76 h; najdaljši Internet-Advertisements 13 min 50 s, Bioresponse 12 min 41 s, adult 10 min 13 s; MaxRSS ≤ 4,8 GiB (Bioresponse) |

## Izid

`python scripts/merge_results.py cc18_leakage` (na Arnesu): **1530 vrstic
(34 naborov × 3 različice × 3 algoritmi × 5 foldov), 0 z napako, 0 `raw_error`.**
Mediani časov na učenje: CatBoost 3,4 s, TabPFN-3.5 0,2 + 0,3 s, TabICL 0,4 + 0,2 s
(učenje + napoved).

## Preverbe (niso del testa)

- **Izvirna različica proti diplomskim rezultatom** (`cc18_v4/combined`, foldi 0–4,
  170 učenj na algoritem): TabPFN-3.5 enak v 167/170, največ 9,3e-5; TabICL enak v
  95/170, največ 4,4e-3 (razlika GPU, kot med `cc18_v2` in `cc18_v3`); CatBoost enak
  v 165/170 — ostalih 5 je vseh 5 foldov numerai28.6 (do 2,9e-3), ki je tu tekel na
  `gwn10`, v diplomskem zagonu pa na `gwn02`: **CatBoost na 96 000 vrsticah ni bitno
  ponovljiv med vozlišči** (na vseh ostalih 33 naborih je). Na test to ne vpliva, ker
  so vse različice nabora tekle na istem vozlišču.
- **Kontrola:** padec CatBoosta pri monotoni preslikavi je na 34 naborih mediana
  4,4e-7, največ 1,35e-3 — drevesa preslikava praktično ne spremeni.
- **Napovedi:** 1530 datotek `.npz`, kopirane na Kremen 2026-10-22;
  `scripts/compute_metrics.py cc18_leakage`: vsako učenje ima datoteko, ROC-AUC iz
  datotek se z `results.csv` ujema do 9,6e-6 → `metrics.csv` (točnost, log. izguba).

## Rezultat — vnaprej določena analiza

`python scripts/analyze_leakage.py cc18_leakage --reference medic3_160_leakage`
(izpis v `summary/leakage_report.txt`, tabele `summary/leakage_*.csv/.tex`).
Δ = padec ROC-AUC temeljnega modela minus padec CatBoosta (povprečje 5 foldov);
enostranski Wilcoxonov test čez 34 naborov, Holm znotraj modela; referenca:
enostranski popravljeni t-test, ali je Δ na Medic3 manjši od mediane Δ na CC18.

| Model | Različica | mediana Δ | Δ > 0 | p (Holm) | Δ Medic3 | p (Medic3 < mediana) | Sklep po pravilu |
|---|---|---|---|---|---|---|---|
| TabPFN-3.5 | monotona | +0,0026 | 29/34 | 7,7e-7 | −0,0002 | 8e-6 | skladno s pomnjenjem (ni dokaz) |
| TabPFN-3.5 | s šumom | +0,0025 | 23/34 | 0,011 | −0,0004 | 0,0006 | skladno s pomnjenjem (ni dokaz) |
| TabICL | monotona | +0,0013 | 27/34 | 2,8e-6 | −0,0012 | 0,0008 | skladno s pomnjenjem (ni dokaz) |
| TabICL | s šumom | +0,0028 | 23/34 | 0,0098 | +0,0059 | 0,998 | ne moremo ločiti od splošne občutljivosti |

Oba temeljna modela izgubita na spremenjenih kopijah naborov CC18 malo, a značilno
več kot CatBoost; na Medic3-160, ki ga nista mogla videti, TabPFN-3.5 ne izgubi
ničesar. Po vnaprej določenem pravilu je to **skladno s pomnjenjem, ni pa dokaz**:
referenca je en sam nabor, ki se od CC18 močno razlikuje (≈ 94 000 učnih vrstic,
82 % manjkajočih vrednosti, prednost TabPFN pred CatBoostom le +0,0016), in test ne
loči pomnjenja od posrednega vpliva (razvoj na javnih zbirkah) — DESIGN.md, točka 12.

## Raziskovalno (post hoc, NI del vnaprej določenega testa)

- **Velikost učinka:** povprečna prednost TabPFN-3.5 pred CatBoostom na 34 naborih
  je +0,0222 (izvirni), +0,0185 (monotona), +0,0163 (s šumom), torej se zmanjša za
  16 % oz. 26 %; TabPFN-3.5 ostane boljši na 33/34 oz. 32/34 naborov (Wilcoxon p
  1,6e-8 oz. 1,7e-7). Na Medic3-160: +0,0016, +0,0018, +0,0020.
- Δ (TabPFN, monotona) je povezan s prvotno prednostjo (Spearman ρ = +0,42, p =
  0,013), ne pa z velikostjo učne množice (ρ = −0,02).
- Največji Δ (TabPFN, monotona): kc1 0,011, GesturePhaseSegmentationProcessed 0,010,
  kc2 0,010, madelon 0,009, climate-model-simulation-crashes 0,009, pc4 0,009,
  vehicle 0,009, cylinder-bands 0,008.

## Kopije napovedi

Arnes (`results/runs/cc18_leakage/predictions/`), Kremen (2026-10-22, preverjeno);
USB disk in prenosnik: še kopirati (uporabnik).
