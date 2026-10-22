# medic3_160_leakage — referenca testa uhajanja na Medic3-160, 2026-10-21/22

Referenčni del testa uhajanja (CLAUDE.md, What is next, korak 8). Zasnova, viri in
rezultat testa so v `../cc18_leakage/DESIGN.md` in `../cc18_leakage/PROVENANCE.md`.
Medic3-160 ni bil nikoli objavljen, zato ga temeljna modela nista mogla videti.

## Identifikacija (iz manifest.json, results.csv in sacct.txt)

| Postavka | Vrednost |
|---|---|
| Repo commit | `1186a57` (vseh 45 vrstic), čisto delovno drevo |
| Podatki | `data/medic3/Medic3.csv`, sha256 `27b23cbfa0511850…` (enak kot v `medic3_160`), `max_classes: 160`, 117 825 vrstic, 275 številskih atributov; preslikanih vseh 275 |
| TabPFN-3.5 + TabICL | job `20724059` (`run_leakage.sh`, `--array=0 --mem=160G`), `gwn10` (H100 80GB HBM3), vse tri različice zaporedoma, 42 min 12 s, MaxRSS 92,9 GiB (TabICL; zahteva 160G je bila potrebna) |
| CatBoost | job `20724060` (`run_leakage_cpu.sh`, `--array=0-2`, particija `all`, 8 jeder, 32G): task 0 original na `wn060` 9 h 05 min, task 1 monotona na `wn050` 9 h 07 min, task 2 s šumom na `wn052` 9 h 59 min; MaxRSS ≤ 4,1 GiB |
| Čas | prvo učenje 2026-10-21 14:13, zadnje 2026-10-22 00:12; vsi taski COMPLETED |

## Izid in preverbe

`python scripts/merge_results.py medic3_160_leakage`: **45 vrstic (3 različice × 3
algoritmi × 5 foldov), 0 z napako.** Mediana CatBoost 6614 s na fold (1,8 h),
TabPFN-3.5 in TabICL ~75 s napovedi na fold.

- Izvirna različica je enaka diplomskemu zagonu `medic3_160_v2/combined` na foldih
  0–4: CatBoost in TabPFN-3.5 5/5 na bit enako, TabICL do 1,9e-8. CatBoost je torej
  na Medic3 ponovljiv med vozlišči `wn0xx` in `gwn0x`, zato različice, ki so tekle
  na treh različnih CPU vozliščih, smemo primerjati.
- Napovedi: 45 datotek `.npz` (579 MB) na Kremnu (2026-10-22); `compute_metrics.py`:
  vsako učenje ima datoteko, ROC-AUC iz datotek se ujema do 1,8e-9.

## Δ po foldih (padec temeljnega modela minus padec CatBoosta)

| Model | Različica | foldi 0–4 | povprečje |
|---|---|---|---|
| TabPFN-3.5 | monotona | −0,0000 −0,0004 −0,0003 −0,0001 −0,0000 | −0,0002 |
| TabPFN-3.5 | s šumom | −0,0005 −0,0012 −0,0004 +0,0000 +0,0002 | −0,0004 |
| TabICL | monotona | −0,0018 −0,0010 −0,0015 −0,0007 −0,0007 | −0,0012 |
| TabICL | s šumom | +0,0053 +0,0050 +0,0060 +0,0066 +0,0066 | +0,0059 |

## Kopije napovedi

Arnes, Kremen (2026-10-22, preverjeno); USB disk in prenosnik: še kopirati
(uporabnik). Napovedi Medic3 so zaupne — nikoli na GitHub.
