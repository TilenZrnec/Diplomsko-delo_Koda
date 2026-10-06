# subset_v3 — validacija nove zamrznitve (TabPFN-3.5) na Arnesu

Pilotna trojica (OpenML 31, 37, 38) na gruči Arnes z zamrznitvijo 2026-10-05
(okolje `tabular3.5`, tabpfn 9.1.0 = TabPFN-3.5). Namen: (1) izmeriti učinek
nove zamrznitve na istih 270 učenjih kot v `cc18_v2` (iste delitve in semena) in
(2) potrditi, da gruča vrne enake številke kot lokalni pilot `subset_v3_local`,
preden se odda polni CC18 (`cc18_v3`).

## Identifikacija (iz manifest.json)

| Postavka | Vrednost |
|---|---|
| Repo commit | `c503f48`, čisto delovno drevo |
| SLURM job ID | `20132190` (array, 3 taski) |
| Vozlišče | `gwn08.arnes.si`, NVIDIA H100 80GB HBM3, 8 jeder |
| Okolje | micromamba `~/envs/tabular3.5`, Python 3.12.13, torch 2.14.1+cu130, CUDA 13.0 |
| Uteži | `tabpfn-v3.5-20260909.safetensors`, `tabicl-classifier-v2-20260212.ckpt` |
| Datum | 2026-10-06 |
| Lokalna referenca | `results/runs/subset_v3_local` (Kremen, RTX 3060, WSL2) |

Paketi so enaki lokalnemu okolju `subset_v3_local`; razlikujejo se le orodja za
nameščanje (pip 26.2.1, setuptools 84.0.0, wheel 0.48.0, packaging 26.3 na
gruči), ki na numeriko ne vplivajo.

## Izid

`python scripts/merge_results.py subset_v3`: **270 vrstic, 3 nabori, 0 vrstic z
napako**, 0 vrstic z `raw_error` (TabPFN-3.5 in TabICL sta sprejela surovi vhod
povsod).

### Učinek nove zamrznitve: `compare_results.py cc18_v2 subset_v3`, 270/270 vrstic

| Algoritem | mean \|Δ\| | max \|Δ\| | Sodba |
|---|---|---|---|
| random_forest | 0 | 0 | bitno identično |
| xgboost | 0 | 0 | bitno identično |
| lightgbm | 0 | 0 | bitno identično |
| catboost | 0 | 0 | bitno identično |
| tabicl | 7e-6 | 6.1e-5 | ista knjižnica (tabicl 2.2.0); šum GPU in popravna izdaja torch |
| tabpfn | 5.0e-3 | 2.1e-2 | drug model (TabPFN-3 → TabPFN-3.5) |

TabPFN-3.5 proti TabPFN-3, povprečni ROC-AUC čez 15 učenj:

| Nabor | TabPFN-3 (`cc18_v2`) | TabPFN-3.5 (`subset_v3`) | Razlika | Učenj z višjim ROC-AUC |
|---|---|---|---|---|
| credit-g | 0.7937 | 0.8022 | +0.0086 | 14/15 |
| diabetes | 0.8398 | 0.8428 | +0.0030 | 11/15 |
| sick | 0.9981 | 0.9989 | +0.0009 | 14/15 |

Nova zamrznitev torej spremeni le TabPFN; razlike med `cc18_v2` in `cc18_v3`
bodo pri ostalih petih algoritmih na ravni šuma GPU ali nič.

### Prenos na gručo: `compare_results.py subset_v3_local subset_v3`, 270/270 vrstic

| Algoritem | mean \|Δ\| | max \|Δ\| | Sodba |
|---|---|---|---|
| random_forest, xgboost, lightgbm, catboost | 0 | 0 | bitno identično |
| tabicl | 8e-6 | 1.2e-4 | PASS (nedeterminizem GPU, RTX 3060 proti H100) |
| tabpfn | 3.8e-5 | 2.4e-4 | PASS (nedeterminizem GPU) |

Povprečja po naborih se ujemajo na 4 decimalke. Odstopanja so dva reda velikosti
pod odklonom med foldi (povprečni standardni odklon ROC-AUC čez 15 učenj na par
nabor × algoritem je 0.023) in enakega reda kot pri validaciji `subset_v2`
(TabPFN max 3.7e-4). **ALL PASS**, gruča je pripravljena za `cc18_v3`.

**Opomba o strojni opremi.** `cc18_v2` je v celoti tekel na H100 PCIe (`gwn01`,
`gwn03`–`gwn06`), ta zagon pa na H100 80GB HBM3 (`gwn08`). Omejitev
`--constraint=h100` dopušča obe izvedbi, zato lahko `cc18_v3` teče na obeh;
stolpec `device` pove izvedbo za vsako učenje.

## Potrdilo opravila (sacct)

Celotni izpis: `sacct.txt` v tej mapi
(`sacct -j 20132190 --format=JobID,JobName%20,Elapsed,MaxRSS,State,NodeList`).
Vsi trije taski `COMPLETED` na `gwn08`, 24–65 s na task.
