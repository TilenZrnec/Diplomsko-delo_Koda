# cc18_v2 — polni zagon OpenML-CC18 z nadgrajenim okoljem in tremi ponovitvami

Rezultat, na katerem temelji diploma. Nadomešča arhivski zagon
`results/arnes/cc18/` (avgust 2026, stara koda in stare različice knjižnic).

## Identifikacija (iz manifest.json in results.csv)

| Postavka | Vrednost |
|---|---|
| Repo commit | `dc1aeb2` (vseh 6480 vrstic) |
| SLURM job ID | `19100043` (68 običajnih naborov), `19100044` (4 veliki nabori) |
| Vozlišča | `gwn01`, `gwn03`, `gwn04`, `gwn05`, `gwn06`; NVIDIA H100 PCIe, 8 jeder na opravilo |
| Okolje | micromamba `~/envs/tabular2`, Python 3.12.14, torch 2.14.0+cu130, CUDA 13.0 |
| Oddaja | 2026-09-11 (manifest ustvarjen 20:05:09+02:00) |
| Protokol | 72 naborov × 6 algoritmov × 5 delitev × 3 ponovitve; seme delitev 42, seme modela 42 + ponovitev |

Različice knjižnic: `summary/razlicice.tex` (samodejno iz manifest.json).

## Izid

- `python scripts/merge_results.py cc18_v2`: **6480 vrstic, 72 naborov, 30 vrstic z napako**, brez `.partial`.
- Vsak algoritem ima 1080 vrstic. Naprava: `cpu x8` 4320 vrstic (drevesni ansambli), `cuda: NVIDIA H100 PCIe` 2160 (TabPFN, TabICL).
- **Neuspela učenja (30):** samo CIFAR_10 (OpenML 40927), vseh 15 učenj:
  - TabPFN: `Number of features 3072 ... greater than the maximum number of features 2000 officially supported` — omejitev modela.
  - TabICL: `CPU memory allocation failed ... DefaultCPUAllocator: can't allocate memory` — preseže pomnilnik vozlišča (240 G dodeljenih).
  Oba sta poskusila surov vhod (`raw failed:`); subvzorčenje ali izbor atributov nista bila uporabljena, protokol privzetih nastavitev ostaja nedotaknjen.

### Primerjava z arhivom (2026-08, 1 ponovitev, stare različice)

| Algoritem | povprečni rang 2026-08 | povprečni rang cc18_v2 |
|---|---|---|
| tabicl | 1.58 | 1.576 |
| tabpfn | 1.88 | 1.896 |
| catboost | 3.38 | 3.326 |
| lightgbm | 4.50 | 4.417 |
| xgboost | 4.59 | 4.618 |
| random_forest | 5.07 | 5.167 |

Friedman p = 8.91e-48 (prej 5e-47), Nemenyi CD = 0.889 (prej 0.89), nasičenih
naborov 33 (prej 33). Vrstni red algoritmov je nespremenjen.

## Opravila SLURM

| Job ID | Array | --mem | --time | Izid |
|---|---|---|---|---|
| 19100043 | 0-26,28-59,62-69,71%4 | 64G | 12:00:00 | 68/68 COMPLETED |
| 19100044 | 27,60,61,70 | 240G | 1-12:00:00 | 4/4 COMPLETED |

Nobeno opravilo ni bilo ponovno oddano ali prekinjeno.

Poraba (MaxRSS iz `sacct`, KiB pretvorjeno v GiB):

| Task | Nabor | Trajanje | MaxRSS | Vozlišče |
|---|---|---|---|---|
| 19100043 (68 taskov) | običajni | 0:33 do 2:09:04 | največ 3.8 GiB | gwn01/03/06 |
| 19100044_27 | mnist_784 (554) | 4:12:03 | 130.6 GiB | gwn03 |
| 19100044_60 | Devnagari-Script (40923) | 1-10:53:41 | 173.3 GiB | gwn03 |
| 19100044_61 | CIFAR_10 (40927) | 18:46:47 | 41.1 GiB | gwn04 |
| 19100044_70 | Fashion-MNIST (40996) | 4:35:50 | 136.9 GiB | gwn05 |

Celotni izpis: `sacct.txt` v tej mapi
(`sacct -j 19100043,19100044 --format=JobID,JobName%20,Elapsed,MaxRSS,State,NodeList`).

## Opombe za ponovitev

- **Devnagari-Script je porabil 34 h 54 min od 36 h omejitve.** Pri ponovnem
  zagonu zahtevaj vsaj `--time=2-00:00:00`; checkpointi sicer preprečijo izgubo
  dela, a prekinitev bi pomenila dodatno čakanje v vrsti.
- **Pomnilnik je z novimi različicami večji kot avgusta.** mnist_784 in
  Fashion-MNIST sta avgusta uspela s 120 G, zdaj sta dosegla 131 in 137 GiB,
  torej 120 G ne zadošča več. 240 G je potreben za vse štiri velike nabore.
- **Čas večinoma porabi CatBoost** (vsota `train_time_s` + `inference_time_s`
  čez 15 učenj): na CIFAR_10 14.2 h od 18.8 h (ocena je bila ~11 h), na
  Devnagari-Script 28.5 h od 34.9 h. Nizek MaxRSS na CIFAR_10 (41 GiB) je
  posledica tega, da TabICL pade ob prvem velikem zahtevku za pomnilnik, ne da
  bi ga dobil.
- Časi 68 običajnih naborov so iz različnih vozlišč (gwn01/03/06, vsa H100),
  velikih štirih pa iz gwn03/04/05; časi med nabori niso strogo primerljivi.
