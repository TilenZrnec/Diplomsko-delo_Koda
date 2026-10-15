# cc18_v4 — RandomForest in XGBoost "kot iz škatle" (delni ponovni zagon, 2026-10-14)

Popravek protokola (CLAUDE.md, Preprocessing policy → Audit): vsak algoritem teče
s privzetimi nastavitvami in le s predobdelavo, brez katere knjižnica vhoda ne
sprejme. Revizija 2026-10-14 je pokazala, da sta dva modula predobdelovala
preveč: **XGBoost** je dobival ordinalne kode, čeprav xgboost 3.4.1 privzeto
(`enable_categorical=True`) sam obravnava stolpce tipa `category`;
**RandomForest** je manjkajoče vrednosti nadomeščal, čeprav jih scikit-learn
1.9.1 (od 1.4) obravnava sam. Popravek je v commitu `e3140ce`,
`scripts/audit_defaults.py` potrdi, da zdaj vseh šest algoritmov teče kot iz
škatle. Ta zagon vsebuje **le RandomForest in XGBoost**; LightGBM, CatBoost,
TabPFN-3.5 in TabICL ostanejo iz `cc18_v3`, ker se njihova koda ni spremenila.

## Identifikacija (iz manifest.json in results.csv)

| Postavka | Vrednost |
|---|---|
| Repo commit | `8b601d1` (vseh 2160 vrstic), čisto delovno drevo; modeli enaki kot v `e3140ce` |
| SLURM job ID | `20470896` (68 običajnih naborov, `%4`), `20470897` (indeksi 27, 60, 61, 70, `--mem=120G`, 1 dan) |
| Vozlišča | `gwn09` (70 taskov, H100 SXM), `gwn03` (2 taska, H100 PCIe); oba algoritma tečeta le na CPU, 8 jeder (`device` = `cpu x8`); GPU je bil rezerviran, a neuporabljen, da zagon teče na istih vozliščih kot `cc18_v3` |
| Okolje | micromamba `~/envs/tabular3.5`, Python 3.12.13 (zamrznitev 2026-10-05, isto kot `cc18_v3`) |
| Oddaja | 2026-10-14 s `hpc-login4`; prvo učenje 21:52, zadnje 2026-10-15 01:42 (lokalni čas) |
| Protokol | 72 naborov CC18, 5 delitev × 3 ponovitve, seme delitev 42, seme modela 42 + ponovitev (enako kot `cc18_v3`) |

## Izid

`python scripts/merge_results.py cc18_v4`: **2160 vrstic (72 × 2 × 15), 0 z napako.**

## Preverba proti `cc18_v3` (2026-10-15, Kremen)

Na istih foldih (velikosti učnih in testnih delov enake v vseh 2160 parih):

- **XGBoost na 49 naborih brez kategoričnih stolpcev: razlika 0 v vseh 735
  učenjih** (vhod je enak kot prej).
- **RandomForest na 63 naborih brez manjkajočih vrednosti:** 22 od 945 učenj se
  razlikuje, največ 1,3e-4 — znana nedeterminističnost vzporednega
  `predict_proba` (cc18_v2 proti cc18_v3 z enako kodo: 24 od 945, isti nabori).
  Največja razlika (blood-transfusion, 2 učenji) je natanko 1/(2·35·114) =
  1,2531e-4: en izenačen par pozitivne in negativne testne vrstice (35 : 114) je
  po zaokrožitvi urejen drugače.
- **Spremembe le tam, kjer jih popravek povzroči:**
  - XGBoost na 23 naborih s kategoričnimi stolpci: boljši na 12, slabši na 8,
    nespremenjen na 3 (dna, Internet-Advertisements, ilpd — kategorični stolpci
    z dvema vrednostma, kjer je nativna delitev enaka delitvi po kodi);
    povprečna sprememba −0,0003 (največ credit-g −0,0112, dresses-sales
    +0,0100, connect-4 +0,0076).
  - RandomForest na 9 naborih z manjkajočimi vrednostmi: boljši na 6, slabši na
    3; povprečno −0,0010 (dresses-sales −0,0198, cylinder-bands +0,0079).

## Celoten nabor rezultatov za diplomo: `combined/`

`python scripts/combine_runs.py cc18_v4 --base cc18_v3` → `combined/results.csv`
(6480 vrstic: RandomForest in XGBoost iz `cc18_v4`, ostali štirje iz `cc18_v3`;
`combined/sources.json` pove vir). Skript pred tem preveri enake nastavitve
delitev, nabore in velikosti foldov. Povzetek in statistika v
`combined/summary/` (`src.summary`, `src.stats`, tudi `--no-saturated`).

Učinek na ugotovitve (proti `cc18_v3`): vrstni red povprečnih rangov ostane
TabPFN-3.5 > TabICL > CatBoost > LightGBM > XGBoost > RandomForest (vseh 72);
povprečni ROC-AUC XGBoost 0,9209 → 0,9208, RandomForest 0,9185 → 0,9184;
Friedman/Iman–Davenport še vedno zavrne H0 (F_F 160,05 → 164,15). Wilcoxon–Holm:
v vsaki od obeh analiz se spremeni le en par — LightGBM proti RandomForest (vseh 72) ni več
značilen (p_Holm 0,042 → 0,057), LightGBM proti XGBoost (37 nenasičenih) postane
značilen (p_Holm 0,082 → 0,026). Na 37 nenasičenih RandomForest in XGBoost
zamenjata mesti po povprečnem rangu (4,95 proti 4,97), po povprečnem ROC-AUC je
XGBoost še vedno pred (0,8516 proti 0,8511).

## Napovedi

`predictions/` (2160 datotek `.npz`) ni v gitu. Kopije: Arnes,
Kremen, prenosnik in USB-ključ (2026-10-15). Preverjeno na Kremnu: vsako učenje
ima datoteko, nobene odvečne, foldi vsake ponovitve razdelijo vse vrstice,
vsi algoritmi imajo na istem foldu iste vrstice, ROC-AUC iz datotek se z
`results.csv` ujema (XGBoost ≤ 1e-16, RandomForest ≤ 3,2e-4 — datoteke hranijo float32).

## Opravila SLURM

| Task | Nabor | Trajanje | MaxRSS | Vozlišče | Izid |
|---|---|---|---|---|---|
| 20470897_60 | Devnagari-Script | 3:28:44 | 4,1 GiB | gwn03 | COMPLETED |
| 20470897_61 | CIFAR_10 | 1:14:47 | 4,8 GiB | gwn09 | COMPLETED |
| 20470897_27 | mnist_784 | 0:21:34 | | gwn03 | COMPLETED |
| ostalih 69 | | skupaj ~0,6 h | ≤ 4,8 GiB | gwn09 | COMPLETED |

Vseh 72 taskov COMPLETED, skupaj 5,7 h, brez ponovnih oddaj. Celotni izpis: `sacct.txt`.
