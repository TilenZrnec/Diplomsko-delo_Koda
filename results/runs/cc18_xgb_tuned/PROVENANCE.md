# cc18_xgb_tuned — poskus uglaševanja: XGBoost z iskanjem po mreži in gnezdenim prečnim preverjanjem

Četrti poskus diplome (CLAUDE.md, What is next, korak 7). Zagon vsebuje **le
algoritem `xgboost_tuned`** (`src/models/xgboost_tuned.py`): XGBoost, ki mu
hiperparametre na vsakem zunanjem foldu izbere iskanje po mreži. Primerja se s
privzetimi algoritmi iz `cc18_v4/combined/`.

- **Zunanji foldi** so isti kot v `cc18_v3` / `cc18_v4` (5 delitev × 3
  ponovitve, seme delitev 42, seme modela 42 + ponovitev).
- **Notranje preverjanje:** `StratifiedKFold(3, shuffle=True)` na učnem delu
  zunanjega folda (seme = seme modela). Zmaga nastavitev z največjim povprečnim
  notranjim ROC-AUC (pri enakosti tista, ki je prej v mreži: privzeta najprej,
  manj dreves najprej); nauči se na celotnem učnem delu in se enkrat oceni na
  testnem.
- **Mreža** (`config.yaml`, `tuning`; Probst, Boulesteix in Bischl 2019):
  `learning_rate` {0,3; 0,1; 0,03; 0,01} × `n_estimators` {100, 200, 500, 1000}
  × `min_child_weight` {1, 4} × `colsample_bylevel` {1; 0,5} × `subsample`
  {1; 0,8} = 128 nastavitev; privzeta (0,3 / 100 / 1 / 1 / 1) je ena od njih.
- **Predobdelava** enaka kot pri privzetem XGBoost (`xgboost_model.prepare()`:
  nativna obravnava NaN in kategorij).
- **Preverba v vsakem foldu:** isti proces nauči še privzeti XGBoost.
  `default_test_roc_auc` v stolpcu `hyperparameters` mora biti enak `cc18_v4`,
  `default_train_time_s` pa je izhodišče za ceno uglaševanja na istem vozlišču.
- `tuning/<id>/xgboost_tuned_r<r>_f<f>.csv` = notranji rezultati vseh 128
  nastavitev na foldu (stolpci mreže, `order` = mesto v mreži, `inner_1..3`,
  `inner_mean`, `rank`).

Nabori: `scripts/cc18_tuning_ids.json`, 68 naborov CC18 (brez mnist_784,
Devnagari-Script, CIFAR_10 in Fashion-MNIST zaradi cene); indeksi 0–35 = 36
nenasičenih, 36–67 = 32 nasičenih (po `cc18_v4/combined`). Oba dela tečeta pod
istim RUN_ID.

## Del 1: 36 nenasičenih naborov (indeksi 0–35)

| Postavka | Vrednost |
|---|---|
| Repo commit | `f7d8366` (vseh 540 vrstic), čisto delovno drevo |
| SLURM job ID | `20494570` (`--array=0-35%8`), oddan 2026-10-15 s `hpc-login4` |
| Vozlišča | privzeta CPU particija `all`, 13 vozlišč (`wn023`–`wn111`); 8 jeder, 32 GB, brez GPU (`device` = `cpu x8`) |
| Okolje | micromamba `~/envs/tabular3.5`, Python 3.12.13 (zamrznitev 2026-10-05) |
| Čas | prvo učenje 11:04, zadnje 15:54 (lokalni čas; 4 h 50 min); task od 3 min do 4 h 04 min (madelon), skupaj 26,9 h na 8 jedrih |
| Pomnilnik | MaxRSS največ 0,97 GiB |
| sacct | vseh 36 taskov COMPLETED (`sacct.txt`) |
| Izid | `merge_results.py`: **540 vrstic (36 × 15), 0 z napako**; 540 datotek v `tuning/` |
| Commit z Arnesa | `785257b` |

Napovedi (`predictions/`, 540 datotek) so bile do konca drugega dela le na
Arnesu; prenesene so skupaj z drugim delom (spodaj).

### Preverbe (2026-10-15, Kremen)

- **Privzeti XGBoost v vsakem foldu = `cc18_v4`:** enak na 415 od 540 foldov,
  na ostalih 125 razlika največ 1,1e-16 (zadnji bit). Tolikšne razlike ne more
  povzročiti drugačen vrstni red napovedi — ta spremeni ROC-AUC za vsaj
  1/(2·n_poz·n_neg), pri teh naborih ≥ ~1e-9 —, gre torej za zaokroževanje.
  Privzeti XGBoost se ponovi tudi med vozlišči `wn…` (ta zagon) in `gwn09`
  (`cc18_v4`).
- **Tabele `tuning/` so skladne z `results.csv`** na vseh 540 foldih: 128
  vrstic, vrstica z rangom 1 je izbrana nastavitev, rang in notranji rezultat
  privzete nastavitve (`order` 0) sta enaka zapisu v `hyperparameters`,
  `inner_mean` je povprečje treh notranjih foldov. Pozor: pandas `read_csv`
  brez `float_precision="round_trip"` lahko število prebere za 1 ulp narobe
  (0,9327026845901245 → …244); s to nastavitvijo je ujemanje točno.
- **Arnes = pilot na Kremenu** (`subset_xgb_tuned_local`, isti commit): na
  credit-g in diabetes isti zmagovalec v 30 od 30 foldov in enak testni ROC-AUC
  (razlika 0).
- Ni znakov zloma: najnižji ROC-AUC na foldu je 0,51 (dresses-sales) in 0,52
  (numerai28.6), kjer so vsi algoritmi blizu naključja.

### Rezultati na 36 naborih (dokončne številke po drugem delu)

- **Uglašeni proti privzetemu XGBoost** (isti zagon, isti foldi): boljši na 33
  od 36 naborov (462 od 540 foldov), povprečno +0,0141, mediana +0,0066,
  Wilcoxon p = 1,9e-8. Največ kc2 +0,054, blood-transfusion +0,049, diabetes
  +0,041, cmc +0,039; slabši na ilpd (−0,0088), Internet-Advertisements
  (−0,0007) in churn (−0,0004).
- **Proti ostalim algoritmom** (`cc18_v4/combined`, istih 36 naborov): povprečni
  ROC-AUC TabPFN-3.5 0,8869, TabICL 0,8830, CatBoost 0,8645, **uglašeni XGBoost
  0,8644**, LightGBM 0,8541, RandomForest 0,8511, privzeti XGBoost 0,8503;
  povprečni rangi (7 algoritmov) 1,25 / 2,08 / 3,78 / **3,92** / 5,33 / 5,72 /
  5,92. Wilcoxon + Holm (6 primerjav uglašenega XGBoost): slabši od TabPFN-3.5
  na 36 od 36 naborov (p_Holm 1,7e-10) in od TabICL na 33 (1,5e-7); proti
  CatBoost 15 : 21, razlika −0,0001, p_Holm 0,67 (ni značilno); boljši od
  LightGBM (31 : 5, 3,9e-6) in RandomForest (29 : 7, 4,7e-4). Povprečni
  zaostanek za TabPFN-3.5 se zmanjša z 0,0367 na 0,0225 (za 39 %).
- **Zmagovalne nastavitve** (540 foldov): `learning_rate` 0,01 / 0,03 / 0,1 /
  0,3 = 284 / 169 / 75 / 12; dreves 100 / 200 / 500 / 1000 = 150 / 135 / 131 /
  124; `subsample` 0,8 v 398, `colsample_bylevel` 0,5 v 395, `min_child_weight`
  1 v 360 foldih. Privzeta nastavitev je zmagala enkrat (mediana njenega
  notranjega ranga je 77 od 128).
- **Robovi mreže.** Najšibkejši kot (0,01; 100 dreves) je zmagal v 73 od 540
  foldov (13,5 %; v pilotu 6 od 45), skoraj le na majhnih šumnih naborih: kc2
  12 od 15, analcatdata_dmft 11, dresses-sales 8, cmc 8, blood-transfusion 7,
  numerai28.6 7. Tam je kot premagal najboljšo nastavitev zunaj kota za mediano
  0,0016 notranjega ROC-AUC (največ 0,0100), zadnji korak z 200 na 100 dreves
  pa je prinesel mediano 0,0023. TabPFN-3.5 je na teh šestih naborih pred
  uglašenim XGBoost za 0,009–0,030. Mreža za drugi del ostane nespremenjena
  (določena je bila pred zagonom); v nalogi to navedemo kot omejitev. Zgornji
  rob ne omejuje: kot z največ učenja (0,3; 1000) je zmagal trikrat.
- **Notranja ocena proti testni:** najboljši notranji ROC-AUC je v povprečju za
  0,0043 *nižji* od testnega, ker se notranji modeli učijo na 2/3 učnega dela.
- **Cena:** iskanje in ponovno učenje skupaj = 537- do 868-kratnik (mediana 652)
  enega privzetega učenja na istem vozlišču.

## Del 2: 32 nasičenih naborov (indeksi 36–67)

Ocena = čas privzetega XGBoost v `cc18_v4` × razmerje uglaševanje / `cc18_v4`
na večjih naborih prvega dela (340–1194, pri večrazrednih 340–823): isolet
(indeks 49) 20–69 h, har (54) 7–24 h, ostalih 30 skupaj 14–50 h, najdaljši
letter do 6,3 h.

Oddano 2026-10-15 s `hpc-login4`, **brez `git pull`** (Arnes na `785257b`, učna
pot enaka kot v `f7d8366`):

- job `20511472`: indeksi 36–48, 50–53, 55–67 (`%8`), privzeta particija `all`,
  `--time=12:00:00`;
- isolet (49) in har (54): prva oddaja z `--time=3-00:00:00` je bila zavrnjena
  ("Requested time limit is invalid"), ker particija `all` dovoli največ 2 dni
  (`sinfo -o "%P %l"`, 2026-10-15: `all` 2 dni, `long` 7, `gpu` 4,
  `mig-preempt` / `mig-priority` 1). Zato v particijo `long`:

      RUN_ID=cc18_xgb_tuned sbatch --array=49,54 --partition=long --time=4-00:00:00 scripts/run_tuning.sh

  job `20511665` (sprejet).

- isolet (49) v 4 dneh ni končal: `20511665_49` se je po 4-00:15:18 na `wn119`
  končal s TIMEOUT (pričakovano), z 11 od 15 foldov v
  `per_dataset/300__xgboost_tuned.csv.partial` (fold 10 končan 2026-10-19
  10:11). Nadaljevanje je uporabnik oddal s `hpc-login3`, Arnes še vedno na
  `785257b`:

      RUN_ID=cc18_xgb_tuned sbatch --array=49 --partition=long --time=4-00:00:00 \
          --dependency=afternotok:20511665_49 scripts/run_tuning.sh

  job `20652997`. Začel se je takoj po TIMEOUT (~2026-10-19 21:45, `wn116`), v
  dnevniku `najden partial: 11 učenj že narejenih, nadaljujem.`, preostale 4
  folde je naredil v 1-00:57:27. Izgubljeno je le delo na foldu 11 ob prekinitvi
  (~11,5 h na `wn119`; na `wn116` ga je ponovil v 6,3 h).

| Postavka | Vrednost |
|---|---|
| Repo commit | `785257b` (vseh 480 vrstic); učna pot enaka kot v `f7d8366` (`git diff f7d8366 785257b -- src/config.py src/data.py src/runner.py src/run_one_dataset.py src/utils.py src/models config.yaml scripts/run_tuning.sh` je prazen) |
| SLURM job ID | `20511472` (30 taskov), `20511665` (isolet, har; particija `long`), `20652997` (nadaljevanje isolet) |
| Vozlišča | 10 vozlišč: 30 taskov na osmih (`wn025`–`wn060`), har na `wn119`, isolet na `wn119` (foldi 0–10) in `wn116` (11–14); 8 jeder, 32 GB, brez GPU |
| Čas | prvo učenje 2026-10-15 20:25, zadnje 2026-10-20 22:42; taski joba `20511472` od 2 min (breast-w) do 5 h 00 min (letter), har 31 h 09 min, isolet 96 h 15 min + 24 h 57 min |
| Pomnilnik | MaxRSS največ 1,2 GiB |
| sacct | `20511472` 30/30 COMPLETED, `20511665_54` COMPLETED, `20511665_49` TIMEOUT (korak batch CANCELLED), `20652997_49` COMPLETED; `sacct.txt` zdaj zajema vse štiri jobe |
| Izid | `merge_results.py`: **1020 vrstic (68 × 15), 0 z napako**; 1020 datotek v `tuning/`, 1020 napovedi, nobene `.partial` ali `.tmp` |
| Commit z Arnesa | `0ad1a88` |

Napovedi vseh 68 naborov (1020 datotek, 32 MB) so 2026-10-21 prenesene na
Kremen, prenosnik in USB-ključ.

### Preverbe (2026-10-21, Kremen; vseh 68 naborov)

- **Popolnost:** 68 naborov × 15 foldov, brez podvojenih (nabor, ponovitev,
  fold); nabori = `scripts/cc18_tuning_ids.json`; indeksi 0–35 so po
  `cc18_v4/combined` (prag 0,995) vsi nenasičeni, 36–67 vsi nasičeni.
- **Isti zunanji foldi kot `cc18_v4`:** `n_train`, `n_test`, `n_features`,
  `n_categorical` in `n_classes` enaki v 1020 od 1020 vrstic.
- **Privzeti XGBoost v vsakem foldu = `cc18_v4`:** enak na 693 od 1020 foldov
  (drugi del 278 od 480), na ostalih razlika največ 1,1e-16 (zadnji bit, kot v
  prvem delu).
- **Tabele `tuning/` so skladne z `results.csv`** na vseh 1020 foldih: 128
  vrstic s celotno mrežo, rangi 1–128, zmagovalec = največji `inner_mean` (ob
  izenačenju manjši `order`), vrstica `order` 0 = privzete vrednosti, njen rang
  in notranji rezultat = zapis v `hyperparameters`, `inner_mean` = povprečje
  treh notranjih foldov.
- **Napovedi:** `scripts/compute_metrics.py cc18_xgb_tuned` — 1020 od 1020
  datotek, ROC-AUC iz datotek = `results.csv` do 1,1e-16 (XGBoost vrača
  verjetnosti v float32, zato zapis v float32 ničesar ne izgubi), vsote vrstic
  do 9e-8; zapisana `metrics.csv` in `metrics_info.json`.
- **Arnes = pilot na Kremenu** zdaj tudi na sick (nasičen, drugi del): isti
  zmagovalec in testni ROC-AUC v 15 od 15 foldov (s credit-g in diabetes 45 od
  45).
- Ni znakov zloma: najnižji ROC-AUC na foldu ostaja dresses-sales (0,51) in
  numerai28.6 (0,52); največji padec uglašenega proti privzetemu na enem foldu
  je na ilpd (−0,055 pri testnem foldu s 116–117 primeri).

### Rezultati na vseh 68 naborih

- **Uglašeni proti privzetemu XGBoost:** boljši na 61 od 68 naborov (846 od
  1020 foldov), povprečno +0,0083, mediana +0,0016, Wilcoxon p = 1,6e-10.
  Nenasičeni (36): nespremenjeno +0,0141 (33 od 36). Nasičeni (32): +0,0018,
  mediana +0,0003, boljši na 28 (p = 2,0e-6). Največ na nasičenih balance-scale
  +0,029 (0,932 → 0,961; TabPFN-3.5 0,9996 — nasičenost pomeni, da 0,995 doseže
  najboljši algoritem, ne nujno XGBoost) in electricity +0,013; slabši wdbc
  (−0,0005), splice, kr-vs-kp in car (vsi nad −0,0001). Skupaj slabši na 7
  naborih, opazno le na ilpd (−0,0088, slabši v 11 od 15 foldov).
- **Proti ostalim algoritmom** (`cc18_v4/combined`, istih 68): povprečni
  ROC-AUC TabPFN-3.5 0,9395, TabICL 0,9370, **uglašeni XGBoost 0,9260**,
  CatBoost 0,9255, LightGBM 0,9193, privzeti XGBoost 0,9177, RandomForest
  0,9160; povprečni rangi (7 algoritmov) 1,44 / 1,96 / **3,85** / 3,81 / 5,32 /
  5,63 / 5,99. Wilcoxon + Holm (6 primerjav uglašenega XGBoost): slabši od
  TabPFN-3.5 na 66 od 68 (p_Holm 8,5e-12) in od TabICL na 62 (1,3e-9); proti
  CatBoost 30 : 38, razlika +0,0005, p_Holm 0,52 (ni značilno); boljši od
  LightGBM (58 : 10, 5,9e-8), RandomForest (57 : 11, 4,7e-7) in privzetega
  XGBoost (61 : 7, 7,9e-10). Pred TabPFN-3.5 je le na tic-tac-toe in
  wall-robot-navigation (razlika pod 0,0004). Zaostanek za TabPFN-3.5 se
  zmanjša z 0,0218 na 0,0135 (za 38 %; nenasičeni 39 %, nasičeni 35 %).
- **Dodatni metriki** (iz napovedi): točnost 0,8664 → 0,8731 (boljša na 57 od
  68, p = 1,7e-8), logaritemska izguba 0,3745 → 0,3016 (boljša na 65 od 68 in
  na vseh 36 nenasičenih, p = 1,6e-12), čeprav je izbira optimizirala ROC-AUC:
  zmagovalci imajo manjšo stopnjo učenja in podvzorčenje, privzeti XGBoost
  (0,3; 100 dreves) pa daje bolj skrajne verjetnosti. Povprečni rang (7) po
  točnosti 4,13 (CatBoost 4,12), po izgubi 3,71 (CatBoost 3,74). Izguba je
  slabša le na kr-vs-kp, wall-robot-navigation in wdbc.
- **Zmagovalne nastavitve** (1020 foldov): `learning_rate` 0,01 / 0,03 / 0,1 /
  0,3 = 335 / 361 / 231 / 93; dreves 100 / 200 / 500 / 1000 = 197 / 205 / 241 /
  377; `subsample` 0,8 v 781, `colsample_bylevel` 0,5 v 762, `min_child_weight`
  1 v 791 foldih; privzeta nastavitev je zmagala dvakrat (mediana njenega
  notranjega ranga 65 od 128). Polovici se razlikujeta: nenasičeni nabori
  hočejo najmanj učenja (0,01 v 284 od 540, kot (0,01; 100) 73-krat), nasičeni
  več (1000 dreves v 253 od 480, najpogosteje 0,03 s 1000 drevesi, 130-krat;
  kot (0,01; 100) nikoli). Smer se ujema z optimalnimi privzetimi vrednostmi
  Probsta in sod. (2019, tabela 3: eta 0,018, subsample 0,839,
  colsample_bylevel 0,585).
- **Robovi mreže:** pri zmagovalcih v kotu (0,01; 100) je zadnji korak z 200 na
  100 dreves prinesel mediano 0,0023 notranjega ROC-AUC (73 foldov, največ
  0,019). Pri zmagovalcih s 1000 drevesi je korak s 500 na 1000 na nasičenih
  prinesel mediano le 0,00007 (253 foldov, 90. percentil 0,0006; plato), na
  nenasičenih pa mediano 0,0023 (124 foldov; pri 0,01 in 1000 drevesih 0,0025,
  70 foldov) — tam krivulja še raste. Najmočnejši kot (0,3; 1000) je zmagal
  28-krat. Širša mreža bi torej na teh foldih prinesla kvečjemu nekaj tisočink,
  kar sklepov ne spremeni (zaostanek za TabPFN-3.5 je 0,0135).
- **Notranja ocena proti testni:** na nenasičenih je notranje preverjanje
  obetalo +0,0198 proti privzetemu, test je dal +0,0141; na nasičenih +0,0022
  proti +0,0018 (optimizem izbire najboljše od 128 šumnih ocen). Na ilpd je bil
  notranji dobiček pozitiven v 15 od 15 foldov (0,008–0,048), testni negativen
  v 11.
- **Primerjava z literaturo:** Probst in sod. (2019, tabela 2) navajajo za
  XGBoost povprečno nastavljivost glede na privzete vrednosti paketa 0,043 ±
  0,006 (AUC, 38 binarnih naborov OpenML100 brez manjkajočih vrednosti). To je
  razlika do najboljše nastavitve na posameznem naboru (enačba 4), poiskane na
  nadomestnem modelu s 100 000 naključnimi točkami v širšem prostoru (tudi
  `booster`, `max_depth`, do 5000 dreves), brez šuma izbire — zgornja meja.
  Naš dobiček z gnezdenim preverjanjem (+0,0083, nenasičeni +0,0141) je zato
  pričakovano manjši.
- **Cena:** iskanje in ponovno učenje = mediana 654-kratnika (nenasičeni) oz.
  714-kratnika (nasičeni) enega privzetega učenja na istem vozlišču (razpon
  426–2434), skladno z grobo oceno 32 kombinacij × 3 notranji foldi × 10-krat
  več dreves × 2/3 podatkov ≈ 640. Seštevek časov uglaševanja 26,8 h + 170,3 h;
  vsi taski po sacct 208,7 h na 8 jedrih (≈ 1670 jedrnih ur, z izgubljenim
  foldom isolet). Isolet (109,8 h) in har (31,1 h) sta bila daleč nad oceno,
  čeprav je bilo njuno razmerje običajno (620 in 582): privzeto učenje (8 jeder,
  kot v `cc18_v4`) je na `wn119` / `wn116` trajalo 2,6-krat dlje kot na `gwn09`,
  od koder so bili časi za oceno, na `wn023`–`wn060` pa 0,74- do 1,31-krat
  (mediane po vozliščih; `wn109` 1,5, `wn111` 2,3). Ocena hitrosti vozlišča ni
  upoštevala.
