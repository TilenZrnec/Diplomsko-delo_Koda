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

Napovedi (`predictions/`, 540 datotek) so zaenkrat le na Arnesu; prenesemo jih
po drugem delu.

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

  job ID: (vpišemo ob prejemu).
