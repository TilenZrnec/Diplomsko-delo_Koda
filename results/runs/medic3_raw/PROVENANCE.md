# medic3_raw — zaupni nabor Medic3, vseh 220 razredov (zamrznitev 2026-10-05)

Ena od dveh različic Medic3 (druga je `medic3_160`, omejena na 160 razredov,
kjer teče tudi TabPFN-3.5). Isti protokol kot `cc18_v3`, le en nabor.

## Identifikacija (iz manifest.json in results.csv)

| Postavka | Vrednost |
|---|---|
| Repo commit | `420bdac` (vseh 90 vrstic), čisto delovno drevo |
| SLURM job ID | `20318167` (RF, XGBoost, LightGBM), `20318168` (TabPFN, TabICL), `20318169` (CatBoost) |
| Vozlišče | vsa opravila na `gwn08` (H100 80GB HBM3, 512 GB); 8 jeder na opravilo |
| Okolje | micromamba `~/envs/tabular3.5`, Python 3.12.13, torch 2.14.1+cu130 |
| Uteži | `tabpfn-v3.5-20260909.safetensors`, `tabicl-classifier-v2-20260212.ckpt` (kot `cc18_v3`) |
| Podatki | `data/medic3/Medic3.csv`, sha256 `27b23cbfa0511850c1b030482a0cc08d3c7ea3d7862eee64f18f6b40d2a5c005` — enak na Kremnu in na Arnesu (preverjeno ob oddaji in znova 2026-10-12 na Arnesu) |
| Oddaja | 2026-10-10 ~10:45 s `hpc-login4` (manifest 11:08:20 UTC), konec 2026-10-11 22:22 UTC (zadnje učenje CatBoost) |
| Protokol | 122 093 vrstic × 275 atributov (vsi numerični), 220 razredov; 5 delitev × 3 ponovitve; seme delitev 42, seme modela 42 + ponovitev; en algoritem na SLURM task |

## Izid

`python scripts/merge_results.py medic3_raw`: **90 vrstic, 15 z napako** — vseh
15 učenj TabPFN-3.5 z razlogom `raw failed: Number of classes `220` exceeds the
maximum number of classes `160` officially supported by TabPFN.; fallback failed:
…` (pričakovano: meja 160 razredov, izmerjena 2026-10-06). Makro-OvR je v vsakem
učenju zajel vseh 220 razredov (`n_classes_scored = n_classes`).

| Algoritem | ROC-AUC (15 učenj) | mediana učenja | mediana napovedovanja | naprava |
|---|---|---|---|---|
| catboost | 0.9551 ± 0.0010 | 8143 s | 0.30 s | cpu x8 |
| tabicl | 0.9499 ± 0.0016 | 10.3 s | 79.8 s | H100 HBM3 |
| xgboost | 0.9306 ± 0.0023 | 152 s | 2.4 s | cpu x8 |
| random_forest | 0.8149 ± 0.0031 | 13.1 s | 0.60 s | cpu x8 |
| lightgbm | **0.5028 ± 0.0048** | 60 s | 1.2 s | cpu x8 |
| tabpfn (TabPFN-3.5) | — (15 × meja razredov) | | | H100 HBM3 |

`python -m src.stats medic3_raw` (en nabor → popravljeni t-test za ponovljeno
prečno preverjanje, Bouckaert in Frank 2004, Holm čez 10 parov; TabPFN izključen
zaradi neuspeha): **vseh 10 parov je značilnih**, najšibkejši CatBoost proti
TabICL (+0.0052, 15 : 0 učenj, p_Holm = 5.2e-06). Vrstni red: CatBoost > TabICL >
XGBoost > naključni gozd > LightGBM. `summary/overall.csv` ima v stolpcu
`mean_roc_auc_complete` NaN, ker TabPFN ni uspel (pravilo: povprečje le čez nabore,
kjer so uspeli vsi); številke so v `summary/per_dataset.csv`.

## LightGBM: privzete nastavitve divergirajo (ni napaka naše kode)

LightGBM je propadel v **vseh 15 učenjih (ROC-AUC 0.4994–0.5125)**, brez napake
ali izjeme. Vzrok je preverjen s `scripts/diagnose_lightgbm_medic3.py` na foldu 0
(rezultati v `diagnostics/`, prenosnik, 8 niti; datoteke `lightgbm_4.7.0_*`
so 2026-10-12 nastale iz commita `a5a2197`, datoteke `lightgbm_4.6.0_*` isti dan z
delovno različico skripta pred tem commitom — del brez naše kode je v obeh enak;
kaj je naredilo katero datoteko, piše v `*_info.json`):

- **Neodvisna pot brez naše kode** (pandas, sklearnov `RepeatedStratifiedKFold`,
  nativni `lgb.train` s privzetimi parametri, `sklearn.roc_auc_score`) da isti
  propad: učna izguba (multi_logloss, računa jo LightGBM sam) **raste** od 4.91 po
  prvi iteraciji do ~32.5; največja absolutna vrednost lista naraste s 129 v prvi
  iteraciji na red velikosti 10^6; napovedi so skoraj gotove (povprečna največja
  verjetnost 0.99999) in napačne (točnost ~0.057 ≈ delež največjega razreda
  0.061); ROC-AUC ~0.50. Po nekaj deset iteracijah učenje obstane (ni več
  dovoljene delitve), LightGBM pa izpiše le opozorili `No further splits with
  positive gain` in `Stopped training because there are no more leaves that meet
  the split requirements` — o naraščajoči izgubi ne poroča.
- **Ista pot kot zagon** (`src.models.lightgbm_model.run`): učna izguba je **enaka
  neodvisni poti v prvih iteracijah**, dokler se listi ne razletijo (razlika ≤ 1e-12;
  v prvem poskusu 8 iteracij, v ponovitvi iz commita le 2 — največji list je že v
  2. iteraciji ~2000; stolpec
  `identical_iterations_vs_independent` v `diagnostics/lightgbm_4.7.0_benchmark.csv`),
  potem se poti razideta. Modela sta torej ista; razhajanje po eksploziji je
  numerično: LightGBM privzeto ni bitno ponovljiv (`deterministic = false`),
  divergenca pa ojača razlike v zaokroževanju — že dva enaka klica na istem
  stroju data različen ROC-AUC (stolpca `roc_auc_module` in `roc_auc_replay`),
  Arnes je dal 0.5000. **Točne številke propadlega LightGBM zato niso ponovljive,
  propad pa je** (vsak poskus: ROC-AUC 0.50–0.52).
- **Ena sama sprememba** na istem foldu učenje ustali: `reg_lambda=1` → ROC-AUC
  0.938, `min_child_weight=1` → 0.938, učna izguba pada (do ~0.3), največji list
  ≤ 14. To sta privzeti vrednosti XGBoost 3.4.1 (`lambda` 1, `min_child_weight` 1);
  CatBoost 1.2.10 ima `l2_leaf_reg` 3; LightGBM 4.7.0 ima `reg_lambda` 0.0 in
  `min_child_weight` 0.001 (prebrano iz nameščenih knjižnic, `lightgbm_*_info.json`).
- **LightGBM 4.6.0** (okolje `tabularOriginal`) se obnaša enako (ROC-AUC ~0.50,
  izguba 4.91 → ~32.5): ne gre za napako zadnje različice.
- Mehanizem (naša razlaga, skladna z meritvami): vrednost lista je −G/(H+λ);
  pri softmaxu z 220 razredi je hessian vrstice p(1−p) za redke razrede blizu 0,
  zato z λ = 0 in minimalno vsoto hessiana 10⁻³ listi z malo vrsticami dobijo
  ogromne vrednosti, ki model potisnejo v gotove napačne napovedi.

Diploma poroča privzeti rezultat (odločitev uporabnika 2026-10-12, možnost A);
vrednosti z zavoro so le diagnostika, ne rezultat LightGBM.

## Opravila SLURM

| Task | Algoritem | --mem | --time | Trajanje | MaxRSS | Izid |
|---|---|---|---|---|---|---|
| 20318167_0 | random_forest | 64G | 1-00:00:00 | 0:03:54 | 12.7 GiB | COMPLETED |
| 20318167_1 | xgboost | 64G | 1-00:00:00 | 0:39:13 | 2.2 GiB | COMPLETED |
| 20318167_2 | lightgbm | 64G | 1-00:00:00 | 0:15:45 | 2.1 GiB | COMPLETED |
| 20318168_4 | tabpfn | 160G | 1-00:00:00 | 0:00:28 | 0.0 GiB | COMPLETED (15 × meja razredov) |
| 20318168_5 | tabicl | 160G | 1-00:00:00 | 0:23:42 | 99.4 GiB | COMPLETED |
| 20318169_3 | catboost | 64G | 3-00:00:00 | 1-09:48:20 | 4.7 GiB | COMPLETED |

Celotni izpis: `sacct.txt`. Brez ponovnih oddaj.

## Napovedi

`predictions/` (75 datotek `.npz` — neuspela učenja TabPFN datoteke nimajo —
~892 MiB) **ni v gitu in ne sme biti** (zaupno). Kopije: Arnes (mapa zagona),
prenosnik (preverjeno 2026-10-12: vsako uspelo učenje ima datoteko; ROC-AUC iz
datotek se z `results.csv` ujema na ≤ 5.5e-9 pri vseh algoritmih razen LightGBM,
kjer je razlika ≤ 7.5e-4, ker so njegove verjetnosti v float32 skoraj vse točno 0
in vezi spremenijo vrstni red) in uporabnikov USB-ključ (2026-10-12). Kremen: še ne.

## Opombe za ponovitev

- **TabICL potrebuje ~100 GiB RAM** (MaxRSS 99.4 GiB): `--mem=160G` je bil nujen,
  s privzetimi 64G bi ga cgroup ubil brez vrstice.
- **CatBoost je bil počasnejši od ocene:** 2.26 h na učenje (mediana 8143 s) na
  `gwn08` (SXM) proti oceni ~1.65 h iz lokalne sonde; `gwn08` je hkrati gostil
  večino ostalih opravil obeh zagonov. Skupaj 33 h 48 min od 72 h.
- XGBoost je bil hitrejši od ocene po CC18 (~2.5 min na učenje namesto ~25 min),
  verjetno zaradi 82 % manjkajočih vrednosti, ki jih histogrami preskočijo.
- Ker LightGBM pri propadu ne vrže izjeme, ga mehko lovljenje napak ne zazna;
  zazna ga šele pregled ROC-AUC (ali učne izgube) po zagonu.
