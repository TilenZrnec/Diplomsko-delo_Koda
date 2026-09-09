# Razlaga repozitorija po korakih

Ta dokument spremlja štiri diagrame v `zaporedje.puml` in razloži isto zgodbo
z besedami, za nekoga, ki Python šele spoznava. Vsak del ustreza enemu
diagramu. Datoteke so navedene s polno potjo, tako kot v diagramih.

Nekaj pojmov, ki se pojavljajo povsod:

- **Modul** je ena datoteka `.py`. **Funkcija** je poimenovan kos kode v modulu,
  ki sprejme vhode (argumente) in vrne rezultat. Ko piše `load_config()` iz
  `src/config.py`, pomeni »funkcija `load_config` v datoteki `src/config.py`«.
- `python -m src.run_benchmark` pomeni »poženi modul `src/run_benchmark.py` kot
  program«. Oblika `-m src.ime` (s piko, brez `.py`) poskrbi, da Python najde
  tudi ostale datoteke v mapi `src/`, ker jo obravnava kot **paket**. Zato
  obstaja prazna datoteka `src/__init__.py`: samo označuje, da je `src/` paket.
- **Slovar** (`dict`) je zbirka parov ključ → vrednost, na primer
  `{"name": "sick", "X": ..., "y": ...}`. Skoraj vse funkcije tu vračajo slovar.
- **DataFrame** je tabela iz knjižnice pandas: vrstice so primeri, stolpci so
  atributi. `X` je tabela vhodnih atributov, `y` je stolpec ciljnih oznak
  (kaj napovedujemo).
- **Argumenti ukazne vrstice** so besede za imenom programa, na primer
  `--dataset-set cc18`. Bere jih knjižnica `argparse`; vsak skript ima na vrhu
  seznam, katere argumente sprejme in kaj je privzeto, če jih izpustiš.

---

## Del 1: Priprava (`01_priprava`)

Tu se nič ne uči. Priprava določi **kaj** bomo merili (kateri nabori) in
**s katerimi nastavitvami**. Vse se zgodi enkrat, lokalno.

### `config.yaml`: edini vir nastavitev

YAML je berljiv zapis oblike `ključ: vrednost`. Vsak skript v repozitoriju
prebere to datoteko, nikjer drugje ni nastavitev zapisanih trdo. Ključi:

- `random_state: 42`. **Naključno seme.** Računalnik ne zna generirati pravih
  naključnih števil, ampak zaporedje, ki je videti naključno in ga v celoti
  določa začetna vrednost, seme. Isto seme, isto zaporedje. Zato dobi vsak, ki
  požene kodo, iste delitve na folde in iste »naključne« odločitve v modelih.
  Številka 42 nima posebnega pomena, pomembno je, da je vedno ista.
- `n_splits: 5`. **Prečno preverjanje** (angl. cross-validation). Nabor
  razdelimo na 5 enakih delov, foldov. Petkrat zapored en del vzamemo za
  testiranje, ostale štiri za učenje. Vsak primer je natanko enkrat v testnem
  delu. Rezultat je 5 ocen namesto ene, kar pove tudi, koliko se ocene med
  seboj razlikujejo.
- `n_repeats: 1`. Koliko krat ponovimo celotno prečno preverjanje z drugačnim
  mešanjem. Pri 1 dobimo 5 foldov, pri 3 dobimo 15. Več ponovitev da bolj
  stabilno oceno, a stane sorazmerno več računanja.
- `algorithms`. Seznam imen; vsako ime je ključ v `REGISTRY` v
  `src/models/__init__.py`, ki pove, katera datoteka izvede ta algoritem.
- `dataset_sets`. Slovar ime → JSON datoteka z opisi naborov. `subset` je
  pilotna trojica, `cc18` je 72 naborov OpenML-CC18, `medic3` je lokalni CSV.
- `cache_dir`. Kam OpenML shrani prenesene nabore, da jih ne prenaša znova.
- `results_dir`. Koren map z rezultati, `results/runs/`.
- `save_predictions: true`. Ali naj se za vsako učenje shranijo napovedane
  verjetnosti. Da, ker iz njih lahko pozneje izračunamo katerokoli metriko.
- `saturation_threshold: 0.995` in `alpha: 0.05` uporablja analiza, razložena
  v delu 4.

### `src/config.py`

Tri funkcije. `load_config()` prebere `config.yaml`, doda privzete vrednosti
za ključe, ki bi manjkali, in relativne poti (`data/openml_cache`) spremeni v
absolutne, tako da skripti delujejo iz katerekoli mape. `dataset_specs(config,
"cc18")` odpre JSON datoteko, na katero kaže `dataset_sets.cc18`, in vrne
seznam opisov naborov. `spec_id(spec)` iz opisa naredi kratek niz za ime
datoteke, na primer `31` ali `medic3`.

### Opisi naborov: `scripts/subset_ids.json`, `scripts/cc18_ids.json`, `scripts/medic3.json`

Opis nabora je bodisi celo število, OpenML ID (`31`), bodisi slovar za lokalno
datoteko: `{"source": "csv", "path": "../Medic3.csv.zip", "target": "Class",
"drop_cols": ["Index", "Field"]}`. `target` je stolpec, ki ga napovedujemo,
`drop_cols` so stolpci, ki niso atributi (zaporedna številka vrstice in
oznaka oddelka, ki bi izdala razred).

### `scripts/gen_cc18_ids.py`

OpenML-CC18 je uradna zbirka 72 naborov za primerjavo klasifikatorjev. Skript
jo prebere z OpenML (`openml.study.get_suite(99)`), vzame ID-je, odstrani
podvojene, jih uredi in zapiše v `cc18_ids.json`. Če OpenML ne vrne natanko
72 naborov, se skript ustavi z napako, ker bi tiha sprememba zbirke
spremenila obseg eksperimenta. Datoteka je v gitu, zato je nabor naborov
zamrznjen; to je »verzioniranje podatkov« iz smernic FRI.

### `scripts/profile_datasets.py`

Izpiše velikost vsakega nabora (vrstice, atributi, kategorični atributi,
razredi), urejeno padajoče po številu celic. Privzeto vpraša OpenML samo za
metapodatke, brez prenosa. Z `--from-cache` nabore dejansko naloži prek
`load_dataset()`, kar je edina pot za lokalne CSV nabore. Namen je izbira
`--mem` in `--time` za gručo: največji nabor določa, koliko pomnilnika mora
dobiti en task.

---

## Del 2: Lokalni pilot (`02_lokalni_pilot`)

Ukaz: `python -m src.run_benchmark --dataset-set subset`. Tu se koda razvija
in preverja na treh majhnih naborih, preden gre na gručo.

### `src/run_benchmark.py`: vstop

Razčleni argumente (`--dataset-set`, `--run-id`, `--algorithms`), pokliče
`load_config()` in `dataset_specs()`. Če `--run-id` ni podan, ga sestavi
`make_run_id()` iz `src/runner.py` kot `<datum-čas>_<nabor>_<stroj>`, na
primer `20260909-143000_subset_Kremen`. Tako se noben zagon ne prepiše.
`run_dir_for()` ustvari mapo `results/runs/<run_id>/`. Nato `write_manifest()`
zapiše `manifest.json`: git commit, ali je koda spremenjena glede na commit,
ime stroja, Python, torch in CUDA, GPU, celotna konfiguracija in seznam vseh
nameščenih paketov z različicami (`pip freeze`). To je »identiteta zagona«,
ki jo zahteva ponovljivost: čez pol leta se natanko ve, kaj je teklo.

Potem za vsak nabor pokliče `run_dataset()` in na koncu `merge_run()`.

### `src/runner.py`: `run_dataset()`, srce vsega

To je edino mesto, kjer je zapisana zanka »za vsak algoritem, za vsak fold«.

1. **Že narejeno?** Če `per_dataset/<id>.csv` obstaja, izpiše »already done«
   in konča. Če obstaja `<id>.csv.partial`, ga prebere in si zapiše, kateri
   pari (algoritem, fold) so že opravljeni; te bo preskočil.
2. **Nalaganje podatkov** prek `load_dataset()` iz `src/data.py`, glej spodaj.
3. **Za vsak algoritem** vzame modul iz `REGISTRY`. Če ima modul
   `USES_GPU = True` (TabPFN, TabICL) in v tem procesu še ni bil uporabljen,
   opravi **ogrevanje**: en klic na majhnem izmišljenem naboru 64 × 4. Prvi
   klic modela na GPU vsebuje nalaganje utež z diska na grafično kartico in
   pripravo CUDA, kar traja sekunde. Brez ogrevanja bi ta strošek pristal v
   času folda 0 in pokvaril primerjavo časov. Čas ogrevanja se zapiše v
   stolpec `warmup_s`.
4. **Za vsak fold** iz `X` in `y` izreže učni in testni del po indeksih folda
   (`X.iloc[train_idx]` pomeni »vrstice s temi zaporednimi številkami«),
   pokliče `run()` modula in dobi slovar z rezultatom.
5. **Shrani napovedi** v `predictions/<id>/<algoritem>_r<ponovitev>_f<fold>.npz`.
   `.npz` je stisnjen zapis več numpy polj: matrika verjetnosti, prave
   oznake testnih primerov in njihovi indeksi.
6. **Doda vrstico** s stolpci iz `RESULT_COLUMNS` (nabor, algoritem, fold,
   velikosti, ROC-AUC, časi, naprava, predobdelava, napake, git commit, stroj,
   čas) in **takoj** prepiše `<id>.csv.partial`. Zapis je **atomaren**: najprej
   v `.tmp`, nato `os.replace()`, ki datoteko preimenuje v enem koraku. Če
   proces umre sredi pisanja, ostane stari veljavni partial, nikoli pol
   napisana datoteka.
7. Ko je vse narejeno, `os.replace(partial, <id>.csv)`. Končna datoteka torej
   obstaja šele, ko je popolna; nepopoln rezultat se ne more pretvarjati, da
   je popoln.

`merge_run()` na koncu združi vse `per_dataset/*.csv` v `results.csv` in
glasno našteje morebitne `.partial`, torej nedokončane nabore.

### `src/data.py`: `load_dataset()`

- Za OpenML nabor pokliče `openml.datasets.get_dataset(id)` in `get_data()`,
  ki vrne `X`, `y` in `categorical_indicator`, seznam »je ta stolpec
  kategoričen« po metapodatkih OpenML. Kategoričen atribut ima končno
  množico vrednosti brez vrstnega reda (barva, poklic); številski ima
  števila. Ta razlika je pomembna, ker jo algoritmi obravnavajo različno.
- Za CSV nabor pokliče `pd.read_csv(path)` (pandas zna brati tudi `.zip`),
  odstrani `drop_cols`, vzame `target` kot `y`. Kategorični so stolpci, ki
  niso številski, razen če jih opis našteje izrecno.
- `X.reset_index(drop=True)` oštevilči vrstice 0, 1, 2, ..., ker foldi vračajo
  zaporedne številke vrstic.
- `LabelEncoder` pretvori oznake razredov (`"good"`, `"bad"`) v števila
  (0, 1). To ni uhajanje informacije, ker uporabi le imena razredov, ne
  atributov.
- **`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.**
  »Stratified« pomeni, da ima vsak fold **enako razmerje razredov** kot
  celoten nabor. Če je v naboru 10 % bolnih, ima vsak fold 10 % bolnih.
  Brez tega bi lahko redek razred po naključju pristal ves v enem foldu in
  bi bil model v drugih foldih testiran na razredu, ki ga ni nikoli videl.
  `shuffle=True` pred delitvijo premeša vrstice, ker so nabori pogosto
  urejeni (najprej vsi enega razreda). `random_state` naredi mešanje
  ponovljivo. Pri `n_repeats > 1` se uporabi `RepeatedStratifiedKFold`, ki
  isto naredi večkrat z drugačnim mešanjem.
- **`list(splitter.split(X, y))`** folde izračuna enkrat in jih shrani v
  seznam. Vsi algoritmi potem berejo iste pare (učni indeksi, testni
  indeksi). To je bistveno za pošteno primerjavo: vsak algoritem vidi
  natanko iste učne in iste testne primere.

### `src/models/*.py`: šest algoritmov z istim podpisom

Vsaka datoteka ima funkcijo `run(X_train, y_train, X_test, y_test,
categorical_cols, random_state)`, ki vrne slovar z istimi ključi. Runner zato
z njimi ravna enako in ne ve, kateri algoritem kliče. Vsaka `run()`:

1. opravi **svojo predobdelavo**, ki je zavestno najmanjša možna, samo
   toliko, da algoritem vhod sploh sprejme (to je eksperimentalna
   spremenljivka diplome);
2. ustvari model s privzetimi nastavitvami in podanim `random_state`;
3. izmeri `fit()` z `time.perf_counter()` (natančna štoparica), nato
   `predict_proba()`, ki za vsak testni primer vrne verjetnost vsakega
   razreda;
4. pokliče `compute_roc_auc()` iz `src/utils.py`;
5. vse ovije v `try/except`: če karkoli vrže izjemo, se besedilo napake
   zapiše v `result["error"]` in program teče naprej (»fail-soft«). En
   neuspel fold tako ne sesuje večurnega zagona.

Predobdelave po algoritmih:

- **RandomForest** ne zna manjkajočih vrednosti (NaN) niti kategorij. Zato
  **imputacija**: manjkajoče številske vrednosti nadomesti mediana, manjkajoče
  kategorične najpogostejša vrednost; kategorije se **ordinalno kodirajo**,
  vsaka dobi celo število. Imputer se **prilega samo na učni del** (`fit` na
  `X_train`, `transform` na obeh), sicer bi testni podatki vplivali na učenje.
- **XGBoost** zna NaN, ne zna nizov. Kategorije ordinalno kodira, NaN ostane
  NaN. Ordinalno kodiranje kategorijam vsili umeten vrstni red, kar mu lahko
  škodi; to je namerno del meritve.
- **LightGBM** zna NaN in kategorije, če so stolpci pandas tipa `category`.
  Ostale stolpce pretvori v `float`, ker OpenML kak povsem prazen stolpec
  naloži kot `object`.
- **CatBoost** zna NaN v številskih stolpcih in kategorije prek
  `cat_features`, a v kategoričnih stolpcih ne dovoli NaN; zato se vsaka
  vrednost pretvori v niz in NaN v niz `'nan'`, ki postane lastna kategorija.
- **TabPFN in TabICL** dobita surove podatke. Če surovi klic vrže izjemo, se
  ta zabeleži v `raw_error`, uporabi se minimalni popravek in poskus se
  ponovi. Trenutno surovi klic deluje povsod.

### `src/utils.py`

`compute_roc_auc(y_true, proba)`. **ROC-AUC** meri, kako dobro model **razvršča**
primere: vzemi naključen pozitiven in naključen negativen primer; AUC je
verjetnost, da model pozitivnemu pripiše višjo verjetnost. 1,0 je popolno,
0,5 je ugibanje. Pri več kot dveh razredih se uporabi **one-vs-rest**: za vsak
razred posebej »ta razred proti vsem ostalim«, nato **macro** povprečje, kjer
ima vsak razred enako težo ne glede na pogostost. `describe_device()` vrne
`cpu x8` ali `cuda: NVIDIA H100`, da je pri vsakem času jasno, na čem je tekel.

---

## Del 3: Zagon na gruči Arnes (`03_arnes`)

Gruča je veliko računalnikov s skupno čakalno vrsto, ki jo upravlja
**SLURM**. Ti ne poganjaš programov neposredno, ampak oddaš **opravilo** (job)
z zahtevo po virih (GPU, jedra, pomnilnik, čas), SLURM pa ga požene, ko so
viri prosti. **Prijavno vozlišče** (login node) je računalnik, na katerega se
prijaviš s `ssh`; ima internet, nima GPU-ja. **Računska vozlišča** imajo
GPU-je, a **nimajo interneta**. Zato je zagon v treh korakih.

### Korak 1: predpriprava na prijavnem vozlišču, `scripts/prestage.py`

Vse, kar bi računsko vozlišče hotelo prenesti, mora biti na disku vnaprej.
Skript prek `config.py` prebere seznam naborov in za vsakega pokliče isti
`load_dataset()` kot benchmark, tako da pristanejo v istem predpomnilniku.
Nato enkrat pokliče TabPFN in TabICL na izmišljenih podatkih, da se preneseta
njuni uteži. Na koncu izpiše velikost predpomnilnika (domači imenik ima kvoto
100 GB). Napaka pri enem naboru ne prekine ostalih; nabori z napako se
izpišejo na koncu.

### Korak 2: oddaja polja, `scripts/run_cc18.sh`

`sbatch` odda skripto. Vrstice `#SBATCH` na vrhu so zahteve za SLURM:
`--array=0-71%4` pomeni **72 samostojnih opravil**, oštevilčenih 0 do 71,
največ 4 hkrati; vsako dobi 1 GPU (`--gres=gpu:1`), 8 jeder, 64 GB pomnilnika
in 12 ur. Vsako opravilo je isti skript, le spremenljivka
`SLURM_ARRAY_TASK_ID` ima drugo vrednost. Skript nato:

1. `set -euo pipefail`: ustavi se ob prvi napaki, namesto da bi tiho tekel
   naprej.
2. `source ~/.tabpfn_token` naloži žeton za TabPFN (licenca) iz datoteke, ki
   je samo na gruči.
3. `export HF_HUB_OFFLINE=1` pove knjižnici Hugging Face, naj ne poskuša na
   internet. `OMP_NUM_THREADS=8` omeji ansamble na dodeljenih 8 jeder, sicer
   bi poskusili uporabiti vsa jedra vozlišča.
4. `RUN_ID` vzame iz okolja (`RUN_ID=cc18_v2 sbatch ...`) ali sestavi
   `cc18_<številka polja>`. Vsa opravila polja pišejo v isto mapo
   `results/runs/<RUN_ID>/`.
5. Varovalka: prek `config.py` prešteje nabore in preveri, da je zgornja meja
   polja enaka številu naborov minus 1; sicer bi zadnji nabori tiho izpadli.
6. Požene `micromamba run -p ~/envs/tabular2 python -m src.run_one_dataset
   --dataset-set cc18 --index $SLURM_ARRAY_TASK_ID --run-id $RUN_ID`.
   `micromamba run -p <pot>` pomeni »poženi v tem okolju«, ker skripte v
   ozadju ne morejo klicati `conda activate`.

`src/run_one_dataset.py` je dvojček `run_benchmark.py` za en nabor: prebere
konfiguracijo, vzame `dataset_specs(...)[index]`, pokliče `write_manifest()`
(manifest zapiše prvo opravilo, ki pride do tja, ostala ga najdejo) in
`run_dataset()`. Od tu naprej je pot **enaka lokalni**, ker teče isti
`src/runner.py`. Zato ima arneški rezultat iste stolpce, iste kontrolne točke
in isti zapis napovedi kot lokalni.

**Ponovna oddaja.** Če opravilo preseže čas ali zmanjka pomnilnika, oddaš
polje znova z **istim** `RUN_ID`; vsako opravilo prebere svoj partial in
nadaljuje pri prvem nenarejenem učenju. Za posamezne nabore z več pomnilnika:
`ALLOW_SPARSE_ARRAY=1 RUN_ID=cc18_v2 sbatch --array=27,60,61,70 --mem=240G
scripts/run_cc18.sh`. `ALLOW_SPARSE_ARRAY=1` izklopi varovalko iz točke 5.

### Korak 3: združevanje, `scripts/merge_results.py`

`python scripts/merge_results.py cc18_v2` pokliče `merge_run()` iz runnerja,
ki zlepi 72 datotek v `results.csv` in našteje `.partial`, če kateri ostane.
Pričakovanih je 72 × 6 × `n_splits` × `n_repeats` vrstic; manj vrstic pomeni
nedokončano, nikoli neuspešno, ker neuspešno učenje vseeno zapiše vrstico z
razlogom v `error`.

### Točni ukazi za preizkus na Arnesu

Na prijavnem vozlišču, v mapi repozitorija. Najprej okolje (enkrat):

```bash
cd ~/Diplomsko-delo_Koda        # ali kjerkoli je repozitorij na gruči
git pull
~/bin/micromamba create -y -p ~/envs/tabular2 -c conda-forge python=3.12
~/bin/micromamba run -p ~/envs/tabular2 pip install --upgrade pip
~/bin/micromamba run -p ~/envs/tabular2 pip install -r requirements.txt
~/bin/micromamba run -p ~/envs/tabular2 python -c "import torch, sklearn, xgboost, tabpfn; print(torch.__version__, sklearn.__version__, xgboost.__version__)"
```

Na prijavnem vozlišču `torch.cuda.is_available()` vrne `False`, ker tam ni
GPU-ja; to je pričakovano. Nato predpriprava in validacijski zagon pilotne
trojice:

```bash
source ~/.tabpfn_token
~/bin/micromamba run -p ~/envs/tabular2 python scripts/prestage.py --dataset-set subset
RUN_ID=subset_v2 sbatch scripts/run_subset.sh
squeue -u $USER                  # dokler se ne izprazni
tail -n 20 logs/subset-*_0.out   # izpis prvega opravila
```

Ko so vsa tri opravila v stanju `COMPLETED`:

```bash
~/bin/micromamba run -p ~/envs/tabular2 python scripts/merge_results.py subset_v2
~/bin/micromamba run -p ~/envs/tabular2 python scripts/compare_results.py subset_v2_local subset_v2
sacct -j <JOBID> --format=JobID,JobName%20,Elapsed,MaxRSS,State,NodeList
```

Pričakovano: RandomForest, XGBoost, LightGBM, CatBoost razlika 0,0, TabPFN in
TabICL do približno 1e-4 do 1e-3 (druga grafična kartica). Če se ujema,
rezultate commitaš in potisneš, izpis `sacct` pa prilepiš v
`results/runs/subset_v2/PROVENANCE.md`:

```bash
git add results/runs/subset_v2
git commit -m "Add Arnes validation run subset_v2"
git push
```

Nato polni CC18. Pred oddajo se odloči za `n_repeats` v `config.yaml` in
spremembo commitaj, ker manifest zapiše konfiguracijo ob zagonu:

```bash
~/bin/micromamba run -p ~/envs/tabular2 python scripts/prestage.py --dataset-set cc18
du -sh ~                         # kvota 100 GB
RUN_ID=cc18_v2 sbatch scripts/run_cc18.sh
```

Po koncu (ali ko del opravil pade zaradi pomnilnika):

```bash
~/bin/micromamba run -p ~/envs/tabular2 python scripts/merge_results.py cc18_v2
# če merge našteje .partial ali so opravila 27/60/61/70 padla:
ALLOW_SPARSE_ARRAY=1 RUN_ID=cc18_v2 sbatch --array=27,60,61,70 --mem=240G scripts/run_cc18.sh
```

---

## Del 4: Analiza (`04_analiza`)

Vsi ukazi sprejmejo `run_id`, mapo zagona ali pot do CSV-ja. Pot je obvezna,
da se nikoli ne povzame napačna tabela; prva vrstica izpisa je vedno
`Vir: <pot>`.

### `src/summary.py`

`python -m src.summary cc18_v2`. Korak za korakom:

1. `per_dataset_table`: za vsak par (nabor, algoritem) povprečje in standardni
   odklon ROC-AUC čez folde ter **mediani** časov. Mediana je srednja vrednost
   po velikosti; za razliko od povprečja je ne pokvari en sam počasen fold.
   Če je katerikoli fold neuspel, je povprečje `NaN`; sicer bi povprečje
   uspelih foldov lepšalo sliko.
2. `dataset_pivot`: tabela nabor × algoritem s temi povprečji.
3. `rank_matrix`: v vsaki vrstici (naboru) algoritme razvrsti, 1 je najboljši.
   **Rang se računa na nabor**, ne na fold, ker foldi istega nabora niso
   neodvisni; če bi rangirali po foldih, bi statistični test mislil, da ima
   petkrat več vzorcev, kot jih res ima. **Neuspeh dobi najslabši rang**, ker
   je algoritem, ki nabora ne zmore, na tem naboru najslabši. Izenačeni si
   delijo povprečje rangov.
4. `saturated_mask`: nabor je **nasičen**, če najboljši algoritem doseže vsaj
   `saturation_threshold` (0,995). Tam vsi dosežejo skoraj popoln rezultat in
   razlike so v četrti decimalki, torej šum. Vsak povzetek se zato poda še
   brez nasičenih naborov.
5. `overall_table`: na algoritem povprečni ROC-AUC **samo na naborih, kjer so
   uspeli vsi** (sicer bi vsak algoritem imel povprečje na drugi podmnožici),
   povprečni rang, število zmag, število neuspelih naborov, mediani časov.
6. Vse tabele zapiše v `summary/` kot `.csv` in `.tex`. Tabele v diplomi se
   vključijo iz teh datotek, nikoli s prepisom številk.

### `src/stats.py`

`python -m src.stats cc18_v2 [--no-saturated]`. Odgovarja na vprašanje: ali so
razlike med algoritmi večje, kot bi jih pričakovali po naključju?

- **Ničelna domneva** je trditev »vsi algoritmi so enakovredni«. **p-vrednost**
  je verjetnost, da bi ob resnični ničelni domnevi videli tako velike (ali
  večje) razlike. Če je p manjša od **`alpha`** (0,05), ničelno domnevo
  zavrnemo in rečemo, da so razlike **statistično značilne**.
- **Friedmanov test** to naredi nad matriko rangov (nabor × algoritem). Je
  neparametričen: ne predpostavlja normalne porazdelitve, uporablja le range.
  Rezultat sta `chi2_F` in p.
- **Nemenyijev post-hoc**: če je Friedman značilen, kateri pari se razlikujejo?
  Dva algoritma se značilno razlikujeta, če se njuna povprečna ranga
  razlikujeta za vsaj **kritično razdaljo** `CD = q_alpha * sqrt(k(k+1)/(6N))`,
  kjer je k število algoritmov, N število naborov in `q_alpha` tabelirana
  konstanta (za k = 6 in alpha = 0,05 je 2,850). **Diagram kritične razdalje**
  to nariše: vodoravna os so rangi, vsak algoritem je črta na svojem
  povprečnem rangu, debela vodoravna črta povezuje skupine, ki niso značilno
  različne. Če sta dva algoritma povezana, iz podatkov ne smemo trditi, da je
  eden boljši.
- **Wilcoxonov test predznačenih rangov** primerja dva algoritma neposredno:
  na vsakem naboru izračuna razliko njunih ROC-AUC in preveri, ali so razlike
  sistematično na eni strani. Je občutljivejši od Nemenyija. Ker naredimo 15
  parnih testov, se poveča možnost, da kak par »slučajno« izpade značilen;
  **Holmov popravek** p-vrednosti zato zaostri, tako da skupna verjetnost
  lažnega alarma ostane pod alpha. Izpis pokaže p in `p_holm`; šteje `p_holm`.

### `scripts/gen_version_table.py`

Iz `manifest.json` zagona naredi LaTeX tabelo različic (`summary/razlicice.tex`)
v obliki, ki jo diploma že uporablja. Različice v besedilu tako vedno pridejo
iz zagona, ki je dal rezultate.

### `scripts/compare_results.py`

Združi dva zagona po (nabor, algoritem, fold) in izpiše povprečno in največjo
absolutno razliko ROC-AUC po algoritmih ter pet najslabših vrstic. Uporablja
se za preverjanje prenosa na gručo in za merjenje učinka nadgradnje knjižnic.

### `scripts/profile_medic3.py`

Podatkovna kartica nabora Medic3: bere CSV vrstico po vrstici (142 MB se ne
naloži v pomnilnik naenkrat), prešteje razrede, manjkajoče vrednosti in
preveri dve bližnjici (ali stolpec `Field` izda razred; ali sam vzorec
manjkajočosti napoveduje razred). Sam benchmark Medic3 teče po isti poti kot
OpenML nabori, prek `scripts/medic3.json`.
