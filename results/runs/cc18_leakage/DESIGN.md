# cc18_leakage — test uhajanja podatkov: vnaprej določena zasnova

Tretji poskus diplome (CLAUDE.md, What is next, korak 8). Ta dokument je
zapisan **pred** vsakim zagonom modela na spremenjenih podatkih CC18 in Medic3:
mentor je zasnovo odobril 2026-10-20 (»Test na 36 množicah kar naredite, tako
kot ste si zamislili je čisto OK, ne glede na izid bova sum na uhajanje navedla
kot omejitev.«), parametri so bili določeni 2026-10-21. Pred zapisom sta tekla le
preverba podatkov brez modelov (`scripts/gen_leakage_ids.py --check`,
`diagnostics/perturbation_check.csv`) in tehnični preizkus na naboru sick, ki
zaradi nasičenosti ni del testa (`leakage_smoke_local`). Parametri so v
`config.yaml` (razdelek `leakage`), analiza v `scripts/analyze_leakage.py`.

## Vprašanje

Ali TabPFN-3.5 (in TabICL) na zbirki OpenML-CC18 dosega tako dobre rezultate
tudi zato, ker je nabore videl med razvojem in si zapomnil njihove vrednosti
(kontaminacija temeljnega modela)? Ne gre za uhajanje v našem cevovodu — tega
ta test ne preverja.

**Načelo.** Nabor spremenimo tako, da se vrednosti spremenijo, učni problem pa
ostane enak. Model, ki se uči le iz konteksta, ima na spremenjenem naboru
približno enak rezultat; model, ki nabor prepozna po zapomnjenih vrednostih,
bi na kopiji izgubil več. Padec temeljnega modela primerjamo s padcem
CatBoosta, ki ga monotona preslikava skoraj ne spremeni.

## Zasnova

| Element | Odločitev |
|---|---|
| Nabori | 36 nenasičenih naborov CC18 (`cc18_tuning`, indeksi 0–35; najboljši ROC-AUC < 0,995) **brez** analcatdata_dmft (469) in connect-4 (40668), ki nimata nobenega številskega atributa → **34 naborov** |
| Referenca | Medic3-160 (zaupni bolnišnični nabor, nikoli objavljen) v istih različicah |
| Modeli | TabPFN-3.5 (glavna hipoteza), TabICL (drugotni rezultat), CatBoost (kontrola); vsi s privzetimi nastavitvami, enaka koda kot v diplomskih zagonih |
| Različice | `original`; `monotone`: vsak številski atribut z = (x − povprečje)/std, nato x' = cbrt(z) (predznačeni kubični koren); `monotone_noise`: x' = cbrt(z + ε), ε ~ N(0; 0,1²), en žreb na nabor, seme 42 |
| Ostalo nespremenjeno | kategorični atributi, manjkajoče vrednosti, konstantni atributi, oznake razredov, foldi |
| Foldi | foldi 0–4 (ponovitev 0) diplomskih zagonov, 5-kratno stratificirano prečno preverjanje, seme delitve 42, seme modela 42; vse tri različice nabora v istem opravilu na istem vozlišču |
| Metrika | ROC-AUC (glavna metrika diplome; točnost in log. izgubo lahko pozneje izračunamo iz shranjenih napovedi) |
| Statistika | na nabor: padec = AUC(original) − AUC(različica), povprečje 5 foldov; Δ = padec(model) − padec(CatBoost); enostranski Wilcoxonov test predznačenih rangov čez 34 naborov, H1: Δ > 0; ničle `zsplit`; Holmov popravek čez obe spremenjeni različici znotraj modela; α = 0,05 |
| Referenčno pravilo | značilen rezultat na CC18 je skladen s pomnjenjem le, če je Δ na Medic3-160 **manjši od mediane** Δ na CC18; sicer ga pojasni splošna občutljivost modela na preslikavo |
| Izločitev | nabor, kjer katerikoli algoritem v katerikoli različici ne uspe na vseh 5 foldih, je izločen iz testa in naveden |

## Odločitve, utemeljitve in viri

Strani so tiskane številke strani v PDF-ju navedene različice (preverjeno
2026-10-21 iz besedil, prebranih s `pdftotext`). Kjer vira ni, je odločitev
utemeljena z lastno izpeljavo ali meritvijo in je tako tudi označena.

1. **Načelo testa (spremenimo vrednosti, ne problema; primerjamo z novimi nabori).**
   Bordt in sod. (2024) za LLM: »we add small amounts of noise to numerical
   values … contamination is more likely to have an effect if the LLM can
   'recognize' a datapoint« (str. 2); »It is possible to make changes to the
   format of the data without (significantly) affecting the underlying
   classification problem« (razd. 4.1, str. 6); padec na zapomnjenih naborih,
   »completely absent on the novel datasets« (razd. 4.2, str. 7). Potrebo po
   takem orodju za tabelarične temeljne modele navaja TabArena (Erickson in sod.,
   2025, dodatek E.6, str. 55–56): »For tabular foundation models, we currently
   have no way of detecting contamination … future research and tools, akin to
   the work by Bordt et al. [122], are needed.«

2. **CatBoost kot kontrola.** Odločitvena drevesa so »generally invariant under
   strictly monotone transformations of the individual predictor variables«,
   knjiga CART pa: »invariant under all monotone transformations of individual
   ordered variables« (Galili in Meilijson, 2016, str. 2, citat Breiman in sod.
   1984, str. 57); izjema so nove vrednosti blizu točke delitve (prav tam, str. 1).
   Pri natanko tej preslikavi je imel XGBoost spremembo točnosti 0,0 [0,0; 0,0]
   (Biloš in sod., 2026, dodatek H.5, str. 44). Lastna meritev (sintetični
   podatki, 2026-10-20): CatBoost razdeli učne primere v iste koše, ROC-AUC enak
   na 6 decimalk. Tudi Bordt in sod. z logistično regresijo in gradientnim
   boostingom preverijo, da preoblikovanja ne spremenijo naloge (tabeli 9 in 10,
   str. 30). CatBoost je najboljši drevesni ansambel v naših rezultatih CC18.

3. **Zakaj monotona preslikava sploh doseže TabPFN-3.5 in zakaj je potreben šum.**
   Poročilo TabPFN-3.5 (Jäger in sod., 2026, razd. 3.1, str. 9): vsaka celica
   vstopi kot standardizirana vrednost (Fourierjeve značilke) in kot rang v
   stolpcu (ECDF); »midranks are unchanged by any strictly increasing
   transformation of a column … the standardized values are also passed
   through, so the exact magnitudes survive«. Monotona preslikava torej spremeni
   le kanal vrednosti, kanal rangov ostane enak — zato druga različica s šumom.
   TabPFN-3.5 nima dodatnih preslikav, ki bi spremembo izničile (razd. 3.2,
   str. 11: »does not rely on additional feature transformations such as
   quantile transformations or robust scaling«).

4. **Predznačeni kubični koren.** Objavljena monotona preslikava za tabelarične
   temeljne modele: »signed cube root warp«, eksponent 1/3, določen »before any
   model evaluation« (Biloš in sod., 2026, tabela 7, str. 16; definicija v
   dodatku H.1, str. 42: preslikave, ki »transform individual features while
   preserving their rank order«). Splošna občutljivost je majhna: TabPFNv2
   −1,1 o. t. [−2,5; +0,1], TabICLv2 −0,3 o. t. (neznačilno; tabela 3, str. 9);
   »consistent with their pretraining prior already covering smooth monotone
   warps« (razd. 6, str. 9). Majhna splošna občutljivost pomeni, da bi padec
   zaradi pomnjenja izstopal. Kubični koren je definiran za vsa realna števila
   in ne more preseči obsega float32 (eksponentna funkcija bi ga pri
   bank-marketing, |z| = 119, presegla).

5. **Centriranje pred korenom (lastna izpeljava).** Biloš in sod. ne navajajo,
   ali pred preslikavo standardizirajo. cbrt ni neodvisen od premika: na
   atributu daleč od nič (npr. vrednosti 1000–1010) je skoraj linearen, model
   pa vsak atribut sam standardizira (poročilo TabPFN-3.5, sl. 8, str. 10), zato
   bi sprememba skoraj izginila. Centriranje postavi ukrivljenost tja, kjer so
   podatki. Deljenje s std obliko le poenoti (cbrt(cx) = cbrt(c)·cbrt(x)).
   Posledica: atribut z dvema vrednostma (npr. 0/1) ostane po standardizaciji
   enak — strogo naraščajoča preslikava dveh točk je vedno linearna.

6. **Šum: namen, enota in velikost.** Namen: spremeniti rang, ki ga monotona
   preslikava ne more (točka 3). Enota: standardni odklon atributa, kot pri šumu
   v tabeli 7 (Biloš in sod., 2026, str. 16: »σ multiplier of feature std«).
   Velikost 0,1 std je **naša izbira** (noben vir ne predpiše številke),
   utemeljena takole: (a) varianca šuma je 1 % variance atributa — »rahel« šum v
   smislu Bordt in sod. (dodatek E, str. 23: sprememba »with a relative error of
   approximately 1%«; naš šum je merjen glede na razpršenost atributa, ne glede
   na vrednost, da atributi z velikim odmikom od nič, npr. leto, ne razpadejo);
   (b) TabPFN-3.5 rang zapiše le s K = 4 harmoniki nizke frekvence (poročilo,
   sl. 8, str. 10), zato mora šum rang premakniti za nekaj odstotkov, da ga model
   sploh zazna. Za normalno porazdeljen atribut je pričakovani premik
   E|Δu| ≈ 0,0225 (izpeljava: E[φ(z)]·E|ε| = (2√π)⁻¹·0,1·√(2/π)); izmerjeno na
   podatkih (`diagnostics/perturbation_check.csv`): od 0,022 pri naborih s
   pretežno zveznimi atributi (climate-model-simulation-crashes 0,0224,
   numerai28.6 0,0228 — skladno z izpeljavo) do 0,20 tam, kjer šum razbije vezi
   diskretnih atributov (Bioresponse 0,20, spambase 0,18); Medic3-160 0,098.
   Razbite vezi so del motnje za vse modele, tudi za kontrolo.

7. **TabICL kot drugotni rezultat.** Tudi TabICL je temeljni model diplome.
   Del njegove občutljivosti sam izniči: »TabICL applies z-normalization with or
   without power transformation to all input features« (Qu in sod., 2025,
   str. 7); v tabicl 2.2.0 štirje od osmih članov ansambla uporabijo
   Yeo-Johnsonovo preslikavo (izvorna koda, `norm_methods` privzeto
   `["none", "power"]`).

8. **Nabori.** 36 nenasičenih je odobril mentor (e-pošta 2026-10-20). Nabora
   brez številskega atributa izpadeta po definiciji (lastna izpeljava): obe
   preslikavi sta na njiju identiteta, zato delta ne nosi informacije.

9. **Medic3-160 kot referenca.** Nabor ni bil nikoli objavljen, zato ga noben
   temeljni model ni mogel videti. Primerjava zapomnjenih in novih naborov je
   osrednji del zasnove Bordt in sod. (razd. 4.2, tabela 4, panel B, str. 7).
   Pravilo »manjši od mediane« je **naše**, določeno vnaprej; z enim referenčnim
   naborom je le groba ločnica.

10. **Foldi in vozlišče (lastna meritev).** Foldi 0–4 so isti kot v diplomskih
    zagonih, zato izvirna različica ponovi diplomske rezultate (preverba v
    analizi). Vse različice nabora tečejo v istem opravilu, ker se TabPFN-3.5
    med vrstami GPU razlikuje do ~1e-3 (CLAUDE.md, Environment → Measured effects).

11. **Statistični test.** Wilcoxonov test predznačenih rangov je priporočeni test
    za primerjavo dveh količin čez več naborov (Demšar, 2006, razd. 3.1.3,
    str. 7–8); ničle razdelimo kot drugje v diplomi (`zsplit`). Enostranski test
    na razliki po naborih, določen vnaprej, uporabijo Biloš in sod. (2026,
    dodatek H.6, str. 44: »A pre-specified one-sided Wilcoxon signed-rank test on
    the per-dataset paired difference«), prav tam tudi Holmov popravek znotraj
    modela (»Within each model … Holm's step-down procedure«); Holmov postopek po
    Demšar (2006, str. 12–13). Enostranski, ker pomnjenje napoveduje le večji
    padec temeljnega modela.

12. **Omejitve (navedemo jih ne glede na izid, po dogovoru z mentorjem).** Test
    zazna le pomnjenje, ki je vezano na vrednosti. Ne zazna posrednega vpliva:
    razvijalci so med razvojem uporabljali realne nabore — poročilo TabPFN-3.5
    (dodatek C.5, str. 29) izloči »the 26 that were used during the development of
    TabPFN-2 and TabICLv2«, TabArena (dodatek E.6, str. 55) opozarja, da lahko
    razvijalci »overfit the hyperparameters of their model on the TabArena
    datasets, or use them for pretraining a foundation model«. Ne zazna niti
    pomnjenja lastnosti, ki jih obe različici ohranita (število atributov in
    razredov, deleži razredov, vzorec manjkajočih vrednosti). Neznačilen
    rezultat torej ni dokaz, da modela nabora nista videla.

## Preverba podatkov pred zagonom (brez modelov)

`python scripts/gen_leakage_ids.py --check` (Kremen, 2026-10-21): za vseh 34
naborov in Medic3-160 so foldi in oznake razredov v vseh različicah enaki
foldom 0–4 diplomskih zagonov; kategorični atributi in vzorec manjkajočih
vrednosti so nespremenjeni; monotona različica ohrani vrstni red in vezi vsakega
atributa. Po zaokrožitvi na float32 (vhod CatBoosta) se pri 4 naborih v nekaj
stolpcih spremeni število različnih vrednosti za največ ~1 % (npr. numerai28.6
attribute_5: 998 → 991; mfeat-zernike att43: 1939 → 1981, tu preslikava vezi
razreši) — zato je kontrola »skoraj«, ne povsem nespremenljiva.

## Zagon (Arnes, po `git pull`)

```
RUN_ID=cc18_leakage sbatch --array=0-33 scripts/run_leakage.sh
DATASET_SET=medic3_160_leakage ALGORITHMS="tabpfn tabicl" RUN_ID=medic3_160_leakage sbatch --array=0 --mem=160G scripts/run_leakage.sh
DATASET_SET=medic3_160_leakage RUN_ID=medic3_160_leakage sbatch --array=0-2 scripts/run_leakage_cpu.sh
```

Pričakovano: `cc18_leakage` 34 × 3 × 3 × 5 = 1530 vrstic, `medic3_160_leakage`
3 × 3 × 5 = 45 vrstic. Nato `python scripts/merge_results.py <zagon>` za oba in
`python scripts/analyze_leakage.py cc18_leakage --reference medic3_160_leakage`.

## Viri

- Bordt, S., Nori, H., Rodrigues, V., Nushi, B., Caruana, R. (2024). Elephants
  Never Forget: Memorization and Learning of Tabular Data in Large Language
  Models. COLM 2024. arXiv:2404.06209 (v3).
- Biloš, M., Wilson, J. T., Schneider, A., Nevmyvaka, Y. (2026). A Mechanistic
  Study of Tabular Foundation Models. arXiv:2605.21288 (v1, prednatis).
- Galili, T., Meilijson, I. (2016). Splitting matters: how monotone
  transformation of predictor variables may improve the predictions of decision
  tree models. arXiv:1611.04561 (v1, prednatis).
- Jäger, B., Erickson, N., Grinsztajn, L., in sod. (2026). TabPFN-3.5: Technical
  Report. arXiv:2609.17895 (v2).
- Erickson, N., Purucker, L., Tschalzev, A., Holzmüller, D., Desai, P. M.,
  Salinas, D., Hutter, F. (2025). TabArena: A Living Benchmark for Machine
  Learning on Tabular Data. NeurIPS 2025 Datasets and Benchmarks.
  arXiv:2506.16791 (v4).
- Qu, J., Holzmüller, D., Varoquaux, G., Le Morvan, M. (2025). TabICL: A Tabular
  Foundation Model for In-Context Learning on Large Data. ICML 2025.
  arXiv:2502.05564.
- Demšar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data
  Sets. JMLR 7, 1–30.
