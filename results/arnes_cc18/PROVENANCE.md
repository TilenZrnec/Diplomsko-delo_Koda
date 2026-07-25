# Polni zagon OpenML-CC18 (72 naborov) na gruči Arnes

## Rezultat
- Izhodna datoteka: results/results_arnes_cc18.csv
- 2160 vrstic = 72 naborov x 6 algoritmov x 5 delitev
- 10 neuspelih učenj: CIFAR_10 (OpenML 40927) x {tabpfn, tabicl}, vseh 5 delitev.
  TabICL je za eno delitev zahteval ~378 GB pomnilnika, kar presega 256 GB na vozlišče.
  Napake so zabeležene v stolpcu error. Subvzorčenje in disk_offload_dir NISTA bila
  uporabljena - protokol privzetih nastavitev ostaja nedotaknjen.

## Protokol
- Vseh 6 algoritmov s privzetimi hiperparametri
- 5-kratno stratificirano prečno preverjanje, identične delitve za vse algoritme
- random_state = 42 (delitve, foldi, vsi modeli)
- Metrika: ROC-AUC (binarno in večrazredno one-vs-rest)

## Strojna oprema in okolje
- Particija gpu, constraint h100, RealMemory = 256000 MB
- 1 GPU, 8 CPU jeder na opravilo (OMP_NUM_THREADS = 8)
- Okolje: micromamba prefix ~/envs/tabular
- HF_HUB_OFFLINE = 1, nabori predhodno preneseni s scripts/prestage.py na prijavnem vozlišču

## Opravila SLURM

| Job ID | Array | --mem | --time | Izid |
|---|---|---|---|---|
| 17746573 | 0-71%4 | 64G | 08:00:00 | 68/72 uspešnih; taski 27, 60, 61, 70 OUT_OF_MEMORY |
| 17775994 | 27,60,61,70 | 240G | 2-00:00:00 | FAILED (1:0) po ~2 s - tipkarska napaka v skripti, brez rezultatov |
| 17827352 | 27,60,61,70 | 120G | 12:00:00 | taski 27, 61, 70 uspešni; task 60 OUT_OF_MEMORY |
| 17829347 | 60 | 240G | 12:00:00 | uspešno - Devnagari-Script (40923) dokončan |

Preslikava indeks -> OpenML ID (scripts/cc18_ids.json):
27 -> 554 (mnist_784), 60 -> 40923 (Devnagari-Script),
61 -> 40927 (CIFAR_10), 70 -> 40996 (Fashion-MNIST)

## Spremembe kode med zagonom
Vsebina src/ se NI spreminjala - rezultati vseh 72 naborov izvirajo iz iste
različice benchmark kode. Spremenjena je bila le skripta za oddajo
scripts/run_cc18.sh:
1. Dodana možnost ALLOW_SPARSE_ARRAY=1, ki dovoli ponovno oddajo posameznih
   indeksov (prej je preverjanje velikosti arraya zavrnilo vsak nepopoln array).
2. Dodano preverjanje posameznega indeksa proti številu ID-jev.
3. Popravljena tipkarska napaka: --ids-file \$IDS_FIL -> --ids-file "\$IDS_FILE"
   (zaradi set -u je opravilo 17775994 padlo v 2 sekundah).

## Opozorilo glede časov
Stolpca train_time_s in inference_time_s za nabore 554, 40923, 40927 in 40996
izvirajo iz ponovnih zagonov na drug dan, na drugih vozliščih in z drugačno
dodeljeno količino pomnilnika. Vrednosti ROC-AUC to ne prizadene (fiksen
random_state, ista koda), časi pa med nabori niso neposredno primerljivi.

## Nadaljevanje po prekinitvi
Ponovni zagoni so se nadaljevali iz datotek <id>.csv.partial, zato noben
že opravljen fit ni bil ponovljen.
EOF
