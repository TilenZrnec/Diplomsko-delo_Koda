#!/bin/bash
#SBATCH --job-name=tfm-medic3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/medic3-%A_%a.out
#SBATCH --account=fri-users

# Zaupni nabor Medic3 (122 093 x 275, 220 razredov, 82 % manjkajočih vrednosti).
#
# ZAKAJ DRUGAČE KOT PRI CC18: Medic3 je EN nabor, zato bi ga shema "en nabor na
# array task" stlačila v eno samo opravilo, v katerem bi vseh šest algoritmov
# teklo zaporedno (ocena po CC18: ~60 h, čez mejo --time). Tu je zato en
# ALGORITEM na array task; indeks je mesto algoritma v config.yaml (algorithms):
#     0 random_forest   1 xgboost   2 lightgbm   3 catboost   4 tabpfn   5 tabicl
# src/runner.py ob izrecnem --algorithms piše v ločeno datoteko
# per_dataset/<nabor>__<algoritem>.csv, zato si vzporedni taski ne prepisujejo
# kontrolnih točk; merge_results.py vse skupaj združi v en results.csv.
#
# DVE RAZLIČICI NABORA (spremenljivka DATASET_SET, privzeto medic3):
#   medic3      - vseh 220 razredov. TabPFN-3.5 ima (kot TabPFN-3) trdo mejo
#                 160 razredov (izmerjeno 2026-10-06), zato TabPFN tu mehko
#                 odpove in ima v results.csv 15 vrstic z razlogom v stolpcu
#                 error. To je rezultat, ne okvara - ne popravljaj ga.
#   medic3_160  - samo 160 najpogostejših razredov (scripts/medic3_160.json,
#                 ključ max_classes); tu stečejo vsi štirje ansambli IN oba
#                 temeljna modela.
#
# STATISTIKA: python -m src.stats <RUN_ID> zazna, da je v zagonu en sam nabor, in
# namesto Friedmana/Nemenyija/Wilcoxona naredi popravljene t-teste za ponovljeno
# prečno preverjanje (Bouckaert in Frank 2004) s Holmovim popravkom.
#
# ODDAJA (iz korena repozitorija). CatBoost gre v svoje opravilo z daljšim
# časom: pri 220 razredih je po oceni iz CC18 (Devnagari-Script, 46 razredov,
# 28,5 h) daleč najdražji, ostali skupaj ne dosežejo niti 10 h.
#     ALLOW_SPARSE_ARRAY=1 RUN_ID=medic3_raw sbatch --array=0,1,2,4,5 scripts/run_medic3.sh
#     ALLOW_SPARSE_ARRAY=1 RUN_ID=medic3_raw sbatch --array=3 --time=1-12:00:00 --mem=120G scripts/run_medic3.sh
# Za različico s 160 razredi isto, le z drugim naborom in RUN_ID:
#     DATASET_SET=medic3_160 ALLOW_SPARSE_ARRAY=1 RUN_ID=medic3_160 sbatch --array=0,1,2,4,5 scripts/run_medic3.sh
#     DATASET_SET=medic3_160 ALLOW_SPARSE_ARRAY=1 RUN_ID=medic3_160 sbatch --array=3 --time=1-12:00:00 --mem=120G scripts/run_medic3.sh
#
# PONOVNA ODDAJA po prekoračenem --time: isti ukaz, ISTI RUN_ID. Kontrolne točke
# so po posameznem učenju, zato CatBoost nadaljuje pri prvem nenarejenem foldu.
# Računaj, da bo CatBoost verjetno potreboval eno ali dve ponovni oddaji.
#
# PODATKI: data/medic3/Medic3.csv glede na koren repozitorija - ISTA relativna
# pot na vseh strojih in na gruči. Mapa data/ je v .gitignore, zato datoteka
# ostane zaupna - prenesi jo s scp in zapiši sha256 v PROVENANCE.md zagona. Predpriprava (prestage.py) ni potrebna, ker nabor ni z OpenML; uteži
# TabPFN/TabICL pa morajo biti predpomnjene, ker vozlišča tečejo offline.

set -euo pipefail

DATASET_SET="${DATASET_SET:-medic3}"
MAMBA="$HOME/bin/micromamba run -p ${TABULAR_ENV:-$HOME/envs/tabular3.5}"

# TABPFN_TOKEN za prenos in uporabo utež TabPFN brez brskalnika
source ~/.tabpfn_token

export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

RUN_ID="${RUN_ID:-${DATASET_SET}_${SLURM_ARRAY_JOB_ID}}"

mkdir -p logs

# Seznam algoritmov beremo iz config.yaml, da ni nikjer podvojen (edini vir
# parametrov). tail -n1: micromamba run lahko na stdout doda uvodne vrstice.
N_ALGOS=$($MAMBA python -c "from src.config import load_config; print(len(load_config()['algorithms']))" | tail -n1)
if [ "${ALLOW_SPARSE_ARRAY:-0}" -ne 1 ] && [ "$SLURM_ARRAY_TASK_MAX" -ne "$((N_ALGOS - 1))" ]; then
    echo "NAPAKA: --array=0-$SLURM_ARRAY_TASK_MAX ne ustreza $N_ALGOS algoritmom v config.yaml." >&2
    echo "Popravi direktivo #SBATCH --array na 0-$((N_ALGOS - 1)) ali nastavi ALLOW_SPARSE_ARRAY=1." >&2
    exit 1
fi
if [ "$SLURM_ARRAY_TASK_ID" -ge "$N_ALGOS" ]; then
    echo "NAPAKA: index $SLURM_ARRAY_TASK_ID je izven obsega (0..$((N_ALGOS - 1)))." >&2
    exit 1
fi

ALGO=$($MAMBA python -c "from src.config import load_config; print(load_config()['algorithms'][$SLURM_ARRAY_TASK_ID])" | tail -n1)
echo "RUN_ID=$RUN_ID  (task $SLURM_ARRAY_TASK_ID, nabor $DATASET_SET, algoritem $ALGO)"

# --index 0: nabora medic3 in medic3_160 imata vsak natanko en opis nabora.
$MAMBA python -m src.run_one_dataset \
    --dataset-set "$DATASET_SET" --index 0 --run-id "$RUN_ID" --algorithms "$ALGO"
