#!/bin/bash
#SBATCH --job-name=tfm-leak
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/leak-%A_%a.out
#SBATCH --account=fri-users

# Test uhajanja podatkov (CLAUDE.md, What is next, korak 8; zasnova, utemeljitve
# in viri v results/runs/cc18_leakage/DESIGN.md). Vsak nabor teče v treh
# različicah (config.yaml, leakage: arms): original, monotone, monotone_noise.
#
# EN ARRAY TASK = EN NABOR v vseh različicah, zaporedoma na ISTEM vozlišču: tako
# razlika med različicami ni razlika med GPU-ji (TabPFN se med vrstami H100 in
# RTX 3060 razlikuje do ~1e-3). Opisi v seznamu so urejeni po naborih (glej
# scripts/gen_leakage_ids.py), zato task i požene opise i*R ... i*R+R-1, kjer je R
# število različic. Vsaka različica je svoj klic src.run_one_dataset in piše svojo
# datoteko per_dataset/<id>_<različica>__<algoritmi>.csv.
#
# ODDAJA (iz korena repozitorija, po git pull; vozlišča so brez interneta, nabori
# CC18 in uteži TabPFN/TabICL so že predpomnjeni, Medic3.csv je na gruči):
#   a) CC18: 34 naborov x 3 različice x CatBoost, TabPFN-3.5, TabICL x 5 foldov
#      (%4 = največ štirje GPU hkrati, kot pri run_cc18.sh):
#        RUN_ID=cc18_leakage sbatch --array=0-33%4 scripts/run_leakage.sh
#   b) Medic3-160, oba temeljna modela (TabICL potrebuje ~92 GiB RAM, glej
#      scripts/run_medic3.sh, zato --mem=160G):
#        DATASET_SET=medic3_160_leakage ALGORITHMS="tabpfn tabicl" RUN_ID=medic3_160_leakage \
#            sbatch --array=0 --mem=160G scripts/run_leakage.sh
#   c) Medic3-160, CatBoost (brez GPU): scripts/run_leakage_cpu.sh
#
# Ocena (časi ponovitve 0 iz cc18_v4/combined, H100 in 8 jeder): CC18 skupaj
# ~1,4 h računanja (CatBoost 0,73 h, TabPFN 0,36 h, TabICL 0,31 h), najdaljši
# task numerai28.6 ~0,2 h; Medic3 temeljna modela ~1 h (15 + 15 učenj).
#
# Algoritmi: privzeto config.yaml (leakage: algorithms), drugače ALGORITHMS="a b".
# Ponovna oddaja po prekinitvi: isti ukaz, ISTI RUN_ID (runner nadaljuje pri prvem
# nenarejenem učenju, dokončane različice preskoči).

set -euo pipefail

DATASET_SET="${DATASET_SET:-cc18_leakage}"
MAMBA="$HOME/bin/micromamba run -p ${TABULAR_ENV:-$HOME/envs/tabular3.5}"

# TABPFN_TOKEN za uporabo utež TabPFN brez brskalnika
source ~/.tabpfn_token

export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

if [ -z "${RUN_ID:-}" ]; then
    echo "NAPAKA: nastavi RUN_ID (npr. RUN_ID=cc18_leakage), sicer bi vsaka oddaja dobila svojo mapo." >&2
    exit 1
fi

mkdir -p logs

# tail -n1: micromamba run lahko na stdout doda uvodne vrstice, zanima nas le zadnja.
if [ -z "${ALGORITHMS:-}" ]; then
    ALGORITHMS=$($MAMBA python -c "from src.config import load_config; print(' '.join(load_config()['leakage']['algorithms']))" | tail -n1)
fi
read -r -a ALGO_LIST <<< "$ALGORITHMS"
read -r N_ARMS N_SPECS <<< "$($MAMBA python -c "from src.config import load_config, dataset_specs; c = load_config(); print(len(c['leakage']['arms']), len(dataset_specs(c, '$DATASET_SET')))" | tail -n1)"

if [ $((N_SPECS % N_ARMS)) -ne 0 ]; then
    echo "NAPAKA: $N_SPECS opisov v '$DATASET_SET' ni deljivo s $N_ARMS različicami - poženi scripts/gen_leakage_ids.py." >&2
    exit 1
fi
N_DATASETS=$((N_SPECS / N_ARMS))
if [ "${ALLOW_SPARSE_ARRAY:-0}" -ne 1 ] && [ "$SLURM_ARRAY_TASK_MAX" -ne "$((N_DATASETS - 1))" ]; then
    echo "NAPAKA: --array=0-$SLURM_ARRAY_TASK_MAX ne ustreza $N_DATASETS naborom v '$DATASET_SET'." >&2
    echo "Oddaj z --array=0-$((N_DATASETS - 1)); za posamezne nabore nastavi ALLOW_SPARSE_ARRAY=1." >&2
    exit 1
fi
if [ "$SLURM_ARRAY_TASK_ID" -ge "$N_DATASETS" ]; then
    echo "NAPAKA: indeks $SLURM_ARRAY_TASK_ID je izven obsega (0..$((N_DATASETS - 1)))." >&2
    exit 1
fi

echo "RUN_ID=$RUN_ID  (nabor $SLURM_ARRAY_TASK_ID od $N_DATASETS v $DATASET_SET, $N_ARMS različice, algoritmi: $ALGORITHMS)"
for ((ARM = 0; ARM < N_ARMS; ARM++)); do
    $MAMBA python -m src.run_one_dataset \
        --dataset-set "$DATASET_SET" --index "$((SLURM_ARRAY_TASK_ID * N_ARMS + ARM))" \
        --run-id "$RUN_ID" --algorithms "${ALGO_LIST[@]}"
done
