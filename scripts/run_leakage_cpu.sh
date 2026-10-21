#!/bin/bash
#SBATCH --job-name=tfm-leak-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/leak-cpu-%A_%a.out
#SBATCH --account=fri-users

# Test uhajanja podatkov - del brez GPU: CatBoost na referenčnem naboru Medic3-160
# (CLAUDE.md, What is next, korak 8; zasnova v results/runs/cc18_leakage/DESIGN.md).
#
# Zakaj posebej: CatBoost na Medic3-160 potrebuje 1,5-2,3 h na fold (medic3_160:
# 1,54 h na gwn09, 2,26 h na deljenem gwn08), skupaj 15 učenj (3 različice x 5
# foldov). Grafične kartice ne uporablja, zato teče na privzeti CPU particiji
# (all, največ 2 dni), EN ARRAY TASK = ENA RAZLIČICA (indeks = mesto opisa v
# seznamu), da trije taski tečejo hkrati (~8-11 h vsak; na počasnem vozlišču
# wn116/wn119, ki je bil pri uglaševanju ~2,6-krat počasnejši, še vedno pod 2 dni).
# Različice so lahko na različnih vozliščih, ker je CatBoost na CPU ponovljiv
# med stroji (Kremen in Arnes, cc18_v2 in cc18_v3: enak ROC-AUC na bit natančno).
#
# ODDAJA (iz korena repozitorija, po git pull):
#     DATASET_SET=medic3_160_leakage RUN_ID=medic3_160_leakage sbatch --array=0-2 scripts/run_leakage_cpu.sh
# Temeljna modela na istem naboru: scripts/run_leakage.sh (točka b v njegovi glavi).
# Ponovna oddaja po prekinitvi: isti ukaz, ISTI RUN_ID (runner nadaljuje).

set -euo pipefail

DATASET_SET="${DATASET_SET:-medic3_160_leakage}"
ALGORITHMS="${ALGORITHMS:-catboost}"
MAMBA="$HOME/bin/micromamba run -p ${TABULAR_ENV:-$HOME/envs/tabular3.5}"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

if [ -z "${RUN_ID:-}" ]; then
    echo "NAPAKA: nastavi RUN_ID (npr. RUN_ID=medic3_160_leakage), sicer bi vsaka oddaja dobila svojo mapo." >&2
    exit 1
fi

mkdir -p logs
read -r -a ALGO_LIST <<< "$ALGORITHMS"

# tail -n1: micromamba run lahko na stdout doda uvodne vrstice, zanima nas le število.
N_SPECS=$($MAMBA python -c "from src.config import load_config, dataset_specs; print(len(dataset_specs(load_config(), '$DATASET_SET')))" | tail -n1)
if [ "$SLURM_ARRAY_TASK_ID" -ge "$N_SPECS" ]; then
    echo "NAPAKA: indeks $SLURM_ARRAY_TASK_ID je izven obsega (0..$((N_SPECS - 1)))." >&2
    exit 1
fi

echo "RUN_ID=$RUN_ID  (opis $SLURM_ARRAY_TASK_ID od $N_SPECS v $DATASET_SET, algoritmi: $ALGORITHMS)"
$MAMBA python -m src.run_one_dataset \
    --dataset-set "$DATASET_SET" --index "$SLURM_ARRAY_TASK_ID" --run-id "$RUN_ID" \
    --algorithms "${ALGO_LIST[@]}"
