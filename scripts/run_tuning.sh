#!/bin/bash
#SBATCH --job-name=tfm-tune
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/tune-%A_%a.out
#SBATCH --account=fri-users

# Poskus uglaševanja: XGBoost z iskanjem po mreži in gnezdenim prečnim
# preverjanjem (src/models/xgboost_tuned.py, mreža v config.yaml: tuning) na
# naborih scripts/cc18_tuning_ids.json - en nabor na array task.
#
# Seznam ima 68 naborov (CC18 brez štirih velikih slikovnih): indeksi 0-35 so
# NENASIČENI, 36-67 nasičeni. Najprej teče prvih 36, ostalih 32 kasneje pod ISTIM
# RUN_ID in z ISTIM commitom (runner preskoči že narejene nabore):
#     RUN_ID=cc18_xgb_tuned sbatch --array=0-35%8 scripts/run_tuning.sh
# Ocena (2026-10-15, mreža 128 nastavitev; iz časov privzetega XGBoost v cc18_v4 x
# razmerje uglaševanje / privzeto učenje 450-800 iz pilota, x 4/3 za learning_rate
# 0.01): prvih 36 skupaj ~28-49 h računanja, najdaljši task (Bioresponse) do ~5 h,
# zato --time=12:00:00. Pri ostalih 32 prevladujeta isolet (indeks 49, do ~62 h)
# in har (54, do ~21 h):
#     RUN_ID=cc18_xgb_tuned sbatch --array=36-48,50-53,55-67%8 scripts/run_tuning.sh
#     RUN_ID=cc18_xgb_tuned sbatch --array=49,54 --time=3-00:00:00 scripts/run_tuning.sh
# (časovne meje za drugi del še preveri z dejanskimi časi prvega dela na Arnesu)
#
# Samo CPU, brez GPU: XGBoost teče na 8 jedrih, zato particije ne določamo (privzeta
# particija gruče). Če jo sbatch zavrne, isti ukaz z
#     --partition=gpu --gres=gpu:1 --constraint=h100
# (vozlišča prejšnjih zagonov; GPU tedaj ostane neuporabljen).
#
# Vsak zunanji fold v istem procesu nauči tudi privzeti XGBoost: njegov ROC-AUC
# mora biti enak cc18_v4 (preverba) in njegov čas je izhodišče za ceno
# uglaševanja na istem vozlišču - zato vrsta vozlišča za primerjavo ni pomembna.
#
# Kontrolne točke po vsakem zunanjem foldu (per_dataset/<id>__xgboost_tuned.csv.partial);
# prekinjen task nadaljuje ob ponovni oddaji z istim RUN_ID.

set -euo pipefail

DATASET_SET=cc18_tuning
MAMBA="$HOME/bin/micromamba run -p ${TABULAR_ENV:-$HOME/envs/tabular3.5}"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

if [ -z "${RUN_ID:-}" ]; then
    echo "NAPAKA: nastavi RUN_ID (npr. RUN_ID=cc18_xgb_tuned), sicer bi vsaka oddaja dobila svojo mapo." >&2
    exit 1
fi

mkdir -p logs

# tail -n1: micromamba run lahko na stdout doda uvodne vrstice, zanima nas le število.
N_IDS=$($MAMBA python -c "from src.config import load_config, dataset_specs; print(len(dataset_specs(load_config(), '$DATASET_SET')))" | tail -n1)
if [ "$SLURM_ARRAY_TASK_ID" -ge "$N_IDS" ]; then
    echo "NAPAKA: indeks $SLURM_ARRAY_TASK_ID je izven obsega (0..$((N_IDS - 1)))." >&2
    exit 1
fi
echo "RUN_ID=$RUN_ID  (task $SLURM_ARRAY_TASK_ID od $N_IDS, nabor $DATASET_SET, algoritem xgboost_tuned)"

$MAMBA python -m src.run_one_dataset \
    --dataset-set "$DATASET_SET" --index "$SLURM_ARRAY_TASK_ID" --run-id "$RUN_ID" \
    --algorithms xgboost_tuned
