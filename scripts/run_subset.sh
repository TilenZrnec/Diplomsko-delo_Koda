#!/bin/bash
#SBATCH --job-name=tfm-subset
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-2
#SBATCH --output=logs/subset-%A_%a.out
#SBATCH --account=fri-users

# Pilotna trojica (subset: 31/37/38) na gruči - hitra validacija novega okolja,
# preden se odda celoten CC18. Ista pot kot run_cc18.sh, le manjša.
#
# Oddaja:   RUN_ID=subset_v3 sbatch scripts/run_subset.sh
# Primerjava po združitvi (cc18_v2 vsebuje iste tri nabore z istimi foldi in semeni):
#   python scripts/merge_results.py subset_v3
#   python scripts/compare_results.py cc18_v2 subset_v3

set -euo pipefail

DATASET_SET=subset

# TABPFN_TOKEN za prenos in uporabo utež TabPFN brez brskalnika
source ~/.tabpfn_token

export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

RUN_ID="${RUN_ID:-${DATASET_SET}_${SLURM_ARRAY_JOB_ID}}"
echo "RUN_ID=$RUN_ID  (task $SLURM_ARRAY_TASK_ID, nabor $DATASET_SET)"

mkdir -p logs

$HOME/bin/micromamba run -p "${TABULAR_ENV:-$HOME/envs/tabular3.5}" \
    python -m src.run_one_dataset \
        --dataset-set "$DATASET_SET" --index "$SLURM_ARRAY_TASK_ID" --run-id "$RUN_ID"
