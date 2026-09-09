#!/bin/bash
#SBATCH --job-name=tfm-cc18
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-71%4
#SBATCH --output=logs/cc18-%A_%a.out
#SBATCH --account=fri-users

# Celotna zbirka OpenML-CC18 (72 naborov), en nabor na array task.
#
# ODDAJA (iz korena repozitorija, z izbrano oznako zagona):
#     RUN_ID=cc18_v2 sbatch scripts/run_cc18.sh
# Brez RUN_ID se oznaka izpelje iz ID-ja polja (cc18_<SLURM_ARRAY_JOB_ID>) in
# se izpiše v dnevnik. Rezultati gredo v results/runs/<RUN_ID>/per_dataset/.
#
# PONOVNA ODDAJA po prekinitvi ali za posamezne nabore: uporabi ISTI RUN_ID,
# sicer začne nov zagon od začetka namesto da bi nadaljeval iz partialov:
#     ALLOW_SPARSE_ARRAY=1 RUN_ID=cc18_v2 sbatch --array=27,60,61,70 --mem=240G scripts/run_cc18.sh
#
# --array=0-71: zgornja meja = število naborov v cc18 (config.yaml: dataset_sets)
#   minus 1. Varovalka spodaj to preveri ob zagonu vsakega taska.
#
# %4 (THROTTLE) = največ toliko taskov hkrati; polje je zaradi resume logike
#   varno ponovno oddati, zato prenizek throttle stane le čas, ne rezultatov.
#
# Dimenzioniranje (izkušnja iz zagona 2026-08, glej results/arnes/cc18/PROVENANCE.md):
#   --mem=64G je zadoščal za 68/72 naborov; indeksi 27 (mnist_784), 60
#   (Devnagari-Script), 61 (CIFAR_10) in 70 (Fashion-MNIST) so potrebovali
#   120-240G. --time=12:00:00 je zadoščal povsod. Na task se izvede
#   6 algoritmov x n_splits x n_repeats učenj (config.yaml).
#
# Kontrolne točke: src/runner.py po vsakem (algoritem, fold) učenju zapiše
#   <id>.csv.partial in šele na koncu atomarno preimenuje v <id>.csv. Prekinjen
#   task torej ne izgubi dela - ponovna oddaja z istim RUN_ID nadaljuje pri
#   prvem nenarejenem učenju.

set -euo pipefail

DATASET_SET=cc18
# Okolje na gruči: privzeto ~/envs/tabular2 (Python 3.12, requirements.txt);
# drugo pot podaš s TABULAR_ENV=... pred sbatch.
MAMBA="$HOME/bin/micromamba run -p ${TABULAR_ENV:-$HOME/envs/tabular2}"

# TABPFN_TOKEN za headless uporabo TabPFN v3
source ~/.tabpfn_token

export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

RUN_ID="${RUN_ID:-${DATASET_SET}_${SLURM_ARRAY_JOB_ID}}"
echo "RUN_ID=$RUN_ID  (task $SLURM_ARRAY_TASK_ID, nabor $DATASET_SET)"

mkdir -p logs

# Velikost polja mora ustrezati številu naborov, sicer bi zadnji tiho izpadli.
# tail -n1: micromamba run lahko na stdout doda uvodne vrstice, zanima nas le število.
N_IDS=$($MAMBA python -c "from src.config import load_config, dataset_specs; print(len(dataset_specs(load_config(), '$DATASET_SET')))" | tail -n1)
if [ "${ALLOW_SPARSE_ARRAY:-0}" -ne 1 ] && [ "$SLURM_ARRAY_TASK_MAX" -ne "$((N_IDS - 1))" ]; then
    echo "NAPAKA: --array=0-$SLURM_ARRAY_TASK_MAX ne ustreza $N_IDS naborom v '$DATASET_SET'." >&2
    echo "Popravi direktivo #SBATCH --array na 0-$((N_IDS - 1)) in oddaj znova." >&2
    echo "Za ponovni zagon posameznih naborov nastavi ALLOW_SPARSE_ARRAY=1." >&2
    exit 1
fi

# Vedno preveri posamezen indeks - velja tudi za redek array.
if [ "$SLURM_ARRAY_TASK_ID" -ge "$N_IDS" ]; then
    echo "NAPAKA: index $SLURM_ARRAY_TASK_ID je izven obsega (0..$((N_IDS - 1)))." >&2
    exit 1
fi

$MAMBA python -m src.run_one_dataset \
    --dataset-set "$DATASET_SET" --index "$SLURM_ARRAY_TASK_ID" --run-id "$RUN_ID"
