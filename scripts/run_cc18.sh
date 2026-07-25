#!/bin/bash
#SBATCH --job-name=tfm-cc18
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-71%4
#SBATCH --output=logs/cc18-%A_%a.out
#SBATCH --account=fri-users

# Celotna zbirka OpenML-CC18 (72 datasetov), en dataset na array task.
#
# --array=0-71: zgornja meja = število ID-jev v scripts/cc18_ids.json minus 1.
#   Preveri z:  python -c "import json; print(len(json.load(open('scripts/cc18_ids.json'))) - 1)"
#   Spodnje preverjanje v telesu skripte to ujame tudi ob oddaji.
#
# %4 (THROTTLE) = največ toliko taskov hkrati. TODO: potrdi na gruči, da je
#   vrednost <= zgornje meje GPU-jev na uporabnika, in jo po potrebi zvišaj:
#       sacctmgr show qos normal format=Name,MaxTRESPU%40,MaxJobsPU
#   Privzeto je konzervativno %4; polje je zaradi resume logike varno
#   ponovno oddati, zato prenizek throttle stane le čas, ne rezultatov.
#
# Dimenzioniranje --time in --mem (pilot je SPODNJA meja, ne zgornja):
#   Pilot (job 17731379, dataseti 31/37/38) je trajal ~10-60 s na task, ker so
#   ti dataseti drobni. CC18 vsebuje bistveno večje: CIFAR_10 (60000 x 3072),
#   Devnagari-Script (92000 x 1024), mnist_784 in Fashion-MNIST (70000 x 784).
#   Na task se izvede 6 algoritmov x 5 foldov = 30 učenj; na največjih
#   datasetih sta CatBoost (1000 iteracij) in RandomForest lahko po več deset
#   minut na fold, TabPFN/TabICL pa sta na GPU dolga repa (ali
#   pa fail-soft padeta na omejitvah velikosti - to je sprejemljivo in se
#   zabeleži). Vsi štirje ansambli uporabljajo vseh 8 dodeljenih jeder
#   (RF prek n_jobs=-1, ostali prek OMP_NUM_THREADS spodaj).
#   8 h na task je torej velikodušna rezerva; ker je polje
#   omejeno (throttle) in odporno na ponovni zagon, je preveliki --time
#   poceni, prekratki pa le podaljša skupni čas (ne izgubi dela - glej
#   kontrolne točke spodaj).
#   TODO: pred zagonom preveri zgornjo mejo particije in QOS ter --time
#   dvigni proti njej (manj ponovnih oddaj):
#       scontrol show partition gpu | grep -i maxtime
#       sacctmgr show qos normal format=Name,MaxWall
#
# Kontrolne točke: src/run_one_dataset.py po vsakem (algoritem, fold) učenju
#   zapiše <id>.csv.partial in šele na koncu atomarno preimenuje v <id>.csv.
#   Prekinjen task (prekoračen --time, preemption) torej ne izgubi dela -
#   ponovna oddaja nadaljuje pri prvem nenarejenem učenju. Tudi posamezen
#   dolg algoritem (CatBoost na Devnagari-Script) lahko svojih 5 foldov
#   razporedi čez več taskov, namesto da bi se v nedogled ponavljal.
#   --mem=64G: CIFAR_10 kot float64 zasede ~1,4 GiB na kopijo; predobdelava,
#   train/test razrez in CatBoostova kvantizacija držijo več kopij hkrati -
#   64G da ~40x rezerve nad surovo matriko. (Pilotni MaxRSS iz sacct je le
#   spodnja meja, glej results/arnes_cc18/PROVENANCE.md.)

set -euo pipefail

IDS_FILE=scripts/cc18_ids.json
MAMBA="$HOME/bin/micromamba run -p $HOME/envs/tabular"

# TABPFN_TOKEN za headless uporabo TabPFN v3
source ~/.tabpfn_token

export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p logs results/per_dataset

# Velikost polja mora ustrezati številu ID-jev, sicer bi zadnji dataseti tiho izpadli.
# tail -n1: micromamba run lahko na stdout doda uvodne vrstice, zanima nas le število.
N_IDS=$($MAMBA python -c "import json; print(len(json.load(open('$IDS_FILE'))))" | tail -n1)
if [ "${ALLOW_SPARSE_ARRAY:-0}" -ne 1 ] && [ "$SLURM_ARRAY_TASK_MAX" -ne "$((N_IDS - 1))" ]; then
    echo "NAPAKA: --array=0-$SLURM_ARRAY_TASK_MAX ne ustreza $N_IDS ID-jem v $IDS_FILE." >&2
    echo "Popravi direktivo #SBATCH --array na 0-$((N_IDS - 1)) in oddaj znova." >&2
    echo "Za ponovni zagon posameznih datasetov nastavi ALLOW_SPARSE_ARRAY=1." >&2
    exit 1
fi

# Vedno preveri posamezen indeks - velja tudi za redek array.
if [ "$SLURM_ARRAY_TASK_ID" -ge "$N_IDS" ]; then
    echo "NAPAKA: index $SLURM_ARRAY_TASK_ID je izven obsega (0..$((N_IDS - 1)))." >&2
    exit 1
fi

$MAMBA python -m src.run_one_dataset --index $SLURM_ARRAY_TASK_ID --ids-file "$IDS_FILE"
