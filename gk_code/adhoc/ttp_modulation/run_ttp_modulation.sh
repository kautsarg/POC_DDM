#!/bin/bash
#SBATCH --job-name=ttp_modulation
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --array=0-1
#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

set -e

mkdir -p "logs/${SLURM_JOB_NAME}"

source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate

SCRIPT_DIR=/vol/bitbucket/gk225/POC_DDM/gk_code/main/adhoc/ttp_modulation
DATASET_DIR="${SCRIPT_DIR}/dataset"

DATASETS=(
    "${DATASET_DIR}/area_strategy.xlsx"   # index 0
    "${DATASET_DIR}/range_strategy.csv"   # index 1
)

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

echo "Task ${SLURM_ARRAY_TASK_ID}: ${DATASET}"

python -u "${SCRIPT_DIR}/main.py" \
    --dataset "${DATASET}" \
    --one_to_one \
    --ttp_aligned \
    --normalise_curve

deactivate
