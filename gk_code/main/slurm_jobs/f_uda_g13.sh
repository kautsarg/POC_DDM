#!/bin/bash
#SBATCH --job-name=f_uda_g13
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --array=0-5
#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

set -e

mkdir -p "logs/${SLURM_JOB_NAME}"

# Activate venv
source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate

export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
case "$GPU_NAME" in
    *A30*)  BATCH_SIZE=1024 ;;
    *A100*) BATCH_SIZE=1024 ;;
    *)      BATCH_SIZE=2048 ;;
esac

DOMAIN=binary
DOMAIN_WEIGHT=1.0

echo "  [*] GPU: ${GPU_NAME} -> batch_size=${BATCH_SIZE}"
echo "  [*] task ${SLURM_ARRAY_TASK_ID} -> held_out=${SLURM_ARRAY_TASK_ID} domain=${DOMAIN}"

python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/uda_dann_lofo_training.py \
    --batch_size ${BATCH_SIZE} \
    --held_out ${SLURM_ARRAY_TASK_ID} \
    --domain ${DOMAIN} \
    --domain_weight ${DOMAIN_WEIGHT}

deactivate
