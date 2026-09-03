#!/bin/bash
#SBATCH --job-name=f_lwo_g13_v1
#SBATCH --time=12:00:00

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --array=0-6
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
    *A30*) BATCH_SIZE=1024 ;;
    *)     BATCH_SIZE=2048 ;;
esac
echo "  [*] GPU: ${GPU_NAME}  ->  batch_size=${BATCH_SIZE}"

COMMON="--batch_size ${BATCH_SIZE} --combo_tag v1"

case "$SLURM_ARRAY_TASK_ID" in
    0) python -u lowo_custom_training.py ${COMMON} --model cnn_gru_dual_attn_recon ;;
    1) python -u lowo_custom_training.py ${COMMON} --model cnn_gru_dual_attn_recon_aug ;;
    2) python -u lowo_custom_training.py ${COMMON} --model cnn_gru_dual_attn_recon_dann ;;
    3) python -u lowo_custom_training.py ${COMMON} --model cnn_gru_dual_attn_recon_supcon3 ;;
    4) python -u lowo_custom_training.py ${COMMON} --model cnn_gru_dual ;;
    5) python -u lowo_custom_training.py ${COMMON} --model knn ;;
    6) python -u lowo_custom_training.py ${COMMON} --model cnn_gru_dual_attn_recon_mtl --mtl ;;
    7) python -u lowo_custom_training.py ${COMMON} --model cnn_gru_dual_attn_recon_coral ;;
    8) python -u lowo_custom_training.py ${COMMON} --model cnn_gru_dual_attn_recon_dann_conc ;;
    9) python -u lowo_custom_training.py ${COMMON} --model cnn_trans_dual_attn_recon ;;
esac

deactivate
