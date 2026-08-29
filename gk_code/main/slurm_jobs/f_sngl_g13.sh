#!/bin/bash
#SBATCH --job-name=f_sngl_g13
#SBATCH --time=36:00:00

# Request resources for a single array task
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --array=4-9

# Output and Error logs (using SLURM variables to prevent overwriting)
#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

set -e

mkdir -p "logs/${SLURM_JOB_NAME}"

# Activate venv
source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate
export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main

RAW_EXP_FOLDER="/vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_final"
# EXP_FOLDER="${RAW_EXP_FOLDER}_nc_subtract"
EXP_FOLDER="${RAW_EXP_FOLDER}"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
case "$GPU_NAME" in
    *A30*) BATCH_SIZE=1024; AE_BATCH_SIZE=128 ;;
    *)     BATCH_SIZE=2048; AE_BATCH_SIZE=256 ;;
esac
echo "  [*] GPU: ${GPU_NAME}  ->  batch_size=${BATCH_SIZE}  lstm_ae_batch_size=${AE_BATCH_SIZE}"


python -u 01_curve_preprocessing_v6.py \
    --task_id $SLURM_ARRAY_TASK_ID --exp_folder ${RAW_EXP_FOLDER} \
    --n_wells 10 --n_a_type v06 --sg_p4 --normalize_curves --drop_pc
    # --nc_subtract

python -u 02_outlier_detection_pipeline.py \
    --task_id $SLURM_ARRAY_TASK_ID --exp_folder ${EXP_FOLDER} \
    --filters

python -u ablations/ablation6_chip_outlier_model_ablation.py \
    --task_id $SLURM_ARRAY_TASK_ID --exp_folder ${EXP_FOLDER} \
    --curve_type ori_curve_sg_p4_norm \
    --n_splits 5 \
    --batch_size ${BATCH_SIZE}

deactivate
