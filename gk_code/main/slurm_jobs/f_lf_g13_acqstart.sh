#!/bin/bash
#SBATCH --job-name=f_lf_g13_acqstart
#SBATCH --time=72:00:00

# Request resources for a single array task
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --array=0-6
# Output and Error logs (using SLURM variables to prevent overwriting)
#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

# Same as f_lf_g13.sh, but only tasks 0-6 (drops 7=coral, 8=dann_conc,
# 9=cnn_trans_dual_attn_recon) and ALIGN_ARGS uses acquisition_start instead of pc_ttp.

set -e

mkdir -p "logs/${SLURM_JOB_NAME}"

# Activate venv
source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate

export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main

EXP_FOLDER="/vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_final/"

CURVE_TYPES="ori_curve_sg_p4_norm"
FILTERS="noamp_remove"
ALIGN_ARGS="--curve_alignment acquisition_start"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
case "$GPU_NAME" in
    *A30*) BATCH_SIZE=1024 ;;
    *)     BATCH_SIZE=2048 ;;
esac
echo "  [*] GPU: ${GPU_NAME}  ->  batch_size=${BATCH_SIZE}"


LOFO_TASK_ID=13
GROUP_NAME=$(python3 -c "import config; print(list(config.CROSS_DATASET_GROUPS.keys())[${LOFO_TASK_ID}])" | tail -n 1)


case "$SLURM_ARRAY_TASK_ID" in
    0)
        python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/04_cross_dataset_training.py \
            --exp_folder ${EXP_FOLDER} --task_id ${LOFO_TASK_ID} \
            --supcon 0 --curve_type ${CURVE_TYPES} --models cnn_gru_dual_attn_recon \
            --outlier_filter ${FILTERS} --batch_size ${BATCH_SIZE} ${ALIGN_ARGS} --train_center_frac 0.5
        ;;
    1)
        python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/04_cross_dataset_training.py \
            --exp_folder ${EXP_FOLDER} --task_id ${LOFO_TASK_ID} \
            --supcon 0 --curve_type ${CURVE_TYPES} --models cnn_gru_dual_attn_recon_aug \
            --outlier_filter ${FILTERS} --batch_size ${BATCH_SIZE} ${ALIGN_ARGS} --train_center_frac 0.5
        ;;
    2)
        python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/04_cross_dataset_training.py \
            --exp_folder ${EXP_FOLDER} --task_id ${LOFO_TASK_ID} \
            --dann --supcon 0 --curve_type ${CURVE_TYPES} --models cnn_gru_dual_attn_recon_dann \
            --outlier_filter ${FILTERS} --batch_size ${BATCH_SIZE} ${ALIGN_ARGS} --train_center_frac 0.5
        ;;
    3)
        python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/04_cross_dataset_training.py \
            --exp_folder ${EXP_FOLDER} --task_id ${LOFO_TASK_ID} \
            --supcon 3 --curve_type ${CURVE_TYPES} --models cnn_gru_dual_attn_recon_supcon3 \
            --outlier_filter ${FILTERS} --batch_size ${BATCH_SIZE} ${ALIGN_ARGS} --train_center_frac 0.5
        ;;
    4)
        python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/04_cross_dataset_training.py \
            --exp_folder ${EXP_FOLDER} --task_id ${LOFO_TASK_ID} \
            --supcon 0 --curve_type ${CURVE_TYPES} --models cnn_gru_dual \
            --outlier_filter ${FILTERS} --batch_size ${BATCH_SIZE} ${ALIGN_ARGS} --train_center_frac 0.5
        ;;
    5)
        python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/04_cross_dataset_training.py \
            --exp_folder ${EXP_FOLDER} --task_id ${LOFO_TASK_ID} \
            --supcon 0 --curve_type ${CURVE_TYPES} --models knn \
            --outlier_filter ${FILTERS} --batch_size ${BATCH_SIZE} ${ALIGN_ARGS} --train_center_frac 0.5
        ;;
    6)
        python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/04_cross_dataset_training.py \
            --exp_folder ${EXP_FOLDER} --task_id ${LOFO_TASK_ID} \
            --mtl --supcon 0 --curve_type ${CURVE_TYPES} --models cnn_gru_dual_attn_recon_mtl \
            --outlier_filter ${FILTERS} --batch_size ${BATCH_SIZE} ${ALIGN_ARGS} --train_center_frac 0.5
        ;;
esac

deactivate
