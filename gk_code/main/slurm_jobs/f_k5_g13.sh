#!/bin/bash
#SBATCH --job-name=f_k5_g13
#SBATCH --time=72:00:00

# Request resources for a single array task
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --array=0-1

# Output and Error logs (using SLURM variables to prevent overwriting)
#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

mkdir -p "logs/${SLURM_JOB_NAME}"

# Activate venv
source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate

export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main

# EXP_FOLDER="/vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_final_nc_subtract/"
EXP_FOLDER="/vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_final/"
FILTERS="noamp_remove"
TASK_ID=13  #### ===>  13 (nc vs non nc?
GROUP_NAME=$(python3 -c "import config; print(list(config.CROSS_DATASET_GROUPS.keys())[${TASK_ID}])" | tail -n 1)
# CURVE_TYPES="ori_curve_norm ori_curve_sg_p4_norm"
CURVE_TYPES="ori_curve_sg_p4_norm"
SUPCON=0


case "$SLURM_ARRAY_TASK_ID" in
    0) MODELS="cnn_gru_dual" ;;
    1) MODELS="cnn_gru_dual_attn_recon" ;;
esac

python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/04_cross_dataset_training.py \
    --exp_folder ${EXP_FOLDER} --task_id ${TASK_ID} \
    --mode kfold --n_splits 5 \
    --supcon ${SUPCON} --curve_type ${CURVE_TYPES} --models ${MODELS} \
    --outlier_filter ${FILTERS} --batch_size 2048 --train_center_frac 0.5

# python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/06b_cross_dataset_prediction_report.py \
#     --exp_folder ${EXP_FOLDER} --group ${GROUP_NAME} \
#     --mode kfold --n_splits 5 \
#     --curve_type ${CURVE_TYPES} --outlier_filter None

deactivate
