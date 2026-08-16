#!/bin/bash
#SBATCH --job-name=fnl_lofo_crossval
#SBATCH --time=72:00:00

# Request resources for a single job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --array=2-3   # 0=acq_start+SC3, 1=acq_start+SC0, 2=pc_ttp_min+SC3, 3=pc_ttp_min+SC0

# Output and Error logs (using SLURM variables to prevent overwriting)
#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

set -e

mkdir -p "logs/${SLURM_JOB_NAME}"

# Activate venv
source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate

export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main

EXP_FOLDER="/vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_final_nc_subtract/"

MODELS="knn cnn_gru_dual cnn_gru_dual_attn_recon"
CURVE_TYPES="ori_curve_wavelet_bior35_norm ori_curve_norm" # ori_curve_norm or ori_curve_wavelet_bior35_norm

####################################################################################################
### none or lofo_ae [N/L]
FILTERS="none"
# FILTERS="lofo_ae"

#### CHECK THE GROUP!
LOFO_TASK_ID=5
GROUP_NAME=$(python3 -c "import config; print(list(config.CROSS_DATASET_GROUPS.keys())[${LOFO_TASK_ID}])" | tail -n 1)
####################################################################################################

ALIGN_ARGS_LIST=("" "" "--curve_alignment pc_ttp --pc_ttp_anchor min" "--curve_alignment pc_ttp --pc_ttp_anchor min")
SUPCON_LIST=(3 0 3 0)
ALIGN_ARGS="${ALIGN_ARGS_LIST[$SLURM_ARRAY_TASK_ID]}"
SUPCON="${SUPCON_LIST[$SLURM_ARRAY_TASK_ID]}"

if [ "$SUPCON" -eq 3 ]; then
    DANN_MODELS="cnn_gru_dual_supcon3_dann cnn_gru_dual_attn_recon_supcon3_dann"
    CORAL_MODELS="cnn_gru_dual_supcon3_coral cnn_gru_dual_attn_recon_supcon3_coral"
else
    DANN_MODELS="cnn_gru_dual_dann cnn_gru_dual_attn_recon_dann"
    CORAL_MODELS="cnn_gru_dual_coral cnn_gru_dual_attn_recon_coral"
fi

python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/04_cross_dataset_training.py \
    --exp_folder ${EXP_FOLDER} --task_id ${LOFO_TASK_ID} \
    --coral --supcon ${SUPCON} --curve_type ${CURVE_TYPES} --models ${CORAL_MODELS} \
    --outlier_filter ${FILTERS} --train_full ${ALIGN_ARGS}

python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/04_cross_dataset_training.py \
    --exp_folder ${EXP_FOLDER} --task_id ${LOFO_TASK_ID} \
    --supcon ${SUPCON} --curve_type ${CURVE_TYPES} --models ${MODELS} \
    --outlier_filter ${FILTERS} --train_full ${ALIGN_ARGS}

python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/04_cross_dataset_training.py \
    --exp_folder ${EXP_FOLDER} --task_id ${LOFO_TASK_ID} \
    --dann --supcon ${SUPCON} --curve_type ${CURVE_TYPES} --models ${DANN_MODELS} \
    --outlier_filter ${FILTERS} --train_full ${ALIGN_ARGS}

python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/06b_cross_dataset_prediction_report.py \
    --exp_folder ${EXP_FOLDER} --group ${GROUP_NAME} \
    --mode lofo \
    --curve_type ${CURVE_TYPES} --outlier_filter ${FILTERS}  ${ALIGN_ARGS}

deactivate
