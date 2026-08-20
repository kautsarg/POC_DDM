#!/bin/bash
#SBATCH --job-name=fnl_4_lofo_nn_filter
#SBATCH --time=72:00:00

# Request resources for a single job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --array=0-1   # 0=base+dann, 1=coral+supcon3 (2 parallel tasks)

# Output and Error logs (using SLURM variables to prevent overwriting)
#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

set -e

mkdir -p "logs/${SLURM_JOB_NAME}"

source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate

export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main

EXP_FOLDER="/vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_final_nc_subtract/"

BASE_MODELS="cnn_gru_dual cnn_gru_dual_attn_recon"
DANN_MODELS="cnn_gru_dual_dann cnn_gru_dual_attn_recon_dann"
CORAL_MODELS="cnn_gru_dual_coral cnn_gru_dual_attn_recon_coral"

CURVE_TYPES="ori_curve_sg_p4_norm"

FILTERS="noamp_remove"

ALIGN_ARGS="--curve_alignment pc_ttp --pc_ttp_anchor min"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
case "$GPU_NAME" in
    *A30*) BATCH_SIZE=1024 ;;
    *)     BATCH_SIZE=2048 ;;
esac
echo "  [*] GPU: ${GPU_NAME}  ->  batch_size=${BATCH_SIZE}"

LOFO_TASK_ID=4 # 4: final_4_chip_cleanv2_nn
GROUP_NAME=$(python3 -c "import config; print(list(config.CROSS_DATASET_GROUPS.keys())[${LOFO_TASK_ID}])" | tail -n 1)

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/04_cross_dataset_training.py \
        --exp_folder ${EXP_FOLDER} --task_id ${LOFO_TASK_ID} \
        --supcon 0 --curve_type ${CURVE_TYPES} --models ${BASE_MODELS} \
        --outlier_filter ${FILTERS} --batch_size ${BATCH_SIZE} ${ALIGN_ARGS}

    python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/04_cross_dataset_training.py \
        --exp_folder ${EXP_FOLDER} --task_id ${LOFO_TASK_ID} \
        --dann --supcon 0 --curve_type ${CURVE_TYPES} --models ${DANN_MODELS} \
        --outlier_filter ${FILTERS} --batch_size ${BATCH_SIZE} ${ALIGN_ARGS}
else
    python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/04_cross_dataset_training.py \
        --exp_folder ${EXP_FOLDER} --task_id ${LOFO_TASK_ID} \
        --coral --supcon 0 --curve_type ${CURVE_TYPES} --models ${CORAL_MODELS} \
        --outlier_filter ${FILTERS} --batch_size ${BATCH_SIZE} ${ALIGN_ARGS}

    python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/04_cross_dataset_training.py \
        --exp_folder ${EXP_FOLDER} --task_id ${LOFO_TASK_ID} \
        --supcon 3 --curve_type ${CURVE_TYPES} --models ${BASE_MODELS} \
        --outlier_filter ${FILTERS} --batch_size ${BATCH_SIZE} ${ALIGN_ARGS}
fi

python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/06b_cross_dataset_prediction_report.py \
    --exp_folder ${EXP_FOLDER} --group ${GROUP_NAME} \
    --mode lofo \
    --curve_type ${CURVE_TYPES} --outlier_filter ${FILTERS} ${ALIGN_ARGS}

deactivate
