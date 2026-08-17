#!/bin/bash
#SBATCH --job-name=fnl_lofo_clean_nn_full
#SBATCH --time=72:00:00

# Request resources for a single job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --array=0-1   # 0=supcon 0, 1=supcon 3 -- group is fixed (task_id 3), parallelized by supcon instead

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

# Full baseline for final_4_chip_clean_nn only: 1 LOFO fold, pc_ttp/min alignment,
# noamp_remove filter, both cnn_gru_dual and attn_recon architectures (base/dann/coral),
# no --train_center_frac (full training pool, no sampling) -- see 04_output_map.md for
# what each of these controls in the output layout.
BASE_MODELS="cnn_gru_dual cnn_gru_dual_attn_recon"
CURVE_TYPES="ori_curve_wavelet_bior35_norm ori_curve_norm"
FILTERS="noamp_remove"
ALIGN_ARGS="--curve_alignment pc_ttp --pc_ttp_anchor min"

# Batch size scaled to whichever GPU SLURM actually placed this job on -- A30 has ~24GB
# VRAM vs A40's ~48GB / A100's 40-80GB, so halve the batch size there to avoid OOM.
# A40/A100/unrecognized all keep 04_cross_dataset_training.py's own default (2048).
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
case "$GPU_NAME" in
    *A30*) BATCH_SIZE=1024 ;;
    *)     BATCH_SIZE=2048 ;;
esac
echo "  [*] GPU: ${GPU_NAME}  ->  batch_size=${BATCH_SIZE}"

#### Fixed group -- final_4_chip_clean_nn (CROSS_DATASET_GROUPS index 3)
LOFO_TASK_ID=3
GROUP_NAME=$(python3 -c "import config; print(list(config.CROSS_DATASET_GROUPS.keys())[${LOFO_TASK_ID}])" | tail -n 1)

# 2 parallel array tasks split by supcon (instead of by group, like the baseline_attn_recon
# job does) -- each gets its own GPU allocation and runs concurrently with the other.
SUPCON_VALUES=(0 3)
SUPCON=${SUPCON_VALUES[$SLURM_ARRAY_TASK_ID]}

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
    --outlier_filter ${FILTERS} --lofo_limit 1 --batch_size ${BATCH_SIZE} ${ALIGN_ARGS}

python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/04_cross_dataset_training.py \
    --exp_folder ${EXP_FOLDER} --task_id ${LOFO_TASK_ID} \
    --dann --supcon ${SUPCON} --curve_type ${CURVE_TYPES} --models ${DANN_MODELS} \
    --outlier_filter ${FILTERS} --lofo_limit 1 --batch_size ${BATCH_SIZE} ${ALIGN_ARGS}

python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/04_cross_dataset_training.py \
    --exp_folder ${EXP_FOLDER} --task_id ${LOFO_TASK_ID} \
    --supcon ${SUPCON} --curve_type ${CURVE_TYPES} --models ${BASE_MODELS} \
    --outlier_filter ${FILTERS} --lofo_limit 1 --batch_size ${BATCH_SIZE} ${ALIGN_ARGS}

python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/06b_cross_dataset_prediction_report.py \
    --exp_folder ${EXP_FOLDER} --group ${GROUP_NAME} \
    --mode lofo \
    --curve_type ${CURVE_TYPES} --outlier_filter ${FILTERS} ${ALIGN_ARGS}

deactivate
