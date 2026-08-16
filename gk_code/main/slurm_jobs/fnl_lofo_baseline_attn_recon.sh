#!/bin/bash
#SBATCH --job-name=fnl_lofo_baseline_attn_recon
#SBATCH --time=72:00:00

# Request resources for a single job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30,a40,a100
#SBATCH --array=0-2   # 0=group idx 3, 1=group idx 4, 2=group idx 5

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

# Baselining config: 1 LOFO fold, pc_ttp/min alignment only, noamp_remove filter,
# attn_recon architecture only (base/dann/coral), SC0 and SC3 -- see 04_output_map.md
# for what each of these controls in the output layout.
BASE_MODELS="cnn_gru_dual_attn_recon"
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

#### CHECK THE GROUPS!
TASK_IDS=(3 4 5)
LOFO_TASK_ID=${TASK_IDS[$SLURM_ARRAY_TASK_ID]}
GROUP_NAME=$(python3 -c "import config; print(list(config.CROSS_DATASET_GROUPS.keys())[${LOFO_TASK_ID}])" | tail -n 1)

for SUPCON in 0 3; do
    if [ "$SUPCON" -eq 3 ]; then
        DANN_MODELS="cnn_gru_dual_attn_recon_supcon3_dann"
        CORAL_MODELS="cnn_gru_dual_attn_recon_supcon3_coral"
    else
        DANN_MODELS="cnn_gru_dual_attn_recon_dann"
        CORAL_MODELS="cnn_gru_dual_attn_recon_coral"
    fi

    python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/04_cross_dataset_training.py \
        --exp_folder ${EXP_FOLDER} --task_id ${LOFO_TASK_ID} \
        --coral --supcon ${SUPCON} --curve_type ${CURVE_TYPES} --models ${CORAL_MODELS} \
        --outlier_filter ${FILTERS} --lofo_limit 1 --batch_size ${BATCH_SIZE} --train_center_frac 0.5 ${ALIGN_ARGS}

    python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/04_cross_dataset_training.py \
        --exp_folder ${EXP_FOLDER} --task_id ${LOFO_TASK_ID} \
        --dann --supcon ${SUPCON} --curve_type ${CURVE_TYPES} --models ${DANN_MODELS} \
        --outlier_filter ${FILTERS} --lofo_limit 1 --batch_size ${BATCH_SIZE} --train_center_frac 0.5 ${ALIGN_ARGS}

    python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/04_cross_dataset_training.py \
        --exp_folder ${EXP_FOLDER} --task_id ${LOFO_TASK_ID} \
        --supcon ${SUPCON} --curve_type ${CURVE_TYPES} --models ${BASE_MODELS} \
        --outlier_filter ${FILTERS} --lofo_limit 1 --batch_size ${BATCH_SIZE} --train_center_frac 0.5 ${ALIGN_ARGS}
done

python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/06b_cross_dataset_prediction_report.py \
    --exp_folder ${EXP_FOLDER} --group ${GROUP_NAME} \
    --mode lofo \
    --curve_type ${CURVE_TYPES} --outlier_filter ${FILTERS} --train_center_frac 0.5 ${ALIGN_ARGS}

deactivate
