#!/bin/bash
#SBATCH --job-name=f_new6_to_4chip_test_only
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --array=0-4
# Output and Error logs (using SLURM variables to prevent overwriting)
#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

# Test-only counterpart to f_new6_to_4chip.sh -- reuses the models that job trains
# (--train_full_only on final_6_new) and just re-runs both test passes (base, then
# +pc_recenter) against the 4 held-out chips. No training, no retraining.
#
# Exists because f_new6_to_4chip.sh's array task 2 (DANN, job 281338) crashed with a
# GPU OOM-style error (tensorflow.python.framework.errors_impl.InternalError: Dst
# tensor is not initialized) inside 04_cross_dataset_training.py's own post-training
# evaluation step -- AFTER the model was already saved to disk, but BEFORE the sbatch
# script's testing loop ever ran (set -e killed the task at the Python traceback). So
# DANN has zero chip-test results, neither base nor pc_recenter. This script re-runs
# both passes standalone for whichever array index needs it, without re-training.
#
# Submit with a dependency on the training job so it doesn't race it, e.g.:
#   sbatch --dependency=afterok:<f_new6_to_4chip_job_id> f_new6_to_4chip_test_only.sh
# Or, for just the crashed DANN slot once its model is confirmed saved:
#   sbatch --array=2 f_new6_to_4chip_test_only.sh

set -e

mkdir -p "logs/${SLURM_JOB_NAME}"

# Activate venv
source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate

export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main

EXP_FOLDER="/vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_final/"

CURVE_TYPES="ori_curve_sg_p4_norm"
FILTERS="noamp_remove"
ALIGN_ARGS="--curve_alignment pc_ttp --pc_ttp_anchor min"

LOFO_TASK_ID=13
GROUP_NAME=$(python3 -c "import config; print(list(config.CROSS_DATASET_GROUPS.keys())[${LOFO_TASK_ID}])" | tail -n 1)

EXCLUSION_GROUP="final_4_chip_clean_nn"
TEST_CHIPS=(
    "D20260806_E00_C00_F4500KHz_U_DDM_01_06"
    "D20260807_E00_C00_F4500KHz_U_DDM_02_07"
    "D20260808_E00_C00_F4500KHz_U_DDM_03_01"
    "D20260810_E00_C00_F4500KHz_U_DDM_04_01"
)

# Must match the MODEL selection in f_new6_to_4chip.sh exactly (same array index ->
# same trained model on disk).
case "$SLURM_ARRAY_TASK_ID" in
    0)
        MODEL="cnn_gru_dual_attn_recon"
        ;;
    1)
        MODEL="cnn_gru_dual_attn_recon_aug"
        ;;
    2)
        MODEL="cnn_gru_dual_attn_recon_dann"
        ;;
    3)
        MODEL="cnn_gru_dual_attn_recon_supcon3"
        ;;
    4)
        MODEL="cnn_gru_dual_attn_recon_mtl"
        ;;
esac

for CHIP in "${TEST_CHIPS[@]}"; do
    echo "  [*] Testing ${MODEL} -> ${CHIP}"
    python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/08_cross_dataset_predict_new_chip.py \
        --exp_folder ${EXP_FOLDER} --group ${GROUP_NAME} --exclusion_group ${EXCLUSION_GROUP} \
        --new_chip_folder ${EXP_FOLDER}/${CHIP} \
        --curve_type ${CURVE_TYPES} --model ${MODEL} --outlier_filter ${FILTERS} \
        ${ALIGN_ARGS}

    echo "  [*] Testing ${MODEL} -> ${CHIP} (+ pc_recenter)"
    python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/08_cross_dataset_predict_new_chip.py \
        --exp_folder ${EXP_FOLDER} --group ${GROUP_NAME} --exclusion_group ${EXCLUSION_GROUP} \
        --new_chip_folder ${EXP_FOLDER}/${CHIP} \
        --curve_type ${CURVE_TYPES} --model ${MODEL} --outlier_filter ${FILTERS} \
        ${ALIGN_ARGS} --pc_recenter
done

deactivate
