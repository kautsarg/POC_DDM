#!/bin/bash
#SBATCH --job-name=fnl_3to5_sc3_aug_parallel
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

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
case "$GPU_NAME" in
    *A30*) BATCH_SIZE=1024 ;;
    *)     BATCH_SIZE=2048 ;;
esac
echo "  [*] GPU: ${GPU_NAME}  ->  batch_size=${BATCH_SIZE}"

# Fixed to the same group/config as 279825_13 (fnl_3to5_lofo_full.sh, --array=13 ->
# final_6_new) so results land in the exact same output directory/files that job will
# eventually write to itself. Safe to run concurrently: save_partitioned's per-(filter,
# model) file split + filelock-protected read-merge-write (utils/cross_dataset_result_io.py)
# means these two model keys never touch the file 279825_13's own current DANN run (or
# eventual SC3 run) is writing to at the same moment -- and if 279825_13 later reaches its
# own SC3 call after this job has already saved SC3 results, evaluate_outlier_filters'
# _split_signature check will just cache-hit and skip retraining rather than clobber
# anything.
LOFO_TASK_ID=13
GROUP_NAME=$(python3 -c "import config; print(list(config.CROSS_DATASET_GROUPS.keys())[${LOFO_TASK_ID}])" | tail -n 1)

# 2 parallel array tasks: task 0 = SupCon3 (the same model 279825_13 will eventually
# reach on its own, just done earlier here), task 1 = the new temporal-augmentation
# variant (cnn_gru_dual_attn_recon_aug didn't exist yet when 279825_13 was submitted).
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/04_cross_dataset_training.py \
        --exp_folder ${EXP_FOLDER} --task_id ${LOFO_TASK_ID} \
        --supcon 3 --curve_type ${CURVE_TYPES} --models cnn_gru_dual_attn_recon_supcon3 \
        --outlier_filter ${FILTERS} --batch_size ${BATCH_SIZE} ${ALIGN_ARGS} --train_center_frac 0.5
else
    python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/04_cross_dataset_training.py \
        --exp_folder ${EXP_FOLDER} --task_id ${LOFO_TASK_ID} \
        --supcon 0 --curve_type ${CURVE_TYPES} --models cnn_gru_dual_attn_recon_aug \
        --outlier_filter ${FILTERS} --batch_size ${BATCH_SIZE} ${ALIGN_ARGS} --train_center_frac 0.5
fi

deactivate
