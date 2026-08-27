#!/bin/bash
#SBATCH --job-name=chip_cross_dataset
#SBATCH --time=72:00:00

#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=a100

#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

set -e
mkdir -p "logs/${SLURM_JOB_NAME}"

source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate
cd /vol/bitbucket/gk225/POC_DDM/main_code

EXP_FOLDER=/vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_final

TASK_ID=8   # final_6_chip_clean_nn

MODELS="cnn_gru_dual cnn_gru_dual_attn_recon \
        cnn_gru_dual_supcon1 cnn_gru_dual_attn_recon_supcon1 \
        cnn_gru_dual_supcon3 cnn_gru_dual_attn_recon_supcon3 \
        cnn_gru_dual_dann cnn_gru_dual_attn_recon_dann \
        cnn_gru_dual_pc_recentering cnn_gru_dual_attn_recon_pc_recentering"
CURVE_TYPES="ori_curve_sg_p4_norm"

python -u main_chip.py cross-dataset \
    --task_id "$TASK_ID" \
    --exp_folder "$EXP_FOLDER" \
    --mode lofo \
    --curve_type $CURVE_TYPES \
    --models $MODELS \
    --outlier_filter none
    # --force_rerun
    # --batch_size 2048
    # --n_splits 5            # only used by --mode kfold
    # --test_size 0.1         # only used by --mode random_split
    # --held_out_chip <name>  # restrict a lofo run to a single fold

deactivate
