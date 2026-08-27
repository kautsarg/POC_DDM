#!/bin/bash
#SBATCH --job-name=chip_train
#SBATCH --time=48:00:00

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --array=0-6   # one task per chip subfolder in EXP_FOLDER

#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

set -e
mkdir -p "logs/${SLURM_JOB_NAME}"

source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate
cd /vol/bitbucket/gk225/POC_DDM/main_code

EXP_FOLDER=/vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_final

python -u main_chip.py train \
    --task_id "$SLURM_ARRAY_TASK_ID" \
    --exp_folder "$EXP_FOLDER" \
    --n_splits 5
    # --force_rerun
    # --models kNN CNN BiGRU Transformer cnn_gru_dual cnn_gru_dual_attn_recon
    # --curve_type ori_curve ori_curve_norm ori_curve_sg_p4 ori_curve_sg_p4_norm
    # --outlier_filter none

deactivate
