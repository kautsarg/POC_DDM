#!/bin/bash
#SBATCH --job-name=lab_train
#SBATCH --time=24:00:00

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --array=0-3   # one task per folder in LAB_DATASETS_IN_SCOPE

#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

set -e
mkdir -p "logs/${SLURM_JOB_NAME}"

source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate
cd /vol/bitbucket/gk225/POC_DDM/main_code

EXP_FOLDER=/vol/bitbucket/gk225/POC_DDM_datasets/LAB_DDM_paper

python -u main_lab.py train \
    --task_id "$SLURM_ARRAY_TASK_ID" \
    --exp_folder "$EXP_FOLDER" \
    --curve_type ori_curve \
    --n_splits 5
    # --force_rerun
    # --models kNN CNN BiGRU Transformer cnn_gru_dual cnn_gru_dual_attn_recon
    # --batch_size 512

deactivate
