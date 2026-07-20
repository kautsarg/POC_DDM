#!/bin/bash
#SBATCH --job-name=lab_rc_poc_training
#SBATCH --time=72:00:00

# Request resources for a single array task
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --array=0-1

# Output and Error logs
#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

set -e

mkdir -p "logs/${SLURM_JOB_NAME}"

# Activate venv
source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate
export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main

EXP_FOLDER=/vol/bitbucket/gk225/POC_DDM_datasets/LAB_DDM_paper
TRAIN_FOLDER="$EXP_FOLDER"

# 1 task ID = 1 exp_folder; runs all 16 RC-POC models (no-fold, n_splits=1 default)
# Pass 1 (CLS/MTL/RCFD) and Pass 2 (CCFD-POC) run sequentially within the same invocation.
python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/03e_rc_poc_training.py \
    --task_id $SLURM_ARRAY_TASK_ID \
    --exp_folder "$TRAIN_FOLDER" \
    --curve_type ori_curve

deactivate
