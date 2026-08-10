#!/bin/bash
#SBATCH --job-name=abl2_curve_preprocessing
#SBATCH --time=24:00:00

# Request resources for a single array task
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
# Phase-2-only: task_id 0 (LAB_DDM_paper) is a no-op for this ablation (nc_subtract
# needs POC_DDM_final-style chip metadata LAB data doesn't have), so only task_id 1
# is submitted. That one task loops over every POC_DDM_final subfolder internally
# (plain half always runs; nc_subtract half runs wherever a matching sibling exists).
#SBATCH --array=1

# Output and Error logs (using SLURM variables to prevent overwriting)
#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

set -e

mkdir -p "logs/${SLURM_JOB_NAME}"

# Activate venv
source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate
export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main

python -u ablations/ablation2_curve_preprocessing.py \
    --task_id $SLURM_ARRAY_TASK_ID \
    --curve_type ori_curve \
    --n_splits 5 \
    --batch_size 512

deactivate
