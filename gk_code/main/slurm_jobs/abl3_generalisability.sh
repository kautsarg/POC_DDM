#!/bin/bash
#SBATCH --job-name=abl3_generalisability
#SBATCH --time=48:00:00

# Request resources for a single array task
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
# task_id 0 = LAB_DDM_paper (folders 01,02,03,09,10, looped internally by the
# script; *_attn_recon* models soft-skipped -- no pixel-grid metadata), task_id 1 =
# every POC_DDM_final subfolder (looped internally, all 8 models train). One task
# per phase now, instead of one array task per subfolder.
#SBATCH --array=0

# Output and Error logs (using SLURM variables to prevent overwriting)
#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

set -e

mkdir -p "logs/${SLURM_JOB_NAME}"

# Activate venv
source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate
export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main

python -u ablations/ablation3_generalisability.py \
    --task_id $SLURM_ARRAY_TASK_ID \
    --curve_type ori_curve \
    --n_splits 5 \
    --batch_size 512

deactivate
