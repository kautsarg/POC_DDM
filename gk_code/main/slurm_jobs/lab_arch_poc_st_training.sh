#!/bin/bash
#SBATCH --job-name=lab_arch_poc_st_training
#SBATCH --time=24:00:00

# Request resources for a single array task
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30,a40,a100
#SBATCH --array=0-2,9-12

# Output and Error logs
#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

set -e

mkdir -p "logs/${SLURM_JOB_NAME}"

source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate
export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main

EXP_FOLDER=/vol/bitbucket/gk225/POC_DDM_datasets/LAB_DDM_paper

python -u 03g_arch_poc_st_training.py \
    --task_id "$SLURM_ARRAY_TASK_ID" \
    --exp_folder "$EXP_FOLDER" \
    --curve_type ori_curve \
    --n_splits 1

deactivate
