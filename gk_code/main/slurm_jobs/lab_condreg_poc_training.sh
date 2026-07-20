#!/bin/bash
#SBATCH --job-name=lab_condreg_poc
#SBATCH --time=24:00:00

# Request resources for a single array task
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a40,a100
#SBATCH --array=0-2,9-10,11-12,13

# Output and Error logs (using SLURM variables to prevent overwriting)
#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

set -e

mkdir -p "logs/${SLURM_JOB_NAME}"

# Activate venv
source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate
export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main

EXP_FOLDER=/vol/bitbucket/gk225/POC_DDM_datasets/LAB_DDM_paper

# GT-FiLM Oracle: 4 models (CGD SC0-3), all run in one invocation
python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/03c_condreg_oracle_poc.py \
    --task_id "$SLURM_ARRAY_TASK_ID" \
    --exp_folder "$EXP_FOLDER" \
    --n_splits 5 \
    --mode native \
    --curve_type ori_curve

deactivate
