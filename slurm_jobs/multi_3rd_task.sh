#!/bin/bash
#SBATCH --job-name=multi_3rd_task
#SBATCH --time=48:00:00

# Request resources for a single job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --array=5,7,9,11

# Output and Error logs (using SLURM variables to prevent overwriting)
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# CRITICAL: Slurm will fail if the logs directory doesn't exist yet
mkdir -p /vol/bitbucket/gk225/POC_DDM/logs

# Activate venv
source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate
export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/outlier_detection

python -u /vol/bitbucket/gk225/POC_DDM/gk_code/outlier_detection/03_par-main_training.py --task_id $SLURM_ARRAY_TASK_ID --exp_folder /vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_multi --n_splits 1

deactivate