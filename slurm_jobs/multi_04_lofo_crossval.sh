#!/bin/bash
#SBATCH --job-name=multi_04_loco_crossval
#SBATCH --time=72:00:00

# Request resources for a single job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --array=0

# Output and Error logs (using SLURM variables to prevent overwriting)
#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

mkdir -p "logs/${SLURM_JOB_NAME}"

# Activate venv
source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate

export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main

python  -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/04_cross_dataset_training.py --exp_folder /vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_multi_nc_subtract/ --task_id 0 --supcon

deactivate