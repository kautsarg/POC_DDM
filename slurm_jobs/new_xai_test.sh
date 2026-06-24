#!/bin/bash
#SBATCH --job-name=new_xai_test
#SBATCH --time=72:00:00
#SBATCH --array=0

# Request resources for a single job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30

#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

mkdir -p "logs/${SLURM_JOB_NAME}"

source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main

TASK_ID=3
REAL_TASK_ID=$((8-TASK_ID))

EXP_FOLDER=/vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_multi
TRAIN_FOLDER="${EXP_FOLDER}_nc_subtract"

python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/07b_cross_dataset_attribution_vis.py --exp_folder "$TRAIN_FOLDER" --force_rerun
python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/07_attribution_vis_all.py --task_id $REAL_TASK_ID --exp_folder "$TRAIN_FOLDER" --force_rerun

deactivate
