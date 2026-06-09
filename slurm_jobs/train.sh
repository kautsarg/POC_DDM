#!/bin/bash
#SBATCH --job-name=v06_poc_multi
#SBATCH --time=48:00:00

# Request resources for a single job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --array=0-1

# Output and Error logs (using SLURM variables to prevent overwriting)
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# CRITICAL: Slurm will fail if the logs directory doesn't exist yet
mkdir -p /vol/bitbucket/gk225/POC_DDM/logs

cd /vol/bitbucket/gk225/POC_DDM/

# Activate venv
source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate

export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"

cd /vol/bitbucket/gk225/POC_DDM/gk_code/outlier_detection

REAL_TASK_ID=8
OUTLIER_TRAINING_ID=$((REAL_TASK_ID + SLURM_ARRAY_TASK_ID))

# FIX: Correct Bash if/else syntax
if [ "$REAL_TASK_ID" -eq "$OUTLIER_TRAINING_ID" ]; then
    python -u /vol/bitbucket/gk225/POC_DDM/gk_code/outlier_detection/01_par-curve_preprocessing_v6.py --task_id $REAL_TASK_ID --exp_folder /vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_multi --n_wells 10 --n_a_type v06
else
    python -u /vol/bitbucket/gk225/POC_DDM/gk_code/outlier_detection/01_par-curve_preprocessing_v6.py --task_id $REAL_TASK_ID --exp_folder /vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_multi --n_wells 10 --n_a_type v06 --nc_subtract
fi

python -u /vol/bitbucket/gk225/POC_DDM/gk_code/outlier_detection/02_par-outlier_detection_pipeline.py --task_id $OUTLIER_TRAINING_ID --exp_folder /vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_multi
python -u /vol/bitbucket/gk225/POC_DDM/gk_code/outlier_detection/03_par-main_training.py --task_id $OUTLIER_TRAINING_ID --exp_folder /vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_multi --n_splits 1

deactivate