#!/bin/bash
#SBATCH --job-name=lab_full_pipeline
#SBATCH --time=72:00:00

# Request resources for a single job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --array=0-2

# Output and Error logs (using SLURM variables to prevent overwriting)
#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

mkdir -p "logs/${SLURM_JOB_NAME}"

# Activate venv
source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate
export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main

REAL_TASK_ID=$SLURM_ARRAY_TASK_ID
EXP_FOLDER=/vol/bitbucket/gk225/POC_DDM_datasets/LAB_DDM_paper
TRAIN_FOLDER="$EXP_FOLDER"

python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/01b_lab_curve_preprocessing.py --task_id $REAL_TASK_ID --exp_folder "$EXP_FOLDER" --normalize_curves
python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/02_outlier_detection_pipeline.py --task_id $REAL_TASK_ID --exp_folder "$TRAIN_FOLDER"

python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/03_main_training.py --task_id $REAL_TASK_ID --exp_folder "$TRAIN_FOLDER" --n_splits 1 --curve_type ori_curve  --n_splits 5
python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/06_model_prediction_report.py --task_id $REAL_TASK_ID --exp_folder "$TRAIN_FOLDER" --curve_type ori_curve  --n_splits 5

python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/03_main_training.py --task_id $REAL_TASK_ID --exp_folder "$TRAIN_FOLDER" --n_splits 5 --curve_type ori_curve_norm
python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/06_model_prediction_report.py --task_id $REAL_TASK_ID --exp_folder "$TRAIN_FOLDER" --curve_type ori_curve_norm --n_splits 5

python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/07_attribution_vis_all.py --task_id $REAL_TASK_ID --exp_folder "$TRAIN_FOLDER" 

deactivate
