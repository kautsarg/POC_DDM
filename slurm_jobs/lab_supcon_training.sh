#!/bin/bash
#SBATCH --job-name=lab_supcon_training
#SBATCH --time=72:00:00

# Request resources for a single array task
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --array=11-12

# Output and Error logs (using SLURM variables to prevent overwriting)
#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

set -e

mkdir -p "logs/${SLURM_JOB_NAME}"

# Activate venv
source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate
export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main

PREP_TASK_ID=$((SLURM_ARRAY_TASK_ID-3))
EXP_FOLDER=/vol/bitbucket/gk225/POC_DDM_datasets/LAB_DDM_paper
TRAIN_FOLDER="$EXP_FOLDER"

python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/01b_lab_curve_preprocessing.py --task_id $PREP_TASK_ID --exp_folder "$EXP_FOLDER"
python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/02_outlier_detection_pipeline.py --task_id $SLURM_ARRAY_TASK_ID --exp_folder "$TRAIN_FOLDER" --filters

# ST Normal
python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/03_main_training.py \
    --task_id $SLURM_ARRAY_TASK_ID \
    --exp_folder "$TRAIN_FOLDER" \
    --curve_type ori_curve \
    --n_splits 5

# MTL Normal
python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/03_main_training.py \
    --task_id $SLURM_ARRAY_TASK_ID \
    --exp_folder "$TRAIN_FOLDER" \
    --curve_type ori_curve \
    --n_splits 5 \
    --mtl

# ST SupCon
python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/03_main_training.py \
    --task_id $SLURM_ARRAY_TASK_ID \
    --exp_folder "$TRAIN_FOLDER" \
    --curve_type ori_curve \
    --n_splits 5 \
    --supcon

# MTL SupCon
python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/03_main_training.py \
    --task_id $SLURM_ARRAY_TASK_ID \
    --exp_folder "$TRAIN_FOLDER" \
    --curve_type ori_curve \
    --n_splits 5 \
    --supcon --mtl

python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/06_model_prediction_report.py \
    --task_id $SLURM_ARRAY_TASK_ID \
    --exp_folder "$TRAIN_FOLDER" \
    --curve_type ori_curve \
    --n_splits 5 \
    --force_rerun

python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/07_attribution_vis_all.py \
    --task_id $SLURM_ARRAY_TASK_ID \
    --exp_folder "$TRAIN_FOLDER" \
    --force_rerun

deactivate
