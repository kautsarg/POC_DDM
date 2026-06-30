#!/bin/bash
#SBATCH --job-name=lab_aligned_full_pipeline
#SBATCH --time=72:00:00

# Request resources for a single job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --array=3-5

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

# Step 1: build/refresh the '<name>_aligned' sibling folder for this task's source experiment
# (--ori_curve_aligned: TTP-align + baseline-correct, dropping curves past the knee-detected
# max_ttp -- a separate self-contained dataset, not a variant of the regular one, since TTP
# filtering changes the row count).
python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/01b_lab_curve_preprocessing.py --task_id $REAL_TASK_ID --exp_folder "$EXP_FOLDER" --ori_curve_aligned

# Step 2: find that '<name>_aligned' folder's own task_id in 02/03/06/07's generic folder
# listing (config.EXCLUDED_FOLDERS-filtered) -- different from $REAL_TASK_ID, which only
# indexes into 01b's FILE_MAPPING-filtered source-folder list.
ALIGNED_TID=$(python3 -c "
import os
import config
import importlib
mod = importlib.import_module('01b_lab_curve_preprocessing')
src_names = sorted([n for n in os.listdir('$EXP_FOLDER')
                     if os.path.isdir(os.path.join('$EXP_FOLDER', n)) and n in mod.FILE_MAPPING])
source_name = src_names[$REAL_TASK_ID]
aligned_name = source_name + '_aligned'
all_names = sorted([n for n in os.listdir('$EXP_FOLDER')
                     if os.path.isdir(os.path.join('$EXP_FOLDER', n)) and n not in config.EXCLUDED_FOLDERS])
print(all_names.index(aligned_name))
" | tail -n 1)

echo "Source task_id=$REAL_TASK_ID -> aligned folder task_id=$ALIGNED_TID"

python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/02_outlier_detection_pipeline.py --task_id $ALIGNED_TID --exp_folder "$TRAIN_FOLDER"
python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/03_main_training.py --task_id $ALIGNED_TID --exp_folder "$TRAIN_FOLDER" --n_splits 5 --curve_type ori_curve
python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/06_model_prediction_report.py --task_id $ALIGNED_TID --exp_folder "$TRAIN_FOLDER" --n_splits 5 --curve_type ori_curve
python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/07_attribution_vis_all.py --task_id $ALIGNED_TID --exp_folder "$TRAIN_FOLDER"

deactivate
