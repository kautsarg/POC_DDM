#!/bin/bash
#SBATCH --job-name=lab_1to1_full_pipeline
#SBATCH --time=72:00:00

# Request resources for a single job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --array=2

# Output and Error logs (using SLURM variables to prevent overwriting)
#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

mkdir -p "logs/${SLURM_JOB_NAME}"

# Activate venv
source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate
export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main

EXP_FOLDER=/vol/bitbucket/gk225/POC_DDM_datasets/LAB_OneToOne
TRAIN_FOLDER="$EXP_FOLDER"

STRATEGIES=(1_area 2_range 3_range_filtered)
STRATEGY=${STRATEGIES[$SLURM_ARRAY_TASK_ID]}

# TASK_IDS=$(python3 -c "
# import importlib
# mod = importlib.import_module('01b_lab_curve_preprocessing')
# combos = mod.discover_one_to_one_combos('$EXP_FOLDER')
# ids = [i for i, c in enumerate(combos) if c[0] == '$STRATEGY']
# print(' '.join(map(str, ids[::-1])))
# " | tail -n 1)

TASK_IDS=$(python3 -c "
import importlib
mod = importlib.import_module('01b_lab_curve_preprocessing')
combos = mod.discover_one_to_one_combos('$EXP_FOLDER')
ids = [i for i, c in enumerate(combos) if c[0] == '$STRATEGY']
print(' '.join(map(str, ids)))
" | tail -n 1) 

echo "Strategy: $STRATEGY -- flat task_ids: $TASK_IDS"

for TID in $TASK_IDS; do
    echo "=== [$STRATEGY] Processing combo flat-task_id=$TID ==="
    python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/01b_lab_curve_preprocessing.py --task_id $TID --exp_folder "$EXP_FOLDER" --one_to_one
    # python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/01b_lab_curve_preprocessing.py --task_id $TID --exp_folder "$EXP_FOLDER" --one_to_one --normalize_curves
    python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/02_outlier_detection_pipeline.py --task_id $TID --exp_folder "$TRAIN_FOLDER"

    python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/03_main_training.py --task_id $TID --exp_folder "$TRAIN_FOLDER" --n_splits 5 --curve_type ori_curve
    python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/06_model_prediction_report.py --task_id $TID --exp_folder "$TRAIN_FOLDER" --n_splits 5 --curve_type ori_curve
    
    # python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/03_main_training.py --task_id $TID --exp_folder "$TRAIN_FOLDER" --n_splits 5 --curve_type ori_curve_norm
    # python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/06_model_prediction_report.py --task_id $TID --exp_folder "$TRAIN_FOLDER" --n_splits 5 --curve_type ori_curve_norm
done

deactivate
