#!/bin/bash
#SBATCH --job-name=multi_full_pipeline
#SBATCH --time=48:00:00

# Request resources for a single job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --array=2-6

# Output and Error logs (using SLURM variables to prevent overwriting)
#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

mkdir -p "logs/${SLURM_JOB_NAME}"

cd /vol/bitbucket/gk225/POC_DDM/

# Activate venv
source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate

export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"

cd /vol/bitbucket/gk225/POC_DDM/gk_code/main

REAL_TASK_ID=$SLURM_ARRAY_TASK_ID

EXP_FOLDER=/vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_multi

for nc_subtract in 0 1; do
    if [ "$nc_subtract" -eq 0 ]; then
        python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/01_curve_preprocessing_v6.py --task_id $REAL_TASK_ID --exp_folder "$EXP_FOLDER" --n_wells 10 --n_a_type v06 --force_rerun
        TRAIN_FOLDER="$EXP_FOLDER"
    else
        python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/01_curve_preprocessing_v6.py --task_id $REAL_TASK_ID --exp_folder "$EXP_FOLDER" --n_wells 10 --n_a_type v06 --nc_subtract --force_rerun
        TRAIN_FOLDER="${EXP_FOLDER}_nc_subtract"
    fi

    python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/02_outlier_detection_pipeline.py --task_id $REAL_TASK_ID --exp_folder "$TRAIN_FOLDER"
    python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/03_main_training.py --task_id $REAL_TASK_ID --exp_folder "$TRAIN_FOLDER" --n_splits 1

done

deactivate