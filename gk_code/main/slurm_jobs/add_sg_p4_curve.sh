#!/bin/bash
#SBATCH --job-name=add_sg_p4_curve
#SBATCH --time=72:00:00

# Request resources for a single job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --array=0-3%1

# Output and Error logs (using SLURM variables to prevent overwriting)
#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

set -e

mkdir -p "logs/${SLURM_JOB_NAME}"

export PYTHONIOENCODING=utf-8

# Activate venv
source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate
export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main

REAL_TASK_ID=$SLURM_ARRAY_TASK_ID
EXP_FOLDER=/vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_final

for nc_subtract in 0 1; do
    if [ "$nc_subtract" -eq 1 ]; then
        ARGS="--task_id $REAL_TASK_ID --exp_folder $EXP_FOLDER --n_wells 10 --n_a_type v06 --nc_subtract --sg_p4 --normalize_curves --drop_pc"
        TRAIN_FOLDER="${EXP_FOLDER}_nc_subtract"
    else
        ARGS="--task_id $REAL_TASK_ID --exp_folder $EXP_FOLDER --n_wells 10 --n_a_type v06 --sg_p4 --normalize_curves --drop_pc"
        TRAIN_FOLDER="$EXP_FOLDER"
    fi
    python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/01_curve_preprocessing_v6.py $ARGS
    python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/01_curve_preprocessing_v6.py $ARGS

    python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/02_outlier_detection_pipeline.py --task_id $REAL_TASK_ID --exp_folder "$TRAIN_FOLDER" --fast_mode --filters
done

deactivate
