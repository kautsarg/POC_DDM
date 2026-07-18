#!/bin/bash
#SBATCH --job-name=lab_multiplex_crf_training
#SBATCH --time=24:00:00

# Request resources for a single array task
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a40,a100
#SBATCH --array=0

# Output and Error logs
#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

set -e

mkdir -p "logs/${SLURM_JOB_NAME}"

# Activate venv
source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate
export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main/multiplex

EXP_FOLDER=/vol/bitbucket/gk225/POC_DDM_datasets/LAB_Multiplex

run_train() {
    python -u 03_main_training.py \
        --task_id $SLURM_ARRAY_TASK_ID \
        --exp_folder "$EXP_FOLDER" \
        --n_splits 5 \
        "$@"
}

# CRF structured-output variants (MRF + chain): SC 0-3
# Writes to: classification_performances_ml_crf[_10fold].joblib
for SC in 0 1 2 3; do
    SARG=(); [ "$SC" -gt 0 ] && SARG=(--supcon "$SC")
    run_train --crf "${SARG[@]}"
done

deactivate
