#!/bin/bash
#SBATCH --job-name=lab_multiplex_cattn_crf
#SBATCH --time=24:00:00

# Trains all 24 CAttn+CRF models (flat/factored/bilinear × CGD+CTD × SC0-3) sequentially
# in a single job so they all write to classification_performances_ml_crf.joblib safely.

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a40,a100

#SBATCH --output=logs/%x/%A.out
#SBATCH --error=logs/%x/%A.err

set -e

mkdir -p "logs/${SLURM_JOB_NAME}"

source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate
export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main/multiplex

EXP_FOLDER=/vol/bitbucket/gk225/POC_DDM_datasets/LAB_Multiplex

for SC in 0 1 2 3; do
    python -u 03_main_training.py \
        --task_id 0 \
        --exp_folder "$EXP_FOLDER" \
        --n_splits 5 \
        --crf \
        --supcon "$SC"
done

deactivate
