#!/bin/bash
#SBATCH --job-name=lab_multiplex_source_sep
#SBATCH --time=4:00:00

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

source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate
export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main/multiplex

EXP_FOLDER=/vol/bitbucket/gk225/POC_DDM_datasets/LAB_Multiplex

# Phase 1 — source separation pretraining
# Trains encoder + 3 parametric decoders on all curves.
# Saves: {exp_path}/source_sep_encoder_weights.weights.h5
# Skipped automatically if weights already exist (use --force_rerun to retrain).
python -u 03b_source_sep_pretraining.py \
    --task_id $SLURM_ARRAY_TASK_ID \
    --exp_folder "$EXP_FOLDER" \
    --d_shared 16 \
    --d_target 10 \
    --epochs_p1 200 \
    --lambda_cons 1.0 \
    --lambda_anch 0.5 \
    --nn_k 5 \
    --validate

run_train() {
    python -u 03_main_training.py \
        --task_id $SLURM_ARRAY_TASK_ID \
        --exp_folder "$EXP_FOLDER" \
        --n_splits 5 \
        "$@"
}

# Phase 2 — frozen-encoder classification (SC0: base, SC1: SupCon backbone)
# Loads pretrained encoder weights and freezes the encoder during training.
# Writes to: classification_performances_ml_source_sep[_10fold].joblib
run_train --source_sep --supcon 0
run_train --source_sep --supcon 1

deactivate
