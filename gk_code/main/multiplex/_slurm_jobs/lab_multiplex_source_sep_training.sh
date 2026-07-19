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

export PYTHONIOENCODING=utf-8
source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate
export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main/multiplex

EXP_FOLDER=/vol/bitbucket/gk225/POC_DDM_datasets/LAB_Multiplex

# Phase 1 — source separation pretraining
# Trains encoder + 3 parametric decoders on all curves.
# Saves: {exp_path}/source_sep_encoder_weights.weights.h5
#        {exp_path}/source_sep_decoder_{0,1,2}_weights.weights.h5
# Skipped automatically if weights already exist.
# Pass --force_rerun to retrain (needed once after architecture changes).
# --lambda_supcon 0.0 = SC0 encoder (no SupCon on z_j); set >0 for SC1-style encoder.
python -u 03b_source_sep_pretraining.py \
    --task_id $SLURM_ARRAY_TASK_ID \
    --exp_folder "$EXP_FOLDER" \
    --d_shared 16 \
    --d_target 10 \
    --epochs_p1 200 \
    --lambda_cons 1.0 \
    --lambda_anch 0.5 \
    --lambda_var 0.1 \
    --lambda_supcon 0.0 \
    --nn_k 5 \
    --force_rerun \
    --validate

run_train() {
    python -u 03_main_training.py \
        --task_id $SLURM_ARRAY_TASK_ID \
        --exp_folder "$EXP_FOLDER" \
        --n_splits 5 \
        "$@"
}

# Phase 2+3 — frozen-encoder head training (Phase 2) then end-to-end fine-tune (Phase 3).
# Loads pretrained encoder weights, freezes encoder for Phase 2, then unfreezes for Phase 3.
# Writes to: classification_performances_ml_source_sep[_10fold].joblib
# --force_rerun clears only source_sep keys in the source_sep result file;
# it does NOT touch other group result files (crf, cattn_v2, etc.).
run_train --source_sep --supcon 0 --force_rerun
run_train --source_sep --supcon 1 --force_rerun

deactivate
