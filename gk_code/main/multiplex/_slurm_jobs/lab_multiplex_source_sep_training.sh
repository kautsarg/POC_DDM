#!/bin/bash
#SBATCH --job-name=lab_multiplex_source_sep
#SBATCH --time=12:00:00

# Request resources for a single array task
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30,a40,a100
#SBATCH --array=1

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

P1_ARGS=(
    --task_id "$SLURM_ARRAY_TASK_ID"
    --exp_folder "$EXP_FOLDER"
)

P23_ARGS=(
    --task_id "$SLURM_ARRAY_TASK_ID"
    --exp_folder "$EXP_FOLDER"
    --n_splits 5
    --train_phase23
    --force_rerun_phase23
)

# Phase 1 — Family A standard encoder (lambda_supcon=0.0, SC0)
# Saves: source_sep_encoder_weights.weights.h5
python -u 03b_source_sep_pretraining.py "${P1_ARGS[@]}" --force_rerun --validate

# Phase 2+3 — Family A, SC0: 4 variants (_p2, _crf_p2, _p3, _crf_p3)
python -u 03b_source_sep_pretraining.py "${P23_ARGS[@]}" --supcon 0

# Phase 2+3 — Family A, SC1: 2 variants (_supcon_p3, _crf_supcon_p3)
python -u 03b_source_sep_pretraining.py "${P23_ARGS[@]}" --supcon 1

# Phase 2+3 — Family A, SC2: 2 variants (_supcon2_p3, _crf_supcon2_p3)
python -u 03b_source_sep_pretraining.py "${P23_ARGS[@]}" --supcon 2

# Phase 2+3 — Family A, SC3: 2 variants (_supcon3_p3, _crf_supcon3_p3)
python -u 03b_source_sep_pretraining.py "${P23_ARGS[@]}" --supcon 3

deactivate
