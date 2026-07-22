#!/bin/bash
#SBATCH --job-name=lab_multiplex_source_sep_precon
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

P1_ARGS_BASE=(
    --task_id "$SLURM_ARRAY_TASK_ID"
    --exp_folder "$EXP_FOLDER"
    --d_shared 16
    --d_target 10
    --epochs_p1 400
    --lambda_cons 2.0
    --lambda_anch 0.3
    --lambda_var 0.05
    --nn_k 5
    --precon
    --force_rerun
    --validate
)

P23_ARGS=(
    --task_id "$SLURM_ARRAY_TASK_ID"
    --exp_folder "$EXP_FOLDER"
    --n_splits 5
    --precon
    --train_phase23
    --force_rerun_phase23
)

# Phase 1 — Family B SC1 precon encoder
# Saves: source_sep_precon_sc1_encoder_weights.weights.h5
python -u 03b_source_sep_pretraining.py "${P1_ARGS_BASE[@]}" --supcon 1

# Phase 1 — Family B SC2 precon encoder
# Saves: source_sep_precon_sc2_encoder_weights.weights.h5
python -u 03b_source_sep_pretraining.py "${P1_ARGS_BASE[@]}" --supcon 2

# Phase 1 — Family B SC3 precon encoder
# Saves: source_sep_precon_sc3_encoder_weights.weights.h5
python -u 03b_source_sep_pretraining.py "${P1_ARGS_BASE[@]}" --supcon 3

# Phase 2+3 — Family B, SC1: 4 variants (_precon_supcon_p2/p3, _precon_crf_supcon_p2/p3)
python -u 03b_source_sep_pretraining.py "${P23_ARGS[@]}" --supcon 1

# Phase 2+3 — Family B, SC2: 4 variants
python -u 03b_source_sep_pretraining.py "${P23_ARGS[@]}" --supcon 2

# Phase 2+3 — Family B, SC3: 4 variants
python -u 03b_source_sep_pretraining.py "${P23_ARGS[@]}" --supcon 3

deactivate
