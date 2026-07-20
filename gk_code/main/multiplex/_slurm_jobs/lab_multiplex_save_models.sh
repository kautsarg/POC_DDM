#!/bin/bash
#SBATCH --job-name=lab_multiplex_save_models
#SBATCH --time=16:00:00

# One task per model family — all 8 families run in parallel.
# Task → family mapping:
#   0 = ST (default: cnn/gru/transformer/dual)
#   1 = RCFD (concentration-conditioned dual)
#   2 = Cross-attn (label-query cross-attn)
#   3 = Cross-attn v2 (deepkv/deephead/v2-CGD/CTD)
#   4 = Cross-attn AuxDet
#   5 = Cross-attn QuerCon
#   6 = CRF (MRF + chain + CAttn+CRF flat/factored/bilinear)
#   7 = Source separation (SC0-1 only; requires Phase 1 encoder weights)
#SBATCH --array=0-7

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30

#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

set -e

mkdir -p "logs/${SLURM_JOB_NAME}"

source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate
export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main/multiplex

EXP_FOLDER=/vol/bitbucket/gk225/POC_DDM_datasets/LAB_Multiplex

# task_id=0: only one dataset (01_ACA_qdPCR).
# n_splits=1: single 90/10 StratifiedShuffleSplit — fold_idx=0 triggers model save.
# --save_models: writes {exp_path}/models/{key}_{filter}_{mode}.keras for each model.
run_train() {
    python -u 03_main_training.py \
        --task_id 0 \
        --exp_folder "$EXP_FOLDER" \
        --n_splits 1 \
        "$@"
}

case $SLURM_ARRAY_TASK_ID in
    0)  # ST (default): cnn/gru/transformer/dual SC0-3
        for SC in 0 1 2 3; do
            SARG=(); [ "$SC" -gt 0 ] && SARG=(--supcon "$SC")
            run_train "${SARG[@]}"
        done ;;
    1)  # RCFD (concentration-conditioned): SC0-3
        for SC in 0 1 2 3; do
            SARG=(); [ "$SC" -gt 0 ] && SARG=(--supcon "$SC")
            run_train --condreg "${SARG[@]}"
        done ;;
    2)  # Cross-attn (label-query cross-attn + inter-label SA): SC0-3
        for SC in 0 1 2 3; do
            run_train --cross_attn --supcon "$SC"
        done ;;
    3)  # Cross-attn v2 (deepkv / deephead / v2-CGD / v2-CTD): SC0-3
        for SC in 0 1 2 3; do
            run_train --cross_attn_v2 --supcon "$SC"
        done ;;
    4)  # Cross-attn v2 AuxDet: SC0-3
        for SC in 0 1 2 3; do
            run_train --cross_attn_auxdet --supcon "$SC"
        done ;;
    5)  # Cross-attn v2 QuerCon: SC0-3
        for SC in 0 1 2 3; do
            run_train --cross_attn_quercon --supcon "$SC"
        done ;;
    6)  # CRF (MRF + chain + CAttn+CRF flat/factored/bilinear): SC0-3
        for SC in 0 1 2 3; do
            SARG=(); [ "$SC" -gt 0 ] && SARG=(--supcon "$SC")
            run_train --crf "${SARG[@]}"
        done ;;
    7)  # Source separation (requires Phase 1 encoder weights): SC0-1 only
        for SC in 0 1; do
            SARG=(); [ "$SC" -gt 0 ] && SARG=(--supcon "$SC")
            run_train --source_sep "${SARG[@]}"
        done ;;
    *)
        echo "Unknown SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID (expected 0-7)" >&2
        exit 1 ;;
esac

deactivate
