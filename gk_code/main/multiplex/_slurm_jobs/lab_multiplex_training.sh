#!/bin/bash
#SBATCH --job-name=lab_multiplex_training
#SBATCH --time=72:00:00

# Request resources for a single array task
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a40,a100
#SBATCH --array=0

# Output and Error logs (using SLURM variables to prevent overwriting)
#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

set -e

mkdir -p "logs/${SLURM_JOB_NAME}"

# Activate venv
source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate
export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main/multiplex

EXP_FOLDER=/vol/bitbucket/gk225/POC_DDM_datasets/LAB_Multiplex

# # 01b: Preprocessing (run once; idempotent — skipped automatically if cached)
# python -u 01b_lab_curve_preprocessing.py --task_id $SLURM_ARRAY_TASK_ID --exp_folder "$EXP_FOLDER"

# # 02: Outlier detection (LSTM-AE only; spatial gracefully skipped on flat-CSV lab data)
# python -u 02_outlier_detection_pipeline.py --task_id $SLURM_ARRAY_TASK_ID --exp_folder "$EXP_FOLDER"

run_train() {
    python -u 03_main_training.py \
        --task_id $SLURM_ARRAY_TASK_ID \
        --exp_folder "$EXP_FOLDER" \
        --n_splits 5 \
        "$@"
}

# ST (standard multi-label): SC 0-3
for SC in 0 1 2 3; do
    SARG=(); [ "$SC" -gt 0 ] && SARG=(--supcon "$SC")
    run_train "${SARG[@]}"
done

# RCFD (concentration-conditioned): SC 0-3
for SC in 0 1 2 3; do
    SARG=(); [ "$SC" -gt 0 ] && SARG=(--supcon "$SC")
    run_train --condreg "${SARG[@]}"
done

# Cross-attention head (label query cross-attn + inter-label self-attn): SC 0-3
for SC in 0 1 2 3; do
    run_train --cross_attn --supcon "$SC"
done

# Cross-attn v2 ablation (deepkv / deephead / v2-CGD / v2-CTD): SC 0-3
for SC in 0 1 2 3; do
    run_train --cross_attn_v2 --supcon "$SC"
done

# Cross-attn v2 AuxDet (per-block aux BCE): SC 0-3
for SC in 0 1 2 3; do
    run_train --cross_attn_auxdet --supcon "$SC"
done

# Cross-attn v2 QuerCon (query contrastive + backbone SC): SC 0-3
for SC in 0 1 2 3; do
    run_train --cross_attn_quercon --supcon "$SC"
done

# CRF structured-output variants (MRF + chain): SC 0-3
for SC in 0 1 2 3; do
    SARG=(); [ "$SC" -gt 0 ] && SARG=(--supcon "$SC")
    run_train --crf "${SARG[@]}"
done

# python -u 06_model_prediction_report.py \
#     --task_id $SLURM_ARRAY_TASK_ID \
#     --exp_folder "$EXP_FOLDER" \
#     --n_splits 5 \
#     --force_rerun

deactivate
