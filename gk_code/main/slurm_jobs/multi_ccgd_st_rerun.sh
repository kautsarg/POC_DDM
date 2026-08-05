#!/bin/bash
#SBATCH --job-name=multi_ccgd_st_rerun
#SBATCH --time=04:00:00

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30,a40,a100
#SBATCH --array=0-1

#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

set -e

mkdir -p "logs/${SLURM_JOB_NAME}"

export PYTHONIOENCODING=utf-8

source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate
export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main

EXP_FOLDER=/vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_final

# Retrain only ccgd_arch_poc_st* so model_utils.py saves the .keras files.
# --rerun_models filters the models list; --force_rerun bypasses cached results.
# --curve_type ori_curve: only the curve type needed by the XAI notebook.

python -u 03_main_training.py \
    --task_id "$SLURM_ARRAY_TASK_ID" \
    --exp_folder "$EXP_FOLDER" \
    --supcon 0 \
    --rerun_models ccgd_arch_poc_st \
    --force_rerun \
    --curve_type ori_curve \
    --n_splits 1 \
    --fast_mode

python -u 03_main_training.py \
    --task_id "$SLURM_ARRAY_TASK_ID" \
    --exp_folder "$EXP_FOLDER" \
    --supcon 1 \
    --rerun_models ccgd_arch_poc_st_sc1 \
    --force_rerun \
    --curve_type ori_curve \
    --n_splits 1 \
    --fast_mode

python -u 03_main_training.py \
    --task_id "$SLURM_ARRAY_TASK_ID" \
    --exp_folder "$EXP_FOLDER" \
    --supcon 2 \
    --rerun_models ccgd_arch_poc_st_sc2 \
    --force_rerun \
    --curve_type ori_curve \
    --n_splits 1 \
    --fast_mode

python -u 03_main_training.py \
    --task_id "$SLURM_ARRAY_TASK_ID" \
    --exp_folder "$EXP_FOLDER" \
    --supcon 3 \
    --rerun_models ccgd_arch_poc_st_sc3 \
    --force_rerun \
    --curve_type ori_curve \
    --n_splits 1 \
    --fast_mode

deactivate
