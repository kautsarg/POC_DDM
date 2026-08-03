#!/bin/bash
#SBATCH --job-name=multi_arch_poc_lofo_training
#SBATCH --time=72:00:00

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30,a40,a100

# Array 0-2 = curve_type_id:  0=ori_curve  1=ori_curve_avg  2=ori_curve_wavelet_sym8
# TASK_ID = index into config.CROSS_DATASET_GROUPS (which group to run LOFO on)
# Set via env when submitting:  sbatch --export=TASK_ID=0 multi_arch_poc_lofo_training.sh
#SBATCH --array=0-2

#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

set -e

mkdir -p "logs/${SLURM_JOB_NAME}"

# TASK_ID defaults to 0 if not exported by the caller
TASK_ID=${TASK_ID:-0}

source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate
export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main

EXP_FOLDER=/vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_multi_nc_subtract

python -u 04b_arch_poc_curve_training.py \
    --task_id       "$TASK_ID" \
    --curve_type_id "$SLURM_ARRAY_TASK_ID" \
    --exp_folder    "$EXP_FOLDER"

deactivate
