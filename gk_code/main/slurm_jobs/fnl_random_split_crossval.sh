#!/bin/bash
#SBATCH --job-name=fnl_random_split
#SBATCH --time=48:00:00

# Request resources for a single job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --array=1   # 0=raw curves, 1=norm curves (2 parallel jobs, separate result files)

# Output and Error logs (using SLURM variables to prevent overwriting)
#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

mkdir -p "logs/${SLURM_JOB_NAME}"

# Activate venv
source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate

export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main

EXP_FOLDER="/vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_final_nc_subtract/"
MODELS="cnn_gru_dual cnn_gru_dual_attn_recon"
FILTERS="none"
TASK_ID=3

# Task 0: raw curves (ori_curve, ori_curve_wavelet_bior35)
# Task 1: norm curves (ori_curve_norm, ori_curve_wavelet_bior35_norm)
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    CURVE_TYPES="ori_curve ori_curve_wavelet_bior35"
else
    CURVE_TYPES="ori_curve_norm ori_curve_wavelet_bior35_norm"
fi

python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/04_cross_dataset_training.py \
    --exp_folder ${EXP_FOLDER} --task_id ${TASK_ID} \
    --mode random_split --test_size 0.2 \
    --supcon 0 --curve_type ${CURVE_TYPES} --models ${MODELS} \
    --outlier_filter ${FILTERS} --train_full --fast_mode \
    --rerun_models cnn_gru_dual_attn_recon cnn_gru_dual --force_rerun

python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/04_cross_dataset_training.py \
    --exp_folder ${EXP_FOLDER} --task_id ${TASK_ID} \
    --mode random_split --test_size 0.2 \
    --supcon 3 --curve_type ${CURVE_TYPES} --models ${MODELS} \
    --outlier_filter ${FILTERS} --train_full --fast_mode \
    --rerun_models cnn_gru_dual_attn_recon cnn_gru_dual --force_rerun

deactivate
