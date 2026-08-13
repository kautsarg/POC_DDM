#!/bin/bash
#SBATCH --job-name=final_04_kfold5_crossval
#SBATCH --time=72:00:00

# Request resources for a single job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --array=0-1   # 0=raw curves, 1=norm curves (2 parallel jobs, separate result files)

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
# FILTERS="none lstm_ae_glb_ds1_label_elbow"
FILTERS="none"
TASK_ID=3
GROUP_NAME="final_4_chip_clean"   # must match list(config.CROSS_DATASET_GROUPS.keys())[TASK_ID]

# Task 0: raw curves (ori_curve, ori_curve_wavelet_bior35)
# Task 1: norm curves (ori_curve_norm, ori_curve_wavelet_bior35_norm)
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    CURVE_TYPES="ori_curve ori_curve_wavelet_bior35"
else
    # CURVE_TYPES="ori_curve_norm ori_curve_wavelet_bior35_norm"
    CURVE_TYPES="ori_curve_wavelet_bior35_norm"
fi

# supcon 0 and 3 run sequentially per curve type (same file per curve type, no race)
python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/04_cross_dataset_training.py \
    --exp_folder ${EXP_FOLDER} --task_id ${TASK_ID} \
    --mode kfold --n_splits 5 \
    --supcon 3 --curve_type ${CURVE_TYPES} --models ${MODELS} \
    --outlier_filter ${FILTERS} --train_full

python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/04_cross_dataset_training.py \
    --exp_folder ${EXP_FOLDER} --task_id ${TASK_ID} \
    --mode kfold --n_splits 5 \
    --supcon 0 --curve_type ${CURVE_TYPES} --models ${MODELS} \
    --outlier_filter ${FILTERS} --train_full

python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/06b_cross_dataset_prediction_report.py \
    --exp_folder ${EXP_FOLDER} --group ${GROUP_NAME} \
    --mode kfold --n_splits 5 \
    --curve_type ${CURVE_TYPES} --outlier_filter None

deactivate
