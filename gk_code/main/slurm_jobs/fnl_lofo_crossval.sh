#!/bin/bash
#SBATCH --job-name=fnl_loco_crossval
#SBATCH --time=72:00:00

# Request resources for a single job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --array=0-1   # 0=acquisition_start, 1=pc_ttp min, 2=pc_ttp percentile; %3 caps concurrency

# Output and Error logs (using SLURM variables to prevent overwriting)
#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

mkdir -p "logs/${SLURM_JOB_NAME}"

# Activate venv
source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate

export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main

EXP_FOLDER="/vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_final_nc_subtract/"

MODELS="knn cnn_gru_dual cnn_gru_dual_attn_recon"
FILTERS="none"
CURVE_TYPES="ori_curve_wavelet_bior35_norm"
SUPCON=3

LOFO_TASK_ID=3
GROUP_NAME=$(python3 -c "import config; print(list(config.CROSS_DATASET_GROUPS.keys())[${LOFO_TASK_ID}])" | tail -n 1)

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    ALIGN_ARGS=""
elif [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    ALIGN_ARGS="--curve_alignment pc_ttp --pc_ttp_anchor min"
else
    ALIGN_ARGS="--curve_alignment pc_ttp --pc_ttp_anchor percentile"
fi

if [ "$SUPCON" -eq 3 ]; then
    DANN_MODELS="cnn_gru_dual_supcon3_dann cnn_gru_dual_attn_recon_supcon3_dann"
else
    DANN_MODELS="cnn_gru_dual_dann cnn_gru_dual_attn_recon_dann"
fi

python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/04_cross_dataset_training.py \
    --exp_folder ${EXP_FOLDER} --task_id ${LOFO_TASK_ID} \
    --dann --supcon ${SUPCON} --curve_type ${CURVE_TYPES} --models ${DANN_MODELS} \
    --outlier_filter ${FILTERS} --train_full ${ALIGN_ARGS} \
    --lofo_limit 1

python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/04_cross_dataset_training.py \
    --exp_folder ${EXP_FOLDER} --task_id ${LOFO_TASK_ID} \
    --supcon ${SUPCON} --curve_type ${CURVE_TYPES} --models ${MODELS} \
    --outlier_filter ${FILTERS} --train_full ${ALIGN_ARGS} \
    --lofo_limit 1

python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/06b_cross_dataset_prediction_report.py \
    --exp_folder ${EXP_FOLDER} --group ${GROUP_NAME} \
    --mode lofo \
    --curve_type ${CURVE_TYPES} --outlier_filter None ${ALIGN_ARGS}

deactivate
