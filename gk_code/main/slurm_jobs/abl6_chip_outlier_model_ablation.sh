#!/bin/bash
#SBATCH --job-name=abl6_chip_outlier_model_ablation
# gnn_gat is now a Keras model (model_utils_gnn_recon.py) trained inside the same
# evaluate_outlier_filters loop as cnn_gru_dual/cnn_gru_dual_attn_recon -- not the old
# PyTorch 03b_gnn_spatial_training.py GNN, which no longer runs here at all. 3 Keras
# models x n_splits=5 x 3 filters = 45 lightweight, mini-batched fit cycles (vs. the
# original 2-model/30-cycle budget this job used before gnn_gat existed) -- 1.5x the
# original 24h.
#SBATCH --time=36:00:00

# Request resources for a single array task
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --array=0-3

# Output and Error logs (using SLURM variables to prevent overwriting)
#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

set -e

mkdir -p "logs/${SLURM_JOB_NAME}"

# Activate venv
source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate
export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main

RAW_EXP_FOLDER="/vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_final"
EXP_FOLDER="${RAW_EXP_FOLDER}_nc_subtract"


# AE_BATCH_SIZE stays much smaller than BATCH_SIZE -- the LSTM autoencoder
# reconstructs the whole curve per sample (TimeDistributed), so its memory
# scales with batch_size x curve_length x features, unlike the classifier's
# fixed-size output. 1024 OOM'd on A30 (see logs/.../276398_0.err) even
# though the classifier's 1024 is fine; 128/256 matches the flat 128 default
# this step always used before it was made adaptive.
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
case "$GPU_NAME" in
    *A30*) BATCH_SIZE=1024; AE_BATCH_SIZE=128 ;;
    *)     BATCH_SIZE=2048; AE_BATCH_SIZE=256 ;;
esac
echo "  [*] GPU: ${GPU_NAME}  ->  batch_size=${BATCH_SIZE}  lstm_ae_batch_size=${AE_BATCH_SIZE}"


python -u 01_curve_preprocessing_v6.py \
    --task_id $SLURM_ARRAY_TASK_ID --exp_folder ${RAW_EXP_FOLDER} \
    --n_wells 10 --n_a_type v06 --nc_subtract --sg_p4 --normalize_curves --drop_pc

python -u 02_outlier_detection_pipeline.py \
    --task_id $SLURM_ARRAY_TASK_ID --exp_folder ${EXP_FOLDER} \
    --filters amf lstm_ae --batch_size ${AE_BATCH_SIZE}

python -u ablations/ablation6_chip_outlier_model_ablation.py \
    --task_id $SLURM_ARRAY_TASK_ID \
    --curve_type ori_curve_sg_p4_norm \
    --n_splits 5 \
    --batch_size ${BATCH_SIZE}

deactivate
