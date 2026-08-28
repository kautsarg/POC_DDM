#!/bin/bash
#SBATCH --job-name=fnl_prep
#SBATCH --time=72:00:00

# Request resources for a single job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --array=4,5

# Output and Error logs (using SLURM variables to prevent overwriting)
#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

set -e

mkdir -p "logs/${SLURM_JOB_NAME}"

export PYTHONIOENCODING=utf-8

# Activate venv
source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate
export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main

REAL_TASK_ID=$SLURM_ARRAY_TASK_ID
EXP_FOLDER=/vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_final


# PREPROCESSING AND OUTLIER DETECTION
for nc_subtract in 0 1; do
    if [ "$nc_subtract" -eq 1 ]; then
        python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/01_curve_preprocessing_v6.py --task_id $REAL_TASK_ID --exp_folder "$EXP_FOLDER" --n_wells 10 --n_a_type v06 --nc_subtract --sg_p4 --normalize_curves --drop_pc
        TRAIN_FOLDER="${EXP_FOLDER}_nc_subtract"
    else
        python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/01_curve_preprocessing_v6.py --task_id $REAL_TASK_ID --exp_folder "$EXP_FOLDER" --n_wells 10 --n_a_type v06 --sg_p4 --normalize_curves --drop_pc
        TRAIN_FOLDER="$EXP_FOLDER" 
    fi

    python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/02_outlier_detection_pipeline.py --task_id $REAL_TASK_ID --exp_folder "$TRAIN_FOLDER" --fast_mode --filters
done


# TRAINING AND REPORTS
# for nc_subtract in 0 1; do
#     if [ "$nc_subtract" -eq 1 ]; then
#         TRAIN_FOLDER="${EXP_FOLDER}_nc_subtract"
#     else
#         TRAIN_FOLDER="$EXP_FOLDER"
#     fi

#     python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/03_main_training.py --task_id $REAL_TASK_ID --exp_folder "$TRAIN_FOLDER" --n_splits 1 --fast_mode --supcon 0
#     python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/03_main_training.py --task_id $REAL_TASK_ID --exp_folder "$TRAIN_FOLDER" --n_splits 1 --fast_mode --supcon 1
#     python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/03_main_training.py --task_id $REAL_TASK_ID --exp_folder "$TRAIN_FOLDER" --n_splits 1 --fast_mode --supcon 2
#     python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/03_main_training.py --task_id $REAL_TASK_ID --exp_folder "$TRAIN_FOLDER" --n_splits 1 --fast_mode --supcon 3

#     python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/05_outlier_visualization_report.py --task_id $REAL_TASK_ID --exp_folder "$TRAIN_FOLDER"  --force_rerun
#     python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/06_model_prediction_report.py --task_id $REAL_TASK_ID --exp_folder "$TRAIN_FOLDER" --force_rerun
#     # python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/07_attribution_vis_all.py --task_id $REAL_TASK_ID --exp_folder "$TRAIN_FOLDER" --force_rerun

# done

### NC SUBTRACT ONLY

# TRAIN_FOLDER="${EXP_FOLDER}_nc_subtract"
# CURVE_TYPES="ori_curve ori_curve_norm ori_curves_sg_p4 ori_curves_sg_p4_norm"
# # CURVE_TYPES="ori_curve_norm ori_curve_wavelet_bior35_norm"
# # CURVE_TYPES="ori_curve_wavelet_bior35_norm"

# # python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/03_main_training.py --task_id $REAL_TASK_ID --exp_folder "$TRAIN_FOLDER" --n_splits 1 --fast_mode --supcon 0 --rerun_models cnn_gru_dual_attn_recon cnn_gru_dual --force_rerun
# # python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/03_main_training.py --task_id $REAL_TASK_ID --exp_folder "$TRAIN_FOLDER" --n_splits 1 --fast_mode --supcon 3 --rerun_models cnn_gru_dual_attn_recon_supcon3 cnn_gru_dual_supcon3 --force_rerun

# # python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/03_main_training.py --task_id $REAL_TASK_ID --exp_folder "$TRAIN_FOLDER" --n_splits 1 --supcon 3 --curve_type $CURVE_TYPES
# # python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/03_main_training.py --task_id $REAL_TASK_ID --exp_folder "$TRAIN_FOLDER" --n_splits 1 --fast_mode --supcon 1 --curve_type $CURVE_TYPES
# # python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/03_main_training.py --task_id $REAL_TASK_ID --exp_folder "$TRAIN_FOLDER" --n_splits 1 --fast_mode --supcon 2 --curve_type $CURVE_TYPES
# python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/03_main_training.py --task_id $REAL_TASK_ID --exp_folder "$TRAIN_FOLDER" --n_splits 1 --supcon 0 --curve_type $CURVE_TYPES

# python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/05_outlier_visualization_report.py --task_id $REAL_TASK_ID --exp_folder "$TRAIN_FOLDER"  --force_rerun
# python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/06_model_prediction_report.py --task_id $REAL_TASK_ID --exp_folder "$TRAIN_FOLDER" --curve_type $CURVE_TYPES --outlier_filter None --force_rerun
# python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/07_attribution_vis_all.py --task_id $REAL_TASK_ID --exp_folder "$TRAIN_FOLDER" --force_rerun

deactivate
