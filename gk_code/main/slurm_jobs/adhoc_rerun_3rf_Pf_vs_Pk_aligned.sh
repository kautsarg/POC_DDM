#!/bin/bash
#SBATCH --job-name=adhoc_3rf_Pf_vs_Pk_aligned
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err

mkdir -p "logs/${SLURM_JOB_NAME}"

source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate
export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main

EXP_FOLDER=/vol/bitbucket/gk225/POC_DDM_datasets/LAB_OneToOne
TRAIN_FOLDER="$EXP_FOLDER"
ALIGNED_TID=40   # 3_range_filtered__Pf_vs_Pk_aligned

python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/03_main_training.py --task_id $ALIGNED_TID --exp_folder "$TRAIN_FOLDER" --n_splits 5 --curve_type ori_curve
python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/03_main_training.py --task_id $ALIGNED_TID --exp_folder "$TRAIN_FOLDER" --n_splits 5 --curve_type ori_curve_norm
python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/06_model_prediction_report.py --task_id $ALIGNED_TID --exp_folder "$TRAIN_FOLDER" --n_splits 5 --curve_type ori_curve
python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/06_model_prediction_report.py --task_id $ALIGNED_TID --exp_folder "$TRAIN_FOLDER" --n_splits 5 --curve_type ori_curve_norm

deactivate
