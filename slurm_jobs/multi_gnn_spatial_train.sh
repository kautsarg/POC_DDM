#!/bin/bash
#SBATCH --job-name=multi_gnn_spatial_train
#SBATCH --time=48:00:00

# Request resources for a single job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a40

# By default, only index 6 -D20260611_E00_C00_F4500KHz_U_lambda_test_manifold_01
#SBATCH --array=6

#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

mkdir -p "logs/${SLURM_JOB_NAME}"

# Activate venv
source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate
export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main

EXP_FOLDER=/vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_multi_nc_subtract

python -u 03b_gnn_spatial_training.py --task_id $SLURM_ARRAY_TASK_ID --exp_folder "$EXP_FOLDER" --force_rerun
python -u 06_model_prediction_report.py --task_id $SLURM_ARRAY_TASK_ID --exp_folder "$EXP_FOLDER" --force_rerun

deactivate
