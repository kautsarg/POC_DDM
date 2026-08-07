#!/bin/bash
#SBATCH --job-name=multi_04_loco_crossval
#SBATCH --time=72:00:00

# Request resources for a single job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --array=2

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
CURVE_TYPES="ori_curve ori_curve_wavelet_bior35"

python  -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/04_cross_dataset_training.py --exp_folder ${EXP_FOLDER} --task_id $SLURM_ARRAY_TASK_ID --curve_type ${CURVE_TYPES} --models ${MODELS} --train_full
# python  -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/04_cross_dataset_training.py --exp_folder ${EXP_FOLDER} --task_id $SLURM_ARRAY_TASK_ID --supcon 1 --curve_type ${CURVE_TYPES} --models ${MODELS} --train_full
# python  -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/04_cross_dataset_training.py --exp_folder ${EXP_FOLDER} --task_id $SLURM_ARRAY_TASK_ID --supcon 2 --curve_type ${CURVE_TYPES} --models ${MODELS} --train_full
python  -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/04_cross_dataset_training.py --exp_folder ${EXP_FOLDER} --task_id $SLURM_ARRAY_TASK_ID --supcon 3 --curve_type ${CURVE_TYPES} --models ${MODELS} --train_full
# python  -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/04_cross_dataset_training.py --exp_folder ${EXP_FOLDER} --task_id $SLURM_ARRAY_TASK_ID --supcon_staged --supcon 1 --curve_type ${CURVE_TYPES} --models ${MODELS} --train_full
# python  -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/04_cross_dataset_training.py --exp_folder ${EXP_FOLDER} --task_id $SLURM_ARRAY_TASK_ID --supcon_staged --supcon 2 --curve_type ${CURVE_TYPES} --models ${MODELS} --train_full
# python  -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/04_cross_dataset_training.py --exp_folder ${EXP_FOLDER} --task_id $SLURM_ARRAY_TASK_ID --supcon_staged --supcon 3 --curve_type ${CURVE_TYPES} --models ${MODELS} --train_full

deactivate