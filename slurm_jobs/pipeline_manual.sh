#!/bin/bash
#SBATCH --job-name=pipeline_manual
#SBATCH --time=48:00:00

# Request resources for a single job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition a30
 
#SBATCH --array=5-6,8-9

# Output and Error logs (using SLURM variables to prevent overwriting)
#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

mkdir -p "logs/${SLURM_JOB_NAME}"

cd /vol/bitbucket/gk225/POC_DDM/

# Activate venv
source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate

export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"

# Run python script
cd /vol/bitbucket/gk225/POC_DDM/gk_code/outlier_detection

# python -u /vol/bitbucket/gk225/POC_DDM/gk_code/outlier_detection/01_par-curve_preprocessing_v6.py --task_id $SLURM_ARRAY_TASK_ID --exp_folder /vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_multi --n_wells 10 --n_a_type v06
# python -u /vol/bitbucket/gk225/POC_DDM/gk_code/outlier_detection/02_par-outlier_detection_pipeline.py --task_id $SLURM_ARRAY_TASK_ID --exp_folder /vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_multi 
# python -u /vol/bitbucket/gk225/POC_DDM/gk_code/outlier_detection/03_par-main_training.py --task_id $SLURM_ARRAY_TASK_ID --exp_folder /vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_multi --n_splits 1

python -u /vol/bitbucket/gk225/POC_DDM/gk_code/outlier_detection/01_par-curve_preprocessing_v6.py --task_id $SLURM_ARRAY_TASK_ID --exp_folder /vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_chip_init --n_wells 10 --n_a_type v04
python -u /vol/bitbucket/gk225/POC_DDM/gk_code/outlier_detection/02_par-outlier_detection_pipeline.py --task_id $SLURM_ARRAY_TASK_ID --exp_folder /vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_chip_init 
python -u /vol/bitbucket/gk225/POC_DDM/gk_code/outlier_detection/03_par-main_training.py --task_id $SLURM_ARRAY_TASK_ID --exp_folder /vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_chip_init --n_splits 1


deactivate