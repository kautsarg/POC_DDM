#!/bin/bash
#SBATCH --job-name=e2e_ddm_pipeline
#SBATCH --time=48:00:00

# Request resources for a single job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition a30
 
# Launch 10 clones of this job (SLURM array indices 1 through 10)
#SBATCH --array=2-3

# Output and Error logs (using SLURM variables to prevent overwriting)
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

cd /vol/bitbucket/gk225/POC_DDM/

# # Load modules
# module load Python/3.12.3-GCCcore-13.3.0
# module load CUDA/12.6.0
# module load cuDNN/9.10.2.21-CUDA-12.6.0

# Activate venv
source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate

export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"

# Run python script
cd /vol/bitbucket/gk225/POC_DDM/gk_code/outlier_detection

# Replace PBS_ARRAY_INDEX with SLURM_ARRAY_TASK_ID
PY_INDEX=$(($SLURM_ARRAY_TASK_ID - 1))

# python -u /vol/bitbucket/gk225/POC_DDM/gk_code/outlier_detection/01_par-curve_preprocessing.py --task_id $PY_INDEX
# python -u /vol/bitbucket/gk225/POC_DDM/gk_code/outlier_detection/02_par-outlier_detection_pipeline.py --task_id $PY_INDEX
python -u /vol/bitbucket/gk225/POC_DDM/gk_code/outlier_detection/03_par-main_training.py --task_id $PY_INDEX --n_splits 10 --exp_folder "/vol/bitbucket/gk225/lab_dataset"

deactivate