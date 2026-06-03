#!/bin/bash
#SBATCH --job-name=e2e_ddm_xai
#SBATCH --time=72:00:00

# Request resources for a single job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition a40
 
#SBATCH --array=1

# Output and Error logs (using SLURM variables to prevent overwriting)
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

cd /vol/bitbucket/gk225/POC_DDM/

# Activate venv
source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate

export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"

# Run python script
cd /vol/bitbucket/gk225/POC_DDM/gk_code/model_interpretation

# python -u /vol/bitbucket/gk225/POC_DDM/gk_code/outlier_detection/01_par-curve_preprocessing.py --task_id $PY_INDEX
# python -u /vol/bitbucket/gk225/POC_DDM/gk_code/outlier_detection/02_par-outlier_detection_pipeline.py --task_id $PY_INDEX
python -u /vol/bitbucket/gk225/POC_DDM/gk_code/model_interpretation/model_for_xai.py --exp_folder "/vol/bitbucket/gk225/POC_DDM_dataset"

deactivate