#!/bin/bash
#SBATCH --job-name=xai_viz
#SBATCH --time=72:00:00
#SBATCH --array=1-3

# Request resources for a single job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30

#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

mkdir -p "logs/${SLURM_JOB_NAME}"

cd /vol/bitbucket/gk225/POC_DDM/

# Activate venv
source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate

export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"

# Run python script
cd /vol/bitbucket/gk225/POC_DDM/gk_code/model_interpretation

# Map SLURM_ARRAY_TASK_ID -> dataset subfolder under POC_DDM_datasets (sorted, discovered at runtime)
DATASETS_ROOT="/vol/bitbucket/gk225/POC_DDM_datasets"
mapfile -t EXP_FOLDERS < <(find "$DATASETS_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)

EXP_FOLDER="${DATASETS_ROOT}/${EXP_FOLDERS[$((SLURM_ARRAY_TASK_ID-1))]}"
# python -u /vol/bitbucket/gk225/POC_DDM/gk_code/model_interpretation/model_for_xai.py --exp_folder "$EXP_FOLDER"
python -u /vol/bitbucket/gk225/POC_DDM/gk_code/model_interpretation/attribution_vis_all.py --exp_folder "$EXP_FOLDER" --force_rerun

deactivate