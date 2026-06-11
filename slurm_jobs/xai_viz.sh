#!/bin/bash
#SBATCH --job-name=xai_viz
#SBATCH --time=72:00:00

# Request resources for a single job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30

#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err

mkdir -p "logs/${SLURM_JOB_NAME}"

cd /vol/bitbucket/gk225/POC_DDM/

# Activate venv
source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate

export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"

# Run python script
cd /vol/bitbucket/gk225/POC_DDM/gk_code/model_interpretation

python -u /vol/bitbucket/gk225/POC_DDM/gk_code/model_interpretation/model_for_xai.py --exp_folder "/vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_multi"
python -u /vol/bitbucket/gk225/POC_DDM/gk_code/model_interpretation/attribution_vis_all.py --exp_folder "/vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_multi"

deactivate