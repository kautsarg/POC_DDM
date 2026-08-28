#!/bin/bash
#SBATCH --job-name=abl4_poc_ddm_final_saliency
#SBATCH --time=24:00:00

# Single sequential job -- no array, ablation4_poc_ddm_final_saliency.py already
# loops over all 6 chips internally.
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30

#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err

set -e

mkdir -p "logs/${SLURM_JOB_NAME}"

# Activate venv
source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate

export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main

python -u ablations/ablation4_poc_ddm_final_saliency.py

deactivate
