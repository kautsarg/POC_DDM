#!/bin/bash
#SBATCH --job-name=chip_embedding_analysis
#SBATCH --time=12:00:00

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30

#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

set -e
mkdir -p "logs/${SLURM_JOB_NAME}"

source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate
cd /vol/bitbucket/gk225/POC_DDM/main_code

EXP_FOLDER=/vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_final

python -u main_chip.py embedding-analysis --exp_folder "$EXP_FOLDER"
    # --batch_n 800 --seed 42

deactivate
