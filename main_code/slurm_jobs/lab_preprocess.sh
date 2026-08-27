#!/bin/bash
#SBATCH --job-name=lab_preprocess
#SBATCH --time=12:00:00

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=a30
#SBATCH --array=0-14   # one task per LAB subfolder -- check `ls $EXP_FOLDER | wc -l` and adjust

#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

set -e
mkdir -p "logs/${SLURM_JOB_NAME}"

source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate
cd /vol/bitbucket/gk225/POC_DDM/main_code

EXP_FOLDER=/vol/bitbucket/gk225/POC_DDM_datasets/LAB_DDM_paper

python -u main_lab.py preprocess \
    --task_id "$SLURM_ARRAY_TASK_ID" \
    --exp_folder "$EXP_FOLDER"
    # --force_rerun
    # --one_to_one

deactivate
