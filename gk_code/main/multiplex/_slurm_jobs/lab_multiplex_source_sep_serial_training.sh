#!/bin/bash
#SBATCH --job-name=lab_multiplex_source_sep_serial
#SBATCH --time=12:00:00

# Request resources for a single array task
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30,a40,a100
#SBATCH --array=1

# Output and Error logs
#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

set -e

mkdir -p "logs/${SLURM_JOB_NAME}"

export PYTHONIOENCODING=utf-8
source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate
export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main/multiplex

EXP_FOLDER=/vol/bitbucket/gk225/POC_DDM_datasets/LAB_Multiplex

P1_ARGS=(
    --task_id "$SLURM_ARRAY_TASK_ID"
    --exp_folder "$EXP_FOLDER"
    --force_rerun
    --validate
    --train_phase2
    --n_splits 5
    --force_rerun_phase2
)

# Variant A: L_active, sum L_consist
python -u 03c_serial_source_sep_pretraining.py "${P1_ARGS[@]}" --variant active

# Variant B: L_balance, sum L_consist
python -u 03c_serial_source_sep_pretraining.py "${P1_ARGS[@]}" --variant balance

# Variant C: L_active, avg L_consist (multi-target signal = mean of single-target signals)
python -u 03c_serial_source_sep_pretraining.py "${P1_ARGS[@]}" --variant active  --avg_consist

# Variant D: L_balance, avg L_consist
python -u 03c_serial_source_sep_pretraining.py "${P1_ARGS[@]}" --variant balance --avg_consist

deactivate
