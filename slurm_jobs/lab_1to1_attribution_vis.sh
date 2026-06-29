#!/bin/bash
#SBATCH --job-name=lab_1to1_attribution_vis
#SBATCH --time=72:00:00

# Request resources for a single job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30
#SBATCH --array=0-1

# Output and Error logs (using SLURM variables to prevent overwriting)
#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

mkdir -p "logs/${SLURM_JOB_NAME}"

# Activate venv
source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate
export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main

EXP_FOLDER=/vol/bitbucket/gk225/POC_DDM_datasets/LAB_OneToOne

STRATEGIES=(1_area 2_range)
MODELS=(bigru cnn_gru_dual)
STRATEGY=${STRATEGIES[$SLURM_ARRAY_TASK_ID]}
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

TASK_IDS=$(python3 -c "
import os
exp_folder = '$EXP_FOLDER'
names = sorted([n for n in os.listdir(exp_folder)
                if os.path.isdir(os.path.join(exp_folder, n)) and n not in ['.DS_Store', 'model_interpretation']])
ids = [i for i, n in enumerate(names) if n.startswith('${STRATEGY}__')]
print(' '.join(map(str, ids)))
" | tail -n 1)

echo "Strategy: $STRATEGY -- model: $MODEL -- flat task_ids: $TASK_IDS"

for TID in $TASK_IDS; do
    echo "=== [$STRATEGY] Processing combo flat-task_id=$TID ==="
    python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/07_attribution_vis_all.py --task_id $TID --exp_folder "$EXP_FOLDER" --curve_type ori_curve --model_names $MODEL
done

deactivate
