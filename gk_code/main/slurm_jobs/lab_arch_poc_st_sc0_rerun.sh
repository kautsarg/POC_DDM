#!/bin/bash
#SBATCH --job-name=lab_arch_poc_st_sc0_rerun
#SBATCH --time=04:00:00

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a30,a40,a100
#SBATCH --array=0-2,9-12,14

#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

set -e

mkdir -p "logs/${SLURM_JOB_NAME}"

source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate
export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main

EXP_FOLDER=/vol/bitbucket/gk225/POC_DDM_datasets/LAB_DDM_paper

# Clear SC0 result keys so they retrain; SC1-SC3 stay cached.
python3 - "$SLURM_ARRAY_TASK_ID" "$EXP_FOLDER" <<'PYEOF'
import sys, joblib
from pathlib import Path
sys.path.insert(0, '.')
sys.path.insert(0, 'utils')
from pipeline_utils import get_exp_paths

SC0_KEYS = {
    'y_preds_AC_cgd_arch_poc_st_',  'y_probs_AC_cgd_arch_poc_st_',  'classes_AC_cgd_arch_poc_st_',
    'y_preds_AC_cgs_arch_poc_st_',  'y_probs_AC_cgs_arch_poc_st_',  'classes_AC_cgs_arch_poc_st_',
    'y_preds_AC_ccgd_arch_poc_st_', 'y_probs_AC_ccgd_arch_poc_st_', 'classes_AC_ccgd_arch_poc_st_',
}
task_id, exp_folder = int(sys.argv[1]), sys.argv[2]
exp_paths = get_exp_paths(exp_folder)
result_file = Path(exp_paths[task_id]) / 'arch_poc' / 'arch_poc_st_results.joblib'
if not result_file.exists():
    print(f'  -> [SKIP] No result file at {result_file}')
    sys.exit(0)
results = joblib.load(result_file)
n = 0
for ds in results.values():
    for mode in ds.values():
        for entry in mode.values():
            if isinstance(entry, dict):
                for k in list(entry):
                    if k in SC0_KEYS:
                        del entry[k]; n += 1
print(f'  -> Cleared {n} SC0 key(s) from {result_file}')
joblib.dump(results, result_file, compress=3)
PYEOF

python -u 03g_arch_poc_st_training.py \
    --task_id "$SLURM_ARRAY_TASK_ID" \
    --exp_folder "$EXP_FOLDER" \
    --curve_type ori_curve \
    --n_splits 1

deactivate
