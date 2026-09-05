#!/bin/bash
#SBATCH --job-name=f_tinv_g13
#SBATCH --time=24:00:00

# Temperature-head LOFO on final_6_new. One array task per model, with the 6
# LOFO folds looped inside the task -- same shape as f_lf_g13.sh.
#
#   task 0 -> adversarial : GRL  -> encoder pushed to make temp UNPREDICTABLE
#   task 1 -> mtl         : none -> encoder pushed to make temp PREDICTABLE
#   task 2 -> off         : zero-weighted head -> baseline control (should
#                           reproduce the plain attn_recon LOFO macro, 45.99%)
#
# All three share one architecture and differ ONLY in the gradient direction
# into the encoder, so any accuracy gap is attributable to that alone.
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --array=0-1
#SBATCH --output=logs/%x/%A_%a.out
#SBATCH --error=logs/%x/%A_%a.err

set -e

mkdir -p "logs/${SLURM_JOB_NAME}"

# Activate venv
source /vol/bitbucket/gk225/venv_poc_ddm/bin/activate

export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
case "$GPU_NAME" in
    *A30*) BATCH_SIZE=1024 ;;
    *)     BATCH_SIZE=2048 ;;
esac

MODES=(adversarial mtl off)
MODE=${MODES[${SLURM_ARRAY_TASK_ID}]}

# switch to wellcentered for the ablation that strips the class-correlated
# between-well component by construction
VARIANT=raw

echo "  [*] GPU: ${GPU_NAME} -> batch_size=${BATCH_SIZE}"
echo "  [*] task ${SLURM_ARRAY_TASK_ID} -> mode=${MODE} variant=${VARIANT}, 6 LOFO folds"

# One bad fold shouldn't discard the other five, so don't let errexit abort the
# loop -- record the failure and surface it in the exit code instead.
FAILED=0
for CHIP in 0 1 2 3 4 5; do
    echo ""
    echo "===== ${MODE} | held-out chip index ${CHIP} ====="
    if ! python -u /vol/bitbucket/gk225/POC_DDM/gk_code/main/tempinv_lofo_training.py \
            --batch_size ${BATCH_SIZE} \
            --held_out ${CHIP} \
            --mode ${MODE} \
            --variant ${VARIANT} \
            --temp_weight 1.0; then
        echo "  [!] mode=${MODE} chip=${CHIP} FAILED"
        FAILED=1
    fi
done

echo ""
echo "[ALL FOLDS DONE] mode=${MODE} failed=${FAILED}"
deactivate
exit ${FAILED}
