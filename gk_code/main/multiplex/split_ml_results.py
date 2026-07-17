"""
Split classification_performances_ml[_10fold].joblib into per-group files.
Run once after moving to per-group result files. Safe to re-run (overwrites).

Usage:
    python split_ml_results.py              # splits the 1-fold file
    python split_ml_results.py --10fold     # splits the 10-fold file
"""
import sys, argparse, joblib
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from multiplex.config_multiplex import (
    RESULT_FILE_BY_FLAG, RESULT_10FOLD_FILE_BY_FLAG, MODEL_FLAG_MAP
)
from multiplex.utils.model_training.model_utils_multilabel import ML_MODEL_KEY_MAP

parser = argparse.ArgumentParser()
parser.add_argument('--10fold',    dest='use_10fold', action='store_true')
parser.add_argument('--exp_path',  default='/vol/bitbucket/gk225/POC_DDM_datasets/LAB_Multiplex/01_ACA_qdPCR')
parser.add_argument('--dry_run',   action='store_true', help='Print routing without writing files')
args = parser.parse_args()

EXP_PATH = Path(args.exp_path)
FILE_MAP  = RESULT_10FOLD_FILE_BY_FLAG if args.use_10fold else RESULT_FILE_BY_FLAG
SRC_NAME  = 'classification_performances_ml_10fold.joblib' if args.use_10fold \
            else 'classification_performances_ml.joblib'
SRC_FILE  = EXP_PATH / SRC_NAME

if not SRC_FILE.exists():
    print(f'Source not found: {SRC_FILE}')
    sys.exit(0)

# Build result-key → flag lookup from ML_MODEL_KEY_MAP
# ML_MODEL_KEY_MAP: model_key → (preds_key, probs_key, classes_key)
RKEY_TO_FLAG = {}
for model_key, (preds_key, probs_key, classes_key) in ML_MODEL_KEY_MAP.items():
    flag = MODEL_FLAG_MAP.get(model_key, 'default')
    for rk in (preds_key, probs_key, classes_key):
        RKEY_TO_FLAG[rk] = flag

print(f'Loading {SRC_FILE} ...')
data = joblib.load(SRC_FILE)

group_data    = {flag: {} for flag in FILE_MAP}
routing_count = Counter()
shared_count  = Counter()

for filter_key, filter_res in data.items():
    if not isinstance(filter_res, dict):
        continue
    for flag in group_data:
        group_data[flag][filter_key] = {}

    for rk, val in filter_res.items():
        flag = RKEY_TO_FLAG.get(rk)
        if flag is not None:
            group_data[flag][filter_key][rk] = val
            routing_count[flag] += 1
        else:
            for g in group_data:
                group_data[g][filter_key][rk] = val
            shared_count[rk] += 1

print(f'\nRouting summary (per filter key):')
for flag, n in sorted(routing_count.items()):
    print(f'  {flag}: {n} result keys')
print(f'  shared: {len(shared_count)} key types → all groups')

if args.dry_run:
    print('\n[dry run] no files written.')
    sys.exit(0)

print()
for flag, fname in FILE_MAP.items():
    dest = EXP_PATH / fname
    has_model_keys = any(
        any(RKEY_TO_FLAG.get(k) == flag for k in rv)
        for rv in group_data[flag].values()
        if isinstance(rv, dict)
    )
    if has_model_keys:
        joblib.dump(group_data[flag], dest, compress=3)
        print(f'  [{flag}] → {dest.name}')
    else:
        print(f'  [{flag}] skipped (no model results in source)')

print('\nDone. Verify each file, then archive or remove the source.')
