# Per-Group Result Files — Parallel SLURM Safety

## Problem

The training script follows a **load → train → save** pattern on a single shared joblib:

```
cached_results = joblib.load(results_path)   # T₀ snapshot
...train models...
safe_joblib_dump(cached_results, results_path)  # write back T₀ + own results
```

`safe_joblib_dump` is atomic against corruption (`.tmp` + `os.replace()`) but has no
inter-process mutual exclusion. When 3 jobs run in parallel:

```
T₀  Job1, Job2, Job3 all load the same file  →  each holds the same dict in memory
T₁  Job1 finishes  →  file = { cross_attn_v2: ... }
T₂  Job2 finishes  →  file = { auxdet: ... }          # LOST Job1 results
T₃  Job3 finishes  →  file = { quercon: ... }          # LOST Job1+Job2 results
```

Last writer wins. Within a single job the for-loop over SC values is sequential, so
no intra-job race — only the inter-job race matters.

---

## Solution: One Joblib Per Training Flag

Each flag (`--cross_attn`, `--cross_attn_v2`, `--auxdet`, `--quercon`, `--condreg`,
or none of the above = `default`) writes to its own file. Jobs never touch each other's
files, so no locking or sequencing is needed.

The mapping lives in `config_multiplex.py` so both the training script and the
analysis notebook share a single source of truth.

---

## Result Dict Structure (existing)

```python
# classification_performances_ml.joblib
{
    None: {                              # filter_key (None = no outlier filter)
        'y_trues_':          [...],      # shared: list of fold arrays (N_fold, n_targets)
        'mask_count':        int,        # shared
        'y_true_count':      int,        # shared
        'y_preds_{model}_':  [...],      # per-model predictions
        'y_probs_{model}_':  [...],      # per-model probabilities
        'classes_{model}_':  [...],      # per-model class labels
    },
    'lstm_ae_glb_ds4_label_elbow': { ... },   # other filter_keys
}
```

Shared keys (`y_trues_`, `mask_count`, `y_true_count`) must appear in every group
file so each file is self-contained for the notebook.

---

## Changes

### 1. `config_multiplex.py`

Add after `TRAINING_10FOLD_RESULT_PATH`:

```python
# Per-group result files — one per parallel-job flag (keeps SLURM jobs race-free)
RESULT_FILE_BY_FLAG = {
    'default':    'classification_performances_ml.joblib',
    'cross_attn': 'classification_performances_ml_cross_attn.joblib',
    'cattn_v2':   'classification_performances_ml_cattn_v2.joblib',
    'auxdet':     'classification_performances_ml_auxdet.joblib',
    'quercon':    'classification_performances_ml_quercon.joblib',
    'condreg':    'classification_performances_ml_condreg.joblib',
}
RESULT_10FOLD_FILE_BY_FLAG = {
    k: v.replace('.joblib', '_10fold.joblib')
    for k, v in RESULT_FILE_BY_FLAG.items()
}


def _flag_for_model(key):
    if 'quercon'    in key: return 'quercon'
    if 'auxdet'     in key: return 'auxdet'
    if any(x in key for x in ('deepkv', 'deephead', '_v2')): return 'cattn_v2'
    if 'cross_attn' in key: return 'cross_attn'
    if 'rcfd'       in key: return 'condreg'
    return 'default'


# model_key → flag; used by split_ml_results.py and for reference
MODEL_FLAG_MAP = {k: _flag_for_model(k) for k in MULTIPLEX_MODELS}
```

`MODEL_FLAG_MAP` is derived (not hand-written) so it stays in sync automatically when
`MULTIPLEX_MODELS` grows.

---

### 2. `03_main_training.py`

Replace lines 149–152 (the `if args.n_splits > 1` branch):

```python
_result_flag = (
    'cross_attn' if args.cross_attn        else
    'cattn_v2'   if args.cross_attn_v2     else
    'auxdet'     if args.cross_attn_auxdet  else
    'quercon'    if args.cross_attn_quercon else
    'condreg'    if args.condreg            else
    'default'
)
_fmap        = config.RESULT_10FOLD_FILE_BY_FLAG if args.n_splits > 1 else config.RESULT_FILE_BY_FLAG
results_path = os.path.join(exp_path, _fmap[_result_flag])
```

No other changes — `load_or_init_results`, `checkpoint_fn`, and `safe_joblib_dump`
all operate on `results_path` which now points to the group-specific file.

---

### 3. `multiplex/split_ml_results.py` (new ad-hoc script)

Run once to split the existing monolithic joblib into per-group files.
After the split, delete (or archive) the monolithic file to avoid confusion.

```python
"""
Split classification_performances_ml[_10fold].joblib into per-group files.
Run once; safe to re-run (overwrites destination files).
"""
import sys, os, joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from multiplex.config_multiplex import (
    RESULT_FILE_BY_FLAG, RESULT_10FOLD_FILE_BY_FLAG, MODEL_FLAG_MAP
)
from multiplex.utils.model_training.model_utils_multilabel import ML_MODEL_KEY_MAP

# ── Config ────────────────────────────────────────────────────────────────────
EXP_PATH = Path('/vol/bitbucket/gk225/POC_DDM_datasets/LAB_Multiplex/01_ACA_qdPCR')
USE_10FOLD = False   # set True to split the _10fold variant

SRC_FILE   = EXP_PATH / ('classification_performances_ml_10fold.joblib' if USE_10FOLD
                          else 'classification_performances_ml.joblib')
FILE_MAP   = RESULT_10FOLD_FILE_BY_FLAG if USE_10FOLD else RESULT_FILE_BY_FLAG

# ── Build result-key → flag lookup ────────────────────────────────────────────
# ML_MODEL_KEY_MAP: model_key → (preds_key, probs_key, classes_key)
RKEY_TO_FLAG = {}
for model_key, (preds_key, probs_key, classes_key) in ML_MODEL_KEY_MAP.items():
    flag = MODEL_FLAG_MAP.get(model_key, 'default')
    for rk in (preds_key, probs_key, classes_key):
        RKEY_TO_FLAG[rk] = flag

# Keys not in RKEY_TO_FLAG are shared (y_trues_, mask_count, y_true_count, etc.)

# ── Split ─────────────────────────────────────────────────────────────────────
if not SRC_FILE.exists():
    print(f'Source not found: {SRC_FILE}')
    sys.exit(0)

print(f'Loading {SRC_FILE} ...')
data = joblib.load(SRC_FILE)

group_data = {flag: {} for flag in FILE_MAP}

for filter_key, filter_res in data.items():
    if not isinstance(filter_res, dict):
        continue
    for flag in group_data:
        group_data[flag][filter_key] = {}

    for rk, val in filter_res.items():
        flag = RKEY_TO_FLAG.get(rk)
        if flag is not None:
            group_data[flag][filter_key][rk] = val
        else:
            # shared key — copy to every group so each file is self-contained
            for g in group_data:
                group_data[g][filter_key][rk] = val

# ── Save ──────────────────────────────────────────────────────────────────────
for flag, fname in FILE_MAP.items():
    dest = EXP_PATH / fname
    # only write if the group has at least one model result key
    has_model_keys = any(
        any(k in rv for k in RKEY_TO_FLAG if RKEY_TO_FLAG[k] == flag)
        for rv in group_data[flag].values()
        if isinstance(rv, dict)
    )
    if has_model_keys:
        joblib.dump(group_data[flag], dest, compress=3)
        print(f'  [{flag}] → {dest.name}')
    else:
        print(f'  [{flag}] skipped (no model results)')

print('Done.')
```

---

### 4. `multiplex/notebooks/lab_threshold_search.ipynb`

Replace the single `joblib.load(RESULTS_PATH)` in Cell 2 with a merge loop:

```python
from multiplex.config_multiplex import RESULT_FILE_BY_FLAG, RESULT_10FOLD_FILE_BY_FLAG

file_map = RESULT_10FOLD_FILE_BY_FLAG if USE_10FOLD else RESULT_FILE_BY_FLAG
all_results = {}
for flag, fname in file_map.items():
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        continue
    d = joblib.load(path)
    for fk, fv in d.items():
        all_results.setdefault(fk, {}).update(fv)

res_entry = all_results[FILTER_KEY]
print(f'Merged {len(file_map)} files  |  result keys: {len(res_entry)}')
```

`dict.update()` is safe here because no key appears in two group files.

---

## SLURM Script

No change needed — the 3 parallel loops already use different flags, which now
automatically route to different files.

---

## Verification

After implementing, run a dry-run forward pass:
```bash
cd /vol/bitbucket/gk225/POC_DDM/gk_code
python3 -c "
import sys; sys.path.insert(0,'main/multiplex'); sys.path.insert(0,'main')
from multiplex.config_multiplex import RESULT_FILE_BY_FLAG, MODEL_FLAG_MAP, MULTIPLEX_MODELS
assert len(set(MODEL_FLAG_MAP.values())) == len(RESULT_FILE_BY_FLAG), 'flag mismatch'
from collections import Counter
c = Counter(MODEL_FLAG_MAP.values())
for flag, n in sorted(c.items()):
    print(f'  {flag}: {n} models → {RESULT_FILE_BY_FLAG[flag]}')
print('OK')
" 2>/dev/null
```

Expected output:
```
  cattn_v2:   16 models → classification_performances_ml_cattn_v2.joblib
  auxdet:      4 models → classification_performances_ml_auxdet.joblib
  condreg:    16 models → classification_performances_ml_condreg.joblib
  cross_attn:  8 models → classification_performances_ml_cross_attn.joblib
  default:    14 models → classification_performances_ml.joblib
  quercon:     4 models → classification_performances_ml_quercon.joblib
```
