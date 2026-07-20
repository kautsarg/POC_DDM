# Branch-level SupCon v2 and v3 — CNN-GRU + CNN-Trans, ST + MTL

## Context

Current SupCon dual models apply one L_SupCon at the fused 96-dim embedding (implicit "v1").
New variants as separate model keys for comparison:
- **v2** — 2 branch SupCon losses (CNN branch + Seq branch, no fused)
- **v3** — 3 SupCon losses (CNN branch + Seq branch + fused)

Both for CNN-GRU dual and CNN-Trans dual, each with ST and MTL → **8 new model keys total**.
Existing SupCon models are untouched.

---

## λ values

**Keep CE dominant; total SupCon budget matches existing models.**

Existing: ST `λ=0.2`, MTL `λ=0.1` (single head at fused).

### Version 2 — branch only (2 heads, split budget equally)
| Variant | Loss formula |
|---|---|
| ST  | `L = 0.8 × CE + 0.1 × L_SC_cnn + 0.1 × L_SC_seq`  (total SC = 0.2) |
| MTL | `L = UW-SO(CE,MSE) + 0.05 × L_SC_cnn + 0.05 × L_SC_seq`  (total SC = 0.1) |

### Version 3 — branch + fused (3 heads, each gets same λ as existing single-head)
`CE` drops 0.8 → 0.7 to keep ST total = 1.0; each head at 0.1 so no individual head is starved.
| Variant | Loss formula |
|---|---|
| ST  | `L = 0.7 × CE + 0.1 × L_SC_cnn + 0.1 × L_SC_seq + 0.1 × L_SC_fused`  (total SC = 0.3) |
| MTL | `L = UW-SO(CE,MSE) + 0.033 × L_SC_cnn + 0.033 × L_SC_seq + 0.033 × L_SC_fused`  (total SC ≈ 0.1) |

Why 0.1 per head for v3 ST instead of equal-split 0.067: each head is an independent signal so
0.1 per head is not too strong, and 0.7 CE is still well above 0.5. Equal λ per head is also
simpler to tune.

### 1 vs 2 vs 3 SupCon — Pros/Cons

| | 1 (current: fused only) | 2 (branch only) | 3 (branch + fused) |
|---|---|---|---|
| PROs | Already works; most expressive level supervised | Each branch independently discriminative; direct gradient to branch weights | Multi-level supervision; branch specialization AND fused integration both supervised |
| CONs | Branches may be noisy before fusion; SupCon can't fix individual branch confusion | No contrastive signal at fused level; lower-dim projections; loses cross-modality contrastive signal | 3 projection heads (slight overhead); CE drops from 0.8 to 0.7 to fit 3 heads |

---

## New model keys (8 total)

```python
BRANCH_SUPCON2_MODEL_KEYS     = ['cnn_gru_dual_supcon2',     'cnn_trans_dual_supcon2']
BRANCH_SUPCON3_MODEL_KEYS     = ['cnn_gru_dual_supcon3',     'cnn_trans_dual_supcon3']
BRANCH_SUPCON2_MTL_MODEL_KEYS = ['cnn_gru_dual_supcon2_mtl', 'cnn_trans_dual_supcon2_mtl']
BRANCH_SUPCON3_MTL_MODEL_KEYS = ['cnn_gru_dual_supcon3_mtl', 'cnn_trans_dual_supcon3_mtl']
```

---

## Model output shapes

| Model type | Keras outputs | Count |
|---|---|---|
| v2 ST  | `cls_out, cnn_proj, seq_proj` | 3 |
| v2 MTL | `cls_out, reg_out, cnn_proj, seq_proj` | 4 |
| v3 ST  | `cls_out, cnn_proj, seq_proj, fused_proj` | 4 |
| v3 MTL | `cls_out, reg_out, cnn_proj, seq_proj, fused_proj` | 5 |

(Compare: existing SupCon ST = 2 outputs, SupCon MTL = 3 outputs)

---

## Files to change

### 1. `model_utils_mtl.py`
Add `return_branches=False` to both dual-branch builders. Existing callers unchanged.
```python
def _build_cnn_gru_dual_branches_mtl(input_tensor, return_branches=False):
    ...
    if return_branches:
        return cnn_emb, gru_emb, fused   # (32-dim, 64-dim, 96-dim)
    return fused
```
Same for `_build_cnn_trans_dual_branches_mtl`.

### 2. `model_utils_supcon.py`
**4 new Keras model classes:**
```python
class SupConBranch2STModel(keras.Model):   # 0.8*CE + 0.1*(L_SC_cnn + L_SC_seq)
class SupConBranch2MTLModel(keras.Model):  # UW-SO(CE,MSE) + 0.05*(L_SC_cnn + L_SC_seq)
class SupConBranch3STModel(keras.Model):   # 0.7*CE + 0.1*(L_SC_cnn + L_SC_seq + L_SC_fused)
class SupConBranch3MTLModel(keras.Model):  # UW-SO(CE,MSE) + 0.033*(L_SC_cnn + L_SC_seq + L_SC_fused)
```

**4 helper wrap functions** (mirror pattern of existing `_supcon_wrap` / `_supcon_mtl_wrap`):
`_branch2_supcon_wrap`, `_branch2_supcon_mtl_wrap`, `_branch3_supcon_wrap`, `_branch3_supcon_mtl_wrap`

Note: v2 wrap receives `fused` for the cls output but creates no fused projection head.

**8 new factory functions** — `create_cnn_{gru|trans}_dual_supcon{2|3}[_mtl]_model`

Export all 4 new key lists from this file.

### 3. `model_utils.py`
- 4 new dispatch blocks (one per key list)
- Predict unpack: `cls_prob = model.predict(...)[0]` for all v2/v3 (first output = cls_out)
- Regression output: `reg_pred = model.predict(...)[1]` for MTL variants
- Regression key guard extended:
  ```python
  if (_base_m in MTL_MODEL_KEYS
      or _base_m in SUPCON_MTL_MODEL_KEYS
      or _base_m in BRANCH_SUPCON2_MTL_MODEL_KEYS
      or _base_m in BRANCH_SUPCON3_MTL_MODEL_KEYS) and reg_preds_per_fold:
  ```

### 4. `config.py`
- Add 8 entries to `MODEL_KEY_MAP` (y_preds_ / y_probs_ / classes_ keys)
- Add all 8 new keys to `_SUPCON_MODEL_KEYS` (prevents `_inc` variants from being generated)
- Add 8 entries to `MODEL_PRINT_MAP`, e.g.:
  ```python
  'cnn_gru_dual_supcon2':     'CNN-GRU Dual SupCon2',
  'cnn_gru_dual_supcon3':     'CNN-GRU Dual SupCon3',
  'cnn_trans_dual_supcon2':   'CNN-Trans Dual SupCon2',
  'cnn_trans_dual_supcon3':   'CNN-Trans Dual SupCon3',
  # + MTL variants
  ```

### 5. `03_main_training.py`
```python
parser.add_argument('--branch_supcon', type=int, choices=[2, 3], default=None)
```
Selection logic alongside existing `--supcon` block:
```python
elif args.branch_supcon == 2:
    model_keys = BRANCH_SUPCON2_MTL_MODEL_KEYS if args.mtl else BRANCH_SUPCON2_MODEL_KEYS
elif args.branch_supcon == 3:
    model_keys = BRANCH_SUPCON3_MTL_MODEL_KEYS if args.mtl else BRANCH_SUPCON3_MODEL_KEYS
```
Force-rerun key deletion: add suffix patterns `supcon2_mtl_` / `supcon2_` and `supcon3_mtl_` / `supcon3_`.

### 6. `07_attribution_vis_all.py`
Two sections need updating: `extract_xai_artifacts` and `plot_gradcam_per_label`.

**`load_saved_models` (line ~118)** — add 8 new strings to hardcoded model names list.

**Type detection** — extend booleans BEFORE existing `is_supcon` line (both sections):
```python
is_supcon_mtl3 = 'supcon3_mtl' in _base_name   # 5 outputs
is_supcon_mtl2 = 'supcon2_mtl' in _base_name   # 4 outputs
is_supcon3     = 'supcon3' in _base_name and not is_supcon_mtl3   # 4 outputs
is_supcon2     = 'supcon2' in _base_name and not is_supcon_mtl2   # 3 outputs
# update existing to exclude v2/v3:
is_supcon_mtl  = 'supcon_mtl' in _base_name and not is_supcon_mtl3 and not is_supcon_mtl2
is_supcon      = 'supcon' in _base_name and not is_supcon_mtl and not is_supcon3 and not is_supcon2
```

**Output unpack** — add 4 elif branches (before existing `is_supcon_mtl` branch):
```python
elif is_supcon_mtl3: cls_out, reg_out, *_projs = model(x, training=False); is_type = "mtl"
elif is_supcon_mtl2: cls_out, reg_out, *_projs = model(x, training=False); is_type = "mtl"
elif is_supcon3:     cls_out, *_projs = model(x, training=False);           is_type = "st"
elif is_supcon2:     cls_out, *_projs = model(x, training=False);           is_type = "st"
```

`*_projs` discards projection outputs — XAI only needs `cls_out` (and `reg_out` for MTL).
`plot_latent_saliency_heatmap` needs no change: it branches on `is_type`, set correctly above.

### 7. `lab_supcon_training.sh`
4 new blocks after existing SupCon blocks:
```bash
python -u .../03_main_training.py ... --branch_supcon 2
python -u .../03_main_training.py ... --branch_supcon 2 --mtl
python -u .../03_main_training.py ... --branch_supcon 3
python -u .../03_main_training.py ... --branch_supcon 3 --mtl
```

### Files NOT requiring changes
- `06_model_prediction_report.py` — iterates `MODEL_KEY_MAP` dynamically; reg detection uses key presence. No changes needed once `config.py` is updated.
- `08_statistical_comparison.py` — same: uses `MODEL_KEY_MAP` uniformly.
- `05_outlier_visualization_report.py` — no model-key logic at all.

---

## Implementation order
1. `model_utils_mtl.py` — expose branch embeddings
2. `model_utils_supcon.py` — new classes, wraps, factory functions, key lists
3. `model_utils.py` — dispatch, predict unpack, reg guard
4. `config.py` — MODEL_KEY_MAP, _SUPCON_MODEL_KEYS, MODEL_PRINT_MAP
5. `03_main_training.py` — `--branch_supcon` flag
6. `07_attribution_vis_all.py` — model names list, type detection, unpack branches
7. `lab_supcon_training.sh` — 4 new training blocks

---

## Verification
1. `--branch_supcon 2` on one task → 4 new keys with cls metrics in joblib
2. `--branch_supcon 2 --mtl` → RMSE/MAE/R² present for v2 MTL models
3. Repeat for `--branch_supcon 3` and `--branch_supcon 3 --mtl`
4. `06_model_prediction_report.py` → 8 new keys visible in report
5. `07_attribution_vis_all.py` → XAI runs without unpack errors for all 8 models
6. Leaderboard: compare `supcon` (v1) vs `supcon2` vs `supcon3` across tasks
