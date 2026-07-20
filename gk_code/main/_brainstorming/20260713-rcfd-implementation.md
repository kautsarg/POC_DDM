# RCFD Implementation Plan

**Regression-Conditioned Feature Dual (RCFD)** — Approach A from `20260713-sequential_multi_task_learning.md`.
Renamed from "Regression-Primed Dual (RPD)" to avoid confusion with primal-dual in Double Machine Learning (DML).
Implementation file: `model_utils_rcfd.py` (not cascade — matches the model name).

---

## Architecture Summary

A lightweight early regression encoder (CNN/GRU/Transformer) predicts concentration `c_pred` as a scalar. That scalar conditions the entire 96-dim fused dual backbone embedding via **scalar-to-feature FiLM** (`γ·z + β`) before the classification head.

Unlike `cnn_gru_film_mtl` (branch-to-branch FiLM), RCFD uses scalar-to-feature conditioning — one concentration estimate reshapes all 96 dimensions. Both heads train end-to-end jointly with `tf.stop_gradient` on `c_pred` before FiLM.

Options 7–9 from the brainstorm (shared encoder, confidence-gated FiLM, multi-level FiLM) are deferred.

---

## Model Keys — 24 total (3 encoders × 2 dual backbones × 4 supcon variants)

| Variant | Key pattern | Outputs | Loss |
|---------|------------|---------|------|
| Base | `{enc}_rcfd_{dual}` | `[cls_out, reg_out]` | UW-SO (CE + MSE) |
| SupCon v1 | `{enc}_rcfd_{dual}_supcon_mtl` | `[cls_out, reg_out, proj]` | + supcon on z_cond |
| SupCon v2 | `{enc}_rcfd_{dual}_supcon2_mtl` | `[cls_out, reg_out, cnn_proj, seq_proj]` | + supcon on branches |
| SupCon v3 | `{enc}_rcfd_{dual}_supcon3_mtl` | `[cls_out, reg_out, cnn_proj, seq_proj, fused_proj]` | + supcon on branches + z_cond |

Where `{enc}` ∈ {cnn, gru, trans} and `{dual}` ∈ {cgd, ctd}.

**SupCon projection targets:**
- v1: `z_cond` (post-FiLM, 96-dim) — contrastive learning in concentration-conditioned space
- v2: `cnn_emb` (32-dim, pre-FiLM) + `seq_emb` (64-dim, pre-FiLM) — pure branch representations
- v3: all of v2 + `z_cond` (96-dim, post-FiLM) — adds third projection on the conditioned fused space

Note: all RCFD variants are MTL-only (no ST). The early encoder requires concentration labels to learn a meaningful `c_pred` — without regression supervision, FiLM conditioning degrades to noise.

---

## FiLM Hardening

Three measures applied in `_apply_film_scalar`:

### 1. stop_gradient
```python
c_sg = tf.stop_gradient(c_pred)
```
Classification loss must not flow into the early encoder through FiLM. Without this, `c_pred` would be shaped to help classification rather than predict concentration — destroying the regression head.

### 2. BatchNormalization on c_pred
```python
c_bn = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_c_bn')(c_sg)
```
Early in training the early encoder is untrained — c_pred can be arbitrary. BN clips extreme values before they reach the FiLM Dense layers. Side-effect: sentinel samples (no concentration label → unsupervised encoder → c_pred near-zero after BN) → with identity init → γ≈1, β≈0 → FiLM is identity → graceful neutral behavior for missing-concentration samples.

### 3. Identity Initialization (FiLM collapse prevention)
```python
gamma = Dense(emb_dim, kernel_initializer='zeros', bias_initializer='ones',  name='film_gamma')(c_bn)
beta  = Dense(emb_dim, kernel_initializer='zeros', bias_initializer='zeros', name='film_beta')(c_bn)
```
At epoch 0: `z_cond = 1·z_raw + 0 = z_raw`. FiLM starts as exact identity; training learns deviations from identity. Prevents the zero-gain collapse where random γ≈0 kills the entire backbone signal.

---

## Early Encoder Architectures (lightweight by design)

The early encoder produces one scalar — increasing its complexity adds parameters without enriching what FiLM can do. The three types test which architectural family is most *compatible* with each dual backbone (gradient flow, feature complementarity), not which one predicts concentration most accurately.

| Type | Architecture |
|------|-------------|
| cnn | Conv1D(16,5,relu) → Conv1D(8,3,relu) → GAP → Dense(32,relu) → reg_head |
| gru | BiGRU(16,no_seq) → Dense(32,relu) → reg_head |
| transformer | Conv1D(32,5,s=2) → MaxPool(2) + pos_emb → 1×TransBlock(2h,kd16,FFN32) → GAP → Dense(32,relu) → reg_head |

reg_head: `Dense(16,relu) → Dense(8,relu) → Dense(1,linear, name='reg_out')`

`_build_early_encoder` returns `(reg_emb, reg_out)`. `reg_emb` (32-dim) is exposed for future SupCon projection heads on the early encoder itself.

---

## Files to Create / Modify

### New: `utils/model_training/model_utils_rcfd.py`

**4 model classes** (pure subclasses — no logic override, just different Keras serialization package):
```python
@tf.keras.utils.register_keras_serializable(package='rcfd')
class RCFDModel(MTLModel): pass

@tf.keras.utils.register_keras_serializable(package='rcfd_sc1')
class RCFDSupConMTLModel(SupConMTLModel): pass

@tf.keras.utils.register_keras_serializable(package='rcfd_sc2')
class RCFDBranch2MTLModel(SupConBranch2MTLModel): pass

@tf.keras.utils.register_keras_serializable(package='rcfd_sc3')
class RCFDBranch3MTLModel(SupConBranch3MTLModel): pass
```

**4 build functions:**
- `_build_rcfd_model(T, n, enc_type, dual_type)` → `RCFDModel` — base
- `_build_rcfd_supcon_model(...)` → `RCFDSupConMTLModel` — adds `_proj_head(z_cond, 'fused')`
- `_build_rcfd_supcon2_model(...)` → `RCFDBranch2MTLModel` — needs `return_branches=True`; `_proj_head` on `cnn_emb` and `seq_emb`
- `_build_rcfd_supcon3_model(...)` → `RCFDBranch3MTLModel` — v2 + `_proj_head(z_cond, 'fused')`

**24 factory functions** (6 per variant), `_RCFD_ALL_FACTORIES` dict, and key lists:
```python
RCFD_MODEL_KEYS             = [...]   # 6 base keys
RCFD_SUPCON_MTL_MODEL_KEYS  = [...]   # 6 sc1 keys
RCFD_BRANCH2_MTL_MODEL_KEYS = [...]   # 6 sc2 keys
RCFD_BRANCH3_MTL_MODEL_KEYS = [...]   # 6 sc3 keys
ALL_RCFD_KEYS               = [...]   # all 24
```

Imports: `MTLModel`, `_build_cnn_gru_dual_branches_mtl`, `_build_cnn_trans_dual_branches_mtl` from `model_utils_mtl`; `SupConMTLModel`, `SupConBranch2MTLModel`, `SupConBranch3MTLModel`, `_proj_head`, `SUPCON_TEMP` from `model_utils_supcon`.

### `model_utils.py`

- Import all 4 classes + key lists + `_RCFD_ALL_FACTORIES` from `model_utils_rcfd`
- `_XAI_SAVE_NAME.update({k: k for k in ALL_RCFD_KEYS})`
- 4 dispatch blocks (one per variant group) after CL_MTL block
  - All use `jit_compile=False`, epochs=500, same conc normalization as MTL block
  - Predict: `cls_prob, reg_pred_scaled, *_ = model.predict(X_test_curve, verbose=0)` — `*_` discards proj outputs
- Extend reg results saving: `if (...or _base_m in ALL_RCFD_KEYS) and reg_preds_per_fold:`

### `config.py`

- **`MODEL_KEY_MAP`**: 24 entries (3-tuple: `y_preds_`, `y_probs_`, `classes_`) after CL section
- **`_RCFD_MODEL_KEYS`**: one set of all 24 keys (after `_CL_MTL_MODEL_KEYS`)
- **`_NO_INC`**: add `| _RCFD_MODEL_KEYS`
- **`MODEL_PRINT_MAP`**: 24 entries, pattern `"CNN RCFD CGD"`, `"CNN RCFD CGD SC1"`, `"CNN RCFD CGD SC2"`, `"CNN RCFD CGD SC3"` etc.

### `03_main_training.py`

- New arg: `--condreg` boolean flag (`action="store_true"`); implies `--mtl` — no need to declare both
- Model selection matrix (`--condreg` boolean × `--supcon 0/1/2/3`):
  ```python
  if getattr(args, 'condreg', False):
      if args.supcon == 0:   models = list(RCFD_MODEL_KEYS)
      elif args.supcon == 1: models = list(RCFD_SUPCON_MTL_MODEL_KEYS)
      elif args.supcon == 2: models = list(RCFD_BRANCH2_MTL_MODEL_KEYS)
      elif args.supcon == 3: models = list(RCFD_BRANCH3_MTL_MODEL_KEYS)
  ```
  Each call trains all 6 models for that variant (3 encoders × 2 dual backbones).
- Banner: append `f" RCFD SC{args.supcon}"`
- force_rerun: add `_rcfd_result_keys` set and clearing `elif`
- Import `RCFD_MODEL_KEYS, RCFD_SUPCON_MTL_MODEL_KEYS, RCFD_BRANCH2_MTL_MODEL_KEYS, RCFD_BRANCH3_MTL_MODEL_KEYS` from `model_utils_rcfd`

### `04_cross_dataset_training.py`

Same changes as `03_main_training.py`.

### `07_attribution_vis_all.py`

- Add all 24 RCFD keys to `model_names`
- Add all 4 classes to `custom_objects`
- Import from `model_utils_rcfd`

### `06_model_prediction_report.py`

No changes — auto-detects from `MODEL_KEY_MAP` and `y_reg_preds_{mk}_` entries.

---

## SLURM: `slurm_jobs/lab_rcfd_training.sh`

```bash
for SC in 0 1 2 3; do
    SARG=(); [ "$SC" -gt 0 ] && SARG=(--supcon "$SC")
    run_train --condreg "${SARG[@]}"
done
# 4 training calls × 6 models each = 24 models total
# then 06_model_prediction_report.py, 07_attribution_vis_all.py
```

---

## SupCon Extension Notes

The current design already *is* the SupCon extension (v1/v2/v3 are implemented). For further extensions:
- Adding SupCon to the early encoder itself: tap `reg_emb` from `_build_early_encoder` and add `_proj_head(reg_emb, 're')` — a fourth projection head. Would require a new model class.
- Options 7–9 (shared encoder, confidence-gated, multi-level FiLM): `_apply_film_scalar` accepts `name_prefix` to support multi-level reuse; confidence-gating and shared encoder require separate build functions.

---

## Verification

```bash
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main

# All 24 factories
python -c "
from utils.model_training.model_utils_rcfd import _RCFD_ALL_FACTORIES
import numpy as np
for key, factory in _RCFD_ALL_FACTORIES.items():
    m = factory(100, 3)
    x = np.random.randn(4, 100, 1).astype('float32')
    outs = m(x, training=False)
    assert outs[0].shape == (4, 3), key
    print(f'  {key}: {len(outs)} outputs OK')
"

# Import chain
python -c "import utils.model_training.model_utils; print('OK')"

# Quick training (2 folds, all 4 supcon variants)
for SC in 0 1 2 3; do
    python -u 03_main_training.py --task_id 0 \
        --exp_folder /vol/bitbucket/gk225/POC_DDM_datasets/LAB_DDM_paper \
        --curve_type ori_curve --n_splits 2 \
        --condreg --supcon $SC
done
```
