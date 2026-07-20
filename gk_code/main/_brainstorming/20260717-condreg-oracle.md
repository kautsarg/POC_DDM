# CondReg Oracle — GT-FiLM Concept Proof (`03c_condreg_oracle_poc.py`)

**Date:** 2026-07-17
**Status:** Planning

---

## 1. Motivation

The RCFD models use a lightweight early encoder to *predict* concentration from the curve,
then condition the dual backbone via FiLM using that prediction. Two failure modes exist:

1. The early encoder predicts concentration poorly → FiLM receives a bad signal → no gain over baseline
2. FiLM itself is not useful even with a perfect regressor → the architectural idea is wrong

This concept proof isolates failure mode 2 by **bypassing the early encoder entirely** and
injecting the ground-truth (GT) concentration directly into the FiLM layer. If this "oracle"
model outperforms RCFD, the regression quality is the bottleneck, not the FiLM design. If it
performs no better, the FiLM architecture itself needs rethinking.

---

## 2. Architecture

### Current RCFD

```
Input (batch, T, 1)
    ├──→ Early Encoder (CNN/GRU/Trans) → reg_emb → Dense(1) → reg_out (batch, 1)  ← REGRESSION
    │                                                               │
    │         stop_gradient + BN + Dense(96) gamma/beta            │
    │         ↓                                                     │
    └──→ Dual Backbone (CNN+GRU or CNN+Trans) → z_raw (batch, 96)  │
                                                      ↓             ↓
                                              _apply_film_scalar(reg_out, z_raw)
                                                      ↓
                                              z_cond → Dropout(0.2) → _cls_head → cls_out
```

### GT-FiLM Oracle (new — identical everywhere except the source of `c_pred`)

```
GT conc_input (batch, 1) ─ pre-normalized, per-fold scaler ─────────────────────────┐
                                                                                      │
Input (batch, T, 1) ──→ Dual Backbone (CNN+GRU or CNN+Trans) → z_raw (batch, 96)   │
                         [IDENTICAL architecture + parameters to RCFD]               ↓
                                                              _apply_film_scalar(conc_input, z_raw)
                                                              [IDENTICAL FiLM layers + parameters]
                                                                      ↓
                                                              z_cond → Dropout(0.2) → _cls_head → cls_out
                                                              [IDENTICAL cls head + parameters]
```

**The ONLY structural difference:**

| Component | RCFD | GT-FiLM Oracle |
|---|---|---|
| Early encoder | CNN/GRU/Trans → `reg_out (batch,1)` | **removed** |
| `c_pred` to FiLM | model's own `reg_out` (predicted) | **GT `conc_input` (batch,1)** |
| Dual backbone | `_build_cnn_gru/trans_dual_branches_mtl` | **identical** |
| FiLM layer | `_apply_film_scalar(reg_out, z_raw, 96)` | **identical** (same weights, same BN) |
| Classification head | `Dropout(0.2) → _cls_head` | **identical** |
| Model class | `RCFDModel(MTLModel)` — CE + masked MSE | plain `tf.keras.Model` — CE only |
| Inputs | single `[curve_input]` | `[curve_input, conc_input]` |

The dual backbone, FiLM conditioning layers, and classification head have the **same architecture
and the same parameter count** as the corresponding RCFD model. The only parameters removed are
those of the early encoder (which IS the regression component being replaced).

`_apply_film_scalar`'s `_StopGradient` is a no-op for a model input tensor — safe to reuse
unchanged. The BN inside still normalises the GT concentration scalar the same way it normalises
`reg_out` in RCFD.

---

## 3. Data Flow

```
curve_for_training.joblib["concentration"]
    → y_concentration (N,) float array, sentinel=-1.0 for controls
    → same array main/03 already loads

Per fold (in 03c's own training loop):
    conc_train_scaled, scaler = _normalize_concentration(conc_f[tr])
        # masks out REG_SENTINEL, log10 of valid values, StandardScaler fit on train
    conc_test_scaled = apply scaler to conc_f[te] (sentinel entries stay = -1.0)

model.fit(
    [X_f[tr], conc_train_scaled.reshape(-1,1)],   # GT concentration as second input
    y_f[tr],                                        # same class labels as main/03
)
model.predict([X_f[te], conc_test_scaled.reshape(-1,1)])
```

Concentration normalization is **identical** to what RCFD's training block already does
(`model_utils_mtl._normalize_concentration`). Per-fold scaler fitted on train only.

---

## 4. Sentinel Handling

Controls (e.g., NTC) have concentration = `REG_SENTINEL = -1.0`. After
`_apply_film_scalar`'s internal BatchNorm, sentinel values (which cluster tightly at -1.0)
produce a near-zero BN activation → γ ≈ 1, β ≈ 0 → near-identity FiLM.

This is **the same graceful degradation** as RCFD: sentinel samples effectively skip FiLM
conditioning and use `z_raw` directly. No special sentinel handling needed in 03c.

---

## 5. Single New File: `main/03c_condreg_oracle_poc.py`

Everything lives in this one file — model builder, registry, training loop. Zero other new files.

### Model builder (inside 03c)

```python
# ── imports at top of 03c ──────────────────────────────────────────────────
from model_utils_mtl  import (_normalize_concentration, REG_SENTINEL,
                               _build_cnn_gru_dual_branches_mtl,
                               _build_cnn_trans_dual_branches_mtl)
from model_utils_rcfd import _apply_film_scalar, _cls_head

# ── model factory ──────────────────────────────────────────────────────────
def _build_gt_film_model(T, n_classes, dual_type='cgd'):
    """RCFD-identical backbone; GT concentration replaces early encoder reg_out."""
    curve_input = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    conc_input  = tf.keras.layers.Input(shape=(1,),   name='conc_input')

    if dual_type == 'cgd':
        z_raw = _build_cnn_gru_dual_branches_mtl(curve_input)   # identical to RCFD CGD
        pfx   = 'film_cgd'
    else:
        z_raw = _build_cnn_trans_dual_branches_mtl(curve_input)  # identical to RCFD CTD
        pfx   = 'film_ctd'

    z_cond  = tf.keras.layers.Dropout(0.2, name=f'{pfx}_drop')(
        _apply_film_scalar(conc_input, z_raw, emb_dim=96, name_prefix=pfx))  # identical FiLM
    cls_out = _cls_head(z_cond, n_classes)                                    # identical cls head

    model = tf.keras.Model(inputs=[curve_input, conc_input], outputs=cls_out,
                           name=f'gt_film_{dual_type}')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model

# ── local registry (no config.MODEL_KEY_MAP changes) ──────────────────────
_GT_FILM_MODELS = ['gt_film_cgd', 'gt_film_ctd']

_GT_FILM_KEY_MAP = {
    'gt_film_cgd': ('y_preds_AC_gt_film_cgd_', 'y_probs_AC_gt_film_cgd_', 'classes_AC_gt_film_cgd_'),
    'gt_film_ctd': ('y_preds_AC_gt_film_ctd_', 'y_probs_AC_gt_film_ctd_', 'classes_AC_gt_film_ctd_'),
}

_GT_FILM_PRINT_MAP = {
    'gt_film_cgd': 'GT-FiLM CGD',
    'gt_film_ctd': 'GT-FiLM CTD',
}
```

> **`name_prefix` is required** to avoid Keras layer name collisions when CGD and CTD models
> are both built within the same session (both call `_apply_film_scalar`).

### Training loop (inside 03c)

CLI mirrors main/03: `--task_id`, `--exp_folder`, `--n_splits`, `--mode`, `--filters`, `--force_rerun`.

```python
# Load same curve_for_training.joblib that main/03 uses
training_data = joblib.load(curve_for_training_path)
X, y_encoded, y_concentration, n_classes = extract_mode_data(training_data, mode)

# Load existing results and append
all_results = joblib.load(results_path) if os.path.exists(results_path) else {}

for f in outlier_filters:
    apply filter mask → X_f, y_f, conc_f

    for m in _GT_FILM_MODELS:
        preds_key, probs_key, classes_key = _GT_FILM_KEY_MAP[m]
        dual_type = 'cgd' if 'cgd' in m else 'ctd'

        preds_folds, probs_folds, classes_folds = [], [], []
        t0 = perf_counter()

        for fold_idx, (tr, val, te) in enumerate(splits):
            conc_train_scaled, scaler = _normalize_concentration(conc_f[tr])
            conc_val_scaled           = _apply_scaler(conc_f[val], scaler)
            conc_test_scaled          = _apply_scaler(conc_f[te],  scaler)

            model = _build_gt_film_model(T, n_classes, dual_type)
            model.fit(
                [X_f[tr], conc_train_scaled.reshape(-1, 1)],
                y_f[tr],
                validation_data=([X_f[val], conc_val_scaled.reshape(-1, 1)], y_f[val]),
                callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                epochs=200, batch_size=64, verbose=0,
            )
            probs = model.predict([X_f[te], conc_test_scaled.reshape(-1, 1)], verbose=0)
            preds = probs.argmax(-1)

            preds_folds.append(preds)
            probs_folds.append(probs)
            classes_folds.append(le.classes_)
            tf.keras.backend.clear_session(); gc.collect()

        # Append-only: existing keys in the result entry are untouched
        entry = all_results[dataset_title][mode][f]
        entry[preds_key]   = preds_folds
        entry[probs_key]   = probs_folds
        entry[classes_key] = classes_folds
        safe_joblib_dump(all_results, results_path)

        duration = perf_counter() - t0
        accs = [accuracy_score(entry['y_trues_'][i], preds_folds[i])
                for i in range(len(preds_folds))]
        hh, mm, ss = int(duration)//3600, int(duration)%3600//60, int(duration)%60
        print(f"  [+] {_GT_FILM_PRINT_MAP[m]} | "
              f"acc={np.mean(accs)*100:.2f}%±{np.std(accs)*100:.2f}% | "
              f"{hh:02d}:{mm:02d}:{ss:02d}")
```

---

## 6. Result Structure

New keys are appended inside the existing `classification_performances.joblib` entry for each
`(dataset_title, mode, filter)` combination. **No existing keys are changed.**

```python
# Before 03c:
entry = {
    'y_trues_': [...],
    'y_preds_AC_cnn_rcfd_cgd_': [...],
    'y_reg_preds_cnn_rcfd_cgd_': [...],
    ...
}

# After 03c:
entry = {
    'y_trues_': [...],                           # unchanged
    'y_preds_AC_cnn_rcfd_cgd_': [...],           # unchanged
    'y_reg_preds_cnn_rcfd_cgd_': [...],          # unchanged
    ...
    'y_preds_AC_gt_film_cgd_': [...],            # NEW
    'y_probs_AC_gt_film_cgd_': [...],            # NEW
    'classes_AC_gt_film_cgd_': [...],            # NEW
    'y_preds_AC_gt_film_ctd_': [...],            # NEW
    'y_probs_AC_gt_film_ctd_': [...],            # NEW
    'classes_AC_gt_film_ctd_': [...],            # NEW
}
```

`y_trues_` already exists from when main/03 ran. 03c reads it directly to compute accuracy.
No `y_reg_*` keys — GT-FiLM has no regression output.

---

## 7. Reuse Map (import-only, zero file changes)

All imports inside `03c_condreg_oracle_poc.py`:

| What | From |
|---|---|
| `_build_cnn_gru_dual_branches_mtl(inputs)` | `model_utils_mtl.py` |
| `_build_cnn_trans_dual_branches_mtl(inputs)` | `model_utils_mtl.py` |
| `_apply_film_scalar(c_pred, z_raw, emb_dim, name_prefix)` | `model_utils_rcfd.py` (line 87) |
| `_cls_head(z_cond, n_classes)` | `model_utils_rcfd.py` (line 128) |
| `_normalize_concentration`, `REG_SENTINEL` | `model_utils_mtl.py` |
| `safe_joblib_dump` | `safe_io.py` |
| `get_exp_paths`, `check_task_id` | `pipeline_utils.py` |
| `set_global_determinism` | `model_utils.py` |
| `config.*` (EXP_FOLDER, OUTLIER_FILTERS, TRAINING_RESULT_PATH, …) | `config.py` |

---

## 8. What is NOT Touched

- `main/config.py` — no `MODEL_KEY_MAP` additions; `_GT_FILM_KEY_MAP` is local to `03c`
- `main/03_main_training.py` — completely unchanged
- `main/utils/model_training/model_utils*.py` — all imported read-only, no modifications
- Existing `classification_performances.joblib` content — only new keys appended

---

## 9. Expected Outcomes and Interpretation

| Outcome | Interpretation |
|---|---|
| GT-FiLM >> RCFD (e.g., +5% acc) | Regression quality is the bottleneck; better regressor = better classifier |
| GT-FiLM ≈ RCFD (within noise) | FiLM benefits saturate early; early encoder already captures the useful signal |
| GT-FiLM ≈ baseline no-FiLM | FiLM architecture is not helpful; concentration does not add useful information |
| GT-FiLM > baseline but GT-FiLM ≈ RCFD | FiLM helps but the regressor is good enough; concept validated |

---

## 10. Checklist

**Single new file: `main/03c_condreg_oracle_poc.py`** ← IMPLEMENTED 2026-07-17
- [x] `_build_gt_film_model(T, n_classes, dual_type)` — 2-input Keras model
- [x] `_GT_FILM_MODELS`, `_GT_FILM_KEY_MAP`, `_GT_FILM_PRINT_MAP` — local registry
- [x] CLI args (`--task_id`, `--exp_folder`, `--n_splits`, `--mode`, `--curve_type`, `--force_rerun`)
- [x] Data loading (same `curve_for_training.joblib` keys as main/03)
- [x] K-fold loop with dual-input `model.fit([X_curves, conc_scaled], y)`
- [x] Append-only joblib write (existing keys untouched)
- [x] `--force_rerun` clears ONLY GT-FiLM keys (existing RCFD/standard results preserved)
- [x] Progress print: `[+] GT-FiLM CGD | acc=XX.XX%±Y.YY% | HH:MM:SS`
- [x] Leaderboard summary at end (GT-FiLM rows alongside RCFD rows from loaded results)
- [x] SLURM script: `POC_DDM/slurm_jobs/lab_condreg_poc_training.sh`

**Verification:**
- [ ] `gt_film_cgd` trains without shape errors (two-input model)
- [ ] `gt_film_ctd` trains without shape errors
- [ ] Sentinel concentration samples handled gracefully (BN near-identity FiLM)
- [ ] `classification_performances.joblib` now has 6 new keys per result entry
- [ ] Existing RCFD/standard model keys in joblib are unmodified

---

## 11. Verification Commands

```bash
# Smoke: 2 folds, Native mode, baseline filter only
python 03c_condreg_oracle_poc.py \
  --task_id 0 --n_splits 2 --mode Native --force_rerun

# Expected:
# - GT-FiLM CGD and GT-FiLM CTD train and complete without errors
# - classification_performances.joblib now has y_preds_AC_gt_film_cgd_ etc.
# - Leaderboard shows GT-FiLM rows with accuracy comparable to or above RCFD rows

# Quick comparison print:
python -c "
import joblib, numpy as np
from sklearn.metrics import accuracy_score
d = joblib.load('classification_performances.joblib')
for ds, modes in d.items():
    for mode, filters in modes.items():
        for f, res in filters.items():
            for key in ['y_preds_AC_cnn_rcfd_cgd_', 'y_preds_AC_gt_film_cgd_']:
                if key in res:
                    accs = [accuracy_score(yt, yp) for yt, yp in zip(res['y_trues_'], res[key])]
                    print(f'{key}: {np.mean(accs)*100:.2f}%')
"
```
