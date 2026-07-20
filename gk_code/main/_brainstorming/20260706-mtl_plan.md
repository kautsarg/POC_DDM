# Plan: Multi-Task Learning (MTL) — Classification + Concentration Regression

## Context

Existing models train only on the classification task (dPCR target prediction). Concentration varies across experiments, and the model may overfit to concentration-correlated artefacts rather than learning true target identity signals. The MTL extension adds a **regression head** that predicts concentration jointly with the classification head, forcing the shared backbone to learn representations that are useful for both tasks — producing classifiers that generalise regardless of concentration.

The `concentration` key now exists in every `curve_for_training.joblib` (added by `01b_lab_curve_preprocessing.py` and back-patched by `adhoc_patch_concentration.py`). Training scripts `03_main_training.py` currently never read it. This plan wires it in.

Loss weighting follows **Kendall et al. (2018)** homoscedastic uncertainty: two learnable log-variance scalars `s_cls`, `s_reg` automatically balance the classification and regression losses during training, avoiding brittle fixed-weight hyperparameters.

References: Caruana (1997) shared representation; Kokkinos UberNet (2017) multi-task single backbone; Kendall et al. (2018) uncertainty weighting; Kirchdorfer (2024/2025) MTL kinetics; Xiao LDC-MTL (2025); Durmus MICCAI (2024).

---

## Architecture

Each MTL model shares the **exact same backbone** as its classification counterpart — only the final output layer changes.

```
curve input (N, T, 1)
        ↓
  [shared backbone — CNN, GRU, Transformer, or dual CNN+GRU/Trans]
        ↓
  embedding (N, D)          ← same as current Dense(64, relu) → Dropout(0.2) output
       ├──→ Dense(n_classes, softmax, name='cls_out')    [classification head]
       └──→ Dense(1, linear, name='reg_out')             [regression head]
```

For single-branch models (GRU, Transformer, CNN): backbone ends at the `Dropout` layer that currently precedes `Dense(output_size, softmax)` — both heads attach to that dropout output.

For dual models (`cnn_gru_dual`, `cnn_trans_dual`) and gated models: backbone ends at the existing `Dense(64, relu) → Dropout(0.2)` merge — both heads attach to the Dropout output.

---

## Loss Function — Kendall Uncertainty Weighting

```
L_total = exp(-s_cls) * L_cls + s_cls
        + exp(-s_reg) * L_reg_masked + s_reg
```

- `s_cls`, `s_reg` are **trainable scalars** (log-variance), initialised to 0
- `L_cls` = sparse categorical cross-entropy on classification output
- `L_reg_masked` = masked MSE: zero contribution for samples where concentration is `None` (stored as sentinel `–1.0`)

```python
mask = tf.cast(y_reg != REG_SENTINEL, tf.float32)   # REG_SENTINEL = -1.0
L_reg_masked = tf.reduce_sum(mask * tf.square(y_reg - reg_out[:,0])) / (tf.reduce_sum(mask) + 1e-8)
```

This means experiments without concentration data contribute 0 regression loss; the classification task is unaffected.

---

## Implementation

### New file: `utils/model_training/model_utils_mtl.py`

Contains:

1. **`MTLModel` — custom Keras Model subclass**

   Overrides `train_step` and `test_step` to compute the Kendall-weighted loss internally. Holds `log_var_cls` and `log_var_reg` as trainable weights added via `self.add_weight(...)`. Compiled with `model.compile(optimizer=..., metrics=['accuracy'])` — no `loss=` argument needed.

   ```python
   REG_SENTINEL = -1.0

   class MTLModel(tf.keras.Model):
       def __init__(self, *args, reg_sentinel=REG_SENTINEL, **kwargs):
           super().__init__(*args, **kwargs)
           self.reg_sentinel = reg_sentinel
           self.log_var_cls = self.add_weight('log_var_cls', shape=(), initializer='zeros', trainable=True)
           self.log_var_reg = self.add_weight('log_var_reg', shape=(), initializer='zeros', trainable=True)

       def _compute_loss(self, y_cls, y_reg, cls_out, reg_out):
           ce = tf.reduce_mean(
               tf.keras.losses.sparse_categorical_crossentropy(y_cls, cls_out))
           mask = tf.cast(tf.not_equal(y_reg, self.reg_sentinel), tf.float32)
           mse = tf.reduce_sum(mask * tf.square(y_reg - reg_out[:, 0])) / (tf.reduce_sum(mask) + 1e-8)
           loss = (tf.exp(-self.log_var_cls) * ce + self.log_var_cls +
                   tf.exp(-self.log_var_reg) * mse + self.log_var_reg)
           return loss, mse

       def train_step(self, data):
           x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
           with tf.GradientTape() as tape:
               cls_out, reg_out = self(x, training=True)
               loss, mse = self._compute_loss(y_dict['cls_out'], y_dict['reg_out'], cls_out, reg_out)
           self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
           self.compiled_metrics.update_state(y_dict['cls_out'], cls_out)
           return {m.name: m.result() for m in self.metrics} | {'loss': loss, 'reg_mse': mse}

       def test_step(self, data):
           x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
           cls_out, reg_out = self(x, training=False)
           loss, mse = self._compute_loss(y_dict['cls_out'], y_dict['reg_out'], cls_out, reg_out)
           self.compiled_metrics.update_state(y_dict['cls_out'], cls_out)
           return {m.name: m.result() for m in self.metrics} | {'loss': loss, 'reg_mse': mse}
   ```

2. **MTL model factory functions** — one per primary neural architecture:
   - `create_cnn_mtl_model(T, n_classes)`
   - `create_lstm_mtl_model(T, n_classes)`
   - `create_gru_mtl_model(T, n_classes)`
   - `create_rnn_mtl_model(T, n_classes)`
   - `create_transformer_mtl_model(T, n_classes)`
   - `create_cnn_gru_dual_mtl_model(T, n_classes)`  ← also reused for cosine_recon_mtl (no new factory)
   - `create_cnn_trans_dual_mtl_model(T, n_classes)`
   - `create_cnn_gru_dual_attn_recon_mtl_model(k_plus_1, T, n_classes)` ← dedicated factory; see below

   **LF models (`cnn_lf`, `lstm_lf`, `gru_lf`, `trans_lf`) and GNN models (`gnn_gat`, `gnn_gcn`)** — deferred to follow-up. LF models take a mixed `(curve, features)` input that complicates the MTL training path; GNN models require a graph adjacency input. Both need separate handling and are out of scope for Phase 1.

   **`lstm_ae_clf`** — deferred. Its classification head is not a standard Dense softmax (it wraps a pretrained encoder); MTL adaptation requires loading the encoder separately before adding the regression head.

   The shared backbone is **unchanged in depth and width** — extra capacity is not needed. MTL's benefit comes from two gradient signals regularising the same weights, not from adding parameters. With the small per-fold sample counts typical of these experiments, more parameters would increase overfitting risk. If both task training losses remain high simultaneously (capacity-bound), widening can be reconsidered, but underfitting is unlikely given the existing models already converge well.

   **Regression head**: the plan uses `Dense(1, linear)` directly from the shared embedding. If the concentration–curve relationship shows nonlinearity, optionally deepen it to `Dense(16, relu) → Dense(1, linear)` without touching the shared backbone. Start shallow; only add the hidden layer if the regression MAE on validation folds is noticeably worse than a standalone regression baseline.

   Each function duplicates its classification counterpart's backbone verbatim — stopping before `Dense(output_size, softmax)` — then attaches the two heads and wraps in `MTLModel`:
   ```python
   def create_cnn_gru_dual_mtl_model(T, n_classes):
       # ... (same CNN+GRU backbone as create_cnn_gru_dual_model / _build_cnn_gru_dual_branches) ...
       embedding = Dropout(0.2)(Dense(64, 'relu')(merged))
       cls_out = Dense(n_classes, 'softmax', name='cls_out')(embedding)
       reg_out = Dense(1, 'linear', name='reg_out')(embedding)  # optionally Dense(16,relu)→Dense(1)
       return MTLModel(inputs=[curve_input], outputs=[cls_out, reg_out])
   ```

   **`cnn_gru_dual_cosine_recon_mtl`** — the cosine reconstruction is a pure data preprocessing step (`reconstruct_curves_cosine()` is called before training, producing a plain `(N, T)` curve). The model architecture is **identical to `cnn_gru_dual_mtl`**; no new factory is needed. In `evaluate_outlier_filters`, when the model key is `cnn_gru_dual_cosine_recon_mtl`, feed `X_AC_cosine_recon` (same as the non-MTL variant) and use the MTL training path. Add `cnn_gru_dual_cosine_recon_mtl` to `_SPATIAL_RECON_MODELS` so the neighbor stack computation is triggered.

   **`cnn_gru_dual_attn_recon_mtl`** — needs a dedicated factory. Same as `create_cnn_gru_dual_attn_recon_model` but replace the final `Dense(output_size, softmax)` with two heads:
   ```python
   def create_cnn_gru_dual_attn_recon_mtl_model(k_plus_1, T, n_classes, attn_dim=16):
       # ... (same stack_input → per_curve_encoder → attention → reconstructed → _build_cnn_gru_dual_branches) ...
       z = _build_cnn_gru_dual_branches(reconstructed)   # 64-dim embedding
       cls_out = Dense(n_classes, 'softmax', name='cls_out')(z)
       reg_out = Dense(1, 'linear', name='reg_out')(z)
       return MTLModel(inputs=stack_input, outputs=[cls_out, reg_out])
   ```
   Input remains `(N, k+1, T)` stack; uses `X_AC_stack` in `evaluate_outlier_filters`.

   Gated model variants (8 factories) are **deferred to a follow-up** — Phase 1 targets the 7 models above.

3. **`_normalize_concentration(conc_array)` helper**:
   Fits `StandardScaler` on non-sentinel training values; returns `(scaled_array, scaler)` — sentinel values are preserved as `REG_SENTINEL` (not scaled).

4. **`MTL_MODEL_KEYS`** list — names for the 9 MTL models (Phase 1), matching `model_key_map` convention:
   ```python
   MTL_MODEL_KEYS = [
       'cnn_mtl', 'lstm_mtl', 'gru_mtl', 'rnn_mtl', 'transformer_mtl',
       'cnn_gru_dual_mtl', 'cnn_trans_dual_mtl',
       'cnn_gru_dual_cosine_recon_mtl', 'cnn_gru_dual_attn_recon_mtl',
   ]
   ```
   Gated variants (8) remain deferred — added to `MTL_MODEL_KEYS` in Phase 2 when their factories are implemented.

---

### Modified: `utils/model_training/model_utils.py`

**`evaluate_outlier_filters` signature change** — add two parameters only; do NOT add `result_path` (saving is done by the caller, not this function):
```python
def evaluate_outlier_filters(
    ...,
    multitask=False,
    y_concentration=None,   # float array (N,); sentinel-encoded upstream
):
```

**Inside the per-fold training loop**, when `multitask=True` and the current model key is in `MTL_MODEL_KEYS`:
1. Extract fold concentration: `conc_train = y_concentration[train_idx]`, `conc_val = y_concentration[val_idx]`
2. Normalize: `conc_train_scaled, scaler = _normalize_concentration(conc_train)`; transform val (skipping sentinels)
3. Replace targets with dict form: `{'cls_out': y_train, 'reg_out': conc_train_scaled}`
4. Compile: `model.compile(optimizer=optimizer, metrics=['accuracy'])` — no `loss=` argument
5. After prediction: unpack `cls_probs, reg_preds_scaled = model.predict(X_val)`, inverse-transform reg preds
6. Store extra result keys alongside the existing classification keys (same results dict entry):
   ```python
   results[f'y_reg_preds_{model_name}_'].append(reg_preds_original_scale)
   results[f'y_reg_trues_{model_name}_'].append(conc_val_original_scale)
   ```

**`_NO_INC_INCEPTION` must include all MTL keys** (line ~704 inside `evaluate_outlier_filters`):
```python
_NO_INC_INCEPTION = {"rf", "knn", "ffi", "cnn_gru_dual_attn_recon"} | set(MTL_MODEL_KEYS)
```
MTL models never use inception smoothing; adding them prevents spurious `_mtl_inc` entries.

**`_SPATIAL_RECON_MODELS` must include both MTL spatial variants** (line ~712):
```python
_SPATIAL_RECON_MODELS = (
    "cnn_gru_dual_cosine_recon", "cnn_gru_dual_attn_recon",
    "cnn_gru_dual_cosine_recon_mtl", "cnn_gru_dual_attn_recon_mtl",
)
```
The existing `removesuffix('_inc')` check in the spatial-recon routing block becomes `re.sub(r'_(inc|mtl)$', '', _base_m)` — or more simply, just include the full MTL key names in the tuple and match exactly.

Non-MTL models and all sklearn models are unaffected — the new parameters default to `False` / `None`.

Add import at top of `model_utils.py`:
```python
from model_utils_mtl import (
    create_cnn_mtl_model, create_lstm_mtl_model, create_gru_mtl_model,
    create_rnn_mtl_model, create_transformer_mtl_model,
    create_cnn_gru_dual_mtl_model, create_cnn_trans_dual_mtl_model,
    create_cnn_gru_dual_attn_recon_mtl_model,
    _normalize_concentration, MTL_MODEL_KEYS, REG_SENTINEL,
)
```

Add MTL entries to the local `model_key_map` dict inside `evaluate_outlier_filters` (lines ~665–710):
```python
# Single-branch MTL
'cnn_mtl':                       ('y_preds_AC_cnn_mtl_',                    'y_probs_AC_cnn_mtl_',                    'classes_AC_cnn_mtl_'),
'lstm_mtl':                      ('y_preds_AC_lstm_mtl_',                   'y_probs_AC_lstm_mtl_',                   'classes_AC_lstm_mtl_'),
'gru_mtl':                       ('y_preds_AC_gru_mtl_',                    'y_probs_AC_gru_mtl_',                    'classes_AC_gru_mtl_'),
'rnn_mtl':                       ('y_preds_AC_rnn_mtl_',                    'y_probs_AC_rnn_mtl_',                    'classes_AC_rnn_mtl_'),
'transformer_mtl':               ('y_preds_AC_trans_mtl_',                  'y_probs_AC_trans_mtl_',                  'classes_AC_trans_mtl_'),
# Dual-branch MTL
'cnn_gru_dual_mtl':              ('y_preds_AC_cnn_gru_dual_mtl_',           'y_probs_AC_cnn_gru_dual_mtl_',           'classes_AC_cnn_gru_dual_mtl_'),
'cnn_trans_dual_mtl':            ('y_preds_AC_cnn_trans_dual_mtl_',         'y_probs_AC_cnn_trans_dual_mtl_',         'classes_AC_cnn_trans_dual_mtl_'),
# Spatial-reconstruction MTL (same input routing as their non-MTL counterparts)
'cnn_gru_dual_cosine_recon_mtl': ('y_preds_AC_cnn_gru_dual_cosine_recon_mtl_', 'y_probs_AC_cnn_gru_dual_cosine_recon_mtl_', 'classes_AC_cnn_gru_dual_cosine_recon_mtl_'),
'cnn_gru_dual_attn_recon_mtl':   ('y_preds_AC_cnn_gru_dual_attn_recon_mtl_',   'y_probs_AC_cnn_gru_dual_attn_recon_mtl_',   'classes_AC_cnn_gru_dual_attn_recon_mtl_'),
```

The factory callable for each MTL key is selected by an `if model_key in MTL_MODEL_KEYS` branch inside the per-fold model construction block — not stored in the dict.

---

### Modified: `03_main_training.py`

1. **New CLI arg — `--mtl`**:
   ```python
   parser.add_argument("--mtl", action="store_true",
       help="Also train MTL models (dual classification+regression heads). "
            "Results are merged into the same classification_performances*.joblib "
            "so standard models do not need to be re-run.")
   ```

   **Note on results file**: MTL results are stored in the **same** `results_file_path` as standard models — the existing `all_ml_results` dict is loaded, MTL model keys are added under the same nested structure, and the file is saved back. No separate file is created. This means `--mtl` can be run after a standard run without touching any existing keys.

2. **Model list gating** — the inline `models` list (lines ~217–235) must exclude MTL keys by default. When `--mtl` is set, MTL model keys are appended:
   ```python
   from model_utils_mtl import MTL_MODEL_KEYS
   # (standard models list as-is, no MTL keys in it)
   models = ["knn", "cnn", "cnn_lf", "gru", "cnn_gru_dual", ...]   # unchanged

   if args.mtl:
       mtl_models = [
           "cnn_mtl", "lstm_mtl", "gru_mtl", "rnn_mtl", "transformer_mtl",
           "cnn_gru_dual_mtl", "cnn_trans_dual_mtl",
           "cnn_gru_dual_cosine_recon_mtl", "cnn_gru_dual_attn_recon_mtl",
       ]
       models = models + mtl_models   # or replace models = mtl_models for MTL-only runs
   ```
   Whether to APPEND (train both standard+MTL together) or REPLACE (train MTL only) is a run-time choice. REPLACE is preferred for dedicated MTL HPC jobs (faster, no redundant retraining of already-saved standard models). APPEND is useful for a combined first-time run.

3. **Load and encode concentration** (after loading the joblib, only when `--mtl`):
   ```python
   from model_utils_mtl import REG_SENTINEL
   y_concentration = None
   if args.mtl:
       raw_conc = training_data.get("concentration", None)
       if raw_conc is not None:
           y_concentration = np.where(np.isnan(raw_conc.astype(float)), REG_SENTINEL, raw_conc.astype(float))
       if y_concentration is None:
           y_concentration = np.full(len(Y_well), REG_SENTINEL, dtype=float)
   ```

4. **Pass to `evaluate_outlier_filters`** — use the EXISTING `models=` parameter; add only the two new MTL params:
   ```python
   evaluate_outlier_filters(
       ...,
       models=models,               # already filtered above
       multitask=args.mtl,
       y_concentration=y_concentration,
   )
   ```

   The caller continues to save via `safe_joblib_dump(all_ml_results, results_file_path, ...)` as before — no change to the save path logic.

---

### Modified: `06_model_prediction_report.py` — MTL-aware reporting

**No new CLI flag needed.** Because MTL results are stored in the same `classification_performances.joblib` alongside standard model keys, the report script loads one file and auto-detects MTL models by presence of `y_reg_preds_{model}_` keys.

**Classification section** (unchanged): accuracy table, confusion matrix, per-fold accuracy curves — exactly as current for all models including MTL classification head.

**Regression section** (new, rendered only when `--mtl` is set): for each MTL model key that has `y_reg_preds_{model}_` present in the results:

1. **Per-fold regression summary table**  
   Columns: model, fold, MAE, RMSE, R², Pearson-r, n_valid (samples where true concentration ≠ sentinel).  
   Aggregated row: mean ± std across folds.

2. **Predicted vs actual scatter** (one subplot per model, aggregated across folds)  
   - x-axis: true concentration (original scale, log₁₀ tick labels)  
   - y-axis: predicted concentration  
   - Identity line (y=x) in dashed black  
   - Points coloured by target label  
   - Title includes R² and MAE

3. **Residual by concentration level** (boxplot)  
   - x-axis: unique concentration values (log-spaced)  
   - y-axis: residual (pred − true) in original scale  
   - One box per concentration bin; median close to zero = unbiased across concentrations  
   - Useful to spot whether the model under/over-predicts specific concentration ranges

4. **Regression performance by target** (grouped bar chart)  
   - x-axis: target label  
   - y-axis: MAE per target  
   - One bar group per model key  
   - Reveals whether some targets are harder to localise by concentration

Metrics computation note: sentinel values (`y_reg_trues == REG_SENTINEL`) must be masked out before computing any regression metric. The count of valid samples per fold is reported as `n_valid`.

---

### Modified: `07_attribution_vis_all.py` — dual-task attribution for MTL models

The XAI pipeline computes `GradientTape` attribution in `extract_xai_artifacts()`. MTL models output `[cls_out, reg_out]` from the same backbone, enabling **two independent saliency maps**: one showing which timesteps drive classification, one showing which drive concentration prediction.

**Detection**: add `is_mtl = 'mtl' in _base_name` check before the existing `is_dual` / `is_lf` checks.

**New `is_mtl` branch in `extract_xai_artifacts`**:
```python
elif is_mtl:
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x_tf_curve)
        cls_out, reg_out = model(x_tf_curve, training=False)
        target_cls = tf.reduce_max(cls_out, axis=1)   # max predicted class score
        target_reg = reg_out[:, 0]                     # scalar regression output

    cls_saliency = np.mean(np.abs(tape.gradient(target_cls, x_tf_curve).numpy()), axis=(0, 2))
    reg_saliency = np.mean(np.abs(tape.gradient(target_reg, x_tf_curve).numpy()), axis=(0, 2))
    del tape

    artifacts[model_name] = {
        "is_type": "mtl",
        "master_saliency": cls_saliency,   # backward-compatible key used by existing plot code
        "cls_saliency": cls_saliency,      # d(cls)/d(input)
        "reg_saliency": reg_saliency,      # d(reg)/d(input)
    }
```

For `cnn_gru_dual_attn_recon_mtl`, `x_tf_curve` is the `(N, k+1, T)` stack. Gradient has shape `(N, k+1, T)` — take `[:, 0, :]` (own curve slice) for both saliency maps; the neighbour slices show how much each neighbour's curves influenced the prediction.

**New visualization function** (add alongside existing `plot_*` functions):
```python
def plot_mtl_saliency_comparison(artifacts, model_names, title):
    """Two-row heatmap: classification saliency (top) vs regression saliency (bottom).
    
    Normalise each map independently so the colour scale represents relative importance
    within each task rather than across tasks (scales differ: cross-entropy gradient vs
    MSE gradient).
    """
```

Renders per-model dual-panel figure:
- Row 1: classification attribution heatmap (which timesteps matter for target prediction)
- Row 2: regression attribution heatmap (which timesteps matter for concentration)
- Optional Row 3: element-wise min(cls, reg) — "shared" attribution used by both tasks

This makes visible whether the model learns task-specific representations (disjoint saliency) or shared representations (overlapping saliency), which directly addresses the research question about whether MTL forces the backbone to generalise.

### Modified: `config.py`

**No `TRAINING_MTL_RESULT_PATH` constant needed** — MTL results go into the same file as standard results.

Add all 9 MTL model keys to `MODEL_KEY_MAP` (lines ~266–301):
```python
# Single-branch MTL
"cnn_mtl":                       ("y_preds_AC_cnn_mtl_",                    "y_probs_AC_cnn_mtl_",                    "classes_AC_cnn_mtl_"),
"lstm_mtl":                      ("y_preds_AC_lstm_mtl_",                   "y_probs_AC_lstm_mtl_",                   "classes_AC_lstm_mtl_"),
"gru_mtl":                       ("y_preds_AC_gru_mtl_",                    "y_probs_AC_gru_mtl_",                    "classes_AC_gru_mtl_"),
"rnn_mtl":                       ("y_preds_AC_rnn_mtl_",                    "y_probs_AC_rnn_mtl_",                    "classes_AC_rnn_mtl_"),
"transformer_mtl":               ("y_preds_AC_trans_mtl_",                  "y_probs_AC_trans_mtl_",                  "classes_AC_trans_mtl_"),
# Dual-branch MTL
"cnn_gru_dual_mtl":              ("y_preds_AC_cnn_gru_dual_mtl_",           "y_probs_AC_cnn_gru_dual_mtl_",           "classes_AC_cnn_gru_dual_mtl_"),
"cnn_trans_dual_mtl":            ("y_preds_AC_cnn_trans_dual_mtl_",         "y_probs_AC_cnn_trans_dual_mtl_",         "classes_AC_cnn_trans_dual_mtl_"),
# Spatial-reconstruction MTL
"cnn_gru_dual_cosine_recon_mtl": ("y_preds_AC_cnn_gru_dual_cosine_recon_mtl_", "y_probs_AC_cnn_gru_dual_cosine_recon_mtl_", "classes_AC_cnn_gru_dual_cosine_recon_mtl_"),
"cnn_gru_dual_attn_recon_mtl":   ("y_preds_AC_cnn_gru_dual_attn_recon_mtl_",   "y_probs_AC_cnn_gru_dual_attn_recon_mtl_",   "classes_AC_cnn_gru_dual_attn_recon_mtl_"),
```

MTL keys go into `_NO_INC` to prevent spurious `_inc` twin generation. **Do NOT import `MTL_MODEL_KEYS` from `model_utils_mtl` in `config.py`** — that would transitively import TensorFlow at `import config` time. Instead, hardcode the list as a plain set:
```python
_MTL_MODEL_KEYS = {
    "cnn_mtl", "lstm_mtl", "gru_mtl", "rnn_mtl", "transformer_mtl",
    "cnn_gru_dual_mtl", "cnn_trans_dual_mtl",
    "cnn_gru_dual_cosine_recon_mtl", "cnn_gru_dual_attn_recon_mtl",
}
_NO_INC = {"rf", "knn", "ffi", "gnn_gat", "gnn_gcn", "cnn_gru_dual_attn_recon"} | _MTL_MODEL_KEYS
```

### No changes needed

- `01b_lab_curve_preprocessing.py` — `concentration` key already written
- `adhoc_patch_concentration.py` — already complete
- `02_outlier_detection_pipeline.py` — unrelated to MTL

---

## Data Flow Summary

```
curve_for_training.joblib
  ├── "curves"        → X_full (N, T)
  ├── "well_labels"   → y_full (N,)    [classification target]
  └── "concentration" → float array (N,) or None

03_main_training.py
  → encode: None → REG_SENTINEL (-1.0)
  → pass y_concentration to evaluate_outlier_filters

evaluate_outlier_filters (per fold, per outlier filter):
  train_idx, val_idx
  → StandardScaler on conc_train (non-sentinel only)
  → y_train_dict = {'cls_out': y_cls_train, 'reg_out': conc_scaled_train}
  → MTLModel.fit(X_train, y_train_dict, validation_data=(X_val, y_val_dict))
  → cls_probs, reg_preds = model.predict(X_val)
  → inverse_transform reg_preds using fold scaler
  → store: y_reg_preds_{model}_, y_reg_trues_{model}_
```

---

## Verification

1. **Unit check** — create and call a single MTL model:
   ```python
   from model_utils_mtl import create_cnn_gru_dual_mtl_model
   m = create_cnn_gru_dual_mtl_model(T=200, n_classes=5)
   m.compile(optimizer='adam', metrics=['accuracy'])
   m.summary()   # should show cls_out (5,) and reg_out (1,)
   ```

2. **Training smoke test** — run one experiment with `--mtl`:
   ```bash
   python 03_main_training.py \
     --task_id 0 --exp_folder /vol/bitbucket/gk225/POC_DDM_datasets/LAB_DDM_paper \
     --mtl --force_rerun
   ```
   Confirm: (a) `log_var_cls`/`log_var_reg` appear in `model.trainable_variables`, (b) training loss prints `loss` + `reg_mse`, (c) `val_accuracy` tracked normally by EarlyStopping.

3. **Results check** — after training, inspect saved results joblib:
   `y_reg_preds_cnn_gru_dual_mtl_` keys present; predictions are in original concentration scale.

4. **Graceful fallback** — run on a folder with `concentration=None` (e.g., `Z_area`):
   MTL models train without error; `L_reg_masked ≈ 0`; classification accuracy comparable to non-MTL baseline.

5. **`06_model_prediction_report.py`** — run on the MTL-trained results; confirm no crash from unknown `y_reg_*` keys.
