# Staged SupCon Training (2-Stage ST)

**Flag:** `--supcon_staged`
**Scope:** CNN+GRU dual only, ST only (SC1 / SC2 / SC3)
**Date:** 2026-07-15
**Updated:** 2026-07-15 (post-revalidation)

---

## 1. Background & Theory

### Joint SupCon (current)

Current ST SupCon mixes CE and SupCon losses from epoch 1:

| Variant | Loss formula |
|---------|-------------|
| SC1 | `(1 − λ)·CE + λ·SC_fused`  (λ = 0.2) |
| SC2 | `(1 − 2λ)·CE + λ·(SC_cnn + SC_seq)`  (λ = 0.1) |
| SC3 | `(1 − 3λ)·CE + λ·(SC_cnn + SC_seq + SC_fused)`  (λ = 0.1) |

The two objectives can conflict early in training: CE exploits whatever shortcut features exist immediately; SupCon pushes for compact, class-separated clusters in projection space. When both compete from epoch 1, neither gets clean gradients to fully develop. The λ coefficients are also a dataset-specific hyperparameter.

### 2-Stage Staged SupCon (proposed)

More closely follows the original SupCon paper (Khosla et al. 2020), which treats SC as a **pretraining** objective, not a joint-training regulariser.

**Stage 1 — Representation learning (SC only):**
Train backbone + projection head(s) with pure SupCon loss. No classification pressure. The backbone organises its representation space purely around class-compact structure. Training continues until `val_loss` (= SC loss in Stage 1) plateaus, or for a fixed epoch count via `--cl_phase1_epochs`.

**Stage 2 — Linear classification (CE only, frozen backbone):**
Freeze backbone and projection heads. Train only `cls_feat` + `cls_out` with CE loss. With the feature space fixed, this is a near-convex problem — converges fast and is less prone to overfitting than joint training.

**Expected benefits:**
- Cleaner SupCon gradient → better-structured representation space
- CE in Stage 2 gets a good fixed feature space → less sensitive to initialisation
- No λ to tune: the two objectives are fully decoupled
- Directly benchmarks the quality of the learned representation (analogous to linear probe evaluation)

**Trade-off:** Wall-clock time is longer (two full training phases). For datasets where SC loss plateaus quickly this is offset. For very small datasets Stage 1 might converge before the backbone saturates.

---

## 2. What to Change

### New flag

`--supcon_staged` (store_true) — mutually exclusive with `--mtl`, requires `--supcon ∈ {1,2,3}`.

**Name rationale:** `--supcon_staged` pairs naturally with `--supcon N` (which selects SC type) and mirrors how `--mtl_cl` adds staged behaviour on top of `--mtl`. The `_staged` suffix avoids the awkward `_2stages` and aligns with "staged training" terminology in the literature.

### New model keys (9 total)

| SC type | Keys |
|---------|------|
| SC1 | `cnn_gru_dual_supcon_staged`, `cnn_gru_dual_cosine_recon_supcon_staged`, `cnn_gru_dual_attn_recon_supcon_staged` |
| SC2 | `cnn_gru_dual_supcon2_staged`, `cnn_gru_dual_cosine_recon_supcon2_staged`, `cnn_gru_dual_attn_recon_supcon2_staged` |
| SC3 | `cnn_gru_dual_supcon3_staged`, `cnn_gru_dual_cosine_recon_supcon3_staged`, `cnn_gru_dual_attn_recon_supcon3_staged` |

### New Keras model classes (3)

`StagedSupConSTModel`, `StagedBranch2STModel`, `StagedBranch3STModel` — inherit from the corresponding existing ST classes (`SupConModel`, `SupConBranch2STModel`, `SupConBranch3STModel`). They add:
- A `curriculum_phase` `tf.Variable(0, trainable=False)` — same mechanism as CL models
- A modified `train_step` **and `test_step`** that branch on phase (0 = SC only, 1 = CE only)

`test_step` must also branch so that `val_loss` during Stage 1 = pure SC loss (not joint CE+SC). Without this, `AutoPhaseTransitionSTCallback` and `EarlyStopping` see a mixed signal in Stage 1.

### New freeze function

`_freeze_backbone_staged_supcon(model)` — uses `_STAGED_HEAD_LAYERS = frozenset({'cls_feat', 'cls_out'})` (tighter than `_freeze_backbone` which also keeps `reg_feat`/`reg_out`). Projection layers (`proj`, `cnn_proj`, `seq_proj`, `fused_proj`, etc.) are also frozen in Stage 2.

Must be compiled with `jit_compile=False` and `metrics=['accuracy']` (see considerations below).

### New callbacks (2)

**`FixedPhaseTransitionSTCallback`** — fires at a fixed epoch (like the existing `FixedPhaseTransitionCallback` in `model_utils_mtl.py` but calling `_freeze_backbone_staged_supcon` instead of `_freeze_backbone`). Existing class cannot be reused since it hardcodes the MTL freeze function and does not accept a `freeze_fn` parameter.

**`AutoPhaseTransitionSTCallback`** — monitors `val_loss` plateau and calls `_freeze_backbone_staged_supcon`. Accepts `early_stop_cb` and `rlrp_cb` explicitly (same as `AutoPhaseTransitionCallback` in `model_utils_mtl.py`), and resets them directly on transition — `self.params.get('callbacks', [])` does not expose other callbacks in Keras.

Both callbacks live in `model_utils_supcon.py`.

### New factory functions (6, + 3 wrap helpers)

Six dedicated factory functions (canonical + attn_recon per SC type). `cosine_recon` variants share the same factory as the canonical variant — this is identical to the existing pattern (confirmed: `cnn_gru_dual_cosine_recon_supcon` already reuses `create_cnn_gru_dual_supcon_model`; there is no separate `cosine_recon` factory). `attn_recon` needs a dedicated factory since its input shape is `(k+1, T)` not `(T, 1)`.

---

## 3. What to Consider

### `test_step` must branch on `curriculum_phase`
Without overriding `test_step`, `val_loss` in Stage 1 = joint CE+SC (parent class default). The plateau callback monitors `val_loss`, so it would see a mixed signal. `EarlyStopping` could also fire prematurely if CE dominates early. Override `test_step` in all 3 staged classes to match `train_step` branching.

### `jit_compile=False` required
`train_step` branches on `self.curriculum_phase` using a Python `if`. TF traces `train_step` once and caches the graph. After `make_train_function(force=True)`, a retrace occurs — at retrace time the Python `if` sees the new value of `curriculum_phase`. This retrace mechanism requires `jit_compile=False` (same pattern as `_freeze_backbone` in `model_utils_mtl.py`). Both the initial compile and the re-compile inside `_freeze_backbone_staged_supcon` must use `jit_compile=False`.

### `metrics=['accuracy']` must survive re-compile
`_freeze_backbone_staged_supcon` recompiles the model. If `metrics` is omitted, `compiled_metrics` is reset to empty — `self.compiled_metrics.update_state(...)` in `train_step`/`test_step` becomes a no-op and `val_accuracy` disappears. Always include `metrics=['accuracy']` in the re-compile call.

### `val_split_ok` fallback
When no validation split is available, auto-transition cannot fire. Guard: if `not _val_split_ok` and `cl_phase1_epochs` is None, raise a clear `ValueError`. The existing `--cl_phase1_epochs` arg (currently used for CL-MTL) is reused unchanged — it is already a named parameter of `evaluate_outlier_filters`.

### Single `fit()` call
Use a single `model.fit()` call with the transition callback. The callback freezes the backbone mid-training via `make_train_function(force=True)` — the same mechanism CL-MTL already uses.

### Dedicated `_st_es` / `_st_rlrp` (do not reuse `_fit_callbacks`)
The shared `_fit_callbacks` list does not hold direct references accessible at callback construction time. Create dedicated `_st_es` and `_st_rlrp` in the dispatch block and pass them to the transition callback — matching the CL-MTL pattern with `_cl_es` / `_cl_rlrp`.

### `attn_recon` dispatch uses shape[1], shape[2]
`attn_recon` factory signature is `(k_plus_1, T, n_classes)`, not `(T, n_classes)`. In the dispatch block, detect `'attn_recon' in _base_m` and call with `X_train_curve.shape[1]`, `X_train_curve.shape[2]`. This is identical to the existing `cnn_gru_dual_attn_recon_supcon` dispatch pattern.

### `_is_staged` in `force_rerun`'s `_is_standard`
Without this, a standard-mode run (no `--supcon_staged`) would delete staged model results when clearing. Follow the `_is_lc` pattern: build `_staged_result_keys` (full universe of staged keys), add `_is_staged = _rk in _staged_result_keys`, extend `_is_standard = (...and not _is_staged...)`.

### `_NO_INC` membership
All 9 staged keys must be in `_NO_INC` — no inception smoothing for SupCon variants.

### Incompatibility guards
- `--supcon_staged + --mtl` → sys.exit (staged is ST-only)
- `--supcon_staged + --supcon 0` → sys.exit (SC0 has no projection head)

---

## 4. Technical Changes

### A. `utils/model_training/model_utils_supcon.py`

**Key-set constants** (add after `ALL_LC_KEYS`):
```python
STAGED_SUPCON_MODEL_KEYS = [
    'cnn_gru_dual_supcon_staged',
    'cnn_gru_dual_cosine_recon_supcon_staged',
    'cnn_gru_dual_attn_recon_supcon_staged',
]
STAGED_BRANCH_SUPCON2_MODEL_KEYS = [
    'cnn_gru_dual_supcon2_staged',
    'cnn_gru_dual_cosine_recon_supcon2_staged',
    'cnn_gru_dual_attn_recon_supcon2_staged',
]
STAGED_BRANCH_SUPCON3_MODEL_KEYS = [
    'cnn_gru_dual_supcon3_staged',
    'cnn_gru_dual_cosine_recon_supcon3_staged',
    'cnn_gru_dual_attn_recon_supcon3_staged',
]
ALL_STAGED_SUPCON_KEYS = (STAGED_SUPCON_MODEL_KEYS
                          + STAGED_BRANCH_SUPCON2_MODEL_KEYS
                          + STAGED_BRANCH_SUPCON3_MODEL_KEYS)
```

**Freeze function** (`jit_compile=False` and `metrics=['accuracy']` are required):
```python
_STAGED_HEAD_LAYERS = frozenset({'cls_feat', 'cls_out'})

def _freeze_backbone_staged_supcon(model):
    for layer in model.layers:
        layer.trainable = layer.name in _STAGED_HEAD_LAYERS
    model.curriculum_phase.assign(1)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
                  metrics=['accuracy'], jit_compile=False)
    model.make_train_function(force=True)
```

**`FixedPhaseTransitionSTCallback`** (mirrors existing `FixedPhaseTransitionCallback` but calls staged freeze):
```python
class FixedPhaseTransitionSTCallback(tf.keras.callbacks.Callback):
    """Fixed-epoch Stage 1→2 transition for staged SupCon (ST only)."""
    def __init__(self, phase1_epochs, early_stop_cb=None, rlrp_cb=None):
        super().__init__()
        self.phase1_epochs = phase1_epochs
        self.early_stop_cb = early_stop_cb
        self.rlrp_cb = rlrp_cb
        self._transitioned = False

    def on_epoch_begin(self, epoch, logs=None):
        if not self._transitioned and epoch == self.phase1_epochs:
            self._transitioned = True
            _freeze_backbone_staged_supcon(self.model)
            if self.early_stop_cb is not None:
                self.early_stop_cb.best = float('inf')
            if self.rlrp_cb is not None:
                self.rlrp_cb.best = float('inf')
            print(f'\n[StagedSupCon] Phase 1→2 fixed at epoch {epoch}')
```

**`AutoPhaseTransitionSTCallback`** (explicit callback refs, not `self.params.get('callbacks')`):
```python
class AutoPhaseTransitionSTCallback(tf.keras.callbacks.Callback):
    """Auto plateau-detection Stage 1→2 transition for staged SupCon (ST only)."""
    def __init__(self, min_phase1_epochs=30, patience=15,
                 early_stop_cb=None, rlrp_cb=None):
        super().__init__()
        self.min_phase1_epochs = min_phase1_epochs
        self.patience = patience
        self.early_stop_cb = early_stop_cb
        self.rlrp_cb = rlrp_cb
        self._best = np.inf
        self._wait = 0
        self._transitioned = False

    def on_epoch_end(self, epoch, logs=None):
        if self._transitioned or epoch < self.min_phase1_epochs:
            return
        val_loss = (logs or {}).get('val_loss', np.inf)
        if val_loss < self._best - 1e-4:
            self._best = val_loss
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                print(f'\n[StagedSupCon] Phase 1→2 at epoch {epoch+1} '
                      f'(val_loss={val_loss:.4f}); freezing backbone.')
                _freeze_backbone_staged_supcon(self.model)
                if self.early_stop_cb is not None:
                    self.early_stop_cb.best = np.inf
                    self.early_stop_cb.wait = 0
                if self.rlrp_cb is not None:
                    self.rlrp_cb.best = np.inf
                    self.rlrp_cb.wait = 0
                self._transitioned = True
```

**Model classes** — all 3 override both `train_step` and `test_step`:

SC1 (2 outputs: `[cls_out, proj_norm]`):
```python
@tf.keras.utils.register_keras_serializable(package='staged_supcon')
class StagedSupConSTModel(SupConModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.curriculum_phase = tf.Variable(0, trainable=False, dtype=tf.int32)

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        y_cls = y_dict['cls_out']
        with tf.GradientTape() as tape:
            cls_out, proj_norm = self(x, training=True)
            sc = supcon_loss(proj_norm, y_cls, self.supcon_temp)
            ce = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(y_cls, cls_out))
            loss = tf.cond(tf.equal(self.curriculum_phase, 0), lambda: sc, lambda: ce)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y_cls, cls_out)
        return {m.name: m.result() for m in self.metrics} | {'loss': loss}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        y_cls = y_dict['cls_out']
        cls_out, proj_norm = self(x, training=False)
        sc = supcon_loss(proj_norm, y_cls, self.supcon_temp)
        ce = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(y_cls, cls_out))
        loss = tf.cond(tf.equal(self.curriculum_phase, 0), lambda: sc, lambda: ce)
        self.compiled_metrics.update_state(y_cls, cls_out)
        return {m.name: m.result() for m in self.metrics} | {'loss': loss}
```

SC2 (3 outputs: `[cls_out, cnn_proj, seq_proj]`):
```python
@tf.keras.utils.register_keras_serializable(package='staged_supcon2')
class StagedBranch2STModel(SupConBranch2STModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.curriculum_phase = tf.Variable(0, trainable=False, dtype=tf.int32)

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        y_cls = y_dict['cls_out']
        with tf.GradientTape() as tape:
            cls_out, cnn_proj, seq_proj = self(x, training=True)
            sc = (supcon_loss(cnn_proj, y_cls, self.supcon_temp)
                  + supcon_loss(seq_proj, y_cls, self.supcon_temp))
            ce = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(y_cls, cls_out))
            loss = tf.cond(tf.equal(self.curriculum_phase, 0), lambda: sc, lambda: ce)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y_cls, cls_out)
        return {m.name: m.result() for m in self.metrics} | {'loss': loss}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        y_cls = y_dict['cls_out']
        cls_out, cnn_proj, seq_proj = self(x, training=False)
        sc = (supcon_loss(cnn_proj, y_cls, self.supcon_temp)
              + supcon_loss(seq_proj, y_cls, self.supcon_temp))
        ce = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(y_cls, cls_out))
        loss = tf.cond(tf.equal(self.curriculum_phase, 0), lambda: sc, lambda: ce)
        self.compiled_metrics.update_state(y_cls, cls_out)
        return {m.name: m.result() for m in self.metrics} | {'loss': loss}
```

SC3 (4 outputs: `[cls_out, cnn_proj, seq_proj, fused_proj]`):
```python
@tf.keras.utils.register_keras_serializable(package='staged_supcon3')
class StagedBranch3STModel(SupConBranch3STModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.curriculum_phase = tf.Variable(0, trainable=False, dtype=tf.int32)

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        y_cls = y_dict['cls_out']
        with tf.GradientTape() as tape:
            cls_out, cnn_proj, seq_proj, fused_proj = self(x, training=True)
            sc = (supcon_loss(cnn_proj,   y_cls, self.supcon_temp)
                  + supcon_loss(seq_proj,   y_cls, self.supcon_temp)
                  + supcon_loss(fused_proj, y_cls, self.supcon_temp))
            ce = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(y_cls, cls_out))
            loss = tf.cond(tf.equal(self.curriculum_phase, 0), lambda: sc, lambda: ce)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y_cls, cls_out)
        return {m.name: m.result() for m in self.metrics} | {'loss': loss}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        y_cls = y_dict['cls_out']
        cls_out, cnn_proj, seq_proj, fused_proj = self(x, training=False)
        sc = (supcon_loss(cnn_proj,   y_cls, self.supcon_temp)
              + supcon_loss(seq_proj,   y_cls, self.supcon_temp)
              + supcon_loss(fused_proj, y_cls, self.supcon_temp))
        ce = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(y_cls, cls_out))
        loss = tf.cond(tf.equal(self.curriculum_phase, 0), lambda: sc, lambda: ce)
        self.compiled_metrics.update_state(y_cls, cls_out)
        return {m.name: m.result() for m in self.metrics} | {'loss': loss}
```

**Wrap helpers** (add after `_branch3_supcon_wrap`):
```python
def _staged_supcon_wrap(inputs, embedding, n_classes):
    """Same head as _supcon_wrap but returns StagedSupConSTModel."""
    cls_feat  = tf.keras.layers.Dense(16, activation='relu',  name='cls_feat')(embedding)
    cls_out   = tf.keras.layers.Dense(n_classes, activation='softmax', name='cls_out')(cls_feat)
    proj      = tf.keras.layers.Dense(64, activation='relu',  name='proj_hidden')(embedding)
    proj_norm = tf.keras.layers.UnitNormalization(axis=1, name='proj')(proj)
    return StagedSupConSTModel(inputs=inputs, outputs=[cls_out, proj_norm])

def _staged_branch2_wrap(inputs, cnn_emb, seq_emb, fused, n_classes):
    """Same head as _branch2_supcon_wrap but returns StagedBranch2STModel."""
    cls_feat = tf.keras.layers.Dense(16, activation='relu', name='cls_feat')(fused)
    cls_out  = tf.keras.layers.Dense(n_classes, activation='softmax', name='cls_out')(cls_feat)
    cnn_proj = _proj_head(cnn_emb, 'cnn')
    seq_proj = _proj_head(seq_emb, 'seq')
    return StagedBranch2STModel(inputs=inputs, outputs=[cls_out, cnn_proj, seq_proj])

def _staged_branch3_wrap(inputs, cnn_emb, seq_emb, fused, n_classes):
    """Same head as _branch3_supcon_wrap but returns StagedBranch3STModel."""
    cls_feat   = tf.keras.layers.Dense(16, activation='relu', name='cls_feat')(fused)
    cls_out    = tf.keras.layers.Dense(n_classes, activation='softmax', name='cls_out')(cls_feat)
    cnn_proj   = _proj_head(cnn_emb, 'cnn')
    seq_proj   = _proj_head(seq_emb, 'seq')
    fused_proj = _proj_head(fused,   'fused')
    return StagedBranch3STModel(inputs=inputs, outputs=[cls_out, cnn_proj, seq_proj, fused_proj])
```

**Factory functions** (6 — add after spatial-reconstruction+branch factories):
```python
# cosine_recon variants share the same factory as the canonical variant
# (different input data, identical architecture) — same as existing supcon pattern.

def create_cnn_gru_dual_supcon_staged_model(T, n_classes):
    inputs = tf.keras.layers.Input(shape=(T, 1))
    return _staged_supcon_wrap(inputs, _build_cnn_gru_dual_branches_mtl(inputs), n_classes)

def create_cnn_gru_dual_attn_recon_supcon_staged_model(k_plus_1, T, n_classes, attn_dim=16):
    stack_input = tf.keras.layers.Input(shape=(k_plus_1, T), name='neighbor_stack_input')
    return _staged_supcon_wrap(
        stack_input, _build_cnn_gru_dual_attn_recon_embedding_mtl(stack_input, T, attn_dim), n_classes)

def create_cnn_gru_dual_supcon2_staged_model(T, n_classes):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    cnn_emb, gru_emb, fused = _build_cnn_gru_dual_branches_mtl(inputs, return_branches=True)
    return _staged_branch2_wrap(inputs, cnn_emb, gru_emb, fused, n_classes)

def create_cnn_gru_dual_attn_recon_supcon2_staged_model(k_plus_1, T, n_classes, attn_dim=16):
    stack_input = tf.keras.layers.Input(shape=(k_plus_1, T), name='neighbor_stack_input')
    cnn_emb, gru_emb, fused = _build_cnn_gru_dual_attn_recon_embedding_mtl(
        stack_input, T, attn_dim, return_branches=True)
    return _staged_branch2_wrap(stack_input, cnn_emb, gru_emb, fused, n_classes)

def create_cnn_gru_dual_supcon3_staged_model(T, n_classes):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    cnn_emb, gru_emb, fused = _build_cnn_gru_dual_branches_mtl(inputs, return_branches=True)
    return _staged_branch3_wrap(inputs, cnn_emb, gru_emb, fused, n_classes)

def create_cnn_gru_dual_attn_recon_supcon3_staged_model(k_plus_1, T, n_classes, attn_dim=16):
    stack_input = tf.keras.layers.Input(shape=(k_plus_1, T), name='neighbor_stack_input')
    cnn_emb, gru_emb, fused = _build_cnn_gru_dual_attn_recon_embedding_mtl(
        stack_input, T, attn_dim, return_branches=True)
    return _staged_branch3_wrap(stack_input, cnn_emb, gru_emb, fused, n_classes)
```

---

### B. `utils/model_training/model_utils.py`

**Extend imports** (add to existing `from model_utils_supcon import (...)` block):
```python
    create_cnn_gru_dual_supcon_staged_model,
    create_cnn_gru_dual_attn_recon_supcon_staged_model,
    create_cnn_gru_dual_supcon2_staged_model,
    create_cnn_gru_dual_attn_recon_supcon2_staged_model,
    create_cnn_gru_dual_supcon3_staged_model,
    create_cnn_gru_dual_attn_recon_supcon3_staged_model,
    STAGED_SUPCON_MODEL_KEYS, STAGED_BRANCH_SUPCON2_MODEL_KEYS,
    STAGED_BRANCH_SUPCON3_MODEL_KEYS, ALL_STAGED_SUPCON_KEYS,
    _freeze_backbone_staged_supcon,
    AutoPhaseTransitionSTCallback, FixedPhaseTransitionSTCallback,
```

**`_XAI_SAVE_NAME`** (add after LC/CL entries):
```python
_XAI_SAVE_NAME.update({k: k for k in ALL_STAGED_SUPCON_KEYS})
```

**Dispatch block** (add after Branch SC3 ST block, before CL-MTL block):
```python
elif _base_m in ALL_STAGED_SUPCON_KEYS:
    tf.keras.backend.clear_session()
    epochs = 500
    if _base_m in STAGED_SUPCON_MODEL_KEYS:
        if 'attn_recon' in _base_m:
            model = create_cnn_gru_dual_attn_recon_supcon_staged_model(
                X_train_curve.shape[1], X_train_curve.shape[2], n_classes)
        else:  # canonical + cosine_recon share the same factory
            model = create_cnn_gru_dual_supcon_staged_model(X_train_curve.shape[1], n_classes)
    elif _base_m in STAGED_BRANCH_SUPCON2_MODEL_KEYS:
        if 'attn_recon' in _base_m:
            model = create_cnn_gru_dual_attn_recon_supcon2_staged_model(
                X_train_curve.shape[1], X_train_curve.shape[2], n_classes)
        else:
            model = create_cnn_gru_dual_supcon2_staged_model(X_train_curve.shape[1], n_classes)
    elif _base_m in STAGED_BRANCH_SUPCON3_MODEL_KEYS:
        if 'attn_recon' in _base_m:
            model = create_cnn_gru_dual_attn_recon_supcon3_staged_model(
                X_train_curve.shape[1], X_train_curve.shape[2], n_classes)
        else:
            model = create_cnn_gru_dual_supcon3_staged_model(X_train_curve.shape[1], n_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
                  metrics=['accuracy'], jit_compile=False)
    if _val_split_ok:
        _st_es   = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=100, restore_best_weights=True)
        _st_rlrp = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=30, min_lr=1e-5)
        _st_phase_cb = (
            FixedPhaseTransitionSTCallback(cl_phase1_epochs,
                                           early_stop_cb=_st_es, rlrp_cb=_st_rlrp)
            if cl_phase1_epochs else
            AutoPhaseTransitionSTCallback(min_phase1_epochs=30, patience=15,
                                          early_stop_cb=_st_es, rlrp_cb=_st_rlrp))
        model.fit(X_train_curve_fit, {'cls_out': y_train_fit},
                  validation_data=(X_val_curve, {'cls_out': y_val}),
                  epochs=epochs, batch_size=512, shuffle=True, verbose=0,
                  callbacks=[_st_es, _st_rlrp, _st_phase_cb])
    else:
        if not cl_phase1_epochs:
            raise ValueError(
                f'[{_base_m}] No val split. Set --cl_phase1_epochs for Stage 1 length.')
        _st_phase_cb = FixedPhaseTransitionSTCallback(cl_phase1_epochs)
        model.fit(X_train_curve, {'cls_out': y_train},
                  epochs=epochs, batch_size=512, shuffle=True, verbose=0,
                  callbacks=[_st_phase_cb])
    if _do_xai_save:
        _xai_path = Path(save_model_dir) / f"{m}_{f}_{save_model_curve_type}_model.keras"
        safe_keras_save(model, _xai_path)
        print(f"     [XAI] Saved {m} -> {_xai_path}")
    raw_out  = model.predict(X_test_curve, verbose=0)
    cls_prob = raw_out[0] if isinstance(raw_out, (list, tuple)) else raw_out
    pred = np.argmax(cls_prob, axis=1)
    cls  = np.unique(y_encoded)
    preds.append(pred); probs.append(cls_prob); classes_list.append(cls)
    tf.keras.backend.clear_session()
```

Note: `cl_phase1_epochs` is already a named parameter of `evaluate_outlier_filters` — no new wiring needed.

---

### C. `config.py`

**`MODEL_KEY_MAP`** — 9 new 3-tuple entries (add after LC entries):
```python
# Staged SupCon ST — SC1
"cnn_gru_dual_supcon_staged":               ("y_preds_AC_cnn_gru_dual_supcon_staged_",               "y_probs_AC_cnn_gru_dual_supcon_staged_",               "classes_AC_cnn_gru_dual_supcon_staged_"),
"cnn_gru_dual_cosine_recon_supcon_staged":  ("y_preds_AC_cnn_gru_dual_cosine_recon_supcon_staged_",  "y_probs_AC_cnn_gru_dual_cosine_recon_supcon_staged_",  "classes_AC_cnn_gru_dual_cosine_recon_supcon_staged_"),
"cnn_gru_dual_attn_recon_supcon_staged":    ("y_preds_AC_cnn_gru_dual_attn_recon_supcon_staged_",    "y_probs_AC_cnn_gru_dual_attn_recon_supcon_staged_",    "classes_AC_cnn_gru_dual_attn_recon_supcon_staged_"),
# SC2 staged
"cnn_gru_dual_supcon2_staged":              ("y_preds_AC_cnn_gru_dual_supcon2_staged_",              "y_probs_AC_cnn_gru_dual_supcon2_staged_",              "classes_AC_cnn_gru_dual_supcon2_staged_"),
"cnn_gru_dual_cosine_recon_supcon2_staged": ("y_preds_AC_cnn_gru_dual_cosine_recon_supcon2_staged_", "y_probs_AC_cnn_gru_dual_cosine_recon_supcon2_staged_", "classes_AC_cnn_gru_dual_cosine_recon_supcon2_staged_"),
"cnn_gru_dual_attn_recon_supcon2_staged":   ("y_preds_AC_cnn_gru_dual_attn_recon_supcon2_staged_",   "y_probs_AC_cnn_gru_dual_attn_recon_supcon2_staged_",   "classes_AC_cnn_gru_dual_attn_recon_supcon2_staged_"),
# SC3 staged
"cnn_gru_dual_supcon3_staged":              ("y_preds_AC_cnn_gru_dual_supcon3_staged_",              "y_probs_AC_cnn_gru_dual_supcon3_staged_",              "classes_AC_cnn_gru_dual_supcon3_staged_"),
"cnn_gru_dual_cosine_recon_supcon3_staged": ("y_preds_AC_cnn_gru_dual_cosine_recon_supcon3_staged_", "y_probs_AC_cnn_gru_dual_cosine_recon_supcon3_staged_", "classes_AC_cnn_gru_dual_cosine_recon_supcon3_staged_"),
"cnn_gru_dual_attn_recon_supcon3_staged":   ("y_preds_AC_cnn_gru_dual_attn_recon_supcon3_staged_",   "y_probs_AC_cnn_gru_dual_attn_recon_supcon3_staged_",   "classes_AC_cnn_gru_dual_attn_recon_supcon3_staged_"),
```

**`_STAGED_SUPCON_MODEL_KEYS`** set + extend `_NO_INC`** (add after `_LC_MODEL_KEYS` ~line 465):
```python
_STAGED_SUPCON_MODEL_KEYS = {
    "cnn_gru_dual_supcon_staged",
    "cnn_gru_dual_cosine_recon_supcon_staged",
    "cnn_gru_dual_attn_recon_supcon_staged",
    "cnn_gru_dual_supcon2_staged",
    "cnn_gru_dual_cosine_recon_supcon2_staged",
    "cnn_gru_dual_attn_recon_supcon2_staged",
    "cnn_gru_dual_supcon3_staged",
    "cnn_gru_dual_cosine_recon_supcon3_staged",
    "cnn_gru_dual_attn_recon_supcon3_staged",
}

_NO_INC = ({"rf", "knn", "ffi", "gnn_gat", "gnn_gcn", "cnn_gru_dual_attn_recon"}
           | _MTL_MODEL_KEYS | _SUPCON_MODEL_KEYS | _BRANCH_SUPCON_MODEL_KEYS
           | _CL_MTL_MODEL_KEYS | _RCFD_MODEL_KEYS | _LC_MODEL_KEYS
           | _STAGED_SUPCON_MODEL_KEYS)
```

Note: `config.py` defines all its sets as inline literals (no imports from `model_utils_supcon`). The staged set must be defined the same way — do NOT add an import from `model_utils_supcon` here.

**`MODEL_PRINT_MAP`** — 9 entries (after LC print entries):
```python
"cnn_gru_dual_supcon_staged":               "CNN+GRU SC1 Staged",
"cnn_gru_dual_cosine_recon_supcon_staged":  "CNN+GRU CosRecon SC1 Staged",
"cnn_gru_dual_attn_recon_supcon_staged":    "CNN+GRU AttnRecon SC1 Staged",
"cnn_gru_dual_supcon2_staged":              "CNN+GRU SC2 Staged",
"cnn_gru_dual_cosine_recon_supcon2_staged": "CNN+GRU CosRecon SC2 Staged",
"cnn_gru_dual_attn_recon_supcon2_staged":   "CNN+GRU AttnRecon SC2 Staged",
"cnn_gru_dual_supcon3_staged":              "CNN+GRU SC3 Staged",
"cnn_gru_dual_cosine_recon_supcon3_staged": "CNN+GRU CosRecon SC3 Staged",
"cnn_gru_dual_attn_recon_supcon3_staged":   "CNN+GRU AttnRecon SC3 Staged",
```

---

### D. `03_main_training.py`

**New argument** (add after `--lbl_conc`):
```python
parser.add_argument('--supcon_staged', action='store_true',
    help='2-stage ST SupCon: Stage 1 SC-only until plateau; Stage 2 CE-only frozen backbone.')
```

**Validation** (add after LC guard):
```python
if getattr(args, 'supcon_staged', False):
    if args.mtl:
        sys.exit('[!] --supcon_staged is ST-only; cannot combine with --mtl.')
    if args.supcon == 0:
        sys.exit('[!] --supcon_staged requires --supcon 1, 2, or 3.')
```

**`_mode` banner** — append `' Staged'` suffix when `args.supcon_staged`.

**Imports** — extend the existing bare import block at lines 15-21 (same style as existing):
```python
from model_utils_supcon import (SUPCON_MODEL_KEYS, ...,  # existing entries
                                ALL_LC_KEYS,
                                STAGED_SUPCON_MODEL_KEYS, STAGED_BRANCH_SUPCON2_MODEL_KEYS,
                                STAGED_BRANCH_SUPCON3_MODEL_KEYS, ALL_STAGED_SUPCON_KEYS)
```

**Model selection** (add as the first condition inside the ST `else:` block, before the existing `if args.supcon == 1:` line ~541):
```python
if getattr(args, 'supcon_staged', False):
    if args.supcon == 1:
        models = STAGED_SUPCON_MODEL_KEYS
    elif args.supcon == 2:
        models = STAGED_BRANCH_SUPCON2_MODEL_KEYS
    elif args.supcon == 3:
        models = STAGED_BRANCH_SUPCON3_MODEL_KEYS
elif args.supcon == 1:           # existing — unchanged
    models = ['cnn_gru_dual_supcon', ...]
...
```

**`force_rerun` — build staged key sets** (add after `_lc_result_keys` block):
```python
_staged_result_keys = set()
_staged_sc_result_keys = set()
for _k, (_pk, _probk, _clsk) in config.MODEL_KEY_MAP.items():
    if _k not in config._STAGED_SUPCON_MODEL_KEYS:
        continue
    _staged_result_keys.update([_pk, _probk, _clsk])
    if 'supcon3_staged' in _k:   _k_staged_sc = 3
    elif 'supcon2_staged' in _k: _k_staged_sc = 2
    elif 'supcon_staged' in _k:  _k_staged_sc = 1
    else:                         _k_staged_sc = 0
    if _k_staged_sc == args.supcon:
        _staged_sc_result_keys.update([_pk, _probk, _clsk])
```

**`_is_staged` in `_is_standard`** — extend the existing expression:
```python
_is_staged   = _rk in _staged_result_keys
_is_standard = (_is_model_key and not _is_mtl and not _is_supcon_st
                and not _is_supcon_mtl and not _is_bsc_all
                and not _is_any_cl and not _is_rcfd and not _is_lc
                and not _is_staged
                and (not args.rerun_models or any(_m in _rk for _m in args.rerun_models)))
```

**Deletion clause** (add after LC deletion clause):
```python
elif getattr(args, 'supcon_staged', False) and _rk in _staged_sc_result_keys and _mm:
    del _filter_res[_rk]
```

---

### E. `main/README.md`

Add to the SupCon section:

```markdown
### Staged SupCon (`--supcon_staged`)

Two-stage alternative to joint SupCon. CNN+GRU dual only, ST only.

| Stage | Loss | Trainable layers | Trigger |
|-------|------|-----------------|---------|
| 1 | SupCon only | All | `val_loss` plateau (auto) or `--cl_phase1_epochs N` |
| 2 | CE only | `cls_feat`, `cls_out` | EarlyStopping (patience=100) |

```bash
# SC1, auto plateau detection (requires val split):
python 03_main_training.py --supcon_staged --supcon 1 ...

# SC3, fixed 150-epoch Stage 1:
python 03_main_training.py --supcon_staged --supcon 3 --cl_phase1_epochs 150 ...
```

Constraints: incompatible with `--mtl`; `--supcon 0` is invalid; `--cl_phase1_epochs` required when no val split exists.
```

---

## Verification

```bash
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main

# 1. All imports + key sets
python -c "
from utils.model_training.model_utils_supcon import (
    create_cnn_gru_dual_supcon_staged_model,
    create_cnn_gru_dual_attn_recon_supcon_staged_model,
    create_cnn_gru_dual_supcon2_staged_model,
    create_cnn_gru_dual_attn_recon_supcon2_staged_model,
    create_cnn_gru_dual_supcon3_staged_model,
    create_cnn_gru_dual_attn_recon_supcon3_staged_model,
    ALL_STAGED_SUPCON_KEYS, AutoPhaseTransitionSTCallback, FixedPhaseTransitionSTCallback,
)
print('imports OK'); print(ALL_STAGED_SUPCON_KEYS)
"

# 2. Forward pass shape + curriculum_phase + test_step val_loss = SC in phase 0
python -c "
import numpy as np, tensorflow as tf
from utils.model_training.model_utils_supcon import (
    create_cnn_gru_dual_supcon_staged_model,
    create_cnn_gru_dual_supcon2_staged_model,
    create_cnn_gru_dual_supcon3_staged_model,
    create_cnn_gru_dual_attn_recon_supcon_staged_model,
)
x  = np.random.randn(4, 100, 1).astype('float32')
xs = np.random.randn(4, 5, 100).astype('float32')
for name, fn, inp in [
    ('SC1',      lambda: create_cnn_gru_dual_supcon_staged_model(100, 3),           x),
    ('SC2',      lambda: create_cnn_gru_dual_supcon2_staged_model(100, 3),          x),
    ('SC3',      lambda: create_cnn_gru_dual_supcon3_staged_model(100, 3),          x),
    ('SC1-attn', lambda: create_cnn_gru_dual_attn_recon_supcon_staged_model(5,100,3), xs),
]:
    m = fn()
    m.compile(optimizer='adam', metrics=['accuracy'], jit_compile=False)
    out = m(inp, training=False)
    assert hasattr(m, 'curriculum_phase'), f'{name}: missing curriculum_phase'
    cls = out[0]; assert cls.shape == (4, 3), f'{name}: bad shape {cls.shape}'
    print(f'{name}: OK — phase={m.curriculum_phase.numpy()}')
"

# 3. Phase transition smoke test (fixed 5-epoch Stage 1)
python -u 03_main_training.py \
    --task_id 0 \
    --exp_folder /vol/bitbucket/gk225/POC_DDM_datasets/LAB_DDM_paper \
    --curve_type ori_curve --n_splits 2 \
    --supcon_staged --supcon 1 --cl_phase1_epochs 5
# Expect: '[StagedSupCon] Phase 1→2 fixed at epoch 5'
# Expect: results written for cnn_gru_dual_supcon_staged* keys

# 4. SC2 and SC3
python -u 03_main_training.py ... --supcon_staged --supcon 2 --cl_phase1_epochs 5
python -u 03_main_training.py ... --supcon_staged --supcon 3 --cl_phase1_epochs 5

# 5. force_rerun scopes correctly
python -u 03_main_training.py ... --supcon_staged --supcon 1 --force_rerun
# Expect: staged SC1 keys cleared; SC2/SC3 staged and non-staged keys untouched

# 6. Incompatibility guards
python -u 03_main_training.py ... --supcon_staged --mtl       # expect sys.exit
python -u 03_main_training.py ... --supcon_staged --supcon 0  # expect sys.exit
```
