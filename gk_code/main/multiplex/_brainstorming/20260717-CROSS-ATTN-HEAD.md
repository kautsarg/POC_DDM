# Label Query Cross-Attention Head (`--cross_attn`)

**Date:** 2026-07-17
**Status:** Planning

---

## 1. Motivation

The current multi-label classification head (`_ml_cls_head`) collapses the backbone embedding
into a flat `(batch, D)` vector, then fires two Dense layers to produce per-target logits.
This means:

- The temporal structure of the PCR curve is destroyed before prediction.
- All `n_targets` are predicted from a single shared embedding with no target-specific attention.
- Chemical interference (co-amplification suppressing one target's signal) can only be captured
  implicitly, via the fused backbone weights — not explicitly.

**The idea**: treat classification as retrieval. Each target maintains a learnable query vector
that "scans" the raw temporal feature sequence output by the backbone, then targets exchange
information via inter-label self-attention before producing independent sigmoid probabilities.

---

## 2. Architecture

```
Input: (batch, T, 1)
       |
   Backbone (CNN+GRU or CNN+Trans dual)
       |
       |--- fused 2D embedding z (batch, D)    [for SupCon projectors, future]
       |
       +--- sequence kv_seq (batch, T', C)     ← KEY CHANGE: expose pre-pooling 3D tensor
                |
                v
    ┌─────────────────────────────────────────────────────────────┐
    │  Label Query Cross-Attention Head                           │
    │                                                             │
    │  LabelQueryEmbedding: (n_targets, query_dim) learnable      │
    │    → tile to (batch, n_targets, query_dim)                  │
    │                                                             │
    │  kv_proj: Dense(query_dim) → kv: (batch, T', query_dim)    │
    │                                                             │
    │  Cross-Attn: MHA(query=label_queries, key=kv, value=kv)    │
    │    → (batch, n_targets, query_dim)                          │
    │  Residual + LayerNorm                                       │
    │                                                             │
    │  Inter-Label Self-Attn: MHA(queries, queries)              │
    │    → (batch, n_targets, query_dim)   ← "the magic step"    │
    │  Residual + LayerNorm                                       │
    │                                                             │
    │  Dense(1, sigmoid) per label + Reshape(n_targets,)         │
    │    → cls_out: (batch, n_targets)                           │
    └─────────────────────────────────────────────────────────────┘
       |
   cls_out: (batch, n_targets)   ← same output shape as _ml_cls_head
```

---

## 3. What to Add

### 3.1 `model_utils_multilabel.py`

#### New custom Keras layer
```python
@tf.keras.utils.register_keras_serializable(package='ml_cross_attn')
class LabelQueryEmbedding(tf.keras.layers.Layer):
    """Learnable label query embeddings; batch dimension extracted from ref_tensor."""
    def __init__(self, n_targets, query_dim, **kwargs):
        super().__init__(**kwargs)
        self.n_targets = n_targets
        self.query_dim = query_dim

    def build(self, input_shape):
        self.query_emb = self.add_weight(
            shape=(self.n_targets, self.query_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='query_emb',
        )

    def call(self, ref_tensor):
        # ref_tensor is just used for batch size; its values are ignored
        batch = tf.shape(ref_tensor)[0]
        q = tf.expand_dims(self.query_emb, 0)         # (1, n_targets, query_dim)
        return tf.tile(q, [batch, 1, 1])              # (batch, n_targets, query_dim)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'n_targets': self.n_targets, 'query_dim': self.query_dim})
        return cfg
```

#### New head function
```python
def _ml_cross_attn_head(kv_seq, ref_tensor, n_targets, query_dim=64, num_heads=4):
    """
    kv_seq:     (batch, T', C) — pre-pooling 3D sequence from backbone
    ref_tensor: any tensor for batch-size extraction (e.g., the model input)
    returns:    cls_out (batch, n_targets) sigmoid
    """
    queries = LabelQueryEmbedding(n_targets, query_dim, name='label_q_emb')(ref_tensor)
    kv = tf.keras.layers.Dense(query_dim, name='ca_kv_proj')(kv_seq)

    # Cross-attention: each label query scans the temporal sequence
    ca = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=query_dim // num_heads, name='cross_attn'
    )(query=queries, key=kv, value=kv)
    ca = tf.keras.layers.LayerNormalization(name='ca_ln')(queries + ca)

    # Inter-label self-attention: targets exchange context
    sa = tf.keras.layers.MultiHeadAttention(
        num_heads=2, key_dim=query_dim // 2, name='inter_label_sa'
    )(ca, ca)
    sa = tf.keras.layers.LayerNormalization(name='sa_ln')(ca + sa)

    # Per-label sigmoid
    logits = tf.keras.layers.Dense(1, name='label_logit')(sa)   # (batch, n_t, 1)
    cls_out = tf.keras.layers.Activation('sigmoid', name='cls_out')(
        tf.keras.layers.Reshape((n_targets,))(logits))
    return cls_out
```

#### New backbone helpers (expose 3D pre-pooling tensor)

Both `_build_cnn_gru_dual_branches_mtl` and `_build_cnn_trans_dual_branches_mtl` currently
collapse the time dimension internally before returning `z`. We need new helpers that tap the
3D tensor before that collapse. Both accept `return_branches=False` (SC0/SC1) or
`return_branches=True` (SC2/SC3 — also returns the per-branch 2D embeddings for projectors).

**`_build_gru_dual_seq_backbone(inputs, return_branches=False)`**:
- Call `_build_cnn_backbone_mtl(inputs)` → `cnn_emb (batch, 32)` (2D — fine)
- Replicate GRU path but stop after the FIRST `BiGRU(return_sequences=True)`:
  - `gru_seq = BiGRU(units, return_sequences=True, name='seq_gru1')(inputs)` → **(batch, T, 2U)**
  - This is `kv_seq`
- Continue GRU path: `LayerNorm → BiGRU(return_sequences=False) → Dropout → Dense(32)` → `gru_emb (batch, 32)`
- Fuse: `Concat([cnn_emb, gru_emb]) → Dense(64) → Dropout` → `z (batch, 64)`
- Return `(z, gru_seq)` if `return_branches=False`
- Return `(z, gru_seq, cnn_emb, gru_emb)` if `return_branches=True`

**`_build_trans_dual_seq_backbone(inputs, return_branches=False)`**:
- Call `_build_cnn_backbone_mtl(inputs)` → `cnn_emb (batch, 32)` (2D)
- Replicate Transformer path but expose the transformer block output (before `GlobalAveragePooling1D`):
  - Conv1D projection → `MHA(x, x)` → `Add+Norm` → FF → `Add+Norm` → `trans_seq (batch, T, 32)` ← kv_seq
  - Then `GlobalAveragePooling1D → Dropout → Dense(32)` → `trans_emb (batch, 32)`
- Fuse: same as above → `z (batch, 64)`
- Return `(z, trans_seq)` if `return_branches=False`
- Return `(z, trans_seq, cnn_emb, trans_emb)` if `return_branches=True`

> **Note**: these helpers replicate the layer structure from the existing dual backbone builders.
> The layer names MUST use distinct suffixes (e.g., `name='cattn_seq_gru1'`) to avoid
> Keras name collisions when both a standard dual model and a cross-attn model are built
> in the same session.

#### New factory functions

**SC0 (base, no SupCon)** — `_StandardMultiLabelModel`, 1 output:
```python
def create_ml_cnn_gru_dual_cross_attn_model(T, n_targets):
    inputs     = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq  = _build_gru_dual_seq_backbone(inputs)
    cls_out    = _ml_cross_attn_head(kv_seq, inputs, n_targets, query_dim=64)
    return _StandardMultiLabelModel(inputs=inputs, outputs=cls_out,
                                    name='cnn_gru_dual_cross_attn')

def create_ml_cnn_trans_dual_cross_attn_model(T, n_targets):
    inputs      = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, trans_seq = _build_trans_dual_seq_backbone(inputs)
    cls_out     = _ml_cross_attn_head(trans_seq, inputs, n_targets, query_dim=64)
    return _StandardMultiLabelModel(inputs=inputs, outputs=cls_out,
                                    name='cnn_trans_dual_cross_attn')
```

**SC1 (Jaccard SupCon on fused `z`)** — `MultiLabelSupConModel`, 2 outputs `[cls_out, proj_norm]`:
```python
def create_ml_cnn_gru_dual_cross_attn_supcon_model(T, n_targets):
    inputs    = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq = _build_gru_dual_seq_backbone(inputs)
    cls_out   = _ml_cross_attn_head(kv_seq, inputs, n_targets)
    proj_norm = _proj_head(z, 'fused')
    return MultiLabelSupConModel(inputs=inputs, outputs=[cls_out, proj_norm])

def create_ml_cnn_trans_dual_cross_attn_supcon_model(T, n_targets):
    inputs       = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, trans_seq = _build_trans_dual_seq_backbone(inputs)
    cls_out      = _ml_cross_attn_head(trans_seq, inputs, n_targets)
    proj_norm    = _proj_head(z, 'fused')
    return MultiLabelSupConModel(inputs=inputs, outputs=[cls_out, proj_norm])
```

**SC2 (Jaccard SupCon on per-branch embeddings)** — `MultiLabelSupConBranch2STModel`, 3 outputs `[cls_out, cnn_proj, seq_proj]`:
```python
def create_ml_cnn_gru_dual_cross_attn_supcon2_model(T, n_targets):
    inputs                       = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq, cnn_emb, gru_emb  = _build_gru_dual_seq_backbone(inputs, return_branches=True)
    cls_out  = _ml_cross_attn_head(kv_seq, inputs, n_targets)
    cnn_proj = _proj_head(cnn_emb, 'cnn')
    seq_proj = _proj_head(gru_emb, 'seq')
    return MultiLabelSupConBranch2STModel(inputs=inputs, outputs=[cls_out, cnn_proj, seq_proj])

def create_ml_cnn_trans_dual_cross_attn_supcon2_model(T, n_targets):
    inputs                         = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, trans_seq, cnn_emb, tr_emb  = _build_trans_dual_seq_backbone(inputs, return_branches=True)
    cls_out  = _ml_cross_attn_head(trans_seq, inputs, n_targets)
    cnn_proj = _proj_head(cnn_emb, 'cnn')
    seq_proj = _proj_head(tr_emb,  'seq')
    return MultiLabelSupConBranch2STModel(inputs=inputs, outputs=[cls_out, cnn_proj, seq_proj])
```

**SC3 (Jaccard SupCon on all three: CNN, seq, fused)** — `MultiLabelSupConBranch3STModel`, 4 outputs `[cls_out, cnn_proj, seq_proj, fused_proj]`:
```python
def create_ml_cnn_gru_dual_cross_attn_supcon3_model(T, n_targets):
    inputs                       = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq, cnn_emb, gru_emb  = _build_gru_dual_seq_backbone(inputs, return_branches=True)
    cls_out    = _ml_cross_attn_head(kv_seq, inputs, n_targets)
    cnn_proj   = _proj_head(cnn_emb, 'cnn')
    seq_proj   = _proj_head(gru_emb, 'seq')
    fused_proj = _proj_head(z,       'fused')
    return MultiLabelSupConBranch3STModel(inputs=inputs, outputs=[cls_out, cnn_proj, seq_proj, fused_proj])

def create_ml_cnn_trans_dual_cross_attn_supcon3_model(T, n_targets):
    inputs                         = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, trans_seq, cnn_emb, tr_emb  = _build_trans_dual_seq_backbone(inputs, return_branches=True)
    cls_out    = _ml_cross_attn_head(trans_seq, inputs, n_targets)
    cnn_proj   = _proj_head(cnn_emb, 'cnn')
    seq_proj   = _proj_head(tr_emb,  'seq')
    fused_proj = _proj_head(z,       'fused')
    return MultiLabelSupConBranch3STModel(inputs=inputs, outputs=[cls_out, cnn_proj, seq_proj, fused_proj])
```

> **Note**: `MultiLabelSupConBranch2STModel` and `MultiLabelSupConBranch3STModel` are already
> defined in `model_utils_multilabel.py` (lines 183, 215). No new model classes needed.

#### Registry updates
```python
# ML_FACTORIES additions (8 total: 2 × 4 SC levels):
'cnn_gru_dual_cross_attn':          create_ml_cnn_gru_dual_cross_attn_model,
'cnn_gru_dual_cross_attn_supcon':   create_ml_cnn_gru_dual_cross_attn_supcon_model,
'cnn_gru_dual_cross_attn_supcon2':  create_ml_cnn_gru_dual_cross_attn_supcon2_model,
'cnn_gru_dual_cross_attn_supcon3':  create_ml_cnn_gru_dual_cross_attn_supcon3_model,
'cnn_trans_dual_cross_attn':         create_ml_cnn_trans_dual_cross_attn_model,
'cnn_trans_dual_cross_attn_supcon':  create_ml_cnn_trans_dual_cross_attn_supcon_model,
'cnn_trans_dual_cross_attn_supcon2': create_ml_cnn_trans_dual_cross_attn_supcon2_model,
'cnn_trans_dual_cross_attn_supcon3': create_ml_cnn_trans_dual_cross_attn_supcon3_model,

# ML_MODEL_KEY_MAP additions (same pattern, no AC_ prefix):
'cnn_gru_dual_cross_attn':           ('y_preds_cnn_gru_dual_cross_attn_',           ...),
'cnn_gru_dual_cross_attn_supcon':    ('y_preds_cnn_gru_dual_cross_attn_supcon_',    ...),
'cnn_gru_dual_cross_attn_supcon2':   ('y_preds_cnn_gru_dual_cross_attn_supcon2_',   ...),
'cnn_gru_dual_cross_attn_supcon3':   ('y_preds_cnn_gru_dual_cross_attn_supcon3_',   ...),
'cnn_trans_dual_cross_attn':          ('y_preds_cnn_trans_dual_cross_attn_',          ...),
'cnn_trans_dual_cross_attn_supcon':   ('y_preds_cnn_trans_dual_cross_attn_supcon_',   ...),
'cnn_trans_dual_cross_attn_supcon2':  ('y_preds_cnn_trans_dual_cross_attn_supcon2_',  ...),
'cnn_trans_dual_cross_attn_supcon3':  ('y_preds_cnn_trans_dual_cross_attn_supcon3_',  ...),

# ML_MODEL_PRINT_MAP additions:
'cnn_gru_dual_cross_attn':           'CNN+GRU CAttn SC0',
'cnn_gru_dual_cross_attn_supcon':    'CNN+GRU CAttn SC1',
'cnn_gru_dual_cross_attn_supcon2':   'CNN+GRU CAttn SC2',
'cnn_gru_dual_cross_attn_supcon3':   'CNN+GRU CAttn SC3',
'cnn_trans_dual_cross_attn':          'CNN+Tr CAttn SC0',
'cnn_trans_dual_cross_attn_supcon':   'CNN+Tr CAttn SC1',
'cnn_trans_dual_cross_attn_supcon2':  'CNN+Tr CAttn SC2',
'cnn_trans_dual_cross_attn_supcon3':  'CNN+Tr CAttn SC3',

# _SUPCON_ML_KEYS additions (6 new, SC1/2/3 only — SC0 base is not a SupCon key):
'cnn_gru_dual_cross_attn_supcon',   'cnn_gru_dual_cross_attn_supcon2',   'cnn_gru_dual_cross_attn_supcon3',
'cnn_trans_dual_cross_attn_supcon', 'cnn_trans_dual_cross_attn_supcon2', 'cnn_trans_dual_cross_attn_supcon3',
```

### 3.2 `03_main_training.py`

- Add arg: `parser.add_argument("--cross_attn", action="store_true")`
- Add `_cross_attn_by_sc` dict (mirrors `_rcfd_by_sc` pattern, `--supcon` selects SC level):
  ```python
  _cross_attn_by_sc = {
      0: ['cnn_gru_dual_cross_attn',         'cnn_trans_dual_cross_attn'],
      1: ['cnn_gru_dual_cross_attn_supcon',   'cnn_trans_dual_cross_attn_supcon'],
      2: ['cnn_gru_dual_cross_attn_supcon2',  'cnn_trans_dual_cross_attn_supcon2'],
      3: ['cnn_gru_dual_cross_attn_supcon3',  'cnn_trans_dual_cross_attn_supcon3'],
  }
  ```
- Model list selection (evaluated BEFORE the `--supcon` ML branches):
  ```python
  if args.condreg:
      models = _rcfd_by_sc[args.supcon]
  elif args.cross_attn:
      models = _cross_attn_by_sc[args.supcon]
  elif args.supcon == 0: ...
  ```
- `_mode` update:
  ```python
  _mode = (
      "RCFD"  if args.condreg   else
      "CATTN" if args.cross_attn else
      "ML"
  ) + f" SC{args.supcon}"
  ```

### 3.3 `config_multiplex.py`

Add to `MULTIPLEX_MODELS` list (after `cnn_trans_dual_supcon3`, before RCFD section):
```python
# Dual-branch CNN+GRU cross-attn: base + SC1/2/3
'cnn_gru_dual_cross_attn',
'cnn_gru_dual_cross_attn_supcon', 'cnn_gru_dual_cross_attn_supcon2', 'cnn_gru_dual_cross_attn_supcon3',
# Dual-branch CNN+Trans cross-attn: base + SC1/2/3
'cnn_trans_dual_cross_attn',
'cnn_trans_dual_cross_attn_supcon', 'cnn_trans_dual_cross_attn_supcon2', 'cnn_trans_dual_cross_attn_supcon3',
```

---

## 4. What to Adjust

| Item | Current | After |
|---|---|---|
| `_mode` string in 03 | `"RCFD"` or `"ML"` | + `"CATTN"` branch |
| `_cross_attn_by_sc` dict in 03 | — | 4-entry dict (SC0/1/2/3 × 2 models) |
| `MULTIPLEX_MODELS` count | 30 | 38 |
| `ML_FACTORIES` count | 30 | 38 |
| `ML_MODEL_KEY_MAP` count | 30 | 38 |
| `ML_MODEL_PRINT_MAP` count | 30 | 38 |
| `_SUPCON_ML_KEYS` count | 21 | 27 (+ 6 cross-attn SC1/2/3) |

---

## 5. What to Consider

### 5.1 Backbone layer names for cross-attn helpers
When `create_ml_cnn_gru_dual_model` and `create_ml_cnn_gru_dual_cross_attn_model` are both
built in the same training loop, Keras auto-increments duplicate layer names. To avoid silent
name collisions producing confusing model graphs, prefix all cross-attn backbone layers with
`'ca_'` (e.g., `'ca_seq_gru1'`, `'ca_cnn_conv1'`).

### 5.2 query_dim vs backbone width
The fused 2D embedding is 64-dim for both dual-branch models. Setting `query_dim=64` matches
this; the `kv_proj Dense(64)` projects the K/V sequence into the same space. No mismatch.

### 5.3 GRU sequence length
The first BiGRU is applied to the full `(batch, T, 1)` input where T=45 timepoints. Its output
is `(batch, 45, 2*GRU_units)`. This is the K/V sequence. 45 tokens is lightweight — cross-attn
cost is O(n_targets × T) ≈ O(3 × 45) = trivial.

### 5.4 Transformer sequence length
Transformer branches typically reduce T via `Conv1D(strides)`. The actual T' after striding
needs to be confirmed when reading `_build_cnn_trans_dual_branches_mtl`. If T' is small (e.g.,
22), cross-attn is even cheaper.

### 5.5 Output compatibility with existing training loop
`_StandardMultiLabelModel` with a `cls_out` activation='sigmoid' output `(batch, n_targets)` is
identical to the standard ML models. `evaluate_outlier_filters_ml` dispatches by key; since
neither new key is in `_RCFD_ML_KEYS`, the single-output training path is used automatically.

### 5.6 SupCon variants (SC1/2/3) — planned

The key insight: **only the cls head changes** (cross-attn vs dense); the SupCon projectors
attach to the same 2D embeddings (`z`, `cnn_emb`, `seq_emb`) as the standard dual-branch SC
variants. No new model class is required — the existing `MultiLabelSupConModel` (SC1),
`MultiLabelSupConBranch2STModel` (SC2), and `MultiLabelSupConBranch3STModel` (SC3) cover all
output shapes exactly.

**What changes per SC level:**

| SC | cls head | projectors | outputs | model class |
|---|---|---|---|---|
| SC0 | `_ml_cross_attn_head(kv_seq, inputs, n)` | none | `[cls_out]` | `_StandardMultiLabelModel` |
| SC1 | `_ml_cross_attn_head(kv_seq, inputs, n)` | `_proj_head(z, 'fused')` | `[cls_out, proj_norm]` | `MultiLabelSupConModel` |
| SC2 | `_ml_cross_attn_head(kv_seq, inputs, n)` | `_proj_head(cnn_emb), _proj_head(seq_emb)` | `[cls_out, cnn_proj, seq_proj]` | `MultiLabelSupConBranch2STModel` |
| SC3 | `_ml_cross_attn_head(kv_seq, inputs, n)` | all three projectors | `[cls_out, cnn_proj, seq_proj, fused_proj]` | `MultiLabelSupConBranch3STModel` |

SC2/3 require `return_branches=True` from `_build_*_seq_backbone` to get `cnn_emb` and `seq_emb`.
Factory code and registry entries are fully specified in §3.1 above.

### 5.7 RCFD + cross-attn — deferred
`_build_rcfd_backbone` returns `z_cond` (2D) after FiLM conditioning. The pre-FiLM dual backbone
sequences could also be exposed. Deferred.

### 5.8 SLURM job
The existing `lab_multiplex_training.sh` loops with `--supcon 0/1/2/3` and `--condreg`. Add a
`--cross_attn` loop that also iterates `--supcon 0/1/2/3` (same pattern as `--condreg`):
```bash
for SC in 0 1 2 3; do
    sbatch ... python 03_main_training.py --cross_attn --supcon $SC ...
done
```
This runs 4 jobs and covers all 8 cross-attn models (2 per SC level).

### 5.9 The `z` (2D) embedding from `_build_*_seq_backbone` is unused in base models
The backbone helpers return `(z, kv_seq)` but `z` is only passed to `_ml_cross_attn_head` as
`ref_tensor` for batch-size extraction. `z` itself is discarded. This is intentional for SC0;
when SupCon variants are added, `z` will feed the projector. No dead-variable issue since this
is Keras functional API — unused tensors in the computation graph are simply not connected to any
output and don't appear in the model.

---

## 6. Verification

```bash
# Smoke SC0: cross-attn base, 2 folds
python -u 03_main_training.py \
  --task_id 0 --exp_folder $EXP_FOLDER \
  --n_splits 2 --cross_attn --supcon 0 --force_rerun
# Expected: _mode="CATTN SC0", models=['cnn_gru_dual_cross_attn', 'cnn_trans_dual_cross_attn']

# Smoke SC1: fused SupCon projector
python -u 03_main_training.py \
  --task_id 0 --exp_folder $EXP_FOLDER \
  --n_splits 2 --cross_attn --supcon 1 --force_rerun
# Expected: _mode="CATTN SC1", 2 outputs per model [cls_out, proj_norm]

# Smoke SC2: branch projectors (triggers return_branches=True path)
python -u 03_main_training.py \
  --task_id 0 --exp_folder $EXP_FOLDER \
  --n_splits 2 --cross_attn --supcon 2 --force_rerun
# Expected: _mode="CATTN SC2", 3 outputs [cls_out, cnn_proj, seq_proj]

# Smoke SC3: all three projectors
python -u 03_main_training.py \
  --task_id 0 --exp_folder $EXP_FOLDER \
  --n_splits 2 --cross_attn --supcon 3 --force_rerun
# Expected: _mode="CATTN SC3", 4 outputs [cls_out, cnn_proj, seq_proj, fused_proj]

# All smokes: y_binary.shape=(N, 3), no shape/scope errors,
# results in classification_performances_ml_10fold.joblib under key None (Baseline)
```

---

## 7. New Model Key Checklist

**Backbone helpers (`model_utils_multilabel.py`)**
- [x] `_build_gru_dual_seq_backbone(inputs, return_branches=False)` — returns `(z, kv_seq)` or `(z, kv_seq, cnn_emb, gru_emb)`
- [x] `_build_trans_dual_seq_backbone(inputs, return_branches=False)` — returns `(z, trans_seq)` or `(z, trans_seq, cnn_emb, trans_emb)`

**`LabelQueryEmbedding` layer + `_ml_cross_attn_head` function**
- [x] `LabelQueryEmbedding` registered keras layer
- [x] `_ml_cross_attn_head(kv_seq, ref_tensor, n_targets, query_dim, num_heads)` function

**Factory functions (8 × model_utils_multilabel.py)**
- [x] `create_ml_cnn_gru_dual_cross_attn_model`
- [x] `create_ml_cnn_gru_dual_cross_attn_supcon_model`
- [x] `create_ml_cnn_gru_dual_cross_attn_supcon2_model`
- [x] `create_ml_cnn_gru_dual_cross_attn_supcon3_model`
- [x] `create_ml_cnn_trans_dual_cross_attn_model`
- [x] `create_ml_cnn_trans_dual_cross_attn_supcon_model`
- [x] `create_ml_cnn_trans_dual_cross_attn_supcon2_model`
- [x] `create_ml_cnn_trans_dual_cross_attn_supcon3_model`

**Registry entries (`model_utils_multilabel.py`)**
- [x] All 8 keys in `ML_FACTORIES`
- [x] All 8 keys in `ML_MODEL_KEY_MAP`
- [x] All 8 keys in `ML_MODEL_PRINT_MAP`
- [x] 6 SC1/2/3 keys added to `_SUPCON_ML_KEYS`

**`config_multiplex.py`**
- [x] All 8 keys in `MULTIPLEX_MODELS`

**`03_main_training.py`**
- [x] `--cross_attn` arg added
- [x] `_cross_attn_by_sc` dict (SC0/1/2/3)
- [x] `elif args.cross_attn: models = _cross_attn_by_sc[args.supcon]` in model-list selection
- [x] `"CATTN"` branch in `_mode` string

**SLURM**
- [ ] `--cross_attn --supcon 0/1/2/3` loop in `lab_multiplex_training.sh` (file not created yet)
