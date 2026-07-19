# Cross-Attention + CRF-MRF Combination — Brainstorm

**Date**: 2026-07-19  
**Motivation**: CRF-MRF ranks #1–5 (80–83%); cross-attn ranks #8–15 (80–81%). CRF wins on
joint-state inference; cross-attn wins on sequence-aware, label-specific feature extraction.
Question: can combining them beat 83.03%?

---

## What Exists Today

### CRF-MRF (current #1 at 83.03%)

```
inputs (T, 1)
  → _build_cnn_gru_dual_branches_mtl()         # CNN+BiGRU dual, GlobalAvgPool
  → z: (batch, 96)                             # fully pooled — no sequence left
  → Dense(2^n_targets=8, name='crf_logits')    # (batch, 8) joint state logits
  → MultiLabelCRFMRFModel NLL loss             # argmax over 8 states at prediction
```

**Strength**: joint-state NLL captures co-occurrence (e.g., KPC+NDM+VIM tri-positive is rare
→ the logit for state 111 is learned to be appropriately rare).  
**Weakness**: the 96-dim z is fully pooled before the CRF head — all temporal/positional
information is averaged out before any label-specific inference.

### Cross-Attention V2 (current #8 at 81.28%)

```
inputs (T, 1)
  → _build_gru_dual_deep_seq_backbone()        # CNN+BiGRU×2 dual
  → z: (batch, 96)                             # pooled summary (unused in CA head)
  → deep_gru_seq: (batch, T, 32)              # per-timestep sequence (kv_seq)

  → LabelQueryEmbedding(n_targets, 64)         # n_targets learnable query vectors
  → [cross-attn + inter-label SA + FFN] × 2   # queries attend over sequence
  → queries: (batch, n_targets, 64)           # per-label attended representations
  → Dense(1) per label → sigmoid              # INDEPENDENT per-label outputs
```

**Strength**: each label query independently reads relevant temporal windows (Ct, Fm, etc.);
inter-label SA then lets labels inform each other.  
**Weakness**: final step is still independent sigmoid — the CRF constraint (valid joint state)
is absent. Two rare labels can both be predicted at 0.6 threshold simultaneously with no penalty.

---

## Core Idea: Replace Per-Label Sigmoid with CRF NLL

The cross-attn head produces `queries: (batch, n_targets, query_dim)` after the final block.
Instead of projecting each to `Dense(1) → sigmoid`, use the query vectors to derive joint-state
logits and apply CRF NLL loss.

Key insight: the inter-label SA already makes the queries co-occurrence-aware. The CRF head then
formally constrains the output to a valid probability distribution over all 2^n_targets states.
This combines:
- Per-label sequence attention (cross-attn advantage)
- Valid joint-state inference (CRF advantage)
- Explicit co-occurrence in the loss function (not just in the head architecture)

---

## Design Options

### Option A — Flatten-Project (simplest, most expressive)

```
queries: (batch, n_targets, query_dim)
  → Flatten → (batch, n_targets × query_dim)       # concatenate all label queries
  → Dense(2^n_targets, name='crf_logits')           # joint state logits
  → MultiLabelCRFMRFModel NLL loss
```

- `query_dim=64`, `n_targets=3` → flatten = 192 dims → Dense(8)
- Parameter count for CRF head: 192×8 + 8 = 1,544 (trivial)
- **Pro**: fully expressive — the Dense layer can learn any combination of per-label query info
- **Con**: flattening destroys the per-label slot structure; `Dense(192 → 8)` can learn spurious
  correlations between query dims that cross label boundaries

### Option B — Per-Label Projection + Outer Product (structured, interpretable)

```
queries: (batch, n_targets, query_dim)
  → Dense(1, name='ca_crf_emit') per label     # (batch, n_targets, 1) per-label emission scores
  → outer-product expansion into 2^n_targets   # enumerate all combinations
  → CRF bias correction (optional Dense(8))    # add pairwise interaction terms
  → NLL loss
```

Concretely, for n_targets=3 and labels (a, b, c) with per-label logits (l_a, l_b, l_c):
```
state score[s] = Σ_j sign(s_j) × l_j + w[s]    # w[s] = learned CRF bias per state
```
where `sign(s_j) = l_j` if bit j of s is 1, else `-l_j` (or 0).
`w[s]` is a learned 8-vector of state biases.

- **Pro**: per-label logits stay interpretable; `w[s]` captures pure co-occurrence bias
- **Con**: the outer-product factorisation assumes label independence in the emission, corrected
  only by the 8 bias terms. This is a constrained model — less expressive than Option A.

### Option C — Per-Label MLP + Factored Pairwise Interaction

```
queries: (batch, n_targets, query_dim)
  → Dense(d, relu, name='ca_crf_proj') per label     # (batch, n_targets, d)
  → pairwise: for each (i,j) pair, dot product        # (batch, n_pairs) interaction scores
  → concat([per-label scores, pairwise scores])
  → Dense(8, name='crf_logits')                       # joint state logits
  → NLL loss
```

- **Pro**: explicitly captures pairwise label interactions from the query embeddings
- **Con**: more complex; n_pairs = 3 for n_targets=3, manageable but adds a custom tensor op
- Better suited if n_targets scales up (6–10 targets in future datasets)

### Option D — Dual Head MTL (cross-attn sigmoid + CRF NLL on z)

```
backbone → (z, kv_seq)
kv_seq → cross-attn deep head → cls_sig (sigmoid, auxiliary BCE, weight λ_aux)
z      → Dense(8, 'crf_logits') → CRF NLL (main loss)
```

- Two-task: the cross-attn head provides per-label BCE supervision; the CRF head on pooled z
  provides joint-state supervision. The backbone must serve both.
- **Pro**: easiest to implement — wraps existing code; both loss signals flow through backbone
- **Con**: z (pooled) already discards sequence; no improvement in temporal resolution for CRF.
  The CRF head is still the same as existing CRF-MRF. This is an ensemble-in-one-model, not
  a genuine architectural combination.
- This is the weakest option — likely would not outperform the pure CRF-MRF.

### Option E — Query-to-State Attention (most elegant)

```
queries: (batch, n_targets, query_dim)
state_embs: (2^n_targets, query_dim)                  # learned state embeddings
  → state_logits = state_embs @ queries.T             # (batch, n_targets, n_states)
  → sum over n_targets dimension                       # (batch, n_states) joint logits
  → NLL loss
```

Each of the 8 joint states has a learned embedding. The logit for state s is the sum of
dot products between each label's query and the state embedding. This is a bilinear model
that decomposes cleanly per label while still capturing joint state structure.

- **Pro**: interpretable — state_embs[s] encodes what the joint signal for state s "looks like";
  fully differentiable; parameter count = n_states × query_dim = 8 × 64 = 512
- **Con**: novel architecture requiring more careful initialisation; the state embeddings may
  need warmup

---

## Recommendation: Option A first, Option B as ablation

### Rationale

Option A (flatten → Dense(8)) is the direct path from the existing
`_ml_cross_attn_deep_head` function. The only change is replacing the final
`Dense(1) → sigmoid` per label with `Flatten → Dense(8, name='crf_logits')` and wrapping
in `MultiLabelCRFMRFModel` instead of `_StandardMultiLabelModel`. The implementation fits
entirely inside a new `_ml_cross_attn_deep_head_crf()` function.

Option B is a natural ablation: it enforces per-label emission structure and quantifies how
much the label-crossing info in Option A actually helps.

Option E is theoretically the most elegant and worth a third experiment if A/B show promise.

Option D is explicitly not recommended — it doesn't genuinely combine the two architectures.

---

## Proposed Architecture: `cnn_gru_dual_cross_attn_v2_crf`

### Data flow

```
inputs: (batch, T, 1)
  ↓
_build_gru_dual_deep_seq_backbone(inputs)
  → z:           (batch, 96)    [pooled, used for SupCon proj head only]
  → deep_gru_seq: (batch, T, 32) [per-timestep kv for cross-attn]
  ↓
_ml_cross_attn_deep_head_crf(deep_gru_seq, inputs, n_targets)
  ↓
  LabelQueryEmbedding(n_targets, 64)         # queries: (batch, n_targets, 64)
  ca2_kv_proj: Dense(64)                    # kv:      (batch, T, 64)
  [cross-attn + inter-label SA + FFN] × 2   # queries refined per block
  ↓
  Flatten: (batch, n_targets × 64 = 192)   ← Option A only
  Dense(8, name='crf_logits')
  ↓
MultiLabelCRFMRFModel (NLL + argmax/marginals)
```

### SupCon variants

Same SC0–SC3 pattern as existing CRF-MRF. SupCon projection heads attach to `z`:
- SC0: no SupCon
- SC1: `_proj_head(z, 'fused')` → Jaccard SupCon on fused embedding
- SC2: `_proj_head(cnn_emb)` + `_proj_head(gru_emb)` → SupCon on branches
- SC3: all three proj heads

The SupCon loss classes `MultiLabelCRFMRFSC1Model` etc. are already implemented and can be
reused if the model outputs match the convention `[state_logits, proj_norm, ...]`.

However, it may be more practical to start with SC0 only and verify the base combination
works before adding SC variants.

---

## Implementation Sketch

### New function in `model_utils_multilabel.py` (multiplex)

```python
def _ml_cross_attn_deep_head_crf(kv_seq, ref_tensor, n_targets,
                                   query_dim=64, num_heads=4, n_ca_blocks=2):
    """Cross-attn deep head with CRF-MRF output instead of per-label sigmoid.

    Same cross-attn stack as _ml_cross_attn_deep_head, but final output is
    (batch, 2^n_targets) joint state logits for MultiLabelCRFMRFModel NLL.
    """
    queries = LabelQueryEmbedding(n_targets, query_dim, name='ca2_label_q_emb')(ref_tensor)
    kv      = tf.keras.layers.Dense(query_dim, name='ca2_kv_proj')(kv_seq)
    for i in range(n_ca_blocks):
        ca      = tf.keras.layers.MultiHeadAttention(...)(query=queries, key=kv, value=kv)
        queries = tf.keras.layers.LayerNormalization()(queries + ca)
        sa      = tf.keras.layers.MultiHeadAttention(...)(queries, queries)
        queries = tf.keras.layers.LayerNormalization()(queries + sa)
        ffn     = tf.keras.layers.Dense(query_dim*2, activation='relu')(queries)
        ffn     = tf.keras.layers.Dense(query_dim)(ffn)
        queries = tf.keras.layers.LayerNormalization()(queries + ffn)
    flat        = tf.keras.layers.Flatten(name='ca2_crf_flat')(queries)    # (batch, n_t*q_dim)
    state_logits= tf.keras.layers.Dense(2**n_targets, name='crf_logits')(flat)
    return state_logits
```

### New factory functions

```python
def create_ml_cnn_gru_dual_cross_attn_v2_crf_model(T, n_targets):
    inputs    = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq = _build_gru_dual_deep_seq_backbone(inputs)
    state_logits = _ml_cross_attn_deep_head_crf(kv_seq, inputs, n_targets)
    return MultiLabelCRFMRFModel(inputs=inputs, outputs=state_logits, n_targets=n_targets)

def create_ml_cnn_gru_dual_cross_attn_v2_crf_supcon_model(T, n_targets):
    # SC1: SupCon on fused z
    inputs    = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq = _build_gru_dual_deep_seq_backbone(inputs)
    state_logits = _ml_cross_attn_deep_head_crf(kv_seq, inputs, n_targets)
    proj_norm    = _proj_head(z, 'fused')
    return MultiLabelCRFMRFSC1Model(inputs=inputs, outputs=[state_logits, proj_norm],
                                     n_targets=n_targets)
```

Model keys: `'cnn_gru_dual_cross_attn_v2_crf'`, `'cnn_gru_dual_cross_attn_v2_crf_supcon'`

### Result file

These should go into `classification_performances_ml_cattn_v2.joblib` (the existing cattn_v2
result file, same `--cattn_v2` flag group in `03_main_training.py`). No new flag or result
file needed.

### Where models live

All new code stays in `main/multiplex/utils/model_training/model_utils_multilabel.py` and
`main/multiplex/config_multiplex.py`. No changes to `main/utils/` or `main/config.py`.

---

## Expected Parameter Count

Current `_ml_cross_attn_deep_head`:
- LabelQueryEmbedding(3, 64): 3×64 = 192 params
- kv_proj Dense(64): T-dependent, ~64×64 = 4,096
- 2× cross-attn MHA(4 heads, key_dim=16): ~16k each
- 2× inter-label SA(2 heads, key_dim=32): ~8k each
- 2× FFN (64→128→64): ~12k each
- Final Dense(1): 64 params per label = 192 total
Total CA head: ~72k params

Proposed change (Option A):
- Replace final `3 × Dense(1)` (192 params) with `Flatten → Dense(8)` (192×8+8 = 1,544 params)
- Net difference: +1,352 params — negligible
- The backbone (270k params) is unchanged

---

## Expected Gain and Why

**Hypothesis**: the combination should outperform both individually:

1. **vs pure CRF-MRF**: the cross-attn backbone provides richer per-label features than
   the pooled 96-dim z. Each of the 3 label queries attends to different temporal windows
   (KPC Ct ≈ 17-20, VIM Ct ≈ 22-25, NDM somewhere in between). After the inter-label SA,
   the queries carry label-specific temporal features + co-occurrence awareness. The CRF head
   can then learn a cleaner joint-state distribution.

2. **vs pure cross-attn**: the CRF NLL replaces independent sigmoid, which means:
   - Prediction is argmax over a valid joint state (not three independent 0.5 thresholds)
   - Rare co-occurrences (e.g., KPC+NDM+VIM tri-positive) are penalised appropriately
   - High-confidence single-positive predictions are not diluted by adjacent label noise

3. **Potential ceiling**: if the cross-attn queries already capture co-occurrence through
   inter-label SA, the additional CRF constraint may provide only marginal improvement
   (the inter-label SA is doing soft co-occurrence modelling already). The gain could be
   0–1pp or could be larger if the CRF NLL provides a more direct gradient signal.

**Realistic expectation**: 0.5–2pp improvement over CNN+GRU CRF-MRF SC1 (83.03%).
If the approach works, it should rank #1.

---

## Risk Factors

1. **Layer name collision**: `_ml_cross_attn_deep_head_crf` must use 'ca2_' prefix for all layers
   (same as `_ml_cross_attn_deep_head`). If the new CRF factory and old V2 factory are ever
   instantiated in the same `tf.keras.backend` session, layer name conflicts occur.
   Mitigation: use `name='crf_...'` for new layers; `clear_session()` is called before each
   factory build in the training loop (already the convention).

2. **Inter-label SA already partially does what the CRF does**: if the two forms of
   co-occurrence modelling interfere rather than complement each other, performance could be
   flat or slightly worse. This is the main uncertainty.

3. **Flatten destroys slot structure**: in Option A, the `Dense(8)` can learn cross-label
   combinations from the concatenated query vectors. If the inter-label SA has already
   specialised each query to its label, the cross-label mixing from Flatten may add noise.
   Option B (per-label emission + learned bias) is a safer fallback.

---

## Variants to Evaluate (in priority order)

| Model key | Description | Priority |
|---|---|---|
| `cnn_gru_dual_cross_attn_v2_crf_flat` | flat SC0 (Flatten→Dense(8)) | 1st — base experiment |
| `cnn_gru_dual_cross_attn_v2_crf_flat_supcon` | flat SC1 (+ fused SupCon on z) | 2nd |
| `cnn_gru_dual_cross_attn_v2_crf_factored` | factored SC0 (per-label emit + bias) | 3rd — ablation |
| `cnn_gru_dual_cross_attn_v2_crf_bilinear` | bilinear SC0 (einsum state_embs) | 4th |
| `cnn_trans_dual_cross_attn_v2_crf_flat` | flat SC0 with CNN+Trans backbone | 5th |

All 24 models (3 heads × 4 SC × 2 backbones) are implemented; run in order of priority.

---

## Open Questions

1. **`_CRF_ML_KEYS` membership**: new models must be in `_CRF_ML_KEYS` for
   `predict_binary`/`predict_marginals` evaluation path — and in the cattn_v2 result
   group (not a new group). Does the `--cattn_v2` flag need updating, or does the
   existing group already cover any model with `cattn_v2` in the key?
   → Check `_flag_for_model` in `config_multiplex.py`.

2. **Output shape compatibility**: `MultiLabelCRFMRFSC1Model._step()` expects
   `self(x, training=True)` to return `(state_logits, proj_norm)`. Since the new
   factory uses the same SC1Model class, this should work if the Keras functional model
   outputs are `[state_logits, proj_norm]`. Verify that `_get_crf_output` unpacking
   is correct.

3. **query_dim tuning**: the current `query_dim=64` is shared across both the CA head
   and the flattened input to `Dense(8)`. If flattening 192 dims into 8 is too
   compressive, consider reducing `query_dim` to 32 (96 dims flatten → Dense(8)) for
   the CRF variant.

---

## Correction: Why Option E Was Omitted from the Recommendation

The original recommendation listed only A and B because it implicitly described Option E as
a weaker model: `state_embs: (n_states, query_dim)` → `(sum_j queries[b,j]) @ state_embs.T`.
That formulation pools all label queries before hitting the state embeddings, which loses
per-label identity. It is genuinely less expressive than A.

But the intended formulation — **per-label-per-state embeddings** `state_embs: (n_states, n_targets, query_dim)` with `logit[b,s] = Σ_j dot(queries[b,j], state_embs[s,j])` — is as
expressive as Option A in raw capacity (1,536 vs 1,544 head params), with a different
structural inductive bias:

| | Option A | Option E (factored) | Option B |
|---|---|---|---|
| CRF head params | 192×8+8 = **1,544** | 8×3×64 = **1,536** | 3×65+8 = **203** |
| Cross-label mixing in head | ✓ (Dense sees all 192 dims) | ✗ (per-label slots, no cross) | ✗ |
| Bias term | ✓ | ✗ | ✓ (8 state biases) |
| Gradient structure | any query → any state logit | `queries[b,j]` → only via `state_embs[s,j]` | emit_j → only via per-label Dense(1) |

Key difference vs A: in Option E the gradient from `state_logits[b,s]` back to `queries[b,j]`
flows only through `state_embs[s,j,:]` — the CRF head cannot mix query dimensions across
label slots. Any cross-label inference has to happen in the inter-label SA inside the CA blocks,
not in the CRF projection. This is a stronger structural prior: the backbone is responsible
for label interaction, the CRF head only for joint-state scoring.

All three options are worth experimenting with. Option E is included in the full design below.

---

## Full Implementation Design: Options A, B, E × SC0–3 × CGD+CTD

### Backbone adequacy

**Short answer: the current backbone is sufficient. Do not widen it.**

| Model | Params | Exact acc |
|---|---|---|
| CNN+GRU CRF-MRF SC1 (#1) | **37,424** | 83.03% |
| CNN+GRU CAttn-V2 SC3 (#8) | **117,569** | 81.28% |
| Proposed A/B/E (current backbone) | **~119,000** | — |
| Proposed A (wider: kv=64, z=128, 3 blocks, q=96) | **374,056** | too large |

The current CRF-MRF at 37k params outperforms the 117k cross-attn. Adding more params
did not help the cross-attn. On N=26,571 wells with 5-fold CV (~21k training samples per
fold), a 374k-param model is 17.8 params/sample — high risk of overfitting.

The current deep backbone (`kv_seq=32`, `z=96`, 2 CA blocks, `query_dim=64`) is what the
existing cross-attn v2 uses at 81%. The new CRF head adds only ~1,500 params. The backbone
is not the bottleneck — the output head is. Keep it unchanged and test A, B, E on equal footing.

If SC0 of any option beats 83%, a moderate capacity increase is justified then:
- Increase BiGRU2 from 16→32 units: kv_seq 32→64-dim, +16k params (~136k total)
- Keep 2 CA blocks and `query_dim=64`
- Do NOT increase to 3 CA blocks or z=128 until data justifies it

**CNN+GRU vs CNN+Trans**: both are worth running. CNN+Trans CRF-MRF (#2) is 0.14pp behind
CNN+GRU CRF-MRF (#1), so both backbones are competitive and should be included.

---

### New head functions (to add in `model_utils_multilabel.py`)

The three head functions share the same CA block stack — only the final projection differs.
All use the `'ca2_'` layer name prefix (same as existing v2 models) and are called with
the `deep_gru_seq` / `trans_seq` from the existing v2 backbones.

#### `_ml_cross_attn_deep_head_crf_flat` — flat (Flatten → Dense(8))

```
queries: (batch, n_targets, query_dim)        ← output of CA block stack
  → Flatten('crf_flat')                       → (batch, n_targets * query_dim)
  → Dense(2^n_targets, name='crf_logits')     → (batch, 8)   [joint state logits]
```

Same as `_ml_cross_attn_deep_head` up to the last two lines; replace with Flatten+Dense.
No sigmoid activation — raw logits feed into `MultiLabelCRFMRFModel` NLL loss.

#### `_ml_cross_attn_deep_head_crf_factored` — factored (per-label emit + state bias)

```
queries: (batch, n_targets, query_dim)
  → CRFFactoredHead(n_targets, name='crf_factored')
      emit   = Dense(1)(queries)               → (batch, n_targets)   per-label logit
      logits = einsum('bj,sj->bs', emit, decode_mat) + state_bias
                                               → (batch, 8)           state logits
```

`decode_mat: (n_states, n_targets)` is a constant (0/1 bit matrix). `state_bias: (n_states,)`
is a trainable weight initialised to zero. Both live inside `CRFFactoredHead`.

#### `_ml_cross_attn_deep_head_crf_bilinear` — bilinear (per-label-per-state einsum)

```
queries: (batch, n_targets, query_dim)
  → CRFBilinearHead(n_targets, query_dim, name='crf_bilinear')
      state_embs: trainable (n_states, n_targets, query_dim)  [8×3×64 = 1536 params]
      logits[b, s] = Σ_j dot(queries[b,j], state_embs[s,j])
                   = einsum('bjd,sjd->bs', queries, state_embs)
                                               → (batch, 8)
```

Per-label-per-state bilinear projection. No cross-label mixing in the CRF head — all
label interaction stays in the inter-label SA blocks.

---

### Model key naming convention

```
cnn_gru_dual_cross_attn_v2_crf_{flat,factored,bilinear}              SC0
cnn_gru_dual_cross_attn_v2_crf_{flat,factored,bilinear}_supcon        SC1
cnn_gru_dual_cross_attn_v2_crf_{flat,factored,bilinear}_supcon2       SC2
cnn_gru_dual_cross_attn_v2_crf_{flat,factored,bilinear}_supcon3       SC3

cnn_trans_dual_cross_attn_v2_crf_{flat,factored,bilinear}             SC0
cnn_trans_dual_cross_attn_v2_crf_{flat,factored,bilinear}_supcon      SC1
cnn_trans_dual_cross_attn_v2_crf_{flat,factored,bilinear}_supcon2     SC2
cnn_trans_dual_cross_attn_v2_crf_{flat,factored,bilinear}_supcon3     SC3
```

Total: 3 options × 4 SC variants × 2 backbones = **24 new model keys**.

---

### SupCon variants (SC1–3)

Same model classes as existing CRF-MRF SC variants, same output convention:

| SC | Output tuple from `call()` | Model class |
|---|---|---|
| SC0 | `state_logits` | `MultiLabelCRFMRFModel` |
| SC1 | `[state_logits, proj_fused]` | `MultiLabelCRFMRFSC1Model` |
| SC2 | `[state_logits, proj_cnn, proj_seq]` | `MultiLabelCRFMRFSC2Model` |
| SC3 | `[state_logits, proj_cnn, proj_seq, proj_fused]` | `MultiLabelCRFMRFSC3Model` |

SupCon projection heads attach to `z` from the backbone (same as existing CRF-MRF). The
backbone must return branches to support SC2/SC3: `_build_gru_dual_deep_seq_backbone(inputs,
return_branches=True)` → `(z, kv_seq, cnn_emb, gru_emb)`.

`_get_crf_output` in SC1–3 already unpacks `self(x)[0]` — no change needed if the state
logits are always first in the output tuple.

---

### Factory structure (one per option × SC variant × backbone)

Each factory follows the same pattern. Shown for flat + CGD backbone:

```python
# SC0
def create_ml_cnn_gru_dual_cross_attn_v2_crf_flat_model(T, n_targets):
    inputs    = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq = _build_gru_dual_deep_seq_backbone(inputs)
    logits    = _ml_cross_attn_deep_head_crf_flat(kv_seq, inputs, n_targets)
    return MultiLabelCRFMRFModel(inputs=inputs, outputs=logits, n_targets=n_targets)

# SC1
def create_ml_cnn_gru_dual_cross_attn_v2_crf_flat_supcon_model(T, n_targets):
    inputs    = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq = _build_gru_dual_deep_seq_backbone(inputs)
    logits    = _ml_cross_attn_deep_head_crf_flat(kv_seq, inputs, n_targets)
    proj_norm = _proj_head(z, 'fused')
    return MultiLabelCRFMRFSC1Model(inputs=inputs, outputs=[logits, proj_norm],
                                     n_targets=n_targets)
# ...and so on for SC2/SC3 and factored/bilinear variants
```

Replace `_crf_flat` with `_crf_factored` or `_crf_bilinear` for the other options.
Replace `_build_gru_dual_deep_seq_backbone` with `_build_trans_dual_deep_seq_backbone` for CTD.

**24 factories total** — implemented in `model_utils_multilabel.py`.

---

### `_flag_for_model` routing issue

`_flag_for_model` checks `'crf' in key` before `'_v2' in key`, so all 24 new models
would be routed to the existing `'crf'` result file (`classification_performances_ml_crf.joblib`).
This is acceptable — the new models run in the same SLURM job group as the existing CRF-MRF
models, sharing one result file. No new flag or result file is needed.

If separation is desired later, add `'cattn_v2_crf'` check before `'crf'` in `_flag_for_model`
and introduce a new result file key `'cattn_crf'`.

---

### `_CRF_ML_KEYS` and `_SUPCON_ML_KEYS`

- All 24 keys must be added to `_CRF_ML_KEYS` (to use `predict_binary`/`predict_marginals`).
- SC1–3 variants must NOT be added to `_SUPCON_ML_KEYS` (that set is for the non-CRF supcon
  models; CRF SC variants use the `MultiLabelCRFMRFSCnModel` classes directly).

---

### Parameter counts (verified)

| Variant | Params |
|---|---|
| flat SC0 (CGD backbone + Flatten→Dense(8), 1,544 head params) | **~119,048** |
| factored SC0 (CGD backbone + CRFFactoredHead, 203 head params) | **~117,600** |
| bilinear SC0 (CGD backbone + CRFBilinearHead, 1,536 head params) | **~119,040** |
| SC1/2/3 add proj head(s) each | **+~2,000 per proj** |
| CNN+Trans variants | **+6,000–7,000** (trans backbone is slightly larger) |

All are well within the safe param/sample budget for N=26,571. Widening the backbone is
not recommended for the initial experiments.

---

### Files modified (implemented 2026-07-19)

| File | Change |
|---|---|
| `main/multiplex/utils/model_training/model_utils_multilabel.py` | `CRFFactoredHead`, `CRFBilinearHead` custom layers; 3 head functions; 24 factory functions; updated `_CRF_ML_KEYS`, `ML_FACTORIES`, `ML_MODEL_PRINT_MAP` |
| `main/multiplex/config_multiplex.py` | 24 keys added to `MULTIPLEX_MODELS` |
| `main/multiplex/03_main_training.py` | `_crf_by_sc` extended with 6 new keys per SC level |

No changes to `main/utils/` or `main/config.py`.

---

### Evaluation protocol

The 24 models will appear in the leaderboard alongside the existing CRF-MRF and cross-attn
models. Key comparisons:

1. **flat/factored/bilinear SC0 vs CNN+GRU CRF-MRF SC0** — is sequence-aware attention better than
   pooled z for CRF state scoring?
2. **flat/factored/bilinear SC1 vs CNN+GRU CRF-MRF SC1 (#1 at 83.03%)** — does cross-attn + CRF
   beat the current best?
3. **flat vs factored vs bilinear** — which structural inductive bias works best for this task?
4. **Best option SC0 vs SC1 vs SC2 vs SC3** — does SupCon help the combined model?
