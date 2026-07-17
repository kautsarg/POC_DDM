# Cross-Attention Head — Improvements Brainstorm

**Date**: 2026-07-17  
**Status**: Implementation-ready (2×2 ablation); future ideas listed at end

---

## Why the current implementation underperforms

Two root issues diagnosed by inspecting the architecture:

### Issue 1 — kv_seq is too shallow

`gru_seq` (the current key-value sequence) is the **first BiGRU output before LayerNorm**:

```
inputs → ca_bigru1 → gru_seq  ← kv_seq tapped here
                  ↓
              ca_gru_ln → ca_bigru2 → ... → classification
```

Shape: `(batch, T, 64)` — raw-ish hidden states from the first recurrent pass. The network hasn't compressed temporal patterns or suppressed noise yet. Cross-attention is asked to discriminate between KPC/NDM/VIM from the **least-processed** representation.

Comparison: in DETR and decoder-only transformers, cross-attention keys/values come from **deep encoder features** — the encoder has already learned high-level feature abstractions before the decoder queries them.

### Issue 2 — Head has one cross-attn block and no FFN

The current head:
```
label_queries  →  MHA(cross-attn)  →  LN
                  MHA(inter-label SA)  →  LN  →  Dense(1)  →  sigmoid
```

This is one cross-attn pass + one self-attn pass with no feed-forward sublayer. Standard transformer decoders (e.g., DETR, original "Attention is All You Need") always stack:
```
cross-attn → self-attn → FFN  (×N blocks)
```
The FFN (`Dense(d_model×4) → Dense(d_model)`) provides the per-query non-linear transformation that "digests" the attended context. Without it, each label query updates linearly from the attention output with no further feature mixing.

---

## Proposed 2×2 ablation + combination variants (SC0)

| Variant | kv_seq | Head depth | Extra loss | Factory key |
|---|---|---|---|---|
| `_cross_attn` *(existing)* | First BiGRU (shallow, `T×64`) | 1 CA block, no FFN | — | — |
| `_cross_attn_deepkv` | **Second BiGRU + sinusoidal pos-enc** (`T×32`) | 1 CA block, no FFN | — | `cnn_gru_dual_cross_attn_deepkv` |
| `_cross_attn_deephead` | First BiGRU (shallow, `T×64`) | **2 CA blocks + FFN** | — | `cnn_gru_dual_cross_attn_deephead` |
| `_cross_attn_v2` | Second BiGRU + pos-enc | 2 CA blocks + FFN | — | `cnn_gru_dual_cross_attn_v2` |
| `_cross_attn_v2_auxdet` | Second BiGRU + pos-enc | 2 CA blocks + FFN | **Per-block aux BCE** | `cnn_gru_dual_cross_attn_v2_auxdet` |

Once the winner is identified from the 2×2 ablation, add SC1/2/3 SupCon variants and QuerCon variant (see below) for that winner only.

---

## Architecture details

### New backbone helper: `_build_gru_dual_deep_seq_backbone`

Layer name prefix: `'ca2_'` (prevents Keras name collision with existing `'ca_'` layers).

```
inputs (batch, T, 1)
    │
    ├─── CNN branch ────────────────────────────────────────────────────
    │    ca2_cnn_conv1: Conv1D(16, 5, relu)
    │    ca2_cnn_conv2: Conv1D(8,  3, relu)
    │    ca2_cnn_flat:  Flatten
    │    ca2_cnn_emb:   Dense(32, relu)  →  cnn_emb (batch, 32)
    │
    └─── GRU branch ────────────────────────────────────────────────────
         ca2_bigru1: BiGRU(32, return_sequences=True)  →  (batch, T, 64)
         ca2_gru_ln: LayerNorm
         ca2_bigru2: BiGRU(16, return_sequences=True)  ← KEY CHANGE: return_sequences=True
                     →  (batch, T, 32)  ← deep_gru_seq = kv_seq
         [+ sinusoidal positional encoding on deep_gru_seq]
         ca2_bigru2_pool: GlobalAveragePooling1D  →  (batch, 32)
         Dropout(0.2) → ca2_gru_emb: Dense(64, relu)  →  gru_emb (batch, 64)

Fusion:
    ca2_cgd_merge: Concat([cnn_emb, gru_emb])  →  (batch, 96)
    ca2_cgd_fused: Dense(96, relu)  →  (batch, 96)
    ca2_cgd_drop:  Dropout(0.2)  →  z (batch, 96)

Returns: (z, deep_gru_seq)  or  (z, deep_gru_seq, cnn_emb, gru_emb) if return_branches=True
```

The sinusoidal positional encoding is added element-wise to `deep_gru_seq` before it is used as kv:
`pos_enc[pos, 2i] = sin(pos / 10000^(2i/d))`, `pos_enc[pos, 2i+1] = cos(...)` — standard formulation. Implemented as a non-trainable `Embedding` table or a closed-form Lambda layer.

**Why deeper kv_seq?** The second BiGRU has seen LayerNorm'd first-pass features; its hidden states encode higher-level temporal patterns (e.g., "rising exponential in cycles 15–25") rather than raw timestep noise. Positional encoding re-injects position information that GAP would discard.

---

### New head: `_ml_cross_attn_deep_head`

Signature: `_ml_cross_attn_deep_head(kv_seq, ref_tensor, n_targets, query_dim=64, num_heads=4, n_ca_blocks=2)`

```
queries = LabelQueryEmbedding(n_targets, query_dim)    # (batch, 3, 64) — learnable
kv      = Dense(query_dim, name='ca2_kv_proj')(kv_seq) # (batch, T', 64)

for block_i in range(n_ca_blocks):
    # Cross-attention: label queries attend over sequence
    ca      = MHA(num_heads, key_dim=query_dim//num_heads,
                  name=f'ca2_cross_attn_{block_i}')(query=queries, key=kv, value=kv)
    queries = LN(queries + ca, name=f'ca2_ca_ln_{block_i}')

    # Inter-label self-attention: labels exchange information
    sa      = MHA(num_heads=2, key_dim=32,
                  name=f'ca2_inter_sa_{block_i}')(queries, queries)
    queries = LN(queries + sa, name=f'ca2_sa_ln_{block_i}')

    # Per-query FFN: non-linear feature mixing
    ffn     = Dense(query_dim * 2, relu, name=f'ca2_ffn1_{block_i}')(queries)
    ffn     = Dense(query_dim,           name=f'ca2_ffn2_{block_i}')(ffn)
    queries = LN(queries + ffn, name=f'ca2_ffn_ln_{block_i}')

logits  = Dense(1, name='ca2_label_logit')(queries)              # (batch, 3, 1)
cls_out = sigmoid(Reshape((n_targets,), name='ca2_reshape')(logits))  # (batch, 3)
```

Each block is: **cross-attn → LN → inter-label SA → LN → FFN → LN** with residual connections throughout. With 2 blocks, the label queries refine their representation twice before making the final prediction.

---

### Combination variant: `_cross_attn_v2_auxdet` — per-label auxiliary detection head

**Problem it solves**: With only a final `Dense(1)` loss signal, all label queries may collapse toward attending the same discriminative time steps ("free-riding"). Adding an intermediate loss after each CA block forces each query to independently develop label-discriminative features before the SA step lets them communicate.

**Architecture change** — inside each block of `_ml_cross_attn_deep_head`, add after the CA + LN step:

```
for block_i in range(n_ca_blocks):
    ca      = MHA(...)(query=queries, key=kv, value=kv)
    queries = LN(queries + ca)

    # ─── Auxiliary detection head (before SA, per block) ────────────
    aux_logit = Dense(1, name=f'ca2_aux_logit_{block_i}')(queries)  # (batch, n_targets, 1)
    aux_pred  = sigmoid(Reshape((n_targets,))(aux_logit))            # (batch, n_targets)
    # aux_pred added to model outputs for loss; gradient flows back through queries
    # ──────────────────────────────────────────────────────────────────

    sa      = MHA(...)(queries, queries)
    queries = LN(queries + sa)
    ffn     = Dense(query_dim * 2)(queries)
    ffn     = Dense(query_dim)(ffn)
    queries = LN(queries + ffn)

final_logit = Dense(1)(queries)
cls_out     = sigmoid(Reshape((n_targets,))(final_logit))
```

**Loss**:
```
L = BCE(cls_out, y_true) + lambda_aux * sum_i BCE(aux_pred_i, y_true)
```
`lambda_aux = 0.3` — auxiliary heads are scaffolding, not the primary signal. With `n_ca_blocks=2`, there are 2 auxiliary losses + 1 final loss = 3 BCE terms total.

**Model subclass**: Extend `_StandardMultiLabelModel`. Override `train_step` to unpack `[cls_out, aux0, aux1]` and compute the weighted sum. At inference, only `cls_out` is used.

**Factory key**: `cnn_gru_dual_cross_attn_v2_auxdet`  
**Print map**: `'CNN+GRU CAttn-V2-AuxDet SC0'`

---

## New SupCon variation: Query Contrastive (QuerCon)

**Motivation**: SC1–SC3 apply contrastive loss in the **backbone projection space** (`z` → projection head). This is global: the entire sample embedding is pulled/pushed. QuerCon applies contrastive loss in the **label query space** — per-label, after the final attention block. It incentivises each label query vector to encode label-specific features: `queries[:, j, :]` should cluster across samples that share label `j`, regardless of what other labels are active.

This is orthogonal to SC1–SC3 (different space, different level of abstraction) and can be combined with them.

### Architecture

After the final block of `_ml_cross_attn_deep_head`:

```
queries_final  # (batch, n_targets, query_dim) — output of last FFN+LN

# L2-normalise along query_dim
queries_norm = queries_final / (||queries_final||_2 + eps)  # (batch, n_targets, query_dim)
# Shape: (batch, 3, 64) — 3 per-label embeddings in a 64-dim normed space
```

**QuerCon loss** — per label `j`, treat `queries_norm[:, j, :]` as a `(batch, 64)` embedding matrix and apply NT-Xent / SupCon loss:
- **Positive pair** for label `j` in sample `i`: any other sample `k` where `y_k[j] = 1` (same label active)
- **Negative pair**: samples where `y_k[j] = 0` (label inactive)
- Loss per label: `L_qc_j = SupCon(queries_norm[:, j, :], y_true[:, j])`
- Total: `L_quercon = mean_j L_qc_j`

Combined loss:
```
L = BCE(cls_out, y_true) + lambda_qc * L_quercon
```
`lambda_qc = 0.1` — query contrastive is a regulariser, not the primary signal.

**Key difference from SC1**: SC1 operates on `proj_norm = L2(Dense(128)(z))` where `z (batch, 96)` is the backbone fusion embedding — a single vector per sample. QuerCon operates on `queries_norm[:, j, :] (batch, 64)` — a separate vector per label per sample. The label dimension is explicit; there is no need to encode label identity into the projection.

### Factory key & model subclass

```
'cnn_gru_dual_cross_attn_v2_quercon'   # v2 backbone + QuerCon, no backbone SupCon
'cnn_gru_dual_cross_attn_v2_quercon_sc1'  # v2 + QuerCon + SC1 (backbone SupCon)
```

Model subclass `MultiLabelQueryConModel(MultiLabelSupConModel)`:
- Outputs: `[cls_out, queries_norm]` — `(batch, n_targets)` + `(batch, n_targets, query_dim)`
- `train_step`: `loss = BCE(cls_out, y) + lambda_qc * quercon_loss(queries_norm, y)`
- `quercon_loss`: loop over `j in range(n_targets)`, apply standard SupCon loss on `queries_norm[:, j, :]`

**Print map**:
```
'cnn_gru_dual_cross_attn_v2_quercon':      'CNN+GRU CAttn-V2 QuerCon'
'cnn_gru_dual_cross_attn_v2_quercon_sc1':  'CNN+GRU CAttn-V2 QuerCon+SC1'
```

---

## Factory functions to register

```python
# SC0 (base) — 2×2 ablation
'cnn_gru_dual_cross_attn_deepkv':    create_ml_cnn_gru_dual_cross_attn_deepkv_model
'cnn_gru_dual_cross_attn_deephead':  create_ml_cnn_gru_dual_cross_attn_deephead_model
'cnn_gru_dual_cross_attn_v2':        create_ml_cnn_gru_dual_cross_attn_v2_model
'cnn_trans_dual_cross_attn_v2':      create_ml_cnn_trans_dual_cross_attn_v2_model  # optional CTD

# v2 combination variants
'cnn_gru_dual_cross_attn_v2_auxdet':       create_ml_cnn_gru_dual_cross_attn_v2_auxdet_model
'cnn_gru_dual_cross_attn_v2_quercon':      create_ml_cnn_gru_dual_cross_attn_v2_quercon_model
'cnn_gru_dual_cross_attn_v2_quercon_sc1':  create_ml_cnn_gru_dual_cross_attn_v2_quercon_sc1_model
```

`_StandardMultiLabelModel` for ablation variants. `MultiLabelQueryConModel` for QuerCon variants (custom `train_step`; `MultiLabelQueryConAuxDetModel` if combined with AuxDet).

`ML_MODEL_PRINT_MAP` entries:
```
'cnn_gru_dual_cross_attn_deepkv':          'CNN+GRU CAttn-DeepKV SC0'
'cnn_gru_dual_cross_attn_deephead':        'CNN+GRU CAttn-DeepHead SC0'
'cnn_gru_dual_cross_attn_v2':              'CNN+GRU CAttn-V2 SC0'
'cnn_trans_dual_cross_attn_v2':            'CNN+Tr CAttn-V2 SC0'
'cnn_gru_dual_cross_attn_v2_auxdet':       'CNN+GRU CAttn-V2-AuxDet SC0'
'cnn_gru_dual_cross_attn_v2_quercon':      'CNN+GRU CAttn-V2 QuerCon'
'cnn_gru_dual_cross_attn_v2_quercon_sc1':  'CNN+GRU CAttn-V2 QuerCon+SC1'
```

Training: add `--cross_attn_v2` arg to `03_main_training.py`.  
SLURM: `run_train --cross_attn_v2 --supcon 0`

---

## Post-hoc diagnostic (no new training needed)

**Ablate inter-label self-attention** — to check if the SA step in the existing `_cross_attn` model actually contributes:

In a notebook, load the trained model weights, manually zero out the SA block's output projection weights, and re-evaluate on the test set. If metrics are unchanged, the SA step is a dead weight. If metrics drop, SA is contributing (even if subtly).

This diagnostic informs whether the inter-label SA in `_ml_cross_attn_deep_head` should be kept.

---

## Future ideas (not for immediate implementation)

- **Deformable attention**: Instead of attending over all T positions, learn offsets to select a sparse set of k ≤ T relevant time steps per query. Reduces kv_seq length effectively; important if T is large (>200 cycles).

- **Attention weight visualisation**: Extract `MultiHeadAttention` attention scores (via `return_attention_scores=True`), average over heads, plot `(label_query × time_step)` heatmap per sample. Use to sanity-check that each label query focuses on the correct amplification phase for its target.

*Note: Per-label auxiliary detection head and label query contrastive loss are now in the main body above as `_v2_auxdet` and `_v2_quercon` variants.*
