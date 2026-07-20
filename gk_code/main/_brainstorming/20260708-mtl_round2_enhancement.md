# MTL Round 2 — Projection-Tower Regression Fix

## Result summary (post round-1 retraining)

Comparison: `classification_performances_10fold.joblib` (new) vs `*_baseline_mtl.joblib` (pre-fix).

| Model family     | AMCA_qdPCR Δ | Direction |
|------------------|-------------|-----------|
| CNN MTL          | +1.79 pp    | ✓ improved |
| GRU MTL          | +0.61 pp    | ✓ improved |
| Trans MTL        | −2.60 pp    | ✗ worse |
| CNN+Trans MTL    | −1.80 pp    | ✗ worse |
| GRU-LF MTL       | −1.61 pp    | ✗ worse |
| Trans-LF MTL     | −1.03 pp    | ✗ worse |

**Pattern**: CNN/GRU single-branch improved; Transformer-based and LF models regressed.

## Root cause

Fix 3 (projection towers, fixed Dense(32)) applied the same tower to ALL models:
- Transformer backbone embeds to only **16 dim**. Dense(32) is a 2× expansion → overfits on ~3k train samples/fold.
- LF models already merge two input streams into a rich embedding; extra Dense(32) adds noise.
- CNN/GRU (64-dim embedding) benefited from gradient isolation.

Fix 1 (clipnorm) + Fix 2 (UW-SO) are sound for all architectures.

---

## Options (all keep Fix 1 + Fix 2)

### Option A — Simple revert (head-only)

```python
def _mtl_wrap(inputs, embedding, n_classes):
    cls_out = Dense(n_classes, 'softmax', name='cls_out')(embedding)
    reg_h   = Dense(16, 'relu', name='reg_hidden')(embedding)
    reg_out = Dense(1,  'linear', name='reg_out')(reg_h)
    return MTLModel(inputs=inputs, outputs=[cls_out, reg_out])
```
Pro: no added params, works for all. Con: CNN/GRU may lose some gain.

---

### Option C — Dimension-adaptive projection towers (head-only)

```python
def _mtl_wrap(inputs, embedding, n_classes):
    emb_dim  = embedding.shape[-1]
    proj_dim = max(8, emb_dim // 4)           # 64→16, 32→8, 16→4
    cls_feat = Dense(proj_dim, 'relu', name='cls_feat')(embedding)
    cls_out  = Dense(n_classes, 'softmax', name='cls_out')(cls_feat)
    reg_feat = Dense(proj_dim, 'relu', name='reg_feat')(embedding)
    reg_h    = Dense(max(4, proj_dim // 2), 'relu', name='reg_hidden')(reg_feat)
    reg_out  = Dense(1, 'linear', name='reg_out')(reg_h)
    return MTLModel(inputs=inputs, outputs=[cls_out, reg_out])
```
Pro: gradient isolation preserved, proportional to backbone. Con: transformer proj_dim=4 may be too small.

---

### Option E — Shared expansion + proportional towers (head-only)

```python
def _mtl_wrap(inputs, embedding, n_classes):
    # Shared expansion: normalise all backbones to common wider rep
    shared   = Dense(96, 'relu', name='shared_exp')(embedding)
    shared   = Dropout(0.2, name='shared_drop')(shared)
    # Per-task projections
    cls_feat = Dense(32, 'relu', name='cls_feat')(shared)
    cls_out  = Dense(n_classes, 'softmax', name='cls_out')(cls_feat)
    reg_feat = Dense(32, 'relu', name='reg_feat')(shared)
    reg_h    = Dense(16, 'relu', name='reg_hidden')(reg_feat)
    reg_out  = Dense(1,  'linear', name='reg_out')(reg_h)
    return MTLModel(inputs=inputs, outputs=[cls_out, reg_out])
```

**Why better than original Fix 3**:
- Transformer 16-dim → 96 (shared expansion, jointly trained by both tasks) → 32 per task.
- CNN/GRU 64-dim → 96 (modest expansion) → 32 per task.
- `shared_drop` regularises the extra parameters.

**Only file to change**: `model_utils_mtl.py` — `_mtl_wrap` (line 151). All 14 factories delegate to it.

---

### Option F — Backbone widening + proportional head towers (backbone + head)

Deeper fix: several architectures have backbone-level capacity constraints that no head change can fully compensate. Address each bottleneck, then use proportional small towers.

**Current embedding dims entering `_mtl_wrap` and their issues:**

| Model | Current dim | Issue |
|---|---|---|
| Transformer | **16** | far too small for dual-task |
| GRU (standard) | 64 | raw BiGRU output — no learned projection |
| LF models (×4) | **32** | 64→32 bottleneck at merge layer |
| CNN+Trans dual | 64 | trans sub-branch only 32-dim before merge |
| CNN+GRU dual | 64 | GRU sub-branch only 32-dim before merge (minor issue) |
| Gated CNN+Trans / CNN+GRU | 64 | `_gated_mtl_head` already projects to 64 — fine |

---

**Change 1 — Widen transformer backbone** (`create_transformer_mtl_model`, line ~225)

```python
# Before:
embedding = tf.keras.layers.Dense(16, activation='relu')(x)
# After:
embedding = tf.keras.layers.Dense(64, activation='relu')(x)
```

ST transformer keeps Dense(16). MTL transformer gets Dense(64) to handle both tasks.

---

**Change 2 — Remove LF merge bottleneck** (4 factories: `create_{gru,cnn,trans,lstm}_lf_mtl_model`)

```python
# Before:
merged    = Concatenate()([curve_emb, feat_emb])   # 32+32 = 64
z         = Dense(32, 'relu')(merged)              # ← bottleneck: 64→32
# After:
z         = Dense(64, 'relu')(merged)              # keep full 64-dim
```

---

**Change 3 — Add Dense projection to standard GRU** (`create_gru_mtl_model`)

Standard GRU passes the **raw BiGRU dropout output** (64-dim) directly to `_mtl_wrap` — no learned projection. Both task heads attach directly to the GRU hidden state.

```python
# Before:
x = Bidirectional(GRU(16))(x)
embedding = Dropout(0.2)(x)                            # raw 64-dim, no projection
return _mtl_wrap(inputs, embedding, n_classes)

# After:
x = Bidirectional(GRU(16))(x)
x = Dropout(0.2)(x)
embedding = Dense(64, activation='relu')(x)            # learnable projection
return _mtl_wrap(inputs, embedding, n_classes)
```

The Dense(64) learns to reorganise GRU hidden states into a representation the task towers can efficiently project from, rather than projecting directly from the raw recurrent hidden state.

---

**Change 4 — Widen transformer sub-branch in CNN+Trans dual** (`create_cnn_trans_dual_mtl_model`)

CNN+Trans dual has a 64-dim final embedding but the transformer sub-branch only outputs 32-dim before the merge. Attention mechanisms are more sensitive to gradient interference than GRU, so the trans branch needs more capacity.

```python
# Before:
trans_emb = Dense(32, relu)(t)                         # 32-dim trans branch
cnn_emb   = Dense(32, relu)(c)                         # 32-dim CNN branch
merged    = Concatenate()([cnn_emb, trans_emb])        # 64-dim
embedding = Dense(64, relu)(merged)                    # 64-dim final

# After:
trans_emb = Dense(64, relu)(t)                         # 64-dim (was 32)
cnn_emb   = Dense(32, relu)(c)                         # 32-dim (unchanged)
merged    = Concatenate()([cnn_emb, trans_emb])        # 96-dim
embedding = Dense(96, relu)(merged)                    # 96-dim final
```

Gated CNN+Trans variants (gate/hadamard/crossattn/film) all go through `_gated_mtl_head → Dense(64)` — they benefit from Change 5's smaller `_mtl_wrap` towers without needing per-factory changes.

---

**Change 5 — Widen GRU sub-branch in CNN+GRU dual** (`_build_cnn_gru_dual_branches_mtl`)

CNN+GRU dual had a minor −0.28 pp degradation. The GRU sub-branch outputs Dense(32) before the merge, same as CNN+Trans dual — widen for consistency.

```python
# Before:
gru_emb = Dense(32, relu)(g)                           # 32-dim GRU branch
cnn_emb = Dense(32, relu)(c)                           # 32-dim CNN branch
merged  = Concatenate([cnn_emb, gru_emb])              # 64-dim
z       = Dense(64, relu)(merged)                      # 64-dim final

# After:
gru_emb = Dense(64, relu)(g)                           # 64-dim (was 32)
cnn_emb = Dense(32, relu)(c)                           # 32-dim (unchanged)
merged  = Concatenate([cnn_emb, gru_emb])              # 96-dim
z       = Dense(96, relu)(merged)                      # 96-dim final
```

*Priority: low — GRU is less sensitive than Transformer. Skip if training time is constrained.*

---

**Change 6 — Simplify `_mtl_wrap` with proportional small towers**

After Changes 1–5, most backbones output 64-dim (transformer, GRU, LF) or 96-dim (dual models). Replace the current fixed Dense(32) towers — which caused overfitting on small embedding sizes — with fixed Dense(16):

```python
def _mtl_wrap(inputs, embedding, n_classes):
    cls_feat = Dense(16, 'relu', name='cls_feat')(embedding)
    cls_out  = Dense(n_classes, 'softmax', name='cls_out')(cls_feat)
    reg_feat = Dense(16, 'relu', name='reg_feat')(embedding)
    reg_h    = Dense(8,  'relu', name='reg_hidden')(reg_feat)
    reg_out  = Dense(1,  'linear', name='reg_out')(reg_h)
    return MTLModel(inputs=inputs, outputs=[cls_out, reg_out])
```

Head parameter count (64-dim backbone, n=8 classes): 4,880 (round-1 Fix 3) → **2,312** (Option F).

---

**Change 7 — Add Dense projection to LSTM MTL** (`create_lstm_mtl_model`)

Change 3 added `Dense(64, relu)` to GRU MTL because it was passing raw 32-dim BiGRU output directly to `_mtl_wrap`. LSTM has the identical problem (raw `Bidirectional(LSTM(16))` output = 32-dim) but was not updated.

```python
# Before:
x = Bidirectional(LSTM(16))(x)
embedding = Dropout(0.2)(x)                  # raw 32-dim, no projection

# After:
x = Bidirectional(LSTM(16))(x)
x = Dropout(0.2)(x)
embedding = Dense(64, activation='relu')(x)  # learnable projection, consistent with GRU
```

---

**Change 8 — Add Dense projection to RNN MTL** (`create_rnn_mtl_model`)

Same as Change 7 but for SimpleRNN. `Bidirectional(SimpleRNN(16))` outputs 32-dim raw hidden state.

```python
# Before:
x = Bidirectional(SimpleRNN(16))(x)
embedding = Dropout(0.2)(x)                  # raw 32-dim

# After:
x = Bidirectional(SimpleRNN(16))(x)
x = Dropout(0.2)(x)
embedding = Dense(64, activation='relu')(x)  # consistent projection
```

*Priority: low — RNN is the weakest baseline architecture.*

---

**Change 9 — Widen LSTM sub-branch in CNN+LSTM dual** (`_build_cnn_lstm_dual_branches_mtl`)

Change 5 widened CNN+GRU dual (gru_emb 32→64, merge 64→96). CNN+LSTM dual uses the same pattern but was not updated — the LSTM sub-branch stays at 32-dim, and the merge stays at 64-dim.

```python
# Before:
lstm_emb = Dense(32, relu)(l)                 # 32-dim LSTM branch
merged_dual = Concatenate()([cnn_emb, lstm_emb])  # 64-dim
z = Dense(64, relu)(merged_dual)              # 64-dim final

# After:
lstm_emb = Dense(64, relu)(l)                 # 64-dim (was 32)
merged_dual = Concatenate()([cnn_emb, lstm_emb])  # 96-dim
z = Dense(96, relu)(merged_dual)              # 96-dim final
```

---

**Summary of changes (model_utils_mtl.py only):**

| # | Location | Change | Priority | Applied |
|---|---|---|---|---|
| 1 | `create_transformer_mtl_model` (~line 225) | `Dense(16)` → `Dense(64)` | high | ✓ |
| 2 | 4 LF factories — merge Dense | `Dense(32)` → `Dense(64)` | high | ✓ |
| 3 | `create_gru_mtl_model` | add `Dense(64, relu)` projection before `_mtl_wrap` | medium | ✓ |
| 4 | `create_cnn_trans_dual_mtl_model` | `trans_emb Dense(32)→Dense(64)`; merge `Dense(64)→Dense(96)` | high | ✓ |
| 5 | `_build_cnn_gru_dual_branches_mtl` | `gru_emb Dense(32)→Dense(64)`; merge `Dense(64)→Dense(96)` | low | ✓ |
| 6 | `_mtl_wrap` (~line 151) | Dense(32)→Dense(16) per task; reg hidden Dense(16)→Dense(8) | high | ✓ |
| 7 | `create_lstm_mtl_model` | add `Dense(64, relu)` projection (same as Change 3 for GRU) | medium | ✗ |
| 8 | `create_rnn_mtl_model` | add `Dense(64, relu)` projection | low | ✗ |
| 9 | `_build_cnn_lstm_dual_branches_mtl` | `lstm_emb Dense(32)→Dense(64)`; merge `Dense(64)→Dense(96)` | medium | ✗ |

**Embedding dimensions after all 9 changes:**

| Model family | Before F | After 1–6 | After 1–9 |
|---|---|---|---|
| CNN MTL | 64 | 64 | 64 |
| LSTM MTL | 32 (raw) | **32 (raw)** | 64 |
| GRU MTL | 32 (raw) | 64 | 64 |
| RNN MTL | 32 (raw) | **32 (raw)** | 64 |
| Transformer MTL | 16 | 64 | 64 |
| CNN+GRU dual | 64 | 96 | 96 |
| CNN+Trans dual | 64 | 96 | 96 |
| CNN+LSTM dual | 64 | **64** | 96 |
| LF factories (×4) | 32 | 64 | 64 |
| Gated (×8) | 64 | 64 | 64 |

No changes to: `_compute_loss`, gated factories (already 64-dim via `_gated_mtl_head`), `model_utils.py`, `config.py`, training scripts.

**Note on ST comparison fairness**: MTL transformer and dual models get wider sub-branches while ST counterparts remain unchanged — a capacity advantage. For strict capacity-controlled comparison, use CNN/GRU single-branch models where both ST and MTL share 64-dim backbones.

---

## If none of the above beat ST

Next escalation: **PCGrad** — surgical gradient projection that removes conflicting gradient components at each backbone parameter before applying the update. Requires a persistent `GradientTape` and per-parameter gradient surgery in `train_step`.

---

## Verification

1. Backup current joblibs:
   ```bash
   LAB=/vol/bitbucket/gk225/POC_DDM_datasets/LAB_DDM_paper
   for DS in ACA_qdPCR AMCA_qdLAMP AMCA_qdPCR; do
       cp "$LAB/$DS/classification_performances_10fold.joblib" \
          "$LAB/$DS/classification_performances_10fold_before_round2.joblib"
   done
   ```
2. Apply chosen option to `model_utils_mtl.py`
3. `sbatch lab_mtl_training.sh` (`--mtl --force_rerun --n_splits 5`)
4. Check `lab_mtl_conc_analysis.ipynb` `plot_stmtl_comparison`:
   - Trans MTL and LF MTL should recover from regressions
   - CNN/GRU MTL should retain or improve on round-1 gains
   - Goal: MTL ≥ ST on AMCA_qdPCR; MTL > ST = strong result
