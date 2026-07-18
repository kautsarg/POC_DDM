# Implementation Guide: Auxiliary Ct Head (6a) & Structured Output CRF (1c)

**Date**: 2026-07-18  
**Context**: `{0,1}^3` multi-label prediction on multiplex dPCR curves (`[KPC, NDM, VIM]`).  
**Scope**: Edit-mode design only. No code changed.

---

## Part 1 — Auxiliary Ct Regression Head (6a)

### What it is

A secondary `Dense(n_targets)` regression head tapped off the same backbone embedding that feeds the classification head. It is trained to predict per-target cycle threshold (Ct) values alongside the binary labels. The regression signal forces the backbone to encode Ct-related temporal features (onset cycle, slope). At inference the regression head is discarded; only the classification sigmoid is used.

This is **distinct from RCFD**: RCFD uses a concentration estimate to condition the backbone via FiLM *before* classification (input modulation). Here, Ct prediction is a parallel *output* head with no feedback into the backbone computation path.

---

### Architecture

```
curve (T, 1)
    │
 [Backbone: CNN / GRU / Transformer / dual]
    │
 fused_emb (batch, d_emb)  ←── same embedding as today
    │         │
    ▼         ▼
Dense(n_targets)   Dense(n_targets)
+ sigmoid          (linear activation)
    │                   │
cls_out (batch, 3)   ct_pred (batch, 3)
```

**Key constraint**: No architectural change to the backbone or the classification head. The regression head is a sibling branch at the top.

---

### Data requirement: Ct sentinel masking

Not every training sample has known per-target Ct values. The sentinel convention (used in existing RCFD training) is:

```
Ct_true[sample, j] = sentinel_value   →   mask out this target for sample in L_reg
```

Recommended sentinel: `-1.0` (Ct values are strictly positive in real data).

Mask construction in the training step:

```python
ct_valid = tf.cast(ct_true > 0.0, tf.float32)          # (batch, n_targets)
l_reg    = tf.reduce_sum(
    ct_valid * tf.square(ct_pred - ct_true), axis=-1)   # per-sample, zeros where invalid
n_valid  = tf.reduce_sum(ct_valid) + 1e-8
l_reg    = tf.reduce_sum(l_reg) / n_valid               # mean over valid (target, sample) pairs
```

If no samples in a batch have Ct annotations, `l_reg = 0` naturally and the gradient contributes nothing.

---

### Loss formulation

```
L_total = L_cls + λ_reg × L_reg

L_cls = BCE(sigmoid(cls_logits), y_binary)           # standard multi-label loss
L_reg = sentinel-masked MSE(ct_pred, ct_true)
```

Recommended starting value: `λ_reg = 0.1`. Ct values range ~15–40 cycles; MSE on raw Ct dwarfs BCE if unweighted.

Optionally normalise Ct values per-fold with `_normalize_concentration` (already in `evaluate_outlier_filters_ml`) to bring MSE into a similar scale as BCE.

---

### SupCon compatibility

The auxiliary Ct head combines with all existing SupCon variants (SC0–SC3) without conflict:

```
L_total = (1 - λ_sc - λ_reg) × L_cls + λ_sc × L_sc + λ_reg × L_reg
```

The SupCon projection head and the Ct regression head are independent siblings on the backbone embedding. No special handling needed — existing model wrapper classes can be subclassed and `_step` overridden to add `l_reg`.

For SupCon + RCFD MTL models (suffix `_supcon_mtl`): the RCFD early encoder already provides a concentration estimate. Adding a per-target Ct head would be redundant on top of RCFD — prefer applying AuxCt only to non-RCFD families.

---

### Files to modify (when implementing)

#### `model_utils_multilabel.py`

1. **New model wrapper class** analogous to `_StandardMultiLabelModel` but with two outputs:

   ```python
   class MultiLabelAuxCtModel(tf.keras.Model):
       def __init__(self, *args, lambda_reg=0.1, ct_sentinel=-1.0, **kwargs):
           super().__init__(*args, **kwargs)
           self.lambda_reg   = lambda_reg
           self.ct_sentinel  = ct_sentinel

       def _step(self, x, y_dict, training):
           cls_out, ct_pred = self(x, training=training)
           y_bin    = tf.cast(y_dict['y_binary'],  tf.float32)
           ct_true  = tf.cast(y_dict['ct_values'], tf.float32)  # (batch, n_targets)

           l_cls    = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_bin, cls_out))

           ct_valid = tf.cast(ct_true > self.ct_sentinel, tf.float32)
           l_reg    = (tf.reduce_sum(ct_valid * tf.square(ct_pred - ct_true))
                       / (tf.reduce_sum(ct_valid) + 1e-8))
           return l_cls + self.lambda_reg * l_reg, l_cls, l_reg
       ...
   ```

2. **New factory functions** — one per backbone variant that should support AuxCt. Suggested minimal set:
   - `create_ml_cnn_gru_dual_auxct_model`
   - `create_ml_cnn_trans_dual_auxct_model`
   - plus SC1/SC2/SC3 variants for each

   Each factory creates the backbone, adds both `Dense(n_targets, activation='sigmoid')` (cls) and `Dense(n_targets)` (ct) as sibling outputs, returns a `MultiLabelAuxCtModel`.

3. **`_AUXCT_ML_KEYS` frozenset** — analogous to `_RCFD_ML_KEYS`, so `evaluate_outlier_filters_ml` can branch on `is_auxct`.

4. **Entries in `ML_FACTORIES`** for the new keys.

5. **Entries in `ML_MODEL_PRINT_MAP`** for display.

#### `config_multiplex.py`

Add `'auxct'` to `RESULT_FILE_BY_FLAG` / `RESULT_10FOLD_FILE_BY_FLAG` (new joblib group).

Update `_flag_for_model` to route new keys to `'auxct'`.

#### `03_main_training.py`

Add `--auxct` CLI flag alongside the existing `--cross_attn`, `--condreg`, etc. flags.

The training loop already handles RCFD's dual output in `evaluate_outlier_filters_ml` via `is_rcfd` branching. Add equivalent `is_auxct` branching to:
- pass `ct_values` into `y_dict`
- read `ct_pred` from the second model output
- compute and save `ct_rmse` (optional diagnostic metric)

#### `_slurm_jobs/lab_multiplex_training.sh`

Add a loop for `--auxct`:

```bash
for SC in 0 1 2 3; do
    SARG=(); [ "$SC" -gt 0 ] && SARG=(--supcon "$SC")
    run_train --auxct "${SARG[@]}"
done
```

---

### What to expect

**Positive signals** (if Ct data is available and of good quality):
- Backbone embeddings encode temporal onset more precisely (earlier activation cycle visible in the embedding).
- Improvement concentrated in cases where sigmoid uncertainty is high — the regression gradient nudges the encoder to preserve cycle-level detail.
- Modest gain (1–3% exact match) on co-amplification samples where target Cts differ significantly.

**Null result conditions**:
- If Ct values are only weakly correlated with curve shape variability (e.g., all targets amplify in a similar cycle window), the regression head learns the task but adds no new information for classification.
- Normalisation of Ct values is critical: unnormalised MSE ≈ 100× BCE in scale, so `λ_reg = 0.1` may still over-weight regression.

**Failure modes**:
- Missing Ct data for most training samples → sentinel masking causes the regression gradient to fire on only 10–20% of batches → effectively zero regularisation. Check `n_valid` per batch in debug logging.
- Ct ambiguity in multi-positive wells: when KPC + NDM co-amplify, the "KPC Ct" is ill-defined (inhibition shifts cycle). If Ct annotations in this regime are noisy, the head introduces inconsistent gradients.

---

### Diagnostic metrics to add

| Metric | Interpretation |
|---|---|
| `ct_rmse_kpc / ndm / vim` | Per-target cycle prediction error (lower = backbone encodes Ct signal) |
| `ct_mae_per_target` | Mean absolute error in cycles |
| `ct_r2_per_target` | How much Ct variance the head explains |
| `frac_sentinel_per_batch` | Fraction of batch without Ct labels (if > 0.8 the signal is sparse) |

Log these in the `test_step` return dict alongside `cls_bce`.

---

### Interaction with cross-attn heads

AuxCt is most natural on the standard dual-backbone (CGD / CTD) which produces a single fused embedding. For cross-attn models, the head would need to be tapped after the inter-label self-attention, which produces per-label attended features rather than a single embedding. To add AuxCt to a cross-attn model, either:

1. Tap the fused pre-attention embedding (before the label queries), or
2. Global-average-pool the query output `queries_norm` → `(batch, d_query)` → `Dense(n_targets)`.

Option 1 is simpler and keeps the regression head independent of the attention.

---

---

## Part 2 — Structured Output CRF (1c)

### What it is

A Conditional Random Field (CRF) output layer replaces the final `sigmoid` + independent BCE. Instead of predicting each label independently, the CRF scores all `2^n_targets = 8` possible label assignments jointly using both unary (per-label) and pairwise (inter-label) potentials. Training maximises the log-probability of the correct assignment; inference finds the globally best assignment via Viterbi.

With 3 binary labels the full enumeration over 8 states is trivial — no approximation needed.

---

### Architecture options

There are two natural framings for a binary multi-label CRF:

#### Option A — Full state CRF (recommended for n_targets = 3)

Treat the label vector `y ∈ {0,1}^3` as a discrete state with `2^3 = 8` possible values. Enumerate all 8 states explicitly.

```
State index  →  label vector
0            →  [0,0,0]   ∅
1            →  [1,0,0]   KPC
2            →  [0,1,0]   NDM
3            →  [1,1,0]   KPC+NDM
4            →  [0,0,1]   VIM
5            →  [1,0,1]   KPC+VIM
6            →  [0,1,1]   NDM+VIM
7            →  [1,1,1]   KPC+NDM+VIM
```

Unary potentials (emission scores):

```python
phi = backbone_emb → Dense(8)   # (batch, 8), no activation
```

Pairwise potentials: learned `(8, 8)` compatibility matrix `W_pair`.
(With 3 labels this is 64 parameters. Alternatively: `(2^3, 2^3)` — but at n_targets=3 it is small and interpretable.)

Energy of assignment `s`:

```
E(s | x) = phi[s] + W_pair[s_prev, s]   ← for linear-chain
         = phi[s]                         ← for a non-sequential (MRF) formulation
```

For a **non-sequential MRF** (the simpler choice when labels have no natural ordering):

```
score(s | x) = phi[s] + psi[s]     where psi is a learned bias per state
log p(s | x) = score(s) - log Σ_{s'} exp(score(s'))   (partition = logsumexp over 8 states)
```

This is exactly an 8-way softmax with a learned prior over label combinations — interpretable and trivial to implement.

#### Option B — Linear-chain CRF over label sequence

Fix an ordering `(KPC, NDM, VIM)`. The CRF chain has 3 positions, each with 2 states. Transition matrix `T (2, 2)` scores how `y_j` transitions to `y_{j+1}`.

Emission at position `j`: `Dense(2)(backbone_emb)` — one per label position (or shared weights).

Partition function: forward algorithm over 3 steps × 2 states — sum of `2^3 = 8` paths, computed in `O(n_targets × 4)` = 12 operations.

**Recommendation**: For `n_targets = 3`, Option A (full state) is simpler to implement, easier to interpret (the `8 × 8` transition matrix directly reads out which label combinations co-occur), and no slower than Option B.

---

### Full state CRF — detailed implementation plan

#### 1. Emission scores

```python
# backbone_emb: (batch, d_emb)
phi = tf.keras.layers.Dense(2**n_targets, use_bias=True)(backbone_emb)   # (batch, 8)
```

No activation — raw logits over the 8 states.

#### 2. State-level prior — already provided by the Dense bias

The `Dense(8, use_bias=True)` layer in step 1 already contains an `(8,)` trainable bias vector. That bias is mathematically identical to a learned state prior — it shifts each state's logit up or down independently of the input. **Do not add a separate `w_compat` trainable variable**: it is redundant. The network would just split the prior arbitrarily between `b_dense` and `w_compat` and the effective prior is their sum. The Dense bias alone is sufficient and is already learned during training.

For stronger pairwise interactions (if co-occurrence statistics matter beyond a state prior), the true CRF extension is a full `(8, 8)` pairwise compatibility matrix — but with only 8 states and sufficient training data, this is usually unnecessary. Start with just the Dense bias.

#### 3. Training loss — negative log-likelihood

```python
log_Z    = tf.reduce_logsumexp(state_logits, axis=-1)   # (batch,)
# true state index for each sample
y_idx    = y @ tf.constant([1,2,4], dtype=tf.int32)     # (batch,) ∈ [0..7]
true_phi = tf.gather(state_logits, y_idx, batch_dims=1) # (batch,)
nll      = tf.reduce_mean(log_Z - true_phi)
```

This is identical to cross-entropy with 8 classes, but only the true label combination is scored — there is no class weighting by prevalence (all 8 states are scored equally in the partition). This can be a concern if ∅ or single-positive states dominate — see Considerations below.

#### 4. Inference — argmax (Viterbi)

```python
pred_state = tf.argmax(state_logits, axis=-1)           # (batch,)
# decode back to binary vector
states     = tf.constant([[int((i >> j) & 1) for j in range(n_targets)]
                           for i in range(2**n_targets)], dtype=tf.int32)   # (8, 3)
y_pred_bin = tf.gather(states, pred_state)               # (batch, 3)
```

No threshold needed. The model predicts the single most probable label combination.

#### 5. Probability output (for saved probs)

```python
state_probs  = tf.nn.softmax(state_logits, axis=-1)   # (batch, 8)
# Marginalise to per-label probabilities
marginals    = tf.zeros((batch, n_targets))
for j in range(n_targets):
    # states where label j = 1
    pos_mask = [(i >> j) & 1 for i in range(8)]
    marginals_j = tf.reduce_sum(state_probs * pos_mask, axis=-1)
    marginals = tf.concat([marginals, marginals_j[:, None]], axis=-1)
```

Or vectorised:

```python
decode_mat = tf.constant([[int((i>>j)&1) for j in range(n_targets)]
                           for i in range(2**n_targets)], dtype=tf.float32)   # (8, 3)
marginals  = state_probs @ decode_mat   # (batch, 3) — per-label marginal probabilities
```

These marginals slot directly into the existing `y_probs_{model}_` result storage without any change to evaluation code.

---

### Class structure

```python
class MultiLabelCRFModel(tf.keras.Model):
    """Full-state CRF: call() returns (batch,8) raw logits; NLL loss; argmax inference.

    INVARIANT: call() / self(x) MUST always return raw logits, never marginals.
    _step relies on self(x) returning logits to compute the partition function correctly.
    predict_marginals() applies softmax + decode_mat on those logits for saved probs.
    """

    def __init__(self, *args, n_targets=3, state_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_targets   = n_targets
        self.n_states    = 2 ** n_targets
        self.decode_mat  = tf.constant(
            [[int((i>>j)&1) for j in range(n_targets)] for i in range(2**n_targets)],
            dtype=tf.float32)                                       # (8, 3)
        self.powers      = tf.constant([2**j for j in range(n_targets)], dtype=tf.int32)
        # Optional per-state sample weights (inverse class frequency) to handle imbalance.
        # Shape (8,); pass None to use unweighted mean.
        self.state_weights = (tf.constant(state_weights, dtype=tf.float32)
                              if state_weights is not None else None)

    def _step(self, x, y_dict, training):
        # self(x) returns raw (batch, 8) logits — do NOT change call() to return marginals
        state_logits = self(x, training=training)                   # (batch, 8)
        y_bin        = tf.cast(y_dict['y_binary'], tf.int32)        # (batch, 3)
        y_idx        = tf.reduce_sum(y_bin * self.powers, axis=-1)  # (batch,) ∈ [0..7]
        log_Z        = tf.reduce_logsumexp(state_logits, axis=-1)   # (batch,)
        true_score   = tf.gather(state_logits, y_idx, batch_dims=1) # (batch,)
        per_sample   = log_Z - true_score                           # (batch,)

        if self.state_weights is not None:
            w   = tf.gather(self.state_weights, y_idx)              # (batch,)
            nll = tf.reduce_sum(w * per_sample) / tf.reduce_sum(w)
        else:
            nll = tf.reduce_mean(per_sample)
        return nll

    def train_step(self, data):
        x, y_dict = data
        with tf.GradientTape() as tape:
            loss = self._step(x, y_dict, training=True)
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return {'loss': loss}

    def test_step(self, data):
        x, y_dict = data
        return {'loss': self._step(x, y_dict, training=False)}

    def predict_binary(self, x):
        logits     = self(x, training=False)                        # (batch, 8) — logits
        pred_state = tf.argmax(logits, axis=-1)                     # (batch,)
        return tf.gather(tf.cast(self.decode_mat, tf.int32), pred_state).numpy()

    def predict_marginals(self, x):
        logits      = self(x, training=False)                       # (batch, 8) — logits
        state_probs = tf.nn.softmax(logits, axis=-1)                # (batch, 8)
        return (state_probs @ self.decode_mat).numpy()              # (batch, 3)
```

**Computing `state_weights` from training data** (done once before constructing the model):

```python
import numpy as np

def compute_crf_state_weights(y_binary, n_targets=3):
    """Inverse-frequency weights over 2^n_targets states."""
    powers  = np.array([2**j for j in range(n_targets)], dtype=np.int32)
    y_idx   = (y_binary.astype(np.int32) @ powers)   # (N,) ∈ [0..7]
    counts  = np.bincount(y_idx, minlength=2**n_targets).astype(np.float32)
    counts  = np.where(counts == 0, 1.0, counts)      # avoid divide-by-zero for absent states
    weights = 1.0 / counts
    return (weights / weights.sum() * len(weights)).tolist()   # normalised so mean weight = 1
```

For this dataset (no ∅ state, single-target states 4× more frequent than multi-target), `state_weights` gives multi-target states roughly 4× the gradient contribution of single-target states. Pass these to `MultiLabelCRFModel(state_weights=compute_crf_state_weights(y_binary_train))`.

---

### Factory function pattern

The factory creates the backbone + one `Dense(8)` output, wrapped in `MultiLabelCRFModel`:

```python
def create_ml_cnn_gru_dual_crf_model(T, n_targets):
    cnn_out, gru_seq = _build_cnn_backbone_mtl(T)
    _, gru_out       = _build_gru_backbone_mtl(gru_seq)
    fused            = tf.keras.layers.Concatenate()([cnn_out, gru_out])
    fused            = tf.keras.layers.Dense(96, activation='relu')(fused)
    state_logits     = tf.keras.layers.Dense(2**n_targets)(fused)   # (batch, 8)
    model            = MultiLabelCRFModel(inputs=..., outputs=state_logits, n_targets=n_targets)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    return model
```

Note: the `state_logits` output is **not** sigmoid-activated. The wrapper class handles loss and decoding internally.

---

### Integration with evaluate_outlier_filters_ml

The CRF model's `call()` / `self(x)` **must return raw `(batch, 8)` logits** — not marginals. This is a correctness requirement, not a style choice.

If `call()` were changed to return marginals and `_step` contains `state_logits = self(x, training=training)`, then `state_logits` would silently be marginals (probabilities in `[0,1]`), and `tf.reduce_logsumexp` would compute the partition function of the *probabilities* rather than the *logits* — producing garbage gradients with no error message.

**Correct integration strategy: `_CRF_ML_KEYS` frozenset** (analogous to `_RCFD_ML_KEYS`)

Add `_CRF_ML_KEYS` to `model_utils_multilabel.py`. The evaluation loop branches on `is_crf = m in _CRF_ML_KEYS`:
- Predictions: `model.predict_binary(x_val)` — argmax over 8 states → `(batch, 3)` binary
- Probabilities: `model.predict_marginals(x_val)` → `(batch, 3)` per-label marginals

Both methods call `self(x)` internally and apply softmax/argmax on the returned logits — this is safe because `call()` consistently returns logits.

The `(batch, 3)` marginals saved to `y_probs_{model}_` are fully compatible with `lab_threshold_search.ipynb`; see the threshold section below.

---

### SupCon compatibility

The CRF output layer is orthogonal to the SupCon projection head. The embedding before `Dense(8)` is the same fused embedding used for SupCon projection. Combining:

```
L_total = (1 - λ_sc) × L_CRF_NLL + λ_sc × L_supcon
```

Override `_step` in a `MultiLabelCRFSupConModel(MultiLabelCRFModel)` subclass to add `_supcon_loss_jaccard` on the projection output, identically to `MultiLabelSupConModel`.

---

### What to expect

**Best case**:
- The partition function normalises over all 8 label combinations, so the model learns that e.g. `KPC+NDM+VIM` is intrinsically more probable than three independent positives would suggest (or less, if suppression is observed).
- Exact match accuracy improves because the CRF cannot predict inconsistent marginals — if `P(KPC)=0.9` and `P(NDM)=0.9`, the sigmoid model may still predict `[1,1,0]` if VIM marginal is 0.4, but the CRF jointly considers `state=3 (KPC+NDM)` vs `state=7 (all three)`.
- Gains most visible on cases where two targets co-amplify: the `KPC+NDM` state gets a learned boost from pairwise potential if this combination is common in training.

**Null result conditions**:
- If label combinations in the dataset are approximately consistent with independence (`P(KPC,NDM) ≈ P(KPC) × P(NDM)`), the pairwise potential converges to zero and the CRF collapses to softmax-with-argmax — equivalent to the current model with a fixed threshold of argmax-over-8-states.
- With small datasets, the 8 states are unevenly represented (∅ might dominate). The NLL sees few examples of rare combinations; the learned pairwise potential is unreliable.

**Failure modes**:
- **Imbalanced states**: this dataset has no ∅ class, but single-target states are ~4× more frequent than multi-target states. Without `state_weights`, the NLL gradient for `KPC+NDM+VIM` (5.9% of data) is negligible compared to single-target states (~76%). Use `compute_crf_state_weights` from the class structure above to rebalance.
- **Saving probabilities**: marginals from `predict_marginals()` are `(batch, 3)` float values in `[0, 1]` — slot directly into `y_probs_{model}_` storage unchanged. The downstream evaluation code is unaffected.

**Threshold tuning is fully supported via marginals** (this is a strength, not a limitation):

The MAP inference (`predict_binary` / argmax over 8 states) is one inference mode, but the per-label marginals `state_probs @ decode_mat` give per-label probabilities that are fully compatible with `lab_threshold_search.ipynb`. Thresholding the KPC marginal at 0.3 instead of 0.5 works exactly as with independent sigmoids — except the marginal probability of KPC was computed while being *jointly aware* of NDM and VIM probabilities. The CRF provides the joint learning benefit; the marginals give back the per-label threshold flexibility. There is no trade-off between the two.

---

### Considerations common to both approaches

#### Data requirements

| Approach | Extra data needed |
|---|---|
| AuxCt head | Per-sample per-target Ct values (cycle at threshold crossing). Can be sentinel-masked if unavailable. |
| CRF | None — uses same binary label vector. |

Ct values may already be computed by the preprocessing pipeline (`01b_lab_curve_preprocessing.py`). Check whether `y_concentration` (passed to RCFD) can be extended to a `(N, n_targets)` per-target Ct matrix.

#### Model key naming convention

Follow existing patterns:

```
# AuxCt examples
'cnn_gru_dual_auxct'
'cnn_gru_dual_auxct_supcon'
'cnn_gru_dual_auxct_supcon2'
'cnn_gru_dual_auxct_supcon3'
'cnn_trans_dual_auxct'
...

# CRF examples
'cnn_gru_dual_crf'
'cnn_gru_dual_crf_supcon'
'cnn_trans_dual_crf'
...
```

SupCon suffix for CRF follows the same `_supcon` / `_supcon2` / `_supcon3` pattern (no `_mtl` suffix, since CRF is not RCFD-based).

#### Training stability

- **AuxCt**: gradient scale of MSE depends on Ct normalisation. If Ct values are raw (15–40 cycles), MSE ≈ 100–900× BCE. Normalise to `[0,1]` or standardise per fold before computing L_reg.
- **CRF**: NLL loss magnitude is `log(8) ≈ 2.08` at initialisation (uniform prior), comparable to `3 × BCE` at initialisation. No special rescaling needed. Monitor that the partition function does not overflow — `reduce_logsumexp` handles this numerically.

#### Interaction with outlier filtering

`evaluate_outlier_filters_ml` applies `outlier_filters` (e.g., Z-score exclusion on LSTM-AE reconstruction error) before training each fold. Both AuxCt and CRF are unaffected — they operate on the same fold splits as existing models.

#### CV fold predictions and saved results

Both approaches produce `y_preds_{model}_` and `y_probs_{model}_` with the same `(n_folds, n_samples_per_fold, n_targets)` shape as existing models. No change to `ML_MODEL_KEY_MAP` structure needed — just add new entries for the new model keys.

---

### Priority recommendation

| | AuxCt | CRF |
|---|---|---|
| **Implementation effort** | Medium (need `ct_values` data plumbing + new model class) | Low (new wrapper + `Dense(8)` factory) |
| **Data dependency** | Requires per-target Ct annotations | None |
| **Expected gain** | Low–medium; primarily helps backbone feature quality | Low–medium; helps exact match on co-amplification states |
| **Risk** | Sentinel masking complexity; Ct noise in multi-positive wells | Imbalanced states; no threshold flexibility |
| **Try first** | After verifying Ct annotations exist and are reliable | Yes — implement as a quick experiment alongside a baseline |

**Suggested order**: Implement CRF first (no new data needed, low effort), evaluate on existing results, then add AuxCt once Ct data quality is confirmed.
