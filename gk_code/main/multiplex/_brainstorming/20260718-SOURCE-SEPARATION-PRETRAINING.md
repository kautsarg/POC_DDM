# Source Separation Pretraining for Multiplex PCR Classification

**Date**: 2026-07-18  
**Type**: Ideation + Critical Analysis + Implementation Plan  
**Dataset**: `LAB_Multiplex/02_ACA_qdPCR_balanced` (N=10,383, T=45, balanced subset)  
**Context**: `(N=10383, T=45)` normalised dPCR curves → `{0,1}^3` for `[KPC, NDM, VIM]`.  
Single-target: ~4,059 samples. Multi-positive: ~6,324 across 4 combination types. No all-negative samples.  
Note: an unbalanced full dataset with ~26,571 wells exists but is NOT used for training — the balanced subset is the canonical training set.

---

## 1. The Problem This Addresses

Standard multi-label BCE treats each label as independent. The physics violates this: a KPC+NDM co-amplified curve is not two independent signals stacked in a feature matrix — it is the *superposition* of two PCR kinetics running in the same reaction vessel, competing for the same reagents, measured on the same fluorescence channel. The classifier must infer two separate Ct values, two growth rates, and one overall saturation from one entangled 45-point sequence.

Existing mitigations in this codebase:
- **Cross-attn head**: `n_targets` learned query vectors attend over the pre-pooling GRU sequence.
- **SupCon**: pushes embeddings with different label sets apart in metric space.
- **RCFD**: conditions the backbone on bulk concentration.

None of these explicitly forces the encoder to learn individual kinetic components. This is the hypothesis behind source separation pretraining.

---

## 2. The Proposed Approach

Inspired by **Blind Source Separation** ("Cocktail Party Problem") and **Compositional Representation Learning**.

### 2.1 Core idea

Instead of training a classifier directly, first train an **Encoder–Decoder** that takes any PCR curve (single or co-amplified) and **dismantles** it into `n_targets = 3` separate reconstructed channels. Channel `j` must reconstruct the shape of target `j` if target `j` is present, or output a flatline if absent.

```
Input: curve (T,)  ← any label combination
  │
Encoder → z (d_latent,)   ← kinetic summary
  │
Decoder → [ŷ_KPC (T,), ŷ_NDM (T,), ŷ_VIM (T,)]
```

For a KPC+NDM input, the decoder should output:
```
ŷ_KPC ≈ reference KPC curve
ŷ_NDM ≈ reference NDM curve
ŷ_VIM ≈ flatline (0)
```

### 2.2 No synthetic data — use real co-amplified curves directly

**The synthetic data trap**: generating synthetic co-amplified curves requires calibrated `α·A + β·B` mixing weights. Without fitting those weights to real co-amplified data, the synthetic curves may not be physically realistic. Using hallucinated curves as Phase 1 supervision would teach the encoder the wrong physics.

**The real-data alternative**: We already have ~6.3K real co-amplified curves and ~20K single-target curves. The Phase 1 loss can be constructed entirely from these, using three complementary signals:

#### Signal 1 — Absent-channel suppression (fully supervised)

For any curve with known absent labels, the decoder channel for absent target `j` is supervised directly toward zero:

```
L_absent = Σ_{j: y_j=0}  MSE(ŷ_j,  0)
```

This is zero-variance supervision — we know *exactly* what the absent channels should look like. Works for both single-target and multi-positive curves.

#### Signal 2 — Reconstruction consistency (unsupervised, but physically constrained)

For a co-amplified curve, the sum of the active decoder channels must reconstruct the input. We do not know the individual components, but we know their sum:

```
L_consist = MSE(input_curve,  Σ_{j: y_j=1}  ŷ_j)
```

For single-target curves this collapses to standard reconstruction (one active channel).  
For multi-positive curves this enforces decomposition additivity without requiring ground-truth components.

#### Signal 3 — Per-target reference anchoring from single-target curves

This is the key insight: **each decoded component should look like a real single-target curve for that target.** The 20K single-target curves are a reference distribution. For each active channel `j` in a multi-positive curve, the decoded component `ŷ_j` should be indistinguishable from a genuine single-target-`j` curve.

Concretely:

**Chosen approach: nearest-neighbour anchoring (concentration-free)**

For each decoded component `ŷ_j`, find the `k` nearest single-target-`j` curves in
curve space (L2 distance on rendered curves), and minimise the average distance:

```
refs_j = k nearest single-target-j curves to ŷ_j (by curve-space L2)
L_anchor_j = MSE(ŷ_j,  mean(refs_j))
```

Stop-gradient on `refs_j` — gradients flow only through `ŷ_j`. The reference bank
is a `tf.constant` of all single-target-j curves, built once from `X_curves` + `y_binary`
(no concentration needed). `k=5` default.

```python
diffs   = tf.expand_dims(rendered_j, 1) - ref_bank_j   # (batch, N_ref, T)
dists   = tf.reduce_sum(diffs ** 2, axis=-1)            # (batch, N_ref)
_, idx  = tf.math.top_k(-dists, k=k)
anchor  = tf.reduce_mean(tf.gather(ref_bank_j, idx), axis=1)  # (batch, T)
L_anchor_j = tf.reduce_mean((rendered_j - anchor) ** 2)
```

**Why not concentration-prototype:** prototype anchoring requires per-well concentration
at training time (to select the bin). NN anchoring is concentration-free — the reference
is selected by shape similarity to the decoded output, making it applicable regardless of
whether concentration is available.

| | Prototype | NN (chosen) |
|---|---|---|
| Requires `concentration` at training time? | Yes | **No** |
| Reference selection | `bin(log10(Conc))` | k-NN by curve L2 |
| Circularity risk | None | Present (stop-gradient mitigates) |

**Combined Phase 1 loss**:

```
L_phase1 = L_absent
          + λ_cons  × L_consist
          + λ_anch  × Σ_{j: y_j=1} L_anchor_j
```

Suggested initial weights: `λ_cons = 1.0`, `λ_anch = 0.5`.

---

## 3. Data Available (`02_ACA_qdPCR_balanced`)

| Well type | N | Supervisory signal |
|---|---|---|
| Single KPC | 875 | L_absent (NDM,VIM→0) + L_consist (KPC channel = input) + L_anchor_KPC (= input) |
| Single NDM | 1,643 | same pattern for NDM |
| Single VIM | 1,541 | same pattern for VIM |
| KPC+NDM | 1,638 | L_absent (VIM→0) + L_consist (KPC+NDM sum = input) + L_anchor per active channel |
| KPC+VIM | 1,582 | same pattern |
| NDM+VIM | 1,535 | same pattern |
| KPC+NDM+VIM | 1,569 | L_absent (all three active) + L_consist (all three sum = input) + L_anchor per active |
| **Total** | **10,383** | — |

**Important imbalance**: unlike the full dataset, in the balanced subset **multi-positive (6,324) outnumbers single-target (4,059)**. The nearest-neighbour anchor bank for KPC has only 875 single-target references — a much sparser bank than originally assumed. Consider using the full unbalanced dataset (~26,571 wells) for Phase 1 pretraining only, since Phase 1 requires no balanced labels.

**Ct vs. concentration**: Sigmoid params (`sigmoid_curves['original']['params'][:, 3]`) give per-well Ct. Ct correlates strongly (r < −0.9) with log10(Conc) for each single-target group. The `Conc` column is used only to build the prototype bank offline — it is **not** an encoder input; the model sees only the raw time series `(T, 1)`.

---

## 4. Architecture

### 4.1 Structured latent space (per-target blocks)

Split the bottleneck into per-target subspaces plus a shared block:

```
Encoder → [z_shared (d_s,),  z_KPC (d_t,),  z_NDM (d_t,),  z_VIM (d_t,)]
              │                   │               │               │
         Decoder_KPC(z_sh, z_KPC)   Decoder_NDM(z_sh, z_NDM)   Decoder_VIM(z_sh, z_VIM)
```

- `z_shared` captures curve-wide properties (plateau height, baseline).
- `z_j` captures target-`j`-specific kinetics (Ct, slope).
- Decoder for channel `j` reads **only** `z_shared` and `z_j` — structurally prevents NDM information from leaking into the KPC decoder, and vice versa.

For classification: `[z_shared, z_KPC, z_NDM, z_VIM]` concatenated → `Dense(32, relu)` → `Dense(3, sigmoid)`.

### 4.2 Parametric decoder (preferred)

Rather than reconstructing the full 45-point curve, predict 5 sigmoid parameters per channel: `(Fm, Fb, Sc, Ct, shape)`. These are already precomputed in `sigmoid_curves['params']` for all 26,571 wells.

```
z_sh, z_j  →  Dense(32, relu)  →  5-head output with constrained activations
```

**Critical: output activations must enforce physical constraints (Trap 1)**

A raw `Dense(5)` allows negative outputs. Under the reconstruction consistency loss, the network will exploit negative `Fm` or `Sc` to cancel errors across channels (e.g., decoder_KPC predicts `Fm = -0.3` to partially subtract a poor decoder_NDM estimate). This must be forbidden:

| Parameter | Activation | Reason |
|---|---|---|
| `Fm` | `softplus` | Fluorescence amplitude is strictly positive |
| `Sc` | `softplus` | Slope (amplification rate) is strictly positive |
| `Ct` | `sigmoid * T_max` | Cycle threshold must lie in `[0, T]`; free linear allows Ct=200 |
| `Fb` | linear | Baseline can be small negative (noise below zero) — but clip to `[-0.05, 0.2]` |
| `shape` | `softplus + 1` | Shape exponent must be ≥ 1 |

`softplus(x) = log(1 + exp(x))` — smooth, always positive, non-zero gradient everywhere (unlike `relu`).

**L_anchor and L_consist are both computed in rendered curve space, not parameter space (Trap 3)**

Computing MSE directly on the 5 parameters creates a scale imbalance: in this dataset, `Ct ∈ [8, 25]`, `Sc ∈ [0.15, 0.25]`, `Fm ∈ [0.9, 1.0]`. Unweighted MSE is dominated by Ct. Per-parameter normalisation is possible but adds tuning complexity.

The cleaner fix: **render all decoded params into 45-point curves, then compute all losses in curve space**. This:
- Naturally weights parameters by their actual impact on the rendered curve (Ct errors cause a large shift; shape errors have small impact on a normalised curve)
- Unifies L_anchor and L_consist into the same curve-space MSE — one loss in one space
- Is robust to future changes in the parametric model without retuning weights

```
rendered_j   = render_sigmoid(decoded_params_j)    # (batch, T)
rendered_proto_j = render_sigmoid(prototype_params_j)  # (batch, T), pre-rendered and frozen

L_consist = MSE(input,  Σ_{j active} rendered_j)
L_anchor_j = MSE(rendered_j,  rendered_proto_j)
L_absent_j = MSE(rendered_j,  0)
```

**`render_sigmoid` must be implemented in pure TensorFlow (Trap 2)**

This function is on the gradient path for L_consist and L_anchor. Using NumPy silently breaks the computational graph — the decoder learns nothing from those losses.

```python
def render_sigmoid(params, T=45):
    # params: (batch, 5) — [Fm, Fb, Sc, Ct, shape]  (matches sigmoid_5p convention)
    Fm, Fb, Sc, Ct, sh = [params[:, i:i+1] for i in range(5)]
    t = tf.cast(tf.range(T), tf.float32)[tf.newaxis, :]   # (1, T)
    return Fb + Fm / (1.0 + tf.exp(-Sc * (t - Ct))) ** sh  # (batch, T)
```

`sigmoid_5p` in `main/utils/sigmoid_fitting.py` uses the 5-parameter logistic with a
shape/asymmetry exponent. The `** sh` term is required — omitting it produces a 4PL curve
that does not match stored `params`, making `L_anchor` meaningless.

Stored params path: `data['sigmoid_curves']['original']['params']` (shape `(N, 5)`).
Note the `'original'` level — `data['sigmoid_curves']['params']` does not exist.

Broadcasting: `params[:, i:i+1]` has shape `(batch, 1)`; `t` has shape `(1, T)` → output `(batch, T)`. This is a pure TF op and differentiable everywhere.

**Advantages of parametric decoder**: 15-float intermediate target instead of 135-point raw curve; Ct and Sc are directly regularisable; no Conv1DTranspose needed.

### 4.3 Encoder architecture

The existing CNN/GRU dual backbone is over-parameterised for Phase 1. A compact encoder appropriate for the 45-step curves:

```
inputs (batch, 45, 1)
  └─ Conv1D(16, 5, relu)
  └─ Conv1D(8, 3, relu, strides=2)           # → (batch, ~20, 8)
  └─ Bidirectional(GRU(16))                  # → (batch, 32)
  └─ Dense(d_shared + n_targets × d_t, relu)
  └─ Reshape → [z_shared, z_KPC, z_NDM, z_VIM]
```

Suggested: `d_shared = 12`, `d_t = 8` → 36-dim total latent. Larger than the 16–32 proposed originally, correctly reflecting the 15-dim minimum needed just for the parametric description of a triple-positive curve.

### 4.4 Full revised architecture

```
 ─────────────── PHASE 1 ─────────────────────────────────────────────────
 All 26,571 real curves (single-target + co-amplified)

 [Encoder: CNN + BiGRU → z_shared, z_KPC, z_NDM, z_VIM]
                │
   ┌────────────┼───────────────────────────────┐
   │            │                               │
 Decoder_KPC  Decoder_NDM                  Decoder_VIM
 (z_sh, z_KPC) (z_sh, z_NDM)              (z_sh, z_VIM)
   │            │                               │
 5 params     5 params                        5 params
                │
 Loss:
   L_absent   — absent channels → [0,0,0,0,0]
   L_consist  — render params as curves, sum(active) ≈ input
   L_anchor   — active channel params close to prototype_j(Conc)
 ─────────────────────────────────────────────────────────────────────────

 ─────────────── PHASE 2 ─────────────────────────────────────────────────
 FREEZE Encoder.
 Attach: [z_sh, z_KPC, z_NDM, z_VIM] → Dense(32) → Dense(3, sigmoid)
 Train on all 26,571 curves with BCE.
 ─────────────────────────────────────────────────────────────────────────

 ─────────────── PHASE 3 ─────────────────────────────────────────────────
 UNFREEZE Encoder.  lr ≈ 1e-5.
 Train Encoder + Classification head jointly with BCE.
 ─────────────────────────────────────────────────────────────────────────
```

---

## 5. Critical Challenges

### 5.1 Reconstruction consistency is underdetermined for multi-positive curves

`MSE(decoder_KPC + decoder_NDM, input)` has infinitely many solutions. Without the anchor loss, the network exploits the most available degree of freedom: **negative parameters**. If decoder_KPC overshoots, decoder_NDM can cancel the error with a negative Fm. This is Trap 1, and it is solved structurally by the constrained output activations in Section 4.2 — once `softplus` is applied to Fm and Sc, negative component curves are impossible at the architectural level, not just discouraged by the loss.

After enforcing positivity, the remaining degeneracy is symmetric energy assignment (all signal to one channel). L_anchor is the only symmetry-breaker. Its weight `λ_anch` is the key hyperparameter: too low → components collapse; too high → decoder memorises prototypes and ignores the actual input.

### 5.2 Concentration information is available for all wells, but not per-target within a well

For a KPC+NDM well at `Conc = 50,000`, we don't know if KPC and NDM each had `Conc = 25,000` (equal split) or some other ratio. The prototype anchoring assumes each active target contributes roughly `Conc / n_active` — a simplification.

**Mitigation**: The reconstruction consistency loss partially compensates — the network must still decompose the signal additively even when one component dominates. The prototype anchor provides a shape prior that prevents the decoder from assigning all energy to one channel.

### 5.3 Disentanglement cannot be proven — only measured

Even with per-target latent blocks, the encoder may learn `z_j` that encodes "is target `j` present?" (a binary flag) rather than the actual kinetics. In that case Phase 2 is just a lookup table on the binary flags, not a genuine kinetic representation.

**Measurement**: After Phase 1, check Spearman correlation of `z_KPC[0]` (first dim of the KPC subspace) against the known Ct of single-target KPC wells. If `|r| > 0.7`, the subspace is encoding real kinetics.

### 5.4 Phase 3 may erase Phase 1 if BCE gradient dominates

BCE on balanced labels (75% single-target, 25% multi-positive) will push the classifier head quickly in Phase 2. After unfreezing in Phase 3, the BCE gradient can overwrite Phase 1 features if `lr` is not very small.

**Mitigation**: Optionally add a reconstruction regulariser during Phase 3 (`L_BCE + μ × L_consist`) to prevent catastrophic forgetting of kinetic features.

### 5.5 Same-Ct ambiguity — the fundamental limit of this approach

If KPC and NDM co-amplify at nearly the same Ct (e.g., both at cycle 17), their individual sigmoid shapes are virtually identical. The decoder has no curve-shape basis to decide which component goes to channel KPC and which to channel NDM. L_absent and L_consist will be satisfied by any symmetric split (e.g., each channel gets half the input). L_anchor is the only signal that can break this symmetry — but only if the single-target KPC and NDM prototypes at the same concentration are distinguishable in curve shape (they will be if their slopes or plateau heights differ, but the differences may be small for this assay).

This is the **fundamental ambiguity limit** of BSS when two sources have the same Ct. It is not a bug in the implementation — it is an irreducible physical limitation. The model will do no worse than random 50/50 channel assignment in this case, which is equivalent to the standard sigmoid classifier (both targets present → predict both). But it will not improve over the baseline for this specific hard case.

**Measurement**: After Phase 1, plot decoded component Ct values for KPC+NDM co-amplified wells. If decoder_KPC Ct and decoder_NDM Ct are consistently close (< 2 cycles apart), the model is ambiguous on this case. Check whether the 10-fold CV results show improvement specifically on same-Ct co-amplifications vs. cross-target cases.

### 5.6 Three-phase training inside CV is expensive

Each fold requires a complete Phase 1 run. Phase 1 cannot share weights across folds without leaking test data. For 10-fold CV, this is 10 × Phase 1 training runs.

**Mitigation**: Phase 1 can be run with early stopping on reconstruction loss (not classification), which typically converges faster. Alternatively, run Phase 1 on the full dataset once (no classification labels used), accepting that test curves are seen by the encoder — this is the same assumption as normalising features on the full dataset, which is common practice.

---

## 6. Implementation Plan

**Session constraint**: all new code lives exclusively in `main/multiplex/` and `main/multiplex/utils/`.

### 6.1 New file: `main/multiplex/utils/model_training/model_utils_source_sep.py`

- `build_source_sep_encoder(T, d_shared, d_target, n_targets)` → Keras functional model
- `build_parametric_decoder(d_shared, d_target, n_params=5)` → MLP per channel
- `build_source_sep_full_model(T, ...)` → encoder + 3 decoders
- `MultiLabelSourceSepPhase1Model(tf.keras.Model)` — custom `train_step` with `L_absent + L_consist + L_anchor`
- `build_source_sep_classifier(encoder, d_shared, d_target, n_targets)` → Phase 2+3 head
- `render_sigmoid(params, T=45)` — **pure TF ops only** (no NumPy): `Fb + Fm / (1 + exp(-Sc*(t-Ct)))`, returns `(batch, T)`; on the gradient path for both L_consist and L_anchor, so must build a differentiable computational graph

### 6.2 New file: `main/multiplex/utils/model_training/nn_anchor_bank.py`

- `build_nn_anchor_bank(X_curves, y_binary, n_targets=3)` — extracts all single-target-j
  curves for each target j. Returns a list of n_targets arrays, each `(N_single_j, T)`.
- No concentration needed. Pure NumPy; called once before training.
- Result is passed to `MultiLabelSourceSepPhase1Model` as a list of `tf.constant` tensors.
- Placed in `utils/model_training/` (no new directory needed — `utils/data/` does not exist).

No `synthetic_augmentation.py` needed.

### 6.3 New script: `main/multiplex/03b_source_sep_pretraining.py`

Standalone Phase 1 training. Outputs `source_sep_encoder_weights.h5` per experiment folder.

```
--exp_folder
--task_id
--d_shared     default=12
--d_target     default=8
--epochs_p1    default=150
--lambda_cons  default=1.0
--lambda_anch  default=0.5
--nn_k         default=5    (k nearest neighbours for L_anchor)
--validate     flag: print per-channel Ct Spearman r after Phase 1
```

### 6.4 Changes to `03_main_training.py`

- Add `--source_sep` flag: loads frozen encoder from `source_sep_encoder_weights.h5`, attaches classification head.
- Phase 2: encoder frozen, `lr = 1e-3`.
- Phase 3: encoder unfrozen, `lr = 1e-5`. Optional `--phase3_lambda_consist` for reconstruction regularisation.

### 6.5 Changes to `config_multiplex.py`

```python
RESULT_FILE_BY_FLAG = {
    ...
    'source_sep': 'classification_performances_ml_source_sep.joblib',
}
```

### 6.6 New model keys in `model_utils_multilabel.py`

```python
'cnn_gru_source_sep'
'cnn_gru_source_sep_supcon'     # SupCon on [z_sh, z_KPC, z_NDM, z_VIM]
```

---

## 7. What to Expect

### 7.1 If Phase 1 learns kinetic structure

- `z_j` encodes Ct of target `j` (measurable by Spearman correlation on single-target wells).
- Phase 2 classification head achieves competitive BCE with fewer epochs (the latent is already structured).
- Phase 3 fine-tuning is small (encoder already near-optimal for classification).
- Improvement concentrated on same-Ct co-amplification: two targets with similar Ct are hardest for standard models; source sep pretraining gives the encoder a prior on "they have separate kinetics even when Ct is close."

### 7.2 Null / failure modes

| Failure | Diagnostic | Fix |
|---|---|---|
| Phase 1 `L_consist` stays high | Decoder cannot sum to input | Increase encoder/decoder capacity |
| Per-channel Spearman `|r|` < 0.4 | `z_j` not encoding Ct | Upweight `L_anchor`; switch to nearest-neighbour anchoring |
| Phase 3 BCE gain is zero vs frozen Phase 2 | Phase 1 already found classification-optimal features | Skip Phase 3; freeze encoder permanently |
| Phase 3 erases Phase 1 | Reconstruction degrades abruptly | Add `L_consist` auxiliary term during Phase 3 |
| Single-target performance degrades vs baseline | Phase 1 over-constrains encoder | Reduce `λ_anch`; increase `d_shared` |

---

## 8. Comparison to Other Approaches in This Codebase

| Approach | Mechanism | Relationship to Source Sep |
|---|---|---|
| Cross-attn head | Per-label query attends over GRU sequence | Inference-time per-label features; no pretraining — source sep provides a complementary *training-time* prior |
| RCFD | FiLM conditioning on total concentration | Uses concentration to modulate the backbone; source sep uses concentration to anchor the prototype bank |
| AuxCt Head (6a) | Secondary Ct regression output | Similar auxiliary signal but no structural disentanglement |
| SupCon SC1–3 | Metric geometry of full embedding | Shaping the joint embedding space vs. shaping per-target subspaces |
| CRF (1c) | Joint label scoring | Output-side structured prediction; orthogonal — CRF could be applied on top of source sep classification head |

---

## 9. Validation Gates

Run in sequence before full CV training. Do not skip.

1. **Phase 1 single-target reconstruction check**: Overfit Phase 1 on 100 single-target curves. Each curve should reconstruct with RMSE < 0.02 on the active channel and < 0.005 on absent channels. If not — architecture bug.

2. **Per-target Ct Spearman check**: After Phase 1 on full training data, compute Spearman `r` between `z_j[0]` and Ct for single-target-`j` wells. Target `|r| > 0.65`. If not — `L_anchor` is too weak or `d_target` too small.

3. **Multi-positive consistency check**: Sample 20 co-amplified curves. For each, render the decoded component curves and plot: input vs. `Σ active decoder channels`. Should look like a reasonable decomposition visually.

---

## 10. Priority

Implement CRF (1c) first (no new data plumbing, low effort). Then source sep, gated on the validation steps above.  
Estimated effort for source sep: ~3–4 days including all three validation gates.
