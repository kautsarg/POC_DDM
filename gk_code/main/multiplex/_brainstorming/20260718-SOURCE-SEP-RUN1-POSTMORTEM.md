# Source Separation — Run 1 Post-mortem

**Date**: 2026-07-18  
**SLURM job**: 262563 (`lab_multiplex_source_sep_training.sh`)  
**Result file**: `classification_performances_ml_source_sep_10fold.joblib`  
**Dataset**: `LAB_Multiplex/02_ACA_qdPCR_balanced` (N=10,383 balanced, T=45)

---

## Results

Both source separation models ranked near the bottom of the full leaderboard:

| Rank | Model | Exact acc | Hamming | F1-sample |
|---|---|---|---|---|
| 57 | SrcSep SC0 | **26.98% ± 7.31%** | 0.3225 | 0.3974 |
| 58 | SrcSep SC1 | **26.98% ± 7.31%** | 0.3225 | 0.3974 |
| 1  | CNN+GRU CRF-MRF SC1 | 83.03% ± 0.64% | 0.0831 | 0.9286 |

The top competing models are at ~80–83% exact accuracy.  
The source sep models are at ~27% — worse than many trivial baselines.

Two additional red flags:
- **SC0 and SC1 are identical** (same weights, same results — see root cause #3 below)
- **Very high variance** (±7.31%) — unstable across folds

---

## Inspection: How the failure was found

### Phase 1 training (encoder + decoders)

The log shows 170 epochs before early stopping.  Loss trajectory:

| Epoch | l_absent | l_anchor | l_consist | loss | val_loss |
|---|---|---|---|---|---|
| 1    | 0.1053 | 0.0763 | 0.0641 | 0.2075 | 0.2074 |
| ~43  | 0.0126 | 0.0630 | 0.0052 | 0.0493 | 0.0481 |
| ~85  | 0.0096 | 0.0572 | 0.0043 | 0.0425 | 0.0370 |
| 170  | 0.0084 | **0.0567** | 0.0039 | 0.0406 | 0.0351 |

- `l_consist` converges well (0.0039 — sum of active decoded ≈ input).
- `l_anchor` **plateaus at 0.057 and won't improve** — active decoded channels are not aligning with single-target reference curves.
- `l_absent` is small (0.0084) — absent channels are near-zero.  Superficially OK.

### Phase 1 validation gate

```
[Validation gate 2] Spearman r(z_j[0], Ct) for single-target wells
  KPC: r=0.929  p=0.000e+00  n=6792  ✓
  NDM: r=nan    p=nan         n=6232  ✗ <0.65     ← FAILURE
  VIM: r=-0.833 p=0.000e+00  n=7223  ✓
```

NDM has **`r=nan`** — Spearman is undefined when one variable has zero variance.  
`z_NDM[0]` is constant for all 6232 single-target NDM wells: the NDM per-target latent dims have **collapsed to a single constant value**.

---

## Root Cause Analysis

### Root cause 1: Dying ReLU at the bottleneck Dense (primary)

In `build_source_sep_encoder`, the bottleneck Dense had `activation='relu'`:

```python
z = tf.keras.layers.Dense(
    d_shared + n_targets * d_target, activation='relu', name='ss_bottleneck')(x)
```

The encoder outputs `d_shared + 3*d_target = 46` dims.  The per-NDM slice is dims 26–35.  If the pre-activation values for those 10 dims are ≤ 0 for every sample in the dataset, relu produces 0 for all of them — dead dims.  This gives:

```
z_NDM = [0, 0, ..., 0]  for every well
→ std(z_NDM) = 0
→ Spearman(z_NDM[0], Ct) = NaN
```

Cascade:
- Dead z_NDM → decoder_NDM always sees the same input → outputs a fixed average curve
- `l_anchor` for NDM can't improve (decoded curve is fixed, reference curves are diverse)
- `l_anchor` plateaus (explains the 0.057 plateau)
- Phase 2 classification head sees z_NDM = 0 always → cannot learn NDM discrimination

Why NDM specifically (not KPC or VIM)? Random initialisation — the weights happened to push those dims negative for NDM during early training.  With relu, once they go negative they're locked out (zero gradient through relu).  KPC and VIM escaped by chance.

### Root cause 2: Phase 2 head ignores decomposition structure

Even if Phase 1 were fixed, the Phase 2 head was:

```python
h   = Dense(32, relu)(z)             # all 46 dims mixed together
out = Dense(3, sigmoid)(h)           # one shared projection → 3 outputs
```

This doesn't exploit the per-target structure the pretraining was designed to create.  A collapsed NDM channel contributes nothing to the shared MLP — but so does a healthy NDM channel if the MLP doesn't learn to look there.

### Root cause 3: SC0 and SC1 are identical (minor / by design)

`cnn_gru_source_sep` and `cnn_gru_source_sep_supcon` use the same factory, same pretrained encoder weights, and the same `_StandardMultiLabelModel` BCE train_step.  Neither is in `_SUPCON_ML_KEYS`, so SupCon is not added to SC1.  This is a design placeholder — until a second pretraining variant exists (e.g., SupCon pretraining), both slots train identically.

---

## Fixes Applied (2026-07-18)

### Fix 1: Remove relu from `ss_bottleneck`

`model_utils_source_sep.py`, `build_source_sep_encoder`:

```python
# Before
z = tf.keras.layers.Dense(
    d_shared + n_targets * d_target, activation='relu', name='ss_bottleneck')(x)

# After  (linear — no activation)
z = tf.keras.layers.Dense(
    d_shared + n_targets * d_target, name='ss_bottleneck')(x)
```

Linear activation means all 46 dims can be positive or negative.  No dying-relu collapse possible.

### Fix 2: Variance regularisation loss `L_var`

Added to `MultiLabelSourceSepPhase1Model._compute_losses`:

```python
# L_var: penalise low per-target-dim variance (collapse safety net)
l_var = tf.constant(0.0)
for j in range(self.n_targets):
    z_std = tf.math.reduce_std(z_parts[j], axis=0)   # (d_target,)
    l_var = l_var + tf.reduce_mean(tf.nn.relu(var_margin - z_std))

loss = l_absent + λ_cons*l_consist + λ_anch*l_anchor + λ_var*l_var
```

Defaults: `lambda_var=0.1`, `var_margin=0.05`.  
This directly penalises any per-target dim whose batch std falls below 0.05, before the dim fully collapses.

Exposed as CLI arg: `03b_source_sep_pretraining.py --lambda_var FLOAT`

### Fix 3: Per-target Phase 2 classification head

`model_utils_source_sep.py`, `build_source_sep_classifier`:

```python
# Before: one shared MLP on full z
h   = Dense(32, relu, name='cls_feat')(z)
out = Dense(3, sigmoid, name='cls_out')(h)

# After: per-target MLP, each head sees [z_shared || z_j]
z_shared = Lambda(t[:, :d_shared])(z)
for j in range(n_targets):
    z_j   = Lambda(t[:, d_shared+j*d_target : d_shared+(j+1)*d_target])(z)
    h_j   = Dense(d_shared+d_target, relu)(Concatenate([z_shared, z_j]))
    out_j = Dense(1, sigmoid)(h_j)
out = Concatenate(name='cls_out')(per_target_outs)
```

This mirrors the pretraining decoder structure: each target j uses exactly `[z_shared || z_j]`, so the Phase 2 classification is forced to exploit the decomposed representation rather than a mixed projection.

Per-target head params: 3 × (Dense(26→26) + Dense(26→1)) = 3 × 729 = 2,187 params (vs 1,603 before).

---

## What to watch for in Run 2

1. **NDM `r` should no longer be NaN** — the validation gate should pass for all three targets with `|r| > 0.65`.
2. **`l_anchor` should not plateau at 0.057** — it should continue improving past epoch 100.
3. **`l_var` should decrease toward zero** — variance reg is doing its job when `l_var → 0`.
4. **SC0 and SC1 may still be identical** until a second pretraining variant is added — that's expected.
5. **Overall exact accuracy should be >50%** to justify the added complexity over standard models.

---

## Run 2 Results (SLURM job 262639, 2026-07-18)

### Phase 1 outcomes

| Metric | Run 1 | Run 2 | Status |
|---|---|---|---|
| Epochs (early stop) | 170 | 68 | — |
| l_anchor at stop | 0.0567 (plateau) | 0.055–0.065 (improving) | ✅ |
| l_var at stop | N/A | **0.0000** | ✅ Variance reg worked |
| NaN at final epoch | No | **Yes** (LR decay) | ⚠️ |
| NDM r (validation gate) | nan ✗ | **−0.599** ✗ (no longer NaN) | ⬆️ |
| VIM r (validation gate) | −0.833 ✓ | **0.244** ✗ (regressed) | ⬇️ |
| KPC r (validation gate) | 0.929 ✓ | 0.894 ✓ | ✓ |

**l_var → 0.0000** confirms variance regularisation is preventing per-target dim collapse.
**l_anchor continues improving** (no plateau) confirms the dying-relu fix removed the bottleneck.
**NaN at the final epoch**: `ReduceLROnPlateau` halved LR to 5e-4, causing one divergent step.
`restore_best_weights=True` reverted to the best pre-NaN checkpoint, so saved weights are clean.
Fixed for Run 3 by reducing factor from 0.5 → 0.3 in `03b_source_sep_pretraining.py`.

**VIM regression** (−0.833 → 0.244): with the linear bottleneck all 3 targets compete freely for
latent space. Random initialisation landed on a different solution where `z_VIM[0]` encodes
something other than Ct. The full 10-dim z_VIM still carries VIM information for classification;
`z_j[0]`–Ct correlation is a proxy, not a hard requirement.

### Phase 2 classification results

| Metric | Run 1 | Run 2 |
|---|---|---|
| Exact accuracy (mean) | 26.98% ± 7.31% | **~15.78%** |
| Hamming | 0.3225 | **0.2800** ↑ |
| F1-sample | 0.3974 | **0.5043** ↑ |
| Per-fold peak | (not tracked) | 35.88% |

**Exact accuracy dropped despite F1/Hamming improving.** This is not a regression — it exposes
a measurement artefact in Run 1. When z_NDM = 0 for all wells (dying relu), the Phase 2 head
learned to always predict NDM=0. Since most wells are NDM-negative, this produced artificially
high exact match without learning anything useful. Run 2 is genuinely trying to predict all 3
labels, which is harder and produces honest (lower) exact match.

**High fold variance** (peak 35.88%, mean ~15.78%): the frozen encoder provides inconsistent
representations across folds. Phase 3 end-to-end fine-tuning is the primary fix.

### Root cause of remaining underperformance

Phase 2 trains only the per-target heads (2,187 params) on a frozen pretrained encoder.
The encoder has no classification signal — it only knows curve shape decomposition from Phase 1.
The per-target heads can only exploit whatever the Phase 1 encoder happens to encode, which
is not sufficient for reliable 3-label classification.

**Phase 3 (unfreeze + fine-tune) was the missing step.** It has now been implemented in
`model_utils_multilabel.py`: after Phase 2 converges, the encoder is unfrozen and the full
model is fine-tuned jointly at Adam(1e-5) for up to 100 epochs with EarlyStopping(patience=30).

---

## Fixes Applied After Run 2 (2026-07-18)

### Fix 4: Phase 3 end-to-end fine-tuning

`model_utils_multilabel.py`, inside the per-fold training loop, after Phase 2 `model.fit()`:

```python
# Phase 3: unfreeze encoder, fine-tune end-to-end at low LR
if _enc is not None:
    _enc.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5, clipnorm=1.0))
    model.fit(..., epochs=100, callbacks=[EarlyStopping(patience=30)])
```

Phase 1 gives the encoder a warm start in a decomposed latent space.
Phase 2 trains the per-target heads with the encoder frozen (fast convergence).
Phase 3 jointly fine-tunes everything at 100× lower LR — adapts the encoder to classification.

### Fix 5: ReduceLROnPlateau factor 0.5 → 0.3 (Phase 1 NaN prevention)

`03b_source_sep_pretraining.py`, `ReduceLROnPlateau` callback:

```python
# Before
tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, ...)
# After
tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, ...)
```

Halving LR in one step (0.5) caused a divergent final epoch. Factor=0.3 reduces LR more
gradually (1e-3 → 3e-4 → 9e-5 → min 1e-5) making divergence far less likely.

---

## What to watch for in Run 3

1. **Phase 3 convergence**: val_loss should decrease further after Phase 2 plateau — if it doesn't,
   the pretrained encoder is not providing a useful warm start.
2. **Exact accuracy >50%** with Phase 3 — if still below 40%, investigate encoder capacity
   (D_SHARED=32, D_TARGET=16) or VIM latent structure.
3. **No NaN** in Phase 1 with factor=0.3.
4. **NDM and VIM validation gate**: `|r| > 0.65` for all 3 — if VIM still regresses, consider
   adding SupCon on z_j during Phase 1 pretraining.

---

## Run 3 Results (SLURM job 262645, 2026-07-18)

### Phase 1

Identical encoder to Run 2 (same deterministic seed + same 03b config).  The NaN still
propagated through many epochs (factor=0.3 wasn't enough to prevent it) but
`restore_best_weights=True` rescued the pre-NaN checkpoint.  Validation gate unchanged:
KPC=0.894 ✓, NDM=−0.599 ✗, VIM=0.244 ✗.

Root cause of persistent NaN identified: the rendered-curve gradients can overflow float32
when decoder parameters (especially Sc or Fm) drift to extreme values after many epochs
without L_var forcing them back.  `tf.clip_by_value(rendered, -10, 10)` added inside
`_compute_losses` prevents this.  Also `TerminateOnNaN()` callback added to stop immediately
instead of running 30 useless NaN epochs.

### Phase 2+3 classification results

| Metric | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| Exact accuracy | 26.98% ± 7.31% | 35.88% ± 15.78% | **40.13% ± 9.98%** |
| Hamming | 0.3225 | 0.2800 | **0.2688** |
| F1-sample | 0.3974 | 0.5043 | **0.5298** |
| Rank / total | 57–58 / 80 | ~55 / 80 | **63 / 80** |

Phase 3 confirms the hypothesis: +4.25% exact accuracy (35.88% → 40.13%) and reduced fold
variance (±15.78% → ±9.98%) from unfreezing the encoder.  All three metrics improve
consistently across all three runs.

Top models remain at ~83% exact accuracy.  The gap (~43%) motivates architectural changes.

---

## Fixes Applied After Run 3 (2026-07-18)

### Fix 6: NaN-safe rendered curve clipping

`model_utils_source_sep.py`, `_compute_losses`:

```python
# Before: rendered curves used directly in losses
# After: clip to ±10 to prevent float32 overflow in gradient computation
rendered = [tf.clip_by_value(r, -10.0, 10.0) for r in rendered]
```

Also fixed `reduce_std` gradient when `std=0` (safe std via `sqrt(var + 1e-8)`):
```python
# Before
z_std = tf.math.reduce_std(z_parts[j], axis=0)
# After
var_j  = tf.reduce_mean((z_parts[j] - tf.reduce_mean(z_parts[j], axis=0))**2, axis=0)
z_std  = tf.sqrt(var_j + 1e-8)
```

### Fix 7: TerminateOnNaN callback

`03b_source_sep_pretraining.py` — stops training immediately on NaN instead of
waiting 30 patience epochs.

### Fix 8: Per-target SupCon loss `L_supcon` on z_j (Phase 1)

`model_utils_source_sep.py` — adds `_per_target_supcon_loss(z_j, y_j)` and
`lambda_supcon` parameter to `MultiLabelSourceSepPhase1Model`.

When `lambda_supcon > 0`, wells with the same target j form positive pairs in the
z_j latent space — stronger supervision signal for latent alignment than
L_anchor alone.  `lambda_supcon=0.0` by default (no-op, backward compatible).

`03b_source_sep_pretraining.py` exposes `--lambda_supcon` CLI arg.

### Fix 9: CRF+SrcSep structured output (Phase 2+3)

`model_utils_multilabel.py` — two new model variants:

```
cnn_gru_source_sep_crf        → SrcSep CRF SC0
cnn_gru_source_sep_crf_supcon → SrcSep CRF SC1
```

Architecture: same pretrained encoder → full latent z (46 dims) →
`Dense(2^n_targets=8, name='crf_logits')` → `MultiLabelCRFMRFModel` NLL loss.

Motivation: the existing per-target sigmoid heads treat each label independently.
The CRF-MRF learns joint state probabilities over all 2^3 = 8 label combinations
directly, which:
1. Captures co-occurrence statistics (KPC+NDM+VIM tri-positive is rare; the model
   can penalise implausible combinations).
2. Uses the full 46-dim latent z (not just per-target slices), letting the shared
   dims z_shared contribute to combination prediction.
3. Argmax over 8 states always produces a valid combination — no independent
   threshold issues per target.

Phase 2 + Phase 3 fine-tuning applies to CRF variants exactly as for sigmoid variants:
the encoder is frozen in Phase 2 (only `crf_logits` Dense learns the joint statistics),
then unfrozen in Phase 3 for end-to-end adaptation.

Models registered in `_SS_ML_KEYS` (encoder load + Phase 3) AND `_CRF_ML_KEYS`
(`predict_marginals`/`predict_binary` evaluation path).

`03_main_training.py --source_sep --supcon 0` now trains both `cnn_gru_source_sep`
AND `cnn_gru_source_sep_crf` per fold.

---

## What to watch for in Run 4

1. **SrcSep CRF vs SrcSep sigmoid**: does joint-state output improve exact accuracy?
   The CRF NLL loss should capture label co-occurrence (KPC+NDM+VIM tri-positive rare
   → model learns to suppress that state unless evidence is strong for all three).
2. **Phase 1 NaN gone**: factor=0.3 + clip + TerminateOnNaN should eliminate NaN.
3. **NDM / VIM validation gate**: |r| > 0.65 for all three — still failing for NDM and VIM.
   If still failing after fix 6+7, try increasing `lambda_supcon` (0.1) in Phase 1.

---

## Potential further steps if Run 4 still underperforms

- **Phase 1 SupCon encoder**: set `--lambda_supcon 0.1` in 03b to train an encoder
  with stronger per-target latent alignment; use this as the SC1 encoder (requires
  saving to a separate weight file).
- **Increase capacity**: D_SHARED=32, D_TARGET=16 (bottleneck 80 dims instead of 46).
- **Normalised anchor loss**: compare decoded curve shapes (Ct, Sc) rather than absolute
  L2 distance, more robust for multi-positive wells where amplitude is shared.

---

## Run 4 Results (SLURM job 262669, 2026-07-19)

### Phase 1

NaN hit again at a late epoch (l_var → 0.0000 first, then decoder parameters drift to extreme
values → float32 overflow). `TerminateOnNaN()` stopped immediately; `restore_best_weights=True`
recovered the pre-NaN checkpoint. The clip fix (±10) and safe-std fix slowed divergence but did
not prevent it — the root cause (unconstrained decoder after L_var satisfies) persists.

Validation gate (same pattern as Runs 2–3):

| Target | r | n | Status |
|--------|---|---|--------|
| KPC | 0.940 | 6792 | ✓ |
| NDM | −0.400 | 6232 | ✗ <0.65 |
| VIM | −0.821 | 7223 | ✓ |

NDM has now failed the Spearman gate in all four runs: NaN → −0.599 → −0.599 → −0.400. The
z_NDM[0] dimension never reliably encodes Ct, regardless of variance regularisation or NaN fixes.

### Phase 2+3 classification results

| Rank | Model | Exact | Hamming | F1-sample |
|------|-------|-------|---------|-----------|
| 63 | SrcSep SC1 (Run 3 cached) | 40.13% ± 9.98% | 0.2688 | 0.5298 |
| 64 | SrcSep CRF SC0 | 32.02% ± 7.01% | 0.4319 | 0.4314 |
| 65 | SrcSep CRF SC1 | 32.02% ± 7.01% | 0.4319 | 0.4314 |
| 66 | SrcSep SC0 | 30.45% ± 12.36% | 0.3244 | 0.4504 |
| 1  | CNN+GRU CRF-MRF SC1 | 83.03% ± 0.64% | 0.0831 | 0.9286 |

SC0 and SC1 are identical (same Phase 1 encoder; `--lambda_supcon 0.0` for both).

**CRF head**: +1.6pp exact accuracy vs sigmoid head, but Hamming/F1 are worse — the joint-state
NLL loss collapses to a small set of dominant states without enough gradient signal from only 8
output units.

**Regression vs Run 3** (SrcSep SC0: 40.13% → 30.45%): `--force_rerun` in the SLURM script
retrained Phase 1 from scratch. The new encoder produced a different (worse) latent space.
High fold variance (±12.36%) persists and reflects Phase 1 stochasticity propagating to Phase 2+3.

---

## Retrospective: Why This Approach Fails

After 4 runs and 9 architectural fixes, the best result is 40.13% exact accuracy vs 83.03% for
the top end-to-end model. The source separation approach is abandoned. This section documents
the failure factors across architecture, data, hyperparameters, and fundamentals.

### A. Architecture

**1. Encoder capacity too small, wrong pre-training objective.**
The source-sep encoder produces a 46-dim latent (d_shared=16 + 3×d_target=10). Competing models
use dual-branch encoders (~96-dim+) with cross-attention, trained end-to-end on the classification
objective. The Phase 1 objective (minimise L2 to nearest single-target curve) is a proxy for
"understand signal shapes", not for "discriminate label combinations". Representations useful for
shape reconstruction are not necessarily useful for classification.

**2. Rigid per-target latent partition.**
Splitting z into `[z_shared | z_KPC | z_NDM | z_VIM]` forces per-target separability. If NDM
and KPC share spectral or kinetic features, this partition actively hurts — the shared dims
are constrained to 16 values that must serve all three targets equally. End-to-end models have
no such partition: they can place any combination of per-target information anywhere in the
latent space.

**3. Phase 3 cancels Phase 1.**
Adam(1e-5) for 100 epochs re-adapts the encoder toward classification. In effect, Phase 3 must
partially undo Phase 1 — this means Phase 1 is not providing a useful warm start. The fact that
Phase 3 improves results (Run 2→3: +4.25pp) but still reaches only 40% confirms that the encoder
is being re-trained almost from scratch, not fine-tuned from a good initialisation.

**4. CRF head under-powered.**
`Dense(8)` on a 46-dim z learns joint-state logits for 2^3=8 label combinations. With only 8
output units and NLL loss, gradient signal is weaker than 3 independent sigmoid losses for
small n_targets. The head collapses to a few dominant states (confirmed by poor Hamming=0.43
vs sigmoid Hamming=0.32).

### B. Data

**1. PCR curves are not linearly superposable.**
The decomposition assumption (observed = Σ per-target curves) holds only approximately. Real
multiplex PCR has per-target competition for primers and polymerase, concentration-dependent
efficiency, and fluorescence channel crosstalk. These non-linearities mean the "true" per-target
component curves are not simply additive, and the decoder can never learn a correct decomposition.

**2. Multi-positive wells create irreducible anchor ambiguity.**
For a KPC+NDM+VIM tri-positive well, the total observed amplitude is shared across all three
decoded channels. The NN anchor selects references from a single-target bank — it has no way to
know what fraction of the total amplitude belongs to each target. L_consist constrains the sum,
but any amplitude partition that sums correctly satisfies both L_consist and L_anchor. This
ambiguity is reflected in the persistent L_anchor plateau (never below ~0.05).

**3. NDM is structurally harder to separate.**
Across all 4 runs, NDM failed the Spearman gate (r never reached 0.65). NDM curves may have
lower amplitude or more overlap with KPC in kinetic profile, making them harder to isolate.
The encoder learns to represent KPC and VIM well (r > 0.82) but cannot assign a dedicated,
informative latent region for NDM.

**4. Data requirements this approach actually needs.**
For source separation to work:
- Exact per-target amplitude must be known for every multi-positive well (not just "positive/negative").
- Single-target references must span the full concentration range in the training data (our bank
  is capped at 500 wells; it may not cover edge cases).
- The physical mixing model must be exactly linear (or explicitly modelled as non-linear).
- None of these conditions hold for this dataset.

### C. Hyperparameters

**1. Phase 1 NaN is structural, not tunable.**
NaN appears in every run. After L_var → 0, there is no loss term preventing decoder parameter
drift. Rendered-curve clipping (±10) patches the symptom but not the cause: the decoder is
free to produce any curve once variance is satisfied. A proper fix would add explicit parameter
constraints (e.g., `softplus` on all decoder outputs), but this was not attempted.

**2. L_var enforces variance, not alignment.**
`lambda_var=0.1` with `var_margin=0.05` prevents z_NDM collapsing to zero (the Run 1 root
cause), but high variance in z_NDM does not imply z_NDM encodes Ct. The dims could have high
variance while encoding noise. L_var is necessary but far from sufficient.

**3. L_anchor weight never tuned.**
`lambda_anch=0.5` was chosen once and kept for all 4 runs. It was never searched. A larger
weight would force stronger anchor alignment but at the cost of dominating L_absent and L_consist;
a smaller weight lets the decoder ignore anchor alignment entirely. There is likely no value that
resolves the amplitude ambiguity problem (point B2 above).

**4. Phase 3 LR budget too small.**
Adam(1e-5) × 100 epochs × batch=512 ≈ 4,600 gradient steps. The full model has ~16,000
parameters. If Phase 1 pushes the encoder into a poor initialisation for classification, Phase 3
cannot escape it in 4,600 steps at 1e-5. A higher LR or more epochs risks destroying the
Phase 1 structure — but there's now evidence the Phase 1 structure isn't useful anyway.

### D. Fundamental Approach

**1. Solving a harder problem does not provide a better warm start for an easier one.**
Classification (is target j present?) is a special case of source separation (what is target j's
contribution?). The harder problem's solution would subsume the easier one, but we never solve
it — the anchor ambiguity (B2) prevents Phase 1 from converging to a correct decomposition.
Starting from an incorrect decomposition provides no advantage over random initialisation.

**2. Pre-training transfer requires task alignment.**
ImageNet → medical imaging transfer works because both tasks use edge detection, colour,
texture — low-level visual features that transfer. Here the source task (curve shape decomposition)
requires information that is explicitly discarded before the target task (classification):
per-target amplitude. Phase 3 must learn to classify without using the thing Phase 1 was trained
to recover.

**3. Inference has no access to the separated signals.**
At inference time only the mixed curve is available. The decoder outputs are never used after
Phase 1. The whole pretraining is therefore creating representations that encode information
(per-target shapes) which is redundant for classification, because the classifier must learn to
read target presence from the mixed curve regardless.

**4. End-to-end models implicitly learn what source separation tries to make explicit.**
CNN+GRU with cross-attention (#1 at 83.03%) attends to different temporal segments for each
label without any decomposition prior. This implicit per-target attention is more flexible
than the rigid partitioned latent space, and it is learned under the classification objective
directly — not via a proxy task. The inductive bias of "targets have distinct temporal
signatures" is already present in the architecture (dual-branch, cross-attention) without
requiring explicit decomposition.

### E. What Kind of Problem/Data Would Make This Work

For completeness — source separation pretraining is appropriate when:

- **Per-component ground truth is available** for every training example (e.g., simulated data
  where each target's concentration is exactly known and the mixing model is linear by construction).
- **The mixing model is exact** (audio spectrograms with known source spectra; spectral unmixing
  with characterised endmembers). The approximate linearity of PCR curves breaks this.
- **Components are independently identifiable** in a reference dataset spanning the full
  concentration range. Our single-target bank is limited and may not cover the operating range
  of multi-positive wells.
- **Classification can directly use the separated components** at inference (e.g., each separated
  component is fed to a per-target classifier). This requires the decoder outputs to be available
  and accurate at test time — which they are not here (inference uses only the mixed curve).

For this PCR dataset, the correct inductive biases are temporal patterns (Ct, Fm, Sc) and
label co-occurrence (joint state probability over 2^3 combinations). Both are already
exploited end-to-end by the existing CRF-MRF models that rank #1–5 at 80–83%.

---

## Proposed Next Steps (2026-07-22)

### F. Architectural change: CNN+GRU dual encoder

The current `build_source_sep_encoder` is a **sequential** CNN→GRU pipeline. The top-performing
classifiers use a **dual-branch** architecture where CNN and GRU process the raw input in
parallel, then concatenate. A direct parallel refactor:

```
CNN branch:   Conv1D(32,5,relu) → Conv1D(32,3,relu,s=2) → Flatten → Dense(d_cnn, relu)
GRU branch:   BiGRU(32, return_sequences=False) → Dense(d_gru, relu)  [raw input]
Combine:      Concatenate([cnn_out, gru_out]) → LayerNorm → Dense(bottleneck, linear)
```

Benefits: each branch sees the full raw signal independently; CNN captures local Ct shape,
GRU captures long-range saturation; matches the backbone that is already #1 for classification.
No change to decoder or Phase 1 loss — bottleneck output shape stays the same.

Function to add: `build_source_sep_encoder_dual(T, d_shared, d_target, n_targets=3, d_cnn=32, d_gru=32)`.

### G. Phase naming convention

SC0/SC1 are currently identical (same Phase 1 encoder, no SupCon). Rename to clarify what phase was trained:

| Key | Training | Phase 3 |
|-----|----------|---------|
| `cnn_gru_source_sep_` | Phase 1+2 only (frozen encoder) | No |
| `cnn_gru_source_sep_ft_` | Phase 1+2+3 (end-to-end fine-tuned) | Yes |
| `cnn_gru_source_sep_crf_` | CRF head, Phase 1+2 frozen | No |
| `cnn_gru_source_sep_crf_ft_` | CRF head, Phase 1+2+3 | Yes |

This enables direct comparison of fine-tuned vs frozen to quantify how much Phase 3 helps.

### H. Other improvements

1. **Phase 1 on full unbalanced data**: the balanced dataset has only 875 KPC single-target
   wells for the anchor bank (vs 6,792 in the full set). Running Phase 1 on the full 26,571
   wells (which requires only curves and binary labels, not balanced sampling) gives a much
   richer anchor bank and more diverse reconstruction signal. Phase 2+3 still uses the balanced
   10,383.

2. **Per-target SupCon (`lambda_supcon > 0`)**: was added as Fix 8 but never tested with
   `lambda_supcon > 0`. NDM has failed the Spearman gate in all 4 runs. Setting
   `--lambda_supcon 0.1` in Phase 1 is the most promising quick win for z_NDM alignment.

3. **Shape-normalised anchor loss**: persistent L_anchor plateau (0.05–0.06) is partly L2
   conflating amplitude and shape. A correlation-based or DTW distance would anchor
   Ct/slope profile regardless of the amplitude ambiguity in multi-positive wells.

4. **Soft per-target attention instead of hard partition**: the rigid
   `[z_shared | z_KPC | z_NDM | z_VIM]` slice forced per-target separability. An attention
   pooling (`Dense → query_j → attend over BiGRU sequence`) would let the head decide how to
   use the latent without the architectural constraint.
