# Source Separation — Run 1 Post-mortem

**Date**: 2026-07-18  
**SLURM job**: 262563 (`lab_multiplex_source_sep_training.sh`)  
**Result file**: `classification_performances_ml_source_sep_10fold.joblib`

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

## Potential next steps if Run 2 still underperforms

- **Increase capacity**: D_SHARED=32, D_TARGET=16 (bottleneck 80 dims instead of 46).  Update BiGRU to match: BiGRU(48) → 96 dims → Dense(80, linear).
- **Normalised anchor loss**: compare decoded curve shapes (Ct, Sc) rather than absolute L2 distance, which is more robust for multi-positive wells where amplitude is shared across channels.
- **SupCon on z_j during pretraining**: add supervised contrastive loss on the per-target latent, using single-target-j wells as positives.  This gives a stronger supervision signal for latent alignment.
- **Phase 3 unfrozen fine-tuning**: after Phase 2 converges, unfreeze the encoder and fine-tune jointly at a lower lr (1e-5).
