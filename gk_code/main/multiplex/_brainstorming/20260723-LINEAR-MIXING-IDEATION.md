# Ideation: Linear Mixing Model for Source Separation

**Date:** 2026-07-23  
**Status:** Challenged — not recommended as-is; see extensions section

---

## The Idea

Train a global linear model:

```
w1 · curve_KPC + w2 · curve_NDM + w3 · curve_VIM + b = multiplex_curve
```

where `curve_j` are rendered single-target curves from a source-sep decoder.  
Absent targets have `curve_j = 0` (drops out naturally since `w_j × 0 = 0`).

Learned `(w1, w2, w3, b)` would then replace the fixed sum/avg in `L_consist`.

---

## Challenge: Why This Doesn't Work as a Global Linear Model

### 1. Circular dependency — no clean training target

Training the linear model on multi-target wells requires `curve_j` for each active target in that well. But those are exactly what source-sep is trying to learn. The two viable curve sources are:

- **Anchor bank (random single-target samples):** A randomly chosen KPC curve has a different Ct/amplitude than the KPC component inside a KPC+NDM mixture. The regression noise is enormous; the fitted weights absorb the Ct mismatch rather than the true mixing ratio.
- **Model's own decoded outputs:** Weights are valid only for the current model's decoded curves. Retrain with new weights → curves change → weights become stale. Fixed-point problem with no convergence guarantee.

### 2. Global weights can't handle per-sample Ct variability

PCR fluorescence amplitude is inversely proportional to Ct (initial template concentration). Two KPC+NDM wells:

| Well | Ct_KPC | Ct_NDM | Dominant component |
|---|---|---|---|
| A | 25 | 35 | KPC (high amplitude) |
| B | 35 | 25 | NDM (high amplitude) |

A single global `w_KPC` cannot be correct for both. The mixing ratio varies per-sample, not per-dataset.

### 3. On equimolar balanced data, avg already is the optimal linear model

If all multi-target wells are equimolar (same concentration per active target), the theoretical ground-truth is `w_j = 1/n_active` — i.e., the `avg` formulation. The +8.8pp improvement of `avg_consist` over `sum_consist` validates this direction empirically. A learned global linear model on balanced equimolar data would converge to `avg` anyway — no gain over the current formulation.

### 4. Absent-target handling is trivially correct but unidentified

`w_j × 0 = 0` for inactive targets regardless of `w_j`, so inactive channels zero out automatically. However, this means `w_j` for absent targets is unidentified by the regression (any value is equally valid). Not harmful, but not informative either.

---

## What Would Actually Work: Per-Sample Learned Weights

The valuable kernel of the idea is replacing the fixed `1/n_active` with **per-sample weights** that adapt to each well's Ct distribution:

### Option A — Soft gate on z_j (recommended)

```python
# After computing rendered[j] from decoder:
w_j = tf.keras.layers.Dense(1, activation='sigmoid')(z_j)  # per-sample, per-target
recon = sum(w_j * rendered[j] for j in range(n_targets))
```

- Learned jointly with encoder and decoder via L_consist
- Per-sample (adapts to Ct variation)
- Degenerates to avg if gate learns uniform weights → can only be ≥ avg, never worse
- No ground-truth requirement; no circular dependency

### Option B — Concentration-aware decoder output

If `z_j[0]` reliably encodes Ct (currently Spearman WARN for most variants), the decoder could scale output amplitude by `exp(-k·z_j[0])`. Physically motivated but requires reliable Ct encoding in the latent space.

### Option C — Per-combination weight fitting (weak baseline)

Fit separate weights per label combination (KPC+NDM, KPC+VIM, NDM+VIM, all-three) using only single-target wells within matched Ct ranges. Coarse but avoids circularity. Main weakness: no per-sample adaptation, and requires binning by Ct.

---

## Verdict

| Approach | Circular? | Per-sample? | Beats avg? | Implementation cost |
|---|---|---|---|---|
| Global linear model (original idea) | Yes | No | No (converges to avg on balanced data) | Low but pointless |
| Soft gate on z_j | No | Yes | Potentially | Low — one dense layer per decoder output |
| Concentration-aware decoder | No | Yes | Depends on Ct encoding | Medium |
| Per-combination fitting | No | No | Marginal | Medium |

**Conclusion:** The global linear model doesn't add value over `avg_consist` on balanced equimolar data, and has a fundamental circular dependency for training. The extension worth pursuing is a **learnable per-sample gate on decoder outputs** (Option A), jointly trained — worth trying in a future 03c-v2 if `avg_consist` results plateau.
