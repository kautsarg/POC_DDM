# CCFD: Classification-Conditioned Feature Distillation

**Reverse of RCFD** — instead of concentration conditioning classification, label predictions condition the concentration regression.

---

## Motivation

RCFD encodes the hypothesis: *"concentration is a physical confound that obscures target identity — unwarp it first, then classify."*

The reversed hypothesis (CCFD) is: *"which targets are present is a biological prior that constrains plausible amplitude profiles — know the labels first, then estimate concentration."*

This comparison lets us answer:
- Does concentration information help identify targets? (RCFD)
- Does target identity information help estimate concentration? (CCFD)
- Or are they fully symmetric, and both directions give equivalent mutual information?

---

## Architecture Comparison

### RCFD (existing)
```
Input (N, T, 1)
  ├─ Early Encoder (BiGRU-16) ──────────────────► reg_out (scalar, linear)
  │                                                       │
  └─ Dual Backbone (CNN+GRU or CNN+Trans) ─► z_raw ─► FiLM ─► z_cond ─► cls_head ─► cls_out (sigmoid × 3)
                                                       ↑
                                                  stop_gradient(reg_out)
```

- **FiLM input**: scalar concentration prediction  
- **Main output**: `cls_out` (multi-label binary, 3 targets)  
- **Auxiliary output**: `reg_out` (concentration)  
- **Loss**: UW-SO(BCE_cls + MSE_reg)

### CCFD (proposed)
```
Input (N, T, 1)
  ├─ Early Classifier (BiGRU-16) ──────────────► cls_early (sigmoid × n_targets)
  │                                                       │
  └─ Dual Backbone (CNN+GRU or CNN+Trans) ─► z_raw ─► FiLM ─► z_cond ─► reg_head ─► reg_out (scalar)
                                                       ↑
                                                  stop_gradient(cls_early)
```

- **FiLM input**: 3-dim label probability vector (from early classifier)  
- **Main output**: `reg_out` (concentration)  
- **Auxiliary output**: `cls_early` (multi-label binary, from early encoder)  
- **Loss**: UW-SO(BCE_cls_early + MSE_reg) — identical loss function, reversed data flow

---

## Key Design Decisions

### 1. FiLM input: 3-dim sigmoid vector
The existing `_apply_film_scalar(c_pred, z_raw, emb_dim=96)` function works for any input shape — the Dense layers inside handle (batch, 3) just as they do (batch, 1). BatchNorm normalises per-feature over the batch. Reused with `name_prefix='cfilm'` to distinguish from RCFD FiLM layers.

### 2. stop_gradient direction
In RCFD, `stop_gradient` prevents the classification loss from reshaping the early encoder **through FiLM**. In CCFD, it prevents the regression loss from reshaping the early classifier **through FiLM**. Both early encoders still receive direct gradient from their own loss heads (MSE for RCFD, BCE for CCFD).

### 3. Classification output in CCFD
`cls_early` comes from the early encoder, not from `z_cond`. When evaluating CCFD for classification:
- Report both `cls_early` (auxiliary head) performance
- Optionally probe `z_cond` with a linear classifier (frozen) for a fair single-head comparison
- **Important**: CCFD's classification quality is expected to be lower than RCFD's because it is the secondary task

### 4. SupCon in CCFD
SC variants use the same Jaccard label similarity (`_jaccard_weight_matrix`) to define positive pairs. In CCFD this clusters `z_cond` (concentration-conditioned embedding) by label similarity — encouraging the backbone to preserve label structure even after FiLM modulation.

**Alternative worth testing later**: SupCon with concentration-range buckets as positive pairs (e.g., samples within ±500 copies/µL). This would be more semantically aligned with CCFD's primary objective.

### 5. Early encoder architecture
Mirrors `_build_early_encoder` from `model_utils_rcfd.py` exactly (same depth/width), just replacing the final linear head with sigmoid.

```
BiGRU(16) → Dense(32, relu) → Dense(16, relu) → Dense(8, relu) → Dense(n_targets, sigmoid)
```

---

## Model Variants (16 total)

| Key | Early encoder | Dual backbone | SC |
|-----|--------------|---------------|----|
| `gru_ccfd_cgd` | BiGRU | CNN+GRU | none |
| `gru_ccfd_ctd` | BiGRU | CNN+Trans | none |
| `trans_ccfd_cgd` | Transformer | CNN+GRU | none |
| `trans_ccfd_ctd` | Transformer | CNN+Trans | none |
| `*_supcon_mtl`  | same | same | SC1: SupCon on z_cond |
| `*_supcon2_mtl` | same | same | SC2: SupCon on CNN+seq branches |
| `*_supcon3_mtl` | same | same | SC3: SupCon on CNN+seq+z_cond |

Mirrors the 16 RCFD variants in `classification_performances_ml_condreg.joblib` exactly.

---

## Evaluation Plan

| Metric | Where RCFD wins | Where CCFD wins |
|--------|----------------|-----------------|
| Classification accuracy (exact state match) | Higher — cls is main task | Lower — cls is auxiliary |
| Auxiliary regression MSE | This IS the main task | Auxiliary only, may degrade |
| Concentration regression MSE | Auxiliary | **Main task** — should win |
| z_cond linear probe (cls) | Expected better | May still be competitive if label prior helps backbone |

**Comparison matrix to build:**
```
                  Classification acc   Reg MSE
RCFD (sc0)           primary            auxiliary
CCFD (sc0)           auxiliary (early)  primary
Baseline (no cond)   from condreg exp   from condreg exp
```

---

## Implementation

Single file (no existing code modified):
```
main/multiplex/utils/model_training/model_utils_ccfd.py
```

**Imports reused:**
- `_StopGradient`, `_apply_film_scalar` ← `model_utils_rcfd`
- `_build_cnn_gru_dual_branches_mtl`, `_build_cnn_trans_dual_branches_mtl` ← `model_utils_mtl`
- `MultiLabelMTLModel`, `_jaccard_weight_matrix`, `_supcon_loss_jaccard`, `SUPCON_TEMP` ← `model_utils_multilabel`
- `_proj_head` ← `model_utils_supcon`

**To plug into training pipeline (future PR):**
- Add `from model_utils_ccfd import _CCFD_ALL_FACTORIES, ALL_CCFD_KEYS` to `model_utils_multilabel.py`
- Register under `--ccfd` flag in `03_main_training.py`
- Joblib key: `classification_performances_ml_ccfd.joblib`

---

## Expected Hypotheses

**H1 — Concentration helps classification (RCFD wins):**  
The physical delay from concentration is the primary confound. Removing it via regression-conditioned FiLM should yield a cleaner latent space for classification.

**H2 — Labels help concentration (CCFD wins on regression):**  
Each target has a characteristic curve shape. Knowing which targets are present constrains which amplitude profile is physically plausible, giving the backbone a prior that reduces regression ambiguity.

**H3 — Both help (symmetric):**  
Concentration and target identity are mutually informative. Both conditioning directions improve their respective auxiliary tasks compared to an unconditioned baseline.

**H4 — Neither helps (FiLM ignored):**  
The dual backbone is already powerful enough to jointly learn both tasks without explicit conditioning. FiLM degrades to identity (gamma≈1, beta≈0). In this case, the stop_gradient design means FiLM adds overhead with no benefit.
