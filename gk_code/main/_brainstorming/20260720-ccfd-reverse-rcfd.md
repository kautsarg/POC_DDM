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

## Architecture Comparison (Updated: Dual-Backbone Design)

Both new RCFD and new CCFD use two **independent full dual backbones** of the same type.
The early branch is a complete CNN+GRU or CNN+Trans backbone — not a lightweight encoder.

### New CCFD (8 models)
```
Input (N, T, 1)
  ├─ Dual Backbone A (CGD or CTD, independent) ──────────────► cls_early (softmax × n_classes)
  │                                                                    │ stop_gradient
  └─ Dual Backbone B (same type, independent)  ─► z_raw ─► FiLM ─► z_cond ─► reg_head ─► reg_out
```

- **FiLM input**: n_classes-dim softmax probability vector (from early backbone A)
- **Position 0 output**: `cls_early` (classification, auxiliary task)
- **Position 1 output**: `reg_out` (concentration, primary task)
- **FiLM prefix**: `cfilm` (γ/β conditioned on early class probs)
- **Loss**: UW-SO(CE_cls_early + MSE_reg)

### New RCFD (8 models)
```
Input (N, T, 1)
  ├─ Dual Backbone A (CGD or CTD, independent) ──────────────► early_reg (scalar, linear)
  │                                                                    │ stop_gradient
  └─ Dual Backbone B (same type, independent)  ─► z_raw ─► FiLM ─► z_cond ─► cls_head ─► cls_out
```

- **FiLM input**: scalar concentration estimate (from early backbone A)
- **Position 0 output**: `cls_out` (classification, primary task)
- **Position 1 output**: `early_reg` (concentration, auxiliary task)
- **FiLM prefix**: `film` (γ/β conditioned on early reg estimate)
- **Loss**: UW-SO(CE_cls + MSE_early_reg)

---

## Initial RCFD (unchanged — still in `model_utils_rcfd.py`)

The original 24 RCFD models use a **lightweight early encoder** (one of cnn/gru/trans BiGRU-16) 
rather than a full dual backbone:

```
Input (N, T, 1)
  ├─ Early Encoder (BiGRU-16) ──────────────────────────────► reg_out (scalar, linear)
  │                                                                  │ stop_gradient
  └─ Dual Backbone (CNN+GRU or CNN+Trans) ─► z_raw ─► FiLM ─► z_cond ─► cls_head ─► cls_out
```

These 24 models continue to live in `model_utils_rcfd.py` and are unchanged.

---

## Key Design Decisions

### 1. Symmetric dual backbones
Both the early branch and the main branch now have the same expressive capacity.
This eliminates architecture asymmetry as a confound when comparing RCFD vs CCFD.

The critical question becomes purely about **conditioning direction**, not backbone capacity.

### 2. Layer name safety (calling twice)
Both `_build_cnn_gru_dual_branches_mtl` and `_build_cnn_trans_dual_branches_mtl` use
fully anonymous layers (no `name=` arguments). Calling them twice in the same `functional.Model`
is safe — Keras auto-numbers the instances (`conv1d`, `conv1d_1`, etc.).

### 3. stop_gradient direction
CCFD: prevents regression loss from reshaping the early classifier through FiLM.
New RCFD: prevents classification loss from reshaping the early regressor through FiLM.
Both early backbones still receive direct gradient from their own primary heads.

### 4. FiLM conditioning with multi-dim input
`_apply_film_scalar` (from `model_utils_rcfd`) works unchanged for any input shape.
For CCFD, the input is `(batch, n_classes)` softmax probs — Dense layers handle any width.
For new RCFD, the input is `(batch, 1)` scalar.

### 5. `return_branches=True` scope
When `return_branches=True`, branches (cnn_emb, seq_emb) are returned from the **main** 
(FiLM-conditioned) backbone only — not the early backbone. This ensures SupCon loss acts on 
the representation that was shaped by conditioning.

---

## Model Variants

### New CCFD (8 models, dual-backbone early cls)

| Key | Early backbone | Main backbone | SC |
|-----|---------------|---------------|----|
| `ccfd_cgd` | CNN+GRU | CNN+GRU | SC0 |
| `ccfd_ctd` | CNN+Trans | CNN+Trans | SC0 |
| `ccfd_cgd_supcon_mtl` | CNN+GRU | CNN+GRU | SC1: SupCon on z_cond |
| `ccfd_ctd_supcon_mtl` | CNN+Trans | CNN+Trans | SC1 |
| `ccfd_cgd_supcon2_mtl` | CNN+GRU | CNN+GRU | SC2: SupCon on CNN+seq branches |
| `ccfd_ctd_supcon2_mtl` | CNN+Trans | CNN+Trans | SC2 |
| `ccfd_cgd_supcon3_mtl` | CNN+GRU | CNN+GRU | SC3: SupCon on branches+z_cond |
| `ccfd_ctd_supcon3_mtl` | CNN+Trans | CNN+Trans | SC3 |

### New RCFD (8 models, dual-backbone early reg)

Same 8 keys with `rcfd` substituted for `ccfd`:
`rcfd_cgd`, `rcfd_ctd`, `rcfd_cgd_supcon_mtl`, ..., `rcfd_ctd_supcon3_mtl`

### Initial RCFD (24 models, in `model_utils_rcfd.py`) — unchanged

| Key prefix | Early encoder | Dual backbone | SC levels |
|---|---|---|---|
| `cnn_rcfd_{cgd,ctd}` | CNN | CGD or CTD | SC0–3 (6 keys) |
| `gru_rcfd_{cgd,ctd}` | BiGRU | CGD or CTD | SC0–3 (6 keys) |
| `trans_rcfd_{cgd,ctd}` | Transformer | CGD or CTD | SC0–3 (6 keys) |

The absence of an `{enc_type}_` prefix in a key signals dual-backbone early branch (03d).

---

## Implementation

Single standalone file (no existing code modified):
```
main/03d_ccfd_training.py
```
All 16 new models trained in one run. Results saved to the same joblib as initial RCFD
(keys are distinct — no collision).

**Imports reused:**
- `_StopGradient`, `_apply_film_scalar` ← `model_utils_rcfd`
- `MTLModel`, `_normalize_concentration`, `_inverse_normalize_concentration`,
  `_build_cnn_gru_dual_branches_mtl`, `_build_cnn_trans_dual_branches_mtl` ← `model_utils_mtl`
- `SupConMTLModel`, `SupConBranch2MTLModel`, `SupConBranch3MTLModel`, `_proj_head` ← `model_utils_supcon`

**SLURM:** `main/slurm_jobs/lab_ccfd_training.sh` — unchanged, calls 03d correctly.

---

## Evaluation Plan

| Metric | Initial RCFD | New RCFD | New CCFD |
|--------|-------------|---------|---------|
| Classification acc | primary | primary | auxiliary (cls_early) |
| Regression MSE | auxiliary | auxiliary (early_reg) | **primary** |
| z_cond probe | — | expected best | may differ: conc-conditioned z_cond |

**3-way comparison matrix:**
```
                      Classification acc    Regression MSE
Initial RCFD (cnn/gru/trans)    primary        auxiliary
New RCFD (dual-bb)              primary        auxiliary
New CCFD (dual-bb)              auxiliary      primary
```

**Key questions to answer:**
- Does a full dual-backbone early regressor (new RCFD) beat a lightweight one (initial RCFD)?
- Does target identity conditioning (CCFD) help regression more than concentration conditioning (RCFD) helps classification?
- Is the dual-backbone symmetry beneficial or is the capacity redundant?

---

## Expected Hypotheses

**H1 — Concentration helps classification (RCFD variants win on cls):**
The physical delay from concentration is the primary confound. Removing it first yields a cleaner
latent space for classification.

**H2 — Labels help concentration (CCFD wins on regression):**
Each target has a characteristic curve shape. Knowing which targets are present constrains which
amplitude profile is physically plausible.

**H3 — Dual backbone adds value (new RCFD > initial RCFD):**
A more expressive early regressor provides a richer conditioning signal for FiLM,
leading to better classification of the main branch.

**H4 — FiLM ignored (no difference across variants):**
The dual backbone is already powerful enough without explicit conditioning. FiLM degrades to
identity (γ≈1, β≈0). In this case stop_gradient adds overhead with no benefit.
