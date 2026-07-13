# Architecture Brainstorm: Two New Training Approaches for MTL

---

## Why MTL Underperformed

The parallel Y-shaped MTL (current `cnn_gru_dual_mtl`) adds a regression head that branches off the **shared embedding** (`z`, 96-dim) and trains jointly with classification via UW-SO dynamic loss weighting. The hypothesis was that making the backbone aware of concentration would improve class boundary learning at low concentrations.

The failure mode: the backbone does learn concentration, but the classification head receives a generic `z` that is not explicitly modulated by concentration at decision time. The regression task regularises the backbone but does not change **how features flow into the classification head**. The two tasks pull on the same embedding without a mechanism for the concentration signal to shift the classification boundary.

Both new approaches fix this structural gap.

---

## Approach A — Regression-Primed Dual (RPD)

### Concept

Run a lightweight **early regression encoder** first. It predicts concentration as a scalar. That scalar is fed into a **FiLM conditioning layer** that physically modulates the feature maps of a full dual classification backbone. The classification head sees embeddings that have already been amplified/shifted as a function of predicted concentration.

This is fundamentally different from the existing `cnn_gru_film_mtl` model, which uses one branch embedding (CNN) to condition the other branch embedding (GRU). RPD uses a **scalar concentration prediction** to condition the **entire fused dual embedding**. One scalar drives gain and bias for all 96 embedding dimensions.

### Why It Should Work

At low concentrations the early encoder's regression signal will be a small scalar. The FiLM layer learns to respond to that scalar with large gains, amplifying the classification-relevant features. The classifier effectively receives the instruction "this is a low-amplitude signal — look harder". This is an architectural solution, not a loss-weight solution.

### Architecture

```
Input curve (batch, T, 1)
        │
        ├──────── Early Reg Encoder ─────────────────────────────────┐
        │         (CNN / GRU / Transformer, lightweight)              │
        │         → reg_emb (batch, 32)                               │
        │                                                             │
        │         ┌──── Regression Head ────┐                         │
        │         │ Dense(16, relu)          │                         │
        │         │ Dense(8,  relu)          │ → reg_out (batch, 1)  ←┘
        │         │ Dense(1,  linear)        │    c_pred
        │
        ├──────── Dual Classification Backbone ──────────────────────┐
        │         CNN branch  → cnn_emb  (batch, 32)                 │
        │         GRU / Trans → other_emb (batch, 64)                 │
        │         Concatenate → z_raw  (batch, 96)                   │
        │                                                             │
        │         FiLM Conditioning on c_pred:                       │
        │           γ = Dense(96)(reg_out)    — learned gain          │
        │           β = Dense(96)(reg_out)    — learned bias          │
        │           z_cond = γ ⊙ z_raw + β   (batch, 96)            │
        │                                                             │
        │         Classification Head:                                │
        │           Dense(16, relu)                                   │
        │           Dense(n_classes, softmax) → cls_out               │
        └────────────────────────────────────────────────────────────┘
```

The FiLM layer uses **scalar-to-feature** conditioning, not feature-to-feature. `c_pred` is `(batch, 1)` and Dense projects it to `(batch, 96)` — one learned γ and β per feature dimension. The classifier's entire 96-dim space is concentration-aware.

### Training Objective

Uses the same `MTLModel` UW-SO loss:
- `reg_out`: MSE against true log10-normalised concentration (from early encoder)
- `cls_out`: cross-entropy against class labels (from conditioned dual head)
- Dynamic inverse-loss weighting keeps the two tasks balanced without manual tuning

All three parts train jointly (end-to-end gradients through FiLM → dual backbone + through reg head → early encoder).

### Six Initial Combinations

| # | Early Reg Encoder | Dual Cls Backbone | Model Key |
|---|------------------|-------------------|-----------|
| 1 | CNN (lightweight) | CNN + GRU | `cnn_rpd_cgd` |
| 2 | CNN | CNN + Trans | `cnn_rpd_ctd` |
| 3 | GRU (BiGRU) | CNN + GRU | `gru_rpd_cgd` |
| 4 | GRU | CNN + Trans | `gru_rpd_ctd` |
| 5 | Transformer | CNN + GRU | `trans_rpd_cgd` |
| 6 | Transformer | CNN + Trans | `trans_rpd_ctd` |

`rpd` = Regression-Primed Dual; `cgd` = CNN+GRU Dual; `ctd` = CNN+Trans Dual.

**Recommended starting point**: `trans_rpd_cgd` — Transformer for regression (performs best for reg), CNN+GRU for classification (strongest classification dual baseline).

### Hyperparameters to Expose

- `--condreg {cnn|gru|transformer}` flag on `03_main_training.py` (selects early encoder)
- Dual model is selected by the model key as usual
- No new epoch hyperparameters — same EarlyStopping and ReduceLR apply

### Additional Recommendations Beyond the 6

**Option 7 — Shared Encoder (Weight-Tied RPD)**
Use the same Transformer backbone for BOTH early regression AND the classification backbone (shared weights). Only the heads differ.
```
Input → Shared Trans → z  ─┬─→ reg_head → c_pred
                            └─→ FiLM(c_pred) → z_cond → cls_head
```
Half the parameters, but forces the backbone to satisfy both tasks simultaneously without a dedicated early encoder.

**Option 8 — Confidence-Gated FiLM**
Output both `c_pred` and a confidence score `σ` from the early encoder. Scale FiLM influence by confidence:
```python
σ        = Dense(1, sigmoid)(reg_emb)   # how certain is the concentration estimate?
γ_scaled = σ * Dense(96)(c_pred) + (1-σ) * ones_like(z)   # gates towards identity
β_scaled = σ * Dense(96)(c_pred)
z_cond   = γ_scaled * z + β_scaled
```
When concentration is ambiguous (e.g., missing data), FiLM degrades gracefully to identity.

**Option 9 — Multi-level FiLM (two conditioning points)**
Condition at both the CNN branch embedding AND the fused z, using two separate FiLM operations at different abstraction levels.

### Implementation Scope

**New file**: `utils/model_training/model_utils_cascade.py`
- `RPDModel(MTLModel)`: overrides `_compute_loss` to route reg_out from early encoder (no other changes needed — same UW-SO math)
- `_build_early_encoder(inputs, enc_type)`: builds CNN / GRU / Trans early encoder returning `(reg_emb, reg_out)`
- `_apply_film_scalar(c_pred, z, emb_dim=96)`: Dense(emb_dim) × 2 → γ, β → FiLM
- `_build_rpd_model(T, n_classes, reg_enc, dual_type)`: assembles full model
- 6 factory functions (one per combination)
- `RPD_MODEL_KEYS`: list of all 6 keys

**Changes to existing files**:
- `03_main_training.py`: add `--condreg {cnn|gru|transformer}` flag; add RPD model dispatch block
- `model_utils.py`: add `elif _base_m in RPD_MODEL_KEYS:` dispatch branch
- `config.py`: register 6 new keys in `MODEL_KEY_MAP` and `_MTL_MODEL_KEYS`

**Early encoder architectures (lightweight by design):**

*CNN early encoder*:
```
Conv1D(16, kernel=5, relu, padding='same')
Conv1D(8,  kernel=3, relu, padding='same')
GlobalAveragePooling1D()
Dense(32, relu)   → reg_emb (batch, 32)
```

*GRU early encoder*:
```
Bidirectional(GRU(16, return_sequences=False))
Dense(32, relu)   → reg_emb (batch, 32)
```

*Transformer early encoder*:
```
Conv1D(32, kernel=5, strides=2, padding='same', relu)
MaxPooling1D(2)
+ Positional embedding (T//4 positions, dim 32)
1× TransformerBlock (MHA: 2 heads, key_dim 16; FFN: 32)
GlobalAveragePooling1D()
Dense(32, relu)   → reg_emb (batch, 32)
```

These are intentionally ~1/4 the depth of the full dual backbone so training is dominated by the classification path.

---

## Approach B — Phase-Decoupled MTL (PD-MTL)

### Concept

Keep the Y-shaped MTL architecture unchanged. Restructure the training into two explicit phases:

- **Phase 1 — Physics Pre-training** (e.g. first 50 epochs): Set classification loss weight to 0. Train backbone + regression head on concentration only. The backbone learns curve physics (amplitude, slope, delay — all concentration-dependent signals) without being distracted by class boundaries.
- **Phase 2 — Identity Fine-tuning** (remaining epochs): Freeze backbone. Set regression loss weight to 0. Train only the classification head on top of the now-expert backbone features.

### Why It Should Work

The standard MTL joint training has a gradient tug-of-war: regression gradients push the backbone toward concentration-discriminative features, classification gradients push toward class-discriminative features. At low concentrations both pull in different directions because the class boundary is concentration-dependent. By separating the phases, the backbone gets a clean signal in each phase. The classification head then gets excellent frozen features without backbone interference.

### Architecture

Identical to `cnn_gru_dual_mtl`. No structural change.

### Training Schedule

```
Phase 1 (epochs 1 → P₁):
    loss = MSE(reg_out, c_true)   only
    trainable: backbone + reg_head
    frozen: cls_feat, cls_out

Phase 2 (epochs P₁+1 → P_total):
    loss = CE(cls_out, y_true)    only
    frozen: all backbone layers
    trainable: cls_feat, cls_out
```

**Recommended P₁**: 50 epochs (10% of 500 total). Regression typically converges faster than classification. Can tune per-model.

**EarlyStopping behaviour**: Monitor `val_loss` in Phase 1 (regression val MSE). Phase 2 monitors `val_loss` (classification val CE). Two separate callbacks, one per phase.

### Auto Phase Transition (Recommended)

Rather than fixing P₁ = 50, trigger the transition automatically when the **regression validation loss plateaus** — the same signal EarlyStopping uses, but for switching phases instead of stopping.

The model's `test_step` always reports both component losses to `logs`, even though only one drives gradients per phase:

```python
def test_step(self, data):
    x, y = data
    cls_out, reg_out = self(x, training=False)
    ce, mse, n_valid = self._base_losses(y, cls_out, reg_out)
    active = tf.cond(tf.equal(self.curriculum_phase, 0), lambda: mse, lambda: ce)
    return {'loss': active, 'val_reg_loss': mse, 'val_cls_loss': ce}
```

The auto-transition callback monitors `val_reg_loss`:

```python
class AutoPhaseTransitionCallback(tf.keras.callbacks.Callback):
    def __init__(self, min_phase1_epochs=20, patience=15, min_delta=1e-4,
                 backbone_layer_names=None):
        super().__init__()
        self.min_phase1_epochs  = min_phase1_epochs
        self.patience           = patience
        self.min_delta          = min_delta
        self.backbone_layer_names = backbone_layer_names or []
        self._best_reg          = float('inf')
        self._wait              = 0

    def on_epoch_end(self, epoch, logs=None):
        if self.model.curriculum_phase == 1:
            return  # already in Phase 2
        if epoch < self.min_phase1_epochs:
            return  # safety floor

        reg_val = logs.get('val_reg_loss', float('inf'))
        if reg_val < self._best_reg - self.min_delta:
            self._best_reg = reg_val
            self._wait     = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                print(f'\n[PD-MTL] Regression converged at epoch {epoch+1} '
                      f'(val_reg_loss={reg_val:.4f}). Switching to Phase 2.')
                for lname in self.backbone_layer_names:
                    self.model.get_layer(lname).trainable = False
                self.model.curriculum_phase.assign(1)
                self.model.compile(
                    optimizer=tf.keras.optimizers.Adam(lr=0.001, clipnorm=1.0),
                    metrics=['accuracy'])
                self._wait = 0
```

**Why this is better than a fixed count**:
- Easy/clean concentration data → regression may converge in 15–20 epochs
- Noisy/sparse concentration data → may need 80–100 epochs
- The callback adapts per-dataset and per-model automatically

**Typical observed range**: 20–80 epochs across the existing MTL experiments, with the dual models on qdPCR data converging around 30–40.

### Flag Design

```
--mtl_cl                        # activate curriculum/phased training
--cl_auto                       # use auto-convergence detection (default when --mtl_cl)
--cl_phase1_epochs 50           # override: fixed epoch count (disables auto)
--cl_phase1_min_epochs 20       # auto mode: safety floor (default 20)
--cl_phase1_patience   15       # auto mode: plateau patience (default 15)
```

When `--cl_phase1_epochs` is not given, auto-convergence is used. When it is given, the fixed count takes precedence (useful for ablation studies or reproducing a specific run).

Works orthogonally with `--supcon {1|2|3}`:
```
--mtl --mtl_cl --supcon 2   # Phased-MTL + SupCon v2 (auto phase transition)
```

### Model Keys

Mirror of existing MTL keys with `_cl_` infix:

| Key | Description |
|-----|-------------|
| `cnn_gru_dual_cl_mtl` | Phase-decoupled CNN+GRU dual |
| `cnn_trans_dual_cl_mtl` | Phase-decoupled CNN+Trans dual |
| `cnn_gru_dual_cl_supcon_mtl` | + SupCon v1 projection head |
| `cnn_gru_dual_cl_supcon2_mtl` | + SupCon v2 |
| `cnn_gru_dual_cl_supcon3_mtl` | + SupCon v3 |
| `cnn_trans_dual_cl_supcon_mtl` | Trans + SupCon v1 |
| ... | etc. |

Start with just `cnn_gru_dual_cl_mtl` and `cnn_trans_dual_cl_mtl` for initial validation.

### Implementation Scope

**New subclasses in `model_utils_mtl.py`** (or new `model_utils_curriculum.py`):

```python
class CurriculumMTLModel(MTLModel):
    """Phase-decoupled MTL: phase 0 = regression only; phase 1 = classification only."""
    def __init__(self, *args, backbone_layer_names=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.curriculum_phase = tf.Variable(0, trainable=False, dtype=tf.int32)
        self._backbone_layer_names = backbone_layer_names or []

    def _compute_loss(self, y_cls, cls_out, y_reg, reg_out):
        ce, mse, n_valid = self._base_losses(y_cls, cls_out, y_reg, reg_out)
        return tf.cond(
            tf.equal(self.curriculum_phase, 0),
            lambda: mse,   # Phase 1: regression only
            lambda: ce,    # Phase 2: classification only
        )
```

**New Keras callback**:

```python
class PhaseTransitionCallback(tf.keras.callbacks.Callback):
    def __init__(self, phase1_epochs, backbone_layer_names):
        super().__init__()
        self.phase1_epochs = phase1_epochs
        self.backbone_layer_names = backbone_layer_names

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self.phase1_epochs:
            for lname in self.backbone_layer_names:
                self.model.get_layer(lname).trainable = False
            self.model.curriculum_phase.assign(1)
            # Recompile resets Adam momentum — intentional, Phase 2 starts fresh
            self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, clipnorm=1.0),
                               metrics=['accuracy'])
```

**Changes to existing files**:
- `model_utils.py`: in MTL dispatch branch, check `_base_m in CL_MTL_MODEL_KEYS`; add `PhaseTransitionCallback` to `_fit_callbacks`; set `epochs = phase1_epochs + phase2_epochs`
- `03_main_training.py`: add `--mtl_cl`, `--cl_phase1_epochs N` flags; pass into `evaluate_outlier_filters`
- `config.py`: register CL model keys in `MODEL_KEY_MAP` and `_MTL_MODEL_KEYS`
- `model_utils_mtl.py` / new file: `CurriculumMTLModel`, `PhaseTransitionCallback`, `CL_MTL_MODEL_KEYS`

**Key implementation subtlety**: `model.compile()` inside a callback resets the Adam optimizer state (momentum buffers). This is intentional for Phase 2 — Phase 2 starts with a fresh Adam from a frozen backbone, avoiding stale momentum from Phase 1 regression gradients.

---

## Comparison: Approach A vs Approach B

| | RPD (Approach A) | PD-MTL (Approach B) |
|--|-----------------|----------------------|
| Architecture change | Yes — adds early encoder + FiLM | No — same Y-network |
| Training change | No — joint end-to-end | Yes — phased schedule |
| Parameters | +30–50% (early encoder) | Same as MTL |
| Concentration signal path | Explicit (conditions feature space) | Implicit (shapes backbone) |
| Implementation complexity | High (new model file, 6 factories) | Medium (subclass + callback) |
| Interpretability | FiLM γ/β weights show what conc affects | No new interpretability |
| Combined? | Can add phased training to RPD | Can add FiLM to PD-MTL |
| Recommended for first trial | `trans_rpd_cgd` | `cnn_gru_dual_cl_mtl` |

---

## Recommended Experiment Order

1. **Baseline**: Confirm `cnn_gru_dual_mtl` and `cnn_trans_dual_mtl` accuracy on all concentration-having folders as reference point.

2. **First trial PD-MTL** (`cnn_gru_dual_cl_mtl`, P₁=50, total=500): Tests whether phased training alone fixes the gradient conflict. Quick to implement — no architecture changes.

3. **First trial RPD** (`trans_rpd_cgd`): Tests the scalar FiLM conditioning. If trans reg encoder is indeed best for regression, this is the optimal single combination.

4. **Compare cross-over**: Is `trans_rpd_cgd` > `cnn_gru_dual_cl_mtl` > `cnn_gru_dual_mtl`? The architecture approach should dominate if low-concentration misclassification is truly a feature-space problem.

5. **If RPD wins**: Run all 6 combinations to find the best early encoder.

6. **Combine both**: `trans_rpd_cl_cgd` — RPD architecture + phased training. Hypothesis: Phase 1 trains the early encoder as a pure regressor; Phase 2 trains the FiLM + classifier on conditioned frozen features.

---

## Implementation Plan (Ordered Tasks)

### Task 1 — PD-MTL (Approach B, ~2–3 days)

1. Add `CurriculumMTLModel(MTLModel)` and `PhaseTransitionCallback` to `model_utils_mtl.py`
2. Add `CL_MTL_MODEL_KEYS = ['cnn_gru_dual_cl_mtl', 'cnn_trans_dual_cl_mtl']`
3. Add dispatch in `model_utils.py` `evaluate_outlier_filters` for CL keys
4. Add `--mtl_cl`, `--cl_phase1_epochs` to `03_main_training.py`
5. Register keys in `config.py` (`MODEL_KEY_MAP`, `_MTL_MODEL_KEYS`)
6. Syntax check + run one fold locally to verify phase transition fires

### Task 2 — RPD (Approach A, ~3–5 days)

1. Create `utils/model_training/model_utils_cascade.py` with:
   - `_build_early_encoder(inputs, enc_type)` for cnn/gru/transformer
   - `_apply_film_scalar(c_pred, z, emb_dim=96)` for FiLM conditioning
   - `_build_rpd_model(T, n_classes, reg_enc, dual_type)` assembler
   - 6 factory functions (`create_{reg_enc}_rpd_{dual_type}_model`)
   - `RPDModel(MTLModel)` — minimal subclass if any override needed
   - `RPD_MODEL_KEYS` list
2. Add dispatch in `model_utils.py`
3. Add `--condreg` flag to `03_main_training.py`
4. Register keys in `config.py`
5. Test `trans_rpd_cgd` on one fold

### Task 3 — SLURM Jobs

```bash
# Run PD-MTL
sbatch --array=0-N multi_full_pipeline.sh --mtl --mtl_cl --cl_phase1_epochs 50

# Run RPD combinations
sbatch --array=0-N multi_full_pipeline.sh --condreg transformer
sbatch --array=0-N multi_full_pipeline.sh --condreg cnn
sbatch --array=0-N multi_full_pipeline.sh --condreg gru
```

### Task 4 — Analysis

Add PD-MTL and RPD keys to `plot_stmtl_comparison` palette in `lab_mtl_result_analysis.ipynb`.

---

## Open Questions / Risks

1. **Gradient conflict in RPD**: During joint training, gradients from the classification path flow backward through FiLM into `c_pred`. This means `c_pred` is shaped by classification loss too, undermining the "pure regressor" design. Mitigation: use `tf.stop_gradient(c_pred)` before the FiLM Dense layers — condition on a stopped gradient so the early encoder is optimised only for regression.

2. **FiLM collapse**: If `c_pred` is approximately constant across the dataset (narrow concentration range), FiLM learns a constant gain and β = bias, degenerating to a simple bias shift. Check that γ variance across samples is meaningfully > 0 after training.

3. **Early encoder overfitting**: Lightweight encoder on a small dataset may overfit to concentration. Use BatchNorm after each conv/dense in the early encoder and L2 regularisation on the reg_head Dense weights.

4. **Phase 2 optimizer reset** (PD-MTL): Intentional but verify that Phase 2 EarlyStopping patience (100 epochs) is large enough given the reset. If Phase 2 converges fast, reduce patience to 50.

5. **Concentration missing data**: `c_pred` is computed for every sample. FiLM conditioning is applied even when `c_true = REG_SENTINEL`. This is fine at inference (concentration is unknown anyway). During training, the regression loss is masked by `t != REG_SENTINEL` as usual — the FiLM conditioning on those samples still occurs but does not contribute to reg loss.
