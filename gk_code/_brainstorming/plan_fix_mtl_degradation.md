# Plan: Fix MTL Classification Degradation

## Context

Running `--mtl` across ACA_qdPCR, AMCA_qdLAMP, AMCA_qdPCR shows the MTL classification head underperforms its single-task counterpart. Three root causes identified:

1. **Missing `clipnorm`**: MTL compiles with bare `tf.keras.optimizers.Adam()` (`model_utils.py:1173`), but every single-task model uses `Adam(learning_rate=0.001, clipnorm=1.0)`. Without clipping, large regression gradients early in training corrupt the shared backbone.

2. **Kendall oscillation**: Two independent learnable scalars (`log_var_cls`, `log_var_reg`) drift out of sync, causing transient regression-weight spikes that override classification gradients (the "oscillation pathology" described in the UW-SO paper).

3. **Shared embedding gradient interference**: Both heads backpropagate through the same embedding weights. Classification wants concentration-invariant features; regression wants concentration-predictive ones. Forcing a single embedding to satisfy both degrades CE performance.

**On backbone capacity**: Current embedding dimensions (32 for GRU/LSTM/RNN/LF models, 64 for dual/gated, 16 for transformer) are sufficient for the task complexity (3–9 classes + 5–7 concentration levels). Widening would add parameters and overfitting risk without resolving the gradient interference root cause. Fix 3 below addresses capacity at the head level instead.

---

## Step 0 — Backup existing results before any code change

```bash
LAB=/vol/bitbucket/gk225/POC_DDM_datasets/LAB_DDM_paper
for DS in ACA_qdPCR AMCA_qdLAMP AMCA_qdPCR; do
    SRC="$LAB/$DS/classification_performances_10fold.joblib"
    DST="$LAB/$DS/classification_performances_10fold_baseline_mtl.joblib"
    [ -f "$SRC" ] && cp "$SRC" "$DST" && echo "Backed up $DS"
done
```

Run **before** any fix and before the next `--mtl --force_rerun` run.

---

## Fix 1 — `clipnorm=1.0` on MTL optimizer

**File**: `utils/model_training/model_utils.py:1173`

```python
# Before:
model.compile(optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

# After:
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
    metrics=['accuracy'],
)
```

Prerequisite for everything else — without it, regression gradients dominate regardless of loss balancing.

---

## Fix 2 — Switch to UW-SO (already coded as comments in `model_utils_mtl.py`)

In `MTLModel.__init__` (lines 40–51): comment out `log_var_cls`/`log_var_reg`, uncomment `self.log_T`.

In `_compute_loss` (lines 60–80): comment out the Kendall block (lines 60–62), uncomment the UW-SO block (lines 64–80).

In `train_step`/`test_step` return dicts (lines 93–95, 103–104): add `'log_T': self.log_T` for monitoring.

**Why UW-SO over Kendall**: Enforces `w_i ∝ 1/L_i` via `stop_gradient` at every step. When regression MSE is high, regression weight is immediately suppressed rather than oscillating. One fewer trainable parameter.

---

## Fix 3 — Task-specific projection towers in `_mtl_wrap`

**File**: `utils/model_training/model_utils_mtl.py:156–170`

```python
def _mtl_wrap(inputs, embedding, n_classes):
    cls_feat = tf.keras.layers.Dense(32, activation='relu',    name='cls_feat')(embedding)
    cls_out  = tf.keras.layers.Dense(n_classes, activation='softmax', name='cls_out')(cls_feat)
    reg_feat = tf.keras.layers.Dense(32, activation='relu',    name='reg_feat')(embedding)
    reg_h    = tf.keras.layers.Dense(16, activation='relu',    name='reg_hidden')(reg_feat)
    reg_out  = tf.keras.layers.Dense(1,  activation='linear',  name='reg_out')(reg_h)
    return MTLModel(inputs=inputs, outputs=[cls_out, reg_out])
```

**Effect**: `∂L/∂embedding = W_cls^T·∂cls/∂cls_feat + W_reg^T·∂reg/∂reg_feat`. Each task learns its own projection from the shared embedding, so classification and regression gradient directions no longer directly overwrite each other at the embedding.

**Special benefit for `transformer_mtl`**: Its embedding is only 16 units; Fix 3 expands each task's effective view to 32 without touching the backbone.

---

## Implementation order

1. **Step 0** — backup joblibs
2. **Fix 1 only** → rerun → if Δ(ST−MTL) shrinks substantially, clipnorm was the main issue
3. **Fix 1 + Fix 2** → rerun → check `log_T` increases then plateaus in training history
4. **Fix 1 + Fix 2 + Fix 3** → rerun → if degradation persists, add the projection towers

---

## Files to modify

| File | Location | Change |
|---|---|---|
| `utils/model_training/model_utils.py` | line 1173 | `Adam()` → `Adam(learning_rate=0.001, clipnorm=1.0)` |
| `utils/model_training/model_utils_mtl.py` | lines 40–51 | swap `log_var_cls`/`log_var_reg` → `log_T` |
| `utils/model_training/model_utils_mtl.py` | lines 60–80 | swap Kendall → UW-SO loss block |
| `utils/model_training/model_utils_mtl.py` | lines 93–95, 103–104 | add `'log_T': self.log_T` to return dicts |
| `utils/model_training/model_utils_mtl.py` | lines 156–170 | replace `_mtl_wrap` with per-task projection version |

No changes to factories, `03_main_training.py`, `config.py`, or notebooks.

---

## Verification

After each step:
```bash
python 03_main_training.py --task_id 0 --mtl --force_rerun --n_splits 5
```
Then `lab_mtl_conc_analysis.ipynb` cell `b0fa9bd1`:
- `plot_stmtl_comparison` — MTL bars should approach or exceed ST (classification goal)
- `plot_reg_metrics` — RMSE/R² should not significantly worsen vs the baseline backup (regression must not break)

After Fix 2: confirm `log_T` in training history increases monotonically early then stabilises. If it oscillates, tighten `clipnorm` to 0.5.

After Fix 3: the regression head gains an extra Dense(32) layer (`reg_feat`), so RMSE may change — this is expected and acceptable as long as R² stays positive and regression is still learning concentration.
