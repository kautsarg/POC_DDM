# Per-Label Saliency Heatmaps

**Date:** 2026-07-22  
**Notebook:** `main/notebooks/lab_per_label_saliency.ipynb`  
**Dataset:** `LAB_DDM_paper/12_ACA_qdPCR_multiplex_balanced` (DS2, 7-class multiclass)  
**Scope:** Non-MTL models only (excludes `_mtl`, `rcfd`)

---

## 1. Motivation

`07_attribution_vis_all.py`'s `plot_latent_saliency_heatmap` collapses all samples into a single saliency map by averaging `raw_saliency_curve[i]` over the **full** batch. This gives one map per model covering all 7 label combinations at once.

**Problem**: The averaged map is dominated by whichever label combination is most common and has the strongest gradient signal. It is impossible to tell whether, say, the model attends to cycle 15–20 because that is where NDM amplifies or because that is where VIM_NDM co-amplifies.

**Goal**: Produce **7 separate saliency figures per model**, one per class label, each answering:
- *Which time regions activate which latent dimensions for correctly classified samples of this class?*  (View A)
- *Which time points the model looks at when scoring this class specifically?*  (View B)

---

## 2. Background: How latent saliency is computed in `07`

### 2.1 Artifact extraction pipeline

`extract_xai_artifacts` does three things per model:

1. **Compute `cls_sal` (master saliency):**  
   `target = reduce_max(cls_out, axis=1)` → `tape.gradient(target, X)`.  
   This is the gradient of the **highest** output logit w.r.t. the raw curve. Averaged over samples → `master_saliency (T,)`.

2. **Rank latent dims (`curve_order`):**  
   Build an extractor sub-model up to the CNN/GRU embedding layer.  
   Compute `dy/dz` = gradient of `reduce_max(cls_out)` w.r.t. the latent vector `z`.  
   `rank_latents(dy/dz)` → sort latent units by `mean(|dy/dz|)` across the batch → `curve_order` (up to 100 top dims).

3. **Compute per-dim latent saliency maps (`raw_saliency_curve`):**  
   For each ranked dim `d` in `curve_order`:  
   `target = z[:, d]` → `tape.gradient(target, X)` → `abs(grad)`.  
   Result: a list of `(N, T)` arrays — **one per ranked dim, preserving the batch axis.**

### 2.2 What `raw_saliency_curve[i]` is

`raw_saliency_curve[i][n, t]` = `|∂Z[curve_order[i]] / ∂X[n, t]|`

The (i-th most important latent unit)'s sensitivity to time-point `t` **for sample `n`**.

This is independent of the output class — it measures the gradient of a latent unit w.r.t. the raw input. The **class-specificity** lives only in the ranking (`curve_order`), which was computed using `max(cls_out)`.

---

## 3. Theory: Two complementary views

### View A — Correctly-classified latent heatmap

**What it shows:** For samples that the model correctly identifies as class `c`, which latent dimensions activate, and at which time points?

**Sample filter:** `y_true == c AND y_pred == c`

**Construction:**
```
mask_a = (y_true == c) & (y_pred == c)
hm[i, t] = mean over mask_a of raw_saliency_curve[i][:, t]
         = mean_{n ∈ mask_a} |∂Z[dim_i] / ∂X[n, t]|
```

Result: `(n_dims, T)` heatmap — rows = latent dims ranked by global importance, columns = time.

**Interpretation:**
- Row `i` shows at which cycle the `i`-th most globally-important latent unit responds for correct class-c predictions.
- Bright horizontal bands across a time range → that latent unit is consistently important there for class c.
- Comparing rows across labels reveals whether different labels drive the same latent units at different times (shared feature detector, different amplification timing) or entirely different latent units (independent detectors).

**Limitation:** The ranking (`curve_order`) is global — ranked by importance to `max(cls_out)` across **all** samples. It does not prioritise dims most important for class c specifically. This is intentional: it keeps row identities consistent across all 7 label figures, enabling cross-label comparison at the cost of potentially under-representing dims that matter mainly for rare labels.

**For dual models (CNN + GRU):**  
Two heatmaps stacked vertically — CNN branch and GRU branch — each with their own ranked latent dims. Lets you see whether the CNN (local feature) and GRU (sequential context) attend to the same time regions.

---

### View B — Direct class-specific input gradient

**What it shows:** Which time points drive the model's score for class `c`, when the model is explicitly asked about class `c` (not just its top prediction)?

**Sample filter (default):** `y_true == c` — all class-c samples, regardless of whether the model predicted correctly. Toggle `sample_scope='all'` for the full batch.

**Construction:**
```python
x_var = tf.Variable(X[mask_b])
with tf.GradientTape() as tape:
    cls_out = model(x_var, training=False)[0]   # (N_c, 7)
    target  = tf.reduce_sum(cls_out[:, c])       # sum → grad[i] = d(cls[i,c])/d(x[i])
grad = tape.gradient(target, x_var)              # (N_c, T, 1)
sal  = |grad[:, :, 0]|.mean(axis=0)             # (T,)
```

**Why `reduce_sum` not `reduce_mean`:**  
For architectures without cross-sample coupling at inference (no `BatchNorm`, or `BatchNorm` with `training=False`), sample `i`'s output `cls_out[i, c]` depends only on `x_var[i]`. Therefore `d(Σᵢ cls[i,c]) / d(x[j,t]) = d(cls[j,c]) / d(x[j,t])`. The `reduce_sum` gradient decomposes cleanly per sample. We then average the absolute values to get the mean per-time sensitivity.

**Visualisation:** Plotted as a normalised filled curve overlaid on the mean fluorescence curve of the class-c samples. Same x-axis (time) as View A.

**Contrast with View A:**

| Dimension | View A | View B |
|-----------|--------|--------|
| Gradient target | `max(cls_out)` (= `cls_out[:,c]` for correct samples) | `cls_out[:,c]` explicitly |
| Sample filter | Correctly classified only | All class-c (or all, toggle) |
| Latent space | Decomposed (n_dims × T heatmap) | Collapsed (T saliency curve) |
| Shows | Latent unit activation patterns | Raw input-space time attribution |

**When A and B agree:** The model's top-ranked latent dims activate at the same time regions that directly drive the class-c logit → strong, consistent feature detection.

**When A and B diverge:** Suggests the latent dim ranking (global) does not fully capture what matters for class c specifically. Consider whether a class-specific ranking (computed from `d(cls_c)/dZ`) would reveal different dims.

**`scope='all'` vs `scope='class'`:**
- `'class'` (default): "What does the model attend to for real class-c inputs?"
- `'all'`: "What does the model always look for when scoring class c, even on non-class-c inputs?" — this reveals what features the model associates with class c independent of true label, useful for diagnosing false-positive patterns.

---

## 4. What is NOT in View B (and why)

**Branch decomposition for View B (CNN vs GRU):** Not implemented. To decompose the direct input gradient by branch, you would need `d(cls_c) / d(z_cnn)` and `d(cls_c) / d(z_rnn)` separately — which requires a "head sub-model" from the concatenated latent to the output. This is non-trivial to extract from the functional Keras model without knowing the exact concat layer topology.

**Class-specific latent ranking (alternative View B):** Rather than direct input gradient, one could rank latent dims by `d(cls_c)/dZ` per class and reuse `raw_saliency_curve` with that ranking. This would make View B a second latent heatmap with class-specific row ordering — directly comparable to View A. Requires a separate GradientTape pass per class through the extractor + head split. Not implemented yet; the direct input gradient in View B is simpler and gives complementary (not duplicate) information.

---

## 5. Implementation details

### 5.1 Files

| File | Role |
|------|------|
| `main/notebooks/lab_per_label_saliency.ipynb` | Main notebook |
| `main/07_attribution_vis_all.py` | Source for `extract_xai_artifacts`, `normalize_heatmap` etc. (imported via importlib) |

### 5.2 Import approach for `07_attribution_vis_all.py`

The file has a numeric prefix (`07_`) making it an invalid Python identifier, so `import 07_attribution_vis_all` fails. The notebook uses:

```python
spec = importlib.util.spec_from_file_location('xai07', MAIN_DIR / '07_attribution_vis_all.py')
xai07 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xai07)
```

For this to work, sys.path must include `MAIN_DIR`, `UTILS_DIR`, and `UTILS_MT_DIR` before loading (so `config`, `safe_io`, `model_utils_*` all resolve). The notebook prepends these paths and imports model_utils modules first (to register `@keras.saving.register_keras_serializable()` decorators).

### 5.3 Model exclusion logic

```python
non_mtl_files = [f for f in all_keras
                 if '_mtl' not in f.stem and 'rcfd' not in f.stem]
```

- `_mtl`: multi-task models (cls + reg heads). `extract_xai_artifacts` produces only `cls_saliency` and `reg_saliency` scalars for these — **no `raw_saliency_curve` with batch dim preserved**. Cannot filter by label post-hoc.
- `rcfd`: Regression-Conditioned Feature Dual. 07 treats these as MTL (`is_mtl = True` when `'rcfd' in _base_name`). Same exclusion applies.

### 5.4 `raw_saliency_curve` structure

```
art['raw_saliency_curve']  : list of length n_ranked_dims (≤ 100)
art['raw_saliency_curve'][i]: np.ndarray shape (N_batch, T)
  = |∂Z[curve_order[i]] / ∂X[n, t]|  for each sample n
```

For View A, filter before averaging:
```python
hm[i, t] = art['raw_saliency_curve'][i][mask_a].mean(axis=0)[t]
```

No recomputation needed. The batch dimension was preserved in `compute_latent_saliency_batch`.

### 5.5 Key parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `N_DIMS` | 25 | Rows in View A heatmap. 100 is unreadable at 7 panels. 20–30 is optimal. |
| `SAMPLE_SCOPE` | `'class'` | View B sample filter. `'all'` changes the question being answered. |
| `BATCH_N` | 512 | Samples drawn for artifact extraction. Same batch used for View B gradients. |
| `CURVE_IDX` | 0 | Which curve variant: 0=ori_curve, 1=avg, 2=wavelet. |

### 5.6 Output structure

```
EXP_PATH/model_interpretation/per_label_saliency/
  {model_name}/
    KPC.png
    NDM.png
    NDM_KPC.png
    VIM.png
    VIM_KPC.png
    VIM_NDM.png
    VIM_NDM_KPC.png
```

7 PNGs per model, 150 dpi.

### 5.7 Label encoding

`sklearn.preprocessing.LabelEncoder` on `Y_well` sorts alphabetically:

```
0 = KPC
1 = NDM
2 = NDM_KPC
3 = VIM
4 = VIM_KPC
5 = VIM_NDM        ← note: NOT NDM_VIM (raw label in Y_well is 'VIM_NDM')
6 = VIM_NDM_KPC
```

The notebook asserts `enc.classes_ == LABEL_NAMES` so this is always verified at load time.

---

## 6. Figure anatomy

### Base model (1 curve branch)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  suptitle: {class_name}  |  BASE model                                      │
│                                                                              │
│  ┌─── View A (left 3/4) ──────────────────────────────┐  ┌─ View B ───────┐ │
│  │  ax_ctx: mean curve of correctly-classified class-c │  │                │ │
│  │  ─────────────────────────────────────────────────  │  │  mean curve    │ │
│  │  Raw heatmap (n_dims×T) | Norm heatmap (n_dims×T)  │  │  + gradient    │ │
│  │  [inferno colormap]     | [0–1 row-normalised]     │  │  fill overlay  │ │
│  └──────────────────────────────────────────────────────┘  └────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
Title shows:  N_correct / N_class_in_batch  and  scope + N for View B
```

### Dual model (CNN + GRU)

Same right panel (View B). Left panel gains a third row:
```
  Row 0 (short):  ax_ctx — mean curve + min/max band
  Row 1 (tall):   CNN-Raw | CNN-Norm
  Row 2 (tall):   GRU-Raw | GRU-Norm
```

---

## 7. Interpretation guide

### Reading View A

- **Horizontal bright band in a row**: that latent unit activates consistently at that time range for class-c correct samples.
- **Raw vs. normalised comparison**: raw absolute values reveal relative magnitude (which dims have strongest signal); row-normalised reveals temporal shape independent of magnitude (where does each dim focus within the curve?).
- **Cross-label comparison** (look at the same row across 7 figures for one model): if Row 2 is always bright at cycles 18–22 regardless of label, that latent dim is a general amplification detector. If Row 2 is bright at cycles 15–18 for NDM and 22–26 for KPC, the timing encodes identity.

### Reading View B

- **Tight gradient peak at early cycles**: model cares about fast amplifiers (early Ct) for this class.
- **Broad gradient across all cycles**: model uses shape, not just a single threshold.
- **View B bright where View A is dark (for the same label)**: the input gradient is driven by features that are not well-captured by the top global latent dims — consider re-running with class-specific dim ranking.

### When to switch `scope='all'` in View B

Use `scope='all'` when you want to understand false positives: "Why does the model assign a high NDM_KPC score to VIM_NDM samples?" Run View B for `class_idx=NDM_KPC` with `scope='all'` — the gradient will highlight what in non-NDM_KPC samples resembles NDM_KPC to the model.

---

## 8. Known limitations and future work

1. **Global dim ranking**: View A rows are ordered by global importance (`max cls_out`, all samples). A **class-specific ranking** (using `d(cls_c)/dZ`) would show the latent dims most tuned for class c. Implementation requires: `tf.keras.Model(input, [z_cnn, z_rnn, cls_out])` from a functional model — feasible but needs careful graph tracing.

2. **Sample count per label in batch**: With `BATCH_N=512` and 7 balanced classes, each class gets ~73 samples. Rare labels or imbalanced datasets could have very few correctly-classified samples in the batch, making View A noisy. Increasing `BATCH_N` or using the full dataset (memory permitting) would help.

3. **Batch-norm coupling**: The `reduce_sum` gradient decomposition assumes sample independence. If any model uses `BatchNorm` with `training=True`, gradients couple across samples and `grad[i]` ≠ `d(cls[i,c])/d(x[i])`. All models here use `training=False`, so this is safe.

4. **Staged / cosine-recon variants**: These models have the same `dual` artifact structure. The notebook handles them automatically — no special casing needed.

5. **Multiplex DS1 models**: DS1 (LAB_Multiplex, multilabel binary outputs) would require a different label mapping (`state_idx` from binary vectors rather than integer class). The current notebook is DS2-specific.
