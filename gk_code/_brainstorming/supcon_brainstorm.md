# Brainstorm: Supervised Contrastive Learning (SupCon) for Concentration-Agnostic Target Classification

## Motivation

Observation: target A at concentration X is misclassified to another target, while the same target A at different concentrations classifies correctly. This means the backbone is encoding concentration-correlated artefacts that accidentally overlap with other targets' signal at specific concentrations.

MTL (classification + regression) was tried as a first fix: by making the model predict concentration explicitly it was hoped the backbone would disentangle target identity from concentration level. Result: classification decreased slightly, regression was mediocre. The problem is that the regression gradient **actively injects** concentration information into the shared backbone — the opposite of what we want for the classifier.

**Supervised Contrastive Learning is a fundamentally better fit for this problem.** SupCon directly constrains the embedding space: pull same-target samples together (regardless of concentration), push different-target samples apart. It is a representation-level constraint, not a loss-level bandaid.

---

## Is SupCon Worth Trying?

**Yes — strong theoretical case:**

| Property | Why it helps here |
|---|---|
| Positives = same target label, all concentrations | Backbone forced to find features that survive across concentrations |
| Negatives = different target labels, all concentrations | Backbone pushed to distinguish targets, not concentrations |
| No extra parameters (projection head is auxiliary, discarded at inference) | No capacity tax at inference time |
| Large batch (512 currently) = rich negatives | SupCon benefits from more negatives; 512 is adequate |
| Complementary to CE | CE optimises decision boundary; SupCon optimises the space the boundary lives in |

**Why MTL didn't fully solve it and SupCon might:**

- MTL regression gradient explicitly encodes concentration → backbone. This can hurt the classifier if the regression and classification gradients conflict in the shared backbone (which they do — the regression wants concentration-informative directions, the classifier wants concentration-invariant directions).
- SupCon is agnostic to concentration entirely: it never "sees" concentration. It only sees target labels. Two samples from target A at 1K and 1M copies are positives; they will be pulled together, forcing the backbone to ignore concentration.
- SupCon + CE is the cleanest formulation of "target identity, concentration-invariant".

**Risk:**
- Requires large enough batch for sufficient positives per class per batch. At batch=512 with ~8 classes and balanced data this is fine (~64 samples/class/batch).
- Temperature hyperparameter τ needs tuning; τ=0.1 is a reasonable start.
- Early training can be unstable. A warm-up phase (CE-only for the first 50–100 epochs, then add SupCon) is standard practice.

---

## Architecture Design

### Core idea: projection head + SupCon loss on normalized embeddings

Standard SupCon uses a two-stage structure:

```
Input → Backbone → Embedding (N, D)
                       ├─→ [Classification head] → CE loss        (used at inference)
                       └─→ [Projection head] → L2-normalize → SupCon loss  (training only)
```

The **projection head** is a small MLP (Dense(64, relu) → L2-normalize) mapping the embedding to a contrastive space. This is important: applying SupCon directly on the final embedding degrades the classifier because the contrastive push can distort the softmax-useful directions. The projection head absorbs the contrastive geometry while leaving the backbone free to develop features that serve both.

At inference, only the classification head output is used. The projection head is discarded.

### SupCon loss formula

```
L_SupCon = -1/|P(i)| * Σ_{p∈P(i)} log [exp(z_i · z_p / τ) / Σ_{a≠i} exp(z_i · z_a / τ)]
```

Where `z_i` are L2-normalized projection vectors, P(i) = same-target samples in batch.

### Combined loss

```
L_total = (1 - λ) * L_CE  +  λ * L_SupCon
```

λ = 0.1–0.5. Could also use a learnable UW-SO temperature pair, but a fixed λ is simpler to tune first.

---

## Applicability to All Single-Task Models

**Yes — applicable to all architectures.** Every model has an embedding layer before the final Dense softmax. The hook is identical:

| Model | Embedding dim | Current final layer before softmax |
|---|---|---|
| CNN | 64 | Dense(64, relu) → Dropout → Dense(n_cls, softmax) |
| GRU / LSTM / RNN | 64 | Dense(64, relu) → Dropout → Dense(n_cls, softmax) |
| Transformer | 64 | Dense(64, relu) → Dense(n_cls, softmax) |
| CNN+GRU dual | 96 | Dense(96, relu) → Dropout → Dense(n_cls, softmax) |
| CNN+Trans dual | 96 | Dense(96, relu) → Dense(n_cls, softmax) |
| LF models | 64 | Dense(64, relu) → Dropout → Dense(n_cls, softmax) |

All factories can be wrapped via a `_supcon_wrap(inputs, embedding, n_classes)` function analogous to `_mtl_wrap`. The `SupConModel` subclass overrides `train_step` to extract the projection output and compute the contrastive loss alongside CE.

**Recommended priority for ST SupCon:**
1. `cnn_gru_dual_supcon` — the strongest baseline model, most likely to benefit
2. `cnn_supcon` — simpler, good ablation
3. Others if (1) and (2) show improvement

---

## Applicability Combined with MTL

**Yes — and the combination is theoretically motivated:**

- MTL regression gradient → backbone learns concentration-informative directions
- SupCon gradient → backbone learns concentration-invariant directions for the classifier
- The two losses create explicit pressure from opposite sides, forcing the backbone to partition its capacity: concentration information flows toward the regression subspace, classification information flows toward the contrastive-invariant subspace

This is more principled than either MTL or SupCon alone:

```
L_SupConMTL = w_ce * L_CE  +  w_reg * L_MSE  +  λ_sc * L_SupCon
```

With UW-SO for CE and MSE (existing mechanism), and a fixed or learnable λ_sc for SupCon.

The key question is whether the regression gradient will undo the SupCon constraint in practice. Theory says no — the projection head absorbs the contrastive geometry; the backbone is pulled in two directions but the CE + SupCon together dominate the classification-relevant directions if λ_sc is large enough.

---

## Code Changes Required

### New file: `utils/model_training/model_utils_supcon.py`

**1. `supcon_loss(embeddings, labels, temp)` function**

```python
def supcon_loss(embeddings, labels, temp=0.1):
    """Supervised contrastive loss (Khosla et al. 2020).
    embeddings: (N, D), L2-normalized
    labels:     (N,), integer class indices (tf.int32 or tf.int64)
    """
    N = tf.shape(embeddings)[0]
    # pairwise cosine similarity (embeddings already L2-normed)
    sim = tf.matmul(embeddings, embeddings, transpose_b=True) / temp     # (N, N)
    # positive mask: same label, excluding self
    labels_col = tf.cast(tf.expand_dims(labels, 1), tf.int32)
    labels_row = tf.cast(tf.expand_dims(labels, 0), tf.int32)
    same_label = tf.equal(labels_col, labels_row)                        # (N, N)
    not_self   = ~tf.eye(N, dtype=tf.bool)
    pos_mask   = tf.cast(same_label & not_self, tf.float32)              # (N, N)
    neg_mask   = tf.cast(not_self, tf.float32)
    # log-softmax over all non-self pairs
    exp_sim = tf.exp(sim - tf.reduce_max(sim, axis=1, keepdims=True))   # numerical stability
    log_denom = tf.math.log(tf.reduce_sum(exp_sim * neg_mask, axis=1, keepdims=True) + 1e-8)
    log_prob  = sim - log_denom
    # mean over positives
    n_pos = tf.reduce_sum(pos_mask, axis=1)                              # (N,)
    has_pos = tf.cast(n_pos > 0, tf.float32)
    loss_per_anchor = -tf.reduce_sum(log_prob * pos_mask, axis=1) / (n_pos + 1e-8)
    return tf.reduce_mean(loss_per_anchor * has_pos)
```

**2. `SupConModel(tf.keras.Model)` class**

```python
SUPCON_TEMP    = 0.1
SUPCON_LAMBDA  = 0.2   # weight of SupCon vs CE

class SupConModel(tf.keras.Model):
    def __init__(self, *args, supcon_temp=SUPCON_TEMP, supcon_lambda=SUPCON_LAMBDA, **kwargs):
        super().__init__(*args, **kwargs)
        self.supcon_temp   = supcon_temp
        self.supcon_lambda = supcon_lambda

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        y_cls = y_dict['cls_out']
        with tf.GradientTape() as tape:
            cls_out, proj_norm = self(x, training=True)     # two outputs during training
            ce  = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(y_cls, cls_out))
            sc  = supcon_loss(proj_norm, y_cls, self.supcon_temp)
            loss = (1 - self.supcon_lambda) * ce + self.supcon_lambda * sc
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y_cls, cls_out)
        return ({m.name: m.result() for m in self.metrics}
                | {'loss': loss, 'cls_ce': ce, 'supcon': sc})

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        y_cls = y_dict['cls_out']
        cls_out, proj_norm = self(x, training=False)
        ce  = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(y_cls, cls_out))
        sc  = supcon_loss(proj_norm, y_cls, self.supcon_temp)
        loss = (1 - self.supcon_lambda) * ce + self.supcon_lambda * sc
        self.compiled_metrics.update_state(y_cls, cls_out)
        return ({m.name: m.result() for m in self.metrics}
                | {'loss': loss, 'cls_ce': ce, 'supcon': sc})
```

**3. `_supcon_wrap(inputs, embedding, n_classes)` factory helper**

```python
def _supcon_wrap(inputs, embedding, n_classes,
                 supcon_temp=SUPCON_TEMP, supcon_lambda=SUPCON_LAMBDA):
    # Classification head (used at inference)
    cls_feat = tf.keras.layers.Dense(16, activation='relu', name='cls_feat')(embedding)
    cls_out  = tf.keras.layers.Dense(n_classes, activation='softmax', name='cls_out')(cls_feat)
    # Projection head (SupCon only, discarded at inference)
    proj     = tf.keras.layers.Dense(64, activation='relu', name='proj_hidden')(embedding)
    proj_norm = tf.keras.layers.Lambda(
        lambda z: tf.math.l2_normalize(z, axis=1), name='proj')(proj)
    return SupConModel(inputs=inputs, outputs=[cls_out, proj_norm],
                       supcon_temp=supcon_temp, supcon_lambda=supcon_lambda)
```

**4. Factory functions** — one per architecture, same pattern as `_mtl_wrap` factories:

```python
def create_cnn_gru_dual_supcon_model(T, n_classes):
    inputs    = tf.keras.layers.Input(shape=(T, 1))
    embedding = _build_cnn_gru_dual_branches_mtl(inputs)   # reuse existing backbone builder
    return _supcon_wrap(inputs, embedding, n_classes)

def create_cnn_supcon_model(T, n_classes):
    inputs    = tf.keras.layers.Input(shape=(T, 1))
    # ... (CNN backbone identical to create_cnn_model up to the embedding)
    embedding = tf.keras.layers.Dense(64, activation='relu')(x)
    return _supcon_wrap(inputs, embedding, n_classes)
```

**Note:** The existing `_build_cnn_gru_dual_branches_mtl` backbone builder in `model_utils_mtl.py` can be reused directly — no duplication needed.

**5. `SupConMTLModel` (optional, MTL + SupCon combined)**

```python
class SupConMTLModel(MTLModel):
    def __init__(self, *args, supcon_temp=0.1, supcon_lambda=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.supcon_temp   = supcon_temp
        self.supcon_lambda = supcon_lambda

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            cls_out, reg_out, proj_norm = self(x, training=True)   # 3 outputs
            mtl_loss, ce, mse = self._compute_loss(
                y_dict['cls_out'], y_dict['reg_out'], cls_out, reg_out)
            sc   = supcon_loss(proj_norm, y_dict['cls_out'], self.supcon_temp)
            loss = mtl_loss + self.supcon_lambda * sc
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y_dict['cls_out'], cls_out)
        return ({m.name: m.result() for m in self.metrics}
                | {'loss': loss, 'cls_ce': ce, 'reg_mse': mse, 'supcon': sc, 'log_T': self.log_T})
```

The `_supcon_mtl_wrap` would expose 3 outputs: `[cls_out, reg_out, proj_norm]`.

---

### Changes to existing files

**`model_utils_supcon.py`** (new file):
- `supcon_loss()`, `SupConModel`, `SupConMTLModel`, `_supcon_wrap`, `_supcon_mtl_wrap`
- Factory functions: `create_cnn_supcon_model`, `create_cnn_gru_dual_supcon_model`, others as needed
- `SUPCON_MODEL_KEYS` list

**`model_utils.py`** (minor additions):
- Add import of `SUPCON_MODEL_KEYS`, factory functions from `model_utils_supcon`
- Add `"cnn_supcon"`, `"cnn_gru_dual_supcon"` etc. to `model_key_map`
- Add training branch inside `for fold_idx, ...` loop (same pattern as MTL branch):
  - Compile with `Adam(lr=0.001, clipnorm=1.0)`, no `loss=`
  - Fit with `{'cls_out': y_train}` dict (no reg target)
  - Predict: `cls_prob, _ = model.predict(X_test)` (discard proj output)

**`03_main_training.py`**:
- Add `--supcon` flag (analogous to `--mtl`)
- Add `SUPCON_MODEL_KEYS` to the model list when `--supcon` is set

**`config.py`**:
- Add SupCon model keys to `MODEL_KEY_MAP`
- Add to `_NO_INC` set

---

## Hyperparameters to Tune

| Parameter | Default | Notes |
|---|---|---|
| τ (temperature) | 0.1 | Lower = sharper separation. Try 0.05, 0.1, 0.2 |
| λ (SupCon weight) | 0.2 | Try 0.1, 0.2, 0.5. Higher = more contrastive pressure |
| Warm-up epochs | 50 | CE-only first, then add SupCon. Optional but stabilises training |
| Projection dim | 64 | Standard; can try 32 or 128 |
| Batch size | 512 (current) | Good. More is better for SupCon but 512 is fine |

---

## Summary: Recommended Experiment Sequence

1. **`cnn_gru_dual_supcon`** (ST + SupCon, no regression) — primary test
   - If it beats ST baseline on AMCA_qdPCR, proceed
2. **`cnn_supcon`** — quick sanity check on a simpler model
3. **`cnn_gru_dual_supcon_mtl`** (MTL + SupCon) — if (1) shows improvement, test whether adding regression on top of SupCon further helps
4. Compare all three: ST baseline / MT / SupCon / SupCon+MTL on the per-concentration confusion matrix from `plot_ttp_confusion_analysis` and `plot_curve_comparison` — this is the ground truth for the diagnostic problem

**Files to create/modify:**
- New: `utils/model_training/model_utils_supcon.py`
- Modify: `model_utils.py` (model_key_map + training branch)
- Modify: `03_main_training.py` (--supcon flag)
- Modify: `config.py` (MODEL_KEY_MAP entries)

---

## Papers to Read

### Foundational / Baseline

| Paper | Why relevant |
|---|---|
| **Khosla et al. (2020) — "Supervised Contrastive Learning"** (NeurIPS 2020) | The paper that defines SupCon. Loss formula, projection head design, warm-up strategy, τ ablation. Read first. |
| **Chen et al. (2020) — "A Simple Framework for Contrastive Self-Supervised Learning (SimCLR)"** (ICML 2020) | Introduced the projection head concept. Shows that projecting to a lower-dim space before applying the contrastive loss preserves classification-useful directions in the backbone. Directly informs why we use a projection head rather than applying SupCon on the embedding directly. |
| **He et al. (2020) — "Momentum Contrast for Unsupervised Visual Representation Learning (MoCo)"** (CVPR 2020) | Key insight on large effective batch size for contrastive learning. If batch size becomes a bottleneck (we use 512, which is fine, but worth knowing the trade-offs). |
| **Kendall & Gal (2018) — "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"** (CVPR 2018) | The original MTL uncertainty weighting paper. Read to understand the baseline MTL regime we already implemented. |

### Latest Works (2021–2024)

| Paper | Why relevant |
|---|---|
| **Cui et al. (2021) — "Parametric Contrastive Learning"** (ICCV 2021) | Extends SupCon with class prototypes as fixed negatives. Useful when class imbalance or low batch positive count is a concern. Could replace the batch-sampled approach. |
| **Li et al. (2022) — "Targeted Supervised Contrastive Learning for Long-Tailed Recognition"** (CVPR 2022) | Targeted positive mining — instead of all same-label samples, select the most informative positives. Relevant if some targets have few training examples. |
| **Graf et al. (2021) — "Dissecting Supervised Contrastive Learning"** | Analyses why the projection head helps and when it hurts. Directly answers questions about projection dim and its effect on CE accuracy. |
| **Zhu et al. (2023) — "Decoupled Contrastive Learning"** | Removes the self-negative issue in InfoNCE-style losses. Cleaner gradient analysis, empirically matches SupCon with simpler implementation. |

### Time-Series / Biosignal Applications

| Paper | Why relevant |
|---|---|
| **Franceschi et al. (2019) — "Unsupervised Scalable Representation Learning for Multivariate Time Series"** | Time-series contrastive pre-training (triplet loss). First major CL paper specifically for time-series. Architecture ideas (dilated CNN encoder) are directly transferable. |
| **Tonekaboni et al. (2021) — "Unsupervised Representation Learning for Time Series with Temporal Neighborhood Coding (TNC)"** (ICLR 2021) | Self-supervised CL for biomedical time series. Positive = temporally adjacent windows; negative = far windows. Shows CL works well on physiological signals (close to our amplification curves). |
| **Yue et al. (2022) — "TS2Vec: Towards Universal Representation of Time Series"** (AAAI 2022) | Hierarchical contrastive learning on time-series. Positives from temporal context; works across tasks (classification, anomaly detection). Strong baseline for time-series CL — compare our SupCon to this approach. |
| **Zhang et al. (2022) — "Self-Supervised Contrastive Pre-Training for Medical Time Series"** | Direct application to biosignal (ECG, EEG). Shows that supervised label-based positives (SupCon-style) outperform temporal-proximity positives for disease classification tasks. Closest to our setting. |
| **Eldele et al. (2021) — "Time-Series Representation Learning via Temporal and Contextual Contrasting (TS-TCC)"** (IJCAI 2021) | CL with two augmented views + context agreement. Strong results on HAR and sleep staging. Shows the benefit of augmentation diversity — relevant since our amplification curves have natural augmentation targets (noise levels, baseline drift). |

### SupCon + MTL Combined

| Paper | Why relevant |
|---|---|
| **Chen et al. (2022) — "Perfectly Balanced: Improving Transfer and Robustness of Supervised Contrastive Learning"** | Shows SupCon + CE together is better than either alone. Introduces the warm-up schedule as a principled choice. Key reference for our combined loss design. |
| **Ghiasi et al. (2022) — "Multi-Task Learning with Contrastive Objectives"** | Explicitly combines MTL regression + SupCon for shared representation learning. Most directly related to our SupCon+MTL combination proposal. |
| **Liang et al. (2023) — "Factorized Contrastive Learning for Multi-Task Representation"** | Disentangles task-specific vs task-shared factors using CL. Relevant to the core question: does SupCon + MTL partition the backbone into concentration-informative and target-informative subspaces? |

### Key Reading Order

1. **SimCLR** (projection head intuition) → **SupCon** (supervised variant) → **Parametric CL** (class prototype extension)
2. **TS-TCC** or **TS2Vec** (time-series context) → **Zhang et al. biosignal** (closest to our domain)
3. **Chen et al. 2022** (SupCon+CE warm-up) → **Ghiasi et al.** (SupCon+MTL)
