# Other Multi-Label Approaches for Multiplex PCR

**Date**: 2026-07-17  
**Context**: Single-well qPCR curves → binary label vector `{0,1}^3` for `[KPC, NDM, VIM]`.  
Hard cases: co-amplification (two targets in one well) and cross-reactivity suppression.

---

## Current approaches (baseline)

| Approach | Architecture | Loss | Label interaction |
|---|---|---|---|
| Flat dense head (`_ml_cls_head`) | GAP → Dense(96) → Dropout → Dense(3) → sigmoid | 3× BCE (independent) | None |
| Label query cross-attn | Learnable label queries MHA over GRU sequence | 3× BCE | Inter-label SA |
| RCFD | FiLM conditioning on concentration; same cls head | 3× BCE + MSE(Ct) | None |
| SupCon variants (SC1-3) | Above backbones + SupCon projection head | BCE + SupCon loss | Via contrastive pairs |

---

## 1. Label-Correlation Modelling

### 1a. Graph Convolutional Network on label graph (ML-GCN / C-GCN)

**Problem it solves**: Flat BCE assumes label independence. Co-amplification means labels are correlated (e.g., KPC+NDM observed more often than expected by chance). GCN bakes co-occurrence statistics into the classifier weights.

**Architecture**:
1. Build `(3, 3)` co-occurrence matrix `A` from training set: `A[i, j] = P(y_j=1 | y_i=1)`. Renormalize: `Â = D^{-1/2} A D^{-1/2}`.
2. Run 2 GCN layers on label node features (e.g., identity matrix or Word2Vec embeddings of "KPC"/"NDM"/"VIM"):
   ```
   H^{l+1} = ReLU(Â H^l W^l)
   ```
   Output: label embeddings `(3, d_label)` encoding relational label structure.
3. Use these embeddings as linear classifiers:
   ```
   logits = backbone_emb @ label_emb.T  →  (batch, 3)
   ```
   Loss: `BCEWithLogits`. The label embedding matrix replaces the fixed `Dense(3)` weight matrix.

**Advantage**: With only 3 labels, the 3×3 `A` matrix is small and the GCN pass is trivial. In larger label spaces (e.g., 20+ targets), GCN provides significant structural regularisation.

**Reference**: Chen et al., "Multi-Label Image Recognition with Graph Convolutional Networks" (CVPR 2019).

---

### 1b. Label dependency LSTM (auto-regressive)

**Problem it solves**: Bernoulli factorisation (`P(y) = ∏ P(y_j)`) ignores conditional dependencies. Auto-regressive modelling captures `P(y) = P(y_1) P(y_2 | y_1) P(y_3 | y_1, y_2)`.

**Architecture**:
- Fix label order by descending marginal frequency (e.g., KPC first if most prevalent).
- Encoder: shared backbone → `z (batch, 96)`.
- Decoder: one-step LSTM. At step `t`:
  - Input = `concat([z, one_hot(ŷ_{t-1})])` (teacher forcing during training; own prediction at inference)
  - `h_t, c_t = LSTM(input, h_{t-1}, c_{t-1})`
  - `logit_t = Dense(1)(h_t)` → `ŷ_t = sigmoid(logit_t)`
- Loss: `sum_t BCE(ŷ_t, y_t)` — identical total contribution to standard BCE, but the prediction at step `t` sees all prior labels.

**With 3 labels**: 3 LSTM steps, trivial compute. With threshold-free ordering (predicted probabilities → sorted label order), can be made order-invariant.

---

### 1c. Structured output with CRF

**Problem it solves**: Sigmoid outputs are point-wise; they don't enforce global label consistency. CRF adds pairwise potentials that score label combinations jointly, then infers the globally best assignment.

**Architecture**:
- Emission scores: `Dense(n_targets)(backbone_emb)` → `(batch, 3)` — same as current head.
- Transition matrix: learned `(3, 3)` pairwise score tensor. Entry `(i,j)` = how much "KPC=i agrees with NDM=j" (for binary i,j this is a 2×2×2×2 = 16-entry tensor, or simplified to a 3×3 co-emission score).
- Loss: negative log-likelihood of the true label vector under the CRF energy, computed via the forward algorithm.
- Inference: Viterbi decoding over the 2^3 = 8 possible label assignments.

**Note**: With 3 binary labels, the full state space (8 states) makes exact inference trivially fast. CRF adds ~9 trainable parameters (3×3 transition table).

---

## 2. Temporal Structure Exploitation

### 2a. Temporal Convolutional Network (TCN)

**Problem it solves**: GRU is a sequential scan (information bottleneck at each step). TCN uses dilated convolutions to capture long-range temporal context in parallel.

**Architecture**:
- Stack of dilated causal `Conv1D` blocks:
  ```
  dilation_rates = [1, 2, 4, 8, ...]   # doubles each layer
  kernel_size = 3
  receptive_field_depth_k = 2^k × (kernel_size - 1) + 1
  ```
- Each block: `Conv1D(filters, kernel=3, dilation=r, padding='causal') → activation → LayerNorm → Dropout`. Residual connection (1×1 Conv to match channels if needed).
- Per-label readout: `n_targets` separate heads sharing early TCN layers, diverging at the final block → label-specialised receptive fields.
- No hidden state → fully parallelisable during training; no vanishing gradient over time.

**In Keras**:
```python
x = tf.keras.layers.Conv1D(filters, 3, dilation_rate=r, padding='causal')(x)
```

---

### 2b. Feature pyramid on curve segments

**Problem it solves**: The full curve contains phases with different diagnostic value (baseline noise, exponential rise, plateau). Segment-level features expose these phases explicitly.

**Approach**:
1. Partition curve into overlapping windows (e.g., cycles 1–15 baseline, 15–35 exponential, 35–45 plateau).
2. Per window: compute slope (linear regression coefficient), curvature (2nd derivative mean), Ct estimate (threshold crossing cycle), amplitude (max − min).
3. Concatenate all window features → `(n_windows × n_features,)` feature vector → MLP.

**Advantage**: Interpretable, no GPU required. **Disadvantage**: feature engineering is brittle; phase boundaries must be tuned per assay.

---

### 2c. Wavelet / multi-scale decomposition

**Problem it solves**: Standard 1D CNN learns a fixed receptive field. Wavelet decomposition provides multi-scale features at no design cost.

**Approach**:
- **Discrete wavelet transform (DWT)**: Decompose curve into approximation + detail coefficients at `L` levels. Each level halves the sequence length and doubles the scale.
- Concatenate DWT coefficients as additional input channels → feed to CNN/GRU backbone as multi-scale input.
- **Continuous wavelet transform (CWT)**: Produces a `(scales, T)` scalogram (time-frequency image). Feed to 2D CNN. More expensive but richer representation.

**Hypothesis**: Different target amplifications may have distinct frequency signatures in the exponential phase (e.g., different rise rates).

---

## 3. Set-Based Prediction (DETR-Style)

**Problem it solves**: Multi-label prediction with sigmoid + threshold requires per-label threshold tuning. DETR-style prediction treats the label set as an unordered collection and learns to activate exactly the right queries.

**Architecture**:
- `n_targets` learnable object queries `(n_targets, d_query)` — each query specialises in detecting one target.
- Transformer decoder: object queries cross-attend to backbone feature map (similar to current cross-attn head, but trained with a matching loss).
- Each query produces a confidence score; match predicted labels to ground-truth labels via **Hungarian algorithm** (minimum-cost bipartite matching) before computing the loss.
- Loss: `BCE(matched_pred, matched_true) + no-object penalty for unmatched queries`.

**Why it helps**: No fixed sigmoid threshold needed. The model learns to activate exactly `k` queries where `k` = number of true labels per sample.

**Practical note**: With 3 labels, Hungarian matching is trivial (3! = 6 assignments). More relevant for open-vocabulary or large-n_targets settings.

**Reference**: Carion et al., "End-to-End Object Detection with Transformers" (ECCV 2020).

---

## 4. Calibration & Threshold Selection

### 4a. Per-label Platt scaling

**Mechanics**: Fit a 1D logistic regression on **held-out validation** pre-sigmoid logits per label:
```
calibrated_prob_j = sigmoid(a_j × logit_j + b_j)
```
`a_j, b_j` fit with `sklearn.linear_model.LogisticRegression(C=1e5)` on val fold logits. Does not change model ranking order, only sharpens/flattens the score distribution toward true calibration.

**When to use**: After training is complete; adds no training overhead.

### 4b. Prevalence (label-frequency) threshold

**Mechanics**: Set `threshold_j = P(y_j = 1)` estimated from training fold. Under class imbalance, this threshold maximises F1 for a well-calibrated model (Lipton et al., "Thresholding Classifiers to Maximize F1 Score", 2014).

**Example**: If KPC prevalence = 40%, use `threshold_KPC = 0.40`. No grid search needed.

### 4c. Cost-sensitive threshold

**Mechanics**: Define asymmetric costs per label: `C_FP[j]` (cost of false positive for label `j`) and `C_FN[j]` (cost of false negative). Optimal Bayes threshold:
```
t*_j = C_FP[j] / (C_FP[j] + C_FN[j])
```
Requires probability calibration first (Platt scaling or isotonic regression).

**Clinical motivation**: Missing KPC-positive patient (FN) is far more dangerous than a false alarm (FP). Setting `C_FN[KPC] >> C_FP[KPC]` → lower detection threshold → higher recall.

---

## 5. Data Augmentation for Co-Amplification

### 5a. Mixup in embedding space

**Mechanics**: Sample pairs `(x_A, y_A), (x_B, y_B)`. Compute backbone embeddings `z_A, z_B`. Mix:
```python
lam    = np.random.beta(0.4, 0.4)
z_mix  = lam * z_A + (1 - lam) * z_B
y_mix  = y_A | y_B          # hard label: union
# or: y_mix = lam * y_A + (1-lam) * y_B   # soft label
```
Train classifier head on `(z_mix, y_mix)`. Requires freezing backbone after warmup or stop-gradient, or end-to-end training with custom training step sampling pairs each batch.

**Advantage over input-space mixup**: Embedding space interpolation is more geometrically meaningful for non-linear features.

### 5b. Curve superposition

**Mechanics**: Physically simulate co-amplification by adding two single-label curves:
```python
curve_AB = alpha * curve_A + (1 - alpha) * curve_B   # alpha ~ Uniform(0.3, 0.7)
label_AB = label_A | label_B
```
**Assumption**: Fluorescence signals add linearly — approximately true for most qPCR chemistries at moderate concentrations.

**Implementation**: Add a data generator stage that pairs single-target curves at batch construction time. Can double or triple effective multi-label training examples at zero annotation cost.

---

## 6. Multi-Task Learning with Auxiliary Signals

### 6a. Ct regression head (auxiliary)

**Architecture**: Add `Dense(n_targets)` regression head on top of backbone, predicting Ct values (continuous cycle threshold). Sentinel-masked for samples without known concentration.

```
L_total = L_cls + lambda_reg × L_reg
L_cls   = BCE(sigmoid(cls_logits), y_binary)
L_reg   = MSE(Ct_pred, Ct_true) masked by sentinel
```

**Difference from RCFD**: RCFD conditions the backbone via FiLM using concentration as a style vector. Here, Ct regression is a secondary output head — the backbone learns features useful for both tasks without being explicitly conditioned on concentration.

### 6b. Curve reconstruction head (self-supervised auxiliary)

**Architecture**: Lightweight decoder `(Dense → Dense → Reshape → Conv1DTranspose)` reconstructs the normalised input curve from the backbone embedding. Loss: MSE on curve.

Forces the backbone to preserve temporal shape information rather than collapsing it to a fixed-length summary. Equivalent to a classification-bottleneck convolutional autoencoder.

---

## 7. Contrastive / Metric Learning

### 7a. SupCon with Jaccard weighting (already implemented)

Weight positive pairs by label-set Jaccard similarity: `w(i,j) = |y_i ∩ y_j| / |y_i ∪ y_j|`. Implemented in `MultiLabelSupConModel` (SC1–SC3 variants). Pulls embeddings of identical-label-set samples together; pushes different-label-set samples apart.

### 7b. Per-label prototype networks

**Architecture**:
- Maintain one running mean embedding per label: `proto_j = EMA_β(mean_{n: y_n,j=1}(z_n))`.
- Predict: `prob_j = sigmoid(-dist(z, proto_j) / τ)` where `dist` = cosine or Euclidean.
- No classification head parameter; the metric space is the classifier.
- Loss: BCE on prototype distances, or prototypical loss:
  ```
  L_j = -log( exp(-d(z, proto_j)) / sum_k exp(-d(z, proto_k)) )
  ```

**Advantage**: Extremely sample-efficient for rare label combinations — a new label only needs a few examples to define its prototype.

---

## 8. Ranking Formulation

### 8a. Label ranking with ListMLE / LambdaLoss

**Problem it solves**: BCE optimises per-label accuracy, not the ranking of labels by confidence. Ranking losses directly optimise ranking metrics (NDCG, MAP).

**Mechanics**: Score labels by predicted probability; loss penalises wrong orderings (e.g., `prob(KPC) < prob(NDM)` when `true(KPC) = 1, true(NDM) = 0`).

**ListMLE**: Minimise negative log-likelihood of the correct label permutation under the Plackett-Luce model.

**Threshold at inference**: Use a dynamic threshold based on predicted probability distribution rather than fixed 0.5.

### 8b. Pairwise ranking loss (RankNet over labels)

**Mechanics**: For each pair of labels `(i, j)`:
```
L_pair = max(0, -(prob_i - prob_j)) if true_i > true_j
```
Combined: `L = BCE + alpha × L_rank`. Encourages the model to rank positive labels above negative labels even when absolute scores are poorly calibrated.

---

## Summary — When to Use What

| Approach | Primary benefit | Implementation effort | Data requirement |
|---|---|---|---|
| GCN on label graph | Explicit co-occurrence encoding | Low (3×3 A matrix) | All training labels |
| Label dependency LSTM | Conditional probability chain | Low (3 LSTM steps) | All training labels |
| CRF output layer | Globally consistent assignment | Medium | All training labels |
| TCN backbone | Parallel long-range context | Medium | — |
| Curve superposition augmentation | Synthetic multi-label samples | Low | Single-label subsets |
| Platt scaling | Calibrated probabilities | Very low (post-hoc) | Held-out val fold |
| Prototype networks | Few-shot new label support | Medium | At least 5+ pos per label |
| DETR-style set prediction | No threshold tuning | High | — |
