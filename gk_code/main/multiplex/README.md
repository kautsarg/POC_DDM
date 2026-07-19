# multiplex — Multi-Label PCR Classification

Trains multi-label classifiers on multiplex dPCR curves (`(T, 1)` → `{0,1}^n_targets`).
All scripts live under `main/multiplex/`; results land in `{EXP_FOLDER}/01_ACA_qdPCR/`.

---

## Quick start

```bash
cd /vol/bitbucket/gk225/POC_DDM/gk_code/main/multiplex

python -u 03_main_training.py \
    --task_id 0 \
    --exp_folder /vol/bitbucket/gk225/POC_DDM_datasets/LAB_Multiplex \
    --n_splits 5
```

---

## SupCon (SC) levels

All flag groups support SC0–SC3 via `--supcon {0|1|2|3}`. The SC level controls which
contrastive projection heads are added on top of the backbone.

| SC | Flag | Extra loss term | Proj heads attached to |
|---|---|---|---|
| 0 | _(omit)_ | none | — |
| 1 | `--supcon 1` | λ · SupCon(fused) | z (96-dim fused embedding) |
| 2 | `--supcon 2` | λ · (SupCon(CNN) + SupCon(seq)) | CNN branch + seq branch |
| 3 | `--supcon 3` | λ · (SupCon(CNN) + SupCon(seq) + SupCon(fused)) | all three |

λ values (additive, not complement): SC1 λ=0.1, SC2/3 λ_each=0.05/0.033.

---

## Model groups

### Standard (`default` group) — no extra flag

Dual-branch and single-branch backbones with a standard sigmoid classification head.

**Result file:** `classification_performances_ml[_10fold].joblib`

| Model key | Print name | Backbone | SC | Head |
|---|---|---|---|---|
| `cnn` | CNN | Single CNN | 0 | Dense(3)→sigmoid |
| `gru` | GRU | Single BiGRU | 0 | Dense(3)→sigmoid |
| `transformer` | Transformer | Single Transformer | 0 | Dense(3)→sigmoid |
| `cnn_supcon` | CNN SC1 | Single CNN | 1 | sigmoid + SupCon(fused) |
| `gru_supcon` | GRU SC1 | Single BiGRU | 1 | sigmoid + SupCon(fused) |
| `transformer_supcon` | Trans SC1 | Single Transformer | 1 | sigmoid + SupCon(fused) |
| `cnn_gru_dual` | CNN+GRU | CNN+BiGRU dual | 0 | Dense(3)→sigmoid |
| `cnn_gru_dual_supcon` | CNN+GRU SC1 | CNN+BiGRU dual | 1 | + SupCon(fused) |
| `cnn_gru_dual_supcon2` | CNN+GRU SC2 | CNN+BiGRU dual | 2 | + SupCon(CNN+seq) |
| `cnn_gru_dual_supcon3` | CNN+GRU SC3 | CNN+BiGRU dual | 3 | + SupCon(all) |
| `cnn_trans_dual` | CNN+Trans | CNN+Transformer dual | 0 | Dense(3)→sigmoid |
| `cnn_trans_dual_supcon` | CNN+Trans SC1 | CNN+Transformer dual | 1 | + SupCon(fused) |
| `cnn_trans_dual_supcon2` | CNN+Trans SC2 | CNN+Transformer dual | 2 | + SupCon(CNN+seq) |
| `cnn_trans_dual_supcon3` | CNN+Trans SC3 | CNN+Transformer dual | 3 | + SupCon(all) |

**14 models total.**

```bash
for SC in 0 1 2 3; do
    SARG=(); [ "$SC" -gt 0 ] && SARG=(--supcon "$SC")
    python -u 03_main_training.py ... "${SARG[@]}"
done
```

---

### Cross-attention (`cross_attn` group) — `--cross_attn`

Label query cross-attention head: `n_targets` learnable queries attend over the
pre-pooling sequence, followed by inter-label self-attention. Shallow kv_seq backbone.

**Result file:** `classification_performances_ml_cross_attn[_10fold].joblib`

| Model key | Print name | Backbone | SC |
|---|---|---|---|
| `cnn_gru_dual_cross_attn` | CNN+GRU CAttn SC0 | CNN+BiGRU dual | 0 |
| `cnn_gru_dual_cross_attn_supcon` | CNN+GRU CAttn SC1 | CNN+BiGRU dual | 1 |
| `cnn_gru_dual_cross_attn_supcon2` | CNN+GRU CAttn SC2 | CNN+BiGRU dual | 2 |
| `cnn_gru_dual_cross_attn_supcon3` | CNN+GRU CAttn SC3 | CNN+BiGRU dual | 3 |
| `cnn_trans_dual_cross_attn` | CNN+Tr CAttn SC0 | CNN+Trans dual | 0 |
| `cnn_trans_dual_cross_attn_supcon` | CNN+Tr CAttn SC1 | CNN+Trans dual | 1 |
| `cnn_trans_dual_cross_attn_supcon2` | CNN+Tr CAttn SC2 | CNN+Trans dual | 2 |
| `cnn_trans_dual_cross_attn_supcon3` | CNN+Tr CAttn SC3 | CNN+Trans dual | 3 |

**8 models total.**

```bash
for SC in 0 1 2 3; do
    python -u 03_main_training.py ... --cross_attn --supcon "$SC"
done
```

---

### Cross-attention v2 (`cattn_v2` group) — `--cross_attn_v2`

Ablation of two improvements: deeper kv_seq (second BiGRU + positional encoding) and deeper
head (2 CA blocks + FFN). Includes combined `v2` variants (CGD and CTD backbones).

**Result file:** `classification_performances_ml_cattn_v2[_10fold].joblib`

| Model key | Print name | kv_seq depth | Head depth | Backbone |
|---|---|---|---|---|
| `cnn_gru_dual_cross_attn_deepkv` | CNN+GRU CAttn-DKV SC0 | Deep (2nd BiGRU + pos-enc) | Shallow (1 CA block) | CGD |
| `cnn_gru_dual_cross_attn_deepkv_supcon` | CNN+GRU CAttn-DKV SC1 | Deep | Shallow | CGD |
| `cnn_gru_dual_cross_attn_deepkv_supcon2` | CNN+GRU CAttn-DKV SC2 | Deep | Shallow | CGD |
| `cnn_gru_dual_cross_attn_deepkv_supcon3` | CNN+GRU CAttn-DKV SC3 | Deep | Shallow | CGD |
| `cnn_gru_dual_cross_attn_deephead` | CNN+GRU CAttn-DHD SC0 | Shallow (1st BiGRU) | Deep (2 CA + FFN) | CGD |
| `cnn_gru_dual_cross_attn_deephead_supcon` | CNN+GRU CAttn-DHD SC1 | Shallow | Deep | CGD |
| `cnn_gru_dual_cross_attn_deephead_supcon2` | CNN+GRU CAttn-DHD SC2 | Shallow | Deep | CGD |
| `cnn_gru_dual_cross_attn_deephead_supcon3` | CNN+GRU CAttn-DHD SC3 | Shallow | Deep | CGD |
| `cnn_gru_dual_cross_attn_v2` | CNN+GRU CAttn-V2 SC0 | Deep | Deep | CGD |
| `cnn_gru_dual_cross_attn_v2_supcon` | CNN+GRU CAttn-V2 SC1 | Deep | Deep | CGD |
| `cnn_gru_dual_cross_attn_v2_supcon2` | CNN+GRU CAttn-V2 SC2 | Deep | Deep | CGD |
| `cnn_gru_dual_cross_attn_v2_supcon3` | CNN+GRU CAttn-V2 SC3 | Deep | Deep | CGD |
| `cnn_trans_dual_cross_attn_v2` | CNN+Tr CAttn-V2 SC0 | Deep | Deep | CTD |
| `cnn_trans_dual_cross_attn_v2_supcon` | CNN+Tr CAttn-V2 SC1 | Deep | Deep | CTD |
| `cnn_trans_dual_cross_attn_v2_supcon2` | CNN+Tr CAttn-V2 SC2 | Deep | Deep | CTD |
| `cnn_trans_dual_cross_attn_v2_supcon3` | CNN+Tr CAttn-V2 SC3 | Deep | Deep | CTD |

**16 models total.**

```bash
for SC in 0 1 2 3; do
    python -u 03_main_training.py ... --cross_attn_v2 --supcon "$SC"
done
```

---

### Cross-attention v2 + AuxDet (`auxdet` group) — `--cross_attn_auxdet`

Adds a per-CA-block auxiliary BCE head tapped after each CA+LN sublayer (before inter-label SA).
Forces each query to develop independent per-label features.

Loss: `BCE(cls_out) + λ_aux · Σ_block BCE(aux_block)`

**Result file:** `classification_performances_ml_auxdet[_10fold].joblib`

| Model key | Print name | SC |
|---|---|---|
| `cnn_gru_dual_cross_attn_v2_auxdet` | CNN+GRU CAttn-V2 AuxDet SC0 | 0 |
| `cnn_gru_dual_cross_attn_v2_auxdet_supcon` | CNN+GRU CAttn-V2 AuxDet SC1 | 1 |
| `cnn_gru_dual_cross_attn_v2_auxdet_supcon2` | CNN+GRU CAttn-V2 AuxDet SC2 | 2 |
| `cnn_gru_dual_cross_attn_v2_auxdet_supcon3` | CNN+GRU CAttn-V2 AuxDet SC3 | 3 |

**4 models total** (CGD backbone only).

```bash
for SC in 0 1 2 3; do
    python -u 03_main_training.py ... --cross_attn_auxdet --supcon "$SC"
done
```

---

### Cross-attention v2 + QuerCon (`quercon` group) — `--cross_attn_quercon`

Per-label supervised contrastive loss on the L2-normalised attended query vectors.
After the final CA block, `queries_norm[:, j, :]` is treated as an embedding:
positives = samples where `y[:, j] = 1`.

Loss: `(1 − λ_qc) · BCE + λ_qc · mean_j SupCon(queries_norm[:, j, :], y[:, j])`

Combine with `--supcon {1|2|3}` to additionally add backbone projection loss.

**Result file:** `classification_performances_ml_quercon[_10fold].joblib`

| Model key | Print name | SC | QuerCon | Backbone SC |
|---|---|---|---|---|
| `cnn_gru_dual_cross_attn_v2_quercon` | CNN+GRU CAttn-V2 QuerCon | 0 | ✓ | — |
| `cnn_gru_dual_cross_attn_v2_quercon_supcon` | CNN+GRU CAttn-V2 QuerCon SC1 | 1 | ✓ | SupCon(fused) |
| `cnn_gru_dual_cross_attn_v2_quercon_supcon2` | CNN+GRU CAttn-V2 QuerCon SC2 | 2 | ✓ | SupCon(CNN+seq) |
| `cnn_gru_dual_cross_attn_v2_quercon_supcon3` | CNN+GRU CAttn-V2 QuerCon SC3 | 3 | ✓ | SupCon(all) |

**4 models total** (CGD backbone only).

```bash
for SC in 0 1 2 3; do
    python -u 03_main_training.py ... --cross_attn_quercon --supcon "$SC"
done
```

---

### RCFD (`condreg` group) — `--condreg`

Regression-Conditioned Feature Dual: a lightweight early encoder (GRU or Transformer)
predicts concentration first, then conditions the dual-backbone embedding via FiLM before
classification. Trains 4 backbone combinations × 4 SC levels = 16 models.

**Result file:** `classification_performances_ml_condreg[_10fold].joblib`

| Model key | Print name | Early encoder | Late backbone | SC |
|---|---|---|---|---|
| `gru_rcfd_cgd` | GRU RCFD CGD | BiGRU | CNN+BiGRU dual | 0 |
| `gru_rcfd_cgd_supcon_mtl` | GRU RCFD CGD SC1 | BiGRU | CNN+BiGRU dual | 1 |
| `gru_rcfd_cgd_supcon2_mtl` | GRU RCFD CGD SC2 | BiGRU | CNN+BiGRU dual | 2 |
| `gru_rcfd_cgd_supcon3_mtl` | GRU RCFD CGD SC3 | BiGRU | CNN+BiGRU dual | 3 |
| `gru_rcfd_ctd` | GRU RCFD CTD | BiGRU | CNN+Trans dual | 0 |
| `gru_rcfd_ctd_supcon_mtl` | GRU RCFD CTD SC1 | BiGRU | CNN+Trans dual | 1 |
| `gru_rcfd_ctd_supcon2_mtl` | GRU RCFD CTD SC2 | BiGRU | CNN+Trans dual | 2 |
| `gru_rcfd_ctd_supcon3_mtl` | GRU RCFD CTD SC3 | BiGRU | CNN+Trans dual | 3 |
| `trans_rcfd_cgd` | Trans RCFD CGD | Transformer | CNN+BiGRU dual | 0 |
| `trans_rcfd_cgd_supcon_mtl` | Trans RCFD CGD SC1 | Transformer | CNN+BiGRU dual | 1 |
| `trans_rcfd_cgd_supcon2_mtl` | Trans RCFD CGD SC2 | Transformer | CNN+BiGRU dual | 2 |
| `trans_rcfd_cgd_supcon3_mtl` | Trans RCFD CGD SC3 | Transformer | CNN+BiGRU dual | 3 |
| `trans_rcfd_ctd` | Trans RCFD CTD | Transformer | CNN+Trans dual | 0 |
| `trans_rcfd_ctd_supcon_mtl` | Trans RCFD CTD SC1 | Transformer | CNN+Trans dual | 1 |
| `trans_rcfd_ctd_supcon2_mtl` | Trans RCFD CTD SC2 | Transformer | CNN+Trans dual | 2 |
| `trans_rcfd_ctd_supcon3_mtl` | Trans RCFD CTD SC3 | Transformer | CNN+Trans dual | 3 |

**16 models total.**

```bash
for SC in 0 1 2 3; do
    SARG=(); [ "$SC" -gt 0 ] && SARG=(--supcon "$SC")
    python -u 03_main_training.py ... --condreg "${SARG[@]}"
done
```

---

### CRF structured output (`crf` group) — `--crf`

Structured output variants that replace the sigmoid head with a joint-state NLL loss over
all 2³ = 8 label combinations. All models expose `predict_binary()` (argmax over states)
and `predict_marginals()` (`softmax(logits) @ decode_mat`), compatible with
`lab_threshold_search.ipynb` for per-label threshold tuning.

**Result file:** `classification_performances_ml_crf[_10fold].joblib`

```bash
for SC in 0 1 2 3; do
    SARG=(); [ "$SC" -gt 0 ] && SARG=(--supcon "$SC")
    python -u 03_main_training.py ... --crf "${SARG[@]}"
done
```

#### Pooled-z heads

Input to the CRF head is the fully pooled 96-dim `z` from the dual backbone.

**CRF-MRF** (full-state 8-class NLL):
State logits from `Dense(8)(z)`. NLL = `logsumexp(logits) − logit[true_state]`.

| Model key | Print name | Backbone | SC |
|---|---|---|---|
| `cnn_gru_dual_crf_mrf` | CNN+GRU CRF-MRF SC0 | CNN+BiGRU dual | 0 |
| `cnn_gru_dual_crf_mrf_supcon` | CNN+GRU CRF-MRF SC1 | CNN+BiGRU dual | 1 |
| `cnn_gru_dual_crf_mrf_supcon2` | CNN+GRU CRF-MRF SC2 | CNN+BiGRU dual | 2 |
| `cnn_gru_dual_crf_mrf_supcon3` | CNN+GRU CRF-MRF SC3 | CNN+BiGRU dual | 3 |
| `cnn_trans_dual_crf_mrf` | CNN+Tr CRF-MRF SC0 | CNN+Trans dual | 0 |
| `cnn_trans_dual_crf_mrf_supcon` | CNN+Tr CRF-MRF SC1 | CNN+Trans dual | 1 |
| `cnn_trans_dual_crf_mrf_supcon2` | CNN+Tr CRF-MRF SC2 | CNN+Trans dual | 2 |
| `cnn_trans_dual_crf_mrf_supcon3` | CNN+Tr CRF-MRF SC3 | CNN+Trans dual | 3 |

**CRF-chain** (linear-chain, Viterbi decoding):
Emission logits `(batch, 3, 2)` + shared `(2,2)` transition matrix; NLL via forward algorithm.
Label order fixed: index 0 = KPC, 1 = NDM, 2 = VIM.

| Model key | Print name | Backbone | SC |
|---|---|---|---|
| `cnn_gru_dual_crf_chain` | CNN+GRU CRF-chain SC0 | CNN+BiGRU dual | 0 |
| `cnn_gru_dual_crf_chain_supcon` | CNN+GRU CRF-chain SC1 | CNN+BiGRU dual | 1 |
| `cnn_gru_dual_crf_chain_supcon2` | CNN+GRU CRF-chain SC2 | CNN+BiGRU dual | 2 |
| `cnn_gru_dual_crf_chain_supcon3` | CNN+GRU CRF-chain SC3 | CNN+BiGRU dual | 3 |
| `cnn_trans_dual_crf_chain` | CNN+Tr CRF-chain SC0 | CNN+Trans dual | 0 |
| `cnn_trans_dual_crf_chain_supcon` | CNN+Tr CRF-chain SC1 | CNN+Trans dual | 1 |
| `cnn_trans_dual_crf_chain_supcon2` | CNN+Tr CRF-chain SC2 | CNN+Trans dual | 2 |
| `cnn_trans_dual_crf_chain_supcon3` | CNN+Tr CRF-chain SC3 | CNN+Trans dual | 3 |

**Pooled-z subtotal: 16 models.**

#### Cross-attention v2 + CRF heads (CAttn+CRF)

The deep CAttn-V2 backbone (`kv_seq=(batch,T,32)`, 2 CA blocks + FFN) produces per-label
attended query vectors `queries: (batch, 3, 64)`. Three projection variants map these to
`(batch, 8)` state logits (identical NLL loss as CRF-MRF):

| Head | Mechanism | Head params |
|---|---|---|
| `flat` | `Flatten(queries) → Dense(8)` — all queries mixed in the weight matrix | 1,544 |
| `factored` | `Dense(1)` per label → `einsum('bj,sj->bs', emit, decode_mat) + state_bias(8)` | 203 |
| `bilinear` | `einsum('bjd,sjd->bs', queries, state_embs(8,3,64))` — per-label-per-state bilinear | 1,536 |

Available for both CGD (`cnn_gru_dual`) and CTD (`cnn_trans_dual`) backbones.

**CGD backbone (CNN+BiGRU dual):**

| Model key | Print name | Head | SC |
|---|---|---|---|
| `cnn_gru_dual_cross_attn_v2_crf_flat` | CNN+GRU CAttn-V2 CRF-flat SC0 | flat | 0 |
| `cnn_gru_dual_cross_attn_v2_crf_flat_supcon` | CNN+GRU CAttn-V2 CRF-flat SC1 | flat | 1 |
| `cnn_gru_dual_cross_attn_v2_crf_flat_supcon2` | CNN+GRU CAttn-V2 CRF-flat SC2 | flat | 2 |
| `cnn_gru_dual_cross_attn_v2_crf_flat_supcon3` | CNN+GRU CAttn-V2 CRF-flat SC3 | flat | 3 |
| `cnn_gru_dual_cross_attn_v2_crf_factored` | CNN+GRU CAttn-V2 CRF-factored SC0 | factored | 0 |
| `cnn_gru_dual_cross_attn_v2_crf_factored_supcon` | CNN+GRU CAttn-V2 CRF-factored SC1 | factored | 1 |
| `cnn_gru_dual_cross_attn_v2_crf_factored_supcon2` | CNN+GRU CAttn-V2 CRF-factored SC2 | factored | 2 |
| `cnn_gru_dual_cross_attn_v2_crf_factored_supcon3` | CNN+GRU CAttn-V2 CRF-factored SC3 | factored | 3 |
| `cnn_gru_dual_cross_attn_v2_crf_bilinear` | CNN+GRU CAttn-V2 CRF-bilinear SC0 | bilinear | 0 |
| `cnn_gru_dual_cross_attn_v2_crf_bilinear_supcon` | CNN+GRU CAttn-V2 CRF-bilinear SC1 | bilinear | 1 |
| `cnn_gru_dual_cross_attn_v2_crf_bilinear_supcon2` | CNN+GRU CAttn-V2 CRF-bilinear SC2 | bilinear | 2 |
| `cnn_gru_dual_cross_attn_v2_crf_bilinear_supcon3` | CNN+GRU CAttn-V2 CRF-bilinear SC3 | bilinear | 3 |

**CTD backbone (CNN+Trans dual):**

| Model key | Print name | Head | SC |
|---|---|---|---|
| `cnn_trans_dual_cross_attn_v2_crf_flat` | CNN+Tr CAttn-V2 CRF-flat SC0 | flat | 0 |
| `cnn_trans_dual_cross_attn_v2_crf_flat_supcon` | CNN+Tr CAttn-V2 CRF-flat SC1 | flat | 1 |
| `cnn_trans_dual_cross_attn_v2_crf_flat_supcon2` | CNN+Tr CAttn-V2 CRF-flat SC2 | flat | 2 |
| `cnn_trans_dual_cross_attn_v2_crf_flat_supcon3` | CNN+Tr CAttn-V2 CRF-flat SC3 | flat | 3 |
| `cnn_trans_dual_cross_attn_v2_crf_factored` | CNN+Tr CAttn-V2 CRF-factored SC0 | factored | 0 |
| `cnn_trans_dual_cross_attn_v2_crf_factored_supcon` | CNN+Tr CAttn-V2 CRF-factored SC1 | factored | 1 |
| `cnn_trans_dual_cross_attn_v2_crf_factored_supcon2` | CNN+Tr CAttn-V2 CRF-factored SC2 | factored | 2 |
| `cnn_trans_dual_cross_attn_v2_crf_factored_supcon3` | CNN+Tr CAttn-V2 CRF-factored SC3 | factored | 3 |
| `cnn_trans_dual_cross_attn_v2_crf_bilinear` | CNN+Tr CAttn-V2 CRF-bilinear SC0 | bilinear | 0 |
| `cnn_trans_dual_cross_attn_v2_crf_bilinear_supcon` | CNN+Tr CAttn-V2 CRF-bilinear SC1 | bilinear | 1 |
| `cnn_trans_dual_cross_attn_v2_crf_bilinear_supcon2` | CNN+Tr CAttn-V2 CRF-bilinear SC2 | bilinear | 2 |
| `cnn_trans_dual_cross_attn_v2_crf_bilinear_supcon3` | CNN+Tr CAttn-V2 CRF-bilinear SC3 | bilinear | 3 |

**CAttn+CRF subtotal: 24 models.**

**CRF group total: 40 models.**

---

### Source separation pretraining (`source_sep` group) — `--source_sep`

Encoder–decoder pretraining that forces the network to learn per-target kinetic decomposition
before classification. Requires running `03b_source_sep_pretraining.py` first (Phase 1).

**Result file:** `classification_performances_ml_source_sep[_10fold].joblib`

| Model key | Print name | Head | SC |
|---|---|---|---|
| `cnn_gru_source_sep` | SrcSep SC0 | Dense(32,relu)→Dense(3,sigmoid) | 0 |
| `cnn_gru_source_sep_supcon` | SrcSep SC1 | sigmoid + SupCon(fused) | 1 |
| `cnn_gru_source_sep_crf` | SrcSep CRF SC0 | Dense(8) joint-state NLL | 0 |
| `cnn_gru_source_sep_crf_supcon` | SrcSep CRF SC1 | joint-state NLL + SupCon(fused) | 1 |

**4 models total** (only SC0 and SC1 supported — Phase 1 does not train SC2/3 encoders).

**Step 1 — Phase 1 pretraining:**

```bash
python -u 03b_source_sep_pretraining.py \
    --task_id 0 \
    --exp_folder /vol/bitbucket/gk225/POC_DDM_datasets/LAB_Multiplex \
    --d_shared 16 --d_target 10 \
    --epochs_p1 200 --lambda_cons 1.0 --lambda_anch 0.5 --nn_k 5 \
    --validate
```

Outputs: `{exp_path}/source_sep_encoder_weights.weights.h5`

**Step 2 — Phase 2 fine-tuning:**

```bash
for SC in 0 1; do
    python -u 03_main_training.py ... --source_sep --supcon "$SC"
done
```

---

## All model groups at a glance

| Flag | Group | # models | Result file suffix |
|---|---|---|---|
| _(none)_ | `default` | 14 | _(none)_ |
| `--cross_attn` | `cross_attn` | 8 | `_cross_attn` |
| `--cross_attn_v2` | `cattn_v2` | 16 | `_cattn_v2` |
| `--cross_attn_auxdet` | `auxdet` | 4 | `_auxdet` |
| `--cross_attn_quercon` | `quercon` | 4 | `_quercon` |
| `--condreg` | `condreg` | 16 | `_condreg` |
| `--crf` | `crf` | 40 (16 pooled-z + 24 CAttn+CRF) | `_crf` |
| `--source_sep` | `source_sep` | 4 | `_source_sep` |
| **Total** | | **106** | |

---

## Result file paths

Each flag group writes to its own joblib to allow parallel SLURM jobs without file races:

```
{exp_path}/classification_performances_ml.joblib               ← default
{exp_path}/classification_performances_ml_cross_attn.joblib    ← --cross_attn
{exp_path}/classification_performances_ml_cattn_v2.joblib      ← --cross_attn_v2
{exp_path}/classification_performances_ml_auxdet.joblib        ← --cross_attn_auxdet
{exp_path}/classification_performances_ml_quercon.joblib       ← --cross_attn_quercon
{exp_path}/classification_performances_ml_condreg.joblib       ← --condreg
{exp_path}/classification_performances_ml_crf.joblib           ← --crf (all CRF variants)
{exp_path}/classification_performances_ml_source_sep.joblib    ← --source_sep
```

Append `_10fold` before `.joblib` when `--n_splits > 1` (e.g. `--n_splits 5`).

Mapping defined in `config_multiplex.RESULT_FILE_BY_FLAG` / `RESULT_10FOLD_FILE_BY_FLAG`.
`MODEL_FLAG_MAP[key]` gives the group flag for any model key.

---

## SLURM scripts

All scripts are in `_slurm_jobs/`. Run from `main/multiplex/`:

| Script | What it trains | Wall time | Array |
|---|---|---|---|
| `lab_multiplex_training.sh` | Standard + CAttn + CAttn-v2 + AuxDet + QuerCon + RCFD + CRF (all flags, sequential) | 72 h | — |
| `lab_multiplex_crf_training.sh` | CRF-MRF + CRF-chain only (pooled-z, SC0–3) | 24 h | — |
| `lab_multiplex_cattn_crf_training.sh` | CAttn+CRF only (flat/factored/bilinear × CGD+CTD × SC0–3, sequential, **5-fold**) | 24 h | — |
| `lab_multiplex_save_models.sh` | All model families, **1-fold (90/10)**, saves fold-0 `.keras` | 16 h | `0-6` (family) |
| `lab_multiplex_source_sep_training.sh` | Phase 1 pretraining + Phase 2 fine-tuning | 4 h | — |

```bash
# Train only the new CAttn+CRF models (5-fold, SC0–3 sequential, single job):
sbatch _slurm_jobs/lab_multiplex_cattn_crf_training.sh

# Train pooled-z CRF baselines (5-fold):
sbatch _slurm_jobs/lab_multiplex_crf_training.sh

# Save fold-0 models for all families (90/10 single split):
sbatch _slurm_jobs/lab_multiplex_save_models.sh
```

---

## Post-training analysis

`notebooks/lab_threshold_search.ipynb` — merges all per-group result files and runs a
vectorised per-label sigmoid threshold grid search (21³ = 9 261 combos). Set `USE_10FOLD`
and `FILTER_KEY` in Cell 3.

---

## Other flags

| Flag | Effect |
|---|---|
| `--n_splits INT` | CV folds. `1` = single 90/10 StratifiedShuffleSplit; `>1` = stratified k-fold |
| `--force_rerun` | Clears cached results for the current group before training |
| `--rerun_models KEY [KEY ...]` | Restricts `--force_rerun` clearing to named model keys only |
| `--fast_mode` | Disables strict TF op-determinism for faster training |
| `--threshold FLOAT` | Sigmoid threshold for binary prediction (default 0.5) |

Models are saved automatically to `{exp_path}/models/` after each run (fold 0 only, as
`{key}_{filter}_{mode_name}.keras`).

---

## Brainstorming docs

`_brainstorming/` — architecture notes, ablation designs, and postmortems:

| File | Contents |
|---|---|
| `20260716-MULTILABEL-MULTIPLEX-INIT.md` | Initial implementation reference — backbone choices, loss functions, overall design |
| `20260717-CROSS-ATTN-HEAD.md` | Label query cross-attention head design (`--cross_attn`) |
| `20260717-CROSS-ATTN-IMPROVEMENTS.md` | Root-cause analysis and v2 ablation design |
| `20260717-OTHER-ML-APPROACHES.md` | Broader multi-label ML landscape |
| `20260717-PER-GROUP-RESULT-FILES.md` | Race-condition fix design notes |
| `20260718-AUX-CT-CRF-IMPLEMENTATION.md` | AuxDet and CRF-MRF implementation guide |
| `20260718-SOURCE-SEPARATION-PRETRAINING.md` | Source-separation pretraining design |
| `20260718-SOURCE-SEP-RUN1-POSTMORTEM.md` | Postmortem: why source separation fails (Runs 1–4) |
| `20260719-CATTN-CRF-BRAINSTORM.md` | CAttn+CRF combination design (flat/factored/bilinear) |
