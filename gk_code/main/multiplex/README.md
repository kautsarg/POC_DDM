# multiplex — Multi-Label PCR Classification

Trains multi-label classifiers on multiplex dPCR curves (`(T, 1)` → `{0,1}^n_targets`).
All scripts live under `main/multiplex/`; all results land in the experiment folder
(e.g. `LAB_Multiplex/01_ACA_qdPCR/`).

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

## Model families

All flags are mutually exclusive (one family per invocation). Combine with `--supcon {0|1|2|3}` to add contrastive projection heads.

### Standard (`default` group)

No extra flags. Trains single- and dual-branch backbones with a standard sigmoid classification head.

| `--supcon` | Models trained | Projection |
|---|---|---|
| 0 (omit) | `cnn`, `gru`, `transformer`, `cnn_gru_dual`, `cnn_trans_dual` | — |
| 1 | above + `*_supcon` variants | fused embedding |
| 2 | above + `*_supcon2` variants | CNN branch + seq branch |
| 3 | above + `*_supcon3` variants | CNN + seq + fused |

```bash
python -u 03_main_training.py ... --supcon 0   # base models only
python -u 03_main_training.py ... --supcon 1   # + SupCon v1
```

**Result file:** `classification_performances_ml[_10fold].joblib`

---

### Cross-attention (`cross_attn` group) — `--cross_attn`

Label query cross-attention head: `n_targets` learnable query vectors attend over the
pre-pooling GRU/Transformer sequence, followed by inter-label self-attention.
Trains 2 backbone variants (CGD = CNN+GRU dual, CTD = CNN+Transformer dual).

```bash
python -u 03_main_training.py ... --cross_attn --supcon 0
```

**Result file:** `classification_performances_ml_cross_attn[_10fold].joblib`

---

### Cross-attention v2 (`cattn_v2` group) — `--cross_attn_v2`

Controlled ablation of two root-cause fixes over the original cross-attn head:

| Variant | kv_seq depth | Head depth |
|---|---|---|
| `deepkv` | Second BiGRU + pos-enc | 1 CA block (shallow) |
| `deephead` | First BiGRU (shallow) | 2 CA blocks + FFN |
| `v2` (CGD) | Second BiGRU + pos-enc | 2 CA blocks + FFN |
| `v2` (CTD) | Transformer + pos-enc | 2 CA blocks + FFN |

Trains all 4 variants per SC level (16 models total across SC0–3).

```bash
for SC in 0 1 2 3; do
    python -u 03_main_training.py ... --cross_attn_v2 --supcon "$SC"
done
```

**Result file:** `classification_performances_ml_cattn_v2[_10fold].joblib`

---

### Cross-attention v2 + AuxDet (`auxdet` group) — `--cross_attn_auxdet`

Adds a per-block auxiliary BCE head tapped after each CA+LN sublayer (before inter-label SA).
Forces each label query to independently develop features; prevents free-riding across queries.

Loss: `BCE(cls_out) + λ_aux × Σ BCE(aux_block_i)`

```bash
for SC in 0 1 2 3; do
    python -u 03_main_training.py ... --cross_attn_auxdet --supcon "$SC"
done
```

**Result file:** `classification_performances_ml_auxdet[_10fold].joblib`

---

### Cross-attention v2 + QuerCon (`quercon` group) — `--cross_attn_quercon`

Per-label supervised contrastive loss on the attended query vectors.
After the final CA block, `queries_norm[:, j, :]` (L2-normalised, shape `(batch, query_dim)`)
is treated as an embedding for SupCon: positives = samples where `y[:, j] = 1`.

Loss: `(1 − λ_qc) × BCE + λ_qc × mean_j(SupCon(queries_norm[:, j, :], y[:, j]))`

Separate from backbone SupCon (which operates in the projection space).
Combine with `--supcon {1|2|3}` to add backbone projection loss on top.

```bash
for SC in 0 1 2 3; do
    python -u 03_main_training.py ... --cross_attn_quercon --supcon "$SC"
done
```

**Result file:** `classification_performances_ml_quercon[_10fold].joblib`

---

### RCFD (`condreg` group) — `--condreg`

Regression-Conditioned Feature Dual: a lightweight early encoder (GRU or Transformer)
predicts concentration first, then conditions the dual-backbone embedding via FiLM before
classification. Trains 4 combinations (GRU/Trans early encoder × CGD/CTD backbone).

```bash
for SC in 0 1 2 3; do
    python -u 03_main_training.py ... --condreg --supcon "$SC"
done
```

**Result file:** `classification_performances_ml_condreg[_10fold].joblib`

---

### CRF structured output (`crf` group) — `--crf`

Two structured output variants that model label correlations at training time.
Neither uses `--supcon`; both replace the sigmoid head with a structured loss.

**Option A — CRF-MRF** (`cnn_gru_dual_crf_mrf`, `cnn_trans_dual_crf_mrf`):
`Dense(8)` output head over all 2³ joint label states; NLL = `logsumexp(logits) − logit[true_state]`.
Inference: `predict_binary()` = argmax over states; `predict_marginals()` = `softmax(logits) @ decode_mat`.
State imbalance: multi-target combos (~6% each) are ~4× under-represented — pass `state_weights`
via `compute_crf_state_weights(y_binary)` to compensate.

**Option B — CRF-chain** (`cnn_gru_dual_crf_chain`, `cnn_trans_dual_crf_chain`):
Emission logits `(batch, 3, 2)` + learned shared `(2,2)` transition matrix; NLL via forward
algorithm; Viterbi for binary prediction; forward-backward for marginals.
Label order is fixed (index 0 = KPC, 1 = NDM, 2 = VIM).

Both variants expose `predict_marginals()` returning per-label `[0,1]` scores — fully compatible
with `lab_threshold_search.ipynb` for per-label threshold tuning.

Each variant trains at SC0–SC3 with `--supcon`:

| SC level | Loss |
|---|---|
| 0 | CRF NLL only |
| 1 | NLL + λ · SupCon(fused) |
| 2 | NLL + λ · (SupCon(CNN) + SupCon(seq)) |
| 3 | NLL + λ · (SupCon(CNN) + SupCon(seq) + SupCon(fused)) |

λ values follow RCFD precedent (additive, not weighted-complement): SC1 λ=0.1, SC2/3 λ_each=0.05/0.033.

```bash
for SC in 0 1 2 3; do
    python -u 03_main_training.py ... --crf --supcon "$SC"
done
```

**Result file:** `classification_performances_ml_crf[_10fold].joblib`

---

### Source separation pretraining (`source_sep` group) — `--source_sep`

Encoder–decoder pretraining that forces the network to learn per-target kinetic
decomposition _before_ classification.  Two scripts in sequence:

**Step 1 — Phase 1 pretraining** (`03b_source_sep_pretraining.py`)

Trains a compact encoder + 3 parametric decoders jointly on all curves with:

```
L = L_absent  +  λ_cons · L_consist  +  λ_anch · Σ_j L_anchor_j
```

| Loss | What it enforces |
|---|---|
| `L_absent` | absent channels → zero decoded amplitude |
| `L_consist` | Σ active decoded channels ≈ observed curve |
| `L_anchor_j` | active channel j close to its k-NN reference curves |

Anchor references = single-target-j curves from the dataset (no concentration needed).
All losses are in rendered curve space using the 5-parameter sigmoid model
(`Fm / (1 + exp(−Sc·(t−Cs)))^As + Fb`) that matches the preprocessing `sigmoid_5p`.

Encoder architecture: `Conv1D(16,5)` → `Conv1D(16,3,stride=2)` → `BiGRU(32)` →
`LayerNorm` → `Dense(d_shared + 3·d_target)`.  Bottleneck: 16 shared dims +
3 × 10 per-target dims = 46 dims total.

```bash
python -u 03b_source_sep_pretraining.py \
    --task_id 0 \
    --exp_folder /vol/bitbucket/gk225/POC_DDM_datasets/LAB_Multiplex \
    --d_shared 16 --d_target 10 \
    --epochs_p1 200 --lambda_cons 1.0 --lambda_anch 0.5 --nn_k 5 \
    --validate   # prints Spearman r(z_j[0], Ct) for each target
```

Outputs: `{exp_path}/source_sep_encoder_weights.weights.h5`

**Step 2 — Phase 2 fine-tuning** (`03_main_training.py --source_sep`)

Loads pretrained encoder, freezes its weights, trains a sigmoid classification
head (`Dense(32,relu) → Dense(3,sigmoid)`).

```bash
for SC in 0 1; do
    python -u 03_main_training.py ... --source_sep --supcon "$SC"
done
```

| SC level | Model key |
|---|---|
| 0 | `cnn_gru_source_sep` |
| 1 | `cnn_gru_source_sep_supcon` |

**Result file:** `classification_performances_ml_source_sep[_10fold].joblib`

---

## All model families at a glance

| Flag(s) | Group | # models (SC0–3) | Result file suffix |
|---|---|---|---|
| _(none)_ | `default` | 14 | _(none)_ |
| `--cross_attn` | `cross_attn` | 8 | `_cross_attn` |
| `--cross_attn_v2` | `cattn_v2` | 16 | `_cattn_v2` |
| `--cross_attn_auxdet` | `auxdet` | 4 | `_auxdet` |
| `--cross_attn_quercon` | `quercon` | 4 | `_quercon` |
| `--condreg` | `condreg` | 16 | `_condreg` |
| `--crf` | `crf` | 16 | `_crf` |
| `--source_sep` | `source_sep` | 2 | `_source_sep` |
| **Total** | | **80** | |

---

## Result file naming convention

Each training flag writes to its own joblib so parallel SLURM jobs never race on the
same file:

```
classification_performances_ml.joblib               ← default group
classification_performances_ml_cross_attn.joblib    ← --cross_attn
classification_performances_ml_cattn_v2.joblib      ← --cross_attn_v2
classification_performances_ml_auxdet.joblib        ← --cross_attn_auxdet
classification_performances_ml_quercon.joblib       ← --cross_attn_quercon
classification_performances_ml_condreg.joblib       ← --condreg
classification_performances_ml_crf.joblib           ← --crf
classification_performances_ml_source_sep.joblib    ← --source_sep
```

Append `_10fold` before `.joblib` when `--n_splits > 1`.

The mapping is defined in `config_multiplex.RESULT_FILE_BY_FLAG` (and `RESULT_10FOLD_FILE_BY_FLAG`).
`MODEL_FLAG_MAP` maps every model key to its group flag — used by `split_ml_results.py`
and derivable programmatically via `_flag_for_model()`.

### Splitting an existing monolithic file

If you have a legacy monolithic joblib (all groups in one file), split it with:

```bash
python split_ml_results.py --10fold \
    --exp_path /vol/bitbucket/gk225/POC_DDM_datasets/LAB_Multiplex/01_ACA_qdPCR

# Dry-run first to verify routing without writing:
python split_ml_results.py --10fold --dry_run ...
```

---

## SLURM

`_slurm_jobs/lab_multiplex_training.sh` — submit for each group independently:

```bash
# Run 3 groups in parallel (each writes to its own file — no race):
sbatch --export=ALL lab_multiplex_training.sh   # cross_attn loop
sbatch --export=ALL lab_multiplex_training.sh   # cattn_v2 loop
sbatch --export=ALL lab_multiplex_training.sh   # auxdet / quercon loops
```

Edit which `for SC in ...` loops are active before each submission.

`_slurm_jobs/lab_multiplex_source_sep_training.sh` — runs Phase 1 pretraining then Phase 2 fine-tuning:

```bash
sbatch --export=ALL lab_multiplex_source_sep_training.sh
```

Phase 1 is skipped automatically if `source_sep_encoder_weights.weights.h5` already exists.
Use `--force_rerun` inside the script to retrain from scratch.

---

## Post-training analysis

`notebooks/lab_threshold_search.ipynb` — merges all per-group result files and runs a
vectorised per-label sigmoid threshold grid search (21³ = 9 261 combos, < 5 s for all
62 models). Set `USE_10FOLD` and `FILTER_KEY` in Cell 3.

`_brainstorming/` — architecture notes, ablation designs, and approach comparisons:
- `20260717-CROSS-ATTN-IMPROVEMENTS.md` — root-cause analysis + v2 ablation design
- `20260717-OTHER-ML-APPROACHES.md` — broader multi-label ML landscape
- `20260717-PER-GROUP-RESULT-FILES.md` — race-condition fix design notes
- `20260718-AUX-CT-CRF-IMPLEMENTATION.md` — implementation guide for Auxiliary Ct Head (6a) and Structured Output CRF (1c)
- `20260718-SOURCE-SEPARATION-PRETRAINING.md` — Compositional / source-separation pretraining design (Blind Source Separation for co-amplification)

---

## Other flags

| Flag | Effect |
|---|---|
| `--n_splits INT` | CV folds. `1` = single 90/10 split; `>1` = stratified k-fold |
| `--force_rerun` | Clears cached results for the current group before training |
| `--rerun_models KEY [KEY ...]` | Restricts `--force_rerun` to named model keys only |
| `--fast_mode` | Disables strict TF op-determinism for faster training |
| `--threshold FLOAT` | Sigmoid threshold for binary prediction (default 0.5) |

`03b_source_sep_pretraining.py` extra flags:

| Flag | Effect |
|---|---|
| `--validate` | After Phase 1, print Spearman r(z_j[0], Ct) for each target (gate: \|r\| > 0.65) |
| `--force_rerun` | Retrain Phase 1 even if encoder weights already exist |
| `--d_shared INT` | Shared latent dims (default 16) |
| `--d_target INT` | Per-target latent dims (default 10; total bottleneck = d_shared + 3·d_target) |
