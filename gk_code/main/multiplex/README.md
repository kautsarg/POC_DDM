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

## All model families at a glance

| Flag(s) | Group | # models (SC0–3) | Result file suffix |
|---|---|---|---|
| _(none)_ | `default` | 14 | _(none)_ |
| `--cross_attn` | `cross_attn` | 8 | `_cross_attn` |
| `--cross_attn_v2` | `cattn_v2` | 16 | `_cattn_v2` |
| `--cross_attn_auxdet` | `auxdet` | 4 | `_auxdet` |
| `--cross_attn_quercon` | `quercon` | 4 | `_quercon` |
| `--condreg` | `condreg` | 16 | `_condreg` |
| **Total** | | **62** | |

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

---

## Post-training analysis

`notebooks/lab_threshold_search.ipynb` — merges all per-group result files and runs a
vectorised per-label sigmoid threshold grid search (21³ = 9 261 combos, < 5 s for all
62 models). Set `USE_10FOLD` and `FILTER_KEY` in Cell 3.

`_brainstorming/` — architecture notes, ablation designs, and approach comparisons:
- `20260717-CROSS-ATTN-IMPROVEMENTS.md` — root-cause analysis + v2 ablation design
- `20260717-OTHER-ML-APPROACHES.md` — broader multi-label ML landscape
- `20260717-PER-GROUP-RESULT-FILES.md` — race-condition fix design notes

---

## Other flags

| Flag | Effect |
|---|---|
| `--n_splits INT` | CV folds. `1` = single 90/10 split; `>1` = stratified k-fold |
| `--force_rerun` | Clears cached results for the current group before training |
| `--rerun_models KEY [KEY ...]` | Restricts `--force_rerun` to named model keys only |
| `--fast_mode` | Disables strict TF op-determinism for faster training |
| `--threshold FLOAT` | Sigmoid threshold for binary prediction (default 0.5) |
