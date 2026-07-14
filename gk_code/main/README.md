# 03_main_training.py

Trains all classification (and optionally regression) models for one experiment folder, determined by `--task_id`. Results are incrementally checkpointed into `classification_performances*.joblib`.

---

## Required args

| Arg | Default | Description |
|---|---|---|
| `--task_id INT` | `0` | Index into the sorted list of experiment sub-folders |
| `--exp_folder PATH` | config default | Root folder containing the experiment sub-folders |
| `--n_splits INT` | `1` | CV folds. `1` = single 90/10 split; `>1` = stratified k-fold (writes to a separate `*_10fold*` results file) |
| `--curve_type` | `ori_curve ori_curve_avg ori_curve_wavelet_sym8` | Which curve variant(s) to train on. Pass one or more values. |

---

## Model family — flag combinations

The three flags `--mtl`, `--supcon`, `--mtl_cl`, `--condreg` combine to select a model family. They are mutually exclusive in effect (only one family runs per invocation).

### Standard (no MTL flags)

```bash
python 03_main_training.py --task_id 0 --exp_folder ...
```

Trains: `knn`, `cnn`, `gru`, `transformer`, `cnn_gru_dual`, `cnn_trans_dual`, `cnn_gru_dual_cosine_recon`, `cnn_gru_dual_attn_recon`.  
Single classification head. No regression.

---

### MTL — `--mtl`

```bash
python 03_main_training.py ... --mtl
```

Trains the same backbones with a second regression head that predicts concentration.  
Models: `cnn_mtl`, `gru_mtl`, `transformer_mtl`, `cnn_gru_dual_mtl`, `cnn_trans_dual_mtl`, `cnn_gru_dual_cosine_recon_mtl`, `cnn_gru_dual_attn_recon_mtl`.

---

### SupCon — `--supcon {1|2|3}` (ST or MTL)

Adds contrastive projection heads on top of the backbone. Three projection variants:

| `--supcon` | Projection target | Models trained |
|---|---|---|
| `1` | Fused embedding | `*_supcon` (ST) or `*_supcon_mtl` (MTL) |
| `2` | CNN branch + seq branch | `*_supcon2` (ST) or `*_supcon2_mtl` (MTL) |
| `3` | CNN branch + seq branch + fused | `*_supcon3` (ST) or `*_supcon3_mtl` (MTL) |

Add `--mtl` to train the MTL variant (classification + regression + projection):

```bash
# ST + SupCon v1
python 03_main_training.py ... --supcon 1

# MTL + SupCon v3
python 03_main_training.py ... --mtl --supcon 3
```

---

### Curriculum Learning MTL — `--mtl_cl` (implies `--mtl`)

Two-phase training: Phase 1 trains the regression head only; Phase 2 unfreezes the backbone for joint classification + regression.

```bash
python 03_main_training.py ... --mtl_cl
```

Combine with `--supcon` to add contrastive heads in Phase 2:

| Command | Family |
|---|---|
| `--mtl_cl` | CL-MTL base |
| `--mtl_cl --supcon 1` | CL-MTL + SupCon v1 (fused projection) |
| `--mtl_cl --supcon 2` | CL-MTL + SupCon v2 (branch projections) |
| `--mtl_cl --supcon 3` | CL-MTL + SupCon v3 (branch + fused projections) |

Use `--cl_phase1_epochs INT` to fix the Phase 1 duration instead of auto-detecting convergence on `val_reg_mse`.

---

### RCFD — `--condreg` (implies `--mtl`)

Regression-Conditioned Feature Dual: a lightweight early encoder predicts concentration first, then conditions the dual-backbone embedding via FiLM before classification.

```bash
python 03_main_training.py ... --condreg
```

6 models per call (3 early encoders × 2 dual backbones: CNN/GRU/Trans × CGD/CTD).

Combine with `--supcon` to add contrastive heads on the conditioned embedding:

| Command | Family | Projection target |
|---|---|---|
| `--condreg` | RCFD base | — |
| `--condreg --supcon 1` | RCFD + SC1 | FiLM-conditioned embedding |
| `--condreg --supcon 2` | RCFD + SC2 | CNN branch + seq branch (pre-FiLM) |
| `--condreg --supcon 3` | RCFD + SC3 | CNN branch + seq branch + conditioned embedding |

---

## All model families at a glance

| Flags | Family | Outputs |
|---|---|---|
| _(none)_ | Standard ST | `[cls]` |
| `--mtl` | Standard MTL | `[cls, reg]` |
| `--supcon 1` | SupCon v1 ST | `[cls, proj]` |
| `--supcon 2` | SupCon v2 ST | `[cls, cnn_proj, seq_proj]` |
| `--supcon 3` | SupCon v3 ST | `[cls, cnn_proj, seq_proj, fused_proj]` |
| `--mtl --supcon 1` | SupCon v1 MTL | `[cls, reg, proj]` |
| `--mtl --supcon 2` | SupCon v2 MTL | `[cls, reg, cnn_proj, seq_proj]` |
| `--mtl --supcon 3` | SupCon v3 MTL | `[cls, reg, cnn_proj, seq_proj, fused_proj]` |
| `--mtl_cl` | CL-MTL | `[cls, reg]` |
| `--mtl_cl --supcon 1` | CL-MTL + SC1 | `[cls, reg, proj]` |
| `--mtl_cl --supcon 2` | CL-MTL + SC2 | `[cls, reg, cnn_proj, seq_proj]` |
| `--mtl_cl --supcon 3` | CL-MTL + SC3 | `[cls, reg, cnn_proj, seq_proj, fused_proj]` |
| `--condreg` | RCFD base | `[cls, reg]` |
| `--condreg --supcon 1` | RCFD + SC1 | `[cls, reg, proj]` |
| `--condreg --supcon 2` | RCFD + SC2 | `[cls, reg, cnn_proj, seq_proj]` |
| `--condreg --supcon 3` | RCFD + SC3 | `[cls, reg, cnn_proj, seq_proj, fused_proj]` |

---

## Other flags

| Flag | Effect |
|---|---|
| `--force_rerun` | Clears cached results for the current family before training. Other families' results are preserved. |
| `--rerun_models KEY [KEY ...]` | Restricts `--force_rerun` clearing and training to the named model keys only. No effect without `--force_rerun`. |
| `--training_mode native reference` | `native` trains on each dataset's own curves (default). `reference` trains on the original `ori_curves` using this dataset's outlier filters. Pass both to run both. |
| `--fast_mode` | Disables strict TF op-determinism for faster GRU/LSTM/Transformer training. Seeds are still set; reruns are not bit-exact. |
| `--k_neighbors INT` | Neighbour count for `cnn_gru_dual_cosine_recon` / `cnn_gru_dual_attn_recon`. Default `24`. |
| `--cl_phase1_epochs INT` | Fixed Phase 1 epoch count for CL-MTL. Omit to use auto-convergence on `val_reg_mse`. |
