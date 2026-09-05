# `gk_code/main/` — experimentation folder

The finalised/production pipeline lives in [`main_code/`](../../main_code/README.md).

Scripts are numbered in run order and are normally launched as SLURM array jobs
(see `slurm_jobs/`). Each stage caches its result as a `.joblib` inside the
experiment folder, so re-running a stage is a no-op unless `--force_rerun` is
passed.

```
01  raw chip data          -> curve_for_training.joblib
 |  (curves + kinetic features + sigmoid fits)
02  + outlier detection    -> curve_for_training.joblib (outlier label columns)
 |
03  intra-chip training    -> classification_performances*.joblib
04  cross-dataset LOFO/CV  -> cross_dataset_cv/<group>/...
 |
05/06/07  reports          -> HTML / PNG
08        new-chip predict + statistical comparison
```

## Layout

| Path | Contents |
|---|---|
| `0*_*.py` | Numbered pipeline stages (below) |
| `config.py` | Paths, dataset groups, label/concentration mappings, model registry, feature lists |
| `utils/` | Shared code: `01_curve_preprocessing/`, `02_outlier_detection/`, `model_training/`, plus `sigmoid_fitting.py`, `safe_io.py`, `pipeline_utils.py`, `cross_dataset_result_io.py` |
| `notebook_rq/` | Thesis research-question analyses — `RQ1_*` (lab), `RQ2_*` (preprocessing/denoising/XAI), `RQ3_*` (cross-chip / LOCO), `RQALL_*` |
| `notebooks/` | ~47 ad-hoc exploration notebooks (dataset profiles, LOFO inspections, drift diagnostics, baselines) |
| `slurm_jobs/` | ~53 job scripts; `logs/<job-name>/` holds their output |
| `ablations/` | Standalone ablation experiments |
| `multiplex/`, `hpc_jobs/`, `figures/` | Multiplex variant, HPC submission helpers, exported figures |

## Pipeline stages

### `01_curve_preprocessing_v6.py`
Loads raw chip data (`v05`/`v06` via `titan_v6`, `v04` via `titan_v4`), reconstructs
each well's amplification curve, applies per-pixel baseline subtraction and
truncation, and builds the curve variants (`ori_curves`, `_norm`, `_sg_p4`,
`_sg_p4_norm`, optional wavelet/moving-average).

Key flags: `--task_id`, `--exp_folder`, `--n_wells`, `--n_a_type`, `--sg_p4`,
`--normalize_curves`, `--drop_pc`, `--nc_subtract` (writes to a sibling
`<exp_folder>_nc_subtract` tree), `--force_rerun`.
`01b_lab_curve_preprocessing.py` is the LAB-dataset equivalent.

### `02_outlier_detection_pipeline.py`
Extracts kinetic features per curve variant and runs the outlier filters
(autoencoder, spatial kNN/grid consistency, MSC/AMF), attaching per-sample
outlier labels as extra feature columns.

### `03_main_training.py`
Intra-chip training of the classifier suite. Model families are selected by flag
combination — see the **model-family reference** at the end of this file.
`03b`–`03g` are experimental spin-offs (GNN spatial, conditioned regression,
CCFD, architecture PoCs).

### `04_cross_dataset_training.py`
Cross-chip evaluation. Pools multiple experiment folders from
`config.CROSS_DATASET_GROUPS`, optionally aligns them on the PC well's TTP
(`--curve_alignment pc_ttp`), and runs leave-one-chip-out (LOFO), k-fold, or a
random split.

Key flags: `--task_id` (index into `CROSS_DATASET_GROUPS`), `--curve_type`,
`--models`, `--mode {lofo,random_split,kfold}`, `--outlier_filter`,
`--train_center_frac`, `--held_out_chip`, `--train_full`.

Writes to `<exp_folder>/cross_dataset_cv/<group>/[curve_alignment_pc_ttp/anchor_<a>/]`,
including `model_interpretation/<fold>/` with `.keras` models and XAI snapshots.

### `05` / `06` / `07` — reports
`05_outlier_visualization_report.py` (inlier vs outlier curves per well),
`06_model_prediction_report.py` and `06b_cross_dataset_prediction_report.py`
(prediction panels), `07_attribution_vis_all.py` and
`07b_cross_dataset_attribution_vis.py` (latent-space PCA/t-SNE, saliency).

### `08` — new-chip prediction and statistics
`08_cross_dataset_predict_new_chip.py` applies a trained cross-dataset model to
an unseen chip, replaying the saved alignment recipe; supports `--pc_recenter`
(latent alignment using the chip's own PC well).
`08_statistical_comparison.py` runs the significance testing.

### Standalone experiments
`lowo_custom_training.py` (leave-one-well-out with a custom held-out well set)
and `tempinv_lofo_training.py` (temperature-invariance / MTL head on the LOFO
split) reuse `04`'s functions by import without modifying it.

---

# Model-family reference — `03_main_training.py`

Trains all classification (and optionally regression) models for one experiment folder, determined by `--task_id`. Results are incrementally checkpointed into `classification_performances*.joblib`.

## Required args

| Arg | Default | Description |
|---|---|---|
| `--task_id INT` | `0` | Index into the sorted list of experiment sub-folders |
| `--exp_folder PATH` | config default | Root folder containing the experiment sub-folders |
| `--n_splits INT` | `1` | CV folds. `1` = single 90/10 split; `>1` = stratified k-fold (writes to a separate `*_10fold*` results file) |
| `--curve_type` | `ori_curve ori_curve_avg ori_curve_wavelet_sym8` | Which curve variant(s) to train on. Pass one or more values. |

## Model family — flag combinations

The flags `--mtl`, `--supcon`, `--mtl_cl`, `--condreg` combine to select a model family. They are mutually exclusive in effect (only one family runs per invocation).

### Standard (no MTL flags)

```bash
python 03_main_training.py --task_id 0 --exp_folder ...
```

Trains: `knn`, `cnn`, `gru`, `transformer`, `cnn_gru_dual`, `cnn_trans_dual`, `cnn_gru_dual_cosine_recon`, `cnn_gru_dual_attn_recon`.
Single classification head. No regression.

### MTL — `--mtl`

Same backbones with a second regression head predicting concentration.
Models: `cnn_mtl`, `gru_mtl`, `transformer_mtl`, `cnn_gru_dual_mtl`, `cnn_trans_dual_mtl`, `cnn_gru_dual_cosine_recon_mtl`, `cnn_gru_dual_attn_recon_mtl`.

### SupCon — `--supcon {1|2|3}` (ST or MTL)

Adds contrastive projection heads on top of the backbone:

| `--supcon` | Projection target | Models trained |
|---|---|---|
| `1` | Fused embedding | `*_supcon` (ST) or `*_supcon_mtl` (MTL) |
| `2` | CNN branch + seq branch | `*_supcon2` (ST) or `*_supcon2_mtl` (MTL) |
| `3` | CNN branch + seq branch + fused | `*_supcon3` (ST) or `*_supcon3_mtl` (MTL) |

```bash
python 03_main_training.py ... --supcon 1          # ST + SupCon v1
python 03_main_training.py ... --mtl --supcon 3    # MTL + SupCon v3
```

### Curriculum Learning MTL — `--mtl_cl` (implies `--mtl`)

Two-phase: Phase 1 trains the regression head only; Phase 2 unfreezes the backbone for joint classification + regression. Combine with `--supcon` to add contrastive heads in Phase 2. Use `--cl_phase1_epochs INT` to fix the Phase 1 duration instead of auto-detecting convergence on `val_reg_mse`.

### Label Consolidation — `--lbl_conc`

Fuses well label and concentration into a single classification target:

```
class = f"{label}_{concentration}"   ->  "KPC_1M", "KPC_100K", "NDM_10K", ...
```

Output neurons expand from `n_labels` -> `n_labels x n_concentrations`. Same architectures as ST, no regression head. Mutually exclusive with `--mtl`. Combine with `--supcon` for contrastive variants (SC1/SC2/SC3), where the contrastive loss uses the combined label as anchor.

### Staged SupCon — `--supcon_staged --supcon {1|2|3}`

Two-stage decoupled training:

| Stage | Loss | Trainable layers | End condition |
|---|---|---|---|
| 1 | SC loss only | All layers | `val_loss` plateau (auto) or `--cl_phase1_epochs N` |
| 2 | Cross-entropy only | `cls_feat`, `cls_out` | EarlyStopping (patience=100) |

CNN+GRU dual only. Mutually exclusive with `--mtl`. Without a val split, `--cl_phase1_epochs` is required (auto-plateau needs `val_loss`).

### RCFD — `--condreg` (implies `--mtl`)

Regression-Conditioned Feature Dual: a lightweight early encoder predicts concentration first, then conditions the dual-backbone embedding via FiLM before classification. 6 models per call (3 early encoders x 2 dual backbones). Combine with `--supcon` for contrastive heads on the conditioned embedding.

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
| `--lbl_conc` | LC SC0 | `[cls]` |
| `--lbl_conc --supcon 1` | LC SC1 | `[cls, proj]` |
| `--lbl_conc --supcon 2` | LC SC2 | `[cls, cnn_proj, seq_proj]` |
| `--lbl_conc --supcon 3` | LC SC3 | `[cls, cnn_proj, seq_proj, fused_proj]` |

## Other flags

| Flag | Effect |
|---|---|
| `--force_rerun` | Clears cached results for the current family before training. Other families' results are preserved. |
| `--rerun_models KEY [KEY ...]` | Restricts `--force_rerun` clearing and training to the named model keys only. No effect without `--force_rerun`. |
| `--training_mode native reference` | `native` trains on each dataset's own curves (default). `reference` trains on the original `ori_curves` using this dataset's outlier filters. Pass both to run both. |
| `--fast_mode` | Disables strict TF op-determinism for faster GRU/LSTM/Transformer training. Seeds are still set; reruns are not bit-exact. |
| `--k_neighbors INT` | Neighbour count for `cnn_gru_dual_cosine_recon` / `cnn_gru_dual_attn_recon`. Default `24`. |
| `--cl_phase1_epochs INT` | Fixed Phase 1 epoch count for CL-MTL. Omit to use auto-convergence on `val_reg_mse`. |
| `--lbl_conc` | Label Consolidation: combine label + concentration into a single classification target. Pure ST, mutually exclusive with `--mtl`. |
