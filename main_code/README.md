# main_code

Self-contained CHIP + LAB pipeline, split out of `gk_code/main` down to the
agreed final methodology. Nothing here imports from outside this folder;
`gk_code/main`, `titan_v6`, and existing datasets are untouched.

```
main_code/
├── main_chip.py            # dispatcher: preprocess | train | cross-dataset | saliency
├── main_lab.py              # dispatcher: preprocess | train
├── config.py                 # shared config (paths, model registry, chip groups, labels)
├── chip/                     # CHIP (POC_DDM_final) pipeline
├── lab/                       # LAB (LAB_DDM_paper) pipeline
├── utils/                      # shared engine + titan_v6 copy (no outside imports)
└── notebooks/{chip,lab}/        # analysis notebooks
```

Run everything from inside `main_code/` (or with `main_code` on `PYTHONPATH`) using
the same venv as the rest of the project (`venv_poc_ddm`, TF 2.16.2).

---

## CHIP: `main_chip.py`

### `preprocess`

```
python main_chip.py preprocess --exp_folder <dir> --task_id 0 [--force_rerun] [--run_outlier_detection]
```

- `--exp_folder` — parent folder of chip subfolders (default `config.DEFAULT_EXP_FOLDER`,
  i.e. `POC_DDM_final`). `--task_id` indexes into the alphabetically sorted subfolders
  (array-job style).
- `--run_outlier_detection` — also runs the LSTM-AE (global) outlier pass in the same
  process. Off by default, and TensorFlow is only imported when this flag is set, so a
  plain preprocessing run stays lightweight.
- Everything else (n_wells=10, titan v6 pixel layout, drop_pc=True, sg_p4 denoising,
  normalization) is fixed, not a CLI param.

**Writes:**
- `{exp_folder}/{chip}/curve_for_training.joblib` — curves (`ori_curves`,
  `ori_curves_norm`, `ori_curves_sg_p4`, `ori_curves_sg_p4_norm`), kinetic features,
  labels, metadata, and a `pc_wells` snapshot of the dropped PC curves.
- `{exp_folder}/{chip}/pretrained_encoders/` — LSTM-AE encoder, only if
  `--run_outlier_detection` was passed.

### `train`

```
python main_chip.py train --exp_folder <dir> --task_id 0 [--force_rerun]
    [--n_splits 5] [--curve_type ori_curve ori_curve_norm ...]
    [--models kNN CNN BiGRU Transformer cnn_gru_dual cnn_gru_dual_attn_recon]
    [--outlier_filter none ...]
```

Trains per-chip (single `task_id`-selected chip), native mode only, `k_neighbors=24`
fixed. Defaults to all 4 curve-type aliases and all 6 models.

**Writes:**
- `{exp_folder}/{chip}/classification_performances.joblib` (or
  `classification_performances_10fold.joblib` if `--n_splits > 1`).
- `{exp_folder}/{chip}/model_interpretation/` — saved `.keras` models
  (`{model}_{filter}_{curve_type}_model.keras`) and XAI snapshots.
- `{exp_folder}/{chip}/model_performance/` — result plots.

### `cross-dataset`

```
python main_chip.py cross-dataset --exp_folder <dir> --task_id 0 [--force_rerun]
    [--curve_type ...] [--models ...] [--mode lofo|random_split|kfold]
    [--n_splits 5] [--outlier_filter ...] [--batch_size 2048]
    [--train_full] [--test_size 0.1] [--held_out_chip <name>]
```

`--task_id` indexes into `config.CROSS_DATASET_GROUPS` (a group is a fixed named list
of chip subfolders — see `config.py`). LOFO by default: one fold per chip in the group,
each holding that chip out. Curve alignment is always `pc_ttp` (fixed).

`--models` choices: `cnn_gru_dual`, `cnn_gru_dual_attn_recon`, their `_supcon1` /
`_supcon3` / `_dann` variants, and `cnn_gru_dual_pc_recentering` /
`cnn_gru_dual_attn_recon_pc_recentering`.

**pc_recentering** is inference-time only — it does *not* train a separate model. It
reuses the matching base model's saved `.keras` file, shifts its embeddings by
`(training-pool PC-well mean embedding − held-out chip's own PC-well mean embedding)`,
and re-runs the model's unmodified classification head. Requesting
`cnn_gru_dual_attn_recon_pc_recentering` trains `cnn_gru_dual_attn_recon` (if not
already requested) and adds a second cached result entry alongside it — one `.keras`
file on disk, two evaluated result entries.

**Writes**, all under
`{exp_folder}/cross_dataset_cv/{group}/curve_alignment_pc_ttp/anchor_min/`:
- `{filter_token}/cross_dataset_classification_performances_{mode}_{curve_type}_{model}.joblib`
  — per-model result partitions (load with `utils/cross_dataset_result_io.load_partitioned`).
- `model_interpretation/lofo_{chip}/` — saved `.keras` models per fold, XAI snapshots,
  and (for pc_recentering models) `pc_reference_embedding_{model}_{filter}_{curve_type}.joblib`.
- `model_performance_{curve_type}/` — result plots.
- `cross_dataset_resampler_classification_performances_{curve_type}.joblib`,
  `cross_dataset_pc_ttp_recipe_{curve_type}.joblib` — curve alignment artifacts.

### `saliency`

```
python main_chip.py saliency --exp_folder <dir> [--curve_type ori_curve_norm ori_curve_sg_p4_norm]
    [--n_dims 25] [--top_n 5] [--batch_n 512] [--seed 42]
```

Runs integrated-gradients-style saliency for `cnn_gru_dual` / `cnn_gru_dual_attn_recon`
against models saved by `train` (per-chip `model_interpretation/`, **not**
`cross-dataset`'s LOFO layout — matches the original ablation4 script's dependency).

**Writes:** `{exp_folder}/{chip}/xai_saliency/`.

---

## LAB: `main_lab.py`

### `preprocess`

```
python main_lab.py preprocess --exp_folder <dir> --task_id 0 [--force_rerun] [--one_to_one]
```

Default `--exp_folder` is `config.LAB_EXP_FOLDER` (`LAB_DDM_paper`). Only these four
params are exposed; normalization/wavelet/TTP-alignment options from the original
script are removed.

**Writes:** `{exp_folder}/{subfolder}/curve_for_training.joblib`.

### `train`

```
python main_lab.py train --exp_folder <dir> --task_id 0 [--force_rerun]
    [--curve_type ori_curve] [--n_splits 5] [--batch_size 512]
    [--models kNN CNN BiGRU Transformer cnn_gru_dual cnn_gru_dual_attn_recon]
```

`cnn_gru_dual_attn_recon` is soft-skipped for LAB data (no pixel-grid metadata to build
spatial neighbor stacks from) — expected, not a bug.

**Writes:**
- `{exp_folder}/{subfolder}/ablations/model_comparison_performances.joblib`.
- `{exp_folder}/{subfolder}/ablations/model_interpretation/` — saved `.keras` models.

---

## Notebooks (`notebooks/{chip,lab}/`)

Run with `main_code/` as the working directory (each notebook's first code cell sets
up `sys.path`).

- `chip/1_curve_preprocessing_walkthrough.ipynb` — walks the titan_v6 pixel/well
  linearisation steps behind `chip/preprocessing.py`.
- `chip/3_denoising_comparison.ipynb` — compares raw vs. sg_p4-denoised curves.
- `chip/4_visualise_recon_layer.ipynb` — visualises `cnn_gru_dual_attn_recon`'s
  reconstruction/attention layers.
- `chip/0_learning_curve_analysis_cross_dataset.ipynb` — learning-curve analysis over
  `cross-dataset` LOFO results.
- `chip/0_LOFO_INSPECTION.ipynb` — per-(group, curve_type, held-out chip) deep dive:
  data profiling (t-SNE, per-label curves, intra/inter-chip similarity) plus model
  validation (confusion matrices, embedding t-SNE/UMAP, MMD/Wasserstein/silhouette,
  confidence histograms) reading `cross-dataset`'s cached LOFO predictions directly —
  simplified from the original, which needed the out-of-scope
  `08_cross_dataset_predict_new_chip.py`/`07_attribution_vis_all.py` scripts.
- `lab/ablation_significance_analysis.ipynb` — statistical significance (Friedman/Nemenyi,
  McNemar) over `lab/training.py`'s model-comparison results. Trimmed to the
  ablation-1 model-comparison analysis only.

**Known scope caveats** (flagging since they affect what these notebooks can show today):
- `0_learning_curve_analysis_cross_dataset.ipynb` points at a `POC_DDM_final_nc_subtract`
  sibling folder that this pipeline no longer produces (`nc_subtract` is hardcoded off
  in `chip/preprocessing.py`) — you'd need to have generated that folder with the old
  pipeline, or repoint `EXP_FOLDER` in the notebook's config cell.
- `4_visualise_recon_layer.ipynb` had its pre/post-nc_subtract PC-curve comparison cell
  removed for the same reason.

---

## SLURM jobs (`slurm_jobs/`)

Submit from inside `slurm_jobs/` (logs land in `slurm_jobs/logs/<job-name>/`), matching
the convention in `gk_code/main/slurm_jobs/`. One script per stage, plus a chained
`*_full_pipeline.sh` per side:

- `chip_preprocess.sh` / `chip_train.sh` / `chip_cross_dataset.sh` / `chip_saliency.sh`
- `chip_full_pipeline.sh` — preprocess → train, array over chips (one task per chip).
  Saliency needs every chip's models at once, so it isn't chained inside the array —
  run it as a dependent job instead:
  ```
  JOBID=$(sbatch --parsable slurm_jobs/chip_full_pipeline.sh)
  sbatch --dependency=afterok:$JOBID slurm_jobs/chip_saliency.sh
  ```
- `lab_preprocess.sh` / `lab_train.sh`
- `lab_full_pipeline.sh` — preprocess → train, array over LAB subfolders.

`--array` bounds are hardcoded to the current folder counts (7 chip subfolders, 15 LAB
subfolders) — check `ls $EXP_FOLDER | wc -l` and adjust before submitting if that's
changed. `chip_cross_dataset.sh` takes a single `TASK_ID` (index into
`config.CROSS_DATASET_GROUPS`, editable at the top of the script) rather than an array,
since one cross-dataset run already covers every chip in that group.

## Config (`config.py`)

Single shared config for both CHIP and LAB. Notable entries: `DEFAULT_EXP_FOLDER`
(`POC_DDM_final`), `LAB_EXP_FOLDER` (`LAB_DDM_paper`), `MODEL_KEY_MAP` (14 models,
including the two synthetic `*_pc_recentering` entries), `CROSS_DATASET_GROUPS` (10
named chip groups), `OUTLIER_FILTERS`. Feel free to edit chip groups / label mappings
here directly — this file is meant to be adjusted as the dataset grows.
