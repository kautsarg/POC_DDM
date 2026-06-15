# POC_DDM Pipeline

This pipeline turns raw Lacewing chip readouts into trained classifiers and
interpretability reports. Scripts live in `gk_code/main/`, are numbered in
run order, and are normally launched as a SLURM array job (see
`slurm_jobs/`). Shared dependencies live in `gk_code/main/utils/`, split into
`01_curve_preprocessing/`, `02_outlier_detection/`, and `model_training/`
subfolders matching the stages that use them. The lighter-weight
interpretability variant lives in `gk_code/main/light_pipeline/`, and
superseded/unused scripts have been moved to `gk_code/legacy/`. Each
stage caches its result as a `.joblib` (or report file) inside the experiment
folder, so re-running a stage is a no-op unless `--force_rerun` is passed.

```
01v6  raw chip data        -> preprocessed_curves_nonorm.joblib
  v   (curves + sigmoid fits)
02    + outlier detection   -> curve_for_training_nonorm.joblib
  v   (kinetic features + outlier labels)
03    + model training       -> classification_performances_nonorm.joblib
  v
04    (optional) cross-dataset CV -> cross_dataset_cv/<group>/...
05/06/07/attribution_vis_all  -> HTML/PNG reports
```

## `main/01_curve_preprocessing_v6.py`
Loads raw chip data (`v05`/`v06` via `titan_v6`, or `v04` via `titan_v4`), reconstructs each
well's amplification curve, computes derivatives/moving averages, and fits
5-parameter sigmoid curves.

- **Output**: `preprocessed_curves_nonorm.joblib` — `curves` dict with
  `ori_curves`, `ori_curves_avg` (moving-average smoothed, window
  `config.WINDOW_SIZE_ORI`), `ori_curve_dydx`, `ori_dydx_avg`, `cleaned_std`,
  `cleaned_lowest`, plus `sigmoid_curves` fits and well/pixel metadata.
- **Key args**: `--task_id`, `--exp_folder`, `--n_wells`, `--n_a_type`
  (`v04`/`v05`/`v06`), `--nc_subtract` (negative-control baseline subtraction;
  writes to a sibling `<exp_folder>_nc_subtract`), `--force_rerun`.
- Old caches missing `ori_curves_avg`/`window_size_ori` are patched in place.
  A corrupted cache file triggers a full recompute instead of crashing.

## `main/02_outlier_detection_pipeline.py`
Builds the unified training dataset: extracts kinetic features for
`ori_curves`, `ori_curves_avg`, and each sigmoid-fit variant, then runs the
outlier-filter pipelines (autoencoder, spatial kNN/grid consistency, MSC/AMF)
and attaches per-sample outlier labels as extra feature columns.

- **Output**: `curve_for_training_nonorm.joblib` — `dataset_name`/`dataset`
  (curve arrays: `ori_curves`, `ori_curves_avg`, `*_fitted_full`,
  `*_fitted_stretched`), `kinetic_features` (one DataFrame per dataset entry,
  kinetic params + outlier-filter columns), `Y_well`, `timestamps`,
  `metadata_df`.
- **Key args**: `--task_id`, `--exp_folder`, `--force_rerun`.
- Old caches missing `ori_curves_avg` are patched in place, reusing the
  outlier-filter labels already computed on `ori_curves`. This patch step
  tolerates a corrupted preprocessed-curves cache (skips with a warning).

## `main/03_main_training.py`
Trains/evaluates the classifier suite (kNN, CNN, GRU, Transformer, CNN+GRU /
CNN+Transformer dual-branch, plus late-fusion variants) on `ori_curves` and
`ori_curves_avg`, for each outlier filter in `config.OUTLIER_FILTERS`.

Two training modes (`--training_mode`, default `["native"]`):
- **Native**: train on this dataset's own curves, using its own
  outlier-filter labels.
- **Reference**: train on the original `ori_curves`, using *this* dataset's
  outlier-filter labels — compares "filters from the smoothed curve, applied
  to the raw curve". For `ori_curves` itself, Native == Reference; if a cached
  Reference result exists, Native training is skipped and reused.

- **Output**: `classification_performances_nonorm.joblib` (or
  `..._10fold_nonorm.joblib` if `--n_splits > 1`) — per-dataset
  `"Native"`/`"Reference"` result dicts, plus
  `model_performance/<dataset>_Native.png` / `_Reference.png` plots.
- **Key args**: `--task_id`, `--exp_folder`, `--n_splits`, `--training_mode`,
  `--force_rerun`.

## `main/04_cross_dataset_training.py`
Cross-dataset robustness check. Combines curves from multiple experiment
folders sharing an identical well→label mapping
(`config.CROSS_DATASET_GROUPS`), resamples them onto a common time grid
(`CurveResampler`), and evaluates:

- **`lofo`** (leave-one-folder-out): train on all-but-one folder, test on the
  held-out folder.
- **`well_cv`** (leave-one-well-out CV): folds hold out one well-index per
  target class across the combined dataset.

- **Output**: under `<exp_folder>/cross_dataset_cv/<group_name>/` —
  `classification_performances_cross_dataset_{lofo,wellcv}_{curve_type}.joblib`,
  `classification_performances_cross_dataset_resampler_{curve_type}.joblib`,
  plus per-fold plots.
- **Key args**: `--task_id` (index into `CROSS_DATASET_GROUPS`),
  `--exp_folder`, `--mode` (`lofo`/`well_cv`/`both`), `--curve_type` (e.g.
  `ori_curve`, `ori_curve_avg`), `--force_rerun`.

## `main/05_outlier_visualization_report.py`
Static HTML report: for each filter in `config.OUTLIER_FILTERS`, plots
inlier vs. outlier curves per well (one row per well, inliers | outliers).

- **Output**: `<viz_dir>/outlier_visualisation/<exp_name>_<curve_type>_outlier.html`
  (one per `--curve_type`); `<viz_dir>` mirrors the experiment folder under
  `POC_DDM_viz`.
- **Key args**: `--exp_folder`, `--curve_type` (nargs, default
  `["ori_curve", "ori_curve_avg"]`), `--force_rerun`.

## `main/06_model_prediction_report.py`
Static HTML report of trained-model predictions from `03` — per-architecture
panels (CNN/LSTM/GRU/RNN/Transformer/kNN/RF + late-fusion/dual-branch
variants) for a given training mode and outlier filter.

- **Output**: `<viz_dir>/model_performance_viz/<exp_name>__<curve_type>__<mode>__<filter_tag>__nsplits<N>.html`.
- **Key args**: `--exp_folder`, `--mode` (`Native`/`Reference`),
  `--outlier_filter`, `--n_splits` (must match the `03` run), `--curve_type`
  (nargs, default `["ori_curve", "ori_curve_avg"]`), `--force_rerun`.

## `main/07_resampling_check.py`
Sanity-check plots for `CurveResampler` (used by `04`) — `ori_curves` before
vs. after resampling onto a common time grid, overall and per well.

- **Output**: `resampling_check_overall.png` and `resampling_check_by_well.png`
  in the experiment folder's viz directory.
- **Key args**: `--task_id`, `--exp_folder`.

## `main/attribution_vis_all.py`
XAI/interpretability report for the trained deep models — latent-space
PCA/t-SNE, per-class kinetic-feature distributions, and gradient-based
saliency maps over the input curves for each architecture.

- **Output**: under `<viz_dir>/model_interpretation/` —
  `01_latent_space_pca_<exp_name>*.png`, `02_latent_space_tsne_<exp_name>*.png`,
  `04_saliency_<exp_name>*_<model>.png` (one per model),
  `07_latent_mapping_<exp_name>*_<model>.png`.
- **Key args**: `--exp_folder`, `--curve_type` (nargs, default
  `["ori_curve", "ori_curve_avg"]`), `--filter_key` (kinetic-feature column
  used to mask samples), `--normalize`, `--force_rerun`.
