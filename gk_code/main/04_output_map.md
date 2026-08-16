# `04_cross_dataset_training.py` output map

What gets written where, for every `--mode`/`--curve_alignment`/`--outlier_filter`/
`--supcon`/`--dann`/`--coral`/`--models` combination, and for every held-out fold.
Reflects the per-`(filter, model)` result-file split in
[utils/cross_dataset_result_io.py](utils/cross_dataset_result_io.py) — see that
module's docstring for why it's split this way (concurrent `--supcon`/`--dann`/
`--coral` runs used to silently clobber each other's results under one shared file).

## 1. The root: which args pick `out_dir`

Everything lives under one `out_dir`, picked by three args before any model ever trains:

```
out_dir = {exp_folder}/cross_dataset_cv/{group_name}
if --curve_alignment pc_ttp:
    out_dir = out_dir / "curve_alignment_pc_ttp" / "anchor_{pc_ttp_anchor}"
```

| Arg | Picks |
|---|---|
| `--exp_folder` | the dataset root |
| `--task_id` | `group_name = list(config.CROSS_DATASET_GROUPS.keys())[task_id]` |
| `--curve_alignment` | `acquisition_start` (default) → no extra nesting. `pc_ttp` → nests under `curve_alignment_pc_ttp/anchor_{X}/` |
| `--pc_ttp_anchor` | only matters when `--curve_alignment pc_ttp`; `min` or `percentile` → the `anchor_{X}` folder name |

So **`--curve_alignment`/`--pc_ttp_anchor` is the only combination of args that changes
the directory root** — every other arg (mode, filter, supcon, dann/coral, models,
curve_type) only changes *filenames* under this same `out_dir`.

## 2. Directory tree under `out_dir`

```
out_dir/
├── _cache_pc_ttp/                                   # only if --curve_alignment pc_ttp; joblib.Memory cache for PC-TTP fits
├── model_performance_{curve_type}/                  # accuracy/composition plots, one dir per curve_type
│   └── {fold_label}_accuracies.png, {fold_label}_composition.png
├── model_interpretation/
│   ├── {fold_label}/                                # one dir per fold actually processed (see §4)
│   │   ├── {model_key}_{filter_key}_{curve_type}_model.keras   # ONLY for the fold processed first (see §5)
│   │   └── xai_data_{curve_type}.joblib             # test-set snapshot for this fold (consumed by 06b)
│   └── full_data_{mode_str}/                        # only with --train_full (see §6)
│       ├── {model_key}_{filter_key}_{curve_type}_model.keras   # every model, since full_data has one "fold"
│       └── (no xai_data snapshot — full_data has no held-out test set)
├── {filter_token}/                                  # one per outlier_filter actually run (see §3)
│   ├── cross_dataset_classification_performances_{mode}_{curve_type}_{model_key}.joblib   # current layout, one file per (filter, model); --train_center_frac None
│   └── {frac_token}/                                # only if --train_center_frac was set (see §3b); every fold_label except "full_data"
│       └── cross_dataset_classification_performances_{mode}_{curve_type}_{model_key}.joblib
├── cross_dataset_classification_performances_{mode}_{curve_type}.joblib   # LEGACY single-file layout — only present for groups trained before the split (see §7)
├── cross_dataset_resampler_classification_performances_{curve_type}.joblib   # CurveResampler, one per curve_type (used by 08 to align new chips)
├── cross_dataset_pc_ttp_recipe_{curve_type}.joblib  # only if --curve_alignment pc_ttp; {anchor, common_duration}
├── lofo_ae_filter/{fold_label}_{curve_type}/        # only if --outlier_filter lofo_ae; fold-local LSTM-AE filter model+meta
└── pc_reference_embedding_{model}_{filter}_{curve_type}.joblib   # written by 08's --pc_recenter, not by 04
```

## 3. `{filter_token}`: which arg, what values

`--outlier_filter` (`nargs='+'`, e.g. `--outlier_filter none lofo_ae`) — each value run
gets its own subfolder. `filter_token(filter_name)` (in `cross_dataset_result_io.py`):
`None`/`"none"` → `none/`; any other filter string → itself, sanitized to
`[A-Za-z0-9_.-]` (a no-op for real filter names, which are already plain
identifiers/`features_df` column names).

## 3b. `{frac_token}`: `--train_center_frac`

`--train_center_frac` (float, default `None`) spatially center-crops the *training*
pool per well before fitting (test/eval pool and `--train_full` are never affected —
see `evaluate_outlier_filters`'s docstring). Its value gets its own subfolder *nested
inside* `{filter_token}/`, via `_frac_token()`/`model_result_path()` in
`cross_dataset_result_io.py`: `None` (default) → no extra subfolder, same path as
before this option existed; a float `f` → `center{f:g}/` (e.g. `0.5` → `center0.5/`,
`0.25` → `center0.25/`).

This exists so re-running the same (group, alignment, filter, model) combination with
a *different* `--train_center_frac` — or with none at all — doesn't silently overwrite
a previous run's results; each fraction gets its own file. The one exception is fold
label `"full_data"` (§6): since `--train_full` never samples, its results always save
to/load from the unscoped path regardless of what `--train_center_frac` this
invocation used — `save_partitioned`/`load_partitioned` handle this automatically, no
caller-side special-casing needed.

`06b_cross_dataset_prediction_report.py --train_center_frac` must match whatever the
corresponding `04` run used, to find the right subfolder (defaults to `None`, the
unsampled layout).

## 4. `{fold_label}`: which arg, what values

Set by `--mode`:

| `--mode` | fold labels | source |
|---|---|---|
| `lofo` (default) | `lofo_{folder_name}`, one per folder in the group | `build_lofo_splits` |
| `kfold` | `fold_0` … `fold_{n_splits-1}` | `build_nfold_splits` / `build_well_stratified_nfold_splits` |
| `random_split` | `random_split` (single fold) | `build_random_split` / `build_well_stratified_random_split` |

`--lofo_limit N` (lofo mode only) caps this to the first `N` fold labels instead of
running every folder — doesn't change the naming, only how many folds get produced.

`_mode_str` (used in the joblib filename) is `mode` verbatim, except `kfold` becomes
`kfold{n_splits}` (e.g. `--mode kfold --n_splits 5` → `kfold5`).

## 5. `{model_key}`: which args, what values

`{model_key}` is never one arg directly — it's whichever concrete model-key strings
`--supcon`/`--dann`/`--coral`/`--mtl`/`--condreg`/`--supcon_staged`/`--lbl_conc` select
(see the big `if/elif` chain in `04_cross_dataset_training.py`'s `__main__`), filtered
down further by `--models` if given (substring match against the base name, so
`--models cnn_gru_dual` also matches `cnn_gru_dual_supcon3`, `cnn_gru_dual_dann`, etc.).
Examples relevant to the `fnl_*` jobs:

| flags | `--supcon 0` models | `--supcon 3` models |
|---|---|---|
| (none) | `knn`, `cnn_gru_dual`, `cnn_gru_dual_cosine_recon`, `cnn_gru_dual_attn_recon` | `cnn_gru_dual_supcon3`, `cnn_gru_dual_cosine_recon_supcon3`, `cnn_gru_dual_attn_recon_supcon3` |
| `--dann` | `cnn_gru_dual_dann`, `cnn_gru_dual_attn_recon_dann` | `cnn_gru_dual_supcon3_dann`, `cnn_gru_dual_attn_recon_supcon3_dann` |
| `--coral` | `cnn_gru_dual_coral`, `cnn_gru_dual_attn_recon_coral` | `cnn_gru_dual_supcon3_coral`, `cnn_gru_dual_attn_recon_supcon3_coral` |

Full mapping: `config.MODEL_KEY_MAP` (model key → result-dict key prefixes) crossed with
the model-selection `if/elif` chain in `04_cross_dataset_training.py`.

`.keras` model files (§2) are named `{model_key}_{filter_key}_{curve_type}_model.keras`
— note `filter_key` there is the *display* form (`"None"` for baseline, not `"none"`),
different from the `{filter_token}` used for the results-joblib subfolder.

**Only the first fold processed gets a `.keras` file saved** (`fold_idx == 0` in
`evaluate_outlier_filters`, for `07_attribution_vis_all.py`'s benefit) — every other
fold's model is trained, evaluated, and discarded; only its predictions/metrics/history
survive, in the results joblib. `full_data` (§6) is the exception: every model there gets
a `.keras` file, since `--train_full` has exactly one "fold" (all data, no holdout).

## 6. `--train_full`

After the fold loop, trains one additional model per (filter, model) on **all** data
(no holdout), saved under `model_interpretation/full_data_{mode_str}/` and recorded in
the results joblib under fold label `"full_data"`. Its accuracy is train-set (inflated)
— use the real folds' metrics for evaluation; `full_data`'s only purpose is producing a
deployable model + a `class_names` list (used by `06_model_prediction_report`-style
downstream tools) and the CORAL/coefficient reference embeddings `08` needs.

## 7. Per-`(filter, model)` result file: exact contents

`{filter_token}/cross_dataset_classification_performances_{mode}_{curve_type}_{model_key}.joblib`
is a dict `{fold_label: {...}}`. Each fold's entry is self-contained:

- Shared/duplicated across every model's file for that (fold, filter): `y_trues_`,
  `_split_signature`, `mask_count`, `y_true_count`, and (fold-level, not filter-level)
  `top_10_features`, `class_names`.
- That model's own keys (from `config.MODEL_KEY_MAP[model_key]` plus optional extras):
  `y_preds_..._`, `y_probs_..._`, `classes_..._`, and if the model trained with Keras
  `.fit()`: `train_history_{model_key}_` (list of one dict per fold-run, each a Keras
  `History.history` dict — see `learning_curve_analysis_cross_dataset.ipynb`). MTL/RCFD
  models additionally get `y_reg_preds_{model_key}_`/`y_reg_trues_{model_key}_`.

**Legacy fallback**: groups trained before this split has one flat file instead —
`cross_dataset_classification_performances_{mode}_{curve_type}.joblib`, holding every
filter/model together. `utils/cross_dataset_result_io.py`'s `load_partitioned()` checks
the per-`(filter, model)` layout first and only falls back to this if nothing is found
there — new training runs never write this legacy path.

## 8. Worked example

```
python 04_cross_dataset_training.py --exp_folder $EXP --task_id 3 \
    --coral --supcon 3 --curve_type ori_curve_norm --models cnn_gru_dual cnn_gru_dual_attn_recon \
    --outlier_filter none --curve_alignment pc_ttp --pc_ttp_anchor min --train_full
```

writes (group = `list(config.CROSS_DATASET_GROUPS.keys())[3]`, 4-folder group):

```
{EXP}/cross_dataset_cv/{group}/curve_alignment_pc_ttp/anchor_min/
├── none/
│   ├── cross_dataset_classification_performances_lofo_ori_curve_norm_cnn_gru_dual_supcon3_coral.joblib
│   │     -> {lofo_<chip1>, lofo_<chip2>, lofo_<chip3>, lofo_<chip4>, full_data}
│   └── cross_dataset_classification_performances_lofo_ori_curve_norm_cnn_gru_dual_attn_recon_supcon3_coral.joblib
│         -> same 5 fold labels
├── model_interpretation/
│   ├── lofo_<chip1>/   (whichever chip is processed first)
│   │   ├── cnn_gru_dual_supcon3_coral_None_ori_curve_norm_model.keras
│   │   ├── cnn_gru_dual_attn_recon_supcon3_coral_None_ori_curve_norm_model.keras
│   │   └── xai_data_ori_curve_norm.joblib
│   ├── lofo_<chip2>/, lofo_<chip3>/, lofo_<chip4>/   (xai_data_*.joblib only, no .keras)
│   └── full_data_lofo/
│       ├── cnn_gru_dual_supcon3_coral_None_ori_curve_norm_model.keras
│       └── cnn_gru_dual_attn_recon_supcon3_coral_None_ori_curve_norm_model.keras
├── model_performance_ori_curve_norm/
├── cross_dataset_resampler_classification_performances_ori_curve_norm.joblib
└── cross_dataset_pc_ttp_recipe_ori_curve_norm.joblib
```
