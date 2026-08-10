# Ablation Studies: Model Comparison, Curve Preprocessing, Generalisability

## Motivation
- **Goal:** answer three separate research questions with isolated, reusable ablation scripts rather than folding everything into `03_main_training.py`'s CLI.
- Which architecture family performs best on raw curves at all (`ablation1`)?
- Does negative-control (nc) subtraction preprocessing actually help, and does it interact with the attn_recon spatial-reconstruction architecture (`ablation2`)?
- Do the MTL / RCFD / SupCon3 augmentations of the two best dual-branch architectures (`cnn_gru_dual`, `cnn_gru_dual_attn_recon`) actually generalise better than their plain counterparts (`ablation3`)?
- All three reuse `evaluate_outlier_filters` (`model_utils.py:754`) as the training/eval engine — the exact same dispatch logic `03_main_training.py` runs on already, so ablation results are directly comparable to the main pipeline's numbers, model-for-model.

## Ablations

### 1. Model Comparison (`ablations/ablation1_model_comparison.py`)
`knn` vs `cnn` vs `gru` vs `transformer` vs `cnn_gru_dual` vs `cnn_gru_dual_attn_recon`.

| key | family | needs spatial metadata? |
|---|---|---|
| `knn` | sklearn `KNeighborsClassifier(n_neighbors=10)`, fit on raw flattened curves | no |
| `cnn` / `gru` / `transformer` | single-branch DL classifiers | no |
| `cnn_gru_dual` | dual-branch CNN+GRU classifier | no |
| `cnn_gru_dual_attn_recon` | dual-branch, but input is an attention-reconstructed curve built from spatial neighbours | **yes** |

### 2. Curve Preprocessing (`ablations/ablation2_curve_preprocessing.py`)
`cnn_gru_dual` vs `cnn_gru_dual_attn_recon`, each trained twice — once on plain curves, once on nc-subtracted curves (`01_curve_preprocessing_v6.py --nc_subtract`: per-VREF-slice, subtracts the mean `NC-*` well curve from every sample curve, collapsing per-target NC labels into one generic `NC` class). Phase-2-only (see below) — both the attn_recon spatial structure and the nc_subtract VREF-slice grouping need POC_DDM_final-style chip metadata that LAB data doesn't have.

### 3. Generalisability (`ablations/ablation3_generalisability.py`)
`cnn_gru_dual` and `cnn_gru_dual_attn_recon`, each against its MTL / RCFD / SupCon3 variant:

| base | + MTL | + RCFD | + SupCon3 |
|---|---|---|---|
| `cnn_gru_dual` | `cnn_gru_dual_mtl` | `gru_rcfd_cgd` | `cnn_gru_dual_supcon3` |
| `cnn_gru_dual_attn_recon` | `cnn_gru_dual_attn_recon_mtl` | `gru_rcfd_attn_recon` *(new — see below)* | `cnn_gru_dual_attn_recon_supcon3` |

Two corrections made against the original ask, both confirmed in discussion:
- `cnn_trans_crossattn_mtl` was a typo for `cnn_gru_dual_attn_recon_mtl` (the real MTL counterpart of `cnn_gru_dual_attn_recon`; `cnn_trans_crossattn_mtl` is a real key but a different gated cross-attention fusion model).
- `gru_rcfd_attn_recon` did not exist (`_build_rcfd_backbone` only supported `dual_type` `cgd`/`ctd`) — built as new core library support, see below, rather than leaving that cell of the table empty.

## Phased Dataset Rollout
- **Phase 1 (now):** `LAB_DDM_paper` (`config.LAB_EXP_FOLDER`), numbered subfolders `01_ACA_qdPCR`, `02_AMCA_qdLAMP`, `03_AMCA_qdPCR`, `09_Area_Conc`, `10_Range_Conc`. LAB data has `metadata: {}` (hardcoded in `01b_lab_curve_preprocessing.py:351` — no pixel grid, since it's per-tube qPCR/qLAMP reference data, not on-chip). `evaluate_outlier_filters` already soft-skips any `*_attn_recon*` model when spatial metadata is missing (prints a warning, trains everything else) — so `ablation1`/`ablation3` run on Phase 1 with those columns simply absent from the results, no special-casing needed in the scripts. `ablation2` needs both attn_recon *and* nc_subtract, so it defaults to Phase-2 data and exits cleanly (per-subfolder) if the nc_subtract sibling doesn't exist yet.
- **Phase 2 (later):** `POC_DDM_final` / `POC_DDM_final_nc_subtract` (`config.FINAL_EXP_FOLDER` + its `_nc_subtract` sibling, same subfolder-name convention). These have real `pixel_row_idx`/`pixel_col_idx` metadata, unlocking attn_recon everywhere and nc_subtract.

## New Model: RCFD + attn_recon Backbone
`model_utils_rcfd.py` gained `_build_rcfd_attn_recon_backbone(stack_input, T, enc_type, ...)`: the early/regression encoder (`_build_early_encoder`, unmodified) runs on just the center curve (`stack_input[:, 0, :]`) — it has no use for neighbours, it only predicts this sample's own concentration. The classification backbone reuses `_build_cnn_gru_dual_attn_recon_embedding_mtl` (imported from `model_utils_mtl.py`) on the full neighbour stack, then FiLM-conditions it exactly like every other RCFD variant. Three new factories (`cnn`/`gru`/`trans` × attn_recon) registered into `RCFD_MODEL_KEYS` (9 base combos total now, 27 across all SupCon tiers). Wired into `evaluate_outlier_filters`'s `_SPATIAL_RECON_MODELS` gate (so `(n, k+1, T)` neighbour stacks get built for these keys) and its RCFD dispatch block (`model_utils.py:2088`, now branches on `RCFD_ATTN_RECON_MODEL_KEYS` to call the factory with `(k_plus_1, T, n_classes)` instead of `(T, n_classes)`). Registered in `config.MODEL_KEY_MAP`/`MODEL_PRINT_MAP` alongside the existing `gru_rcfd_cgd`-style entries.

## Implementation
- `ablations/ablation1_model_comparison.py`, `ablation2_curve_preprocessing.py`, `ablation3_generalisability.py` — each a thin, self-contained driver: reuses the same `curve_for_training.joblib` loader / label-mapping / spatial-metadata pattern as `03_main_training.py`, a hardcoded `models = [...]` list (not a general `--models` flag — these are fixed research questions), one call into `evaluate_outlier_filters(outlier_filters=[None], ...)`, one `plot_ml_results` call. `--exp_folder`/`--task_id` args mirror `get_exp_paths`/`check_task_id` from `pipeline_utils.py` so they drop into the existing SLURM array-job pattern (`slurm_jobs/`) unchanged for Phase 2.
- No changes to `03_main_training.py` or its `classification_performances*.joblib` — the ablation scripts are entirely additive.

## Result Format
Each script writes its own joblib under `<exp_path>/ablations/`:
- `ablation1_model_comparison_performances.joblib`
- `ablation2_curve_preprocessing_plain_performances.joblib` + `ablation2_curve_preprocessing_nc_subtract_performances.joblib`
- `ablation3_generalisability_performances.joblib`

Every file holds the exact same `results_dict[filter_name] = {y_trues_, y_preds_{tag}_, y_probs_{tag}_, y_classes_{tag}_, train_history_{tag}_, [y_reg_preds_{tag}_, y_reg_trues_{tag}_]}` schema `evaluate_outlier_filters` already produces for `classification_performances.joblib` (just with `outlier_filters=[None]`, one entry). `model_utils.plot_ml_results` reads these files unmodified — nothing pipeline-side needs to know these came from an ablation script instead of `03_main_training.py`.
