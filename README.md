# Point-of-Care Data Driven Multiplexing

<span style="font-size: 1.2em;">
Identifying which of several candidate pathogens is present from a single amplification curve on a point-of-care platform, the Lacewing device. This repository contains the signal preprocessing, spatio-temporal modelling, model generalisability, and interpretability code for a data-driven multiplexing pipeline built for the Lacewing ISFET platform.

</span>

<br />
<br />

---

# Folder Structure (see [`main_code/`](main_code/))


```
main_code/
├── main_chip.py    # dispatcher: preprocess | cross-dataset | intra-chip-training | saliency
│                    #             | embedding-analysis | confidence-shift
├── main_lab.py      # dispatcher: preprocess | train | saliency
├── config.py         # shared config (paths, model registry, chip/LAB scope, labels)
├── chip/              # CHIP (POC_DDM_final) pipeline
├── lab/                # LAB (LAB_DDM_paper) pipeline
├── utils/               # shared engine + titan_v6 copy (no outside imports)
├── notebooks/{chip,lab,multiplex}/  # RQ analysis notebooks
└── slurm_jobs/            # one script per stage, plus chained full_pipeline scripts
```

Run everything from inside `main_code/` (or with `main_code` on `PYTHONPATH`) using
the same venv as the rest of the project (`venv_poc_ddm`, TF 2.16.2).

**Scope**: CHIP = exactly the 6 chips in `config.CROSS_DATASET_GROUPS['final_6_new']`.
LAB = exactly the 4 folders in `config.LAB_DATASETS_IN_SCOPE`
(`01_ACA_qdPCR`, `02_AMCA_qdLAMP`, `03_AMCA_qdPCR`, `12_ACA_qdPCR_multiplex_balanced`),
plus `LAB_Multiplex/02_ACA_qdPCR_balanced` (read directly, not part of `LAB_DDM_paper`).
Every stage's folder discovery is restricted to these lists
(`pipeline_utils.get_scoped_exp_paths`), not "whatever's on disk."

---

## CHIP: `main_chip.py`

### `preprocess`

```
python main_chip.py preprocess --exp_folder <dir> --task_id 0 [--force_rerun] [--run_outlier_detection]
```

`--task_id` indexes into `final_6_new` (0-5). `--run_outlier_detection` also runs the
LSTM-AE (global) outlier pass in the same process; TensorFlow is only imported when
this flag is set, so a plain preprocessing run stays lightweight. Everything else
(n_wells=10, titan v6 pixel layout, drop_pc=True, sg_p4 denoising, normalization) is
fixed, not a CLI param.

**Writes:**
- `{exp_folder}/{chip}/curve_for_training.joblib` — curves (`ori_curves`,
  `ori_curves_norm`, `ori_curves_sg_p4`, `ori_curves_sg_p4_norm`), kinetic features,
  labels, metadata, and a `pc_wells` snapshot of the dropped PC curves.
- `{exp_folder}/{chip}/pretrained_encoders/` — LSTM-AE encoder, only if
  `--run_outlier_detection` was passed.

### `cross-dataset`

```
python main_chip.py cross-dataset --exp_folder <dir> --task_id 0 [--force_rerun]
    [--curve_type ...] [--models ...] [--mode lofo|random_split|kfold]
    [--n_splits 5] [--outlier_filter ...] [--batch_size 2048]
    [--train_center_frac 0.5] [--train_full] [--test_size 0.1] [--held_out_chip <name>]
```

`--task_id` indexes into `config.CROSS_DATASET_GROUPS` (only `final_6_new` exists).
`--mode kfold` uses **plain** curve alignment (no PC-TTP shift) and writes to
`cross_dataset_cv/{group}/`; `lofo`/`random_split` use **PC-TTP-aligned** curves and
write to `cross_dataset_cv/{group}/curve_alignment_pc_ttp/anchor_min/`.

`--models` choices: `cnn_gru_dual`, `cnn_gru_dual_attn_recon`, their `_supcon1` /
`_supcon3` / `_dann` / `_aug` (temporal augmentation) / `_mtl` (multi-task,
Kendall-uncertainty-weighted joint classification + concentration regression)
variants, and `knn`. There is no `_pc_recentering` model choice — pc_recenter is
computed on demand at analysis time (see below), not trained.

**Writes**, under `{exp_folder}/cross_dataset_cv/{group}/[curve_alignment_pc_ttp/anchor_min/]`:
- `{filter_token}/[center{frac}/]cross_dataset_classification_performances_{mode}_{curve_type}_{model}.joblib`
  — per-model result partitions (load with `utils/cross_dataset_result_io.load_partitioned`).
- `model_interpretation/lofo_{chip}/` or `model_interpretation/full_data_{mode}/` —
  saved `.keras` models per fold (`{model}_{filter}_{curve_type}[_center{frac}]_model.keras`).
- `model_performance_{curve_type}/` — result plots.
- `cross_dataset_resampler_classification_performances_{curve_type}.joblib`,
  `cross_dataset_pc_ttp_recipe_{curve_type}.joblib` — curve alignment artifacts
  (only under the PC-TTP-aligned path).

### `intra-chip-training`

```
python main_chip.py intra-chip-training --exp_folder <dir> --task_id 0 [--force_rerun]
    [--curve_type ori_curve_sg_p4_norm] [--n_splits 5] [--batch_size 512]
```

Per-chip, per-fold outlier-filter × model ablation for `cnn_gru_dual` /
`cnn_gru_dual_attn_recon` (`ori_curve_sg_p4_norm`, `--n_splits`-fold CV). Replaces the
old per-chip `train` stage — nothing downstream needs plain, un-ablated per-chip
kNN/CNN/BiGRU/Transformer results.

**Writes:** `{exp_folder}/{chip}/ablations/ablation6_chip_outlier_model_ablation_performances.joblib`,
`{exp_folder}/{chip}/ablations/model_interpretation/`.

### `saliency`

```
python main_chip.py saliency --exp_folder <dir> [--curve_type ori_curve_norm ori_curve_sg_p4_norm]
    [--n_dims 25] [--top_n 5] [--batch_n 512] [--seed 42]
```

Saliency + latent-feature-mapping plots for `cnn_gru_dual` / `cnn_gru_dual_attn_recon`,
against models saved by `intra-chip-training` (`ablations/model_interpretation/`).

**Writes:** `{exp_folder}/{chip}/ablations/xai_saliency/`.

### `embedding-analysis`

```
python main_chip.py embedding-analysis --exp_folder <dir> [--batch_n 800] [--seed 42]
```

Per-model t-SNE embedding grid (base + `_pc_recenter`) for `final_6_new`'s LOFO
cross-dataset results — needs `cross-dataset --mode lofo` to have already populated
`model_interpretation/lofo_{chip}/`.

**Writes:** `notebooks/chip/RQ3_05_embedding_analysis/{model_key}_tsne_grid.png`.

### `confidence-shift`

```
python main_chip.py confidence-shift --exp_folder <dir>
```

Softmax confidence: in-distribution CV vs. LOCO, base vs. pc_recenter. Needs both
`cross-dataset --mode kfold` and `--mode lofo` results populated.

**Writes:** `notebooks/chip/RQ3_06_confidence_shift/confidence_shift_grid.png`.

### On-demand pc_recenter (`utils/pc_recentering.py`)

`pc_recenter` is **not** a trained model — it's a pure inference-time embedding shift:
`(training-pool PC-well mean embedding − held-out chip's own PC-well mean embedding)`
applied before the base model's unmodified classification head. `embedding-analysis`,
`confidence-shift`, and the `RQ3_*`/`RQALL` notebooks all call
`predict_new_chip(..., pc_recenter=True)` directly against an already-trained base
model's `.keras` file — no separate training run or CLI flag needed.

---

## LAB: `main_lab.py`

### `preprocess`

```
python main_lab.py preprocess --exp_folder <dir> --task_id 0 [--force_rerun]
```

`--task_id` indexes into `config.LAB_DATASETS_IN_SCOPE` (0-3). Default `--exp_folder`
is `config.LAB_EXP_FOLDER` (`LAB_DDM_paper`).

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
- `{exp_folder}/{subfolder}/ablations/ablation1_model_comparison_performances.joblib`.
- `{exp_folder}/{subfolder}/ablations/model_interpretation/` — saved `.keras` models.

### `saliency`

```
python main_lab.py saliency [--n_dims 25] [--top_n 5] [--batch_n 512] [--seed 42]
```

`cnn_gru_dual` saliency + latent-feature-mapping plots over the 3 non-multiplex LAB
folders (`01_ACA_qdPCR`, `02_AMCA_qdLAMP`, `03_AMCA_qdPCR`), against models saved by
`train`.

**Writes:** `{exp_folder}/{subfolder}/ablations/xai_saliency/`.

---

## Notebooks

Run with `main_code/` as the working directory (each notebook's first code cell sets
up `sys.path`/`chdir`).

- `lab/RQ1_01_lab_results.ipynb` — LAB model-comparison results + statistical
  significance (Friedman/Nemenyi, McNemar) over `lab/train`'s output.
- `chip/RQ2_01_denoising.ipynb` — raw vs. sg_p4-denoised curve comparison.
- `chip/RQ2_02_e2e_preprocessing.ipynb` — end-to-end walkthrough: raw instrument file
  → titan_v6 pixel/well linearisation → `chip/preprocessing.py` → `cross-dataset`'s
  curve-alignment pipeline, on one example chip.
- `chip/RQ2_03_spatial_attn_pooling.ipynb` — per-chip raw vs. `cnn_gru_dual_attn_recon`
  reconstruction/attention-weighted curves, against `intra-chip-training`'s saved models.
- `chip/RQ2_05_results.ipynb` — `intra-chip-training`'s per-fold outlier-filter × model results
  (`cnn_gru_dual` vs. `cnn_gru_dual_attn_recon`), accuracy/F1/sens/spec tables.
- `chip/RQ3_01_cv_results.ipynb` — 5-fold cross-chip CV (plain-aligned, `kfold` mode)
  accuracy, `knn`/`cnn_gru_dual`/`cnn_gru_dual_attn_recon`.
- `chip/RQ3_02_loco_results.ipynb` — leave-one-chip-out (LOFO) accuracy, base models vs.
  + pc_recenter, all 6 attn_recon-family variants (dann/supcon3/aug/mtl included).
- `chip/RQ3_03_loco_curve_analysis.ipynb` — PC/NC-ALL-augmented mean-curve plots per
  target/chip, PC-TTP-aligned pools.
- `chip/RQ3_04_loco_result_analysis.ipynb` — LOFO confusion matrices, curve-distance
  matrices, per-fold model loading for deeper LOCO diagnostics.
- `chip/RQALL_per_well_accuracy.ipynb` — per-well (majority-vote) accuracy tables
  across RQ2 intra-chip, RQ3 CV, and RQ3 LOCO sections, attn_recon-only.
- `multiplex/Z_FW_01_sw_multiplex.ipynb` — multiplex source-separation vs. multi-label
  classification comparison, entirely read-only over already-computed results (no
  training in this notebook).

**Needs a SLURM run first**: any section reading `_aug`/`_mtl` model results, or a
`train_center_frac`-partitioned result path, will print `[WARN] No results` /
empty tables until `slurm_jobs/chip_cross_dataset.sh` (or a manual `cross-dataset`
run with `--models ... cnn_gru_dual_attn_recon_aug cnn_gru_dual_attn_recon_mtl
--train_center_frac 0.5`) has actually been run — this repo does not run a full-scale
training sweep for you.

---

## SLURM jobs (`slurm_jobs/`)

Submit from inside `slurm_jobs/` (logs land in `slurm_jobs/logs/<job-name>/`), matching
the convention in `gk_code/main/slurm_jobs/`. One script per stage, plus a chained
`*_full_pipeline.sh` per side:

- `chip_preprocess.sh` / `chip_cross_dataset.sh` / `chip_intra_chip_training.sh` /
  `chip_saliency.sh` / `chip_embedding_analysis.sh` / `chip_confidence_shift.sh`
- `chip_full_pipeline.sh` — preprocess → intra-chip-training, array over the 6 `final_6_new`
  chips. Saliency/embedding-analysis/confidence-shift need every chip's models at
  once, so they aren't chained inside the array — run as dependent jobs instead:
  ```
  JOBID=$(sbatch --parsable slurm_jobs/chip_full_pipeline.sh)
  sbatch --dependency=afterok:$JOBID slurm_jobs/chip_saliency.sh
  ```
- `lab_preprocess.sh` / `lab_train.sh` / `lab_saliency.sh`
- `lab_full_pipeline.sh` — preprocess → train, array over the 4
  `LAB_DATASETS_IN_SCOPE` folders.

`--array` bounds are hardcoded to the current scope (6 chips, 4 LAB folders) —
re-check `config.CROSS_DATASET_GROUPS['final_6_new']` / `config.LAB_DATASETS_IN_SCOPE`
and adjust if that scope ever changes. `chip_cross_dataset.sh` takes a single
`TASK_ID` (index into `config.CROSS_DATASET_GROUPS`, editable at the top of the
script) rather than an array, since one cross-dataset run already covers every chip
in that group.

## Config (`config.py`)

Single shared config for both CHIP and LAB, ordered top-to-bottom from
user-editable to internal/utility. Notable entries: `DATASET_ROOT_FOLDER` (single
root, both `POC_DDM_final`/`LAB_DDM_paper` derive from it), `CROSS_DATASET_GROUPS`
(only `final_6_new`), `LAB_DATASETS_IN_SCOPE` (4 folders), `MODEL_KEY_MAP` (14
models, including `_aug`/`_mtl` — no `_pc_recentering` entries, that's on-demand
now), `OUTLIER_FILTERS`.

## Adding a new dataset

### CHIP

1. **Put the raw data** in a new subfolder under `config.DEFAULT_EXP_FOLDER`
   (`POC_DDM_final/`), same raw instrument-file layout (vref sweep + readout time
   `.bin` files, titan_v6 format) as an existing chip folder — copy an existing
   folder's layout as reference.
2. **Add to `config.py`** (all keyed by the new folder's exact name):
   - `CROSS_DATASET_GROUPS['final_6_new']` — append the folder name. This is the
     only group any notebook_rq notebook reads; a chip left out of it (or put in a
     new group) won't show up in cross-dataset training, embedding-analysis,
     confidence-shift, or any `RQ2_0x`/`RQ3_0x`/`RQALL` notebook.
   - `LOFO_EXCLUDE_WELL_MAPPING['final_6_new'][<folder>]` — well indices to drop
     from LOFO pools for this chip (PC/NC-ALL wells at minimum, e.g. `[8, 9]`; add
     any other non-target well like `06_02`'s `[6, 8, 9]`).
   - `LABEL_MAPPINGS[<folder>]` — `{well_idx: target_label}` for all `N_WELLS` (10)
     wells, PC/NC-ALL included (`'PC'` / `'NC-ALL'`).
   - `CONC_MAPPINGS[<folder>]` — `{well_idx: concentration}`, `0` for PC/NC-ALL.
     Feeds `_mtl`'s regression target and XAI concentration coloring.
   - `EXCLUDE_WELL_MAPPING[<folder>]` — well indices to drop for non-LOFO stages
     (`intra-chip-training`, `saliency`, plain `cross-dataset --mode kfold`) — usually the
     same PC/NC-ALL wells as the LOFO mapping.
3. **Update `slurm_jobs/`** — bump `chip_preprocess.sh` / `chip_intra_chip_training.sh` /
   `chip_full_pipeline.sh`'s `--array` upper bound to match the new chip count
   (`len(config.CROSS_DATASET_GROUPS['final_6_new']) - 1`).
4. **Run** `chip/preprocessing.py --task_id <new chip's index>` first (writes
   `curve_for_training.joblib`), then `intra-chip-training`/`cross-dataset` as normal.

### LAB

1. **Put the raw data** in a new subfolder under `config.LAB_EXP_FOLDER`
   (`LAB_DDM_paper/`), containing one CSV with a concentration column and a
   target/label column (see an existing folder's CSV for the row/column shape).
2. **Add to `config.py`**: `LAB_DATASETS_IN_SCOPE` — append the folder name.
3. **Add to `lab/preprocessing.py`** (module-level dicts, not in `config.py` —
   they're LAB-preprocessing-specific, keyed by the new folder's exact name):
   `FILE_MAPPING[<folder>]` (the CSV filename inside that folder),
   `FILE_CONC[<folder>]` (its concentration column name),
   `FILE_TARGET[<folder>]` (its target/label column name).
4. **Update `slurm_jobs/`** — bump `lab_preprocess.sh` / `lab_train.sh` /
   `lab_full_pipeline.sh`'s `--array` upper bound to
   `len(config.LAB_DATASETS_IN_SCOPE) - 1`.
5. **Run** `lab/preprocessing.py --task_id <new folder's index>` first, then
   `lab/training.py` / `lab/saliency.py` as normal.

<br />
<br />
<br />

---
# Folder Mapping


| Folder | Status | What it is |
|---|---|---|
| **`main_code/`** | **Finalised / production** | The cleaned, consolidated pipeline. Two dispatchers (`main_chip.py`, `main_lab.py`), a single `config.py`, and a fixed dataset scope. This is the code to read, run, and build on. |
| **`gk_code/main/`** | **Experimentation** | The exploratory tree the results were developed in — numbered stage scripts, ~90 analysis notebooks, ablations, and SLURM sweeps. Kept for provenance and for the thesis analyses; not the reference implementation. |
| `titan_v4/`, `titan_v6/` | Device layer | Raw chip acquisition/decoding for the v04 and v05/v06 firmware. `main_code/utils/titan_v6/` holds a synchronised copy. |
