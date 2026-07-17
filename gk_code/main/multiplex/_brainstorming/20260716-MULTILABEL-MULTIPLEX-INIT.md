# Multi-Label Multiplex Classification — Implementation Reference
**Originally drafted:** 2026-07-16  
**Last updated:** 2026-07-17  
**Status:** Living document — updated to reflect actual implementation. Where the original plan and the code differ, the code is authoritative.

---

## 1. Problem & Motivation

The existing pipeline treats every label combination (`VIM_NDM`, `NDM_KPC`, …) as an independent flat class using `LabelEncoder`. This ignores the multi-label structure: a droplet containing `VIM` + `NDM` should be learned jointly, not as an opaque class distinct from `VIM` or `NDM` alone.

The multiplex dataset (`dPCR_Dataset_Multiplex.csv`, 26 571 rows × 45 cycles + metadata) has:

| Label        | Count |
|---|---|
| VIM          | 7 223 |
| KPC          | 6 792 |
| NDM          | 6 232 |
| NDM_KPC      | 1 638 |
| VIM_KPC      | 1 582 |
| VIM_NDM_KPC  | 1 569 |
| VIM_NDM      | 1 535 |

Targets (after splitting by `_`): `{KPC, NDM, VIM}` → 3-bit binary vector per sample.

> **Data fix (pre-implementation):** The raw CSV had 1 row with label `VIM_NDM¸` (encoding artifact). This row must be corrected directly in the source CSV at `LAB_Multiplex/01_ACA_qdPCR/dPCR_Dataset_Multiplex.csv` before running 01b. The preprocessing script assumes clean labels and does **not** apply runtime artifact stripping.

---

## 2. Design Principles

1. **Do not modify any file in `main/`** — all existing pipelines stay unchanged.
2. **Import, don't copy** — model builders, fitting utilities, IO helpers, outlier detectors all imported from `main/`.
3. **Multi-label adaptations live in `main/multiplex/` and `main/multiplex/utils/`** only.
4. **Result schema uses `y_preds_<model>_`, `y_probs_<model>_`, `classes_<model>_`** (no `AC_` prefix — the multiplex result file is separate from main/'s `classification_performances.joblib`, so there is no collision risk and no need for the `AC_` disambiguation prefix).
5. Result arrays are **2D** `(N_test, n_targets)` where main/ uses 1D `(N_test,)` integer arrays.

---

## 3. Target Folder Structure

```
main/multiplex/
├── _brainstorming/
│   └── 20260716-MULTILABEL-MULTIPLEX-INIT.md     ← this file
├── config.py                                      ← multiplex-specific constants  [DONE]
├── 01b_lab_curve_preprocessing.py                 ← multi-label preprocessing     [DONE]
├── 02_outlier_detection_pipeline.py               ← subset of filters             [DONE]
├── 03_main_training.py                            ← multi-label training driver   [DONE]
├── 06_model_prediction_report.py                  ← multi-label HTML report       [DONE]
├── _slurm_jobs/
│   └── lab_multiplex_training.sh                  ← mirrors lab_supcon_training.sh [DONE]
└── utils/
    ├── __init__.py
    └── model_training/
        ├── __init__.py
        └── model_utils_multilabel.py              ← core multi-label extensions   [DONE]
```

---

## 4. `config.py`

**Purpose:** All multiplex-specific constants. Imports `main/config` for shared values.

```python
from config import *   # inherit all base constants

LAB_MULTIPLEX_FOLDER = os.path.join(BASE_FOLDER, "LAB_Multiplex")

FILE_MAPPING = {'01_ACA_qdPCR': 'dPCR_Dataset_Multiplex.csv'}
FILE_CONC    = {'01_ACA_qdPCR': 'Conc'}
FILE_TARGET  = {'01_ACA_qdPCR': 'LoadedPanels'}

# Simple string separator (one dataset → no need for per-dataset dict)
MULTIPLEX_SEPARATOR = '_'

MULTIPLEX_TARGETS = ['KPC', 'NDM', 'VIM']   # sorted, deterministic binarization

TRAINING_DATA_PATH          = 'curve_for_training_ml.joblib'
TRAINING_RESULT_PATH        = 'classification_performances_ml.joblib'
TRAINING_10FOLD_RESULT_PATH = 'classification_performances_ml_10fold.joblib'

OUTLIER_FILTERS = [
    None,
    f'lstm_ae_glb_ds{AE_DOWNSAMPLE_FACTOR}_label_elbow',
    'spatial_knn_label_elbow',
    'spatial_grid_label_elbow',
]

MULTIPLEX_MODELS = [
    # Single-branch standard ST (no SupCon)
    'cnn', 'gru', 'transformer',
    # Single-branch SupCon SC1
    'cnn_supcon', 'gru_supcon', 'transformer_supcon',
    # Dual CNN+GRU: base + SC1/2/3
    'cnn_gru_dual',
    'cnn_gru_dual_supcon', 'cnn_gru_dual_supcon2', 'cnn_gru_dual_supcon3',
    # Dual CNN+Trans: base + SC1/2/3
    'cnn_trans_dual',
    'cnn_trans_dual_supcon', 'cnn_trans_dual_supcon2', 'cnn_trans_dual_supcon3',
    # RCFD — GRU early encoder, CNN+GRU dual
    'gru_rcfd_cgd',
    'gru_rcfd_cgd_supcon_mtl', 'gru_rcfd_cgd_supcon2_mtl', 'gru_rcfd_cgd_supcon3_mtl',
    # RCFD — GRU early encoder, CNN+Trans dual
    'gru_rcfd_ctd',
    'gru_rcfd_ctd_supcon_mtl', 'gru_rcfd_ctd_supcon2_mtl', 'gru_rcfd_ctd_supcon3_mtl',
    # RCFD — Transformer early encoder, CNN+GRU dual
    'trans_rcfd_cgd',
    'trans_rcfd_cgd_supcon_mtl', 'trans_rcfd_cgd_supcon2_mtl', 'trans_rcfd_cgd_supcon3_mtl',
    # RCFD — Transformer early encoder, CNN+Trans dual
    'trans_rcfd_ctd',
    'trans_rcfd_ctd_supcon_mtl', 'trans_rcfd_ctd_supcon2_mtl', 'trans_rcfd_ctd_supcon3_mtl',
]  # 30 total
```

**Design decisions:**
- `MULTIPLEX_SEPARATOR` is a plain string (not a dict) — one dataset, no per-dataset routing needed.
- `cosine_recon` and `attn_recon` variants **omitted** — they require spatial pixel coordinates (`row`/`col`) which the flat-CSV lab data does not have.
- `ML_MODEL_KEY_MAP` and `ML_MODEL_PRINT_MAP` live in `model_utils_multilabel.py` (not in config).
- Spatial filters listed in `OUTLIER_FILTERS` for structural consistency; they produce no column for flat-CSV lab data and are gracefully skipped.

---

## 5. `01b_lab_curve_preprocessing.py`

**What to reuse (imports from `main/`):**
```python
import sigmoid_fitting as sp
from safe_io import safe_joblib_dump
from pipeline_utils import get_exp_paths
```

**New multi-label logic:**
```python
def parse_multilabel_column(raw_labels, separator='_'):
    """Split 'VIM_NDM' → ['VIM', 'NDM']. CSV must already be clean."""
    return [[t.strip() for t in str(lbl).split(separator) if t.strip()] for lbl in raw_labels]
```

**Output `curve_for_training_ml.joblib` keys** (actual implementation):

| Key | Type | Description |
|---|---|---|
| `curves` | `dict` | `{'ori_curves': np.ndarray (N, T)}` — curve array |
| `timestamps` | `np.ndarray` | Time axis for the curve dimension |
| `label_lists` | `list[list[str]]` | Per-sample target lists e.g. `[['VIM','NDM'], ['KPC'], …]` |
| `all_targets` | `list[str]` | Sorted unique targets: `['KPC', 'NDM', 'VIM']` |
| `well_labels` | `np.ndarray[str]` | Raw combo strings e.g. `'VIM_NDM'` (used for curve plot labels) |
| `concentration` | `np.ndarray[float]` | Per-sample concentration (optional, may be None-filled) |
| `kinetic_features` | `dict` | Populated by `02_outlier_detection_pipeline.py` |

> **Note:** `label_binarized` is **not** saved to the joblib. It is recomputed on-demand in 03 and 06 via `encode_multilabel_for_training(label_lists, all_targets)` — this keeps the joblib schema lean and avoids stale cached arrays if `all_targets` changes.

**Argparse:** `--task_id` (int, default 0), `--exp_folder` (str, default `config.LAB_MULTIPLEX_FOLDER`).

---

## 6. `02_outlier_detection_pipeline.py`

**What to reuse (imports from `main/`):**
```python
from lstm_autoencoder_outlier import run_lstm_autoencoder_pipeline
from spatial_consistency_outlier import (
    run_spatial_consistency_knn_pipeline,
    run_spatial_consistency_grid_pipeline,
)
from model_utils import set_global_determinism
from safe_io import safe_joblib_dump
import sigmoid_fitting as sp
```

**Key differences from `main/02`:**
- Loads `curve_for_training_ml.joblib` (not `curve_for_training.joblib`)
- No per-well encoding (`per_well=False`) — flat-CSV lab data has no well structure
- Spatial filters listed but gracefully skipped (no row/col coords → no column written)
- LSTM-AE downsample factor: `config.AE_DOWNSAMPLE_FACTOR` (= 1)

**Argparse:**

| Flag | Default | Notes |
|---|---|---|
| `--task_id` | `0` | same as main/02 |
| `--exp_folder` | `config.LAB_MULTIPLEX_FOLDER` | — |
| `--force_rerun` | off | same |
| `--fast_mode` | off | same |
| `--filters` | `[]` (empty) | `nargs="*"`, `choices=sorted(ALL_FILTERS)`. **Omitting `--filters` = no filter pipelines run.** Pass `--filters lstm_ae` to run LSTM-AE. |

> **Important:** Unlike `main/02` whose `--filters` defaults to running lstm_ae, multiplex `02` defaults to `[]` — no pipelines run unless explicitly requested. This avoids inadvertently running the expensive AE during debugging.

---

## 7. `utils/model_training/model_utils_multilabel.py`

### 7a. Label encoding

```python
def encode_multilabel_for_training(labels_list, all_targets):
    """Returns (y_binary, y_combo_int, mlb, combo_enc).

    y_binary    (N, n_targets) int8 binary indicator — actual training target
    y_combo_int (N,) int — encoded combination string, used ONLY for StratifiedKFold
    mlb         fitted MultiLabelBinarizer
    combo_enc   fitted LabelEncoder on combination strings
    """
```

Returns **4 values** (original plan said 3 — `mlb` was added to expose the fitted binarizer for downstream use).

### 7b. Model head replacement

```python
def adapt_for_multilabel(model, n_targets):
    """Replace last softmax Dense with sigmoid Dense(n_targets, name='cls_out').
    Returns _StandardMultiLabelModel — a subclass that injects BCE train/test steps.
    """
```

`_StandardMultiLabelModel` is needed because standard functional models have no loss specified at compile time. The subclass provides custom `train_step` / `test_step` with `binary_crossentropy`. The output layer is named `'cls_out'` (not `'cls_out_ml'`) to match the training dict key.

### 7c. Model class hierarchy

All multi-label model classes are **standalone** (not subclasses of the corresponding main/ classes — avoids coupling to main/ private internals):

```
tf.keras.Model
├── _StandardMultiLabelModel          # standard ST (adapt_for_multilabel)
├── MultiLabelSupConModel             # SC1: BCE + Jaccard SupCon on fused embedding
├── MultiLabelSupConBranch2STModel    # SC2: BCE + Jaccard SupCon on CNN + seq branches
├── MultiLabelSupConBranch3STModel    # SC3: BCE + Jaccard SupCon on all 3 branches
└── MultiLabelMTLModel                # RCFD base: UW-SO(BCE + MSE)
    ├── MultiLabelRCFDModel           # RCFD SC0 (base, no SupCon)
    ├── MultiLabelRCFDSupConMTLModel  # RCFD SC1: MTL + Jaccard SupCon on z_cond
    ├── MultiLabelRCFDBranch2MTLModel # RCFD SC2: MTL + Jaccard SupCon on CNN + seq
    └── MultiLabelRCFDBranch3MTLModel # RCFD SC3: MTL + Jaccard SupCon on all 3
```

### 7d. Multi-label metrics

```python
def multilabel_metrics(y_true_binary, y_pred_binary, target_names):
    # Returns dict with: exact_acc, hamming, f1_samples, f1_macro, f1_micro, f1_per_label
```

Signature takes `(y_true_binary, y_pred_binary, target_names)` — no `y_pred_proba` argument (probabilities are accessed directly from the result dict when needed in 06).

### 7e. Model factory dispatch

`ML_FACTORIES` maps model key → `factory_fn(T, n_targets)`. **30 models total:**

| Group | Keys | Count |
|---|---|---|
| Single-branch ST | `cnn`, `gru`, `transformer` | 3 |
| Single-branch SC1 | `cnn_supcon`, `gru_supcon`, `transformer_supcon` | 3 |
| CNN+GRU dual ST | `cnn_gru_dual` | 1 |
| CNN+GRU dual SC1/2/3 | `cnn_gru_dual_supcon/2/3` | 3 |
| CNN+Trans dual ST | `cnn_trans_dual` | 1 |
| CNN+Trans dual SC1/2/3 | `cnn_trans_dual_supcon/2/3` | 3 |
| RCFD GRU+CGD SC0/1/2/3 | `gru_rcfd_cgd[_supcon/2/3_mtl]` | 4 |
| RCFD GRU+CTD SC0/1/2/3 | `gru_rcfd_ctd[_supcon/2/3_mtl]` | 4 |
| RCFD Trans+CGD SC0/1/2/3 | `trans_rcfd_cgd[_supcon/2/3_mtl]` | 4 |
| RCFD Trans+CTD SC0/1/2/3 | `trans_rcfd_ctd[_supcon/2/3_mtl]` | 4 |

RCFD naming: `{seq_encoder}_rcfd_{dual_backbone}` where `seq_encoder ∈ {gru, trans}` and `dual_backbone ∈ {cgd=CNN+GRU dual, ctd=CNN+Trans dual}`.

### 7f. Result schema

`evaluate_outlier_filters_ml()` writes to `results_dict[outlier_filter]` (flat, no Native/Reference nesting):

| Key | Type | Description |
|---|---|---|
| `y_trues_` | `list[(N_test, n_targets) int8]` | Ground-truth binary arrays per fold |
| `y_preds_<m>_` | `list[(N_test, n_targets) int8]` | Thresholded binary predictions per fold |
| `y_probs_<m>_` | `list[(N_test, n_targets) float32]` | Raw sigmoid scores per fold |
| `classes_<m>_` | `list[np.ndarray[str]]` | Target names per fold (always `all_targets`) |
| `y_reg_preds_<m>_` | `list[(N_test,) float]` | RCFD-only: predicted concentrations |
| `y_reg_trues_<m>_` | `list[(N_test,) float]` | RCFD-only: true concentrations (sentinel-masked) |
| `mask_count` | `int` | N used after outlier filter application |
| `y_true_count` | `int` | N after rare-combination drop |

Key pattern: `y_preds_{model_key}_` (no `AC_` prefix — the multiplex result file is separate from main/).

---

## 8. `03_main_training.py`

**Argparse (actual implementation):**

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--task_id` | int | `0` | same as main/03 |
| `--exp_folder` | str | `config.LAB_MULTIPLEX_FOLDER` | — |
| `--n_splits` | int | `1` | 1 = StratifiedShuffleSplit; >1 = StratifiedKFold |
| `--force_rerun` | flag | off | same |
| `--fast_mode` | flag | off | same |
| `--supcon` | int [0-3] | `0` | same as main/03 |
| `--condreg` | flag | off | same as main/03 |
| `--threshold` | float | `0.5` | sigmoid → binary prediction threshold |
| `--rerun_models` | nargs+ | None | restrict `--force_rerun` scope |

**Flags intentionally absent (not applicable for flat CSV):**
- `--training_mode` / `--curve_type` — only `ori_curves` exists; no reference/native split
- `--k_neighbors` — no cosine_recon/attn_recon (no spatial coords)
- `--mtl`, `--lbl_conc`, `--supcon_staged` — not applicable

**`mode_name = _mode` is a cosmetic log label only.** Examples: `"ML"`, `"ML SC1"`, `"RCFD SC0"`. Unlike main/03's `"Native"` / `"Reference"` (which are structural result-dict keys), `_mode` in multiplex/03 is only printed to the console. Results are keyed only by `outlier_filter` (flat structure).

**Model group selection (`_rcfd_by_sc`):**

```python
# ST (non-RCFD)
--supcon 0: ['cnn', 'gru', 'transformer', 'cnn_gru_dual', 'cnn_trans_dual']
--supcon 1: ['cnn_supcon', 'gru_supcon', 'transformer_supcon',
             'cnn_gru_dual_supcon', 'cnn_trans_dual_supcon']
--supcon 2: ['cnn_gru_dual_supcon2', 'cnn_trans_dual_supcon2']
--supcon 3: ['cnn_gru_dual_supcon3', 'cnn_trans_dual_supcon3']

# RCFD (--condreg flag)
--condreg --supcon 0: ['gru_rcfd_cgd',  'gru_rcfd_ctd',  'trans_rcfd_cgd',  'trans_rcfd_ctd']
--condreg --supcon 1: ['gru_rcfd_cgd_supcon_mtl',  'gru_rcfd_ctd_supcon_mtl',
                       'trans_rcfd_cgd_supcon_mtl',  'trans_rcfd_ctd_supcon_mtl']
--condreg --supcon 2: [... supcon2_mtl variants ...]
--condreg --supcon 3: [... supcon3_mtl variants ...]
```

**Checkpoint semantics** (mirrors `make_checkpoint_fn` in main/03):
```python
def checkpoint_fn(current_results):   # current_results = full results_dict
    cached_results.update(current_results)
    safe_joblib_dump(cached_results, results_path, compress=3)
```

**Result structure:** `cached_results = {outlier_filter: res_entry}` — flat, no Native/Reference nesting. All runs with different `--supcon` / `--condreg` combinations accumulate into the same file (same checkpoint pattern as main/03).

---

## 9. `06_model_prediction_report.py`

**Loads:** `classification_performances_ml.joblib` (or `_10fold.joblib` when `--n_splits > 1`).

**Split reconstruction:** `compute_ml_filtered_splits(y_binary, y_combo_int, features_df, outlier_filter, n_splits)` — defined locally (not imported from main/06) because it uses `y_combo_int` for stratification whereas main/06 uses 1D `y_encoded`.

**Report tabs (8 total):**

| # | Tab ID | Content |
|---|---|---|
| 1 | `overview` | Exact-match accuracy bar + F1-macro bar + per-fold stability strip chart |
| 2 | `combo_freq` | Grouped bar: per-combination sample count in train vs test (avg across folds) |
| 3 | `curves` | Correct vs wrong signal plots per label combination |
| 4 | `cm` | Per-label 2×2 confusion matrix (one per gene: KPC, NDM, VIM) |
| 5 | `metrics` | Per-label P/R/F1 table + macro/micro/samples aggregates |
| 6 | `conf` | Max sigmoid score histogram: exact-match vs wrong |
| 7 | `roc` | Per-label ROC + PR curves |
| 8 | `regression` | RCFD scatter + metrics table (auto-detected, tab absent for non-RCFD) |

**Argparse:** `--task_id`, `--exp_folder`, `--n_splits` (must match the 03 run), `--force_rerun`, `--outlier_filter` (nargs*, default `config.OUTLIER_FILTERS`; pass `'None'` string for baseline).

---

## 10. Critical Multi-Label Algorithmic Considerations

### 10a. SupCon Contrastive Loss — Continuous Jaccard

Binary `pos_mask[i,j] = (y[i] == y[j])` is too coarse for multi-label: `VIM` and `VIM_NDM` share real biological signal yet form pure negative pairs.

**Decision: continuous Jaccard — no threshold.**

```python
def _jaccard_weight_matrix(y_binary):
    y            = tf.cast(y_binary, tf.float32)
    intersection = tf.matmul(y, y, transpose_b=True)
    row_sums     = tf.reduce_sum(y, axis=1)
    union        = (tf.expand_dims(row_sums, 1) + tf.expand_dims(row_sums, 0) - intersection)
    return (intersection / (union + 1e-8)) * (1.0 - tf.eye(tf.shape(y)[0]))
```

Semantics: identical combos → J=1.0 (full pull); partial overlap → fractional; disjoint → J=0.0 (pure negative). No new hyperparameter; no `--supcon_jaccard_threshold` flag.

**Weighted loss:**
```python
def _supcon_loss_jaccard(embeddings, pos_mask, temp=SUPCON_TEMP):
    # Standard Khosla denominator (all non-self pairs)
    # Numerator weighted by continuous pos_mask instead of binary
    pos_sum     = tf.reduce_sum(pos_mask, axis=1)
    has_pos     = tf.cast(pos_sum > 0, tf.float32)
    per_anchor  = -tf.reduce_sum(log_prob * pos_mask, axis=1) / (pos_sum + 1e-8)
    return tf.reduce_mean(per_anchor * has_pos)
```

### 10b. RCFD with Multi-Label

`_build_rcfd_backbone(inputs, enc_type, dual_type)` is unchanged (imported from `main/`):
- `enc_type`: `'gru'` or `'transformer'` (early regression encoder architecture)
- `dual_type`: `'cgd'` (CNN+GRU dual) or `'ctd'` (CNN+Trans dual)

The classification head is replaced with sigmoid in `_ml_cls_head(z, n_targets)`. Regression head (concentration MSE) is unchanged. Combined loss uses UW-SO (Uncertainty Weighting Single Output) balancing BCE + MSE.

All 16 RCFD variants: `{gru,trans}_rcfd_{cgd,ctd}[_supcon{,2,3}_mtl]`.

### 10c. Rare Combination Handling

`evaluate_outlier_filters_ml` drops combinations with fewer than 2 samples before splitting (same guard as main/02 for rare classes). With the current dataset all 7 combinations have >1500 samples — the guard is a safety net only.

### 10d. Concentration Sentinel Masking

When `--condreg` is passed, missing or zero-concentration samples are set to `_REG_SENTINEL = -1.0` before encoding (mirrors main/03 MTL handling):
```python
y_concentration = np.where(
    np.isnan(_float_arr) | (_float_arr == 0.0),
    _REG_SENTINEL, _float_arr)
```
The regression MSE loss only accumulates over samples where `y_reg != REG_SENTINEL`.

---

## 11. Implementation Checklist

### `multiplex/config.py` ✅
- [x] `from config import *` — inherits base constants
- [x] `LAB_MULTIPLEX_FOLDER`, `FILE_MAPPING`, `FILE_CONC`, `FILE_TARGET`
- [x] `MULTIPLEX_SEPARATOR = '_'` (string, not dict)
- [x] `TRAINING_DATA_PATH`, `TRAINING_RESULT_PATH`, `TRAINING_10FOLD_RESULT_PATH`
- [x] `OUTLIER_FILTERS` (4 entries; spatial listed but inert for flat-CSV lab data)
- [x] `MULTIPLEX_MODELS` (30 entries)

### Pre-implementation: CSV fix
- [ ] Edit `LAB_Multiplex/01_ACA_qdPCR/dPCR_Dataset_Multiplex.csv`: change `VIM_NDM¸` → `VIM_NDM`

### `multiplex/01b_lab_curve_preprocessing.py` ✅
- [x] `parse_multilabel_column(raw_labels, separator)` — splits combo strings to lists
- [x] Saves `label_lists`, `all_targets`, `well_labels` to joblib (no `label_binarized` — computed on demand)
- [x] Argparse: `--task_id=0`, `--exp_folder=LAB_MULTIPLEX_FOLDER`

### `multiplex/02_outlier_detection_pipeline.py` ✅
- [x] `ALL_FILTERS = {"lstm_ae", "spatial_knn", "spatial_grid"}`, `_DEFAULT_FILTERS = {"lstm_ae"}`
- [x] `--filters nargs="*", choices=sorted(ALL_FILTERS), default=[]` — omitting = no pipelines
- [x] `per_well=False`, `has_spatial=False` (flat lab CSV)
- [x] `--fast_mode`, `--force_rerun` flags

### `multiplex/utils/model_training/model_utils_multilabel.py` ✅
- [x] `_jaccard_weight_matrix(y_binary)` — continuous Jaccard float matrix
- [x] `_supcon_loss_jaccard(embeddings, pos_mask)` — weighted SupCon loss
- [x] `encode_multilabel_for_training()` — returns `(y_binary, y_combo_int, mlb, combo_enc)`
- [x] `_StandardMultiLabelModel` — functional model wrapper with BCE train/test steps
- [x] `adapt_for_multilabel(model, n_targets)` — replaces softmax with sigmoid, returns `_StandardMultiLabelModel`
- [x] `MultiLabelSupConModel`, `MultiLabelSupConBranch2STModel`, `MultiLabelSupConBranch3STModel`
- [x] `MultiLabelMTLModel`, `MultiLabelRCFDModel`, `MultiLabelRCFDSupConMTLModel`, `MultiLabelRCFDBranch2MTLModel`, `MultiLabelRCFDBranch3MTLModel`
- [x] `multilabel_metrics(y_true_binary, y_pred_binary, target_names)` — metric dict
- [x] `evaluate_outlier_filters_ml(...)` — full training loop with checkpoint
- [x] `print_ml_results_summary(...)` — console text summary (replaces `plot_ml_results_ml`)
- [x] `ML_FACTORIES` (30 entries), `_RCFD_ML_KEYS` (16), `_SUPCON_ML_KEYS` (21)
- [x] `ML_MODEL_KEY_MAP` — `{m: (f'y_preds_{m}_', f'y_probs_{m}_', f'classes_{m}_')}` for all 30
- [x] `ML_MODEL_PRINT_MAP` — display names for all 30 models

### `multiplex/03_main_training.py` ✅
- [x] Argparse: `--task_id`, `--exp_folder`, `--n_splits`, `--force_rerun`, `--fast_mode`, `--supcon`, `--condreg`, `--threshold`, `--rerun_models`
- [x] Inapplicable flags omitted: `--training_mode`, `--curve_type`, `--k_neighbors`, `--mtl`, `--lbl_conc`
- [x] `--rerun_models` nullified without `--force_rerun`
- [x] Concentration masking with `_REG_SENTINEL` scoped to `--condreg` only
- [x] `_rcfd_by_sc` dict: 4 models per SC level (all 4 RCFD families)
- [x] `checkpoint_fn`: `cached_results.update(current_results)` then `safe_joblib_dump`
- [x] Flat result structure: `{outlier_filter: res_entry}` (no Native/Reference nesting)

### `multiplex/06_model_prediction_report.py` ✅
- [x] `compute_ml_filtered_splits()` — mirrors training split logic exactly with `y_combo_int`
- [x] 8 tabs: Overview, Combo Freq, Curves, Label CMs, Label Metrics, Confidence, ROC/PR, Regression
- [x] Per-fold stability chart (`fold_accs` already in %; no double-multiply)
- [x] Combination frequency chart (`render_ml_combination_freq_chart`)
- [x] RCFD regression tab auto-detected from result keys
- [x] `--outlier_filter` converts `'None'` string → Python `None`

### `multiplex/_slurm_jobs/lab_multiplex_training.sh` ✅
- [x] `--array=0` (single task, one dataset)
- [x] `02` call has NO `--filters` arg (= `default=[]` = no filter pipelines at training time)
- [x] ST loop: `--supcon 0/1/2/3` (4 runs)
- [x] RCFD loop: `--condreg --supcon 0/1/2/3` (4 runs)

---

## 12. Open Questions / Risks

1. **Spatial outlier filters on lab data** (by design): `spatial_knn_label_elbow` and `spatial_grid_label_elbow` produce no column for flat-CSV data. Training gracefully skips absent columns. Listed in `OUTLIER_FILTERS` for structural consistency with main/.

2. **Continuous Jaccard for SupCon** (resolved): `_jaccard_weight_matrix()` — no threshold, no CLI flag. J=1.0 for identical combos, J=0.0 for disjoint, fractional for partial overlap.

3. **Sigmoid threshold** (configurable): Default 0.5 via `--threshold`. Per-class threshold optimization can be done post-training in a notebook: load `classification_performances_ml.joblib`, sweep `y_probs_*` arrays per class, pick F1-maximising thresholds. No retraining required.

4. **RCFD concentration for multi-label** (unchanged): Concentration is a scalar per droplet regardless of how many targets are present. The regression head and FiLM conditioning are semantically identical to the single-label case.

5. **`iterstrat` dependency**: Not needed — 7 unique combinations with StratifiedKFold on `y_combo_int` is sufficient. Adopt `MultilabelStratifiedKFold` if a future dataset has 20+ sparse combinations.

6. **`plot_ml_results_ml`** (resolved as `print_ml_results_summary`): Console summary is sufficient; full HTML visualisation is in 06.

---

## 13. Dependency & Import Map

```
multiplex/config.py
    ← main/config.py  (from config import *)

multiplex/01b_lab_curve_preprocessing.py
    ← main/sigmoid_fitting.py
    ← main/safe_io.py
    ← main/config.py  (shared constants)
    ← multiplex/config.py  (FILE_MAPPING, FILE_TARGET, MULTIPLEX_SEPARATOR, …)
    NEW: sklearn.preprocessing.MultiLabelBinarizer, numpy, pandas

multiplex/02_outlier_detection_pipeline.py
    ← main/utils/02_outlier_detection/lstm_autoencoder_outlier.py
    ← main/utils/02_outlier_detection/spatial_consistency_outlier.py
    ← main/safe_io.py
    ← main/sigmoid_fitting.py
    ← main/utils/model_training/model_utils.py  (set_global_determinism)
    ← multiplex/config.py

multiplex/utils/model_training/model_utils_multilabel.py
    ← main/utils/model_training/model_utils.py
        (create_cnn_model, create_gru_model, create_transformer_model,
         create_cnn_gru_dual_model, create_cnn_transformer_dual_model)
    ← main/utils/model_training/model_utils_mtl.py
        (MTLModel, REG_SENTINEL, _normalize_concentration, _inverse_normalize_concentration,
         _build_cnn_backbone_mtl, _build_gru_backbone_mtl, _build_transformer_backbone_mtl,
         _build_cnn_gru_dual_branches_mtl, _build_cnn_trans_dual_branches_mtl)
    ← main/utils/model_training/model_utils_supcon.py
        (SUPCON_TEMP, SUPCON_LAMBDA, _proj_head)
    ← main/utils/model_training/model_utils_rcfd.py
        (_build_rcfd_backbone)
    ← main/safe_io.py  (safe_keras_save)
    NEW: sklearn.metrics.*, sklearn.model_selection.StratifiedKFold/ShuffleSplit

multiplex/03_main_training.py
    ← multiplex/config.py
    ← multiplex/utils/model_training/model_utils_multilabel.py
    ← main/safe_io.py
    ← main/utils/model_training/model_utils.py  (set_global_determinism)
    ← main/utils/model_training/model_utils_mtl.py  (REG_SENTINEL)

multiplex/06_model_prediction_report.py
    ← main/utils/html_utils.py  (build_tabbed_html, _panel, _fig_to_buf, _buf_to_img_html)
    ← main/utils/model_training/model_utils_mtl.py  (REG_SENTINEL)
    ← multiplex/config.py
    ← multiplex/utils/model_training/model_utils_multilabel.py
        (ML_MODEL_KEY_MAP, ML_MODEL_PRINT_MAP, encode_multilabel_for_training)
    NEW: sklearn.metrics.*, scipy.stats, matplotlib
```

---

## 14. Verification Plan

After implementation, verify end-to-end with the single `01_ACA_qdPCR` dataset:

0. **CSV fix** (one-time): confirm `dPCR_Dataset_Multiplex.csv` has no `VIM_NDM¸` rows

1. **01b:**
   ```bash
   cd main/multiplex
   python 01b_lab_curve_preprocessing.py --task_id 0
   ```
   Check: `curve_for_training_ml.joblib` has `label_lists`, `all_targets=['KPC','NDM','VIM']`, `well_labels` (combo strings), and `curves['ori_curves'].shape == (N, T)`.

2. **02:**
   ```bash
   python 02_outlier_detection_pipeline.py --task_id 0 --filters lstm_ae
   ```
   Check: `kinetic_features['01_ACA_qdPCR']` has `lstm_ae_glb_ds1_label_elbow` column.

3. **03 (smoke — 1 model, default n_splits=1):**
   ```bash
   python 03_main_training.py --task_id 0 --rerun_models cnn_gru_dual
   ```
   Check: `classification_performances_ml.joblib` with `{None: {'y_trues_': [2D array], 'y_preds_cnn_gru_dual_': [2D array], …}}`.

4. **03 (SupCon variant):**
   ```bash
   python 03_main_training.py --task_id 0 --supcon 1 --rerun_models cnn_gru_dual_supcon
   ```
   Check: `y_preds_cnn_gru_dual_supcon_` merged into same joblib; shape `(N_test, 3)`.

5. **03 (RCFD):**
   ```bash
   python 03_main_training.py --task_id 0 --condreg --rerun_models gru_rcfd_cgd
   ```
   Check: `y_reg_preds_gru_rcfd_cgd_` and `y_reg_trues_gru_rcfd_cgd_` in result entry.

6. **06:**
   ```bash
   python 06_model_prediction_report.py --task_id 0
   ```
   Check: HTML at `model_performance_viz/ml_report__none__nsplits1.html` with 8 tabs including "Combo Freq" and per-label CMs.

---

*End of implementation reference. Code is the ground truth; update this document when the code changes.*
