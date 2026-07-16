# Multi-Label Multiplex Classification — Init Brainstorming
**Date:** 2026-07-16  
**Scope:** Introduce `main/multiplex/` as a self-contained pipeline for multi-label droplet classification, reusing existing `main/` code without touching it.

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
4. **Reuse the same joblib result schema** — `y_trues_`, `y_preds_AC_<model>_`, `y_probs_AC_<model>_`, `classes_AC_<model>_` — but with 2D binary arrays where the existing pipeline uses 1D integer arrays.

---

## 3. Target Folder Structure

```
main/multiplex/
├── _brainstorming/
│   └── 20260716-MULTILABEL-MULTIPLEX-INIT.md     ← this file
├── config.py                                      ← multiplex-specific constants
├── 01b_lab_curve_preprocessing.py                 ← multi-label preprocessing
├── 02_outlier_detection_pipeline.py               ← subset of filters
├── 03_main_training.py                            ← multi-label training driver
├── 06_model_prediction_report.py                  ← multi-label HTML report
└── utils/
    ├── __init__.py
    └── model_training/
        ├── __init__.py
        └── model_utils_multilabel.py              ← core multi-label extensions
```

---

## 4. `config.py`

**Purpose:** All multiplex-specific constants. Imports `main/config` for shared values.

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config as _base_config

# Inherit everything from the base config
from config import *

# ── Multiplex dataset registry ─────────────────────────────────────────────
LAB_MULTIPLEX_FOLDER = '/vol/bitbucket/gk225/POC_DDM_datasets/LAB_Multiplex'

FILE_MAPPING = {
    '01_ACA_qdPCR': 'dPCR_Dataset_Multiplex.csv',
}
FILE_CONC = {
    '01_ACA_qdPCR': 'Conc',
}
FILE_TARGET = {
    '01_ACA_qdPCR': 'LoadedPanels',
}
MULTIPLEX_SEPARATOR = {
    '01_ACA_qdPCR': '_',
}

# ── Result file names (separate from LAB_DDM results) ───────────────────────
TRAINING_DATA_PATH         = 'curve_for_training_ml.joblib'
TRAINING_RESULT_PATH       = 'classification_performances_ml.joblib'
TRAINING_10FOLD_RESULT_PATH = 'classification_performances_ml_10fold.joblib'

# ── Outlier filters for multiplex (subset of main/config) ────────────────────
# NOTE: spatial_knn/grid are listed but will produce NO column in kinetic_features
# for flat-CSV lab data (no row/col spatial coords). Training will skip them with
# a warning. They are listed to keep the filter-set structure consistent.
OUTLIER_FILTERS = [
    None,
    f'lstm_ae_glb_ds{AE_DOWNSAMPLE_FACTOR}_label_elbow',  # = 'lstm_ae_glb_ds1_label_elbow'
    'spatial_knn_label_elbow',
    'spatial_grid_label_elbow',
]

# ── Models to train ────────────────────────────────────────────────────────
MULTIPLEX_MODELS = [
    'cnn_gru_dual',
    'cnn_gru_dual_cosine_recon',
    'cnn_gru_dual_attn_recon',
    'cnn_gru_dual_supcon',
    'cnn_gru_dual_cosine_recon_supcon',
    'cnn_gru_dual_attn_recon_supcon',
    'cnn_gru_dual_supcon2',
    'cnn_gru_dual_cosine_recon_supcon2',
    'cnn_gru_dual_attn_recon_supcon2',
    'cnn_gru_dual_supcon3',
    'cnn_gru_dual_cosine_recon_supcon3',
    'cnn_gru_dual_attn_recon_supcon3',
    'gru_rcfd_cgd',
    'gru_rcfd_cgd_supcon_mtl',
    'gru_rcfd_cgd_supcon2_mtl',
    'gru_rcfd_cgd_supcon3_mtl',
]
```

**Key decisions:**
- `TRAINING_DATA_PATH` uses `_ml` suffix to avoid collision with existing LAB results in the same folder.
- `OUTLIER_FILTERS` uses Python f-string so it resolves correctly when `AE_DOWNSAMPLE_FACTOR=1`.
- Spatial filters listed but effectively inert for lab data (no coords).

---

## 5. `01b_lab_curve_preprocessing.py`

**Purpose:** Read the multiplex CSV, extract curves and kinetic features, produce `curve_for_training_ml.joblib` with the same structure as `main/01b` plus three new keys.

**What to reuse (import from `main/`):**
```python
sys.path.insert(0, parent_dir)
import sigmoid_fitting as sp
from safe_io import safe_joblib_dump
from config import (WINDOW_SIZE_ORI, WINDOW_SIZE_1STDER, AE_DOWNSAMPLE_FACTOR)
# Reuse curve-loading helpers from main/01b by importing them:
from importlib.util import spec_from_file_location, module_from_spec
# Load main/01b functions: sigmoid_fitting_5p, _read_table, get_numeric_sort_key,
# _fit_single_curve, extract_kinetic_features, etc.
```

**New multi-label logic (in `multiplex/01b`):**
```python
from sklearn.preprocessing import MultiLabelBinarizer

def parse_multilabel(raw_labels, separator):
    """Split 'VIM_NDM' → ['VIM', 'NDM']. CSV must already be clean."""
    return [[t.strip() for t in str(lbl).split(separator) if t.strip()] for lbl in raw_labels]

def binarize_labels(label_lists, all_targets=None):
    """MultiLabelBinarizer → binary matrix (N, n_targets)."""
    mlb = MultiLabelBinarizer(classes=all_targets)
    binary = mlb.fit_transform(label_lists)
    return binary, list(mlb.classes_)
```

**New fields added to `curve_for_training_ml.joblib`:**

| Key | Type | Description |
|---|---|---|
| `Y_well` | `np.ndarray[str]` | Raw label strings (`'VIM_NDM'`, etc.) — same as existing |
| `labels` | `list[list[str]]` | Split labels per sample (`[['VIM','NDM'], ['VIM'], ...]`) |
| `all_targets` | `list[str]` | Sorted unique targets (`['KPC','NDM','VIM']`) |
| `label_binarized` | `np.ndarray[int]` | Shape `(N, 3)` — binary indicator matrix |
| `concentration` | `np.ndarray[float]` | Per-sample concentration (same as existing) |

**Argparse defaults** — same as `main/01b_lab_curve_preprocessing.py`:
- `--task_id` default `0`
- `--exp_folder` default `multiplex/config.LAB_MULTIPLEX_FOLDER`

**Output:** `{exp_path}/curve_for_training_ml.joblib`

Same inner structure as `main/01b` output (datasets list, curve arrays, kinetic_features DataFrames) plus the new multi-label fields.

---

## 6. `02_outlier_detection_pipeline.py`

**Purpose:** Run LSTM-AE global outlier filter on multiplex data, write outlier labels to `kinetic_features` in `curve_for_training_ml.joblib`.

**What to reuse (import directly):**
```python
from lstm_autoencoder_outlier import run_lstm_autoencoder_pipeline
# Spatial filters imported but will produce no output (no coords):
from spatial_consistency_outlier import (
    run_spatial_consistency_knn_pipeline,
    run_spatial_consistency_grid_pipeline,
)
```

**What is different from `main/02`:**
- Loads `curve_for_training_ml.joblib` (not `curve_for_training.joblib`)
- Passes `config` = `multiplex.config` so `OUTLIER_FILTERS` uses the subset
- Spatial pipelines called but skipped gracefully (no coords → no column written)
- LSTM-AE runs on `ori_curves` as normal (multi-label has no effect on the AE)
- Kinetic feature extraction (`extract_kinetic_features`, `get_send`) imported from `main/02` directly (no changes needed — features depend only on the curve, not on the label)

**Argparse defaults — same as `main/02_outlier_detection_pipeline.py`:**
- `--task_id` default `0`
- `--exp_folder` default `multiplex/config.LAB_MULTIPLEX_FOLDER`
- `--force_rerun` flag, off
- `--fast_mode` flag, off
- `--filters` nargs+, default `None` (runs all `OUTLIER_FILTERS`)

**Why MSC/AMF/KNN are not included:**
- User specified only `[None, lstm_ae_..., spatial_knn_..., spatial_grid_...]`
- MSC (Mahalanobis) uses `kinetic_features` stratified by class — multi-label class structure complicates this; excluded for now
- AMF (Isolation Forest) is class-agnostic but not requested
- KNN fingerprint filter requires per-class KNN — multi-label makes "same class" ambiguous; excluded

---

## 7. `utils/model_training/model_utils_multilabel.py`

This is the **core adaptation module**. It does not re-implement model training from scratch — it wraps and adapts `model_utils.evaluate_outlier_filters()`.

### 7a. Label encoding

```python
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

def encode_multilabel_for_training(labels_list, all_targets):
    """Returns (y_binary, y_combo_int, encoder) where:
    - y_binary: (N, n_targets) binary indicator matrix — actual training target
    - y_combo_int: integer-encoded combination — used ONLY for StratifiedKFold stratification
    - encoder: fitted LabelEncoder on combination strings
    """
    mlb = MultiLabelBinarizer(classes=all_targets)
    y_binary = mlb.fit_transform(labels_list)
    combo_strs = ['_'.join(sorted(lbl)) for lbl in labels_list]
    enc = LabelEncoder()
    y_combo_int = enc.fit_transform(combo_strs)
    return y_binary, y_combo_int, enc
```

**Rationale:** We need two encodings:
1. `y_binary` — the actual training target (sigmoid output) and the input to the Jaccard-based SupCon mask
2. `y_combo_int` — integer label for `StratifiedKFold` stratification **only** (SupCon now uses Jaccard on `y_binary`)

Since there are only 7 unique combinations, `StratifiedKFold` on `y_combo_int` works fine without an external `iterstrat` dependency.

### 7b. Model head replacement

```python
import tensorflow as tf

def adapt_for_multilabel(model, n_targets, threshold=0.5):
    """Replace the softmax classification output with sigmoid for multi-label.
    
    Finds the last Dense(n_classes, softmax) layer and rebuilds the output.
    Works for standard ST models. For custom subclass models (SupCon, RCFD)
    a different strategy is needed (see 7c).
    """
    # Find the layer before the softmax output
    for layer in reversed(model.layers):
        if hasattr(layer, 'activation') and layer.activation.__name__ == 'softmax':
            pre_output = layer.input
            new_out = tf.keras.layers.Dense(
                n_targets, activation='sigmoid', name='cls_out_ml'
            )(pre_output)
            return tf.keras.Model(inputs=model.inputs, outputs=new_out)
    raise ValueError("Could not find softmax output layer to replace")
```

**For SupCon models** (Keras subclasses overriding `train_step`): their classification output is defined inside `call()`. The `adapt_for_multilabel` function cannot simply remap layers.

**Solution for SupCon/RCFD:** Create thin multi-label subclasses in `model_utils_multilabel.py`. The key change is the contrastive mask construction: instead of using integer class equality, compute pairwise Jaccard similarity on `y_binary` and use it as the positive-pair mask (see Section 10a for details).

```python
class MultiLabelSupConSTModel(SupConSTModel):
    """Multi-label SupCon: sigmoid cls head + Jaccard positive-pair mask."""
    def __init__(self, *args, n_targets, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_targets = n_targets
        self.cls_out = tf.keras.layers.Dense(n_targets, activation='sigmoid', name='cls_out_ml')
    
    def train_step(self, data):
        # x, (y_binary, y_combo_int) = data
        # contrastive_loss uses _jaccard_weight_matrix(y_binary) — continuous, no threshold
        # cls_loss uses binary_crossentropy(y_binary, sigmoid(cls_out))
        ...
```

The standard (non-subclass) models use the simpler wrapper:
```python
def create_multilabel_model(base_factory, T, n_targets):
    base_model = base_factory(T, n_targets)  # build with n_targets, then swap head
    return adapt_for_multilabel(base_model, n_targets)
```

**Summary:** Standard models (cnn_gru_dual + cosine/attn_recon, 3 models) use `adapt_for_multilabel`. SupCon subclass models (SC1/SC2/SC3 × 3 variants = 9 models) use `MultiLabelSupConSTModel` and its Branch2/Branch3 equivalents. RCFD models (4 models) use `MultiLabelRCFDModel` variants.

### 7c. Multi-label metrics

```python
from sklearn.metrics import (
    accuracy_score,      # exact-match (subset accuracy)
    hamming_loss,        # fraction of wrong label bits
    f1_score,            # multi-label F1
)

def multilabel_metrics(y_true_binary, y_pred_binary, y_pred_proba, target_names):
    """Returns a dict of standard multi-label classification metrics."""
    return {
        'exact_acc':   accuracy_score(y_true_binary, y_pred_binary),
        'hamming':     hamming_loss(y_true_binary, y_pred_binary),
        'f1_samples':  f1_score(y_true_binary, y_pred_binary, average='samples',  zero_division=0),
        'f1_macro':    f1_score(y_true_binary, y_pred_binary, average='macro',    zero_division=0),
        'f1_micro':    f1_score(y_true_binary, y_pred_binary, average='micro',    zero_division=0),
        'f1_per_label': f1_score(y_true_binary, y_pred_binary, average=None,     zero_division=0),
    }
```

### 7d. `evaluate_outlier_filters_ml()` — adapted training loop

The function signature mirrors `evaluate_outlier_filters()` from `model_utils.py` but accepts multi-label inputs:

```python
def evaluate_outlier_filters_ml(
    X_curves,           # (N, T) float curve array
    features_df,        # DataFrame with kinetic features + outlier filter columns
    y_binary,           # (N, n_targets) binary indicator matrix
    y_combo_int,        # (N,) int for stratification + SupCon pairing
    all_targets,        # list[str] e.g. ['KPC', 'NDM', 'VIM']
    outlier_filters,    # list from multiplex/config.OUTLIER_FILTERS
    dataset_name,
    mode_name,
    cached_results=None,
    models=MULTIPLEX_MODELS,
    n_splits=5,
    checkpoint_fn=None,
    save_model_dir=None,
    y_concentration=None,   # for RCFD regression head
    threshold=0.5,          # sigmoid → binary prediction threshold
    ...
):
```

**Key differences from `evaluate_outlier_filters()`:**

| Aspect | Original | Multi-label version |
|---|---|---|
| `y_encoded` | 1D int array | 2D binary (N, n_targets) |
| Stratification key | `y_encoded` | `y_combo_int` |
| Splitter | `StratifiedKFold(y_encoded)` | `StratifiedKFold(y_combo_int)` |
| Model compile loss | `sparse_categorical_crossentropy` | `binary_crossentropy` |
| Model output | softmax (n_classes) | sigmoid (n_targets) |
| Predictions | `argmax(proba)` → int | `proba > threshold` → binary |
| `y_trues_` | list of 1D int arrays | list of 2D binary arrays |
| `y_preds_*` | list of 1D int arrays | list of 2D binary arrays |
| `y_probs_*` | list of (N, n_classes) | list of (N, n_targets) |
| `classes_*` | list of int arrays | list of target name arrays |
| Primary metric logged | accuracy (%) | exact_acc + hamming + f1_samples |
| SupCon pair mining | `y_encoded` int | `y_combo_int` |
| Rare class filter | `class_count < 2` | combination count < 2 |

**What to import from `model_utils.py`:**
```python
from model_utils import (
    build_neighbor_curve_stack,   # for cosine/attn_recon spatial models
    reconstruct_curves_cosine,
    set_global_determinism,
    CurveResampler,
    _remap_global_splits,         # reuse split remapping helper
)
```

**Model factories imported from existing utils:**
```python
# Standard models
from model_utils import (
    create_cnn_gru_dual_model,
    create_cnn_gru_dual_cosine_recon_model,
    create_cnn_gru_dual_attn_recon_model,
)
# SupCon models
from model_utils_supcon import (
    create_cnn_gru_dual_supcon_model,
    create_cnn_gru_dual_cosine_recon_supcon_model,
    create_cnn_gru_dual_attn_recon_supcon_model,
    create_cnn_gru_dual_supcon2_model,
    create_cnn_gru_dual_cosine_recon_supcon2_model,
    create_cnn_gru_dual_attn_recon_supcon2_model,
    create_cnn_gru_dual_supcon3_model,
    create_cnn_gru_dual_cosine_recon_supcon3_model,
    create_cnn_gru_dual_attn_recon_supcon3_model,
)
# RCFD models
from model_utils_rcfd import (
    create_gru_rcfd_cgd_model,
    create_gru_rcfd_cgd_supcon_mtl_model,
    create_gru_rcfd_cgd_supcon2_mtl_model,
    create_gru_rcfd_cgd_supcon3_mtl_model,
    RCFDModel, RCFDSupConMTLModel,
)
```

---

## 8. `03_main_training.py`

**Purpose:** CLI driver for multi-label training. Mirrors `main/03_main_training.py` structure exactly, including argparse defaults.

**What to reuse:**
```python
import sys, os
sys.path.insert(0, parent)
from safe_io import safe_joblib_dump
from pipeline_utils import get_exp_paths, check_task_id
from model_utils import set_global_determinism
```

**What differs from `main/03`:**
- Loads `curve_for_training_ml.joblib` (`y_binary`, `labels`, `all_targets` already present)
- Calls `evaluate_outlier_filters_ml()` from `multiplex/utils/model_training/model_utils_multilabel.py`
- No `LabelEncoder` call (binarization already in joblib)
- No `--lbl_conc` (LC label consolidation not applicable for multi-label)
- No `--mtl` toggle (RCFD always has concentration regression head; standard models always use sigmoid cls head)
- No `--supcon_jaccard_threshold` flag (continuous Jaccard has no threshold parameter)
- Result file: `classification_performances_ml.joblib`

**Argparse — same flags and same defaults as `main/03_main_training.py`:**

| Flag | Type | Default | Notes |
|---|---|---|---|
| `--task_id` | int | `0` | same |
| `--exp_folder` | str | `multiplex/config.LAB_MULTIPLEX_FOLDER` | overrides base default |
| `--n_splits` | int | `1` | **same as main/03** (not 5) |
| `--force_rerun` | flag | off | same |
| `--training_mode` | nargs+ | `["native"]` | same |
| `--curve_type` | nargs+ | `["ori_curve","ori_curve_avg","ori_curve_wavelet_sym8"]` | same |
| `--fast_mode` | flag | off | same |
| `--k_neighbors` | int | `24` | same |
| `--supcon` | int choices[0-3] | `0` | **same as main/03** — `0`=standard, `1/2/3`=SupCon SC1/2/3 |
| `--condreg` | flag | off | **same as main/03** — enables RCFD (`gru_rcfd_cgd*`) models |
| `--rerun_models` | nargs+ | `None` | same |
_(no `--supcon_jaccard_threshold` — continuous Jaccard has no threshold)_

**How `--supcon` and `--condreg` select models (same logic as `main/03`):**

```
--supcon 0  (no flag): train cnn_gru_dual, cnn_gru_dual_cosine_recon, cnn_gru_dual_attn_recon
--supcon 1:            train cnn_gru_dual_supcon, ..._cosine_recon_supcon, ..._attn_recon_supcon
--supcon 2:            train cnn_gru_dual_supcon2, ..._cosine_recon_supcon2, ..._attn_recon_supcon2
--supcon 3:            train cnn_gru_dual_supcon3, ..._cosine_recon_supcon3, ..._attn_recon_supcon3
--condreg:             train gru_rcfd_cgd, gru_rcfd_cgd_supcon_mtl (depending on --supcon value)
```

All results merge into the same `classification_performances_ml.joblib` (same checkpoint pattern as `main/03`).

---

## 9. `06_model_prediction_report.py`

**Purpose:** HTML prediction report for multi-label results. Reuses `main/06` rendering infrastructure.

**What to reuse:**
```python
from html_utils import _fig_to_buf, _buf_to_img_html, _panel, build_tabbed_html
# Import the split recreation helper from main/06:
import importlib
_06 = importlib.import_module('..06_model_prediction_report')  # or direct import path
compute_filtered_splits = _06.compute_filtered_splits
```

**What differs:**
- Loads `classification_performances_ml.joblib`
- `y_trues_` and `y_preds_*` are 2D binary arrays → needs threshold/binarization display
- **Confusion matrix** → replaced with per-label confusion matrices (one per target gene)
- **Per-label F1/precision/recall** table (not multi-class confusion)
- **ROC curve** → per-label AUC (one curve per target)
- **Accuracy overview bar** → shows exact-match accuracy + hamming loss side-by-side
- **Multi-label distribution plot** → combination frequency barchart (how often each combination appears in train/test)
- Prediction display: probabilities shown as (KPC=0.82, NDM=0.11, VIM=0.97) format

---

## 10. Critical Multi-Label Algorithmic Considerations

### 10a. SupCon Contrastive Loss with Multi-Label Targets — Continuous Jaccard

The original SupCon loss identifies positive pairs by integer class equality: `pos_mask[i,j] = float(y[i] == y[j])` — a binary 0/1 matrix. For multi-label this is too coarse: `VIM` and `VIM_NDM` share a real biological signal yet form pure negative pairs.

**Decision: continuous Jaccard weight matrix — no threshold.**

Instead of binary, `pos_mask[i,j]` becomes a continuous float:

```
J(a, b) = |a ∩ b| / |a ∪ b|
         = dot(a, b) / (sum(a) + sum(b) - dot(a, b))
```

All values stay in `[0, 1]` with no threshold. This is a strict generalisation of binary SupCon: for disjoint single-target pairs `J=0` (pure negative), for identical pairs `J=1` (pure positive), for partial overlaps a fractional pull.

**Semantic geometry:** a `VIM_NDM` sample is pulled toward `VIM` with 50% force and toward `NDM` with 50% force (J = 1/3 each under VIM/NDM/VIM_NDM), placing its embedding between the two single-target clusters. Exactly what we want for multi-label representation.

**Why no threshold:** A threshold re-introduces a hard binary decision. The continuous form lets the loss gradient scale smoothly by overlap degree — logarithms in the Khosla equation handle the weighting naturally.

**Implementation:**

```python
def _jaccard_weight_matrix(y_binary):
    """Continuous pairwise Jaccard similarity for SupCon positive weighting.
    
    Returns (B, B) float matrix in [0, 1]. Self-pairs excluded (diagonal = 0).
    For identical labels → 1.0 (full pull).
    For partial overlap → fractional pull proportional to Jaccard.
    For disjoint labels → 0.0 (pure negative, no pull).
    """
    y = tf.cast(y_binary, tf.float32)           # (B, n_targets)
    intersection = tf.matmul(y, y, transpose_b=True)   # (B, B)
    sum_labels = tf.reduce_sum(y, axis=1)
    union = (tf.expand_dims(sum_labels, 1)
             + tf.expand_dims(sum_labels, 0)
             - intersection)                     # (B, B)
    jaccard = intersection / (union + 1e-8)      # (B, B), safe div
    # Zero out self-pairs
    not_self = 1.0 - tf.eye(tf.shape(y)[0])
    return jaccard * not_self                    # continuous pos_mask
```

**Loss modification inside `MultiLabelSupConSTModel.train_step`:**

```python
# Old (binary):
# pos_mask = tf.cast(same_label & not_self, tf.float32)  # 0 or 1
# mean_log_prob_pos = tf.reduce_sum(log_prob * pos_mask, axis=1) / tf.reduce_sum(pos_mask, axis=1)

# New (continuous Jaccard):
pos_mask = _jaccard_weight_matrix(y_binary)          # floats in [0, 1]
mean_log_prob_pos = (tf.reduce_sum(log_prob * pos_mask, axis=1)
                     / (tf.reduce_sum(pos_mask, axis=1) + 1e-8))
```

The rest of the Khosla equation (temperature scaling, log-sum-exp denominator) is unchanged.

**Applicability check:**
- ✅ Our embeddings are 96-dim floats (CNN+GRU dual backbone) — continuous weighting works
- ✅ Batch contains all 7 label combinations — full range of Jaccard values appear in each batch
- ✅ Single-target samples (`VIM`, `KPC`, `NDM`) have J=0 with disjoint targets → pure negatives preserved
- ✅ `VIM_NDM_KPC` has J>0 with all other combinations — smoothly pulled to centre of space
- ✅ No new hyperparameter introduced (threshold removed entirely)

**No `--supcon_jaccard_threshold` arg needed.** Remove it from `03_main_training.py` argparse.

**Implementation location:** `multiplex/utils/model_training/model_utils_multilabel.py` — the `MultiLabelSupCon*` subclasses only. Original `SupConSTModel` in `main/` is not touched.

### 10b. RCFD with Multi-Label

`gru_rcfd_cgd` has:
- Early encoder: GRU → scalar concentration prediction
- Dual backbone: CNN+GRU → 96-dim embedding
- FiLM conditioning: concentration modulates embedding
- Classification head: `Dense(n_classes, softmax)` → change to `Dense(n_targets, sigmoid)`
- Regression head: `Dense(1, linear)` → unchanged

In `evaluate_outlier_filters_ml`, for RCFD models:
- Pass `y_concentration` as usual (concentration regression target stays the same)
- Multi-label only affects the classification output/loss
- Combined loss: `cls_weight * binary_crossentropy(y_binary) + reg_weight * mse(y_conc)`

The `RCFDModel` and `RCFDSupConMTLModel` subclasses will need thin multi-label overrides in `model_utils_multilabel.py`:

```python
class MultiLabelRCFDModel(RCFDModel):
    """RCFD with sigmoid multi-label output instead of softmax."""
    # Override _build_cls_head to output sigmoid(n_targets)
    # Override train_step to use binary_crossentropy for classification loss
```

### 10c. Rare Combination Handling

Some combinations have very low counts (e.g., `VIM_NDM¸` = 1 sample after cleaning merges into `VIM_NDM` = 1535). The existing `rare_classes` filter (drops classes with < 2 samples) operates on `y_combo_int`. For multiplex, all cleaned combinations have > 1000 samples, so this is not a concern in practice. The rare-class logic is still kept for safety.

### 10d. Model Checkpoint / XAI

- `save_model_dir` in `evaluate_outlier_filters_ml` saves `.keras` files for XAI
- Multi-label models saved with multi-label head can be loaded normally for saliency analysis
- XAI (script 07) is **not** in the multiplex scope for this phase

---

## 11. File-by-File Implementation Checklist

### `multiplex/config.py`
- [ ] Import base config with `from config import *`
- [ ] Define `LAB_MULTIPLEX_FOLDER`, `FILE_MAPPING`, `FILE_CONC`, `FILE_TARGET`, `MULTIPLEX_SEPARATOR`
- [ ] Define `TRAINING_DATA_PATH`, `TRAINING_RESULT_PATH`, `TRAINING_10FOLD_RESULT_PATH` with `_ml` suffix
- [ ] Define `OUTLIER_FILTERS` (4 entries)
- [ ] Define `MULTIPLEX_MODELS` list (16 model keys)
- [ ] Add `MODEL_KEY_MAP` entries for multiplex (inherit from base + reuse existing keys)

### Pre-implementation: fix CSV
- [ ] Edit `LAB_Multiplex/01_ACA_qdPCR/dPCR_Dataset_Multiplex.csv`: change `VIM_NDM¸` → `VIM_NDM`

### `multiplex/01b_lab_curve_preprocessing.py`
- [ ] Import curve-reading helpers from `main/01b` (via sys.path + direct import)
- [ ] Import `sigmoid_fitting`, `safe_io`, `pywt` from parent
- [ ] Implement `parse_multilabel(raw_labels, separator)` — no artifact cleaning (CSV already clean)
- [ ] Implement `binarize_labels(label_lists)` using `MultiLabelBinarizer`
- [ ] Use `multiplex/config.FILE_MAPPING/FILE_CONC/FILE_TARGET/MULTIPLEX_SEPARATOR`
- [ ] Output `curve_for_training_ml.joblib` with extra keys: `labels`, `all_targets`, `label_binarized`
- [ ] Same wavelet / ori_curve / ori_curve_avg preprocessing as `main/01b`
- [ ] Argparse defaults same as `main/01b`: `--task_id=0`, `--exp_folder=LAB_MULTIPLEX_FOLDER`

### `multiplex/02_outlier_detection_pipeline.py`
- [ ] Import LSTM-AE pipeline from `main/utils/02_outlier_detection/`
- [ ] Import spatial pipelines (call but expect graceful no-op for lab data)
- [ ] Load `curve_for_training_ml.joblib`
- [ ] Extract kinetic features using functions imported from `main/02`
- [ ] Run only `lstm_ae_glb` + spatial (4 filter slots, 3 non-None)
- [ ] Save updated `curve_for_training_ml.joblib` with outlier columns appended

### `multiplex/utils/model_training/model_utils_multilabel.py`
- [ ] `_jaccard_weight_matrix(y_binary)` — continuous pairwise Jaccard float matrix (no threshold)
- [ ] `encode_multilabel_for_training()` — returns `(y_binary, y_combo_int, encoder)`
- [ ] `adapt_for_multilabel(base_model, n_targets)` — replaces softmax head with sigmoid
- [ ] `MultiLabelSupConSTModel`, `MultiLabelSupConBranch2STModel`, `MultiLabelSupConBranch3STModel` — continuous Jaccard pos_mask + `binary_crossentropy` cls loss + sigmoid head
- [ ] `MultiLabelRCFDModel`, `MultiLabelRCFDSupConMTLModel`, `MultiLabelRCFDBranch2MTLModel`, `MultiLabelRCFDBranch3MTLModel` — RCFD variants with sigmoid cls head; continuous Jaccard for SupCon variants
- [ ] `multilabel_metrics(y_true, y_pred, y_proba, target_names)` — returns metric dict
- [ ] `evaluate_outlier_filters_ml(...)` — the main training loop (see Section 7d)
- [ ] `plot_ml_results_ml(...)` — adapted for multi-label metrics display

### `multiplex/03_main_training.py`
- [ ] Argparse with same flags and defaults as `main/03` (no `--supcon_jaccard_threshold`)
- [ ] `--supcon` and `--condreg` select model subsets (same logic as `main/03`)
- [ ] Load `curve_for_training_ml.joblib`
- [ ] Call `encode_multilabel_for_training()` to get `y_binary`, `y_combo_int`
- [ ] Call `evaluate_outlier_filters_ml()` for each curve type
- [ ] Checkpoint into `classification_performances_ml.joblib`

### `multiplex/06_model_prediction_report.py`
- [ ] Load `classification_performances_ml.joblib`
- [ ] Per-label ROC curves + AUC table
- [ ] Per-label confusion matrices (3 separate 2×2 matrices: KPC, NDM, VIM)
- [ ] Exact-match accuracy + hamming loss overview
- [ ] Combination frequency chart
- [ ] Reuse `build_tabbed_html` from `html_utils`

---

## 12. Open Questions / Risks

1. **Spatial outlier filters on lab data**: `spatial_knn_label_elbow` and `spatial_grid_label_elbow` require `row`/`col` coordinates from chip data. For flat-CSV lab data these don't exist. The pipelines produce no column; the training loop skips missing filter columns with a warning. Listed in `OUTLIER_FILTERS` for structural consistency.

2. **Continuous Jaccard for SupCon** (resolved): Uses `_jaccard_weight_matrix()` — no threshold, no CLI flag. `J=1.0` for identical combinations, `J=0.0` for disjoint, fractional for partial overlap. This is the decided approach.

3. **Multi-label sigmoid threshold**: Default 0.5 for sigmoid → binary prediction in `03`. **Per-class threshold optimization can be done post-training in a notebook** — load `classification_performances_ml.joblib`, iterate `y_probs_*` arrays (already cached), sweep thresholds per class, and pick the threshold that maximises per-class F1. No retraining required.

4. **Concentration regression in RCFD for multi-label**: All droplets have a scalar concentration regardless of how many targets are present. The regression head is unchanged; FiLM conditioning by concentration remains semantically correct.

5. **`iterstrat` dependency**: Not needed — 7 unique combinations with `StratifiedKFold` on `y_combo_int` is sufficient. Adopt `MultilabelStratifiedKFold` if a future dataset has 20+ sparse combinations.

6. **`plot_ml_results` compatibility**: The existing `plot_ml_results()` assumes 1D int `y_trues_`. Multi-label needs its own `plot_ml_results_ml()` in `model_utils_multilabel.py`. Do not reuse the original.

---

## 13. Dependency & Import Map

```
multiplex/config.py
    ← main/config.py (from config import *)

multiplex/01b_lab_curve_preprocessing.py
    ← main/sigmoid_fitting.py
    ← main/safe_io.py
    ← main/config.py  (shared constants)
    ← multiplex/config.py  (FILE_MAPPING, FILE_TARGET, etc.)
    NEW: sklearn.preprocessing.MultiLabelBinarizer, pywt, numpy, pandas

multiplex/02_outlier_detection_pipeline.py
    ← main/utils/02_outlier_detection/lstm_autoencoder_outlier.py
    ← main/utils/02_outlier_detection/spatial_consistency_outlier.py
    ← main/utils/pipeline_utils.py
    ← main/safe_io.py
    ← main/sigmoid_fitting.py
    ← multiplex/config.py

multiplex/utils/model_training/model_utils_multilabel.py
    ← main/utils/model_training/model_utils.py  (build_neighbor_curve_stack, set_global_determinism, CurveResampler, _remap_global_splits)
    ← main/utils/model_training/model_utils_supcon.py  (SupCon model subclasses + factory functions)
    ← main/utils/model_training/model_utils_rcfd.py  (RCFD model factories: gru_rcfd_cgd family, RCFDModel base)
    ← main/safe_io.py
    NEW: sklearn.metrics.f1_score, hamming_loss; sklearn.model_selection.StratifiedKFold

multiplex/03_main_training.py
    ← multiplex/config.py
    ← multiplex/utils/model_training/model_utils_multilabel.py
    ← main/utils/pipeline_utils.py
    ← main/safe_io.py
    ← main/utils/model_training/model_utils.py  (set_global_determinism)

multiplex/06_model_prediction_report.py
    ← main/utils/html_utils.py
    ← multiplex/config.py
    ← multiplex/utils/model_training/model_utils_multilabel.py  (multilabel_metrics)
```

---

## 14. Verification Plan

After implementation, verify end-to-end with the single `01_ACA_qdPCR` dataset:

0. **CSV fix** (one-time): confirm `dPCR_Dataset_Multiplex.csv` has no `VIM_NDM¸` rows

1. **01b**: `python multiplex/01b_lab_curve_preprocessing.py --task_id 0`
   - Check: `curve_for_training_ml.joblib` exists, has `labels`, `all_targets=['KPC','NDM','VIM']`, `label_binarized.shape == (26571, 3)`

2. **02**: `python multiplex/02_outlier_detection_pipeline.py --task_id 0`
   - Check: `kinetic_features` DataFrame in updated joblib has `lstm_ae_glb_ds1_label_elbow` column

3. **03** (smoke: 1 model, default `--n_splits 1`):
   ```bash
   python multiplex/03_main_training.py --task_id 0 --rerun_models cnn_gru_dual
   ```
   - Check: `classification_performances_ml.joblib` with `y_trues_` as list of 2D binary arrays

4. **03 SupCon variant**:
   ```bash
   python multiplex/03_main_training.py --task_id 0 --supcon 1 --rerun_models cnn_gru_dual_supcon
   ```
   - Check: results merged into same joblib; `y_preds_AC_cnn_gru_dual_supcon_` is 2D binary

5. **03 RCFD**:
   ```bash
   python multiplex/03_main_training.py --task_id 0 --condreg --rerun_models gru_rcfd_cgd
   ```
   - Check: RCFD model trained with regression head intact

6. **06**: `python multiplex/06_model_prediction_report.py`
   - Check: HTML report renders with per-label ROC curves and per-label confusion matrices

---

*End of brainstorming document. Implementation proceeds from this plan.*
