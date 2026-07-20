# Architecture Brainstorm: Label Consolidation (LC) Variant

**Date:** 2026-07-14

---

## Concept

Current classification models predict the **well label only** (e.g. KPC, NDM, VIM — `n_labels` classes). The Label Consolidation variant fuses label and concentration into a single combined target:

```
class = f"{label}_{concentration}"   →  "KPC_1M", "KPC_100K", "NDM_10K", …
```

This expands the output from `n_labels` → `n_labels × n_concentrations` classes. The model architecture is **unchanged** — same CNN+GRU dual backbone, same softmax head, just more output neurons. No regression head.

**Flag:** `--lbl_conc` (short, consistent with `--mtl`, `--supcon` style).

### Why This Is Interesting

| | ST (current) | MTL | LC (new) |
|---|---|---|---|
| Predicts label? | ✓ | ✓ | ✓ (embedded in class) |
| Predicts concentration? | ✗ | ✓ (continuous regression) | ✓ (discrete, via class) |
| Extra head? | no | regression head | no |
| Output neurons | n_labels | n_labels + 1 | n_labels × n_conc |

LC treats concentration as a **categorical dimension**, not a continuous one. Comparison with MTL: MTL regresses concentration; LC classifies it discretely. The results tell us whether the signal is continuous-regression-amenable or better captured by hard categorical boundaries.

### Explicit non-combinations

- **LC + MTL**: NOT supported. Concentration is consumed during label encoding; adding a regression head would be redundant. A startup guard exits early if both flags are set.
- **LC + cross-dataset training**: separate consideration; not in scope here.

---

## Data Reality

From inspection of `curve_for_training.joblib`:

- `01_ACA_qdPCR`: 3 labels × ~3 concentrations ≈ **~9 classes**
- `02_AMCA_qdLAMP`: 5 labels × 7 concentrations = **35 classes** (small-class risk)

---

## Label Combination Strategy

### Combined label format

```python
def _lc_label(well_label, conc):
    if conc is None or np.isnan(float(conc)) or float(conc) <= 0:
        return f'{well_label}_0'      # NTC/negatives: keep, mark as 0
    c = float(conc)
    if c >= 1e6:  return f'{well_label}_{c/1e6:.3g}M'
    if c >= 1e3:  return f'{well_label}_{c/1e3:.3g}K'
    return f'{well_label}_{c:.3g}'
```

Keeping `_0` (rather than excluding) retains NTC/negative controls — their concentration=0 is semantically meaningful (no template).

### Encoding

`Y_well` at the encoding step is already post-`label_mapping` (applied at line 186 of `03_main_training.py`), so multiple raw well IDs that share the same label (e.g. `KPC_A1`, `KPC_A2` → `KPC`) are correctly unified before combination.

```python
# Y_well here is post-label_mapping — mapped class labels, not raw well IDs
combined = [_lc_label(y, c) for y, c in zip(Y_well, y_concentration)]
encoder  = LabelEncoder()
y_full   = encoder.fit_transform(combined)   # n_classes = n_labels × n_conc_levels
```

`n_classes` propagates automatically — `evaluate_outlier_filters` computes it from `len(np.unique(y_true))`.

---

## Model Keys — 12 Total (3 architectures × 4 SC variants)

| SC | Keys |
|----|------|
| SC0 (base) | `cnn_gru_dual_lc`, `cnn_gru_dual_cosine_recon_lc`, `cnn_gru_dual_attn_recon_lc` |
| SC1 (SupCon fused z) | `cnn_gru_dual_supcon_lc`, `cnn_gru_dual_cosine_recon_supcon_lc`, `cnn_gru_dual_attn_recon_supcon_lc` |
| SC2 (SupCon branches) | `cnn_gru_dual_supcon2_lc`, `cnn_gru_dual_cosine_recon_supcon2_lc`, `cnn_gru_dual_attn_recon_supcon2_lc` |
| SC3 (SupCon branches + fused) | `cnn_gru_dual_supcon3_lc`, `cnn_gru_dual_cosine_recon_supcon3_lc`, `cnn_gru_dual_attn_recon_supcon3_lc` |

### Why SupCon + LC is valuable

In standard SupCon the contrastive anchor is the well label (`KPC` groups together). In LC SupCon the anchor is the combined label (`KPC_1M`), which pulls together samples of the **same label AND same concentration** while separating `KPC_1M` from `KPC_100K`. This is strictly more discriminative — the embedding space learns both categorical identity and concentration ordering, without any explicit regression.

---

## Changes to `03_main_training.py`

### 1. New argument flag

```python
parser.add_argument('--lbl_conc', action='store_true',
    help='Consolidate label+concentration into a single classification target (pure ST)')
```

Guard against invalid combination:
```python
if getattr(args, 'lbl_conc', False) and args.mtl:
    sys.exit('[!] --lbl_conc is not compatible with --mtl.')
```

### 2. Extend concentration loading gate (lines 136–150)

```python
_need_conc = args.mtl or getattr(args, 'lbl_conc', False)
if _need_conc:
    raw_conc = training_data.get("concentration", None)
    ...
    y_concentration = ...
```

### 3. Label encoding block (after line 191)

```python
if getattr(args, 'lbl_conc', False):
    if y_concentration is None:
        print('[!] --lbl_conc requires concentration data in joblib.'); sys.exit(1)
    # Y_well is already post-label_mapping here
    combined = [_lc_label(y, c) for y, c in zip(Y_well, y_concentration)]
    encoder = LabelEncoder()
    y_full  = encoder.fit_transform(combined)
    print(f'[LC] {len(set(combined))} combined classes: {sorted(set(combined))}')
else:
    encoder = LabelEncoder()
    y_full  = encoder.fit_transform(Y_well)
```

### 4. Model selection block

```python
if getattr(args, 'lbl_conc', False):
    _sc_lc = {
        0: ['cnn_gru_dual_lc', 'cnn_gru_dual_cosine_recon_lc', 'cnn_gru_dual_attn_recon_lc'],
        1: ['cnn_gru_dual_supcon_lc', 'cnn_gru_dual_cosine_recon_supcon_lc', 'cnn_gru_dual_attn_recon_supcon_lc'],
        2: ['cnn_gru_dual_supcon2_lc', 'cnn_gru_dual_cosine_recon_supcon2_lc', 'cnn_gru_dual_attn_recon_supcon2_lc'],
        3: ['cnn_gru_dual_supcon3_lc', 'cnn_gru_dual_cosine_recon_supcon3_lc', 'cnn_gru_dual_attn_recon_supcon3_lc'],
    }
    models = _sc_lc[args.supcon]
```

### 5. Evaluation call

```python
evaluate_outlier_filters(
    ...,
    multitask=False,        # no regression head for LC
    y_concentration=None,   # concentration consumed by label encoding
    ...
)
```

### 6. `force_rerun` / banner

Add `_lc_result_keys` set (all 12 keys); clear on `args.lbl_conc`. Banner: `mode_str = f'LC SC{args.supcon}'`.

---

## Changes to `model_utils_supcon.py`

SC0 LC keys reuse the same ST factory functions as their non-LC counterparts — no new classes needed; `n_classes` is automatically larger.

SC1/SC2/SC3 LC keys need **pure-ST SupCon model classes** (existing `SupConMTLModel` etc. output `[cls_out, reg_out, proj]`; LC has no regression head):

```python
@tf.keras.utils.register_keras_serializable(package='supcon_st')
class SupConSTModel(tf.keras.Model):
    """ST SupCon: outputs [cls_out, proj]. train_step: CE + SupCon on y_cls."""
    def train_step(self, data):
        x, (y_cls, y_cls_proj) = data
        with tf.GradientTape() as tape:
            cls_out, proj = self(x, training=True)
            loss = CE(y_cls, cls_out) + supcon_loss(proj, y_cls_proj, SUPCON_TEMP)
        ...

class SupConBranch2STModel(tf.keras.Model):
    """ST Branch2 SupCon: [cls_out, cnn_proj, seq_proj]."""

class SupConBranch3STModel(tf.keras.Model):
    """ST Branch3 SupCon: [cls_out, cnn_proj, seq_proj, fused_proj]."""
```

In `model_utils.py` dispatch: SC1/SC2/SC3 LC blocks unpack `[cls_out, *_projs]` from `model.predict`, use only `cls_out` for accuracy.

---

## Changes Outside `03_main_training.py` (summary)

### `model_utils.py`
- SC0 dispatch: alias `_lc` keys to same factory functions as non-LC counterparts.
- SC1/SC2/SC3 dispatch: new blocks using `SupConSTModel` / `SupConBranch2STModel` / `SupConBranch3STModel`.
- Unpack `[cls_out, *_projs]` from predict; store only `cls_out` and `cls_prob`.
- Add `_lc` keys to `_XAI_SAVE_NAME`.

### `config.py`
- Add 12 entries to `MODEL_KEY_MAP` following `"key": ("y_preds_AC_key_", "y_probs_AC_key_", "classes_AC_key_")` pattern.
- Add `_LC_MODEL_KEYS` set to `_NO_INC`.
- Add to `MODEL_PRINT_MAP`: `"cnn_gru_dual_lc": "CNN+GRU LC"`, `"cnn_gru_dual_supcon_lc": "CNN+GRU LC SC1"`, etc.

---

## Key Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Small class sizes (35 classes for AMCA_qdLAMP) | Stratified CV handles it; existing guard at model_utils.py line 828 |
| ST SupCon `train_step` boilerplate | Mirror existing MTL train_step pattern, omit reg terms |
| `_get_reg_trues_cm` fallback in notebook | LC has no `y_reg_trues_*`; `all_conc = NaN` → labels show class-only, which is correct for LC |
| `n_classes` varies across datasets | Already the case for all models; no issue |

---

## Evaluation / Post-Analysis

Results stored as `y_preds_AC_cnn_gru_dual_lc_` etc. `classes_AC_*` contains combined strings like `['KPC_10K', 'KPC_100K', 'KPC_1M', 'NDM_10K', ...]`.

`plot_confusion_gru` naturally displays the combined labels in the true-axis. To recover per-label accuracy, group rows by label prefix (text before `_`).

Comparison to run:
- SC0 LC vs SC0 ST: does knowing concentration help classification?
- SC1 LC vs SC1 ST: does concentration-aware contrastive pulling improve the embedding?
- LC vs MTL: categorical vs continuous concentration representation — which is more learnable?
