# Code Quality Audit — POC_DDM `gk_code/main/`

> Generated: 2026-07-11  
> Scope: all `.py` files under `gk_code/main/` and `gk_code/main/utils/`, plus `slurm_jobs/*.sh`  
> Priority legend: 🔴 correctness/silent-bug risk · 🟠 high maintenance burden · 🟡 moderate cleanup · 🟢 nice-to-have

---

## Table of Contents

1. [File inventory & line counts](#1-file-inventory--line-counts)
2. [Critical correctness issues](#2-critical-correctness-issues)
3. [Highest-leverage structural changes](#3-highest-leverage-structural-changes)
4. [Dead code & unused imports](#4-dead-code--unused-imports-by-file)
5. [Monolithic functions to decompose](#5-monolithic-functions-to-decompose)
6. [Factory / boilerplate reduction](#6-factory--boilerplate-reduction)
7. [Cross-file repeated patterns](#7-cross-file-repeated-patterns)
8. [config.py issues](#8-configpy-issues)
9. [Slurm script observations](#9-slurm-script-observations)
10. [Quick wins (one-liners)](#10-quick-wins-one-liners)
11. [Prioritised action list](#11-prioritised-action-list)

---

## 1. File Inventory & Line Counts

### `gk_code/main/` (pipeline scripts)

| File | Lines |
|---|---|
| `01_curve_preprocessing_v6.py` | 644 |
| `02_outlier_detection_pipeline.py` | 892 |
| `03_main_training.py` | 456 |
| `04_cross_dataset_training.py` | 443 |
| `05_outlier_visualization_report.py` | 171 |
| `06_model_prediction_report.py` | 823 |
| `07_attribution_vis_all.py` | 2,043 |
| `08_statistical_comparison.py` | 1,049 |
| `config.py` | 883 |

### `gk_code/main/utils/model_training/`

| File | Lines |
|---|---|
| `model_utils.py` | **2,064** |
| `model_utils_supcon.py` | 608 |
| `model_utils_mtl.py` | 568 |
| `model_utils_gated.py` | 553 |
| `xai_gated_utils.py` | 791 |

### `gk_code/main/utils/02_outlier_detection/`

| File | Lines |
|---|---|
| `knn_fingerprint_filter.py` | 569 |
| `sigmoid_fitting.py` | 920 |
| `spatial_consistency_outlier.py` | 291 |
| `outlier_utils.py` | 310 |
| `autoencoder_outlier.py` | 215 |
| `lstm_autoencoder_outlier.py` | 258 |
| `cnn_autoencoder_outlier.py` | 233 |
| `autoencoder_outlier_per_well.py` | 201 |
| `lstm_autoencoder_outlier_per_well.py` | 211 |
| `cnn_autoencoder_outlier_per_well.py` | 216 |
| `mean_std_outlier.py` | 106 |
| `msc_outlier.py` | 138 |
| `amf_outlier.py` | 66 |
| `chip_utilities.py` | 288 |

---

## 2. Critical Correctness Issues

### 🔴 `Lambda` layers break `attn_recon` model reload across processes

**Files:** `model_utils_mtl.py` lines 334–342, `model_utils.py` lines 317–325

Both `_build_cnn_gru_dual_attn_recon_embedding_mtl` and `create_cnn_gru_dual_attn_recon_model` use `tf.keras.layers.Lambda` for the attention-reconstruction step (query slicing, score computation, weighted reconstruction). `model_utils_gated.py` documents this bug explicitly at lines 90–95: *"Lambda's saved bytecode loses its closure's globals on deserialization, so `tf` was unbound and inference raised NameError."*

**Impact:** Loading any `attn_recon` or `attn_recon_supcon*` model from disk in a new Python process (e.g. inside `07_attribution_vis_all.py`) silently raises `NameError: name 'tf' is not defined` at inference time.

**Fix:** Replace the three Lambda layers in each file with small `@tf.keras.utils.register_keras_serializable` subclasses, exactly as done in `model_utils_gated.py` for `_SumPool1D` and `_OneMinus`. Since both files implement the identical logic, a single `_AttnReconEmbedding` registered Layer in `model_utils_mtl.py` can be imported and used in both.

---

### 🔴 Mutable default arguments in all six autoencoder pipeline files

**Files:** `autoencoder_outlier.py:30`, `cnn_autoencoder_outlier.py:37`, `lstm_autoencoder_outlier.py:59`, and their three `_per_well` counterparts

```python
def run_autoencoder_pipeline(..., threshold_percentiles=["elbow", 90, 95]):
```

Python creates this list **once** at function-definition time. Any call-site that mutates the list (`.append`, `.remove`) propagates the mutation into all subsequent calls that omit the argument.

**Fix:** `threshold_percentiles=None`, then `if threshold_percentiles is None: threshold_percentiles = ["elbow", 90, 95]` at the top of the function body.

---

### 🔴 Duplicate key in `_FEAT_GROUP` dict — silent overwrite

**File:** `07_attribution_vis_all.py` ~line 1618

`'distance_asymmetry_index': 'shape'` appears twice in the same dict literal. Python silently uses the last assignment. One entry is a dead key with no effect.

**Fix:** Remove the first occurrence.

---

### 🟠 `model_key_map` / `model_print_map` defined twice — new models must be registered in three places

**Files:** `model_utils.py:731–866` (local dicts inside `evaluate_outlier_filters`) and `config.py:266–457` (`MODEL_KEY_MAP`, `MODEL_PRINT_MAP`)

The local dicts are a near-complete copy of the config module's authoritative maps. The `_inc` expansion loop is also duplicated. The `_XAI_SAVE_NAME` dict at `model_utils.py:669–684` is a third partial mapping. Every new model must be registered in all three places — this session hit this bug exactly (supcon2/3 trained nothing because the local map was not updated).

**Fix:** Remove the local dicts from `evaluate_outlier_filters`. Import `config` at the top of `model_utils.py` (the deferred `import config` at line 1799 exists only to avoid a perceived circular import — see item in §8 for how to break the cycle properly). Reference `config.MODEL_KEY_MAP` and `config.MODEL_PRINT_MAP` directly, with the existing `_inc` expansion already handled by `config.py:396–399`.

---

### 🟠 `_SPATIAL_RECON_MODELS` and `X_train_curve` input dispatch are maintained independently

**File:** `model_utils.py:883–892` (tuple) and `model_utils.py:1078–1089` (dispatch)

Two separate enumerations must be kept in sync by hand. Adding a new spatial model and forgetting the second one silently assigns the wrong `X_train_curve`.

**Fix:** Derive the `X_train_curve` assignment from membership in `_SPATIAL_RECON_MODELS` directly, so one source of truth controls both the pre-loop SKIP check and the input assignment.

---

### 🟠 Misleading comment on `_build_cnn_gru_dual_branches_mtl`

**File:** `model_utils_mtl.py:287`

Docstring says *"Identical to model_utils._build_cnn_gru_dual_branches but self-contained."* This is false: the MTL version uses `Dense(64)` for GRU embedding vs `Dense(32)` in `model_utils.py`, and produces a 96-dim fused output vs 64-dim. If a developer relies on this comment for architectural assumptions, they will get it wrong.

**Fix:** Update the comment to state the intended dimensional difference. Longer term, merge into a single `_build_cnn_gru_dual_branches(input_tensor, gru_emb_dim=32)` factory and parametrise the dimensions.

---

### 🔴 `EXCLUDED_FOLDERS` has duplicate entries

**File:** `config.py:49`

```python
EXCLUDED_FOLDERS = ['.DS_Store', 'model_interpretation', 'model_interpretation_old',
                    'outlier_visualisation', 'outlier_visualisation_old', ...
                    'outlier_visualisation', 'outlier_visualisation_old', ...]  # duplicated
```

`'outlier_visualisation'` and `'outlier_visualisation_old'` each appear twice. Harmless for `in` checks but confusing and suggests the list was grown by concatenation without deduplication.

**Fix:** Use a `set` literal (or deduplicate the list). If order matters for display, use `list(dict.fromkeys([...]))`.

---

## 3. Highest-Leverage Structural Changes

### 🟠 A. Remove the local `model_key_map` from `evaluate_outlier_filters` (→ config.py)

See §2 above. This is the single highest-leverage change: removes ~135 lines of duplicated dict code, eliminates a class of "new model silently trains nothing" bugs, and reduces every future model addition from a 3-file edit to a 1-file edit. Requires solving the circular-import constraint (see §8).

---

### 🟠 B. Extract `evaluate_outlier_filters` into sub-functions

**File:** `model_utils.py:687–1791` — **~1,105 lines**, the largest single function in the codebase

The function mixes: outlier-filter iteration, cache management, class-rarity guards, spatial-neighbour stack construction, a 14-way `elif` model dispatch, per-fold training loops, XAI model saving, result storage, checkpointing, and a printed leaderboard. Suggested decomposition:

```
evaluate_outlier_filters(...)
  ├── _build_model(base_m, T, n_classes, ...) → (model, epochs)
  ├── _prepare_concentration(y_conc, train_idx, test_idx) → (raw, scaled, scaler, valid_mask)
  ├── _train_fold(model, X_tr, y_tr, X_val, y_val, epochs, callbacks, ...) → None
  └── _run_filter(filter_name, filter_fn, ...) → dict (result entry)
```

`_build_model` alone collapses the 14-way `elif` block into a clean dispatch table and can be unit-tested independently.

---

### 🟠 C. Extract HTML utilities into `utils/html_utils.py`

**Files:** `06_model_prediction_report.py:79–100` and `08_statistical_comparison.py:82–104`

`_fig_to_buf`, `_buf_to_img_html`, `_panel`, and `build_tabbed_html` are defined identically in both files. Any change to the HTML template must be applied twice. Create `gk_code/main/utils/html_utils.py` with these four functions and import from both scripts.

---

### 🟡 D. Superseded `_per_well` autoencoder files

**Files:** `autoencoder_outlier_per_well.py`, `cnn_autoencoder_outlier_per_well.py`, `lstm_autoencoder_outlier_per_well.py`

Each is a subset of its companion file with `per_well=True` hardcoded and a different result key prefix. The companion files gained a `per_well` keyword argument after these were written, but the per-well files were never removed. They are still importable from `02_outlier_detection_pipeline.py` via commented-out pipeline blocks. As long as those blocks remain commented out, these files are unreachable dead code.

**Fix:** Verify the result key prefix difference (`ae_well_*` vs `ae_pw_ds1_*`) — if no downstream code reads those keys, delete all three `_per_well` files and their duplicate `build_*autoencoder` functions.

---

## 4. Dead Code & Unused Imports (by file)

### `02_outlier_detection_pipeline.py`

- **Line 32** — `import chip_utilities as utils`: never called anywhere in the file
- **Lines 34–43** — Eight pipeline imports (`run_msc_pipeline`, `run_amf_pipeline`, `run_meanstd_pipeline`, `run_knnfilter_pipeline`, `run_autoencoder_pipeline`, `run_cnn_autoencoder_pipeline`, `run_autoencoder_per_well_pipeline`, `run_lstm_autoencoder_per_well_pipeline`, `run_cnn_autoencoder_per_well_pipeline`) — all only referenced inside fully commented-out blocks. Remove or lazy-import inside the `if` branch that would use them.
- **Lines 634–642** — `msc_configs`, `amf_configs`, `mean_std_configs` are defined but all consumers are commented out.
- **Lines 649–783** — Four fully commented-out outlier pipeline blocks (~135 lines). If permanently disabled, delete them along with their imports and config variables.

### `03_main_training.py`

- **Line 13** — `MTL_MODEL_KEYS` imported but only appears in a comment at line 326. Remove.
- **Line 14** — `ALL_SUPCON_KEYS` imported but never referenced. Remove.
- **Line 181** — `outlier_filters = [None]` with a commented-out multi-filter alternative above. Delete the dead comment.

### `04_cross_dataset_training.py`

- **Line 15** — `MTL_MODEL_KEYS`: same as 03. Remove.
- **Line 16** — `ALL_SUPCON_KEYS`: never referenced. Remove.
- **Lines 197–198** — Two commented-out `for curve_type in ...` alternatives. Remove.
- **Line 199** — `for curve_type in [args.curve_type[1], args.curve_type[2], args.curve_type[0]]` — hardcodes exactly-3-element assumption; crashes with `IndexError` if fewer are supplied. Use a more defensive ordering.
- **Line 354** — `enumerate(reversed(list(lofo_splits.items())))` has a commented-out non-reversed version above it. Document *why* reversed, or revert.
- **Line 99** — `ref_mapping = d["well_to_label"] if "well_to_label" in d else None` — key `"well_to_label"` never exists in the returned dict; `ref_mapping` is always `None` and only checked with `if ref_mapping is None`, making it a dead stub.

### `06_model_prediction_report.py`

- **Line 29** — `_MTL_REG_SENTINEL = -1.0` — duplicates `model_utils_mtl.REG_SENTINEL`. Will silently drift if that module changes. Replace with `from model_utils_mtl import REG_SENTINEL as _MTL_REG_SENTINEL`.
- **Lines 802–804** — `--outlier_filter` default list re-defines the same strings as `config.OUTLIER_FILTERS`. Use `config.OUTLIER_FILTERS` as default directly.

### `07_attribution_vis_all.py`

- **Line 1872** — `exp_paths` exclusion list `['.DS_Store', 'model_interpretation']` is hardcoded instead of using `config.EXCLUDED_FOLDERS`.
- **Lines 1956–1959** — Two separate loops (`for model_name in non_mtl_names`, `for model_name in mtl_model_names`) both call `plot_latent_saliency_heatmap`. Since the function dispatches on `is_type` internally, these can be unified into one loop.

### `model_utils.py`

- **Lines 942–947** — Commented-out curve normalisation block with a debug marker comment `# # THISSS ###`. Delete.
- **Lines 731–866** — Local `model_key_map`/`model_print_map` (see §3A).

### `model_utils_mtl.py`

- **Lines 43–47, 60–63** — Commented-out Kendall (2018) uncertainty weighting code interleaved with live `__init__` and `_compute_loss` code. If it's documentation intent, move it to a docstring. Leaving it as commented-out `self.add_weight(...)` calls creates the misleading impression the feature is nearly ready.

### `model_utils_supcon.py`

- **6 aliased `cosine_recon` factory functions** — `create_cnn_gru_dual_cosine_recon_supcon_model`, `create_cnn_gru_dual_cosine_recon_supcon2_model`, `create_cnn_gru_dual_cosine_recon_supcon3_model`, and their MTL variants are **byte-for-byte identical** to the corresponding non-recon factories. The `cosine_recon` distinction is only in the *input data* (a pre-processed curve passed from outside), not the model architecture. Remove the six aliased functions and use the canonical dual-supcon factories for cosine_recon experiments, adding a comment in `evaluate_outlier_filters` to clarify.

### `sigmoid_fitting.py`

- **Line 4** — `from scipy.interpolate import interp1d` — never called anywhere in the file. Remove.

### `config.py`

- `CURVE_SPLIT` (lines 545–550) — config.py's own comment says "not currently referenced outside config.py"; confirmed unused externally.
- `IMPORTANCE_METRICS` (lines 552–558) — same.
- `MODEL_COLORS` (lines 207–216) — defined for use as a `get_palette()` argument but no external call site passes it.
- `CURVE_COLORS` (lines 220–227) — same.
- `PLOT_DOWNSAMPLE_STEP` (line 94) / `PLOT_DECIMAL_PRECISION` (line 95) — only referenced inside a **commented-out** call in `01_curve_preprocessing_v6.py:642`.
- Line 180 of `03_main_training.py` — `# outlier_filters = config.OUTLIER_FILTERS` commented out, leaving the live code reading from joblib instead. Misleading comment; delete.

---

## 5. Monolithic Functions to Decompose

### `evaluate_outlier_filters` — `model_utils.py:687` (~1,105 lines)

Already described in §3B. Priority: **high**.

### `extract_xai_artifacts` — `07_attribution_vis_all.py:304` (~367 lines)

The single longest function in `07`. Contains a 10-branch `if/elif` dispatch across model variants, each executing a near-identical `GradientTape` saliency loop. The type-detection logic (lines 331–341) is also duplicated verbatim in `plot_gradcam_per_label` (lines 954–964).

**Fix:**
1. Extract `detect_model_type(model_name) -> ModelTypeFlags` (a dataclass or namedtuple) — used by both functions.
2. Extract `_unpack_cls_output(outputs, flags) -> (cls_out, reg_out_or_None)`.
3. Each branch reduces to `cls_out, reg_out = _unpack_cls_output(model(x), flags)` followed by shared saliency code.

### `plot_gradcam_per_label` — `07_attribution_vis_all.py:940` (~152 lines)

Contains a duplicated type-detection block (lines 954–964 mirror lines 331–341 in `extract_xai_artifacts`). Resolved by the `detect_model_type` helper above.

### `plot_latent_feature_mapping` — `07_attribution_vis_all.py:1252` (~253 lines)

### `render_latent_feature_mapping_figure` — `07_attribution_vis_all.py:1524` (~280 lines)

Together ~533 lines for one visualisation type. `render_latent_feature_mapping_figure` is a 280-line grid-layout function that could be split into `_draw_left_panel(...)` and `_draw_right_panel(...)` row-drawing helpers.

### `process_experiment` — `06_model_prediction_report.py:560` (~231 lines)

Loads two joblib files, reconstructs splits, iterates every model to collect fold predictions, then builds six report tabs. Should be split into at least: `_load_experiment_artifacts()`, `_collect_model_results()`, `_build_html_tabs()`.

### `01_curve_preprocessing_v6.py` — NC-subtraction block

**Lines 522–599** (~77 lines) embedded inline in `__main__`. Mixes derivative computation, threshold detection, and per-vref/label subtraction. Extract to `apply_nc_subtraction(X_2d_bs_active, X_time, Y_well, vref_idx, config)`.

---

## 6. Factory / Boilerplate Reduction

### `model_utils_supcon.py` — Branch SupCon train_step duplication

`SupConBranch2STModel.train_step` (lines 305–317) and `SupConBranch3STModel.train_step` (lines 372–386) differ only in the number of `supcon_loss(...)` calls (2 vs 3). The `GradientTape → optimizer.apply_gradients → compiled_metrics.update_state → return dict` boilerplate is copied 8 times across the 4 model classes.

**Fix:** A base class `_SupConBaseModel(keras.Model)` with an abstract `_compute_sc_loss(self, outputs, y_cls) -> tf.Tensor` method. Each subclass implements only that method; the base provides `train_step` and `test_step`.

### `model_utils_supcon.py` — Branch SupCon factory boilerplate

The 8 (or 12 before removing cosine_recon aliases) factory functions differ only in which backbone builder and which wrapper they call. A dispatch table:

```python
_BSUPCON_SPEC = {
    'cnn_gru_dual_supcon2':     (_build_cnn_gru_dual_branches_mtl, _branch2_supcon_wrap),
    'cnn_trans_dual_supcon2':   (_build_cnn_trans_dual_branches_mtl, _branch2_supcon_wrap),
    ...
}
def create_branch_supcon_model(key, T, n_classes):
    backbone_fn, wrap_fn = _BSUPCON_SPEC[key]
    inp = Input(shape=(T, 1))
    cnn_emb, seq_emb, fused = backbone_fn(inp, return_branches=True)
    return wrap_fn(inp, cnn_emb, seq_emb, fused, n_classes)
```

Replaces ~120 lines of factory bodies with ~20.

### `model_utils_gated.py` — 8 near-identical gated fusion factories

Lines 347–509: all follow `Input → [inception] → _cnn_branch → _seq_branch → _fuse → _head → Model → compile`. The only differences are `seq_branch_fn`, `fuse_fn`, model name, and learning rate (1e-3 for GRU, 5e-4 for Trans).

**Fix:** A single `_make_gated_model(seq_branch_fn, fuse_fn, name, lr=1e-3)` collapses 160 lines to ~20. The `_ALL_FACTORIES` dict already catalogues them; the named functions are the redundancy.

### `model_utils_mtl.py` — Late-fusion MTL factory boilerplate

`create_cnn_lf_mtl_model`, `create_gru_lf_mtl_model`, `create_transformer_lf_mtl_model`, `create_lstm_lf_mtl_model` (lines 361–439) all share the same structure: curve branch + features `Dense(32)` + `Concatenate` + `Dense(64)` + `Dropout` + `_mtl_wrap`. Only the curve branch changes.

**Fix:** A single `create_lf_mtl_model(curve_backbone_fn, T, n_features, n_classes)` factory called with the appropriate backbone function.

### `xai_gated_utils.py` — `plot_branch_pca` and `plot_branch_tsne`

Lines 241–285 and 288–330: structural clones that differ only in using `PCA` vs `TSNE`. Extract to:
```python
def _plot_branch_2d(reducer, branch_embs, y, class_names, title_suffix, ...):
plot_branch_pca  = partial(_plot_branch_2d, PCA(n_components=2), ...)
plot_branch_tsne = partial(_plot_branch_2d, TSNE(...), ...)
```

---

## 7. Cross-File Repeated Patterns

### 🟠 A. `get_exp_paths` — folder enumeration repeated 9 times

The pattern:
```python
sorted([Path(args.exp_folder, name) for name in os.listdir(args.exp_folder)
        if os.path.isdir(...) and name not in config.EXCLUDED_FOLDERS])
```
appears in 01, 02, 03, 04, 05, 06, 07, 08, and `lab_aligned_full_pipeline.sh` (inline Python). `get_exp_paths` is defined in 03 and 06 but not shared. Move it to `config.py` (or a new `utils/pipeline_utils.py`) and import everywhere.

### 🟠 B. `task_id` bounds check — 9 scripts

```python
if args.task_id >= len(exp_paths):
    print(f"Task ID {args.task_id} is out of bounds ...")
    sys.exit(0)
```
Identical 3-line block in 01, 02, 03, 04, 05, 06, 07, 08. Extract to `check_task_id(task_id, paths)` in `utils/pipeline_utils.py`.

### 🟠 C. `force_rerun` key-clearing block — 03 and 04

~70-line blocks at `03:188–262` and `04:281–350` are near-identical. Extract to `clear_results_for_rerun(all_results, args)` in a shared training utility.

### 🟠 D. Model selection `if/elif` chain for `--supcon`/`--mtl` — 03 and 04

`03:318–346` and `04:231–265` contain the same 5-branch chain selecting model key lists. Extract to `select_models(args) -> List[str]` importable by both.

### 🟡 E. `print_banner` — 8 scripts

```python
print(f"\n{'='*70}\n[RUNNING] {os.path.basename(__file__)}\n{'='*70}\n")
```
One-liner repeated in all 8 pipeline scripts. A `print_banner(__file__)` function in `utils/pipeline_utils.py` would also enforce consistency (`06b`/`07b` currently omit it).

### 🟡 F. `argparse` defaults repeated across 5–9 scripts

| Argument | Count | Issue |
|---|---|---|
| `--task_id` | 9 | identical |
| `--exp_folder` | 9 | identical default |
| `--force_rerun` | 9 | identical |
| `--curve_type` | 8 | default list inconsistent: `06b`/`07b` omit `ori_curve_wavelet_sym8` |
| `--n_splits` | 3 | identical |
| `--fast_mode` | 3 | identical |

A `add_common_args(parser)` helper in `utils/pipeline_utils.py` would centralise defaults and prevent `--curve_type` drift between scripts.

### 🟡 G. `saved_viz` / `save_plot_flag` pattern — 01 and 02

```python
saved_viz = getattr(config, "SAVED_VIZ", [])
save_plot_flag = bool(saved_viz) and any(s in str(exp_path) for s in saved_viz)
```
Identical in both files. Extract to `config.should_save_plots(exp_path) -> bool`.

### 🟡 H. HTML utilities duplicated — 06 and 08

`_fig_to_buf`, `_buf_to_img_html`, `_panel`, `build_tabbed_html` — see §3C.

### 🟡 I. Label mapping application — 03, 04, 06, 07

`config.get_label_mappings(exp_path)` followed by the same remapping boilerplate appears in four scripts. Extract to `apply_label_mapping(Y_well, exp_path) -> Y_remapped` in `utils/pipeline_utils.py`.

### 🟡 J. `model.compile(...)` boilerplate in `model_utils.py`

`model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0), metrics=['accuracy'])` appears at lines 1279, 1355, 1424, 1465, 1500, 1555, 1590. Extract to a one-liner `_compile_model(model)`.

### 🟡 K. `T = X_train_curve.shape[1]` inside every `elif` branch

Seven branches each independently assign `T`. Compute it once before the dispatch.

---

## 8. `config.py` Issues

### 🟠 Side effects at import time

`config.py` executes three side effects when imported:
1. `set_global_determinism(0)` at line 10 — seeds NumPy/TF/Python globally
2. `plt.rcParams['axes.prop_cycle'] = ...` at line 183 — mutates global matplotlib state
3. `FILTER_COLORS = get_palette(...)` at line 253 — runs a function call

Any script that `import config` only for path constants still gets all three side effects. This is especially surprising for scripts that do their own seeding or that run in contexts where matplotlib's global state should not be touched (e.g. headless HPC workers).

**Fix:** Guard (1) with an environment variable `POC_DDM_DETERMINISM=1` or move the call to an explicit `config.init()` function called only from `__main__` entry points. Guard (2) similarly or only apply inside plotting functions. This is low-urgency but becomes a problem if the codebase is ever unit-tested or imported selectively.

### 🟠 Circular import constraint undocumented

`model_utils.py:1799` says `# deferred import to avoid circular` before `import config`. The risk is that any future addition to `config.py` that imports from `model_utils.py` would create a cycle. This constraint should be documented at the **top** of `model_utils.py`, not buried 1800 lines in.

The root cause: `config.py` builds `MODEL_KEY_MAP` / `MODEL_PRINT_MAP` which are also reproduced locally in `evaluate_outlier_filters`. If the local copies were removed (§3A), `model_utils.py` would need `import config` at module level. To make this safe, extract model registry maps into a new `model_registry.py` that neither `config.py` nor `model_utils.py` depends on at module level.

### 🟡 Mixed concerns in one file

`config.py` contains: root filesystem paths (infra), matplotlib style (presentation), MODEL_KEY_MAP with 60+ entries (training infrastructure), LABEL_MAPPINGS with 8 per-experiment dicts (domain data), `XAI_KINETIC_FEATURE_GROUP` (scientific annotation, 148 lines alone), and `AE_DOWNSAMPLE_FACTOR = 1` (a single training hyperparameter). For a file that every script imports, this breadth makes it hard to scan.

**Suggested split (optional, long-term):**
- `config.py` → paths, folder names, SLURM-relevant constants only
- `model_registry.py` → MODEL_KEY_MAP, MODEL_PRINT_MAP, model key lists
- `xai_config.py` → XAI_KINETIC_FEATURE_GROUP, FEATURE_GROUPS, IMPORTANCE_METRICS
- `label_mappings.py` → LABEL_MAPPINGS, get_label_mappings()

---

## 9. Slurm Script Observations

### Task-ID arithmetic is fragile and undocumented

Several scripts perform ad-hoc arithmetic to convert `$SLURM_ARRAY_TASK_ID` into the actual dataset index:

- `multi_full_pipeline.sh`: `REAL_TASK_ID=$((8-SLURM_ARRAY_TASK_ID))` — the reversal is unexplained
- `lab_supcon_training.sh`: `PREP_TASK_ID=$((SLURM_ARRAY_TASK_ID-3))` for tasks 9–12 — no comment why
- `new_xai_test.sh`: `TASK_ID=3; REAL_TASK_ID=$((8-TASK_ID))` — hardcoded constant, effectively `REAL_TASK_ID=5`
- `lab_1to1_attribution_vis.sh`: discovers task IDs with an inline Python snippet that uses `config.EXCLUDED_FOLDERS` inconsistently (line `and n not in ['.DS_Store', 'model_interpretation']` vs the config list)

**Recommendation:** Add a comment next to every non-trivial `REAL_TASK_ID` arithmetic explaining what index space it maps to and why the offset exists. The inline Python used in `lab_aligned_full_pipeline.sh` for finding `ALIGNED_TID` is a good pattern — it reads `config.EXCLUDED_FOLDERS` consistently and is self-documenting.

### Heavy use of inline Python for task discovery

Six scripts spawn `python3 -c "..."` one-liners to compute dataset task IDs. This works but the code is invisible to syntax highlighters and linters, and any import error inside those strings produces cryptic output.

**Recommendation:** Add a small `gk_code/main/slurm_helpers.py` CLI (`python slurm_helpers.py discover --strategy 1_area --exp_folder $EXP_FOLDER`) that prints the IDs, replacing inline Python snippets with a readable, testable function call.

### Commented-out alternatives accumulate

`multi_full_pipeline.sh` is ~30% commented-out variants (the `nc_subtract=0/1` loop block). `pipeline_manual.sh` similarly has most useful logic commented out. These represent the natural history of exploration but obscure what the script currently does.

**Recommendation:** Move superseded variants to an `_archive/` sub-folder or a `# ARCHIVED` block at the bottom, keeping the active code visually clean.

### Missing `set -e` in most scripts

Only `lab_supcon_training.sh` and `lab_mtl_training.sh` use `set -e` (exit on first error). All other scripts silently continue if a Python step fails. A failed `03_main_training.py` will not prevent `06_model_prediction_report.py` from running with stale data.

**Recommendation:** Add `set -e` to all production pipeline scripts. For scripts where partial completion is acceptable (e.g. `lab_1to1_full_pipeline.sh` looping over combos), add per-step error checking or capture the exit code explicitly.

### `PYTHONPATH` is set inconsistently

Most scripts export:
```bash
export PYTHONPATH="/vol/bitbucket/gk225/POC_DDM:/vol/bitbucket/gk225/POC_DDM/gk_code:$PYTHONPATH"
```
`new_xai_test.sh` omits this and relies on `cd gk_code/main` being sufficient for imports (which works only because `model_utils.py` manipulates `sys.path` internally). Standardise to the explicit export.

### `lab_1to1_attribution_vis.sh` uses a stale exclusion list

Line: `and n not in ['.DS_Store', 'model_interpretation']` — this inline list is not `config.EXCLUDED_FOLDERS`. New exclusions added to config are missed here.

### `multi_04_lofo_crossval.sh` missing `set -e` and MTL runs

No `set -e` (silent failure on any step). No `--mtl` variants — may be intentional for cross-dataset ST-only comparison but worth a comment to make intent explicit.

---

## 10. Quick Wins (one-liners or 2-line fixes)

| # | File | Location | Fix |
|---|---|---|---|
| 1 | `sigmoid_fitting.py` | line 4 | remove `from scipy.interpolate import interp1d` |
| 2 | `03_main_training.py` | line 14 | remove `ALL_SUPCON_KEYS` import |
| 3 | `04_cross_dataset_training.py` | line 16 | remove `ALL_SUPCON_KEYS` import |
| 4 | `03_main_training.py` | line 13 | remove `MTL_MODEL_KEYS` import |
| 5 | `04_cross_dataset_training.py` | line 15 | remove `MTL_MODEL_KEYS` import |
| 6 | `06_model_prediction_report.py` | line 29 | `from model_utils_mtl import REG_SENTINEL as _MTL_REG_SENTINEL` |
| 7 | `07_attribution_vis_all.py` | line 1872 | replace inline exclusion list with `config.EXCLUDED_FOLDERS` |
| 8 | `config.py` | line 49 | deduplicate `EXCLUDED_FOLDERS` list |
| 9 | `07_attribution_vis_all.py` | line 1618 | remove duplicate `distance_asymmetry_index` dict key |
| 10 | `model_utils.py` | lines 942–947 | delete commented-out curve normalisation block |
| 11 | All 6 AE files | `threshold_percentiles=` | fix mutable default argument |
| 12 | `02_outlier_detection_pipeline.py` | line 32 | remove `import chip_utilities as utils` |
| 13 | `06_model_prediction_report.py` | lines 802–804 | use `config.OUTLIER_FILTERS` as default |

---

## 11. Prioritised Action List

### Tier 1 — Fix now (correctness / silent bugs)

1. 🔴 Replace `Lambda` layers in `model_utils_mtl.py` and `model_utils.py` attn_recon models with registered `Layer` subclasses (model reload broken across processes)
2. 🔴 Fix mutable default `threshold_percentiles=[...]` in all 6 AE pipeline files
3. 🔴 Remove duplicate dict key `distance_asymmetry_index` in `07_attribution_vis_all.py`
4. 🔴 Deduplicate `EXCLUDED_FOLDERS` in `config.py`
5. 🔴 Fix `04_cross_dataset_training.py:199` — defensive curve_type indexing

### Tier 2 — High maintenance relief

6. 🟠 Remove local `model_key_map`/`model_print_map` from `evaluate_outlier_filters`; use `config.MODEL_KEY_MAP`/`MODEL_PRINT_MAP` directly (eliminates the "new model trains nothing" bug class)
7. 🟠 Extract shared `utils/pipeline_utils.py`: `get_exp_paths`, `check_task_id`, `add_common_args`, `print_banner`, `apply_label_mapping`
8. 🟠 Extract `utils/html_utils.py`: `_fig_to_buf`, `_buf_to_img_html`, `_panel`, `build_tabbed_html` (shared by 06 and 08)
9. 🟠 Extract `clear_results_for_rerun` and `select_models` shared by 03 and 04
10. 🟠 Correct the misleading `_build_cnn_gru_dual_branches_mtl` docstring; parametrise the dimension difference

### Tier 3 — Structural cleanup

11. 🟡 Remove the 6 `cosine_recon_supcon*` factory function aliases in `model_utils_supcon.py`
12. 🟡 Decompose `evaluate_outlier_filters` into `_build_model`, `_prepare_concentration`, `_train_fold`, `_run_filter`
13. 🟡 Decompose `extract_xai_artifacts` + `plot_gradcam_per_label` with shared `detect_model_type()`
14. 🟡 Delete or verify the three `_per_well` autoencoder files
15. 🟡 Delete the 8 dead imports + 5 fully-commented-out pipeline blocks in `02_outlier_detection_pipeline.py`
16. 🟡 Quick-win imports and constants (items 1–13 in §10)

### Tier 4 — Long-term architecture

17. 🟢 Consolidate `model_key_map` into `model_registry.py` to break circular-import constraint
18. 🟢 Factory-table refactor for Branch SupCon, gated fusion, and LF-MTL factories
19. 🟢 Base class `_SupConBaseModel` to eliminate `train_step` boilerplate
20. 🟢 `slurm_helpers.py` CLI for dataset ID discovery
21. 🟢 Add `set -e` to all production slurm scripts
22. 🟢 Split `config.py` concerns into `model_registry.py`, `xai_config.py`, `label_mappings.py`
