# Plan: Reduce Joblib Cache Redundancy/Space (01-04)

## Context
Two related questions prompted this investigation:
1. Can `01`-`08`'s separate `.joblib` cache files be merged into a single file per experiment to reduce redundancy/space, and can existing cached files be "reformatted" into a new structure without re-running the pipeline?
2. Separately, the sigmoid-fitted curve variants produced by `01`'s `run_all_fits()` (3 variants × {fitted_full, fitted_stretched} = 6 arrays, copied into `02`'s cache, each with its own `kinetic_features` computed) are **no longer used downstream** — confirmed: `03_main_training.py`'s `--curve_type` defaults to `["ori_curve", "ori_curve_avg"]` only (line 68), so under normal runs these 6 variants are computed (expensive per-pixel `curve_fit` in `01`), copied (storage in both `01` and `02`), and have `kinetic_features` extracted for them (more compute, more storage) — all for data nothing reads.

**Decision:** for `01`+`02` specifically, keep the two scripts separate but have them **share one joblib file** — `01` and `02` each patch their own keys into the same file in place, rather than `01` writing one file and `02` copying from it into a second file. Detailed below.

Investigated via two Explore agents (read `01_curve_preprocessing_v6.py`, `02_outlier_detection_pipeline.py`, `03_main_training.py`, `04_cross_dataset_training.py`, `utils/model_training/model_utils.py`, and grepped `05`-`08` for `joblib.load(`) plus my own follow-up verification reads.

---

## Answer: should 01-08 share ONE joblib? No.

`05`-`08` are confirmed pure **readers** — they load `02`'s/`03`'s/`04`'s output and write only HTML/PNG reports, no new joblib. So "one joblib for 01-08" really means "merge 01→02→03→04's outputs into one file per experiment." Not recommended:
- **No partial/lazy loading**: `joblib.load()` reads the whole file into memory. `05` only needs `02`'s curve+feature data; merging would force it to load `03`'s/`04`'s classification results too (and vice versa) — slower, higher peak memory, for every reader.
- **Loses per-stage `--force_rerun` granularity**: today, deleting just `classification_performances_nonorm.joblib` re-runs `03` without touching `01`/`02`'s expensive cached preprocessing. One mega-file would need key-level checkpointing to preserve this — real added complexity for no benefit.
- **Reintroduces race conditions**: a previous session's plan (LOFO array-job parallelism) deliberately split results into per-combo files specifically to avoid concurrent writers clobbering a shared file. Merging back into one shared file per experiment would reintroduce that exact risk for any parallel stage.
- **`.keras` model files can't join anyway** — they're a different format read directly by `tf.keras.models.load_model()`; pickling them into joblib is fragile across TF versions and loses that direct-load path.

**Recommendation:** keep the existing per-stage file boundaries between `02`→`03`→`04`, and instead eliminate the *specific* duplication identified below. The one exception is `01`↔`02`, addressed next — they're a special case because `02` *already* copies `01`'s output wholesale rather than transforming it into something new.

---

## `01` + `02`: one shared joblib, scripts stay separate

**The mechanism that makes this work cleanly:** `02_outlier_detection_pipeline.py` already has a load-modify-save "unified state" pattern for *its own* cache (`_load_or_build_state()`, `_flush_and_save()`, both keyed on `unified_save_path = config.TRAINING_DATA_PATH`). Today that function does two things conditionally:
```python
def _load_or_build_state(exp_path, unified_save_path, force_rerun):
    pipeline_state = {}
    if os.path.exists(unified_save_path) and not force_rerun:
        pipeline_state = joblib.load(unified_save_path)   # 02's OWN cache
    ...
    if "dataset" not in pipeline_state:
        curve_path = exp_path / config.PREPROCESSED_CURVES_PATH   # <-- separate file, separate read
        data = joblib.load(curve_path)
        dataset_name = ["ori_curves"]
        dataset = [data["curves"]["ori_curves"]]
        ...
        pipeline_state.update({
            "dataset_name": np.array(dataset_name),
            "dataset": np.array(dataset),     # <-- np.array() on a list of arrays COPIES the data
            "Y_well": data["well_labels"],
            "timestamps": data["timestamps"],
            "metadata_df": pd.DataFrame(data["metadata"]),
        })
```
Two changes turn this into a true single shared file:

**1. Point `01` at the same file path `02` already uses (`config.TRAINING_DATA_PATH`), and have it patch in place instead of overwrite.**
`01_curve_preprocessing_v6.py`'s save step changes from "write a fresh file at `config.PREPROCESSED_CURVES_PATH`" to:
```python
unified_save_path = os.path.join(save_exp_path, config.TRAINING_DATA_PATH)
existing_state = {}
if os.path.exists(unified_save_path):
    try:
        existing_state = joblib.load(unified_save_path)
    except Exception:
        existing_state = {}
existing_state.update(save_data)   # save_data = curves, sigmoid_curves, idxs, timestamps,
joblib.dump(existing_state, unified_save_path, compress=3)   # well_labels, metadata, baseline_value, window_size_*, margin
```
`existing_state.update(save_data)` overwrites/inserts only `01`'s own keys; any keys `02` already added (`dataset`, `kinetic_features`, `Y_well`, the outlier-filter label columns, etc.) are left untouched. No key-name collisions exist today between what `01` writes and what `02` writes (confirmed by inspection — `02` already renames `well_labels`→`Y_well` on its side; `metadata`→`metadata_df` is also done on `02`'s side today, but see below — that one shouldn't be persisted at all).

**2. Simplify `02`'s `_load_or_build_state()` — it no longer needs a second file read at all.** Once `01` writes directly into the shared file, by the time `02` runs, `pipeline_state` (freshly loaded from `unified_save_path`) **already contains** `curves`, `sigmoid_curves`, `well_labels`, `metadata`, `timestamps`. The `if "dataset" not in pipeline_state:` block just needs to stop doing `data = joblib.load(curve_path)` and instead read directly from the `pipeline_state` it already has in memory:
```python
if "dataset" not in pipeline_state:
    data = pipeline_state   # already loaded — no second file, no second read
    dataset_name = ["ori_curves"]
    dataset = [data["curves"]["ori_curves"]]   # same object, not a copy
    ...
    pipeline_state.update({
        "dataset_name": dataset_name,   # keep as plain list (see point 3)
        "dataset": dataset,             # keep as plain list — NOT np.array(dataset)
        "Y_well": data["well_labels"],
        # "timestamps" already present from 01 — no need to re-set it
        # metadata_df intentionally NOT added here either — see "Within that shared
        # file" below; derive it as a local variable wherever it's needed instead
    })
```

**3. The actual de-duplication trick: stop calling `np.array(dataset)`.** `dataset[0]` and `data["curves"]["ori_curves"]` are, after the change above, the *same ndarray object* (no file boundary in between forcing a copy). `np.array(dataset)` stacks a list of arrays into a new contiguous block — a real memory/disk copy. Keeping `dataset` as a plain Python list preserves object identity, and joblib's pickler (memo-based, like standard `pickle`) recognizes the repeated object by `id()` and serializes it only once. **This is what actually removes the duplication on disk** — merging into one file is necessary (pickle dedup only works within a single `dump()` call, never across two separate files) but not sufficient on its own; avoiding the copy is the other half.
- `03`/`05`/`06`/`07` already index `state["dataset"][idx]` — a Python list supports the same `[idx]` access as a 3D ndarray would, so **no changes needed in any reader.**

**Why `--force_rerun` granularity survives:** each script still only recomputes its own keys. `02 --force_rerun` re-derives `dataset`/`kinetic_features`/outlier-filter columns from whatever curves are *already sitting in the shared file* — it never re-runs `01`'s raw chip-data reconstruction. `01 --force_rerun` only overwrites its own keys (`curves`, `sigmoid_curves`, etc.) via the `existing_state.update(save_data)` pattern, leaving `02`'s previously-computed `kinetic_features`/filters in place (they may now be stale relative to updated curves — same as today, where re-running `01` doesn't auto-invalidate `02`'s cache either; the user still needs to separately `--force_rerun` `02` after changing `01`'s logic).

**Net effect:** `config.PREPROCESSED_CURVES_PATH` becomes unused — only `config.TRAINING_DATA_PATH` (`curve_for_training_nonorm.joblib`) exists per experiment going forward. One file instead of two, `ori_curves`/`ori_curves_avg` stored exactly once, `02`'s incremental caching fully preserved, zero changes required in `03`/`05`/`06`/`07`.

**Migration for existing two-file experiments (no re-run needed):** a one-time merge step — load both existing files, `file2_data.update({k: v for k, v in file1_data.items() if k not in file2_data})` (don't clobber anything `02` already derived), fix up `dataset`/`dataset_name` to reference `file2_data["curves"]["ori_curves"]`/`["ori_curves_avg"]` by object instead of the old separately-stored copies, save the merged dict to `config.TRAINING_DATA_PATH`, then delete the old `preprocessed_curves_nonorm.joblib`. Pure reorganization — no recomputation.

### Within that shared file: `well_labels`/`Y_well` and `metadata`/`metadata_df`

Checked both pairs specifically:

- **`well_labels` / `Y_well` — already non-redundant, no change needed.** `02`'s existing code does `"Y_well": data["well_labels"]` — a direct reference, not a copy. Once `01` and `02` write to the same file (above), `pipeline_state["Y_well"] is pipeline_state["well_labels"]` is `True`, and joblib's pickler memoizes repeated objects by identity — stored once on disk automatically. The *only* requirement is to keep this a plain rebind (`Y_well = well_labels`), never `well_labels.copy()`.

- **`metadata` / `metadata_df` — genuinely redundant, fixable.** Unlike `Y_well`, `"metadata_df": pd.DataFrame(data["metadata"])` is a real constructor call — `pd.DataFrame()` does not guarantee zero-copy from a dict of arrays, so this *does* allocate a second copy of the same values. Checked every downstream script (`03`, `04`, `05`, `06`, `07`, `model_for_xai.py`, `resampling_check.py`) for `metadata_df` — **zero references outside `02_outlier_detection_pipeline.py` itself.** It's used only internally, at 6 call sites within `02` (`build_kinetic_features()`, the spatial-filter `has_spatial_info` check, etc.) — a pure convenience derived from `metadata`, never read from the saved file by anyone else.
  - **Fix:** stop persisting `metadata_df` as a saved key entirely. Compute it as a local variable once per `02` run (`metadata_df = pd.DataFrame(pipeline_state["metadata"])`, right after loading/building `pipeline_state`) and pass it around in-memory for the rest of that run, same as today — just never include it in the dict passed to `joblib.dump()`. Construction from a ~7-column, N_pixels-row dict is microseconds; not worth persisting a duplicate to save that.
  - Net effect: one fewer top-level key on disk, and the metadata values are stored exactly once (under `metadata`), not twice.

---

## Change 1 (primary) — Make sigmoid-curve fitting opt-in, default off

**Expanded finding:** the derivative/cleaning chain that *feeds* sigmoid fitting is itself dead weight. Traced the exact dependency chain in `process_experiment_data()` (line 202):
```
ori_curves (kept — used downstream)
  → ori_curve_dydx = get_derivatives(...)                    # intermediate only
    → ori_dydx_avg = moving_average_vec(ori_curve_dydx, ...)  # intermediate only
      → cleaned_idx, cleaned_lowest_idx = zero-crossing/integral detection on ori_dydx_avg
        → cleaned_std, cleaned_lowest = apply_baseline_cleaning(ori_curves, idx, margin)
          → indices_dict = {"cleaned_idx": ..., "cleaned_lowest_idx": ...}
            → run_all_fits() sigmoid-fits cleaned_std/cleaned_lowest using indices_dict
```
Confirmed via grep that `ori_curve_dydx`/`ori_dydx_avg`/`cleaned_std`/`cleaned_lowest` have **zero consumers outside this chain** — not even the would-be visualization: `plot_interactive_sigmoid_grids` (the only other place these are referenced, in `utils/01_curve_preprocessing/curve_preprocessing_plots.py`) is called from a block that's **commented out** (lines 547-550 of `01_curve_preprocessing_v6.py`). And confirmed via grep on `02_outlier_detection_pipeline.py` (the only other consumer of this data) that it touches exactly `curves.ori_curves`, `curves.ori_curves_avg`, `sigmoid_curves`, `well_labels`, `timestamps`, `metadata` — nothing else.

**File:** `01_curve_preprocessing_v6.py`
- Add `parser.add_argument("--compute_sigmoid_fits", action="store_true", help="Compute the derivative/cleaning chain (ori_curve_dydx, ori_dydx_avg, cleaned_std, cleaned_lowest) and the 5-parameter sigmoid fits derived from it. Unused by 02-08 under default args; off by default to save compute and storage.")`.
- Guard `process_experiment_data()` (or its call site) so that when the flag is off, it skips `get_derivatives`/`moving_average_vec`(1st-deriv)/the cleaning-tasks loop entirely, returning `ori_curve_dydx=None, ori_dydx_avg=None, cleaned_std=None, cleaned_lowest=None` and `indices_dict={"cleaned_idx": None, "cleaned_lowest_idx": None}` — only `ori_curves`/`ori_curves_avg` (already computed independently) are produced.
- Guard the call site (line 535: `fitting_results = run_all_fits(processed_curves, indices_dict, X_time)`): skip when the flag is off (saves the `Parallel(n_jobs=-1)` per-pixel `curve_fit` cost across 3 variants), use `fitting_results = {}` instead.
- `window_size_1stder` (used only as the `moving_average_vec(ori_curve_dydx, window_size_1stder)` window for computing `ori_dydx_avg`) is part of this same chain — store it as `None` when the flag is off too.
- Keep `"sigmoid_curves": fitting_results` and `"curves": {...}` always present in `save_data` (values just `None`/`{}` when disabled) — `02` does `data["sigmoid_curves"].items()`, and an **empty dict iterates zero times safely**; a **missing key would `KeyError`**, so keys must always exist, only values become empty/None.

**File:** `02_outlier_detection_pipeline.py`
- No code change required beyond the shared-file change above — confirmed it never reads `ori_curve_dydx`/`ori_dydx_avg`/`cleaned_std`/`cleaned_lowest`, and the existing `sigmoid_curves` loop already degrades gracefully to zero iterations when empty.

**Migration for existing cached files (no re-run needed):**
Implemented as a standalone one-off utility, **`adhoc_strip_unused_curves.py`** (not a flag on `01`/`02` — this is a "run once per dataset" migration, not a recurring pipeline option, so it lives outside the numbered scripts to keep them free of one-shot maintenance code). For every experiment folder under `--exp_folder`, it loads the shared `curve_for_training_nonorm.joblib` (folding in a legacy separate `preprocessed_curves_nonorm.joblib` if `01`/`02` haven't been re-run since the shared-file merge), sets `sigmoid_curves = {}`, drops `ori_curve_dydx`/`ori_dydx_avg`/`cleaned_std`/`cleaned_lowest`/`window_size_1stder` and the 5 dead `well_*` fields (Change 2), and filters `dataset_name`/`dataset`/`kinetic_features`/`linear_feature_combinations`/`important_feature_combinations`/`importance_dfs` down to entries where `name in {"ori_curves", "ori_curves_avg"}` — a pure filter+resave, no recomputation. Idempotent (already-stripped experiments are skipped).
```bash
python adhoc_strip_unused_curves.py --exp_folder /path/to/POC_DDM_dataset_folder
```

---

## Change 2 (new, recommended) — Remove fields with zero consumers, unconditionally

Beyond the sigmoid-fit chain, grepping every field of `01`'s output against everything downstream shows these have **no consumer at all**, regardless of `--compute_sigmoid_fits` — safe to drop unconditionally:

| Field | Location | Why it's dead |
|---|---|---|
| `well_2d_bs_active` | `curves` | Only other reference is `light_pipeline/00`/`01`, which *reconstructs* it independently from raw experiment data via `reconstruct_data()` — never reads it from this joblib |
| `well_2d_nl_bs_active` | `curves` | No reference anywhere outside `01` |
| `well_temp_lin2d` | `curves` | No reference anywhere outside `01` |
| `well_2d_temp_npr` | `curves` | No reference anywhere outside `01` |
| `well_temp_mean_then_lin` | `curves` | No reference anywhere outside `01` |

**Recommendation:** drop only the above. Confirmed zero consumers anywhere, in any mode, so safe to remove unconditionally.

**Explicitly kept, NOT removed (retained for further analysis even though currently unread by anything downstream):**
- `idxs.idx_start`, `idxs.idx_settled`, `idxs.idx_end`, `idxs.idx_active`, `idxs.max_significant_index`
- `baseline_value` (top-level)
- `margin` (top-level)

These stay exactly as today, unconditionally — not gated by `--compute_sigmoid_fits` either, since they're independent of the sigmoid-fit chain (e.g. `idx_start`/`idx_settled`/`idx_end`/`idx_active` come from `well0.idx_*`, not from `process_experiment_data()`).

Everything else stays as in Change 1: `curves.ori_curves`, `curves.ori_curves_avg`, `sigmoid_curves` + the dydx/cleaned chain + `window_size_1stder` (all gated by `--compute_sigmoid_fits`), `timestamps`, `well_labels`, `metadata` (all sub-fields — confirmed used: `pixel_row_idx`/`pixel_col_idx` feed the spatial-consistency outlier filters in `02`, the rest become DataFrame columns intentionally excluded from ML features via `config.EXCLUDED_FEATURES` but still tracked), and `window_size_ori` (used by `01`'s own cache-freshness check).
- **Migration:** same `adhoc_strip_unused_curves.py` pass drops the 5 `well_*` fields above alongside the sigmoid-chain fields — pure key deletion, no recomputation, fully retroactive on existing cached files.

---

## Change 3 (secondary, safe) — Stop `03` re-saving `02`'s features_df

**Finding:** `03`'s XAI metadata file (`model_interpretation_{curve_type}.joblib`) currently re-saves a byte-identical copy of `kinetic_features`/`features_df` that's already in the shared curve-training joblib (~94 MB extra per dataset in one measured example).
**Verified safe:** read `07_attribution_vis_all.py` directly — it only reads `local_pkg["top_10_features"]` from this file; it gets `dataset`/`features_df` from `curve_for_training_nonorm.joblib` separately, never from `model_interpretation_*.joblib`. So removing the duplicate fields from `03`'s write is a no-op for `07`.

**Also checked `model_paths`** (the other field in this file): grepped every `.py` file for `model_paths` — it's written **only** by `model_for_xai.py`, which is not invoked by any current `.pbs`/`.sh` job script (only appears in `slurm_jobs/logs/_archive/`, i.e. old runs) — superseded by `07_attribution_vis_all.py`. And `07`'s actual model loader, `load_saved_models()`, never reads a `model_paths` key at all — it reconstructs `.keras` file paths itself from a deterministic naming convention (`{model_dir}/{name}_{filter_key}_{curve_type}_model.keras`) and checks `.exists()`. So `model_paths` isn't just duplicate — in the active pipeline it's **dead on both ends** (nothing live writes it from `03`/`04`, nothing reads it).

**File:** `03_main_training.py` (and the equivalent in `04_cross_dataset_training.py`, which writes the same shape of file)
- Stop writing `dataset`/`features_df` into `model_interpretation_{curve_type}.joblib`; keep only `top_10_features` (confirmed `03`/`04` never write `model_paths` themselves — only the unused `model_for_xai.py` does, so there's nothing to keep there beyond `top_10_features`).
- **Migration:** existing files can be reformatted in place — load, keep only `top_10_features`, drop everything else, re-save. Lossless since nothing in the active pipeline reads the other fields.

---

## Change 4 (optional) — Fold `model_interpretation_{curve_type}.joblib` into the results file entirely

Once Change 3 lands, this file holds **only** `{"top_10_features": {filter_key_str: [10 names], ...}}` — a few hundred bytes. At that point, is a separate file still needed? Checked the write site in `03_main_training.py` (line ~247-258):
```python
all_ml_results[clean_title]["Native"] = res_native
joblib.dump(all_ml_results, results_file_path, compress=3)   # classification_performances_*.joblib written here
...
xai_pkg = joblib.load(xai_joblib_path) if xai_joblib_path.exists() else {}   # then a SEPARATE file, right after
xai_pkg["top_10_features"][str(f)] = top_10_features
joblib.dump(xai_pkg, xai_joblib_path)
```
`top_10_features` is computed and available at exactly the point where `all_ml_results[clean_title]` (the per-dataset_name dict already being saved to `classification_performances_nonorm.joblib`) is in scope — and that dict is already keyed by the right granularity (`dataset_name` → `mode` → `filter_key`).

**Not strictly necessary as a separate file.** Recommendation: fold it in directly —
```python
all_ml_results[clean_title]["top_10_features"] = top_10_features   # one line, sibling to "Native"/"Reference"
joblib.dump(all_ml_results, results_file_path, compress=3)
# delete the entire "WRITE XAI METADATA JOBLIB" block — no separate file at all
```
Same pattern applies to `04_cross_dataset_training.py`: `lofo_results[fold_label]["top_10_features"] = top_10_features`, folded into `classification_performances_cross_dataset_lofo_{curve_type}.joblib`.

**Reader side:** `07_attribution_vis_all.py` swaps `joblib.load(model_interp_dir / f"model_interpretation_{curve_type}.joblib")` → `local_pkg["top_10_features"].get(str(filter_key))` for a read of the (already-needed-elsewhere) `classification_performances_*.joblib`, keyed the same way.

**Why "optional" rather than recommended outright:** unlike `01`/`02`, there's no real `--force_rerun` asymmetry here (`top_10_features` is cheap to recompute either way — a single `mutual_info_classif` call, not a multi-minute training run), so the usual "keep separate to protect an expensive cache" argument doesn't apply. The only reason to keep it separate is risk isolation — regenerating/debugging XAI feature-selection logic touches a file that doesn't also hold the (expensive-to-regenerate) classification results, so a bug in the XAI-write path can't corrupt training results. Given the file is now tiny either way, this is a low-stakes style choice — fold it in if you want one fewer file, or leave it split if you'd rather keep XAI metadata changes isolated from training-result writes.

---

## Change 5 (behavior change, not space-related) — Baseline `ori_curves_avg` to start at y=0

**File:** `01_curve_preprocessing_v6.py`, `process_experiment_data()`
- After `ori_curves_avg = moving_average_vec(ori_curves, window_size_ori)`, subtract each curve's own first value: `ori_curves_avg = ori_curves_avg - ori_curves_avg[:, 0:1]`.
- **Why the slice, not a scalar index:** `ori_curves_avg` is `(N_pixels, N_timepoints)` — `ori_curves_avg[0]` would be pixel 0's entire first row, broadcast-subtracted from *every* pixel (wrong). `ori_curves_avg[:, 0:1]` keeps shape `(N_pixels, 1)`, so each row is zeroed by its own start value via broadcasting.
- Pure data-correctness fix, no storage/compute impact — `ori_curves_avg` keeps the same shape/dtype, just shifted per-row.

---

## Explicitly deferred (not in this change, flagged for later if wanted)
- **`04`'s `xai_data_{curve_type}.joblib` per-fold snapshot** (~303 MB/fold: full `X_curves_test`/`features_df_test` instead of just indices). Cannot be cleanly reformatted retroactively — the original `test_idx`/`train_idx` aren't stored anywhere independent of the materialized arrays in *existing* files, so fixing this only helps *future* runs, not a reformat of what's already cached. Skipped for now.

---

## Verification
1. **01+02 shared file:** run `01` then `02` from scratch on a small/test experiment folder; confirm only `curve_for_training_nonorm.joblib` exists on disk (no `preprocessed_curves_nonorm.joblib`), and that it contains both `01`'s keys (`curves`, `sigmoid_curves`, `metadata`, ...) and `02`'s keys (`dataset`, `kinetic_features`, `Y_well`, ...). Confirm `03`/`05`/`06`/`07` run unchanged against this file. Confirm `02 --force_rerun` alone (without re-running `01`) still works and doesn't touch `01`'s keys. Confirm `dataset[0] is <the saved file's> curves["ori_curves"]` (same object) before serialization, and check the on-disk file size doesn't double-count `ori_curves`.
2. **Migration:** run `adhoc_strip_unused_curves.py --exp_folder <dataset>` against an experiment with existing separate `preprocessed_curves_nonorm.joblib` + `curve_for_training_nonorm.joblib` (or an already-merged shared file); confirm the result has all expected keys, `03`/`05`/`06`/`07` still run correctly against it, the legacy file is removed, and a second run prints `[SKIP] ... already stripped`.
3. **`well_labels`/`Y_well` + `metadata`/`metadata_df`:** confirm the saved file has no `metadata_df` key at all (only `metadata`); confirm `02` still works correctly when it derives `metadata_df` locally on each run (`build_kinetic_features()`, the spatial-filter check, etc. all still get a valid DataFrame). Confirm `Y_well` and `well_labels` are present and `Y_well is well_labels` in memory before saving.
4. **Change 1:** run `01` once with default args (no `--compute_sigmoid_fits`); confirm `sigmoid_curves` is `{}`, `ori_curve_dydx`/`ori_dydx_avg`/`cleaned_std`/`cleaned_lowest`/`window_size_1stder` are `None`, and it completes faster than a `--compute_sigmoid_fits` run. Confirm `dataset_name` only contains `ori_curves`/`ori_curves_avg`.
5. **Change 2:** after dropping the 5 zero-consumer `well_*` fields, confirm `02` still runs unchanged and `light_pipeline/00`/`01` still work (they reconstruct `well_2d_bs_active` independently). Confirm `idxs.idx_start`/`idx_settled`/`idx_end`/`idx_active`/`max_significant_index`, `baseline_value`, and `margin` are still present (intentionally kept).
6. **Change 3:** after stripping everything but `top_10_features` from one `model_interpretation_{curve_type}.joblib`, run `07_attribution_vis_all.py` and confirm identical output (it never read the other fields).
7. **Change 4 (if taken):** after folding `top_10_features` into `classification_performances_*.joblib`/the LOFO equivalent and removing the separate file write in `03`/`04`, confirm `07_attribution_vis_all.py`'s updated read site finds the same `top_10_features` values as before, and that no `model_interpretation_*.joblib` file is created on a fresh run.
8. **Change 5:** run `01` on a small experiment; confirm `curves["ori_curves_avg"][:, 0]` is all-zeros (each pixel's row, not just one), and that `curves["ori_curves"]` is unaffected (only `ori_curves_avg` is baselined).

---

## Appendix: Joblib File Reference (structure, producer, consumers)

All paths are relative to an experiment folder unless noted. Sizes are from measured examples during investigation — actual size varies with pixel/well count.

### 1. `curve_for_training_nonorm.joblib` — `config.TRAINING_DATA_PATH` (shared by `01` and `02`)

- **Produced by:** `01_curve_preprocessing_v6.py` (writes/patches `curves`, `sigmoid_curves`, `idxs`, `timestamps`, `well_labels`, `metadata`, `baseline_value`, `window_size_*`, `margin`) **and** `02_outlier_detection_pipeline.py` (writes/patches `dataset`, `dataset_name`, `Y_well`, `kinetic_features`, `linear_feature_combinations`, `important_feature_combinations`, `importance_dfs`) — same file, each script owns its own keys. `metadata_df` is *not* persisted — `02` derives it in-memory from `metadata` each run (see "Within that shared file" above).
- **Used by:** `03_main_training.py`, `04_cross_dataset_training.py` (via `load_curve_data`), `05_outlier_visualization_report.py`, `06_model_prediction_report.py`, `07_attribution_vis_all.py`, `light_pipeline/attribution_vis.py`, `model_for_xai.py`, `resampling_check.py`

**Structure — Before (today: two separate files)**

`preprocessed_curves_nonorm.joblib` (`01`'s output, ~42 MB–529 MB) + `curve_for_training_nonorm.joblib` (`02`'s output, which *copies* curves from the first file, ~218 MB–1 GB):
```python
# File A: preprocessed_curves_nonorm.joblib
{
    "curves": {
        "well_2d_bs_active": ndarray, "well_2d_nl_bs_active": ndarray,
        "well_temp_lin2d": ndarray, "well_2d_temp_npr": ndarray, "well_temp_mean_then_lin": ndarray,
        "ori_curves": ndarray (N_pixels, N_timepoints) float64,
        "ori_curves_avg": ndarray (N_pixels, N_timepoints) float64,
        "ori_curve_dydx": ndarray, "ori_dydx_avg": ndarray, "cleaned_std": ndarray, "cleaned_lowest": ndarray,
    },
    "sigmoid_curves": {"original": {...}, "cleaned_std": {...}, "cleaned_lowest": {...}},
    "idxs": {"idx_start", "idx_settled", "idx_end", "idx_active", "cleaned_idx", "cleaned_lowest_idx", "max_significant_index"},
    "timestamps": ndarray, "well_labels": ndarray,
    "metadata": {"pixel_row_idx", "pixel_col_idx", "temp_group_idx", ...},
    "baseline_value": float, "window_size_ori": int, "window_size_1stder": int, "margin": int,
}

# File B: curve_for_training_nonorm.joblib  (8 curve variants, copies A's arrays)
{
    "dataset_name": ndarray (8,) object,   # ori_curves, ori_curves_avg, + 6 sigmoid-fit variants
    "dataset":      ndarray (8, N_pixels, N_timepoints) float64,   # np.array() COPY of A's curves
    "Y_well": ndarray, "timestamps": ndarray, "metadata_df": pd.DataFrame,
    "kinetic_features": [ 8 DataFrames ],
    "linear_feature_combinations": [...], "important_feature_combinations": [...], "importance_dfs": [...],
}
```

**Structure — After (Change 1 + 2 + shared-file merge, default args)**

One file, `ori_curves`/`ori_curves_avg` stored once (referenced, not copied), unused fields gone/gated:
```python
{
    # --- owned by 01 ---
    "curves": {
        "ori_curves":     ndarray (N_pixels, N_timepoints) float64,   # always computed
        "ori_curves_avg": ndarray (N_pixels, N_timepoints) float64,   # always computed; each row baselined to y=0 (Change 5)
        "ori_curve_dydx": None, "ori_dydx_avg": None,                # Change 1 — gated by --compute_sigmoid_fits
        "cleaned_std": None, "cleaned_lowest": None,                  # Change 1
        # well_2d_bs_active, well_2d_nl_bs_active, well_temp_lin2d, well_2d_temp_npr,
        # well_temp_mean_then_lin  →  REMOVED (Change 2, unconditional)
    },
    "sigmoid_curves": {},   # Change 1
    "idxs": {
        "idx_start", "idx_settled", "idx_end", "idx_active", "max_significant_index",  # kept, for further analysis
        "cleaned_idx": None, "cleaned_lowest_idx": None,   # Change 1
    },
    "timestamps":  ndarray (N_timepoints,) float64,
    "well_labels": ndarray (N_pixels,) int,
    "metadata": {...},            # unchanged, all sub-fields kept
    "baseline_value": float,      # kept, for further analysis
    "window_size_ori": int,       # kept — 01's own cache-freshness check
    "window_size_1stder": None,   # Change 1
    "margin": int,                # kept, for further analysis

    # --- owned by 02 ---
    "dataset_name": ["ori_curves", "ori_curves_avg"],         # plain list, only 2 entries
    "dataset": [<same object as curves["ori_curves"]>,
                <same object as curves["ori_curves_avg"]>],   # plain list of references, NOT np.array() — dedupes on disk
    "Y_well":      <same object as well_labels>,  # same array, renamed key for reader compatibility — pickle dedupes it
    # metadata_df NOT persisted — 02 derives it in-memory from "metadata" each run, never written to disk
    "kinetic_features": [ 2 DataFrames ],         # only 2 now — no compute spent on the unused 6
    "linear_feature_combinations": [...],         # 2 entries
    "important_feature_combinations": [...],      # 2 entries
    "importance_dfs": [...],                      # 2 entries
}
```
With `--compute_sigmoid_fits` passed, the `None`/`{}` placeholders above are populated as before; the 5 unconditionally-removed `well_*` fields stay gone either way.

### 2. `classification_performances_nonorm.joblib` / `classification_performances_10fold_nonorm.joblib` — `config.TRAINING_RESULT_PATH` / `config.TRAINING_10FOLD_RESULT_PATH`

- **Produced by:** `03_main_training.py` → `evaluate_outlier_filters()` in `utils/model_training/model_utils.py`
- **Used by:** `06_model_prediction_report.py`, `08_statistical_comparison.py`
- **Size:** ~6.8 MB (1-fold) – ~24 MB (10-fold), all models/filters combined
- **Structure:**
```python
{
  dataset_name (str, e.g. "Ori Curves"): {
    mode_name (str, "Native" | "Reference"): {
      filter_key (None | str, e.g. 'lstm_ae_glb_ds1_label_elbow'): {
        "y_trues_":   [ndarray(test_size,) int64, ...],     # len = n_splits
        "mask_count": int,
        "y_preds_<MODEL>_": [ndarray(test_size,) int64, ...],
        "y_probs_<MODEL>_": [ndarray(test_size, n_classes) float64, ...],
        "classes_<MODEL>_": [ndarray(n_classes,) int64, ...],
        # <MODEL> repeated per key in MODEL_KEY_MAP: AC, AC_lstm, AC_gru, AC_rnn, AC_trans, AC_rf,
        #   AC_kNN, FFI, AC_cnn_lf, AC_lstm_lf, AC_gru_lf, AC_trans_lf, AC_cnn_gru_dual, AC_cnn_trans_dual
      }
    }
  }
}
```

### 3. `.keras` model files

- **Produced by:** `evaluate_outlier_filters()` in `utils/model_training/model_utils.py` (called from `03` and `04`) — only when `save_model_dir` is set, `fold_idx == 0`, and the model is in `_XAI_SAVE_NAME`
- **Path pattern:** `{save_model_dir}/{model_name}_{filter_key}_{curve_type}_model.keras`
  — `save_model_dir` = `{exp_path}/model_interpretation/` for `03`, or `cross_dataset_cv/{group}/model_interpretation/{fold_label}/` for `04`
- **Used by:** `07_attribution_vis_all.py` (`load_saved_models()`)
- **Size:** ~43 KB – 453 KB each (not duplicated — each is a unique trained model)

### 4. `model_interpretation_{curve_type}.joblib`

- **Produced by:** `03_main_training.py` (one per experiment) and `04_cross_dataset_training.py` (one per fold, under `cross_dataset_cv/{group}/model_interpretation/{fold_label}/`)
- **Used by:** `07_attribution_vis_all.py` — reads **only** `local_pkg["top_10_features"]`
- **Size:** ~94 MB before; shrinks to a few hundred bytes after Change 3; file removed entirely if Change 4 is also taken

**Structure — Before:**
```python
{
  "top_10_features": { filter_key_as_str: [10 feature names], ... },
  "dataset":      ndarray,        # duplicate of curve_for_training_nonorm.joblib's data
  "features_df":  pd.DataFrame,   # duplicate of curve_for_training_nonorm.joblib's data
  # "model_paths" may appear in files written long ago by the now-unused model_for_xai.py —
  # the active 03/04 scripts never write it, and 07 never reads it
}
```
**Structure — After (Change 3, active scripts only write this from now on):**
```python
{
  "top_10_features": { filter_key_as_str: [10 feature names], ... },
}
```
**After Change 4 (optional):** this file doesn't exist at all — `"top_10_features"` becomes a sibling key inside the per-`dataset_name` dict in `classification_performances_nonorm.joblib` (file 2) / per-`fold_label` dict in `classification_performances_cross_dataset_lofo_{curve_type}.joblib` (file 5) instead.

### 5. `classification_performances_cross_dataset_{mode}_{curve_type}.joblib` — `config.CROSS_DATASET_RESULT_PATH`

- **Produced by:** `04_cross_dataset_training.py` (e.g. `classification_performances_cross_dataset_lofo_ori_curve.joblib`)
- **Used by:** cross-dataset equivalents of `05`/`06`/`07`/`08` (same reader logic, pointed at the `cross_dataset_cv/{group}/` folder)
- **Size:** ~5.8 MB
- **Structure:**
```python
{
  fold_label (str, e.g. "lofo_<folder_name>"): {
    filter_key (None | str): {
      "y_trues_": [ndarray(test_size,)],      # len = 1 (LOFO = single train/test split per fold)
      "mask_count": int,
      "y_preds_<MODEL>_": [...], "y_probs_<MODEL>_": [...], "classes_<MODEL>_": [...],
      # models: cnn, gru, transformer, cnn_gru_dual, cnn_trans_dual
    }
  }
}
```

### 6. `classification_performances_cross_dataset_resampler_{curve_type}.joblib` — `config.CROSS_DATASET_RESAMPLER_PATH`

- **Produced by:** `04_cross_dataset_training.py` — result of `CurveResampler.fit()` (fits a common time grid across folders with different timestamp densities)
- **Used by:** re-loaded by `04` itself on subsequent runs to skip recomputation; `resampling_check.py` (sanity-check plots)
- **Structure:** a pickled `CurveResampler` object — holds `t_grid` (ndarray) + fit metadata.

### 7. `xai_data_{curve_type}.joblib` (per-fold snapshot)

- **Produced by:** `04_cross_dataset_training.py`, under `cross_dataset_cv/{group}/model_interpretation/{fold_label}/`
- **Used by:** `07_attribution_vis_all.py` (cross-dataset XAI path)
- **Size:** ~303 MB per fold (dominated by `X_curves_test` + `features_df_test` — see "Explicitly deferred" above)
- **Structure:**
```python
{
  "X_curves_test":    ndarray (test_size, n_timesteps) float32,
  "features_df_test": pd.DataFrame (test_size, n_features),
  "X_man_train":      ndarray (train_size, 10) float32,
  "X_man_test":       ndarray (test_size, 10) float32,
  "y_test":           ndarray (test_size,) int64,
  "timestamps":       ndarray (n_timesteps,) float64,
  "top_10_features":  [10 feature names],
  "group_name":       str,
  "fold_label":       str,
}
```

### Pipeline data-flow summary

```
01 ──┐
     ├──> curve_for_training_nonorm.joblib (shared file) ──> 03, 04, 05, 06, 07, model_for_xai.py, resampling_check.py
02 ──┘
03 ──> classification_performances_(10fold_)nonorm.joblib ──> 06, 08
03 ──> model_interpretation_{curve_type}.joblib ──┐
03 ──> *.keras model files ────────────────────────┼──> 07
04 ──> classification_performances_cross_dataset_{mode}_{curve_type}.joblib ──> 05/06/07/08 (cross-dataset variants)
04 ──> classification_performances_cross_dataset_resampler_{curve_type}.joblib ──> 04 (self, cache), resampling_check.py
04 ──> model_interpretation_{curve_type}.joblib (per fold) ──┐
04 ──> xai_data_{curve_type}.joblib (per fold) ──────────────┼──> 07 (cross-dataset XAI)
04 ──> *.keras model files (per fold) ────────────────────────┘
05, 06, 07, 08 ──> HTML/PNG reports only (no new joblib written)
```
