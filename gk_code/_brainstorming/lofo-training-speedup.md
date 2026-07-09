# Plan: Speed Up LOFO Training in `04_cross_dataset_training.py`

## Context
LOFO cross-validation on a 4-folder group currently trains **4 folds × 4 outlier filters × 5 models = 80 model trainings sequentially in one PBS job** (72h walltime requested). A single GRU fold (~55K train / ~16.5K test rows, 2 classes) already takes ~2h. This will only get worse as classes grow to 10 or multilabel. Investigation (read `utils/model_training/model_utils.py`, `04_cross_dataset_training.py`) found three concrete, independent causes — **not** training-set size (the model is tiny, ~14K params, and 55K rows/epoch is cheap):

1. `set_global_determinism()` sets `TF_CUDNN_DETERMINISTIC=1` + `tf.config.experimental.enable_op_determinism()`, which forces TensorFlow off cuDNN's fast (non-deterministic) RNN kernels and onto a slow fallback — likely the single biggest cost for GRU/LSTM/Transformer.
2. Every model trains a **fixed 500 epochs with no validation set and no EarlyStopping** (`model_utils.py` lines ~603, 629, 646-650) — guaranteed to run far past convergence every time.
3. All 80 trainings run **sequentially in one job** — no parallelism across folds/filters at all.

Decisions made with the user:
- Determinism must stay **bit-exact by default**, but should become a **runtime choice on `04` only** (other scripts unchanged) — so it can be turned off later if reproducibility isn't needed for a given run.
- Add EarlyStopping + a stratified validation split — safe win, no accuracy downside. Apply at the shared training function so both `03` and `04` benefit.
- **Skip data sampling** — not the bottleneck, and risks hurting minority-class performance as classes grow.
- **Split outlier filters into separate parallel array-job tasks** (and fold, since fold is also currently sequential) — pure job-orchestration change.

---

## Change 0 — Fix latent `--task_id` bug in `multi_lofo_crossval.pbs`

**Bug:** `$PBS_ARRAY_INDEX` is reused for two unrelated purposes: (1) selecting the `exp_folder` variant via the if/else (`nc_subtract` vs not — intentional), and (2) being passed straight through as `--task_id`, which indexes into `config.CROSS_DATASET_GROUPS` (only 1 group exists today). Array index 0 → `--task_id 0` → valid → runs. Array index 1 → `--task_id 1` → `1 >= len(group_names)=1` → `04_cross_dataset_training.py` prints "Task ID 1 is out of bounds for 1 groups. Exiting." and calls `sys.exit(0)` — **does nothing**, but exits 0, so PBS reports success. Net effect: the plain (non-`nc_subtract`) LOFO variant has silently never actually run via this job script.

**Fix:** In `hpc_jobs/multi_lofo_crossval.pbs`, change both branches to pass `--task_id 0` instead of `--task_id $PBS_ARRAY_INDEX` (group selection and the `nc_subtract` exp_folder toggle are orthogonal; revisit only if `CROSS_DATASET_GROUPS` ever grows past 1 entry).

---

## Change 1 — Parameterize determinism strictness (04 only)

**File:** `utils/model_training/model_utils.py`
- Change `def set_global_determinism(seed=0):` → `def set_global_determinism(seed=0, strict=True):`
- Keep RNG seeding (`PYTHONHASHSEED`, `random.seed`, `np.random.seed`, `tf.random.set_seed`) unconditional.
- Gate `TF_DETERMINISTIC_OPS`, `TF_CUDNN_DETERMINISTIC`, and `tf.config.experimental.enable_op_determinism()` behind `if strict:`.
- All existing callers (`01`, `02`, `03`, `07`, `model_for_xai.py`, etc.) call `set_global_determinism(0)` with no `strict` arg → behavior unchanged (defaults to `strict=True`).

**File:** `04_cross_dataset_training.py`
- Remove the module-level `set_global_determinism(0)` call (currently runs before argparse even exists).
- Add `parser.add_argument("--fast_mode", action="store_true", help="Disable strict TF determinism (TF_CUDNN_DETERMINISTIC/enable_op_determinism) for faster GRU/LSTM/Transformer training. RNG seeds are still set, but reruns won't be bit-exact.")`.
- Inside `if __name__ == "__main__":`, right after `args = parser.parse_args()`, call `set_global_determinism(0, strict=not args.fast_mode)`.
- Verified safe: `04_cross_dataset_training.py` is only ever run as a script (grep confirms no other file imports it), so moving the call doesn't break anything.

---

## Change 2 — EarlyStopping + stratified validation split (shared, benefits 03 and 04)

**File:** `utils/model_training/model_utils.py`, inside the per-fold training loop in `evaluate_outlier_filters` (~line 571 onward, before the `if m in ["cnn_lf", ...]` branching).

Right after `X_train_curve`/`y_train`/(optional)`X_train_man` are sliced for the fold, add **one** stratified split, reused by all three training branches:
```python
from sklearn.model_selection import train_test_split
tr_idx, val_idx = train_test_split(
    np.arange(len(y_train)), test_size=0.1, stratify=y_train, random_state=0)
```
This avoids the bug of using Keras's `validation_split` (which takes a *trailing slice* of the array — not stratified, and could miss whole classes depending on row ordering).

Apply consistently to all three Keras code paths:
- **Late-fusion models** (`cnn_lf`, `lstm_lf`, `trans_lf`, `gru_lf`, ~line 592-603): pass `validation_data=([X_val_curve, X_val_man], y_val)` + `callbacks=[EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]` to `model.fit(...)`.
- **Dual models** (`cnn_gru_dual`, `cnn_trans_dual`, ~line 619-629): same pattern with `validation_data=(X_val_curve, y_val)`.
- **`KerasModelWrapper` (scikeras) path** (`cnn`, `lstm`, `gru`, `rnn`, `transformer`, ~line 646-650): `KerasModelWrapper(...).pass`-through subclass of `KerasClassifier` — confirmed scikeras forwards extra `.fit()` kwargs to the underlying Keras `model.fit()`, so call `clf.fit(X_train_curve_sub, y_train_sub, validation_data=(X_val_curve, y_val), callbacks=[EarlyStopping(...)])`.
- Existing fixed `epochs=` values (1000 for cnn, 500 for the rest) become an upper-bound safety cap, not the actual training length — EarlyStopping will stop earlier once `val_loss` plateaus.
- `rf`/`knn`/`ffi` branches are untouched (not epoch-based).

---

## Change 3 — Parallelize LOFO across (fold × filter) array tasks

**Problem this avoids:** the current checkpoint pattern reads-modifies-writes ONE shared `lofo_results` joblib (and a shared per-fold `model_interpretation_{curve_type}.joblib` for `top_10_features`) inside the fold loop. If multiple array tasks for different filters (or folds) ran concurrently against these shared files, updates would race and clobber each other.

**File:** `04_cross_dataset_training.py`

1. Add two new CLI args: `--fold_idx` (int, default `None`) and `--filter_idx` (int, default `None`). When both are given, the script processes **only that one (fold, filter) combination** (all 5 models) instead of looping over all folds × all filters.
2. When in combo mode, write to a **per-combo result file** instead of the shared one:
   `classification_performances_cross_dataset_lofo_{curve_type}__fold{fold_idx}_filt{filter_idx}.joblib`, containing `{fold_label: {filter_key: filter_results}}` (same shape as a slice of the canonical `lofo_results`, so merging later is a simple recursive dict update).
3. Similarly scope the per-fold XAI metadata write (`model_interpretation_{curve_type}.joblib`, the `top_10_features` dict) to a per-combo file in combo mode; the `xai_data_{curve_type}.joblib` snapshot is filter-independent (only depends on fold), so it's already safe to write redundantly from every filter-task for that fold — no change needed there.
4. Add `--merge_only` flag: when set, skip `combine_group`/training entirely and instead glob all per-combo files for the given `curve_type`, recursively merge them into the canonical `lofo_results` / `model_interpretation_{curve_type}.joblib` files (the exact paths/format `05`/`06`/`07`/`08` already expect — **no downstream script needs to change**), and exit. Verification: assert the merged result has all `n_folds × n_filters` combinations present before declaring success; warn (don't silently drop) on any missing combo.

**Files:** `hpc_jobs/multi_lofo_crossval.pbs` (+ mirrored `slurm_jobs/multi_04_lofo_crossval.sh` for consistency with this session's established sync convention)
- Change the array dimension from the current `nc_subtract`-toggle (`-J 0-1`) to a flattened `(fold_idx, filter_idx)` index: `n_folds=4, n_filters=4` → `-J 0-15`, decode inside the script via `fold_idx, filter_idx = divmod(PBS_ARRAY_INDEX, n_filters)` (pass these through as `--fold_idx`/`--filter_idx`).
- Keep the existing `nc_subtract` exp_folder toggle as an outer concern (separate job or a second array dimension) — out of scope to redesign further unless it's actually exercised; flag this as a pre-existing latent issue (current script passes `--task_id $PBS_ARRAY_INDEX` straight into the *group* selector, which only has 1 valid value, so the `else` branch silently no-ops today — see Change 0).
- Add a follow-up merge invocation: either a separate small `hpc_jobs/multi_lofo_merge.pbs` (run manually after confirming the array completed, or via `-W depend=afterok:<array_job_id>`), calling `04_cross_dataset_training.py --merge_only --exp_folder ... --curve_type ...`.

---

## Change 4 — Learning rate scheduling (reduces seed-to-seed variance)

**Motivation:** observed that the same architecture/config produces noticeably different performance across random seeds. Part of this is addressed by Change 2 (EarlyStopping stops training at a validated checkpoint instead of an arbitrary fixed epoch), but the underlying optimization is still a fixed learning rate (Adam, lr=0.001) for the whole run — different seeds can settle into qualitatively different optima or oscillate near a plateau differently. Adaptive LR reduction on plateau helps different seeds converge more consistently to similarly-good optima.

**File:** `utils/model_training/model_utils.py` — same three code paths touched in Change 2 (late-fusion models, dual models, `KerasModelWrapper` path), since `ReduceLROnPlateau` needs the same `validation_data`/`val_loss` monitoring already being added there.
- Add `tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-5)` alongside the `EarlyStopping` callback in each `callbacks=[...]` list (`EarlyStopping` patience should stay larger than `ReduceLROnPlateau` patience so LR gets a chance to drop before training stops, e.g. EarlyStopping patience=15-20 vs ReduceLROnPlateau patience=8).
- No new validation split needed — reuses the exact `(X_val_curve, y_val)` / `(X_val_curve, X_val_man, y_val)` split already introduced in Change 2.
- Does not eliminate seed variance (optimization landscape is still non-convex), but should reduce the spread by preventing seeds from getting stuck at a poor LR for the whole run.

---

## Explicitly out of scope
- Data subsampling — confirmed not the bottleneck.
- Mixed precision, batch size changes, sequence-length downsampling — not requested; can revisit later if more speed is still needed after the above.
- Redesigning the `nc_subtract` toggle dimension — flagged as a pre-existing latent bug, not fixed here (beyond the Change 0 one-line fix).
- Multi-seed averaging for `08_statistical_comparison.py` robustness — separate concern from training speed/variance, not in scope here.

---

## Verification
1. **Determinism flag:** run `04_cross_dataset_training.py` twice with default args (no `--fast_mode`) on a small group → byte-identical `lofo_results` joblib both times (bit-exact preserved). Run once more with `--fast_mode` → confirm it still completes and produces valid (not necessarily identical) results, and measure wall-clock improvement on one GRU fold.
2. **EarlyStopping + LR scheduling:** run one fold and confirm via logs that training stops well before the epoch cap (e.g. <100 of 500), that LR visibly drops at least once before stopping, and that held-out test accuracy is comparable to (not worse than) a pre-change baseline run. Optionally run the same fold across 3 seeds before/after this change to confirm the spread in accuracy shrinks.
3. **Parallel combo mode:** run 2-3 `--fold_idx`/`--filter_idx` combinations manually, then `--merge_only`, and diff the merged joblib structure against a full sequential run on the same data to confirm no combos are missing and values match.
4. **End-to-end:** submit the updated `hpc_jobs/multi_lofo_crossval.pbs` array (small group first) + merge job, confirm `05`/`06`/`07`/`08` all still read the merged output correctly with no code changes on their end.
