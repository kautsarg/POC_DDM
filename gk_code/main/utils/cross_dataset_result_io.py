"""
Per-(outlier_filter, model) result-file layout for 04_cross_dataset_training.py
and 06b_cross_dataset_prediction_report.py.

Before this, all supcon/--dann/--coral/--models variants for a given (group,
curve_alignment, mode, curve_type) shared one joblib file
(config.CROSS_DATASET_RESULT_PATH). Each process loaded it once, held it in
memory for the whole run, and periodically overwrote the whole file --
concurrent processes (e.g. two qsub submissions differing only in --supcon)
would silently erase each other's checkpoints (last writer wins; safe_io's
atomic write prevents corruption, not lost updates). Splitting to one file per
(filter, model) makes that collision structurally impossible: two processes
only share a file if they're training the exact same model under the exact
same filter, which they'd only do if truly duplicating each other's work.

save_partitioned()/load_partitioned() are drop-in replacements for the
`safe_joblib_dump(lofo_results, results_file_path, ...)` /
`joblib.load(results_file_path)` calls in 04_cross_dataset_training.py --
they take the exact same in-memory shape
({fold_label: {filter: res_entry, ...}, "top_10_features": ..., "class_names": ...})
so no other code needs to change.
"""
import re
from pathlib import Path

import joblib

import config
from safe_io import safe_joblib_dump

# res_entry (lofo_results[fold_label][filter]) keys that aren't tied to one
# model -- duplicated into every per-model file for that (fold, filter) so
# each file is self-contained.
_SHARED_RES_ENTRY_KEYS = ("y_trues_", "_split_signature", "mask_count", "y_true_count")
# lofo_results[fold_label] keys that are siblings of the filter keys (not
# specific to any filter or model) -- duplicated into every file for that fold.
_SHARED_FOLD_KEYS = ("top_10_features", "class_names")

_FILTER_TOKEN_RE = re.compile(r'[^A-Za-z0-9_.-]+')
_DEFAULT_LEGACY_PATH = object()  # sentinel: "not passed", distinct from legacy_path=None


def filter_token(filter_name):
    """Filesystem-safe subfolder name for an outlier_filter value (None -> 'none')."""
    if filter_name is None:
        return "none"
    return _FILTER_TOKEN_RE.sub("_", str(filter_name))


def _token_to_filter_name(token):
    """Inverse of filter_token: outlier_filter values are already plain
    identifiers (features_df column names), so sanitisation in filter_token()
    is a no-op for every real filter -- the token round-trips as-is."""
    return None if token == "none" else token


def _model_keys_for(model_key, res_entry):
    """All result-dict keys belonging to `model_key` that are present in res_entry."""
    if model_key not in config.MODEL_KEY_MAP:
        return []
    preds_key, probs_key, classes_key = config.MODEL_KEY_MAP[model_key]
    candidates = (preds_key, probs_key, classes_key,
                  f'train_history_{model_key}_',
                  f'y_reg_preds_{model_key}_', f'y_reg_trues_{model_key}_')
    return [k for k in candidates if k in res_entry]


def _models_present(res_entry):
    return [m for m, (preds_key, _, _) in config.MODEL_KEY_MAP.items() if preds_key in res_entry]


def model_result_path(out_dir, mode, curve_type, filter_name, model_key):
    return (Path(out_dir) / filter_token(filter_name)
            / config.CROSS_DATASET_RESULT_PATH_PER_MODEL.format(
                mode=mode, curve_type=curve_type, model=model_key))


def save_partitioned(lofo_results, out_dir, mode, curve_type, compress=3):
    per_file = {}
    for fold_label, fold_res in lofo_results.items():
        if not isinstance(fold_res, dict):
            continue
        fold_shared = {k: fold_res[k] for k in _SHARED_FOLD_KEYS if k in fold_res}
        for filter_name, res_entry in fold_res.items():
            if filter_name in _SHARED_FOLD_KEYS or not isinstance(res_entry, dict):
                continue
            entry_shared = {k: res_entry[k] for k in _SHARED_RES_ENTRY_KEYS if k in res_entry}
            for model_key in _models_present(res_entry):
                path = model_result_path(out_dir, mode, curve_type, filter_name, model_key)
                model_slice = dict(entry_shared)
                for k in _model_keys_for(model_key, res_entry):
                    model_slice[k] = res_entry[k]
                per_file.setdefault(path, {})[fold_label] = {**fold_shared, **model_slice}
    for path, data in per_file.items():
        safe_joblib_dump(data, path, compress=compress)


def load_partitioned(out_dir, mode, curve_type, legacy_path=_DEFAULT_LEGACY_PATH):
    out_dir = Path(out_dir)
    lofo_results = {}
    name_glob = config.CROSS_DATASET_RESULT_PATH_PER_MODEL.format(mode=mode, curve_type=curve_type, model="*")
    per_model_files = sorted(out_dir.glob(f"*/{name_glob}"))

    if not per_model_files:
        if legacy_path is _DEFAULT_LEGACY_PATH:
            legacy_path = out_dir / config.CROSS_DATASET_RESULT_PATH.format(mode=mode, curve_type=curve_type)
        if legacy_path is not None and legacy_path.exists():
            try:
                return joblib.load(legacy_path)
            except Exception:
                return {}
        return {}

    for path in per_model_files:
        filter_name = _token_to_filter_name(path.parent.name)
        try:
            data = joblib.load(path)
        except Exception:
            continue
        for fold_label, entry in data.items():
            fold_res = lofo_results.setdefault(fold_label, {})
            for k in _SHARED_FOLD_KEYS:
                if k in entry:
                    fold_res[k] = entry[k]
            res_entry = fold_res.setdefault(filter_name, {})
            for k, v in entry.items():
                if k in _SHARED_FOLD_KEYS:
                    continue
                res_entry[k] = v
    return lofo_results
