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


def _frac_token(train_center_frac):
    """Filesystem-safe subfolder name for a train_center_frac value, e.g. 0.5 -> 'center0.5'."""
    return f"center{train_center_frac:g}"


def model_result_path(out_dir, mode, curve_type, filter_name, model_key, train_center_frac=None):
    path = Path(out_dir) / filter_token(filter_name)
    if train_center_frac is not None:
        path = path / _frac_token(train_center_frac)
    return path / config.CROSS_DATASET_RESULT_PATH_PER_MODEL.format(
        mode=mode, curve_type=curve_type, model=model_key)


def save_partitioned(lofo_results, out_dir, mode, curve_type, compress=3, train_center_frac=None):
    """train_center_frac path-scopes every fold except "full_data", which always saves
    to the unscoped location regardless of what's passed here -- --train_full never
    uses train_center_frac (04_cross_dataset_training.py), so its output must stay where
    06b/08 already expect to find it no matter what this run's train_center_frac was."""
    per_file = {}
    full_data_paths = set()
    for fold_label, fold_res in lofo_results.items():
        if not isinstance(fold_res, dict):
            continue
        effective_frac = None if fold_label == "full_data" else train_center_frac
        fold_shared = {k: fold_res[k] for k in _SHARED_FOLD_KEYS if k in fold_res}
        for filter_name, res_entry in fold_res.items():
            if filter_name in _SHARED_FOLD_KEYS or not isinstance(res_entry, dict):
                continue
            entry_shared = {k: res_entry[k] for k in _SHARED_RES_ENTRY_KEYS if k in res_entry}
            for model_key in _models_present(res_entry):
                path = model_result_path(out_dir, mode, curve_type, filter_name, model_key,
                                          train_center_frac=effective_frac)
                model_slice = dict(entry_shared)
                for k in _model_keys_for(model_key, res_entry):
                    model_slice[k] = res_entry[k]
                per_file.setdefault(path, {})[fold_label] = {**fold_shared, **model_slice}
                if fold_label == "full_data" and train_center_frac is not None:
                    full_data_paths.add(path)

    for path, data in per_file.items():
        if path in full_data_paths and path.exists():
            try:
                existing = joblib.load(path)
            except Exception:
                existing = {}
            data = {**existing, **data}
        safe_joblib_dump(data, path, compress=compress)


def load_partitioned(out_dir, mode, curve_type, legacy_path=_DEFAULT_LEGACY_PATH, train_center_frac=None):
    """train_center_frac must match what save_partitioned() was called with: None (the
    default) reads the unscoped layout; a float reads that fraction's own subfolder for
    every fold except "full_data", which always lives at the unscoped location (--train_full
    never uses train_center_frac) and is merged in regardless. Legacy single-file fallback
    only applies when train_center_frac is None, since the legacy layout predates this
    parameter entirely."""
    out_dir = Path(out_dir)
    lofo_results = {}
    name_glob = config.CROSS_DATASET_RESULT_PATH_PER_MODEL.format(mode=mode, curve_type=curve_type, model="*")

    def _merge(path, filter_name, only_fold_label=None):
        try:
            data = joblib.load(path)
        except Exception:
            return
        for fold_label, entry in data.items():
            if only_fold_label is not None and fold_label != only_fold_label:
                continue
            fold_res = lofo_results.setdefault(fold_label, {})
            for k in _SHARED_FOLD_KEYS:
                if k in entry:
                    fold_res[k] = entry[k]
            res_entry = fold_res.setdefault(filter_name, {})
            for k, v in entry.items():
                if k in _SHARED_FOLD_KEYS:
                    continue
                res_entry[k] = v

    if train_center_frac is not None:
        frac_token = _frac_token(train_center_frac)
        for path in sorted(out_dir.glob(f"*/{frac_token}/{name_glob}")):
            _merge(path, _token_to_filter_name(path.parent.parent.name))
        # "full_data" always lives at the unscoped location -- merge it in regardless.
        for path in sorted(out_dir.glob(f"*/{name_glob}")):
            _merge(path, _token_to_filter_name(path.parent.name), only_fold_label="full_data")
    else:
        for path in sorted(out_dir.glob(f"*/{name_glob}")):
            _merge(path, _token_to_filter_name(path.parent.name))

    if not lofo_results:
        if train_center_frac is None:
            if legacy_path is _DEFAULT_LEGACY_PATH:
                legacy_path = out_dir / config.CROSS_DATASET_RESULT_PATH.format(mode=mode, curve_type=curve_type)
            if legacy_path is not None and legacy_path.exists():
                try:
                    return joblib.load(legacy_path)
                except Exception:
                    return {}
        return {}

    return lofo_results
