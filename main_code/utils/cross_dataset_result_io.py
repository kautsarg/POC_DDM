import re
from pathlib import Path

import joblib
from filelock import FileLock, Timeout

import config
from safe_io import safe_joblib_dump

_LOCK_TIMEOUT_SECONDS = 300

_SHARED_RES_ENTRY_KEYS = ("y_trues_", "well_ids_test_", "_split_signature", "mask_count", "y_true_count")
_SHARED_FOLD_KEYS = ("top_10_features", "class_names")

_FILTER_TOKEN_RE = re.compile(r'[^A-Za-z0-9_.-]+')


def filter_token(filter_name):
    if filter_name is None:
        return "none"
    return _FILTER_TOKEN_RE.sub("_", str(filter_name))


def _token_to_filter_name(token):
    return None if token == "none" else token


def _model_keys_for(model_key, res_entry):
    """All result-dict keys belonging to `model_key` that are present in res_entry."""
    if model_key not in config.MODEL_KEY_MAP:
        return []
    preds_key, probs_key, classes_key = config.MODEL_KEY_MAP[model_key]
    candidates = (preds_key, probs_key, classes_key, f'train_history_{model_key}_')
    return [k for k in candidates if k in res_entry]


def _models_present(res_entry, restrict_to=None):
    keys = (m for m, (preds_key, _, _) in config.MODEL_KEY_MAP.items() if preds_key in res_entry)
    if restrict_to is not None:
        restrict_to = set(restrict_to)
        keys = (m for m in keys if m in restrict_to)
    return list(keys)


def model_result_path(out_dir, mode, curve_type, filter_name, model_key):
    return Path(out_dir) / filter_token(filter_name) / config.CROSS_DATASET_RESULT_PATH_PER_MODEL.format(
        mode=mode, curve_type=curve_type, model=model_key)


def _read_merge_write(path, data, compress):
    lock = FileLock(str(path) + ".lock", timeout=_LOCK_TIMEOUT_SECONDS)
    try:
        with lock:
            existing = {}
            if path.exists():
                try:
                    existing = joblib.load(path)
                except Exception:
                    existing = {}
            merged = {**existing, **data}
            safe_joblib_dump(merged, path, compress=compress)
    except Timeout:
        raise RuntimeError(
            f"[cross_dataset_result_io] Could not acquire lock for {path} within "
            f"{_LOCK_TIMEOUT_SECONDS}s -- another process may be stuck holding it.")


def save_partitioned(lofo_results, out_dir, mode, curve_type, *, models, compress=3):
    models = {m.lower() for m in models}
    per_file = {}
    for fold_label, fold_res in lofo_results.items():
        if not isinstance(fold_res, dict):
            continue
        fold_shared = {k: fold_res[k] for k in _SHARED_FOLD_KEYS if k in fold_res}
        for filter_name, res_entry in fold_res.items():
            if filter_name in _SHARED_FOLD_KEYS or not isinstance(res_entry, dict):
                continue
            entry_shared = {k: res_entry[k] for k in _SHARED_RES_ENTRY_KEYS if k in res_entry}
            for model_key in _models_present(res_entry, restrict_to=models):
                path = model_result_path(out_dir, mode, curve_type, filter_name, model_key)
                model_slice = dict(entry_shared)
                for k in _model_keys_for(model_key, res_entry):
                    model_slice[k] = res_entry[k]
                per_file.setdefault(path, {})[fold_label] = {**fold_shared, **model_slice}

    for path, data in per_file.items():
        _read_merge_write(path, data, compress)


def load_partitioned(out_dir, mode, curve_type):
    out_dir = Path(out_dir)
    lofo_results = {}
    name_glob = config.CROSS_DATASET_RESULT_PATH_PER_MODEL.format(mode=mode, curve_type=curve_type, model="*")

    def _merge(path, filter_name):
        try:
            data = joblib.load(path)
        except Exception:
            return
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

    for path in sorted(out_dir.glob(f"*/{name_glob}")):
        _merge(path, _token_to_filter_name(path.parent.name))

    return lofo_results
