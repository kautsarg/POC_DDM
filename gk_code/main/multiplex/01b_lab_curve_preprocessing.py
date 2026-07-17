"""Multiplex lab curve preprocessing.

Reads dPCR_Dataset_Multiplex.csv, splits multi-label strings (e.g. "VIM_NDM"),
fits 5-parameter sigmoid, and writes curve_for_training_ml.joblib.
"""
import os
import re
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed

_THIS         = Path(__file__).resolve()
_MULTIPLEX_DIR = _THIS.parent
_MAIN_DIR     = _MULTIPLEX_DIR.parent

sys.path.insert(0, str(_MAIN_DIR))
sys.path.insert(0, str(_MAIN_DIR / "utils"))
sys.path.insert(0, str(_MAIN_DIR / "utils" / "model_training"))
sys.path.insert(0, str(_MULTIPLEX_DIR))

import config_multiplex as config

from safe_io import safe_joblib_dump
import sigmoid_fitting as sp


# ======================================================================
# HELPERS
# ======================================================================

def _get_numeric_sort_key(col):
    m = re.search(r'[\d.]+', str(col))
    return float(m.group()) if m else 0.0


def _fit_single_curve(y_row, x_time, start_idx=0):
    try:
        y_num = np.asanyarray(y_row, dtype=np.float64)
        x_num = np.asanyarray(x_time, dtype=np.float64)
        y_fit = y_num[start_idx:]
        x_fit = x_num[start_idx:]
        params, _ = sp.fit_5p(x_fit, y_fit, normalize=True)
        fitted_full      = sp.sigmoid_5p(x_num, *params)
        fitted_segment   = sp.sigmoid_5p(x_fit, *params)
        old_x = np.linspace(0, 1, len(fitted_segment))
        new_x = np.linspace(0, 1, len(x_num))
        fitted_stretched = np.interp(new_x, old_x, fitted_segment)
        rmse = np.sqrt(np.nanmean(np.square(fitted_segment - y_fit)))
        return fitted_full, fitted_stretched, params, rmse
    except Exception:
        nan_arr = np.full_like(x_time, np.nan, dtype=np.float64)
        return nan_arr, nan_arr, np.full(5, np.nan), np.nan


def sigmoid_fitting_5p(curves, ori_timestamps, starting_idxs=None):
    if starting_idxs is None:
        starting_idxs = np.zeros(len(curves), dtype=int)
    results = Parallel(n_jobs=-1, backend="loky", batch_size='auto')(
        delayed(_fit_single_curve)(y, ori_timestamps, t)
        for y, t in zip(curves, starting_idxs)
    )
    curves_full, curves_stretched, params_out, rmse_out = zip(*results)
    return np.array(curves_full), np.array(curves_stretched), np.array(params_out), np.array(rmse_out)


def parse_multilabel_column(raw_labels, separator=config.MULTIPLEX_SEPARATOR):
    """Split 'VIM_NDM' -> ['VIM','NDM'] for each row."""
    result = []
    for lbl in raw_labels:
        if pd.isna(lbl) or str(lbl).strip() == '':
            result.append([])
        else:
            parts = [p.strip() for p in str(lbl).split(separator) if p.strip()]
            result.append(parts)
    return result


def load_multiplex_curves(exp_path):
    """Read CSV, extract curves, multi-label lists, and concentration."""
    folder = Path(exp_path).name
    file_path = os.path.join(exp_path, config.FILE_MAPPING[folder])
    df = pd.read_csv(file_path)

    cycle_cols = sorted(
        [c for c in df.columns
         if str(c).startswith('Cycle') or re.match(r'^\d+(\.\d+)?$', str(c))],
        key=_get_numeric_sort_key)

    timestamps   = np.array([_get_numeric_sort_key(c) for c in cycle_cols])
    curves       = df[cycle_cols].to_numpy(dtype=float)
    raw_labels   = df[config.FILE_TARGET[folder]].to_numpy()
    label_lists  = parse_multilabel_column(raw_labels, config.MULTIPLEX_SEPARATOR)

    conc_col     = config.FILE_CONC.get(folder)
    concentration = df[conc_col].to_numpy(dtype=float) if conc_col else None

    return curves, timestamps, label_lists, concentration


def save_multiplex_experiment_data(save_path, curves, timestamps, label_lists,
                                   sigmoid_results, concentration=None):
    """Write curve_for_training_ml.joblib (mirrors main/01b save_experiment_data)."""
    from sklearn.preprocessing import MultiLabelBinarizer

    all_targets  = sorted({t for lbl in label_lists for t in lbl})
    mlb          = MultiLabelBinarizer(classes=all_targets)
    y_binary     = mlb.fit_transform(label_lists).astype(np.int8)
    combo_labels = np.array(['_'.join(sorted(lbl)) if lbl else 'EMPTY' for lbl in label_lists])

    fitted_full, fitted_stretched, params, rmse = sigmoid_results

    data = {
        "curves":          {"ori_curves": curves},
        "sigmoid_curves":  {
            "original": {
                "fitted_full":      fitted_full,
                "fitted_stretched": fitted_stretched,
                "params":           params,
                "rmse":             rmse,
            }
        },
        "timestamps":      timestamps,
        "well_labels":     combo_labels,    # combo string, e.g. "VIM_NDM"
        "label_lists":     label_lists,     # list of lists, e.g. [['VIM','NDM'], ...]
        "label_binarized": y_binary,        # (N, n_targets) int8
        "all_targets":     all_targets,     # e.g. ['KPC','NDM','VIM']
        "concentration":   concentration,
        "metadata":        {},
        "baseline_value":  0.0,
    }

    existing = {}
    if os.path.exists(save_path):
        try:
            existing = joblib.load(save_path)
        except Exception as e:
            print(f"  -> [WARNING] Existing file unreadable ({e}). Overwriting.")
    existing.update(data)
    safe_joblib_dump(existing, save_path, compress=3)
    print(f"  -> Saved to {save_path}  "
          f"(N={len(curves)}, targets={all_targets}, "
          f"y_binary.shape={y_binary.shape})")


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":
    print(f"\n{'='*70}\n[RUNNING] {os.path.basename(__file__)}\n{'='*70}\n")
    parser = argparse.ArgumentParser(description="Multiplex Lab Curve Preprocessing")
    parser.add_argument("--task_id",     type=int, default=0)
    parser.add_argument("--exp_folder",  type=str, default=config.LAB_MULTIPLEX_FOLDER)
    parser.add_argument("--force_rerun", action="store_true")
    args = parser.parse_args()

    exp_folder    = args.exp_folder
    subdirs       = sorted([d for d in Path(exp_folder).iterdir() if d.is_dir()])
    all_task_dirs = [d for d in subdirs if d.name in config.FILE_MAPPING]

    if not all_task_dirs:
        print(f"No valid task directories found in {exp_folder}.")
        sys.exit(0)

    task_dirs = all_task_dirs if args.task_id < 0 else [all_task_dirs[args.task_id % len(all_task_dirs)]]

    for exp_path in task_dirs:
        folder = exp_path.name
        print(f"\n[Task] {folder}")
        save_path = os.path.join(exp_path, config.TRAINING_DATA_PATH)

        if os.path.exists(save_path) and not args.force_rerun:
            try:
                cached = joblib.load(save_path)
                if 'label_binarized' in cached:
                    print(f"  -> [SKIP] {save_path} already exists with label_binarized.")
                    continue
            except Exception:
                pass

        curves, timestamps, label_lists, concentration = load_multiplex_curves(exp_path)
        print(f"  -> Loaded {len(curves)} curves, {len(timestamps)} timepoints.")

        # Flatten valid concentration values for diagnostics
        if concentration is not None:
            valid_conc = concentration[np.isfinite(concentration) & (concentration > 0)]
            print(f"  -> Concentration: {len(valid_conc)}/{len(concentration)} valid. "
                  f"Range [{valid_conc.min():.1f}, {valid_conc.max():.1f}]"
                  if len(valid_conc) > 0 else "  -> No valid concentration values.")

        print(f"  -> Fitting 5-parameter sigmoid ({len(curves)} curves)...")
        sigmoid_results = sigmoid_fitting_5p(curves, timestamps)

        save_multiplex_experiment_data(save_path, curves, timestamps, label_lists,
                                       sigmoid_results, concentration)
