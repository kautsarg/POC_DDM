import os
import sys
import re
import itertools
import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from joblib import Parallel, delayed

import config
import sigmoid_fitting as sp

# ==========================================
# LAB_DDM_paper dataset registry
# ==========================================
# Unlike 01_curve_preprocessing_v6.py (pixel-grid chip data with row/col/well
# coordinates), these are flat per-sample qPCR/qdLAMP curves read straight out
# of a CSV -- one row per sample, no spatial structure at all. Each entry here
# is a subfolder of config.LAB_EXP_FOLDER.
FILE_MAPPING = {
    'ACA_qdPCR':   'dPCR_Dataset_Pure.csv',
    'AMCA_qdPCR':  'dPCR_Amplification_Curves.csv',
    'AMCA_qdLAMP': '5Plex_dLAMP_oldData.csv',
    'Z_area': 'area_strategy.csv',
    'Z_range': 'range_strategy.xlsx',
}

FILE_MAPPING_ONE_TO_ONE = {
    '1_area': 'area_strategy.csv',
    '2_range': 'range_strategy.xlsx',
}
FILE_TARGET = {
    'ACA_qdPCR':   'LoadedPanels',
    'AMCA_qdPCR':  'Target',
    'AMCA_qdLAMP': 'Target',
    'Z_area':   'Target',
    'Z_range':  'Target',
}

FILE_TARGET_ONE_TO_ONE = {
    '1_area':   'Target',
    '2_range':  'Target',
}

def get_numeric_sort_key(col_name):
    nums = re.findall(r'\d+\.?\d*', col_name)
    return float(nums[0]) if nums else 0.0


def _fit_single_curve(y_row, x_time, start_idx=0):
    try:
        y_num = np.asanyarray(y_row, dtype=np.float64)
        x_num = np.asanyarray(x_time, dtype=np.float64)
        y_fit = y_num[start_idx:]
        x_fit = x_num[start_idx:]

        params, _ = sp.fit_5p(x_fit, y_fit, normalize=True)
        fitted_full = sp.sigmoid_5p(x_num, *params)
        fitted_segment = sp.sigmoid_5p(x_fit, *params)

        old_x_norm = np.linspace(0, 1, len(fitted_segment))
        new_x_norm = np.linspace(0, 1, len(x_num))
        fitted_stretched = np.interp(new_x_norm, old_x_norm, fitted_segment)
        rmse = np.sqrt(np.nanmean(np.square(fitted_segment - y_fit)))

        return fitted_full, fitted_stretched, params, rmse

    except Exception:
        nan_array = np.full_like(x_time, np.nan, dtype=np.float64)
        return nan_array, nan_array, np.full(5, np.nan), np.nan


def sigmoid_fitting_5p(curves, ori_timestamps, starting_idxs=None):
    if starting_idxs is None:
        starting_idxs = np.zeros(len(curves), dtype=int)

    results = Parallel(n_jobs=-1, backend="loky", batch_size='auto')(
        delayed(_fit_single_curve)(y, ori_timestamps, t)
        for y, t in zip(curves, starting_idxs)
    )

    curves_out_full, curves_out_stretched, params_out, rmse_out = zip(*results)
    return np.array(curves_out_full), np.array(curves_out_stretched), np.array(params_out), np.array(rmse_out)


def load_lab_curves(exp_path):
    """Read this experiment's CSV/Excel file and return (curves, timestamps, well_labels)."""
    folder = exp_path.name
    file_path = os.path.join(exp_path, FILE_MAPPING[folder])
    df = _read_table(file_path)

    cycle_cols = [col for col in df.columns
                  if str(col).startswith('Cycle') or re.match(r'^\d+(\.\d+)?$', str(col))]
    sorted_cycle_cols = sorted(cycle_cols, key=get_numeric_sort_key)

    timestamps = np.array([get_numeric_sort_key(col) for col in sorted_cycle_cols])
    curves = df[sorted_cycle_cols].to_numpy()
    well_labels = df[FILE_TARGET[folder]].to_numpy()

    return curves, timestamps, well_labels



def _read_table(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(file_path)
    return pd.read_csv(file_path)


def discover_one_to_one_combos(exp_folder):
    combos = []
    for strategy in sorted(FILE_MAPPING_ONE_TO_ONE.keys()):
        file_path = os.path.join(exp_folder, strategy, FILE_MAPPING_ONE_TO_ONE[strategy])
        target_col = FILE_TARGET_ONE_TO_ONE[strategy]
        labels = sorted(_read_table(file_path)[target_col].unique())
        for label1, label2 in itertools.combinations(labels, 2):
            combo_name = f"{strategy}__{label1}_vs_{label2}"
            combos.append((strategy, label1, label2, combo_name))
    return combos


def _strategy_fit_cache_path(exp_folder, strategy):
    return os.path.join(exp_folder, f"_one_to_one_fitted_cache__{strategy}.joblib")


def load_and_fit_strategy_once(exp_folder, strategy):
    cache_path = _strategy_fit_cache_path(exp_folder, strategy)
    if os.path.exists(cache_path):
        try:
            return joblib.load(cache_path)
        except Exception as e:
            print(f"  -> [WARNING] Strategy fit cache at {cache_path} unreadable ({e}). Recomputing.")

    file_path = os.path.join(exp_folder, strategy, FILE_MAPPING_ONE_TO_ONE[strategy])
    target_col = FILE_TARGET_ONE_TO_ONE[strategy]
    df = _read_table(file_path)

    cycle_cols = [col for col in df.columns
                  if str(col).startswith('Cycle') or re.match(r'^\d+(\.\d+)?$', str(col))]
    sorted_cycle_cols = sorted(cycle_cols, key=get_numeric_sort_key)
    timestamps = np.array([get_numeric_sort_key(col) for col in sorted_cycle_cols])

    curves = df[sorted_cycle_cols].to_numpy()
    well_labels = df[target_col].to_numpy()

    print(f"  -> Fitting sigmoid curves for ALL of strategy '{strategy}' once ({len(curves)} curves)...")
    sigmoid_results = sigmoid_fitting_5p(curves, timestamps)

    cached = {
        "curves": curves, "timestamps": timestamps, "well_labels": well_labels,
        "sigmoid_results": sigmoid_results,
    }
    joblib.dump(cached, cache_path, compress=3)
    return cached


def load_lab_curves_one_to_one(exp_folder, strategy, label1, label2):
    cached = load_and_fit_strategy_once(exp_folder, strategy)
    mask = np.isin(cached["well_labels"], [label1, label2])

    curves = cached["curves"][mask]
    well_labels = cached["well_labels"][mask]
    fitted_full, fitted_stretched, params, rmse = cached["sigmoid_results"]
    sigmoid_results = (fitted_full[mask], fitted_stretched[mask], params[mask], rmse[mask])

    return curves, cached["timestamps"], well_labels, sigmoid_results


def is_cache_hit(save_path):
    if not os.path.exists(save_path):
        return False
    try:
        existing_data = joblib.load(save_path)
    except Exception as e:
        print(f"  -> [WARNING] Cached file at {save_path} is corrupted ({e}). Recomputing from scratch...")
        return False
    return existing_data is not None and "curves" in existing_data


def patch_missing_norm_variant(save_path, normalize_curves):
    """On a cache hit, add a missing 'ori_curves_norm' from the already-cached 'ori_curves'
    instead of forcing a full re-run (which would re-read the CSV/Excel and re-fit sigmoids)."""
    if not normalize_curves:
        return
    existing_data = joblib.load(save_path)
    if "ori_curves_norm" in existing_data["curves"]:
        return
    existing_data["curves"]["ori_curves_norm"] = normalize_curves_minmax(existing_data["curves"]["ori_curves"])
    joblib.dump(existing_data, save_path, compress=3)
    print(f"  -> Patched missing 'ori_curves_norm' into {save_path}")


def normalize_curves_minmax(curves):
    curves = np.asarray(curves, dtype=np.float64)
    row_min = curves.min(axis=1, keepdims=True)
    row_max = curves.max(axis=1, keepdims=True)
    denom = np.where(row_max - row_min == 0, 1, row_max - row_min)
    return (curves - row_min) / denom


def save_experiment_data(save_exp_path, curves, timestamps, well_labels, sigmoid_results, normalize_curves=False):
    fitted_full, fitted_stretched, params, rmse = sigmoid_results

    curves_dict = {"ori_curves": curves}
    if normalize_curves:
        curves_dict["ori_curves_norm"] = normalize_curves_minmax(curves)

    save_data = {
        "curves": curves_dict,
        "sigmoid_curves": {
            "original": {
                "fitted_full": fitted_full,
                "fitted_stretched": fitted_stretched,
                "params": params,
                "rmse": rmse,
            }
        },
        "timestamps": timestamps,
        "well_labels": well_labels,
        "metadata": {},
        "baseline_value": 0.0,
    }

    save_path = os.path.join(save_exp_path, config.TRAINING_DATA_PATH)
    existing_state = {}
    if os.path.exists(save_path):
        try:
            existing_state = joblib.load(save_path)
        except Exception as e:
            print(f"  -> [WARNING] Existing shared cache at {save_path} unreadable ({e}). Overwriting.")
            existing_state = {}
    existing_state.update(save_data)   # only overwrites this script's own keys -- 02's keys
    joblib.dump(existing_state, save_path, compress=3)   # (dataset, kinetic_features, ...) are left untouched
    print(f"  -> Saved numerical results and metadata to {save_path}")


if __name__ == "__main__":
    print(f"\n{'='*70}\n[RUNNING] {os.path.basename(__file__)}\n{'='*70}\n")
    parser = argparse.ArgumentParser(description="Lab (LAB_DDM_paper) Curve Preprocessing Pipeline")
    parser.add_argument("--task_id", type=int, default=0, help="Array Job ID")
    parser.add_argument("--exp_folder", type=str, default=config.LAB_EXP_FOLDER, help="Path to LAB_DDM_paper experiment datasets")
    parser.add_argument("--force_rerun", action="store_true", help="Recompute and overwrite even if a presaved file already exists")
    parser.add_argument("--one_to_one", action="store_true",
                        help="Pairwise one-to-one mode (use with --exp_folder=config.LAB_1TO1_EXP_FOLDER): "
                             "build one curve_for_training.joblib per (strategy, label-pair) combination, "
                             "in its own subfolder, instead of one shared joblib per strategy folder.")
    parser.add_argument("--normalize_curves", action="store_true",
                        help="Add an 'ori_curves_norm' variant: each curve independently min-max scaled to "
                             "[0,1]. Selectable downstream via --curve_type ori_curve_norm.")
    args = parser.parse_args()

    if args.one_to_one:
        combos = discover_one_to_one_combos(args.exp_folder)
        for _, _, _, combo_name in combos:
            os.makedirs(os.path.join(args.exp_folder, combo_name), exist_ok=True)

        if args.task_id >= len(combos):
            print(f"Task ID {args.task_id} is out of bounds for {len(combos)} combinations. Exiting.")
            sys.exit(0)

        strategy, label1, label2, combo_name = combos[args.task_id]
        exp_path = Path(args.exp_folder, combo_name)
        save_path = os.path.join(exp_path, config.TRAINING_DATA_PATH)

        if is_cache_hit(save_path) and not args.force_rerun:
            patch_missing_norm_variant(save_path, args.normalize_curves)
            print(f"Cache hit: {exp_path}")
            print("  ✓ Experiment complete!\n")
            sys.exit(0)

        print(f"Processing One-to-One Combo: {combo_name}  ({strategy}: {label1} vs {label2})")

        curves, timestamps, well_labels, sigmoid_results = load_lab_curves_one_to_one(
            args.exp_folder, strategy, label1, label2)

        save_experiment_data(exp_path, curves, timestamps, well_labels, sigmoid_results,
                            normalize_curves=args.normalize_curves)

        print(f"  -> X (Curves) shape:       {curves.shape}")
        print(f"  -> well_labels (Targets):  {np.unique(well_labels)}")
        print(f"  -> Mean RMSE Fit Error:    {np.nanmean(sigmoid_results[3]):.4f}")
        print("  ✓ Experiment complete!\n")
        sys.exit(0)

    exp_paths = sorted([Path(args.exp_folder, name) for name in os.listdir(args.exp_folder)
                        if os.path.isdir(os.path.join(args.exp_folder, name)) and name in FILE_MAPPING])

    if args.task_id >= len(exp_paths):
        print(f"Task ID {args.task_id} is out of bounds for {len(exp_paths)} folders. Exiting.")
        sys.exit(0)

    exp_path = exp_paths[args.task_id]
    save_path = os.path.join(exp_path, config.TRAINING_DATA_PATH)

    if is_cache_hit(save_path) and not args.force_rerun:
        patch_missing_norm_variant(save_path, args.normalize_curves)
        print(f"Cache hit: {exp_path}")
        print("  ✓ Experiment complete!\n")
        sys.exit(0)

    print(f"Processing Experiment: {exp_path}")

    curves, timestamps, well_labels = load_lab_curves(exp_path)

    print("  -> Processing Sigmoid Curves...")
    sigmoid_results = sigmoid_fitting_5p(curves, timestamps)

    save_experiment_data(exp_path, curves, timestamps, well_labels, sigmoid_results,
                        normalize_curves=args.normalize_curves)

    print(f"  -> X (Curves) shape:       {curves.shape}")
    print(f"  -> well_labels (Targets):  {np.unique(well_labels)}")
    print(f"  -> Mean RMSE Fit Error:    {np.nanmean(sigmoid_results[3]):.4f}")
    print("  ✓ Experiment complete!\n")
