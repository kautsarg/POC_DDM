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

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "utils"))

import config
from safe_io import safe_joblib_dump
import sigmoid_fitting as sp
from kinetic_features import build_kinetic_features

FILE_MAPPING = {
    '01_ACA_qdPCR':   'dPCR_Dataset_Pure.csv',
    '02_AMCA_qdLAMP': '5Plex_dLAMP_oldData.csv',
    '03_AMCA_qdPCR':  'dPCR_Amplification_Curves.csv',
    '04_Z_area': 'area_strategy.csv',
    '05_Z_range': 'range_strategy.xlsx',
    '06_Z_range_filtered': 'range_strategy_filtered.xlsx',
    '07_ACA_qdPCR_conc_filtered':   'dPCR_Dataset_Pure.csv',
    '08_AMCA_qdPCR_conc_filtered':  'dPCR_Amplification_Curves.csv',
    '09_Area_Conc': 'Area_Strategy_with_conc.csv',
    '10_Range_Conc': 'Range_Strategy_5P_with_conc.csv',
    '11_ACA_qdPCR_multiplex': 'dPCR_Dataset_Multiplex.csv',
    '12_ACA_qdPCR_multiplex_balanced': 'dPCR_Dataset_Mixture.csv',
}
FILE_CONC = {
    '01_ACA_qdPCR': 'Conc',
    '03_AMCA_qdPCR':  'Conc',
    '02_AMCA_qdLAMP': 'Conc',
    '04_Z_area': None,
    '05_Z_range': None,
    '06_Z_range_filtered': None,
    '07_ACA_qdPCR_conc_filtered': 'Conc',
    '08_AMCA_qdPCR_conc_filtered': 'Conc',
    '09_Area_Conc': 'Conc',
    '10_Range_Conc': 'Conc',
    '11_ACA_qdPCR_multiplex': 'Conc',
    '12_ACA_qdPCR_multiplex_balanced': 'Conc',
}

FILE_MAPPING_ONE_TO_ONE = {
    '1_area': 'area_strategy.csv',
    '2_range': 'range_strategy.xlsx',
    '3_range_filtered': 'range_strategy_filtered.xlsx',
}
FILE_TARGET = {
    '01_ACA_qdPCR':   'LoadedPanels',
    '03_AMCA_qdPCR':  'Target',
    '02_AMCA_qdLAMP': 'Target',
    '04_Z_area':   'Target',
    '05_Z_range':  'Target',
    '06_Z_range_filtered': 'Target',
    '07_ACA_qdPCR_conc_filtered': 'LoadedPanels',
    '08_AMCA_qdPCR_conc_filtered': 'Target',
    '09_Area_Conc': 'Target',
    '10_Range_Conc': 'Target',
    '11_ACA_qdPCR_multiplex': 'LoadedPanels',
    '12_ACA_qdPCR_multiplex_balanced': 'LoadedPanels',
}
FILE_TARGET_ONE_TO_ONE = {
    '1_area':   'Target',
    '2_range':  'Target',
    '3_range_filtered': 'Target',
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


def _read_table(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(file_path)
    return pd.read_csv(file_path)


def load_lab_curves(exp_path):
    folder = exp_path.name
    file_path = os.path.join(exp_path, FILE_MAPPING[folder])
    df = _read_table(file_path)

    cycle_cols = [col for col in df.columns
                  if str(col).startswith('Cycle') or re.match(r'^\d+(\.\d+)?$', str(col))]
    sorted_cycle_cols = sorted(cycle_cols, key=get_numeric_sort_key)

    timestamps = np.array([get_numeric_sort_key(col) for col in sorted_cycle_cols])
    curves = df[sorted_cycle_cols].to_numpy()
    well_labels = df[FILE_TARGET[folder]].to_numpy()

    conc_col = FILE_CONC.get(folder)
    concentration = df[conc_col].to_numpy() if conc_col else None

    return curves, timestamps, well_labels, concentration


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
        "sigmoid_results": sigmoid_results, "concentration": None,
    }
    safe_joblib_dump(cached, cache_path, compress=3)
    return cached


def load_lab_curves_one_to_one(exp_folder, strategy, label1, label2):
    cached = load_and_fit_strategy_once(exp_folder, strategy)
    mask = np.isin(cached["well_labels"], [label1, label2])

    curves = cached["curves"][mask]
    well_labels = cached["well_labels"][mask]
    fitted_full, fitted_stretched, params, rmse = cached["sigmoid_results"]
    sigmoid_results = (fitted_full[mask], fitted_stretched[mask], params[mask], rmse[mask])

    return curves, cached["timestamps"], well_labels, sigmoid_results, cached.get("concentration")


def is_cache_hit(save_path):
    if not os.path.exists(save_path):
        return False
    try:
        existing_data = joblib.load(save_path)
    except Exception as e:
        print(f"  -> [WARNING] Cached file at {save_path} is corrupted ({e}). Recomputing from scratch...")
        return False
    return existing_data is not None and "curves" in existing_data


def save_experiment_data(save_exp_path, curves, timestamps, well_labels, sigmoid_results, concentration=None):
    fitted_full, fitted_stretched, params, rmse = sigmoid_results

    # LAB curves have no pixel-grid metadata (flat per-sample data) -- an empty-column,
    # correctly-indexed DataFrame keeps build_kinetic_features' row-aligned concat working.
    metadata_df = pd.DataFrame(index=range(len(curves)))
    kinetic_features = [build_kinetic_features(curves, timestamps, metadata_df)]

    save_data = {
        "curves": {"ori_curves": curves},
        "dataset_name": ["ori_curves"],
        "dataset": [curves],
        "kinetic_features": kinetic_features,
        "Y_well": well_labels,
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
        "concentration": concentration,
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
    existing_state.update(save_data)
    safe_joblib_dump(existing_state, save_path, compress=3)
    print(f"  -> Saved numerical results and metadata to {save_path}")


def main(argv=None):
    print(f"\n{'='*70}\n[RUNNING] lab/preprocessing.py\n{'='*70}\n")
    parser = argparse.ArgumentParser(description="LAB (LAB_DDM_paper) curve preprocessing")
    parser.add_argument("--task_id", type=int, default=0, help="Array Job ID")
    parser.add_argument("--exp_folder", type=str, default=config.LAB_EXP_FOLDER)
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--one_to_one", action="store_true",
                        help="Pairwise one-to-one mode (use with --exp_folder=config.LAB_1TO1_EXP_FOLDER): "
                             "build one curve_for_training.joblib per (strategy, label-pair) combination.")
    args = parser.parse_args(argv)

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
            print(f"Cache hit: {exp_path}")
            print("  ✓ Experiment complete!\n")
            sys.exit(0)

        print(f"Processing One-to-One Combo: {combo_name}  ({strategy}: {label1} vs {label2})")
        curves, timestamps, well_labels, sigmoid_results, concentration = load_lab_curves_one_to_one(
            args.exp_folder, strategy, label1, label2)
        save_experiment_data(exp_path, curves, timestamps, well_labels, sigmoid_results, concentration=concentration)

        print(f"  -> X (Curves) shape:       {curves.shape}")
        print(f"  -> well_labels (Targets):  {np.unique(well_labels)}")
        print(f"  -> Mean RMSE Fit Error:    {np.nanmean(sigmoid_results[3]):.4f}")
        print("  ✓ Experiment complete!\n")
        return

    exp_paths = sorted([Path(args.exp_folder, name) for name in os.listdir(args.exp_folder)
                        if os.path.isdir(os.path.join(args.exp_folder, name)) and name in FILE_MAPPING])

    if args.task_id >= len(exp_paths):
        print(f"Task ID {args.task_id} is out of bounds for {len(exp_paths)} folders. Exiting.")
        sys.exit(0)

    exp_path = exp_paths[args.task_id]
    save_path = os.path.join(exp_path, config.TRAINING_DATA_PATH)

    if is_cache_hit(save_path) and not args.force_rerun:
        print(f"Cache hit: {exp_path}")
        print("  ✓ Experiment complete!\n")
        return

    print(f"Processing Experiment: {exp_path}")
    curves, timestamps, well_labels, concentration = load_lab_curves(exp_path)

    print("  -> Processing Sigmoid Curves...")
    sigmoid_results = sigmoid_fitting_5p(curves, timestamps)
    save_experiment_data(exp_path, curves, timestamps, well_labels, sigmoid_results, concentration=concentration)
    print(f"  -> X (Curves) shape:       {curves.shape}")
    print(f"  -> well_labels (Targets):  {np.unique(well_labels)}")
    print(f"  -> Mean RMSE Fit Error:    {np.nanmean(sigmoid_results[3]):.4f}")
    print("  ✓ Experiment complete!\n")


if __name__ == "__main__":
    main()
