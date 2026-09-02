import os
import sys
import re
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
from pipeline_utils import get_scoped_exp_paths

FILE_MAPPING = {
    '01_ACA_qdPCR':   'dPCR_Dataset_Pure.csv',
    '02_AMCA_qdLAMP': '5Plex_dLAMP_oldData.csv',
    '03_AMCA_qdPCR':  'dPCR_Amplification_Curves.csv',
    '12_ACA_qdPCR_multiplex_balanced': 'dPCR_Dataset_Mixture.csv',
}
FILE_CONC = {
    '01_ACA_qdPCR': 'Conc',
    '03_AMCA_qdPCR':  'Conc',
    '02_AMCA_qdLAMP': 'Conc',
    '12_ACA_qdPCR_multiplex_balanced': 'Conc',
}
FILE_TARGET = {
    '01_ACA_qdPCR':   'LoadedPanels',
    '03_AMCA_qdPCR':  'Target',
    '02_AMCA_qdLAMP': 'Target',
    '12_ACA_qdPCR_multiplex_balanced': 'LoadedPanels',
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
    args = parser.parse_args(argv)

    exp_paths = get_scoped_exp_paths(args.exp_folder, [n for n in config.LAB_DATASETS_IN_SCOPE if n in FILE_MAPPING])

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
