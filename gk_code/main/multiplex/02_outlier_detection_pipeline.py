"""Multiplex outlier detection pipeline.

Mirrors main/02 args and structure; adapted for flat-CSV lab data:
  - only 'ori_curves' variant (no per-well, no metadata, no avg/wavelet variants)
  - LSTM-AE: per_well=False  (no per-well encoder needed)
  - Spatial filters: listed for structural parity but skipped when no pixel
    coordinates are present in the dataset (lab flat-CSV has none)
"""
import os
import sys
import warnings
import gc
import argparse
from io import BytesIO
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
import tensorflow as tf

_THIS          = Path(__file__).resolve()
_MULTIPLEX_DIR = _THIS.parent
_MAIN_DIR      = _MULTIPLEX_DIR.parent

sys.path.insert(0, str(_MAIN_DIR))
sys.path.insert(0, str(_MAIN_DIR / "utils"))
sys.path.insert(0, str(_MAIN_DIR / "utils" / "02_outlier_detection"))
sys.path.insert(0, str(_MAIN_DIR / "utils" / "model_training"))
sys.path.insert(0, str(_MULTIPLEX_DIR))

import config as _base_config
sys.path.insert(0, str(_MULTIPLEX_DIR))
import config

from safe_io import safe_joblib_dump
import sigmoid_fitting as sp
from lstm_autoencoder_outlier import run_lstm_autoencoder_pipeline
from spatial_consistency_outlier import (
    run_spatial_consistency_knn_pipeline,
    run_spatial_consistency_grid_pipeline,
)
from model_utils import set_global_determinism
from pipeline_utils import get_exp_paths


# ======================================================================
# FILTER REGISTRY (mirrors main/02 ALL_FILTERS / _DEFAULT_FILTERS)
# ======================================================================

# Subset of main/02's ALL_FILTERS that are relevant to flat-CSV lab data.
ALL_FILTERS      = frozenset({"lstm_ae", "spatial_knn", "spatial_grid"})
# Spatial filters are listed but gracefully skipped (no pixel coordinates).
_DEFAULT_FILTERS = frozenset({"lstm_ae"})


# ======================================================================
# KINETIC FEATURE EXTRACTION (mirrors main/02's helpers)
# ======================================================================

def _process_single_row(y, X):
    valid = np.isfinite(X) & np.isfinite(y)
    if np.sum(valid) < 3:
        return {}
    try:
        return sp.extract_kinetic_parameters_original(X, y)
    except Exception:
        return {}


def extract_kinetic_features(timestamps, curves, n_jobs=-1):
    features = Parallel(n_jobs=n_jobs)(
        delayed(_process_single_row)(y, timestamps) for y in curves
    )
    return pd.DataFrame(features)


def get_send(timestamps, curves_2d, send_n=(5, 10, 15, 20, 25)):
    dy_dx_list = Parallel(n_jobs=-1, backend="loky", batch_size='auto')(
        delayed(sp.calculate_first_derivative)(timestamps, y) for y in curves_2d
    )
    dy_dx = np.array(dy_dx_list)
    send_dict = {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        for n in send_n:
            send_dict[f"send_{n}"]     = np.nanmean(dy_dx[:, -n:], axis=1)
            send_dict[f"send_abs_{n}"] = np.nanmean(np.abs(dy_dx[:, -n:]), axis=1)
    return send_dict


def build_kinetic_features(curves_2d, timestamps):
    """Kinetic parameters + Send derivatives (no metadata — flat lab data)."""
    features_df  = extract_kinetic_features(timestamps, curves_2d).reset_index(drop=True)
    add_features = get_send(timestamps, curves_2d)
    add_features["FFI"]     = curves_2d[:, -1]
    add_features["F_range"] = curves_2d[:, -1] - curves_2d[:, 0]
    return pd.concat(
        [features_df, pd.DataFrame(add_features).reset_index(drop=True)], axis=1)


# ======================================================================
# PIPELINE STATE HELPERS
# ======================================================================

def _load_or_build_state(data_path, force_rerun):
    """Load curve_for_training_ml.joblib; optionally clear 02's own keys."""
    if not os.path.exists(data_path):
        print(f"  -> [SKIP] {data_path} not found. Run 01b first.")
        sys.exit(0)

    data = joblib.load(data_path)

    if force_rerun:
        print("  -> [FORCE RERUN] Clearing 02's cached kinetic_features "
              "(curves/labels are kept). Recomputing...")
        data.pop('kinetic_features', None)

    return data


def _ensure_kinetic_features(data, folder, data_path):
    """Compute kinetic features for ori_curves if not yet cached."""
    curves     = data['curves']['ori_curves']
    timestamps = data['timestamps']

    kf_dict = data.get('kinetic_features', {})
    if folder in kf_dict and len(kf_dict[folder]) == len(curves):
        print(f"  -> [CACHED] Kinetic features ({len(kf_dict[folder])} rows).")
        return kf_dict[folder]

    print(f"  -> Extracting kinetic features ({len(curves)} curves)...")
    kf_df = build_kinetic_features(curves, timestamps)
    data.setdefault('kinetic_features', {})[folder] = kf_df
    safe_joblib_dump(data, data_path, compress=3)
    return kf_df


# ======================================================================
# OUTLIER PIPELINES
# ======================================================================

def _run_outlier_pipelines(data, kf_df, folder, exp_path, data_path,
                            filters=_DEFAULT_FILTERS):
    """Run applicable outlier sub-pipelines; save after each."""
    print("\n=== RUNNING OUTLIER DETECTION PIPELINES ===")

    curves      = data['curves']['ori_curves']
    timestamps  = data['timestamps']
    well_labels = data['well_labels']    # combo strings, e.g. "VIM_NDM"
    ds          = config.AE_DOWNSAMPLE_FACTOR

    # Flat lab data has no pixel coords → spatial filters gracefully skipped.
    has_spatial = False  # no pixel_row_idx / pixel_col_idx in CSV

    # ── LSTM-AE (Global) ────────────────────────────────────────────────
    if "lstm_ae" not in filters:
        print("  -> [DISABLED] LSTM AutoEncoder (Global): not in --filters.")
    else:
        ae_col = f'lstm_ae_glb_ds{ds}_label_elbow'
        if ae_col in kf_df.columns:
            print(f"  -> [SKIP] LSTM-AE: column already present ({ae_col}).")
        else:
            ae_plot_dir  = os.path.join(exp_path, "ae_outlier")
            encoder_dir  = os.path.join(exp_path, "pretrained_encoders")
            os.makedirs(ae_plot_dir, exist_ok=True)

            print(f"  -> Running LSTM-AE global outlier filter ({len(curves)} curves)...")
            extracted_dfs = run_lstm_autoencoder_pipeline(
                dataset_names=[folder],
                dataset_curves=[curves],
                Y_well=[well_labels],
                ref_curves=curves,
                ae_plot_path=ae_plot_dir,
                threshold_percentiles=['elbow'],
                save_plot=True,
                downsample_factor=ds,
                per_well=False,
                save_encoder_dir=encoder_dir,
            )
            ae_df = extracted_dfs[0]
            for col in ae_df.columns:
                kf_df[col] = ae_df[col].values
            data['kinetic_features'][folder] = kf_df
            safe_joblib_dump(data, data_path, compress=3)
            print(f"  -> LSTM-AE done. Added: {list(ae_df.columns)}")

    # ── Spatial KNN ─────────────────────────────────────────────────────
    if "spatial_knn" not in filters:
        print("  -> [DISABLED] Spatial Consistency (KNN): not in --filters.")
    elif not has_spatial:
        print("  -> [SKIP] Spatial Consistency (KNN): no pixel coordinates in flat-CSV lab data.")

    # ── Spatial Grid ─────────────────────────────────────────────────────
    if "spatial_grid" not in filters:
        print("  -> [DISABLED] Spatial Consistency (Grid): not in --filters.")
    elif not has_spatial:
        print("  -> [SKIP] Spatial Consistency (Grid): no pixel coordinates in flat-CSV lab data.")

    return kf_df


# ======================================================================
# MASTER PIPELINE
# ======================================================================

def run_pipeline(exp_path, force_rerun=False, filters=_DEFAULT_FILTERS):
    """End-to-end outlier detection pipeline for one multiplex experiment."""
    exp_path  = Path(exp_path)
    folder    = exp_path.name
    data_path = exp_path / config.TRAINING_DATA_PATH

    print(f"\n\n{'#'*80}\nSTARTING MULTIPLEX OUTLIER PIPELINE FOR: {folder}\n{'#'*80}")

    data  = _load_or_build_state(str(data_path), force_rerun)
    kf_df = _ensure_kinetic_features(data, folder, str(data_path))
    kf_df = _run_outlier_pipelines(data, kf_df, folder, str(exp_path), str(data_path),
                                    filters=filters)

    print(f"\n  -> {folder} finished gracefully.")
    tf.keras.backend.clear_session()
    gc.collect()


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    print(f"\n{'='*70}\n[RUNNING] {os.path.basename(__file__)}\n{'='*70}\n")
    parser = argparse.ArgumentParser(description="Multiplex Outlier Detection Pipeline")
    parser.add_argument("--task_id",    type=int, default=0, help="Array Job ID")
    parser.add_argument("--exp_folder", type=str, default=config.LAB_MULTIPLEX_FOLDER)
    parser.add_argument("--force_rerun", action="store_true",
                        help="Recompute and overwrite even if cached results exist")
    parser.add_argument("--fast_mode", action="store_true",
                        help="Disable strict TF determinism for faster AE training. "
                             "RNG seeds still set; reruns won't be bit-exact.")
    parser.add_argument("--filters", nargs="*",
                        choices=sorted(ALL_FILTERS),
                        default=[],
                        help="Which outlier filters to run. "
                             f"Available: {', '.join(sorted(ALL_FILTERS))}. "
                             "Default: none (omitting --filters runs no filter pipelines). "
                             "Pass specific names to run only those.")
    args = parser.parse_args()

    filters = set(args.filters or [])

    set_global_determinism(0, strict=not args.fast_mode)

    exp_folder = args.exp_folder
    subdirs    = sorted([d for d in Path(exp_folder).iterdir() if d.is_dir()])
    task_dirs  = [d for d in subdirs if d.name in config.FILE_MAPPING]

    if not task_dirs:
        print(f"No valid task directories found in {exp_folder}.")
        sys.exit(0)

    exp_path = task_dirs[args.task_id % len(task_dirs)]

    run_pipeline(exp_path, force_rerun=args.force_rerun, filters=filters)
