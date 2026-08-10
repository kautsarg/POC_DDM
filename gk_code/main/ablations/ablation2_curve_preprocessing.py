"""
Ablation 2: Curve Preprocessing
cnn_gru_dual vs cnn_gru_dual_attn_recon, each with/without negative-control (nc)
subtraction preprocessing (see 01_curve_preprocessing_v6.py's --nc_subtract).

Phase-2-only: nc_subtract needs the VREF-slice/NC-well structure that only
POC_DDM_final-style data has (LAB_DDM_paper has none -- see 01b_lab_curve_preprocessing.py,
metadata={} for LAB data).
task_id 0 = LAB_DDM_paper: no-op (nothing to compare, printed and exits).
task_id 1 = every POC_DDM_final subfolder: the plain half always runs; the
nc_subtract half runs wherever a matching POC_DDM_final_nc_subtract sibling
(f"{exp_folder}_nc_subtract", same subfolder name -- the convention
01_curve_preprocessing_v6.py writes it under) exists, skipped with a warning otherwise.
One task now loops over every subfolder internally, rather than one SLURM array
task per subfolder.

Each half's results go to its own file under <plain_exp_path>/ablations/:
  ablation2_curve_preprocessing_plain_performances.joblib
  ablation2_curve_preprocessing_nc_subtract_performances.joblib
Same results_dict[filter] schema as classification_performances.joblib, kept
fully separate from the main pipeline's results.
"""
import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# Absolute paths -- robust regardless of the CWD/invocation style, unlike a bare
# sys.path.insert(0, 'utils') (which only resolves if CWD happens to be main/).
_MAIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_MAIN_DIR))
sys.path.insert(0, str(_MAIN_DIR / "utils"))
sys.path.insert(0, str(_MAIN_DIR / "utils" / "model_training"))

from safe_io import safe_joblib_dump
from pipeline_utils import get_exp_paths
from model_utils import evaluate_outlier_filters, plot_ml_results, set_global_determinism

import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.keras.mixed_precision.set_global_policy('mixed_float16')
tf.config.optimizer.set_jit(True)

MODELS = ['cnn_gru_dual', 'cnn_gru_dual_attn_recon']


def load_training_data(exp_path):
    data_path = os.path.join(exp_path, config.TRAINING_DATA_PATH)
    if not os.path.exists(data_path):
        print(f"  [SKIP] {exp_path.name}: '{data_path}' not found.")
        return None
    return joblib.load(data_path)


def filter_datasets(dataset_name, dataset, kinetic_features):
    filtered_names, filtered_dataset, filtered_features = [], [], []
    for name, data, features in zip(dataset_name, dataset, kinetic_features):
        if not name.startswith("avg_") and not name.startswith("original_fitted_stretched"):
            filtered_names.append(name)
            filtered_dataset.append(data)
            filtered_features.append(features)
    return filtered_names, filtered_dataset, filtered_features


def load_spatial_metadata(training_data, Y_well):
    if "metadata" not in training_data:
        print("  [*] No 'metadata' in training data -- cnn_gru_dual_attn_recon will be skipped.")
        return None, None
    metadata_df = pd.DataFrame(training_data["metadata"])
    if not {"pixel_row_idx", "pixel_col_idx"}.issubset(metadata_df.columns):
        print("  [*] No pixel_row_idx/pixel_col_idx in metadata -- "
              "cnn_gru_dual_attn_recon will be skipped.")
        return None, None
    coords = np.stack([
        metadata_df["pixel_row_idx"].values.astype(float),
        metadata_df["pixel_col_idx"].values.astype(float),
    ], axis=1)
    well_ids = (metadata_df["well_id"].values if "well_id" in metadata_df.columns
                else np.array(Y_well).copy())
    return coords, well_ids


def run_variant(exp_path, curve_type, n_splits, batch_size, tag, out_dir, outlier_filter=None):
    """Load + train MODELS for one preprocessing variant. Returns None if the
    dataset folder / training data doesn't exist (caller decides whether that's fatal)."""
    training_data = load_training_data(exp_path)
    if training_data is None:
        return None

    dataset_name = training_data["dataset_name"]
    dataset = training_data["dataset"]
    kinetic_features = training_data["kinetic_features"]
    Y_well = training_data["Y_well"]

    coords_full, well_ids_full = load_spatial_metadata(training_data, Y_well)
    dataset_name, dataset, kinetic_features = filter_datasets(dataset_name, dataset, kinetic_features)

    label_mappings = config.get_label_mappings(exp_path)
    if exp_path.name in label_mappings:
        mapping = label_mappings[exp_path.name]
        Y_well = [mapping.get(w, w) for w in Y_well]

    encoder = LabelEncoder()
    y_full = encoder.fit_transform(Y_well)

    target_name = config.CURVE_TYPE_ALIASES.get(curve_type, curve_type)
    if target_name not in dataset_name:
        print(f"  [!] curve_type '{curve_type}' (-> '{target_name}') not found for "
              f"{tag} at {exp_path} -- skipping this half.")
        return None
    idx = dataset_name.index(target_name)
    curves_2d = dataset[idx]
    features_df = kinetic_features[idx]
    clean_title = dataset_name[idx].replace("_", " ").title()

    print(f"  [*] Ablation 2 [{tag}]: training {MODELS} on curve_type='{curve_type}' "
          f"outlier_filter={outlier_filter!r} ({len(y_full)} samples)")

    # Separate from the main pipeline's exp_path/model_interpretation, and separate
    # per preprocessing variant (plain vs nc_subtract) since both use the same model keys.
    model_interp_dir = out_dir / "model_interpretation" / tag
    # Filter-suffixed filename so a filtered run doesn't clobber the baseline
    # (outlier_filter=None) results -- default case keeps the original clean name.
    _suffix = f"_{outlier_filter}" if outlier_filter else ""

    results_dict = evaluate_outlier_filters(
        X_curves=curves_2d,
        features_df=features_df,
        y_encoded=y_full,
        outlier_filters=[outlier_filter],
        dataset_name=clean_title,
        mode_name=f"ablation2_curve_preprocessing_{tag}",
        models=MODELS,
        n_splits=n_splits,
        coords=coords_full,
        well_ids=well_ids_full,
        save_model_dir=model_interp_dir,
        save_model_curve_type=curve_type,
        batch_size=batch_size,
    )

    results_file_path = out_dir / f"ablation2_curve_preprocessing_{tag}{_suffix}_performances.joblib"
    safe_joblib_dump(results_dict, results_file_path, compress=3)
    print(f"  [+] Saved ablation 2 [{tag}] results -> {results_file_path}")

    plot_ml_results(
        results_dict=results_dict,
        outlier_filters=[outlier_filter],
        dataset_name=clean_title,
        mode_name=f"Ablation 2 [{tag}]: Curve Preprocessing",
        total_count=len(y_full),
        save_prefix=str(out_dir / f"{dataset_name[idx]}_ablation2_{tag}{_suffix}"),
    )
    return results_dict


def run_one(exp_path, exp_folder, curve_type, n_splits, batch_size, outlier_filter=None):
    out_dir = exp_path / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_variant(exp_path, curve_type, n_splits, batch_size, "plain", out_dir, outlier_filter)

    # nc_subtract sibling, same subfolder name under <exp_folder>_nc_subtract
    nc_exp_folder = exp_folder.rstrip('/') + '_nc_subtract'
    if not os.path.isdir(nc_exp_folder):
        print(f"  [!] nc_subtract sibling folder not found: {nc_exp_folder} "
              f"-- skipping the nc_subtract half (this dataset likely hasn't been "
              f"run through 01_curve_preprocessing_v6.py --nc_subtract yet).")
        return
    nc_exp_paths = get_exp_paths(nc_exp_folder)
    nc_exp_path = next((p for p in nc_exp_paths if p.name == exp_path.name), None)
    if nc_exp_path is None:
        print(f"  [!] No subfolder named '{exp_path.name}' under {nc_exp_folder} "
              f"-- skipping the nc_subtract half.")
        return
    run_variant(nc_exp_path, curve_type, n_splits, batch_size, "nc_subtract", out_dir, outlier_filter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation 2: Curve Preprocessing (nc_subtract)")
    parser.add_argument("--task_id", type=int, default=1, choices=[0, 1],
                        help="0 = LAB_DDM_paper (no-op -- nc_subtract needs POC_DDM_final-style "
                             "chip metadata LAB data doesn't have), 1 = POC_DDM_final (every "
                             "subfolder; nc_subtract half runs wherever a matching "
                             "POC_DDM_final_nc_subtract sibling exists). One task loops over "
                             "every subfolder internally.")
    parser.add_argument("--exp_folder", type=str, default=None,
                        help="Override the task_id -> exp_folder mapping (plain half; the "
                             "nc_subtract sibling is still derived automatically). Runs every "
                             "subfolder under it.")
    parser.add_argument("--curve_type", type=str, default="ori_curve")
    parser.add_argument("--n_splits", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--outlier_filter", type=str, default=None,
                        choices=['None', 'lstm_ae_glb_ds1_label_elbow',
                                 'spatial_grid_label_elbow', 'spatial_knn_label_elbow'],
                        help="Outlier filter to train on -- must already be a boolean "
                             "column in features_df, computed by "
                             "02_outlier_detection_pipeline.py. Default: no filtering.")
    parser.add_argument("--fast_mode", action="store_true",
                        help="Disable strict TF determinism for faster training.")
    args = parser.parse_args()
    outlier_filter = None if args.outlier_filter in (None, 'None') else args.outlier_filter

    print(f"\n{'='*70}\n[RUNNING] ablation2_curve_preprocessing.py\n{'='*70}\n")
    set_global_determinism(0, strict=not args.fast_mode)

    if args.task_id == 0 and args.exp_folder is None:
        print("  [SKIP] task_id=0 (LAB_DDM_paper) has no nc_subtract data -- nothing to run. "
              "Use task_id=1 for POC_DDM_final.")
        sys.exit(0)

    exp_folder = args.exp_folder or config.FINAL_EXP_FOLDER
    exp_paths_to_run = get_exp_paths(exp_folder)

    print(f"\n{'#'*80}\nSTARTING ABLATION 2 -- task_id={args.task_id} -- "
          f"{len(exp_paths_to_run)} subfolder(s) under {exp_folder}\n{'#'*80}")

    for exp_path in exp_paths_to_run:
        print(f"\n{'-'*70}\n[{exp_path.name}]\n{'-'*70}")
        try:
            run_one(exp_path, exp_folder, args.curve_type, args.n_splits, args.batch_size, outlier_filter)
        except Exception as e:
            print(f"  [ERROR] {exp_path.name} failed: {type(e).__name__}: {e}")

    print(f"\n{'='*70}\n[DONE] ablation2_curve_preprocessing.py -- task_id={args.task_id}\n{'='*70}\n")
