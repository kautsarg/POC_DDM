import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

_MAIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_MAIN_DIR))
sys.path.insert(0, str(_MAIN_DIR / "utils"))
sys.path.insert(0, str(_MAIN_DIR / "utils" / "model_training"))

from safe_io import safe_joblib_dump
from pipeline_utils import get_exp_paths
from model_utils import evaluate_outlier_filters, plot_ml_results, set_global_determinism
from model_utils_mtl import REG_SENTINEL as _MTL_REG_SENTINEL

import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.keras.mixed_precision.set_global_policy('mixed_float16')
tf.config.optimizer.set_jit(True)

MODELS = [
    'cnn_gru_dual', 'cnn_gru_dual_attn_recon',
    'cnn_gru_dual_mtl', 'cnn_gru_dual_attn_recon_mtl',
    'gru_rcfd_cgd', 'gru_rcfd_attn_recon',
    'cnn_gru_dual_supcon3', 'cnn_gru_dual_attn_recon_supcon3',
]

TASK_EXP_FOLDERS = {0: config.LAB_EXP_FOLDER, 1: config.FINAL_EXP_FOLDER}
TASK_SUBFOLDERS = {
    0: ['01_ACA_qdPCR', '02_AMCA_qdLAMP', '03_AMCA_qdPCR', '09_Area_Conc', '10_Range_Conc'],
    1: None,
}


def load_training_data(exp_path):
    data_path = os.path.join(exp_path, config.TRAINING_DATA_PATH)
    if not os.path.exists(data_path):
        print(f"  [SKIP] {exp_path.name}: '{data_path}' not found.")
        return None
    return joblib.load(data_path)


def load_or_init_results(results_file_path):
    if os.path.exists(results_file_path):
        try:
            return joblib.load(results_file_path)
        except Exception as e:
            print(f"  [WARNING] Results file corrupt ({e}), starting fresh: {results_file_path}")
    return {}


def make_checkpoint_fn(results_file_path):
    def _checkpoint(updated_results):
        safe_joblib_dump(updated_results, results_file_path, compress=3)
    return _checkpoint


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
        print("  [*] No 'metadata' in training data -- *_attn_recon* models will be skipped.")
        return None, None
    metadata_df = pd.DataFrame(training_data["metadata"])
    if not {"pixel_row_idx", "pixel_col_idx"}.issubset(metadata_df.columns):
        print("  [*] No pixel_row_idx/pixel_col_idx in metadata -- "
              "*_attn_recon* models will be skipped.")
        return None, None
    coords = np.stack([
        metadata_df["pixel_row_idx"].values.astype(float),
        metadata_df["pixel_col_idx"].values.astype(float),
    ], axis=1)
    well_ids = (metadata_df["well_id"].values if "well_id" in metadata_df.columns
                else np.array(Y_well).copy())
    return coords, well_ids


def load_concentration(training_data, Y_well):
    """Regression target for the MTL/RCFD heads in MODELS. Sentinel-masked exactly
    like 03_main_training.py's --mtl path: missing/zero concentration (negative
    controls, log10 undefined) -> REG_SENTINEL."""
    raw_conc = training_data.get("concentration", None)
    if raw_conc is not None:
        arr = np.asarray(raw_conc, dtype=object)
        float_arr = np.array([float(v) if v is not None else np.nan for v in arr], dtype=float)
        y_concentration = np.where(np.isnan(float_arr) | (float_arr == 0.0),
                                    _MTL_REG_SENTINEL, float_arr)
        n_zero = int((float_arr == 0.0).sum())
    else:
        y_concentration = np.full(len(Y_well), _MTL_REG_SENTINEL, dtype=float)
        n_zero = 0
    n_valid = int((y_concentration != _MTL_REG_SENTINEL).sum())
    print(f"  [MTL] Concentration loaded: {n_valid} / {len(y_concentration)} samples "
          f"have non-sentinel concentration"
          + (f" ({n_zero} zero/negative-control masked)." if n_zero else "."))
    return y_concentration


def run_one(exp_path, curve_type, n_splits, batch_size, outlier_filter=None, force_rerun=False):
    out_dir = exp_path / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Filter-suffixed filename so a filtered run doesn't clobber the baseline
    # (outlier_filter=None) results -- default case keeps the original clean name.
    _suffix = f"_{outlier_filter}" if outlier_filter else ""
    results_file_path = out_dir / f"ablation3_generalisability{_suffix}_performances.joblib"

    cached_results = {} if force_rerun else load_or_init_results(results_file_path)
    if cached_results:
        print(f"  [*] Found existing results at {results_file_path} -- "
              f"already-trained models will be skipped (use --force_rerun to retrain).")
    # Separate from the main pipeline's exp_path/model_interpretation -- keeps
    # ablation XAI models isolated from 03_main_training.py's own saved models.
    model_interp_dir = out_dir / "model_interpretation"

    training_data = load_training_data(exp_path)
    if training_data is None:
        return

    dataset_name = training_data["dataset_name"]
    dataset = training_data["dataset"]
    kinetic_features = training_data["kinetic_features"]
    Y_well = training_data["Y_well"]

    coords_full, well_ids_full = load_spatial_metadata(training_data, Y_well)
    y_concentration = load_concentration(training_data, Y_well)

    dataset_name, dataset, kinetic_features = filter_datasets(dataset_name, dataset, kinetic_features)

    label_mappings = config.get_label_mappings(exp_path)
    if exp_path.name in label_mappings:
        mapping = label_mappings[exp_path.name]
        Y_well = [mapping.get(w, w) for w in Y_well]

    encoder = LabelEncoder()
    y_full = encoder.fit_transform(Y_well)

    target_name = config.CURVE_TYPE_ALIASES.get(curve_type, curve_type)
    if target_name not in dataset_name:
        print(f"  [SKIP] curve_type '{curve_type}' (-> '{target_name}') not found in "
              f"this dataset's curve variants: {dataset_name}")
        return
    idx = dataset_name.index(target_name)
    curves_2d = dataset[idx]
    features_df = kinetic_features[idx]
    clean_title = dataset_name[idx].replace("_", " ").title()

    print(f"  [*] Ablation 3: training {MODELS} on curve_type='{curve_type}' "
          f"outlier_filter={outlier_filter!r} ({len(y_full)} samples)")

    results_dict = evaluate_outlier_filters(
        X_curves=curves_2d,
        features_df=features_df,
        y_encoded=y_full,
        outlier_filters=[outlier_filter],
        dataset_name=clean_title,
        mode_name="ablation3_generalisability",
        models=MODELS,
        n_splits=n_splits,
        coords=coords_full,
        well_ids=well_ids_full,
        multitask=True,
        y_concentration=y_concentration,
        save_model_dir=model_interp_dir,
        save_model_curve_type=curve_type,
        batch_size=batch_size,
        cached_results=cached_results,
        checkpoint_fn=make_checkpoint_fn(results_file_path),
        rerun_models=config.RERUN_MODELS,
    )

    safe_joblib_dump(results_dict, results_file_path, compress=3)
    print(f"  [+] Saved ablation 3 results -> {results_file_path}")

    plot_ml_results(
        results_dict=results_dict,
        outlier_filters=[outlier_filter],
        dataset_name=clean_title,
        mode_name="Ablation 3: Generalisability",
        total_count=len(y_full),
        save_prefix=str(out_dir / f"{dataset_name[idx]}_ablation3{_suffix}"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation 3: Generalisability")
    parser.add_argument("--task_id", type=int, default=0, choices=[0, 1],
                        help="0 = LAB_DDM_paper (folders 01,02,03,09,10), "
                             "1 = POC_DDM_final (every subfolder). One task loops over "
                             "every subfolder for its phase internally.")
    parser.add_argument("--exp_folder", type=str, default=None,
                        help="Override the task_id -> exp_folder mapping (runs every "
                             "subfolder under it).")
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
    parser.add_argument("--force_rerun", action="store_true",
                        help="Ignore any existing <results>.joblib checkpoint and retrain "
                             "every model from scratch instead of skipping cached ones.")
    args = parser.parse_args()
    outlier_filter = None if args.outlier_filter in (None, 'None') else args.outlier_filter

    print(f"\n{'='*70}\n[RUNNING] ablation3_generalisability.py\n{'='*70}\n")
    set_global_determinism(0, strict=not args.fast_mode)

    exp_folder = args.exp_folder or TASK_EXP_FOLDERS[args.task_id]
    all_exp_paths = get_exp_paths(exp_folder)
    wanted_names = TASK_SUBFOLDERS.get(args.task_id) if args.exp_folder is None else None
    if wanted_names is None:
        exp_paths_to_run = all_exp_paths
    else:
        by_name = {p.name: p for p in all_exp_paths}
        missing = [n for n in wanted_names if n not in by_name]
        if missing:
            print(f"  [WARN] Missing subfolders under {exp_folder}: {missing}")
        exp_paths_to_run = [by_name[n] for n in wanted_names if n in by_name]

    print(f"\n{'#'*80}\nSTARTING ABLATION 3 -- task_id={args.task_id} -- "
          f"{len(exp_paths_to_run)} subfolder(s) under {exp_folder}\n{'#'*80}")

    for exp_path in exp_paths_to_run:
        print(f"\n{'-'*70}\n[{exp_path.name}]\n{'-'*70}")
        try:
            run_one(exp_path, args.curve_type, args.n_splits, args.batch_size, outlier_filter,
                    args.force_rerun)
        except Exception as e:
            print(f"  [ERROR] {exp_path.name} failed: {type(e).__name__}: {e}")

    print(f"\n{'='*70}\n[DONE] ablation3_generalisability.py -- task_id={args.task_id}\n{'='*70}\n")
