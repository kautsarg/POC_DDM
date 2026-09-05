import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "utils"))
sys.path.insert(0, str(_ROOT / "utils" / "model_training"))

import config
from safe_io import safe_joblib_dump
from pipeline_utils import get_scoped_exp_paths, check_task_id

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.keras.mixed_precision.set_global_policy('mixed_float16')
tf.config.optimizer.set_jit(True)

from model_utils import evaluate_outlier_filters, plot_ml_results, set_global_determinism

MODELS = ['cnn_gru_dual', 'cnn_gru_dual_attn_recon']
OUTLIER_FILTERS = [None]
DEFAULT_CURVE_TYPE = "ori_curve_sg_p4_norm"


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
        return None, None
    metadata_df = pd.DataFrame(training_data["metadata"])
    if not {"pixel_row_idx", "pixel_col_idx"}.issubset(metadata_df.columns):
        return None, None
    coords = np.stack([
        metadata_df["pixel_row_idx"].values.astype(float),
        metadata_df["pixel_col_idx"].values.astype(float),
    ], axis=1)
    well_ids = (metadata_df["well_id"].values if "well_id" in metadata_df.columns
                else np.array(Y_well).copy())
    return coords, well_ids


def resolve_available_filters(outlier_filters, features_df):
    available = []
    for f in outlier_filters:
        if f is None or f in features_df.columns:
            available.append(f)
        else:
            print(f"  [!] outlier_filter '{f}' not found -- skipping.")
    return available


def run_one(exp_path, curve_type, n_splits, batch_size, force_rerun=False):
    out_dir = exp_path / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file_path = out_dir / "ablation6_chip_outlier_model_ablation_performances.joblib"

    cached_results = {} if force_rerun else load_or_init_results(results_file_path)
    model_interp_dir = out_dir / "model_interpretation"

    training_data = load_training_data(exp_path)
    if training_data is None:
        return

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
        print(f"  [SKIP] curve_type '{curve_type}' (-> '{target_name}') not found: {dataset_name}")
        return
    idx = dataset_name.index(target_name)
    curves_2d = dataset[idx]
    features_df = kinetic_features[idx]
    clean_title = dataset_name[idx].replace("_", " ").title()

    outlier_filters = resolve_available_filters(OUTLIER_FILTERS, features_df)
    if not outlier_filters:
        print(f"  [SKIP] {exp_path.name}: none of {OUTLIER_FILTERS} are available.")
        return

    results_dict = evaluate_outlier_filters(
        X_curves=curves_2d,
        features_df=features_df,
        y_encoded=y_full,
        outlier_filters=outlier_filters,
        dataset_name=clean_title,
        mode_name="ablation6_chip_outlier_model_ablation",
        models=MODELS,
        n_splits=n_splits,
        coords=coords_full,
        well_ids=well_ids_full,
        save_model_dir=model_interp_dir,
        save_model_curve_type=curve_type,
        batch_size=batch_size,
        cached_results=cached_results,
        checkpoint_fn=make_checkpoint_fn(results_file_path),
    )

    safe_joblib_dump(results_dict, results_file_path, compress=3)
    print(f"  [+] Saved ablation 6 results -> {results_file_path}")

    plot_ml_results(
        results_dict=results_dict,
        outlier_filters=outlier_filters,
        dataset_name=clean_title,
        mode_name="Ablation 6: Chip x Outlier-Detection x Model",
        total_count=len(y_full),
        save_prefix=str(out_dir / f"{dataset_name[idx]}_ablation6"),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ablation 6: Chip x Outlier-Detection x Model")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--exp_folder", type=str, default=config.DEFAULT_EXP_FOLDER)
    parser.add_argument("--curve_type", type=str, default=DEFAULT_CURVE_TYPE)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--force_rerun", action="store_true")
    args = parser.parse_args(argv)

    print(f"\n{'='*70}\n[RUNNING] chip/intra_chip_training.py\n{'='*70}\n")
    set_global_determinism(0, strict=True)

    exp_paths = get_scoped_exp_paths(args.exp_folder, config.CROSS_DATASET_GROUPS['final_6_new'])
    check_task_id(args.task_id, exp_paths)
    exp_path = exp_paths[args.task_id]

    print(f"\n{'#'*80}\nSTARTING ABLATION 6 -- task_id={args.task_id} -- {exp_path.name}\n{'#'*80}")
    run_one(exp_path, args.curve_type, args.n_splits, args.batch_size, args.force_rerun)


if __name__ == "__main__":
    main()
