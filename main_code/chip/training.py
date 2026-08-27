import os
import sys
import gc
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "utils"))
sys.path.insert(0, str(_ROOT / "utils" / "model_training"))

import config
from safe_io import safe_joblib_dump
from pipeline_utils import get_exp_paths, check_task_id

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.keras.mixed_precision.set_global_policy('mixed_float16')
tf.config.optimizer.set_jit(True)

from model_utils import evaluate_outlier_filters, plot_ml_results, set_global_determinism

MODEL_CHOICES = ["kNN", "CNN", "BiGRU", "Transformer", "cnn_gru_dual", "cnn_gru_dual_attn_recon"]
_MODEL_KEY = {
    "kNN": "knn", "CNN": "cnn", "BiGRU": "gru", "Transformer": "transformer",
    "cnn_gru_dual": "cnn_gru_dual", "cnn_gru_dual_attn_recon": "cnn_gru_dual_attn_recon",
}

K_NEIGHBORS = 24


def load_training_data(exp_path):
    data_path = os.path.join(exp_path, config.TRAINING_DATA_PATH)
    if not os.path.exists(data_path):
        print(f"  -> Skipping {exp_path.name}: '{data_path}' not found.")
        sys.exit(0)
    training_data = joblib.load(data_path)
    return config.apply_well_exclusion(training_data, exp_path.name)


def filter_datasets(dataset_name, dataset, kinetic_features):
    filtered_names, filtered_dataset, filtered_features = [], [], []
    for name, data, features in zip(dataset_name, dataset, kinetic_features):
        if not name.startswith("avg_") and not name.startswith("original_fitted_stretched"):
            filtered_names.append(name)
            filtered_dataset.append(data)
            filtered_features.append(features)
    return filtered_names, filtered_dataset, filtered_features


def load_or_init_results(results_file_path):
    if os.path.exists(results_file_path):
        try:
            return joblib.load(results_file_path)
        except Exception as e:
            print(f"  -> [WARNING] Results file corrupt ({e}), starting fresh: {results_file_path}")
    return {}


def make_checkpoint_fn(all_ml_results, results_file_path, clean_title, mode_key):
    def _checkpoint(updated_results):
        all_ml_results[clean_title][mode_key] = updated_results
        safe_joblib_dump(all_ml_results, results_file_path, compress=3)
    return _checkpoint


def _clear_cached_results(all_ml_results, clean_title, models):
    for m in models:
        if m not in config.MODEL_KEY_MAP:
            continue
        pk, pbk, ck = config.MODEL_KEY_MAP[m]
        mode = all_ml_results.get(clean_title, {}).get("Native", {})
        for res in mode.values():
            if isinstance(res, dict):
                for k in (pk, pbk, ck, f'train_history_{m}_'):
                    res.pop(k, None)


def main(argv=None):
    parser = argparse.ArgumentParser(description="CHIP per-experiment training")
    parser.add_argument("--task_id", type=int, default=0, help="Array Job ID")
    parser.add_argument("--exp_folder", type=str, default=config.DEFAULT_EXP_FOLDER)
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--curve_type", type=str, nargs="+",
                        choices=list(config.CURVE_TYPE_ALIASES.keys()),
                        default=list(config.CURVE_TYPE_ALIASES.keys()))
    parser.add_argument("--models", type=str, nargs="+", choices=MODEL_CHOICES, default=MODEL_CHOICES)
    parser.add_argument("--outlier_filter", type=str, nargs="+",
                        choices=["none"] + [f for f in config.OUTLIER_FILTERS if f is not None],
                        default=["none"])
    args = parser.parse_args(argv)

    print(f"\n{'='*70}\n[RUNNING] chip/training.py\n{'='*70}\n")

    set_global_determinism(0, strict=True)

    exp_paths = get_exp_paths(args.exp_folder)
    check_task_id(args.task_id, exp_paths)
    exp_path = exp_paths[args.task_id]
    print(f"\n\n{'#'*80}\nSTARTING TRAINING FOR: {exp_path.name}\n{'#'*80}")

    if args.n_splits > 1:
        results_file_path = os.path.join(exp_path, config.TRAINING_10FOLD_RESULT_PATH)
    else:
        results_file_path = os.path.join(exp_path, config.TRAINING_RESULT_PATH)
    model_plot_path = os.path.join(exp_path, "model_performance")
    os.makedirs(model_plot_path, exist_ok=True)

    training_data = load_training_data(exp_path)

    dataset_name = training_data["dataset_name"]
    dataset = training_data["dataset"]
    kinetic_features = training_data["kinetic_features"]
    Y_well = training_data["Y_well"]

    coords_full, well_ids_full = None, None
    if "metadata" in training_data:
        metadata_df = pd.DataFrame(training_data["metadata"])
        if {"pixel_row_idx", "pixel_col_idx"}.issubset(metadata_df.columns):
            coords_full = np.stack([
                metadata_df["pixel_row_idx"].values.astype(float),
                metadata_df["pixel_col_idx"].values.astype(float),
            ], axis=1)
            well_ids_full = (metadata_df["well_id"].values if "well_id" in metadata_df.columns
                              else np.array(Y_well).copy())
        else:
            print("  [*] No pixel_row_idx/pixel_col_idx in metadata -- cnn_gru_dual_attn_recon will be skipped.")
    else:
        print("  [*] No 'metadata' in training data -- cnn_gru_dual_attn_recon will be skipped.")

    dataset_name, dataset, kinetic_features = filter_datasets(dataset_name, dataset, kinetic_features)

    label_mappings = config.get_label_mappings(exp_path)
    if exp_path.name in label_mappings:
        print(f"  [*] Applying custom target label mapping for experiment: {exp_path.name}")
        Y_well = [label_mappings[exp_path.name].get(w, w) for w in Y_well]
    else:
        print(f"  [*] No custom mapping found for {exp_path.name}. Retaining default well labels.")

    encoder = LabelEncoder()
    y_full = encoder.fit_transform(Y_well)

    outlier_filters = [None if f == "none" else f for f in args.outlier_filter]
    total_samples = len(y_full)
    total_datasets = len(dataset_name)

    models = [_MODEL_KEY[m] for m in args.models]

    _target_names = {config.CURVE_TYPE_ALIASES.get(ct, ct) for ct in args.curve_type}
    _reverse_alias = {v: k for k, v in config.CURVE_TYPE_ALIASES.items()}

    all_ml_results = load_or_init_results(results_file_path)

    model_interp_dir = exp_path / "model_interpretation"

    for idx, (name, features_df, curves_2d) in enumerate(zip(dataset_name, kinetic_features, dataset)):
        if name not in _target_names:
            continue

        clean_title = name.replace("_", " ").title()
        xai_curve_type = _reverse_alias.get(name, name)
        progress_pct = ((idx + 1) / total_datasets) * 100

        print(f"\n{'='*75}")
        print(f"[{idx+1}/{total_datasets} | {progress_pct:.1f}%] Processing Dataset: {clean_title}")
        print(f"{'='*75}")

        if clean_title not in all_ml_results:
            all_ml_results[clean_title] = {}

        if args.force_rerun:
            _clear_cached_results(all_ml_results, clean_title, models)

        cached_native = all_ml_results[clean_title].get("Native", {})
        checkpoint_native = make_checkpoint_fn(all_ml_results, results_file_path, clean_title, "Native")

        res_native = evaluate_outlier_filters(
            X_curves=curves_2d,
            features_df=features_df,
            y_encoded=y_full,
            outlier_filters=outlier_filters,
            dataset_name=clean_title,
            mode_name="Native",
            cached_results=cached_native,
            models=models,
            checkpoint_fn=checkpoint_native,
            n_splits=args.n_splits,
            save_model_dir=model_interp_dir,
            save_model_curve_type=xai_curve_type,
            coords=coords_full,
            well_ids=well_ids_full,
            k_neighbors=K_NEIGHBORS,
        )

        all_ml_results[clean_title]["Native"] = res_native
        safe_joblib_dump(all_ml_results, results_file_path, compress=3)

        prefix_native = os.path.join(model_plot_path, f"{name}_Native")
        plot_ml_results(
            results_dict=all_ml_results[clean_title]["Native"],
            outlier_filters=outlier_filters,
            dataset_name=clean_title,
            mode_name="Native Training",
            total_count=total_samples,
            save_prefix=prefix_native,
        )

        gc.collect()


if __name__ == "__main__":
    main()
