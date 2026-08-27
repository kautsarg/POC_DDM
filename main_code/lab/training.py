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
    return joblib.load(data_path)


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


def load_spatial_metadata(training_data, Y_well):
    """coords/well_ids for attn_recon's neighbour stack -- always None for LAB data
    (flat per-sample curves, no pixel grid), so cnn_gru_dual_attn_recon is soft-skipped."""
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


def _clear_cached_results(results_dict, models):
    for m in models:
        if m not in config.MODEL_KEY_MAP:
            continue
        pk, pbk, ck = config.MODEL_KEY_MAP[m]
        for res in results_dict.values():
            if isinstance(res, dict):
                for k in (pk, pbk, ck, f'train_history_{m}_'):
                    res.pop(k, None)


def main(argv=None):
    parser = argparse.ArgumentParser(description="LAB per-experiment model comparison training")
    parser.add_argument("--task_id", type=int, default=0, help="Array Job ID")
    parser.add_argument("--exp_folder", type=str, default=config.LAB_EXP_FOLDER)
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--curve_type", type=str, default="ori_curve",
                        choices=list(config.CURVE_TYPE_ALIASES.keys()))
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--models", type=str, nargs="+", choices=MODEL_CHOICES, default=MODEL_CHOICES)
    args = parser.parse_args(argv)

    print(f"\n{'='*70}\n[RUNNING] lab/training.py\n{'='*70}\n")

    set_global_determinism(0, strict=True)

    exp_paths = get_exp_paths(args.exp_folder)
    check_task_id(args.task_id, exp_paths)
    exp_path = exp_paths[args.task_id]
    print(f"\n\n{'#'*80}\nSTARTING TRAINING FOR: {exp_path.name}\n{'#'*80}")

    out_dir = exp_path / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file_path = out_dir / "model_comparison_performances.joblib"
    model_interp_dir = out_dir / "model_interpretation"

    training_data = load_training_data(exp_path)

    dataset_name = training_data["dataset_name"]
    dataset = training_data["dataset"]
    kinetic_features = training_data["kinetic_features"]
    Y_well = training_data["Y_well"]

    coords_full, well_ids_full = load_spatial_metadata(training_data, Y_well)

    dataset_name, dataset, kinetic_features = filter_datasets(dataset_name, dataset, kinetic_features)

    label_mappings = config.get_label_mappings(exp_path)
    if exp_path.name in label_mappings:
        Y_well = [label_mappings[exp_path.name].get(w, w) for w in Y_well]

    encoder = LabelEncoder()
    y_full = encoder.fit_transform(Y_well)

    target_name = config.CURVE_TYPE_ALIASES.get(args.curve_type, args.curve_type)
    if target_name not in dataset_name:
        print(f"  [SKIP] curve_type '{args.curve_type}' (-> '{target_name}') not found in "
              f"this dataset's curve variants: {dataset_name}")
        return
    idx = dataset_name.index(target_name)
    curves_2d = dataset[idx]
    features_df = kinetic_features[idx]
    clean_title = dataset_name[idx].replace("_", " ").title()

    models = [_MODEL_KEY[m] for m in args.models]

    results_dict = {} if args.force_rerun else load_or_init_results(results_file_path)
    if args.force_rerun:
        _clear_cached_results(results_dict, models)

    print(f"  [*] Training {models} on curve_type='{args.curve_type}' ({len(y_full)} samples)")

    def checkpoint(updated_results):
        safe_joblib_dump(updated_results, results_file_path, compress=3)

    results_dict = evaluate_outlier_filters(
        X_curves=curves_2d,
        features_df=features_df,
        y_encoded=y_full,
        outlier_filters=[None],
        dataset_name=clean_title,
        mode_name="Native",
        models=models,
        n_splits=args.n_splits,
        coords=coords_full,
        well_ids=well_ids_full,
        save_model_dir=model_interp_dir,
        save_model_curve_type=args.curve_type,
        batch_size=args.batch_size,
        cached_results=results_dict,
        checkpoint_fn=checkpoint,
        k_neighbors=K_NEIGHBORS,
    )

    safe_joblib_dump(results_dict, results_file_path, compress=3)
    print(f"  [+] Saved results -> {results_file_path}")

    plot_ml_results(
        results_dict=results_dict,
        outlier_filters=[None],
        dataset_name=clean_title,
        mode_name="Model Comparison",
        total_count=len(y_full),
        save_prefix=str(out_dir / f"{dataset_name[idx]}_model_comparison"),
    )


if __name__ == "__main__":
    main()
