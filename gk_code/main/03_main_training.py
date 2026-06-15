import os
import sys
import gc
import argparse
import joblib 
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
sys.path.insert(0, 'utils/model_training')
from model_utils import evaluate_outlier_filters, plot_ml_results, set_global_determinism
from sklearn.feature_selection import mutual_info_classif
import numpy as np


import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
set_global_determinism(0)

# ============================================================
# HELPERS
# ============================================================
def get_exp_paths(exp_folder):
    return sorted([
        Path(exp_folder, name)
        for name in os.listdir(exp_folder)
        if (os.path.isdir(os.path.join(exp_folder, name)) and name not in config.EXCLUDED_FOLDERS)
    ])

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
        return joblib.load(results_file_path)
    return {}

def make_checkpoint_fn(all_ml_results, results_file_path, clean_title, mode_key):
    def _checkpoint(updated_results):
        all_ml_results[clean_title][mode_key] = updated_results
        joblib.dump(all_ml_results, results_file_path, compress=3)
    return _checkpoint

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main Training Pipeline")
    parser.add_argument("--task_id", type=int, default=0, help="Array Job ID")
    parser.add_argument("--exp_folder", type=str, default=config.DEFAULT_EXP_FOLDER)
    parser.add_argument("--n_splits", type=int, default=1)
    parser.add_argument("--force_rerun", action="store_true", help="Recompute and overwrite even if presaved results already exist")
    parser.add_argument("--training_mode", type=str, nargs="+", choices=["native", "reference"], default=["native"], help="Which training mode(s) to run: 'native' (train on this dataset's own curves) and/or 'reference' (train on the original ori_curves, using this dataset's outlier filters)")
    args = parser.parse_args()

    exp_paths = get_exp_paths(args.exp_folder)
    n_splits = args.n_splits
    if args.task_id >= len(exp_paths):
        print(f"Task ID {args.task_id} is out of bounds for {len(exp_paths)} folders. Exiting.")
        sys.exit(0)

    exp_path = exp_paths[args.task_id]
    print(f"\n\n{'#'*80}\nSTARTING TRAINING FOR: {exp_path.name}\n{'#'*80}")

    if(n_splits > 1):
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

    # Filter Clean data only
    dataset_name, dataset, kinetic_features = filter_datasets(dataset_name, dataset, kinetic_features)

    if hasattr(config, "LABEL_MAPPINGS") and exp_path.name in config.LABEL_MAPPINGS:
        print(f"  [*] Applying custom target label mapping for experiment: {exp_path.name}")
        mapping = config.LABEL_MAPPINGS[exp_path.name]
        
        # Maps matching keys; falls back to the original index value if not found
        Y_well = [mapping.get(w, w) for w in Y_well]
    else:
        print(f"  [*] No custom mapping found for {exp_path.name}. Retaining default well labels.")

    # Convert to encoded labels
    encoder = LabelEncoder()
    y_full = encoder.fit_transform(Y_well)

    # outlier_filters = config.OUTLIER_FILTERS
    outlier_filters = [None, 'lstm_ae_glb_ds1_label_elbow', 'spatial_knn_label_elbow', 'spatial_grid_label_elbow']

    print(f"[*] Found {len(outlier_filters)-1} Dynamic Outlier Filters to test.")

    if args.force_rerun:
        print(f"  -> [FORCE RERUN] Ignoring presaved results at {results_file_path}. Recomputing everything...")
        all_ml_results = {}
    else:
        all_ml_results = load_or_init_results(results_file_path)

    total_datasets = len(dataset_name)
    total_samples = len(y_full)
    trained_curve = dataset[0].copy()

    for idx, (name, features_df, curves_2d) in enumerate(zip(dataset_name, kinetic_features, dataset)):
        if(name in ["ori_curves", "ori_curves_avg"]): # or name =='original_fitted_full'):
            clean_title = name.replace("_", " ").title()
            progress_pct = ((idx + 1) / total_datasets) * 100
            
            print(f"\n{'='*75}")
            print(f"[{idx+1}/{total_datasets} | {progress_pct:.1f}%] Processing Dataset: {clean_title}")
            print(f"{'='*75}")
    
            if clean_title not in all_ml_results: 
                all_ml_results[clean_title] = {}
    
            # --- FEATURE SELECTION (MUTUAL INFORMATION) ---
            print(f"\n  [*] Calculating Mutual Information for Top 10 Features...")
            
            # Extract candidate features and clean NaNs/Infs (MI function will crash otherwise)
            X_candidates = features_df[config.LD_FEATURES].values
            X_candidates_clean = np.nan_to_num(X_candidates, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Calculate MI scores
            mi_scores = mutual_info_classif(X_candidates_clean, y_full, random_state=0)
            
            # Get indices of the top 10 scores (sorted descending)
            top_10_idx = np.argsort(mi_scores)[-10:][::-1]
            top_10_features = [config.LD_FEATURES[i] for i in top_10_idx]
            
            print(f"  [*] Selected Top 10 Features: {top_10_features}")

            models = ["knn", "cnn", "gru", "transformer", "cnn_gru_dual", "cnn_trans_dual"]

            # --- REFERENCE TRAINING (this dataset's outlier filters, trained on the original curves) ---
            if "reference" in args.training_mode:
                print(f"\n  [MODE] REFERENCE TRAINING")
                cached_ref = all_ml_results[clean_title].get("Reference", {})

                checkpoint_ref = make_checkpoint_fn(all_ml_results, results_file_path, clean_title, "Reference")

                res_ref = evaluate_outlier_filters(
                    X_curves=trained_curve,
                    features_df=features_df,
                    y_encoded=y_full,
                    outlier_filters=outlier_filters,
                    dataset_name=clean_title,
                    mode_name="Reference",
                    cached_results=cached_ref,
                    models=models,
                    checkpoint_fn=checkpoint_ref,
                    KFS=top_10_features,
                    rerun_models=config.RERUN_MODELS,
                    n_splits=args.n_splits
                )

                all_ml_results[clean_title]["Reference"] = res_ref

                joblib.dump(all_ml_results, results_file_path, compress=3)

                prefix_ref = os.path.join(model_plot_path, f"{name}_Reference")
                plot_ml_results(
                    results_dict=all_ml_results[clean_title]["Reference"],
                    outlier_filters=outlier_filters,
                    dataset_name=clean_title,
                    mode_name="Reference Training",
                    total_count=total_samples,
                    save_prefix=prefix_ref
                )

            # --- NATIVE TRAINING (this dataset's outlier filters AND training curves) ---
            if "native" in args.training_mode:
                print(f"\n  [MODE] NATIVE TRAINING")
                cached_reference = all_ml_results[clean_title].get("Reference")
                if np.array_equal(curves_2d, trained_curve) and cached_reference:
                    print(f"  [*] '{clean_title}' curves are identical to the Reference training curves. Reusing saved Reference results, skipping retraining.")
                    res_native = cached_reference
                else:
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
                        KFS=top_10_features,
                        rerun_models=config.RERUN_MODELS,
                        n_splits=args.n_splits
                    )

                all_ml_results[clean_title]["Native"] = res_native
                joblib.dump(all_ml_results, results_file_path, compress=3)

                prefix_native = os.path.join(model_plot_path, f"{name}_Native")
                plot_ml_results(
                    results_dict=all_ml_results[clean_title]["Native"],
                    outlier_filters=outlier_filters,
                    dataset_name=clean_title,
                    mode_name="Native Training",
                    total_count=total_samples,
                    save_prefix=prefix_native
                )

        gc.collect()