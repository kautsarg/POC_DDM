import os
import sys
import gc
import argparse
import joblib 
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
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
        if (os.path.isdir(os.path.join(exp_folder, name)) and name not in [".DS_Store"])
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
    args = parser.parse_args()

    exp_paths = get_exp_paths(args.exp_folder)

    if args.task_id >= len(exp_paths):
        print(f"Task ID {args.task_id} is out of bounds for {len(exp_paths)} folders. Exiting.")
        sys.exit(0)

    exp_path = exp_paths[args.task_id]
    print(f"\n\n{'#'*80}\nSTARTING TRAINING FOR: {exp_path.name}\n{'#'*80}")

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

    encoder = LabelEncoder()
    y_full = encoder.fit_transform(Y_well)

    outlier_filters = [
        None, 
        
        'msc_label_msc_linear_0.001',
        # 'msc_label_msc_baseline_0.001',

        'amf_label_amf_important',
        # 'amf_label_amf_send_5',

        'knn_top_0.95',
        # 'knn_top_0.9',
        # 'knn_top_0.85', 
        
        # f'cnn_ae_pw_ds{config.AE_DOWNSAMPLE_FACTOR}_label_elbow', 
        # f'cnn_ae_pw_ds{config.AE_DOWNSAMPLE_FACTOR}_label_95',
        # f'cnn_ae_pw_ds{config.AE_DOWNSAMPLE_FACTOR}_label_90', 
        
        f'cnn_ae_glb_ds{config.AE_DOWNSAMPLE_FACTOR}_label_elbow',  
        # f'cnn_ae_glb_ds{config.AE_DOWNSAMPLE_FACTOR}_label_95',
        # f'cnn_ae_glb_ds{config.AE_DOWNSAMPLE_FACTOR}_label_90',

        # f'lstm_ae_pw_ds{config.AE_DOWNSAMPLE_FACTOR}_label_elbow', 
        # f'lstm_ae_pw_ds{config.AE_DOWNSAMPLE_FACTOR}_label_95',
        # f'lstm_ae_pw_ds{config.AE_DOWNSAMPLE_FACTOR}_label_90', 

        f'lstm_ae_glb_ds{config.AE_DOWNSAMPLE_FACTOR}_label_elbow', 
        # f'lstm_ae_glb_ds{config.AE_DOWNSAMPLE_FACTOR}_label_95'
        # f'lstm_ae_glb_ds{config.AE_DOWNSAMPLE_FACTOR}_label_90', 
    ]

    print(f"[*] Found {len(outlier_filters)-1} Dynamic Outlier Filters to test.")

    all_ml_results = load_or_init_results(results_file_path)

    total_datasets = len(dataset_name)
    total_samples = len(y_full)
    trained_curve = dataset[0].copy()

    # --- OPTIMIZATION: Extract existing baseline if script was restarted ---
    shared_ref_baseline = None
    for ct in all_ml_results:
        if "Reference" in all_ml_results[ct] and None in all_ml_results[ct]["Reference"]:
            shared_ref_baseline = all_ml_results[ct]["Reference"][None]
            print("  [*] Found cached Reference Baseline. Will skip redundant baseline training for all datasets.")
            break

    for idx, (name, features_df, curves_2d) in enumerate(zip(dataset_name, kinetic_features, dataset)):
        if(name=="ori_curves"):
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

            # --- REFERENCE TRAINING ---
            print(f"\n  [MODE 2/2] REFERENCE TRAINING")
            cached_ref = all_ml_results[clean_title].get("Reference", {})
    
            # Inject the shared baseline so it immediately hits the cache inside evaluate_outlier_filters
            if shared_ref_baseline is not None and None not in cached_ref:
                cached_ref[None] = shared_ref_baseline
    
            checkpoint_ref = make_checkpoint_fn(all_ml_results, results_file_path, clean_title, "Reference")
            
            res_ref = evaluate_outlier_filters(
                trained_curve, 
                features_df, 
                y_full, 
                outlier_filters, 
                clean_title, 
                mode_name="Reference", 
                cached_results=cached_ref, 
                models=["cnn", "cnn_lf", "gru", "gru_lf", "transformer", "trans_lf", "cnn_gru_dual", "cnn_trans_dual"],
                checkpoint_fn=checkpoint_ref,
                KFS=top_10_features,
                rerun_models=config.RERUN_MODELS
            )
            
            # Capture the baseline after the first dataset runs it, so subsequent iterations skip it
            if shared_ref_baseline is None and None in res_ref:
                shared_ref_baseline = res_ref[None]
    
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
            
        gc.collect()