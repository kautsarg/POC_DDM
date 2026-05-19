import os
import sys
import gc
import argparse
import joblib 
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from model_utils import evaluate_outlier_filters, plot_ml_results, set_global_determinism

import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
set_global_determinism(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main Training Pipeline")
    parser.add_argument("--task_id", type=int, default=0, help="Array Job ID")
    parser.add_argument("--exp_folder", type=str, default=config.DEFAULT_EXP_FOLDER)
    args = parser.parse_args()

    exp_paths = sorted([Path(args.exp_folder, name) for name in os.listdir(args.exp_folder) if (os.path.isdir(os.path.join(args.exp_folder, name)) and name not in [".DS_Store"])])

    if args.task_id >= len(exp_paths):
        print(f"Task ID {args.task_id} is out of bounds for {len(exp_paths)} folders. Exiting.")
        sys.exit(0)

    exp_path = exp_paths[args.task_id]
    print(f"\n\n{'#'*80}\nSTARTING TRAINING FOR: {exp_path.name}\n{'#'*80}")

    data_path = os.path.join(exp_path, config.TRAINING_DATA_PATH)
    results_file_path = os.path.join(exp_path, config.TRAINING_RESULT_PATH)

    model_plot_path = os.path.join(exp_path, "model_performance")
    os.makedirs(model_plot_path, exist_ok=True)

    if not os.path.exists(data_path):
        print(f"  -> Skipping {exp_path.name}: '{data_path}' not found.")
        sys.exit(0)

    training_data = joblib.load(data_path)

    dataset_name = training_data["dataset_name"]
    dataset = training_data["dataset"]
    kinetic_features = training_data["kinetic_features"]
    Y_well = training_data["Y_well"]

    # Filter Clean data only
    filtered_names, filtered_dataset, filtered_features = [], [], []

    for name, data, features in zip(dataset_name, dataset, kinetic_features):
        if not name.startswith("avg_") and not name.startswith("original_fitted_stretched"):
            filtered_names.append(name)
            filtered_dataset.append(data)
            filtered_features.append(features)

    dataset_name = filtered_names
    dataset = filtered_dataset
    kinetic_features = filtered_features

    encoder = LabelEncoder()
    y_full = encoder.fit_transform(Y_well)

    outlier_filters = [
        None, 
        # 'msc_label_msc_linear_0.001'
        # 'msc_label_msc_baseline_0.001',

        # 'amf_label_amf_important'
        # 'amf_label_amf_send_5',

        # 'knn_top_0.95',
        # 'knn_top_0.9',
        # # 'knn_top_0.85', 
        
        # f'cnn_ae_pw_ds{config.AE_DOWNSAMPLE_FACTOR}_label_elbow', 
        # f'cnn_ae_pw_ds{config.AE_DOWNSAMPLE_FACTOR}_label_95',
        # # f'cnn_ae_pw_ds{config.AE_DOWNSAMPLE_FACTOR}_label_90', 
        
        # f'cnn_ae_glb_ds{config.AE_DOWNSAMPLE_FACTOR}_label_elbow',  
        # f'cnn_ae_glb_ds{config.AE_DOWNSAMPLE_FACTOR}_label_95',
        # # f'cnn_ae_glb_ds{config.AE_DOWNSAMPLE_FACTOR}_label_90',

        # f'lstm_ae_pw_ds{config.AE_DOWNSAMPLE_FACTOR}_label_elbow', 
        # f'lstm_ae_pw_ds{config.AE_DOWNSAMPLE_FACTOR}_label_95',
        # # f'lstm_ae_pw_ds{config.AE_DOWNSAMPLE_FACTOR}_label_90', 

        # f'lstm_ae_glb_ds{config.AE_DOWNSAMPLE_FACTOR}_label_elbow', 
        # f'lstm_ae_glb_ds{config.AE_DOWNSAMPLE_FACTOR}_label_95'
        # f'lstm_ae_glb_ds{config.AE_DOWNSAMPLE_FACTOR}_label_90', 
    ]

    print(f"[*] Found {len(outlier_filters)-1} Dynamic Outlier Filters to test.")

    if os.path.exists(results_file_path):
        all_ml_results = joblib.load(results_file_path)
    else:
        all_ml_results = {}

    total_datasets = len(dataset_name)
    total_samples = len(y_full)
    trained_curve = dataset[0].copy()

    for idx, (name, features_df, curves_2d) in enumerate(zip(dataset_name, kinetic_features, dataset)):
        clean_title = name.replace("_", " ").title()
        progress_pct = ((idx + 1) / total_datasets) * 100
        
        print(f"\n{'='*75}")
        print(f"[{idx+1}/{total_datasets} | {progress_pct:.1f}%] Processing Dataset: {clean_title}")
        print(f"{'='*75}")

        if clean_title not in all_ml_results: 
            all_ml_results[clean_title] = {}

        # # --- NATIVE TRAINING ---
        # print(f"\n  [MODE 1/2] NATIVE TRAINING")
        # cached_native = all_ml_results[clean_title].get("Native", {})
        
        # res_native = evaluate_outlier_filters(
        #     curves_2d, features_df, y_full, outlier_filters, clean_title, 
        #     mode_name="Native", cached_results=cached_native, models=["cnn"]
        # )
        # all_ml_results[clean_title]["Native"] = res_ref
        
        # joblib.dump(all_ml_results, results_file_path, compress=3)
            
        # prefix_native = os.path.join(model_plot_path, f"{name}_Native")
        # plot_ml_results(
        #     results_dict=all_ml_results[clean_title]["Native"], 
        #     outlier_filters=outlier_filters, 
        #     dataset_name=clean_title, 
        #     mode_name="Native Training", 
        #     total_count=total_samples, 
        #     save_prefix=prefix_native
        # )

        # --- REFERENCE TRAINING ---
        print(f"\n  [MODE 2/2] REFERENCE TRAINING")
        cached_ref = all_ml_results[clean_title].get("Reference", {})

        def checkpoint_ref(updated_results):
            all_ml_results[clean_title]["Reference"] = updated_results
            joblib.dump(all_ml_results, results_file_path, compress=3)
        
        res_ref = evaluate_outlier_filters(
            trained_curve, features_df, y_full, outlier_filters, clean_title, 
            mode_name="Reference", cached_results=cached_ref, models=["cnn", "lstm", "gru", "rnn", "transformer", "rf"],
            checkpoint_fn=checkpoint_ref
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
        
    gc.collect()