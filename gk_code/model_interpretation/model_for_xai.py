import argparse
import sys
import os
import joblib
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import mutual_info_classif
import tensorflow as tf

sys.path.insert(0, "../outlier_detection")
import config
from model_utils import (
    create_cnn_model, create_cnn_lf_model, 
    create_gru_model, create_gru_lf_model, create_cnn_gru_dual_model, 
    create_transformer_model, create_transformer_lf_model, create_cnn_transformer_dual_model, 
    set_global_determinism
)

def main(exp_folder=Path(config.LAB_EXP_FOLDER), curve_type="ori_curve"):
    out_subdir = "model_interpretation"
    exp_paths = sorted([p for p in exp_folder.iterdir() if p.is_dir() and p.name not in config.EXCLUDED_FOLDERS])

    filter_key = None
    filter_str = str(filter_key)

    for exp_path in exp_paths:
        print(f"\n{'='*60}\n[*] Processing Dataset: {exp_path.name} (curve_type: {curve_type})\n{'='*60}")
        out_dir = exp_path / out_subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Localized Joblib Path (Saved inside each specific exp_path)
        joblib_path = out_dir / f"model_interpretation_{curve_type}.joblib"

        if joblib_path.exists():
            print(f"[*] Loading local tracking file from {joblib_path}")
            data_package = joblib.load(joblib_path)
        else:
            print("[*] No local tracking file found. Creating new one.")
            data_package = None

        data = joblib.load(exp_path / config.TRAINING_DATA_PATH)
        dataset_name_list = list(data["dataset_name"])
        try:
            curve_idx, dataset_name = config.resolve_curve_dataset_idx(curve_type, dataset_name_list)
        except ValueError as e:
            print(f"  -> [SKIP] {e}")
            continue

        # 2. Check if the loaded package matches the current dataset
        if data_package is not None and data_package.get("dataset_name") == dataset_name:
            if filter_str not in data_package.get("model_paths", {}):
                print(f"[+] Appending new filter '{filter_str}' to existing dataset.")
                data_package["model_paths"][filter_str] = {}
            else:
                print(f"[*] Found existing entry (Filter: '{filter_str}'). Checking for missing models...")
        else:
            print(f"[+] Creating new entry with filter '{filter_str}'.")
            data_package = {
                "dataset_name" : dataset_name,
                "dataset" : data["dataset"][curve_idx],
                "features_df" : data["kinetic_features"][curve_idx],
                "y_well" : data["Y_well"],
                "timestamps" : data["timestamps"],
                "model_paths" : {filter_str: {}},
                "top_10_features" : {} # Initialize dictionary to hold features per filter
            }
            
        # Ensure the key exists for backward compatibility if loading an old tracking file
        if "top_10_features" not in data_package:
            data_package["top_10_features"] = {}

        # --- DATA PREPARATION ---
        Y_well = data_package["y_well"]
        if hasattr(config, "LABEL_MAPPINGS") and exp_path.name in config.LABEL_MAPPINGS:
            print(f"  [*] Applying custom target label mapping for experiment: {exp_path.name}")
            mapping = config.LABEL_MAPPINGS[exp_path.name]
            
            # Maps matching keys; falls back to the original index value if not found
            Y_well = [mapping.get(w, w) for w in Y_well]
        else:
            print(f"  [*] No custom mapping found for {exp_path.name}. Retaining default well labels.")

        encoder = LabelEncoder()
        y_full = encoder.fit_transform(Y_well)

        if filter_key is None:
            mask = np.ones(len(y_full), dtype=bool)
        else:
            mask = (data_package['features_df'][filter_key] == 1).fillna(False).values
            
        X_curve = data_package["dataset"][mask]
        X_curve = X_curve.astype(np.float32)[..., None]
        y = y_full[mask]
        features_df_masked = data_package['features_df'][mask]

        # --- SPLIT FIRST TO PREVENT DATA LEAKAGE ---
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
        train_idx, test_idx = next(splitter.split(X_curve, y))
        
        X_train_curve, X_test_curve = X_curve[train_idx], X_curve[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # --- CALCULATE MUTUAL INFORMATION STRICTLY ON TRAINING DATA ---
        X_candidates_train = features_df_masked.iloc[train_idx][config.LD_FEATURES].values
        X_candidates_train = np.nan_to_num(X_candidates_train, nan=0.0, posinf=0.0, neginf=0.0)

        mi_scores = mutual_info_classif(X_candidates_train, y_train, random_state=0)
        top_10_idx = np.argsort(mi_scores)[-10:][::-1]
        top_10_features = [config.LD_FEATURES[i] for i in top_10_idx]
        
        # Save the dynamically calculated features safely inside the tracking package
        data_package["top_10_features"][filter_str] = top_10_features
        
        # Extract features for Train and Test based on Top 10
        X_man_train = features_df_masked.iloc[train_idx][top_10_features].values
        X_man_test = features_df_masked.iloc[test_idx][top_10_features].values

        X_man_train = np.nan_to_num(X_man_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        X_man_test = np.nan_to_num(X_man_test, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Scale based on training distributions
        scaler = StandardScaler()
        X_train_man = scaler.fit_transform(X_man_train)
        X_test_man = scaler.transform(X_man_test)

        input_size = X_train_curve.shape[1]
        input_man_size = X_train_man.shape[1]
        output_size = len(np.unique(y))
        
        print(f"  -> Data Prep | X_curve: {X_curve.shape}, X_man: ({len(X_curve)}, {input_man_size}), y: {y.shape}")

        # ---- TRAIN MODELS ----
        set_global_determinism(0)
        
        model_builders = {
            'cnn': lambda: create_cnn_model(input_size, output_size),
            'bigru': lambda: create_gru_model(input_size, output_size),
            'transformer': lambda: create_transformer_model(input_size, output_size),
            
            'cnn_lf': lambda: create_cnn_lf_model(input_size, input_man_size, output_size),
            'bigru_lf': lambda: create_gru_lf_model(input_size, input_man_size, output_size),
            'transformer_lf': lambda: create_transformer_lf_model(input_size, input_man_size, output_size),
            
            'cnn_gru_dual': lambda: create_cnn_gru_dual_model(input_size, output_size),
            'cnn_transformer_dual': lambda: create_cnn_transformer_dual_model(input_size, output_size),
        }

        epochs_map = {
            'cnn': 1000, 'cnn_lf': 1000, 
            'bigru': 500, 'bigru_lf': 500, 'cnn_gru_dual': 500,
            'transformer': 500, 'transformer_lf': 500, 'cnn_transformer_dual': 500 
        }
        
        for name, builder_func in model_builders.items():
            model_filename = f"{name}_{filter_str}_{curve_type}_model.keras"
            model_save_path = out_dir / model_filename

            if model_save_path.exists():
                print(f"  [-] Skipping {name}: Model already trained and saved.")
                data_package["model_paths"][filter_str][name] = str(model_save_path)
                continue

            print(f'  [+] Training {name}...')
            
            model = builder_func()
            
            if name.endswith('_lf'):
                train_inputs = [X_train_curve, X_train_man]
            else:
                train_inputs = X_train_curve

            history = model.fit(
                train_inputs, y_train,
                epochs=epochs_map[name],
                batch_size=512,
                shuffle=True,
                verbose=0,
            )
            print(f"      -> {name} done. val_acc={history.history['accuracy'][-1]:.3f}")

            model.save(model_save_path)
            data_package["model_paths"][filter_str][name] = str(model_save_path)
            
            # 3. Save incrementally to the local joblib file
            joblib.dump(data_package, joblib_path)

            tf.keras.backend.clear_session()

    print(f"\n[*] All datasets successfully processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model for XAI Training Script")
    parser.add_argument("--exp_folder", type=str, default=config.DEFAULT_EXP_FOLDER, help="Path to experiment datasets")
    parser.add_argument("--curve_type", type=str, nargs="+", default=["ori_curve", "ori_curve_avg"], help="Which curve dataset(s) to train on (e.g. 'ori_curve', 'ori_curve_avg', or a raw dataset_name entry)")

    args = parser.parse_args()
    exp_folder = Path(args.exp_folder)
    print(exp_folder)
    for curve_type in args.curve_type:
        main(exp_folder=exp_folder, curve_type=curve_type)