import os
import sys
import gc
import argparse
import joblib
from safe_io import safe_joblib_dump
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
sys.path.insert(0, 'utils')
sys.path.insert(0, 'utils/model_training')
from model_utils import evaluate_outlier_filters, plot_ml_results, set_global_determinism
from sklearn.feature_selection import mutual_info_classif
import numpy as np
import pandas as pd


import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

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

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print(f"\n{'='*70}\n[RUNNING] {os.path.basename(__file__)}\n{'='*70}\n")
    parser = argparse.ArgumentParser(description="Main Training Pipeline")
    parser.add_argument("--task_id", type=int, default=0, help="Array Job ID")
    parser.add_argument("--exp_folder", type=str, default=config.DEFAULT_EXP_FOLDER)
    parser.add_argument("--n_splits", type=int, default=1)
    parser.add_argument("--force_rerun", action="store_true", help="Recompute and overwrite even if presaved results already exist")
    parser.add_argument("--training_mode", type=str, nargs="+", choices=["native", "reference"], default=["native"], help="Which training mode(s) to run: 'native' (train on this dataset's own curves) and/or 'reference' (train on the original ori_curves, using this dataset's outlier filters)")
    parser.add_argument("--curve_type", type=str, nargs="+", default=["ori_curve", "ori_curve_avg", "ori_curve_wavelet_sym8"], help="Which curve variant(s) to train on and save XAI models for (e.g. 'ori_curve' 'ori_curve_avg')")
    parser.add_argument("--fast_mode", action="store_true",
                        help="Disable strict TF determinism (TF_CUDNN_DETERMINISTIC/enable_op_determinism) "
                             "for faster GRU/LSTM/Transformer training. RNG seeds are still set, but reruns "
                             "won't be bit-exact. Only affects this script.")
    parser.add_argument("--k_neighbors", type=int, default=24,
                        help="Neighbours per pixel (within the same well) for "
                             "cnn_gru_dual_cosine_recon/cnn_gru_dual_attn_recon's spatial reconstruction.")
    args = parser.parse_args()

    set_global_determinism(0, strict=not args.fast_mode)
    
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

    # Spatial metadata for cnn_gru_dual_cosine_recon/cnn_gru_dual_attn_recon (see
    # model_utils.build_neighbor_curve_stack). Soft-optional: unlike 03b_gnn_spatial_training.py
    # (which trains GNN models exclusively and exits if metadata is missing), 03 trains many
    # non-spatial models too -- a dataset lacking metadata just means those two models get
    # skipped (with a warning from evaluate_outlier_filters), everything else still runs.
    # well_ids is derived from the RAW (pre label-mapping) Y_well -- the physical/spatial
    # grouping for neighbour-finding, deliberately independent of how config.LABEL_MAPPINGS
    # later buckets labels for the classification target (mirrors 03b's identical comment).
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
            print("  [*] No pixel_row_idx/pixel_col_idx in metadata -- "
                  "cnn_gru_dual_cosine_recon/cnn_gru_dual_attn_recon will be skipped for this dataset.")
    else:
        print("  [*] No 'metadata' in training data -- "
              "cnn_gru_dual_cosine_recon/cnn_gru_dual_attn_recon will be skipped for this dataset.")

    # Filter Clean data only
    dataset_name, dataset, kinetic_features = filter_datasets(dataset_name, dataset, kinetic_features)

    label_mappings = config.get_label_mappings(exp_path)
    if exp_path.name in label_mappings:
        print(f"  [*] Applying custom target label mapping for experiment: {exp_path.name}")
        mapping = label_mappings[exp_path.name]
        
        # Maps matching keys; falls back to the original index value if not found
        Y_well = [mapping.get(w, w) for w in Y_well]
    else:
        print(f"  [*] No custom mapping found for {exp_path.name}. Retaining default well labels.")

    # Convert to encoded labels
    encoder = LabelEncoder()
    y_full = encoder.fit_transform(Y_well)

    # outlier_filters = config.OUTLIER_FILTERS
    # outlier_filters = [None, 'lstm_ae_glb_ds1_label_elbow', 'spatial_knn_label_elbow', 'spatial_grid_label_elbow']
    outlier_filters = [None]

    print(f"[*] Found {len(outlier_filters)-1} Dynamic Outlier Filters to test.")

    if args.force_rerun:
        print(f"  -> [FORCE RERUN] Ignoring presaved results at {results_file_path}. Recomputing everything...")
        all_ml_results = {}
    else:
        all_ml_results = load_or_init_results(results_file_path)

    total_datasets = len(dataset_name)
    total_samples = len(y_full)
    trained_curve = dataset[0].copy()

    # Dataset names that correspond to the requested curve_types.
    _target_names = {config.CURVE_TYPE_ALIASES.get(ct, ct) for ct in args.curve_type}

    # Maps dataset_name back to the CLI curve_type alias used in file names.
    _reverse_alias = {v: k for k, v in config.CURVE_TYPE_ALIASES.items()}

    def _lstm_ae_paths(curve_dataset_name):
        """Path to the pretrained LSTM-AE encoder/scaler for one curve variant, saved
        by 02_outlier_detection_pipeline.py's global LSTM autoencoder (see
        lstm_autoencoder_outlier.py's _save_encoder). Returns (None, None) if either
        file is missing — "lstm_ae_clf" then gets skipped by evaluate_outlier_filters."""
        enc_dir = exp_path / "pretrained_encoders"
        enc_path = enc_dir / f"lstm_ae_encoder_{curve_dataset_name}.keras"
        scaler_path = enc_dir / f"lstm_ae_scaler_{curve_dataset_name}.joblib"
        if enc_path.exists() and scaler_path.exists():
            return str(enc_path), str(scaler_path)
        return None, None

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

        # --- FEATURE SELECTION (MUTUAL INFORMATION) on training data only ---
        # MI is computed on the same 90/10 split used for the None-filter baseline,
        # so test-set labels never influence feature selection.
        print(f"\n  [*] Calculating Mutual Information for Top 10 Features (train split only)...")
        X_candidates = features_df[config.LD_FEATURES].values
        X_candidates_clean = np.nan_to_num(X_candidates, nan=0.0, posinf=0.0, neginf=0.0)

        n_classes = len(np.unique(y_full))
        mi_test_size = max(int(len(y_full) * 0.10), n_classes)
        mi_splitter = StratifiedShuffleSplit(n_splits=1, test_size=mi_test_size, random_state=0)
        mi_train_idx, _ = next(mi_splitter.split(X_candidates_clean, y_full))
        mi_scores = mutual_info_classif(X_candidates_clean[mi_train_idx], y_full[mi_train_idx], random_state=0)

        top_10_idx = np.argsort(mi_scores)[-10:][::-1]
        top_10_features = [config.LD_FEATURES[i] for i in top_10_idx]
        print(f"  [*] Selected Top 10 Features: {top_10_features}")

        # models = ["knn", "cnn", "gru", "transformer", "cnn_lf", "gru_lf", "trans_lf", "cnn_gru_dual", "cnn_trans_dual"]
        models = [
            "knn", "cnn",  # "cnn_inc", 
            "cnn_lf", 
            "gru", "cnn_gru_dual", # "cnn_gru_dual_inc",
             "gru_lf", 
            "transformer", "cnn_trans_dual", # "cnn_trans_dual_inc",
             "trans_lf", 

            # Spatial-reconstruction variants inspired GNN:
            # pixel_row_idx/pixel_col_idx -- see coords_full/well_ids_full above.
            # "cnn_gru_dual_cosine_recon", "cnn_gru_dual_attn_recon",

            # # From pretained outlier unsupervised training encoder
            # "lstm_ae_clf",

            # # New gated dual-branch fusion models
            # "cnn_gru_gate", "cnn_gru_hadamard", "cnn_gru_crossattn", "cnn_gru_film",
            # "cnn_trans_gate", "cnn_trans_hadamard", "cnn_trans_crossattn", "cnn_trans_film",
        ]

        model_interp_dir = exp_path / "model_interpretation"

        # --- REFERENCE TRAINING (this dataset's outlier filters, trained on the original curves) ---
        if "reference" in args.training_mode:
            print(f"\n  [MODE] REFERENCE TRAINING")
            cached_ref = all_ml_results[clean_title].get("Reference", {})
            checkpoint_ref = make_checkpoint_fn(all_ml_results, results_file_path, clean_title, "Reference")

            # Save XAI models from the Reference run for ori_curves (Reference curve = ori_curves).
            ref_save_dir = model_interp_dir if name == "ori_curves" else None
            ref_save_ct = xai_curve_type if ref_save_dir else "ori_curve"

            # Reference always trains on dataset[0] (ori_curves), outlier filters are calculated from other curves
            ref_enc_path, ref_scaler_path = _lstm_ae_paths(dataset_name[0])

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
                n_splits=args.n_splits,
                save_model_dir=ref_save_dir,
                save_model_curve_type=ref_save_ct,
                pretrained_encoder_path=ref_enc_path,
                pretrained_scaler_path=ref_scaler_path,
                coords=coords_full,
                well_ids=well_ids_full,
                k_neighbors=args.k_neighbors,
            )

            all_ml_results[clean_title]["Reference"] = res_ref
            safe_joblib_dump(all_ml_results, results_file_path, compress=3)

            prefix_ref = os.path.join(model_plot_path, f"{name}_Reference")
            plot_ml_results(
                results_dict=all_ml_results[clean_title]["Reference"],
                outlier_filters=outlier_filters,
                dataset_name=clean_title,
                mode_name="Reference Training",
                total_count=total_samples,
                save_prefix=prefix_ref,
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

                native_enc_path, native_scaler_path = _lstm_ae_paths(name)

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
                    n_splits=args.n_splits,
                    save_model_dir=model_interp_dir,
                    save_model_curve_type=xai_curve_type,
                    pretrained_encoder_path=native_enc_path,
                    pretrained_scaler_path=native_scaler_path,
                    coords=coords_full,
                    well_ids=well_ids_full,
                    k_neighbors=args.k_neighbors,
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

        if "top_10_features" not in all_ml_results[clean_title]:
            all_ml_results[clean_title]["top_10_features"] = {}
        for f in outlier_filters:
            all_ml_results[clean_title]["top_10_features"][str(f)] = top_10_features
        safe_joblib_dump(all_ml_results, results_file_path, compress=3)
        print(f"  [XAI] Saved feature metadata into {results_file_path}")

        gc.collect()