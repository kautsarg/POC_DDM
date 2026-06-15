import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gc
import warnings
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Add custom paths
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '0_4_AMCA Code on Chip')
sys.path.insert(0, '../main')
sys.path.insert(0, '../main/utils/02_outlier_detection')
sys.path.insert(0, '../main/utils/model_training')

import config

from titan_v4.Experiment import Experiment
from titan_v4.load_and_preprocessing import titan_load_and_preprocessing
import sigmoid_fitting as sp

from model_utils import set_global_determinism
from msc_outlier import run_msc_pipeline
from amf_outlier import run_amf_pipeline
from mean_std_outlier import run_meanstd_pipeline
from knn_fingerprint_filter import run_knnfilter_pipeline
from lstm_autoencoder_outlier import run_lstm_autoencoder_pipeline
from cnn_autoencoder_outlier import run_cnn_autoencoder_pipeline

warnings.filterwarnings("ignore", category=RuntimeWarning)
set_global_determinism(0)

# ==========================================
# 1. DATA EXTRACTION MODULES (From Script 01)
# ==========================================

def reconstruct_data(exp_data: Experiment, attr_str: str):   
    vstacked = None
    well_ids = []

    for well in exp_data.wells_list:
        temp_x = getattr(well, attr_str).copy()
        temp_x = np.swapaxes(temp_x, 0, 1) 

        if vstacked is None:
            vstacked = temp_x
        else:
            vstacked = np.vstack((vstacked, temp_x))

        well_ids.append(temp_x.shape[0])

    X_time = exp_data.wells_list[0].time
    Y_well = []
    for label, count in enumerate(well_ids):
        Y_well.extend([label] * count)

    return X_time, np.array(Y_well), vstacked

def extract_pixel_temp_dataframes(exp_data):
    df_pix_lin_list, df_pix_nl_list = [], []
    df_temp_lin_list, df_temp_nl_list = [], []

    for w_idx, well in enumerate(exp_data.wells_list):
        idx_settled = well.idx_settled
        idx_active = well.idx_active
        time_npr = well.time_npr

        well_nrows, well_ncols = well.well_nrows, well.well_ncols
        well_temp_nrows, well_temp_ncols = well.well_temp_nrows, well.well_temp_ncols

        well_temp_lin2d = well.well_temp_lin2d
        well_2d_temp_npr = well.well_2d_temp_npr

        y, x = np.indices((well_nrows, well_ncols))
        temp_group_idx = ((y // 5) * well_temp_ncols + (x // 5)).flatten()

        active_y = np.asarray(y.flatten()[idx_active])
        active_x = np.asarray(x.flatten()[idx_active])
        active_temp_mapping = np.asarray(temp_group_idx[idx_active])

        n_time_lin = well.well_3d_lin.shape[2]
        well_2d_lin = well.well_3d_lin.reshape(-1, n_time_lin, order='C').T
        well_2d_bs = well_2d_lin - well_2d_lin[idx_settled, :]
        well_2d_bs_active = well_2d_bs[:, idx_active]

        n_time_nl = well.well_3d_npr.shape[2]
        well_2d_nl = well.well_3d_npr.reshape(-1, n_time_nl, order='C').T
        well_2d_nl_bs = well_2d_nl - well_2d_nl[idx_settled, :]
        well_2d_nl_bs_active = well_2d_nl_bs[:, idx_active]

        time_cols = [f"Cycle_{t}" for t in time_npr]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_temp_lin = np.nanmean(well_temp_lin2d, axis=0)
            mean_temp_nl = np.nanmean(well_2d_temp_npr, axis=0)

        df_pnl = pd.DataFrame(well_2d_nl_bs_active.T, columns=time_cols)
        df_pnl['well_id'] = w_idx
        df_pnl['pixel_row_idx'] = active_y
        df_pnl['pixel_col_idx'] = active_x
        df_pnl['temp_group_idx'] = active_temp_mapping
        counts = df_pnl['temp_group_idx'].value_counts()
        df_pnl['num_active_pixels_in_temp_group'] = df_pnl['temp_group_idx'].map(counts)
        df_pnl['well_temp_lin2d_mean'] = np.asarray(mean_temp_lin[active_temp_mapping])
        df_pnl['well_2d_temp_npr_mean'] = np.asarray(mean_temp_nl[active_temp_mapping])

        meta_cols = ['well_id', 'pixel_row_idx', 'pixel_col_idx', 'temp_group_idx',
                     'num_active_pixels_in_temp_group', 'well_temp_lin2d_mean', 'well_2d_temp_npr_mean']
        df_pnl = df_pnl[meta_cols + time_cols]
        df_pix_nl_list.append(df_pnl)

    return {
        "well_2d_nl_bs_active_df": pd.concat(df_pix_nl_list, ignore_index=True)
    }

# ==========================================
# 2. FEATURE EXTRACTION MODULES (From Script 02)
# ==========================================

def _process_single_row(y, X):
    valid = np.isfinite(X) & np.isfinite(y)
    if np.sum(valid) < 3: return {}
    try: return sp.extract_kinetic_parameters_original(X, y)
    except Exception: return {}

def extract_kinetic_features(timestamps, curves, n_jobs=-1):
    features = Parallel(n_jobs=n_jobs)(
        delayed(_process_single_row)(y, timestamps) for y in curves
    )
    return pd.DataFrame(features)

def get_send(timestamps, curves_2d, send_n=[5, 10, 15, 20, 25]):
    dy_dx_list = Parallel(n_jobs=-1, backend="loky", batch_size='auto')(
        delayed(sp.calculate_first_derivative)(timestamps, y) for y in curves_2d
    )
    dy_dx = np.array(dy_dx_list) 
    send_dict = {}

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        for n in send_n:
            send_dict[f"send_{n}"] = np.nanmean(dy_dx[:, -n:], axis=1)
            send_dict[f"send_abs_{n}"] = np.nanmean(np.abs(dy_dx[:, -n:]), axis=1)
    return send_dict


# ====================================================================
# MASTER PIPELINE EXECUTION
# ====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrated Minimal Curve Pipeline")
    parser.add_argument("--task_id", type=int, default=0, help="Array Job ID")
    parser.add_argument("--exp_folder", type=str, default=config.DEFAULT_EXP_FOLDER)
    args = parser.parse_args()

    exp_paths = sorted([Path(args.exp_folder, name) for name in os.listdir(args.exp_folder) if (os.path.isdir(os.path.join(args.exp_folder, name)) and name not in [".DS_Store"])])

    if args.task_id >= len(exp_paths):
        print(f"Task ID {args.task_id} is out of bounds for {len(exp_paths)} folders. Exiting.")
        sys.exit(0)

    exp_path = exp_paths[args.task_id]

    print(f"\n\n{'#'*80}\nSTARTING MASTER PIPELINE FOR: {exp_path.name}\n{'#'*80}")
    
    # -------------------------------------------------------------
    # 1. LOAD DATA & EXTRACT ori_curves
    # -------------------------------------------------------------
    print("  -> Loading Experiment Data...")
    exp = titan_load_and_preprocessing(exp_path, n_wells=config.N_WELLS, start_type="temperature",
                                       end_time_min=60, n_a_type=config.N_A_TYPE,
                                       print_status=False, plt_gain_calib=False, save_gain_calib=False)
    
    timestamps, Y_well, ori_curves = reconstruct_data(exp, attr_str="well_2d_bs_active")
    
    if np.min(ori_curves) < 0:
        ori_curves = ori_curves - np.min(ori_curves) + 1e-9
        
    pixel_temp_dfs = extract_pixel_temp_dataframes(exp)
    df_meta = pixel_temp_dfs["well_2d_nl_bs_active_df"]
    
    metadata_df = pd.DataFrame({
        "pixel_row_idx": df_meta['pixel_row_idx'].values,
        "pixel_col_idx": df_meta['pixel_col_idx'].values,
        "temp_group_idx": df_meta['temp_group_idx'].values,
        "num_active_pixels_in_temp_group": df_meta['num_active_pixels_in_temp_group'].values,
        "well_temp_lin2d_mean": df_meta['well_temp_lin2d_mean'].values,
        "well_2d_temp_npr_mean": df_meta['well_2d_temp_npr_mean'].values
    })

    dataset_name = np.array(["ori_curves"])
    dataset = np.array([ori_curves])

    # -------------------------------------------------------------
    # 2. EXTRACT KINETICS & APPEND ALIASES
    # -------------------------------------------------------------
    print("  -> Extracting kinetic features (CPU Bound)...")
    base_features_df = extract_kinetic_features(timestamps, ori_curves)
    base_features_df = base_features_df.reset_index(drop=True)
    meta_clean = metadata_df.reset_index(drop=True)
    
    add_features = get_send(timestamps, ori_curves)
    add_features["FFI"] = ori_curves[:, -1]    
    add_features["F_range"] = ori_curves[:, -1] - ori_curves[:, 0]
    
    df_new_features = pd.DataFrame(add_features).reset_index(drop=True)
    kinetic_features_df = pd.concat([base_features_df, meta_clean, df_new_features], axis=1)

    kinetic_features = [kinetic_features_df]

    # -------------------------------------------------------------
    # 3. CORRELATION (Find linear combinations)
    # -------------------------------------------------------------
    print("  -> Extracting Best Linear Triplets (Correlation)...")
    linear_feature_combinations = []
    
    numeric_df = kinetic_features_df.select_dtypes(include=['number', 'float', 'int'])
    if numeric_df.shape[1] > 0:
        corr_matrix = numeric_df.corr().abs() 
        valid_features = [f for f in corr_matrix.columns if f not in config.EXCLUDED_FEATURES]
        cannot_pair_with = {f: set() for f in valid_features}
        
        for f in valid_features:
            for group in config.FEATURE_GROUPS:
                if f in group: cannot_pair_with[f].update(group)
                    
        best_triplet, max_score = [], -1
        for f1, f2, f3 in itertools.combinations(valid_features, 3):
            if f2 in cannot_pair_with[f1] or f3 in cannot_pair_with[f1] or f3 in cannot_pair_with[f2]:
                continue 
            score = corr_matrix.loc[f1, f2] + corr_matrix.loc[f1, f3] + corr_matrix.loc[f2, f3]
            if score > max_score:
                max_score = score
                best_triplet = [f1, f2, f3]
                
        linear_feature_combinations.append(best_triplet.copy() if best_triplet else [])
        print(f"     Best Triplet: {best_triplet} (Avg Inter-Corr: {max_score/3:.3f})")
    else:
        linear_feature_combinations.append([])

    # -------------------------------------------------------------
    # 4. RANDOM FOREST (Top 5 Independent Features)
    # -------------------------------------------------------------
    print("  -> Extracting Top 5 Independent Features (Random Forest)...")
    important_feature_combinations = []
    
    if numeric_df.shape[1] > 0:
        numeric_clean_df = numeric_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_clean_df)
        rf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
        rf.fit(X_scaled, Y_well)
        
        importance_df = pd.DataFrame({'feature': numeric_clean_df.columns, 'importance': rf.feature_importances_}).sort_values(by='importance', ascending=False)
        
        valid_df = importance_df[~importance_df['feature'].isin(config.EXCLUDED_FEATURES)].copy()
        cannot_pair_with = {f: set() for f in valid_df['feature']}
        for f in valid_df['feature']:
            for group in config.FEATURE_GROUPS:
                if f in group: cannot_pair_with[f].update(group)
                    
        selected_features = []
        for _, row in valid_df.iterrows():
            if not any(row['feature'] in cannot_pair_with[sel] for sel in selected_features):
                selected_features.append(row['feature'])
            if len(selected_features) == 5: break
                
        important_feature_combinations.append(selected_features)
        print(f"     Top 5 Features: {selected_features}")
    else:
        important_feature_combinations.append([])

    # -------------------------------------------------------------
    # 5. OUTLIER DETECTION PIPELINES
    # -------------------------------------------------------------
    print("\n=== RUNNING OUTLIER DETECTION PIPELINES ===")
    
    # Define Outlier Configurations
    downsample_factor = config.AE_DOWNSAMPLE_FACTOR
    ae_configs = ["elbow", 90, 95]        
    knn_filter_config = [0.85, 0.90, 0.95]
    
    msc_configs = [
        ("msc_linear_0.001", 0.001, linear_feature_combinations),
        ("msc_baseline_0.001", 0.001, [["Ct", "Cy0", "log_F0"]] * len(dataset_name))
    ]
    amf_configs = [
        ("amf_important", important_feature_combinations),
        ("amf_send_5", [["Fm", "Fb", "Sc", "Cs", "send_5"]] * len(dataset_name)),
    ]
    mean_std_configs = [] 

    features_to_concat = []

    # --- CNN AutoEncoder Per Well ---
    expected_cnn_pw = [f"cnn_ae_pw_ds{downsample_factor}_label_{pct}" for pct in ae_configs]
    extracted_dfs = run_cnn_autoencoder_pipeline(dataset_name, dataset, Y_well, ori_curves, "", ae_configs, save_plot=False, downsample_factor=downsample_factor, per_well=True)
    features_to_concat.append(extracted_dfs[0])

    # --- CNN AutoEncoder Whole Chip ---
    expected_cnn_glb = [f"cnn_ae_glb_ds{downsample_factor}_label_{pct}" for pct in ae_configs]
    extracted_dfs = run_cnn_autoencoder_pipeline(dataset_name, dataset, Y_well, ori_curves, "", ae_configs, save_plot=False, downsample_factor=downsample_factor, per_well=False)
    features_to_concat.append(extracted_dfs[0])

    # --- LSTM AutoEncoder Per Well ---
    expected_lstm_pw = [f"lstm_ae_pw_ds{downsample_factor}_label_{pct}" for pct in ae_configs]
    extracted_dfs = run_lstm_autoencoder_pipeline(dataset_name, dataset, Y_well, ori_curves, "", ae_configs, save_plot=False, downsample_factor=downsample_factor, per_well=True)
    features_to_concat.append(extracted_dfs[0])
    
    # --- LSTM AutoEncoder Whole Chip ---
    expected_lstm_glb = [f"lstm_ae_glb_ds{downsample_factor}_label_{pct}" for pct in ae_configs]
    extracted_dfs = run_lstm_autoencoder_pipeline(dataset_name, dataset, Y_well, ori_curves, "", ae_configs, save_plot=False, downsample_factor=downsample_factor, per_well=False)
    features_to_concat.append(extracted_dfs[0])

    # --- KNN Filter ---
    extracted_dfs = run_knnfilter_pipeline(dataset_name, dataset, Y_well, ori_curves, "", knn_filter_config, save_plot=False)
    features_to_concat.append(extracted_dfs[0])

    # --- MSC Filter ---
    for exp_label, p_val, feats in msc_configs:
        extracted_dfs = run_msc_pipeline(exp_label, p_val, ori_curves, "", dataset_name, kinetic_features, dataset, Y_well, feats, save_plot=False)
        features_to_concat.append(extracted_dfs[0])

    # --- AMF Filter ---
    for exp_label, feats in amf_configs:
        extracted_dfs = run_amf_pipeline(exp_label, feats, ori_curves, "", dataset_name, kinetic_features, dataset, Y_well, save_plot=False)
        features_to_concat.append(extracted_dfs[0])

    # --- Mean/Std Filter ---
    for exp_label, num_std in mean_std_configs:
        extracted_dfs = run_meanstd_pipeline(exp_label, num_std, ori_curves, "", dataset_name, dataset, Y_well, kinetic_features[0].index, save_plot=False)
        features_to_concat.append(extracted_dfs[0])

    # Combine all newly generated outlier masks into the main kinetic_features dataframe
    if features_to_concat:
        kinetic_features[0] = pd.concat([kinetic_features[0]] + features_to_concat, axis=1)

    print(f"\nExperiment {exp_path.name} finished gracefully!")
    
    # Optional: You now have `kinetic_features[0]` in memory holding all features and outlier masks 
    # ready for downstream use or saving if needed.

    tf.keras.backend.clear_session()
    gc.collect()