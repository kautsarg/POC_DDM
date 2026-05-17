import os
import sys
import gc
import base64
import pickle
import itertools
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import tensorflow as tf
from cycler import cycler
from matplotlib.colors import to_hex

# --- CUSTOM MODULES ---
sys.path.insert(0, '..')
import chip_utilities as utils
import sigmoid_fitting as sp
from msc_outlier import run_msc_pipeline
from amf_outlier import run_amf_pipeline
from mean_std_outlier import run_meanstd_pipeline
from knn_fingerprint_filter import run_knnfilter_pipeline
from autoencoder_outlier import run_autoencoder_pipeline
from lstm_autoencoder_outlier import run_lstm_autoencoder_pipeline
from cnn_autoencoder_outlier import run_cnn_autoencoder_pipeline
from autoencoder_outlier_per_well import run_autoencoder_per_well_pipeline
from lstm_autoencoder_outlier_per_well import run_lstm_autoencoder_per_well_pipeline
from cnn_autoencoder_outlier_per_well import run_cnn_autoencoder_per_well_pipeline

# ====================================================================
# GLOBAL CONSTANTS & CONFIGURATIONS
# ====================================================================

from model_utils import set_global_determinism
set_global_determinism(0)

# Matplotlib Color Cycler Configuration
mpl_colors = [
    (0.00, 0.45, 0.70), (0.90, 0.60, 0.00), (0.35, 0.70, 0.90), 
    (0.00, 0.60, 0.50), (0.95, 0.90, 0.25), (0.80, 0.40, 0.70), 
    (0.20, 0.13, 0.53), (0.87, 0.80, 0.47), (0.27, 0.67, 0.60), (0.65, 0.65, 0.65)
]
plt.rcParams['axes.prop_cycle'] = cycler(color=[to_hex(i) for i in mpl_colors])

# Feature Exclusions and Groups for Correlation Logic
EXCLUDED_FEATURES = [
    'TH', 'msc_mahal_dist', 'msc_label_0.001', 'amf_label_Send', 
    'amf_label_Send_abs', 'amf_label_Send_fit', 'amf_label_Send_fit_abs', 
    'mean_std_outlier_label_3sigma',
    "pixel_row_idx", "pixel_col_idx", "temp_group_idx", "num_active_pixels_in_temp_group", "well_temp_lin2d_mean", "well_2d_temp_npr_mean"
]

FEATURE_GROUPS = [
    ['F0', 'log_F0'],
    ['send_5', 'send_10', 'send_15', 'send_20', 'send_25', 'send_abs_5', 'send_abs_10', 'send_abs_15', 'send_abs_20', 'send_abs_25', 'Send', 'Send_abs', 'Send_fit', 'Send_fit_abs'],
    ['F_max', 'Fm', 'FFI', 'F_range'],
    ['ct_idx', 'Ct'],
    ['Cy0', 'Cs', 'As', 'Sc'],
    ['first_half_distance', 'A1', 'xs', 'xms'],
    ['second_half_distance', 'A2', 'xms', 'xe'],
    ['threshold_distance', 'xs', 'xe'],
    ['peak_shifting_distance', 'xp1', 'xp2'],
    ['distance_asymmetry_index', 'first_half_distance', 'second_half_distance'],
    ['area_asymmetry_index', 'A1', 'A2'],
    ['peak_asymmetry_index', 'd2y_xp1', 'd2y_xp2'],
    ['xms', 'y_xms', 'dy_xms'],
    ['xp1', 'y_xp1', 'dy_xp1', 'd2y_xp1'],
    ['xp2', 'y_xp2', 'dy_xp2', 'd2y_xp2'],
    ['xs', 'y_xs'],
    ['xe', 'y_xe'],
    ['amplitude', 'y_xs', 'y_xe'],
    ['y_xms', 'y_xs', 'y_xe', 'y_xp1', 'y_xp2', 'F_max', 'Fm', 'FFI', 'F_range']
]


# ====================================================================
# HELPER FUNCTIONS: KINETICS & ALIAS EXTRACTION
# ====================================================================

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
    for n in send_n:
        send_dict[f"send_{n}"] = np.nanmean(dy_dx[:, -n:], axis=1)
        send_dict[f"send_abs_{n}"] = np.nanmean(np.abs(dy_dx[:, -n:]), axis=1)
    return send_dict


# ====================================================================
# HELPER FUNCTIONS: VISUALIZATION
# ====================================================================

def feature_boxplot(features_df, well_labels, feature_columns, target="well", title="", save_path=None):
    plot_data = features_df.copy()
    plot_data[target] = well_labels
    valid_features = [f for f in feature_columns if f in plot_data.columns]
    
    if not valid_features:
        print(f"Skipping {title}: None of the requested features exist.")
        return

    n_features = len(valid_features)
    
    if n_features == 3:
        fig = plt.figure(figsize=(10, 8))
        axes = [fig.add_subplot(2, 2, i+1) for i in range(3)]
        for i, feature in enumerate(valid_features):
            sns.boxplot(data=plot_data, x=target, y=feature, ax=axes[i])
            axes[i].set_title(feature, fontweight='bold')
            axes[i].grid(True, alpha=0.3, axis='y')
            
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        f1, f2, f3 = valid_features
        unique_targets = np.unique(well_labels)
        palette = sns.color_palette("tab10", len(unique_targets))
        
        for idx, val in enumerate(unique_targets):
            subset = plot_data[plot_data[target] == val]
            ax4.scatter(subset[f1], subset[f2], subset[f3], label=f"Well {val}", color=palette[idx], alpha=0.7, s=20)
            
        ax4.set_xlabel(f1, fontweight='bold')
        ax4.set_ylabel(f2, fontweight='bold')
        ax4.set_zlabel(f3, fontweight='bold')
        ax4.set_title("3D Feature Space", fontweight='bold')
        ax4.legend(title=target, bbox_to_anchor=(1.15, 1), loc='upper left')

    else:
        fig, axes = plt.subplots(1, n_features, figsize=(n_features * 4, 3))
        if n_features == 1: axes = [axes]
        for i, feature in enumerate(valid_features):
            sns.boxplot(data=plot_data, x=target, y=feature, ax=axes[i])
            axes[i].set_title(feature, fontweight='bold')
            axes[i].grid(True, alpha=0.3, axis='y')
            
    fig.suptitle(title, fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path: plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig)

def save_html_report(save_path, title, subtitle, img_buffers, flex_layout=False):
    """Wrapper to generate and save standard HTML reports."""
    html_content = f"""
    <html>
    <head><title>{title}</title></head>
    <body style="font-family: Arial, sans-serif; background-color: #f4f4f9; text-align: center; margin: 0; padding: 20px;">
        <h1 style="color: #333; margin-bottom: 10px;">{title}</h1>
        {f'<p style="color: #666; margin-bottom: 40px;">{subtitle}</p>' if subtitle else ''}
    """
    
    if flex_layout:
        html_content += "<div style='display: flex; flex-wrap: wrap; justify-content: center; gap: 20px;'>"
        width_style = "width: 45%; min-width: 450px;"
    else:
        width_style = "width: 95%; max-width: 1600px; margin: 40px auto;"
        
    for buf in img_buffers:
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        html_content += f'''
        <div style="background: white; padding: 15px; box-shadow: 0px 4px 10px rgba(0,0,0,0.1); border-radius: 8px; {width_style}">
            <img src="data:image/png;base64,{img_base64}" style="width: 100%; height: auto;">
        </div>
        '''
        
    if flex_layout: html_content += "</div>"
    html_content += "</body></html>"
    
    with open(save_path, "w") as f:
        f.write(html_content)


# ====================================================================
# MASTER PIPELINE EXECUTION
# ====================================================================

if __name__ == "__main__":
    # exp_folder = "/Users/kautsarg/Documents/Final Project/Run Data/POC_DDM_dataset"
    exp_folder = "/rds/general/user/gk225/home/Run Data/POC_DDM_dataset/"
    
    # Process specific experiment or iterate over all
    # exp_paths = [Path(exp_folder, "D20250808_E00_C00_F4500KHz_U_Sample_7")]
    exp_paths = sorted([Path(exp_folder, name) for name in os.listdir(exp_folder) if (os.path.isdir(os.path.join(exp_folder, name)) and name not in [".DS_Store"])])

    if len(sys.argv) > 1:
        # PBS_ARRAY_INDEX starts at 1, so we subtract 1 for Python's 0-based indexing
        task_id = int(sys.argv[1]) - 1 
    else:
        # Fallback for local testing
        task_id = 0 

    if task_id >= len(exp_paths):
        print(f"Task ID {task_id} is out of bounds for {len(exp_paths)} folders. Exiting.")
        sys.exit(0)

    # Pick ONLY the specific folder for this specific node
    exp_path = exp_paths[task_id]

    print(f"\n\n{'#'*80}\nSTARTING MASTER PIPELINE FOR: {exp_path.name}\n{'#'*80}")
    
    # -------------------------------------------------------------
    # 1. LOAD PREPROCESSED CURVES & METADATA
    # -------------------------------------------------------------
    curve_path = Path(exp_path, "preprocessed_curves_data.pkl")
    if not curve_path.exists():
        print(f"Skipping {exp_path.name} - 'preprocessed_curves_data.pkl' not found.")
    else:
            
        with open(curve_path, 'rb') as f:
            data = pickle.load(f)

        Y_well = data["well_labels"]
        timestamps = data["timestamps"]
        metadata_df = pd.DataFrame(data["metadata"])
        
        dataset_name = ["ori_curves", "ori_curves_avg"]
        dataset = [data["curves"]["ori_curves"], data["curves"]["ori_curves_avg"]]
        
        for k, v in data["sigmoid_curves"].items():
            dataset_name.append(f"{k}_fitted_full")
            dataset.append(v["fitted_full"])
            dataset_name.append(f"{k}_fitted_stretched")
            dataset.append(v["fitted_stretched"])
            
        dataset_name = np.array(dataset_name)
        dataset = np.array(dataset)

        try:
            colors = pd.factorize(Y_well)[0]
            cmap = 'tab10'
        except NameError:
            colors = '#3498db'
            cmap = None

        # -------------------------------------------------------------
        # 2. EXTRACT KINETICS & APPEND ALIASES
        # -------------------------------------------------------------
        kinetics_path = os.path.join(exp_path, "initial_kinetics.pkl")
        if os.path.exists(kinetics_path):
            print("  -> Loading cached kinetic features...")
            with open(kinetics_path, 'rb') as f:
                kinetic_features = pickle.load(f)
        else:
            print("  -> Extracting initial kinetic features (CPU Bound)...")
            kinetic_features = [extract_kinetic_features(timestamps, curves) for curves in dataset]
            with open(kinetics_path, 'wb') as f: pickle.dump(kinetic_features, f)

        # Append Metadata and 'Send' Aliases
        for idx, (name, features_df, curves_2d) in enumerate(zip(dataset_name, kinetic_features, dataset)):
            features_df = features_df.reset_index(drop=True)
            meta_clean = metadata_df.reset_index(drop=True)
            
            add_features = get_send(timestamps, curves_2d)
            add_features["FFI"] = curves_2d[:, -1]    
            add_features["F_range"] = curves_2d[:, -1] - curves_2d[:, 0]
            
            df_new_features = pd.DataFrame(add_features).reset_index(drop=True)
            kinetic_features[idx] = pd.concat([features_df, meta_clean, df_new_features], axis=1)

        # -------------------------------------------------------------
        # SKIP MOVING AVERAGE DATASET
        # -------------------------------------------------------------

        filtered_names = []
        filtered_dataset = []
        filtered_features = []

        for name, data, features in zip(dataset_name, dataset, kinetic_features):
            if not name.startswith("avg_"):
            # if name == 'ori_curves':
                filtered_names.append(name)
                filtered_dataset.append(data)
                filtered_features.append(features)

        # Reassign the filtered lists back to the original variables
        dataset_name = filtered_names
        dataset = filtered_dataset
        kinetic_features = filtered_features

        # -------------------------------------------------------------
        # 3. GENERATE BOXPLOTS (MSC & AMF Features)
        # -------------------------------------------------------------
        print("\n=== GENERATING FEATURE BOXPLOTS ===")
        msc_features = ["Ct", "Cy0", "log_F0"]
        amf_features_all = ["Fm", "Fb", "Sc", "Cs", "Send", "Send_abs", "Send_fit", "Send_fit_abs"]
        
        msc_plot_path = os.path.join(exp_path, "msc_outlier")
        amf_plot_path = os.path.join(exp_path, "amf_outlier")
        os.makedirs(msc_plot_path, exist_ok=True)
        os.makedirs(amf_plot_path, exist_ok=True)
        
        for name, features_df in zip(dataset_name, kinetic_features):
            clean_title = name.replace("_", " ").title()
            
            # MSC Boxplot
            feature_boxplot(
                features_df=features_df, well_labels=Y_well, feature_columns=msc_features, 
                title=f"MSC Features: {clean_title}", save_path=os.path.join(msc_plot_path, f"{name}_msc_features.png")
            )
            
            # AMF Boxplot
            feature_boxplot(
                features_df=features_df, well_labels=Y_well, feature_columns=amf_features_all, 
                title=f"AMF Features: {clean_title}", save_path=os.path.join(amf_plot_path, f"{name}_amf_features_boxplot.png")
            )
            
        # -------------------------------------------------------------
        # 4. CORRELATION HEATMAPS & BEST TRIPLETS
        # -------------------------------------------------------------
        print("\n=== EXTRACTING BEST LINEAR TRIPLETS (CORRELATION) ===")
        linear_feature_combinations = []
        heatmap_buffers = []
        
        for name, features_df in zip(dataset_name, kinetic_features):
            clean_title = name.replace("_", " ").title()
            numeric_df = features_df.select_dtypes(include=['number', 'float', 'int'])
            corr = numeric_df.corr()
            corr_matrix = corr.abs() 
            
            valid_features = [f for f in corr_matrix.columns if f not in EXCLUDED_FEATURES]
            cannot_pair_with = {f: set() for f in valid_features}
            for f in valid_features:
                for group in FEATURE_GROUPS:
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
            print(f"  -> {clean_title} Best Triplet: {best_triplet} (Avg Inter-Corr: {max_score/3:.3f})")

            # Generate Heatmap Buffer
            fig, ax = plt.subplots(figsize=(24, 20))
            sns.heatmap(corr.mask(corr.abs() < 0.5, 0), cmap="coolwarm", center=0, linewidths=0.5, cbar_kws={"shrink": .75}, ax=ax)
            ax.tick_params(axis='x', rotation=90, labelsize=8); ax.tick_params(axis='y', rotation=0, labelsize=8)
            plt.title(f"Correlation Matrix: {clean_title}", fontsize=22, fontweight='bold', pad=20)
            
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='white')
            buf.seek(0); heatmap_buffers.append(buf)
            plt.close(fig); gc.collect()

        save_html_report(os.path.join(exp_path, "all_correlation_heatmaps.html"), "Experiment Correlation Heatmaps", None, heatmap_buffers)

        # -------------------------------------------------------------
        # 5. GENERATE 3D FEATURE COMBINATIONS
        # -------------------------------------------------------------
        print("\n=== GENERATING 3D COMBINATION PLOTS ===")
        colors = pd.factorize(Y_well)[0]
        plot_3d_buffers = []
        
        for name, kf, dataset_combs in zip(dataset_name, kinetic_features, linear_feature_combinations):
            if not dataset_combs: continue
                
            clean_title = name.replace("_", " ").title()
            x_feat, y_feat, z_feat = dataset_combs[:3]
            
            corr_matrix = kf[[x_feat, y_feat, z_feat]].corr(method='pearson').abs()
            avg_corr = (corr_matrix.loc[x_feat, y_feat] + corr_matrix.loc[x_feat, z_feat] + corr_matrix.loc[y_feat, z_feat]) / 3
            X_vals = kf[[x_feat, y_feat, z_feat]].replace([np.inf, -np.inf], np.nan).fillna(0).values
            
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_vals[:, 0], X_vals[:, 1], X_vals[:, 2], c=colors, cmap='tab10', s=30, alpha=0.8, edgecolor='k')
            ax.set_title(f"{clean_title}\n[{x_feat}, {y_feat}, {z_feat}]\nAvg Inter-Correlation: {avg_corr:.3f}", fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel(x_feat, fontweight='bold', labelpad=10); ax.set_ylabel(y_feat, fontweight='bold', labelpad=10); ax.set_zlabel(z_feat, fontweight='bold', labelpad=15)
            
            buf = BytesIO(); plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            buf.seek(0); plot_3d_buffers.append(buf)
            plt.close(fig); gc.collect()

        save_html_report(os.path.join(exp_path, "best_feature_combinations_3D.html"), "Best 3D Feature Combinations", "Highest inter-correlated valid feature triplet for each experiment.", plot_3d_buffers, flex_layout=True)

        # -------------------------------------------------------------
        # 6. RANDOM FOREST FEATURE IMPORTANCES
        # -------------------------------------------------------------
        print("\n=== EXTRACTING TOP 5 INDEPENDENT FEATURES (RANDOM FOREST) ===")
        importance_dfs, rf_buffers = [], []
        
        for name, features_df in zip(dataset_name, kinetic_features):
            clean_title = name.replace("_", " ").title()
            numeric_df = features_df.select_dtypes(include=['number', 'float', 'int']).replace([np.inf, -np.inf], np.nan).fillna(0)
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(numeric_df)
            rf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
            rf.fit(X_scaled, Y_well)
            
            importance_df = pd.DataFrame({'feature': numeric_df.columns, 'importance': rf.feature_importances_}).sort_values(by='importance', ascending=False)
            importance_dfs.append(importance_df)
            
            fig, ax = plt.subplots(figsize=(12, 18))
            sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis', ax=ax)
            ax.set_title(f"Random Forest Importances: {clean_title}", fontsize=18, fontweight='bold', pad=20)
            ax.set_xlabel("Importance Score", fontweight='bold'); ax.set_ylabel("Features", fontweight='bold')
            ax.grid(axis='x', linestyle='--', alpha=0.6)
            
            buf = BytesIO(); plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='white')
            buf.seek(0); rf_buffers.append(buf)
            plt.close(fig); gc.collect()

        save_html_report(os.path.join(exp_path, "all_feature_importances.html"), "Feature Importance Analysis", None, rf_buffers)

        # Greedy Top 5 Selection
        important_feature_combinations = []
        for name, imp_df in zip(dataset_name, importance_dfs):
            valid_df = imp_df[~imp_df['feature'].isin(EXCLUDED_FEATURES)].copy()
            cannot_pair_with = {f: set() for f in valid_df['feature']}
            for f in valid_df['feature']:
                for group in FEATURE_GROUPS:
                    if f in group: cannot_pair_with[f].update(group)
                        
            selected_features = []
            for _, row in valid_df.iterrows():
                if not any(row['feature'] in cannot_pair_with[sel] for sel in selected_features):
                    selected_features.append(row['feature'])
                if len(selected_features) == 5: break
                    
            important_feature_combinations.append(selected_features)
            print(f"  -> {name.replace('_', ' ').title()}: {selected_features}")

        importance_dfs_path = os.path.join(exp_path, "feature_importance_dfs.pkl")
        save_importance_data = {
            "dataset_name": dataset_name,
            "dataset": dataset,
            "Y_well": Y_well,
            "timestamps": timestamps,
            "importance_dfs": importance_dfs,
            "top_combination": important_feature_combinations,
        }

        with open(importance_dfs_path, 'wb') as f:
            pickle.dump(save_importance_data, f)

        # -------------------------------------------------------------
        # 7. TSNE & 3D FOR TOP 5 FEATURES
        # -------------------------------------------------------------
        print("\n=== GENERATING TSNE & 3D PLOTS FOR TOP 5 FEATURES ===")
        tsne_buffers = []
        for name, top_5_features, kf in zip(dataset_name, important_feature_combinations, kinetic_features):
            clean_title = name.replace("_", " ").title()
            top_3_features = top_5_features[:3]
            df_top5 = kf[top_5_features].replace([np.inf, -np.inf], np.nan).fillna(0)
            
            tsne = TSNE(n_components=2, random_state=0)
            X_tsne = tsne.fit_transform(df_top5.values)
            
            fig = plt.figure(figsize=(22, 9))
            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            ax1.scatter(df_top5[top_3_features[0]], df_top5[top_3_features[1]], df_top5[top_3_features[2]], c=colors, cmap='tab10', s=40, alpha=0.8, edgecolor='k')
            ax1.set_title(f"{clean_title}\n(Top 3 Features: {', '.join(top_3_features)})", fontsize=14, fontweight='bold', pad=15)
            
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, cmap='tab10', s=40, alpha=0.8, edgecolor='k')
            ax2.set_title(f"{clean_title}\n(t-SNE on All 5: {', '.join(top_5_features)})", fontsize=14, fontweight='bold', pad=15)
            ax2.grid(True, linestyle='--', alpha=0.6)
            
            buf = BytesIO(); plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            buf.seek(0); tsne_buffers.append(buf)
            plt.close(fig); gc.collect()

        save_html_report(os.path.join(exp_path, "independent_features_tsne_and_3d.html"), "Top 5 Independent Features Dimensionality Reduction", "Visualizing the highest importance features after removing mathematical redundancies.", tsne_buffers)

        # -------------------------------------------------------------
        # 8. OUTLIER DETECTION PIPELINES
        # -------------------------------------------------------------
        print("\n=== RUNNING OUTLIER DETECTION PIPELINES ===")
        ref_curves = dataset[0]
        all_new_feature_dfs = [[] for _ in range(len(dataset_name))] 

        # --- 8.0: LOAD PREVIOUS PROGRESS IF IT EXISTS ---
        updated_save_path = os.path.join(exp_path, "curve_for_training_latest.pkl")
        if os.path.exists(updated_save_path):
            print(f"  -> Found existing progress in {updated_save_path}. Loading to resume...")
            with open(updated_save_path, 'rb') as f:
                saved_data = pickle.load(f)
            # Update kinetic_features with the one that already has computed outlier columns
            kinetic_features = saved_data["kinetic_features"]

        def save_incremental_progress():
            save_data = {
                "timestamps": timestamps,
                "Y_well": Y_well,
                "dataset_name": dataset_name,
                "dataset": dataset,
                "kinetic_features": kinetic_features,
                "linear_feature_combinations": linear_feature_combinations,
                "important_feature_combinations": important_feature_combinations
            }
            with open(updated_save_path, 'wb') as f:
                pickle.dump(save_data, f)
            print(f"    [SAVED] Progress incrementally written to disk.")
            
        # -------------------------------------------------------------
        # MSC Configs
        # -------------------------------------------------------------
        msc_configs = [
            # ("msc_linear_0.05", 0.05, linear_feature_combinations),
            # ("msc_linear_0.01", 0.01, linear_feature_combinations),
            ("msc_linear_0.001", 0.001, linear_feature_combinations),
            # ("msc_baseline_0.05", 0.05, [["Ct", "Cy0", "log_F0"]] * len(dataset_name)),
            # ("msc_baseline_0.01", 0.01, [["Ct", "Cy0", "log_F0"]] * len(dataset_name)),
            ("msc_baseline_0.001", 0.001, [["Ct", "Cy0", "log_F0"]] * len(dataset_name))
        ]
        
        # -------------------------------------------------------------
        # AMF Configs
        # -------------------------------------------------------------
        amf_configs = [
            ("amf_important", important_feature_combinations),
            ("amf_send_5", [["Fm", "Fb", "Sc", "Cs", "send_5"]] * len(dataset_name)),
            # ("amf_send_10", [["Fm", "Fb", "Sc", "Cs", "send_10"]] * len(dataset_name)),
            # ("amf_send_15", [["Fm", "Fb", "Sc", "Cs", "send_15"]] * len(dataset_name)),
            # ("amf_send_20", [["Fm", "Fb", "Sc", "Cs", "send_20"]] * len(dataset_name)),
            # ("amf_send_25", [["Fm", "Fb", "Sc", "Cs", "send_25"]] * len(dataset_name)),
            # ("amf_send_abs_5", [["Fm", "Fb", "Sc", "Cs", "send_abs_5"]] * len(dataset_name)),
            # ("amf_send_abs_10", [["Fm", "Fb", "Sc", "Cs", "send_abs_10"]] * len(dataset_name)),
            # ("amf_send_abs_15", [["Fm", "Fb", "Sc", "Cs", "send_abs_15"]] * len(dataset_name)),
            # ("amf_send_abs_20", [["Fm", "Fb", "Sc", "Cs", "send_abs_20"]] * len(dataset_name)),
            # ("amf_send_abs_25", [["Fm", "Fb", "Sc", "Cs", "send_abs_25"]] * len(dataset_name))
        ]
        
        # -------------------------------------------------------------
        # Mean Std Configs
        # -------------------------------------------------------------
        mean_std_configs = [
            # ("env_1std", 1), 
            # ("env_2std", 2), 
            # ("env_3std", 3)
        ]
        
        # -------------------------------------------------------------
        # KNN Fingerprint Configs
        # -------------------------------------------------------------
        knn_filter_config = [0.85, 0.90, 0.95]

        # -------------------------------------------------------------
        # Autoencoder Config
        # -------------------------------------------------------------
        ae_configs = ["elbow", 90, 95]        
        downsample_factor=1

        # Only Original Curves for Auto Encoder
        ae_filtered_names = []
        ae_filtered_dataset = []

        for name, data, features in zip(dataset_name, dataset, kinetic_features):
            if name == 'ori_curves':
                ae_filtered_names.append(name)
                ae_filtered_dataset.append(data)

        ae_dataset_name = ae_filtered_names
        ae_dataset = ae_filtered_dataset
        
        # -------------------------------------------------------------
        # 8.1: OUTLIER DETECTION METHODS
        # -------------------------------------------------------------
        
        # --- CNN AutoEncoder Per Well ---
        expected_cnn_pw = [f"cnn_ae_pw_ds{downsample_factor}_label_{pct}" for pct in ae_configs]
        missing_cnn_pw = [pct for pct, label in zip(ae_configs, expected_cnn_pw) if label not in kinetic_features[0].columns]
        if missing_cnn_pw:
            extracted_dfs = run_cnn_autoencoder_pipeline(ae_dataset_name, ae_dataset, Y_well, ref_curves, f"{exp_path}/ae_outlier", missing_cnn_pw, save_plot=("D20250808_E00_C00_F4500KHz_U_Sample_7" in str(exp_path)), downsample_factor=downsample_factor, per_well=True)
            for i, name in enumerate(ae_dataset_name): 
                master_idx = list(dataset_name).index(name)
                kinetic_features[master_idx] = pd.concat([kinetic_features[master_idx], extracted_dfs[i]], axis=1)
            save_incremental_progress()
        else:
            print("  -> [SKIP] CNN AutoEncoder (Per-Well): Already calculated.")

        # --- CNN AutoEncoder Whole Chip ---
        expected_cnn_glb = [f"cnn_ae_glb_ds{downsample_factor}_label_{pct}" for pct in ae_configs]
        missing_cnn_glb = [pct for pct, label in zip(ae_configs, expected_cnn_glb) if label not in kinetic_features[0].columns]
        if missing_cnn_glb:
            extracted_dfs = run_cnn_autoencoder_pipeline(ae_dataset_name, ae_dataset, Y_well, ref_curves, f"{exp_path}/ae_outlier", missing_cnn_glb, save_plot=("D20250808_E00_C00_F4500KHz_U_Sample_7" in str(exp_path)), downsample_factor=downsample_factor, per_well=False)
            for i, name in enumerate(ae_dataset_name): 
                master_idx = list(dataset_name).index(name)
                kinetic_features[master_idx] = pd.concat([kinetic_features[master_idx], extracted_dfs[i]], axis=1)
            save_incremental_progress()
        else:
            print("  -> [SKIP] CNN AutoEncoder (Global): Already calculated.")

        # --- LSTM AutoEncoder Per Well ---
        expected_lstm_pw = [f"lstm_ae_pw_ds{downsample_factor}_label_{pct}" for pct in ae_configs]
        missing_lstm_pw = [pct for pct, label in zip(ae_configs, expected_lstm_pw) if label not in kinetic_features[0].columns]
        if missing_lstm_pw:
            extracted_dfs = run_lstm_autoencoder_pipeline(ae_dataset_name, ae_dataset, Y_well, ref_curves, f"{exp_path}/ae_per_well_outlier", missing_lstm_pw, save_plot=("D20250808_E00_C00_F4500KHz_U_Sample_7" in str(exp_path)), downsample_factor=downsample_factor, per_well=True)
            for i, name in enumerate(ae_dataset_name): 
                master_idx = list(dataset_name).index(name)
                kinetic_features[master_idx] = pd.concat([kinetic_features[master_idx], extracted_dfs[i]], axis=1)
            save_incremental_progress()
        else:
            print("  -> [SKIP] LSTM AutoEncoder (Per-Well): Already calculated.")
        
        # --- LSTM AutoEncoder Whole Chip ---
        expected_lstm_glb = [f"lstm_ae_glb_ds{downsample_factor}_label_{pct}" for pct in ae_configs]
        missing_lstm_glb = [pct for pct, label in zip(ae_configs, expected_lstm_glb) if label not in kinetic_features[0].columns]
        if missing_lstm_glb:
            extracted_dfs = run_lstm_autoencoder_pipeline(ae_dataset_name, ae_dataset, Y_well, ref_curves, f"{exp_path}/ae_per_well_outlier", missing_lstm_glb, save_plot=("D20250808_E00_C00_F4500KHz_U_Sample_7" in str(exp_path)), downsample_factor=downsample_factor, per_well=False)
            for i, name in enumerate(ae_dataset_name): 
                master_idx = list(dataset_name).index(name)
                kinetic_features[master_idx] = pd.concat([kinetic_features[master_idx], extracted_dfs[i]], axis=1)
            save_incremental_progress()
        else:
            print("  -> [SKIP] LSTM AutoEncoder (Global): Already calculated.")
        
        # # Run AutoEncoder Per Well
        # extracted_dfs = run_autoencoder_pipeline(ae_dataset_name, ae_dataset, Y_well, ref_curves, f"{exp_path}/ae_per_well_outlier", ae_configs, save_plot=("D20250808_E00_C00_F4500KHz_U_Sample_7" in str(exp_path)), downsample_factor=downsample_factor, per_well=per_well)
        # for i in range(len(ae_dataset_name)): all_new_feature_dfs[i].append(extracted_dfs[i])

        # --- KNN Filter ---
        expected_knn = [f"knn_top_{pct}" for pct in knn_filter_config]
        missing_knn = [pct for pct, label in zip(knn_filter_config, expected_knn) if label not in kinetic_features[0].columns]
        if missing_knn:
            extracted_dfs = run_knnfilter_pipeline(dataset_name, dataset, Y_well, ref_curves, os.path.join(exp_path, "knnfilter_outlier"), missing_knn, save_plot=("D20250808_E00_C00_F4500KHz_U_Sample_7" in str(exp_path)))
            for i in range(len(dataset_name)): 
                kinetic_features[i] = pd.concat([kinetic_features[i], extracted_dfs[i]], axis=1)
            save_incremental_progress()
        else:
            print("  -> [SKIP] KNN Filter: Already calculated.")

        # --- MSC Filter ---
        msc_configs_to_run = []
        for exp_label, p_val, feats in msc_configs:
            if f"msc_label_{exp_label}" not in kinetic_features[0].columns:
                msc_configs_to_run.append((exp_label, p_val, feats))
            else:
                print(f"  -> [SKIP] MSC [{exp_label}]: Already calculated.")
                
        if msc_configs_to_run:
            for exp_label, p_val, feats in msc_configs_to_run:
                extracted_dfs = run_msc_pipeline(exp_label, p_val, ref_curves, os.path.join(exp_path, "msc_outlier"), dataset_name, kinetic_features, dataset, Y_well, feats, save_plot=False)
                for i in range(len(dataset_name)): 
                    kinetic_features[i] = pd.concat([kinetic_features[i], extracted_dfs[i]], axis=1)
            save_incremental_progress()

        # --- AMF Filter ---
        amf_configs_to_run = []
        for exp_label, feats in amf_configs:
            if f"amf_label_{exp_label}" not in kinetic_features[0].columns:
                amf_configs_to_run.append((exp_label, feats))
            else:
                print(f"  -> [SKIP] AMF [{exp_label}]: Already calculated.")
                
        if amf_configs_to_run:
            for exp_label, feats in amf_configs_to_run:
                extracted_dfs = run_amf_pipeline(exp_label, feats, ref_curves, os.path.join(exp_path, "amf_outlier"), dataset_name, kinetic_features, dataset, Y_well, save_plot=False)
                for i in range(len(dataset_name)): 
                    kinetic_features[i] = pd.concat([kinetic_features[i], extracted_dfs[i]], axis=1)
            save_incremental_progress()

        # --- Mean/Std Filter ---
        mean_std_configs_to_run = []
        for exp_label, num_std in mean_std_configs:
            if f"mean_std_label_{exp_label}" not in kinetic_features[0].columns:
                mean_std_configs_to_run.append((exp_label, num_std))
            else:
                print(f"  -> [SKIP] Mean/Std [{exp_label}]: Already calculated.")
                
        if mean_std_configs_to_run:
            for exp_label, num_std in mean_std_configs_to_run:
                extracted_dfs = run_meanstd_pipeline(exp_label, num_std, ref_curves, os.path.join(exp_path, "meanstd_outlier"), dataset_name, dataset, Y_well, kinetic_features[0].index, save_plot=False)
                for i in range(len(dataset_name)): 
                    kinetic_features[i] = pd.concat([kinetic_features[i], extracted_dfs[i]], axis=1)
            save_incremental_progress()

        # # Run Mean/Std
        # for exp_label, num_std in mean_std_configs:
        #     extracted_dfs = run_meanstd_pipeline(exp_label, num_std, ref_curves, os.path.join(exp_path, "meanstd_outlier"), dataset_name, dataset, Y_well, kinetic_features[0].index, save_plot=False)
        #     for i in range(len(dataset_name)): all_new_feature_dfs[i].append(extracted_dfs[i])

        # -------------------------------------------------------------
        # 9. FINAL MERGE & SAVE
        # -------------------------------------------------------------
        print(f"\nExperiment {exp_path.name} finished gracefully!")
        
    del data, Y_well, timestamps, metadata_df, dataset, dataset_name
    del kinetic_features, linear_feature_combinations, important_feature_combinations
    
    tf.keras.backend.clear_session()
    gc.collect()