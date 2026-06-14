import os
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

import config

# Set global seaborn theme
sns.set_theme(style="white", font_scale=1.1)

# ====================================================================
# LINE PLOT FUNCTION 
# ====================================================================
def plot_dataset_results_line(df_long, sample_name, save_dir=None):
    """
    Generates line plots for a single dataset across all curves and filters.
    Adapts a long-format DataFrame to the requested visual style.
    """
    def clean_filter_name(f):
        if pd.isna(f) or f == "Baseline_None" or f == "None (Baseline)": return "Baseline"
        f_str = str(f)
        if "msc_label" in f_str: return f_str.replace("msc_label_msc_", "MSC: ")
        if "amf_label" in f_str: return f_str.replace("amf_label_amf_", "AMF: ")
        if "knn_top" in f_str: return f_str.replace("knn_top_", "KNN: ")
        if "cnn_ae" in f_str: return f_str.replace("cnn_ae_", "CNN AE: ")
        if "lstm_ae" in f_str: return f_str.replace("lstm_ae_", "LSTM AE: ")
        return f_str[:15] + ".." if len(f_str) > 15 else f_str

    # Make a copy and convert accuracy to percentage for the Y-Axis
    df = df_long.copy()
    df['total_accuracy'] *= 100 
    df['Clean_Filter'] = df['outlier_filter'].apply(clean_filter_name)

    curves = df['curve_name'].unique()

    # Expanded styles to cover all deep learning and classical models.
    # Colors come from config.MODEL_COLORS so each model is drawn in the same
    # color across line plots, bar charts, and the interactive HTML report.
    styles = {
        'CNN': {'color': config.MODEL_COLORS['CNN'], 'marker': 'o', 'label': 'CNN'},
        'LSTM': {'color': config.MODEL_COLORS['LSTM'], 'marker': 's', 'label': 'LSTM'},
        'GRU': {'color': config.MODEL_COLORS['GRU'], 'marker': '^', 'label': 'GRU'},
        'RNN': {'color': config.MODEL_COLORS['RNN'], 'marker': 'D', 'label': 'RNN'},
        'Transformer': {'color': config.MODEL_COLORS['Transformer'], 'marker': 'v', 'label': 'Transformer'},
        'RandomForest': {'color': config.MODEL_COLORS['RandomForest'], 'marker': '*', 'label': 'Random Forest'},
        'KNN': {'color': config.MODEL_COLORS['KNN'], 'marker': 'p', 'label': 'KNN'},
        'LR (FFI)': {'color': config.MODEL_COLORS['LR (FFI)'], 'marker': 'h', 'label': 'LR (FFI)'}
    }

    for c_name in curves:
        df_sub = df[df['curve_name'] == c_name].copy()
        if df_sub.empty: continue

        # Maintain original filter order
        filter_order = df_sub['Clean_Filter'].unique() 

        # Pivot to Wide format: Index = Filters, Columns = Models
        df_wide = df_sub.pivot(index='Clean_Filter', columns='model', values='total_accuracy')
        df_wide = df_wide.reindex(filter_order) # Restore proper order

        x_labels = df_wide.index.tolist()
        x_pos = np.arange(len(x_labels))

        fig, ax = plt.subplots(figsize=(14, 7))
        
        baseline_row = df_wide.loc['Baseline'] if 'Baseline' in df_wide.index else None
        available_models = [m for m in df_wide.columns if m in styles]

        for model_col in available_models:
            style = styles[model_col]
            y_vals = df_wide[model_col].values
            
            # Main Line Plot
            ax.plot(x_pos, y_vals, color=style['color'], marker=style['marker'], 
                    linewidth=2, markersize=8, alpha=0.85, label=style['label'])
            
            # Horizontal Dashed Baseline
            if baseline_row is not None and not pd.isna(baseline_row[model_col]):
                base_val = baseline_row[model_col]
                ax.axhline(base_val, color=style['color'], linestyle='--', linewidth=1.5, alpha=0.4)

        ax.set_title(f"Model Performance Across Outlier Filters\nDataset: {sample_name} | Curve: {c_name}", 
                     fontsize=16, fontweight='bold', pad=15)
        ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
        
        # Dynamic Y limits with padding
        valid_accs = df_wide.values.flatten()
        valid_accs = valid_accs[~np.isnan(valid_accs)]
        if len(valid_accs) > 0:
            ax.set_ylim(max(0, np.min(valid_accs) - 5), min(100, np.max(valid_accs) + 5))
        ax.set_ylim(60, 100)
        ax.set_yticks(range(60, 101, 5))

        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax.grid(True, axis='x', linestyle=':', alpha=0.2)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11)
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            safe_c = c_name.replace("/", "_").replace(" ", "_")
            plt.savefig(os.path.join(save_dir, f"{sample_name}_{safe_c}_LinePlot.png"), dpi=300, bbox_inches='tight')
            
        plt.show() 
        plt.close(fig)


# ==========================================
# MAIN EXECUTION
# ==========================================

root_path = config.DEFAULT_EXP_FOLDER
sample_names = [name for name in os.listdir(root_path) if name != ".DS_Store"]
EXPECTED_CLASSES = list(range(10)) 

model_key_mapping = {
    "CNN": "y_preds_AC_",
    "LSTM": "y_preds_AC_lstm_",
    "GRU": "y_preds_AC_gru_",
    "RNN": "y_preds_AC_rnn_",
    "Transformer": "y_preds_AC_trans_",
    "RandomForest": "y_preds_AC_rf_",
    "KNN": "y_preds_AC_kNN_",
    "LR (FFI)": "y_preds_FFI_"
}

for sample_name in sample_names:
    exp_path = Path(root_path, sample_name)
    data_path = os.path.join(exp_path, config.TRAINING_RESULT_PATH)
    
    print(f"\n{'='*60}")
    print(f"PROCESSING: {sample_name}")
    print(f"{'='*60}")
    
    if not os.path.exists(data_path):
        print(f"[WARNING] {data_path} does not exist.\n")
        continue
        
    # Load via Joblib
    result_data = joblib.load(data_path)    

    rows = []
    print(f"[*] Flattening dictionary into tabular format...")
    
    for curve_name, modes in result_data.items():
        for mode_name, filters in modes.items():
            total_samples = max(metrics.get("mask_count", 0) for metrics in filters.values())
            
            for filter_name, metrics in filters.items():
                clean_filter = filter_name if filter_name is not None else "Baseline_None"
                surviving_samples = metrics.get("mask_count", total_samples)
                outlier_pct = ((total_samples - surviving_samples) / total_samples) * 100
                
                if "y_trues_" not in metrics:
                    continue
                    
                y_true = np.concatenate(metrics["y_trues_"])
                
                for clean_model_name, dict_key in model_key_mapping.items():
                    if dict_key not in metrics:
                        continue
                        
                    y_pred = np.concatenate(metrics[dict_key])
                    total_acc = accuracy_score(y_true, y_pred)
                    
                    row_data = {
                        "curve_name": curve_name,
                        "training_mode": mode_name,
                        "outlier_filter": clean_filter,
                        "model": clean_model_name,
                        "outlier_percentage": round(outlier_pct, 2), 
                        "total_accuracy": total_acc
                    }
                    
                    cm = confusion_matrix(y_true, y_pred, labels=EXPECTED_CLASSES)
                    
                    with np.errstate(divide='ignore', invalid='ignore'):
                        per_class_acc = cm.diagonal() / cm.sum(axis=1)
                    
                    for well_idx in EXPECTED_CLASSES:
                        row_data[f"accuracy_well_{well_idx}"] = per_class_acc[well_idx]
                        
                    rows.append(row_data)

    result_df = pd.DataFrame(rows)

    if result_df.empty:
        print("[WARNING] No valid data found in this result file.\n")
        continue

    # Save to CSV
    csv_out = os.path.join(exp_path, "flattened_results.csv")
    result_df["accuracy_well_avg"] = np.average(result_df.filter(like="accuracy_well_"), axis=1)
    result_df.to_csv(csv_out, index=False)
    print(f"[*] Saved flat CSV to: {csv_out}")

    # --- PLOTTING ---
    df_mask = (result_df["training_mode"] == "Reference") & \
              (result_df["outlier_filter"].isin([
                  "Baseline_None", 
                  "msc_label_msc_linear_0.001", 
                  "msc_label_msc_baseline_0.001", 
                  "amf_label_amf_important",  
                  "amf_label_amf_send_5", 
                  "knn_top_0.9", 
                  "knn_top_0.95", 
                  f"cnn_ae_pw_ds{config.AE_DOWNSAMPLE_FACTOR}_label_95", 
                  f"cnn_ae_pw_ds{config.AE_DOWNSAMPLE_FACTOR}_label_elbow",
                  f"cnn_ae_glb_ds{config.AE_DOWNSAMPLE_FACTOR}_label_95", 
                  f"cnn_ae_glb_ds{config.AE_DOWNSAMPLE_FACTOR}_label_elbow",
                  f"lstm_ae_pw_ds{config.AE_DOWNSAMPLE_FACTOR}_label_95", 
                  f"lstm_ae_pw_ds{config.AE_DOWNSAMPLE_FACTOR}_label_elbow",
                  f"lstm_ae_glb_ds{config.AE_DOWNSAMPLE_FACTOR}_label_95", 
                  f"lstm_ae_glb_ds{config.AE_DOWNSAMPLE_FACTOR}_label_elbow"
              ])) & \
              (result_df["curve_name"].isin([
                  'Ori Curves', 'Original Fitted Full', 'Cleaned Std Fitted Full', 
                  'Cleaned Std Fitted Stretched', 'Cleaned Lowest Fitted Full', 
                  'Cleaned Lowest Fitted Stretched'
              ]))
              
    plot_df = result_df[df_mask].copy()

    if not plot_df.empty:
        print(f"[*] Generating line plots for {sample_name}...")
        # This function loops internally to generate a plot for EACH curve type
        plot_dataset_results_line(plot_df, sample_name=sample_name, save_dir=exp_path)
    else:
        print("[WARNING] Filter mask resulted in empty dataframe, skipping plots.")