import os
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
from sklearn.metrics import accuracy_score, confusion_matrix
import config
import plotly.express as px

# Set global seaborn theme
sns.set_theme(style="white", font_scale=1.1)

def export_interactive_html(master_df, out_file="pipeline_comparison_report.html"):
    """
    Generates a highly interactive HTML dashboard comparing Models, Experiments, 
    Curve Types, and Outlier Filters.
    """
    print(f"\n[*] Generating Interactive HTML Report...")
    df = master_df.copy()
    
    # 1. Format Data
    df['Accuracy (%)'] = df['total_accuracy'] * 100
    
    def clean_filter_name(f):
        if pd.isna(f) or f == "Baseline_None" or f == "None (Baseline)": return "Baseline"
        f_str = str(f)
        if "msc_label" in f_str: return f_str.replace("msc_label_msc_", "MSC: ")
        if "amf_label" in f_str: return f_str.replace("amf_label_amf_", "AMF: ")
        if "knn_top" in f_str: return f_str.replace("knn_top_", "KNN: ")
        if "cnn_ae" in f_str: return f_str.replace("cnn_ae_", "CNN AE: ")
        if "lstm_ae" in f_str: return f_str.replace("lstm_ae_", "LSTM AE: ")
        return f_str[:15] + ".." if len(f_str) > 15 else f_str

    df['Clean_Filter'] = df['outlier_filter'].apply(clean_filter_name)

    # 2. Build the Plotly Facet Grid
    # - X-axis: Outlier Filters
    # - Y-axis: Accuracy
    # - Color: Model Architecture
    # - Subplots (Facets): Curve Types
    # - Lines (Grouping): Connects points from the same Experiment (sample_name)
    fig = px.line(
        df, 
        x="Clean_Filter", 
        y="Accuracy (%)", 
        color="model",
        facet_col="curve_name", 
        facet_col_wrap=3,          # Wraps to a new row after 3 columns
        line_group="sample_name",  # Crucial: Separates lines by experiment
        hover_name="sample_name",  # Shows Experiment name prominently on hover
        hover_data={
            "outlier_percentage": True,
            "Clean_Filter": False,
            "curve_name": False,
            "model": False
        },
        markers=True,
        title="<b>Interactive ML Pipeline Comparison</b><br><sup>Hover over points for Experiment details. Click Legend items to toggle models.</sup>",
        template="plotly_white"
    )
    
    # 3. Enhance Visual Layout
    fig.update_layout(
        height=900, 
        width=1600,
        hovermode="closest",
        legend_title_text='Model Architecture',
        font=dict(size=12)
    )
    
    # Clean up facet titles (removes "curve_name=" from the subplot headers)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    
    # Fix X-axis labels overlapping
    fig.update_xaxes(tickangle=45, title_text="")
    fig.update_yaxes(matches=None) # Allows Y-axis to scale dynamically per row if needed

    # 4. Save to HTML
    fig.write_html(out_file)
    print(f"[*] Successfully saved interactive report to: {out_file}")

def plot_flexible_grouped_bar(df, x_col, y_col, hue_col, ax, title=None, is_percentage=True, is_horizontal=False, baseline_value=None):
    """
    A highly flexible grouped bar chart generator using Seaborn.
    Renders the plot onto the provided matplotlib 'ax'.
    """
    # 1. Map Data Based on Orientation
    x_data = y_col if is_horizontal else x_col
    y_data = x_col if is_horizontal else y_col
    
    # Note: If multiple models are present, sns.barplot will automatically plot the MEAN 
    # accuracy across all models for that specific curve/filter, with error bars for variance.
    sns.barplot(data=df, x=x_data, y=y_data, hue=hue_col, palette="viridis", edgecolor="black", linewidth=1, ax=ax)
    
    ax.set_xlabel(x_data.replace("_", " ").title(), fontsize=14, fontweight='bold')
    ax.set_ylabel(y_data.replace("_", " ").title(), fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=0 if is_horizontal else 90)
    ax.tick_params(axis='y', rotation=0) 
    
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
    # 2. Dynamic Axis & Grid Formatting
    val_axis = ax.xaxis if is_horizontal else ax.yaxis
    grid_axis = 'x' if is_horizontal else 'y'
    
    if is_percentage:
        if is_horizontal: 
            ax.set_xlim(0, 1.0)
        else: 
            ax.set_ylim(0, 1.0)
            
        val_axis.set_major_locator(mtick.MultipleLocator(0.1))
        val_axis.set_minor_locator(mtick.MultipleLocator(0.01))
        val_axis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
        
        ax.grid(which='major', axis=grid_axis, linestyle='-', linewidth=1.5, color='#eaeaea')
        ax.grid(which='minor', axis=grid_axis, linestyle=':', linewidth=1, color='#cccccc', alpha=0.7)
    else:
        ax.grid(which='major', axis=grid_axis, linestyle='-', linewidth=1.5, color='#eaeaea')
        
    ax.set_axisbelow(True) 
    
    # 3. Draw Baseline Threshold
    if baseline_value is not None:
        if is_horizontal:
            ax.axvline(x=baseline_value, color='#e74c3c', linestyle='--', linewidth=2.5, zorder=5)
        else:
            ax.axhline(y=baseline_value, color='#e74c3c', linestyle='--', linewidth=2.5, zorder=5)
    
    # 4. Dynamic Bar Annotations (Text)
    for container in ax.containers:
        labels = []
        for bar in container:
            val = bar.get_width() if is_horizontal else bar.get_height()
            
            if val > 0:
                labels.append(f"{val*100:.1f}%" if is_percentage else f"{val:.2f}")
            else:
                labels.append("")
        
        text_padding = -35 if is_percentage else 5
        text_color = 'white' if is_percentage else '#333333'
        
        ax.bar_label(
            container, 
            labels=labels, 
            label_type='edge',     
            padding=text_padding,          
            fontsize=10, 
            fontweight='bold', 
            color=text_color,         
            rotation=0 if is_horizontal else 90            
        )
    
    # 5. Legend
    legend_title = hue_col.replace("_", " ").title()
    ax.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc='upper left')


# ==========================================
# MAIN EXECUTION
# ==========================================

root_path = config.DEFAULT_EXP_FOLDER
sample_names = [name for name in os.listdir(root_path) if name != ".DS_Store"]
EXPECTED_CLASSES = list(range(10))

master_rows = []

# --- UPDATED: All Latest Model Dictionary Mappings ---
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
        
    result_data = joblib.load(data_path)    
    rows = []
    
    for curve_name, modes in result_data.items():
        for mode_name, filters in modes.items():
            total_samples = max(metrics.get("mask_count", 0) for metrics in filters.values())
            
            for filter_name, metrics in filters.items():
                clean_filter = filter_name if filter_name is not None else "Baseline_None"
                surviving_samples = metrics.get("mask_count", total_samples)
                outlier_pct = ((total_samples - surviving_samples) / total_samples) * 100
                
                y_true = np.concatenate(metrics["y_trues_"])
                
                for clean_model_name, dict_key in model_key_mapping.items():
                    if dict_key not in metrics: continue
                        
                    y_pred = np.concatenate(metrics[dict_key])
                    total_acc = accuracy_score(y_true, y_pred)
                    
                    row_data = {
                        "sample_name": sample_name,
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
                        col_name = f"accuracy_well_{well_idx}"
                        row_data[col_name] = per_class_acc[well_idx]
                        
                    rows.append(row_data)

    result_df = pd.DataFrame(rows)
    if result_df.empty: continue
    
    master_rows.extend(rows)

    if result_df.empty:
        print("[WARNING] No valid data found in this result file.\n")
        continue

    # Save to CSV
    csv_out = os.path.join(exp_path, "flattened_results.csv")
    result_df["accuracy_well_avg"] = np.average(result_df.filter(like="accuracy_well_"), axis=1)
    result_df.to_csv(csv_out, index=False)
    print(f"[*] Saved flat CSV to: {csv_out}")

    # Extract Baseline (Mean across all models for the pure original curve)
    try:
        baseline_mask = (result_df["curve_name"] == "Ori Curves") & \
                        (result_df["outlier_filter"] == "Baseline_None") & \
                        (result_df["training_mode"] == "Reference")
        baseline_value = result_df[baseline_mask].total_accuracy.mean()
    except IndexError:
        baseline_value = None

    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
    
    # --- UPDATED: Uses dynamic f-strings for the AE downsample factors ---
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

    # Tip: If you want to see models instead of curves on the Y-Axis legend, 
    # simply change `hue_col="curve_name"` below to `hue_col="model"`
    
    # 1. Plot Accuracy
    plot_flexible_grouped_bar(
        df=plot_df,
        x_col="outlier_filter",
        y_col="total_accuracy",
        hue_col="curve_name",
        ax=axes[0],
        title=f"Model Accuracy Comparison (Averaged over all architectures)\n{sample_name}",
        is_percentage=True,
        is_horizontal=True,
        baseline_value=baseline_value
    )

    # 2. Plot Outlier Percentage
    plot_flexible_grouped_bar(
        df=plot_df,
        x_col="outlier_filter",
        y_col="outlier_percentage",
        hue_col="curve_name",
        ax=axes[1],
        title=f"Outlier Percentage Comparison\n{sample_name}",
        is_percentage=False,
        is_horizontal=True
    )
    
    plt.tight_layout()
    
    combined_plot_path = os.path.join(root_path, f"{sample_name}_model_performance_comparison.png")
    fig.savefig(combined_plot_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig)
    print(f"[*] Saved combined side-by-side plot to: {combined_plot_path}\n")

    if master_rows:
        master_df = pd.DataFrame(master_rows)
        
        # Filter only the Reference mode and relevant curves to keep the HTML clean
        html_mask = (master_df["training_mode"] == "Reference") & \
                    (master_df["curve_name"].isin([
                        'Ori Curves', 'Original Fitted Full', 'Cleaned Std Fitted Full', 
                        'Cleaned Std Fitted Stretched', 'Cleaned Lowest Fitted Full', 
                        'Cleaned Lowest Fitted Stretched'
                    ]))
                    
        master_plot_df = master_df[html_mask].copy()
        
        html_output_path = os.path.join(root_path, "Master_Pipeline_Comparison.html")
        export_interactive_html(master_plot_df, out_file=html_output_path)