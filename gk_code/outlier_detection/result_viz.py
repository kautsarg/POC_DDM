import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
from sklearn.metrics import accuracy_score, confusion_matrix

# Set global seaborn theme
sns.set_theme(style="white", font_scale=1.1)

def plot_flexible_grouped_bar(df, x_col, y_col, hue_col, ax, title=None, is_percentage=True, is_horizontal=False, baseline_value=None):
    """
    A highly flexible grouped bar chart generator using Seaborn.
    Renders the plot onto the provided matplotlib 'ax'.
    """
    # 1. Map Data Based on Orientation
    x_data = y_col if is_horizontal else x_col
    y_data = x_col if is_horizontal else y_col
    
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

root_path = "/Users/kautsarg/Documents/Final Project/Run Data/trial test data/"
sample_names = [name for name in os.listdir(root_path) if name != ".DS_Store"]

EXPECTED_CLASSES = list(range(10)) 

model_key_mapping = {
    "CNN": "y_preds_AC_",
    # "KNN": "y_preds_AC_kNN_",
    # "LR":  "y_preds_FFI_"
}

for sample_name in sample_names:
    exp_path = Path(root_path, sample_name)
    data_path = os.path.join(exp_path, "classification_performances_with_proba.pkl")
    print(f"**********PROCESSING {sample_name}**********")
    # Skip if file doesn't exist
    if not os.path.exists(data_path):
        print(f"[WARNING] {data_path} is not exist.\n")
        continue
        
    with open(data_path, 'rb') as f:
        result_data = pickle.load(f)    

    rows = []
    print(f"[*] Flattening dictionary into tabular format for {sample_name}...")
    
    for curve_name, modes in result_data.items():
        for mode_name, filters in modes.items():
            total_samples = max(metrics.get("mask_count", 0) for metrics in filters.values())
            
            for filter_name, metrics in filters.items():
                clean_filter = filter_name if filter_name is not None else "Baseline_None"
                surviving_samples = metrics.get("mask_count", total_samples)
                outlier_pct = ((total_samples - surviving_samples) / total_samples) * 100
                
                y_true = np.concatenate(metrics["y_trues_"])
                
                for clean_model_name, dict_key in model_key_mapping.items():
                    # Handle edge cases where models might not exist for a filter
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
                        col_name = f"accuracy_well_{well_idx}"
                        row_data[col_name] = per_class_acc[well_idx]
                        
                    rows.append(row_data)

    result_df = pd.DataFrame(rows)

    # Save to CSV
    csv_out = os.path.join(exp_path, "flattened_results.csv")
    result_df["accuracy_well_avg"] = np.average(result_df.filter(like="accuracy_well_"), axis=1)
    result_df.to_csv(csv_out, index=False)
    print(f"[*] Saved flat CSV to: {csv_out}")

    # Extract Baseline
    try:
        baseline_mask = (result_df["curve_name"] == "Ori Curves") & \
                        (result_df["outlier_filter"] == "Baseline_None") & \
                        (result_df["training_mode"] == "Reference")
        baseline_value = result_df[baseline_mask].total_accuracy.values[0]
    except IndexError:
        baseline_value = None

    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
    
    # 1. Plot Accuracy
    plot_flexible_grouped_bar(
        df=result_df,
        x_col="outlier_filter",
        y_col="total_accuracy",
        hue_col="curve_name",
        ax=axes[0],
        title=f"Model Accuracy Comparison\n{sample_name}",
        is_percentage=True,
        is_horizontal=True,
        baseline_value=baseline_value
    )

    # 2. Plot Outlier Percentage
    plot_flexible_grouped_bar(
        df=result_df,
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
    # os.system(f"open '{combined_plot_path}'")