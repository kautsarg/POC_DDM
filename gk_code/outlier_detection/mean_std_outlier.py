import os
import gc
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from outlier_utils import *

def draw_thresholds(ax, curves, mean_val, std_val, num_std, line_color, title_text):
    if len(curves) > 0:
        ax.plot(curves.T, c=line_color, alpha=0.3, rasterized=True)
        
    if mean_val is not None and std_val is not None:
        ax.plot(mean_val, c="black", linewidth=2.0, label="Mean")
        
        upper_bound = mean_val + (num_std * std_val)
        lower_bound = mean_val - (num_std * std_val)
        
        ax.plot(upper_bound, c="black", linestyle="--", linewidth=1.5, label=f"Outlier Threshold (±{num_std} Std)")
        ax.plot(lower_bound, c="black", linestyle="--", linewidth=1.5)
        
        ax.legend(loc="upper left", fontsize=8)
        
    ax.set_title(title_text, fontweight='bold')


def run_meanstd_pipeline(exp_label, num_std, ref_curves, meanstd_plot_path, dataset_names, dataset_curves, Y_well, features_index):
    os.makedirs(meanstd_plot_path, exist_ok=True)
    results_dfs = []
    
    for name, curves in zip(dataset_names, dataset_curves):
        clean_title = name.replace("_", " ").title()
        print(f"  -> Running Mean/Std [{exp_label}] for {clean_title}...")
        
        # 1. Detection Phase
        new_features = pd.DataFrame(index=features_index)
        label_col = f"mean_std_label_{exp_label}"
        new_features[label_col] = np.nan
        
        unique_wells = np.unique(Y_well)
        for well in unique_wells:
            well_mask = (Y_well == well)
            well_curves = curves[well_mask]
            
            if len(well_curves) == 0: continue
                
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c_mean = np.nanmean(well_curves, axis=0)
                c_std = np.nanstd(well_curves, axis=0)
            
            upper, lower = c_mean + (num_std * c_std), c_mean - (num_std * c_std)
            is_outlier = np.any((well_curves < lower) | (well_curves > upper), axis=1)
            new_features.loc[well_mask, label_col] = np.where(is_outlier, -1, 1)
            
        results_dfs.append(new_features)
        
        # 2. HTML Visualization Phase
        html = init_html_report(
            title=f"Mean/Std Outliers: {clean_title}", 
            subtitle=f"Experiment: {exp_label} | Outlier Threshold: Mean ± {num_std} Std"
        )
        
        for well in unique_wells:
            well_mask = (Y_well == well)
            curr_well_curves = curves[well_mask]
            ref_well_curves = ref_curves[well_mask]
            
            if len(curr_well_curves) == 0: continue
            
            is_outlier = (new_features.loc[well_mask, label_col] == -1).fillna(False).values
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                curr_mean = np.nanmean(curr_well_curves, axis=0)
                curr_std = np.nanstd(curr_well_curves, axis=0)
                ref_mean = np.nanmean(ref_well_curves, axis=0)
                ref_std = np.nanstd(ref_well_curves, axis=0)
            
            # --- THE UPDATE: sharey='row' added here ---
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey='row')
            
            # --- ROW 1: REFERENCE ---
            draw_thresholds(axes[0, 0], ref_well_curves[~is_outlier], ref_mean, ref_std, num_std, "blue", 
                            f"Well {well} - Ref Normal (n={np.sum(~is_outlier)})")
            
            draw_thresholds(axes[0, 1], ref_well_curves[is_outlier], ref_mean, ref_std, num_std, "red", 
                            f"Well {well} - Ref Outliers (n={np.sum(is_outlier)})")
            
            # --- ROW 2: CURRENT DATA ---
            draw_thresholds(axes[1, 0], curr_well_curves[~is_outlier], curr_mean, curr_std, num_std, "green", 
                            f"Curr Normal (n={np.sum(~is_outlier)})")
            
            draw_thresholds(axes[1, 1], curr_well_curves[is_outlier], curr_mean, curr_std, num_std, "red", 
                            f"Curr Outliers (n={np.sum(is_outlier)})")
            
            plt.tight_layout()
            
            html += f"<div style='background: white; padding: 15px; border-radius: 8px; width: 45%; min-width: 450px; box-shadow: 0px 4px 10px rgba(0,0,0,0.05);'><img src='data:image/png;base64,{fig_to_base64(fig)}' style='width: 100%; height: auto;'></div>"
            
        html += "</div></body></html>"
        with open(os.path.join(meanstd_plot_path, f"{name}_{exp_label}.html"), "w") as f: 
            f.write(html)
        gc.collect()
        
    return results_dfs