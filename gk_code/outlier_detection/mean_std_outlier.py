import os
import gc
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from outlier_utils import *

def run_meanstd_pipeline(exp_label, num_std, ref_curves, meanstd_plot_path, dataset_names, dataset_curves, Y_well, features_index):
    os.makedirs(meanstd_plot_path, exist_ok=True)
    results_dfs = []
    
    for name, curves in zip(dataset_names, dataset_curves):
        clean_title = name.replace("_", " ").title()
        print(f"  -> Running Mean/Std [{exp_label}] for {clean_title}...")
        
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
        
        # HTML Gen (Simplified for brevity, uses same 2x2 grid approach as AMF)
        html = init_html_report(title=f"Mean/Std Outliers: {clean_title}", subtitle=f"Experiment: {exp_label} | Std: {num_std}")
        
        for well in unique_wells:
            well_mask = (Y_well == well)
            is_outlier = (new_features.loc[well_mask, label_col] == -1).fillna(False).values
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            
            axes[0].plot(ref_curves[well_mask][~is_outlier].T, c="blue", alpha=0.3)
            axes[0].plot(ref_curves[well_mask][is_outlier].T, c="red", alpha=0.5)
            axes[0].set_title(f"Well {well} (Reference)")
            
            axes[1].plot(curves[well_mask][~is_outlier].T, c="green", alpha=0.3)
            axes[1].plot(curves[well_mask][is_outlier].T, c="red", alpha=0.5)
            axes[1].set_title(f"Current Data")
            
            plt.tight_layout()
            html += f"<div style='background: white; padding: 10px; border-radius: 8px; width: 45%;'><img src='data:image/png;base64,{fig_to_base64(fig)}' width='100%'></div>"
            
        html += "</div></body></html>"
        with open(os.path.join(meanstd_plot_path, f"{name}_{exp_label}.html"), "w") as f: f.write(html)
        gc.collect()
        
    return results_dfs