import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from outlier_utils import *

def run_amf_pipeline(exp_label, amf_features_list, ref_curves, amf_plot_path, dataset_names, kinetic_features, dataset_curves, Y_well):
    os.makedirs(amf_plot_path, exist_ok=True)
    results_dfs = []
    
    for i, (name, feats_df, curves) in enumerate(zip(dataset_names, kinetic_features, dataset_curves)):
        amf_feats = amf_features_list[i]
        clean_title = name.replace("_", " ").title()
        print(f"  -> Running AMF [{exp_label}] for {clean_title}...")
        
        # 1. Detection
        new_features = pd.DataFrame(index=feats_df.index)
        label_col = f"amf_label_{exp_label}"
        new_features[label_col] = np.nan
        
        unique_wells = np.unique(Y_well)
        for well in unique_wells:
            well_mask = (Y_well == well)
            valid_mask = well_mask & ~feats_df[amf_feats].isna().any(axis=1)
            features_clean = feats_df.loc[valid_mask, amf_feats].values
            
            if len(features_clean) > 0:
                clf = IsolationForest(random_state=0).fit(features_clean)
                new_features.loc[valid_mask, label_col] = clf.predict(features_clean)
                
        results_dfs.append(new_features)
        
        # 2. HTML Visualization
        html = init_html_report(
            title=f"AMF Outliers (Isolation Forest): {clean_title}", 
            subtitle=f"Experiment: {exp_label} | Features: {amf_feats}"
        )
        
        for well in unique_wells:
            well_mask = (Y_well == well)
            is_outlier = (new_features.loc[well_mask, label_col] == -1).fillna(False).values
            
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            
            # Ref Normal / Outlier
            axes[0,0].plot(ref_curves[well_mask][~is_outlier].T, c="blue", alpha=0.3, rasterized=True)
            axes[0,0].set_title(f"Well {well} - Ref Normal")
            axes[0,1].plot(ref_curves[well_mask][is_outlier].T, c="red", alpha=0.5, rasterized=True)
            axes[0,1].set_title(f"Well {well} - Ref Outliers")
            
            # Curr Normal / Outlier
            axes[1,0].plot(curves[well_mask][~is_outlier].T, c="green", alpha=0.3, rasterized=True)
            axes[1,0].set_title(f"Curr Normal")
            axes[1,1].plot(curves[well_mask][is_outlier].T, c="red", alpha=0.5, rasterized=True)
            axes[1,1].set_title(f"Curr Outliers")
            
            plt.tight_layout()
            html += f"<div style='background: white; padding: 10px; border-radius: 8px; width: 30%; min-width: 400px;'><img src='data:image/png;base64,{fig_to_base64(fig)}' width='100%'></div>"
            
        html += "</div></body></html>"
        save_path = os.path.join(amf_plot_path, f"{name}_{exp_label}.html")
        with open(save_path, "w") as f: f.write(html)
        gc.collect()
        
    return results_dfs