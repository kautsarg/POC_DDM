import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from outlier_utils import *

def detect_msc(features_df, Y_well, msc_features, p_value, exp_label):
    msc_threshold = calculate_chi2_threshold(p_value=p_value, df=2) 
    unique_wells = np.unique(Y_well)
    
    # Initialize empty dataframe for ONLY the new features
    new_features = pd.DataFrame(index=features_df.index)
    dist_col = f"msc_mahal_dist_{exp_label}"
    label_col = f"msc_label_{exp_label}"
    
    new_features[dist_col] = np.nan
    new_features[label_col] = np.nan
    line_fittings = {}

    for well in unique_wells:
        well_mask = (Y_well == well)
        valid_mask = well_mask & ~features_df[msc_features].isna().any(axis=1)
        features_clean = features_df.loc[valid_mask, msc_features].values
        
        if len(features_clean) > 0:
            well_line_fit = unsupervised_line_fitting(features_clean)
            if well_line_fit:
                q1 = well_line_fit['mean']
                q2 = well_line_fit['mean'] + well_line_fit['direction']
                cov_matrix = np.cov(features_clean, rowvar=False)
                
                distances = calculate_msc_mahalanobis(features_clean, q1, q2, cov_matrix)
                new_features.loc[valid_mask, dist_col] = distances
                new_features.loc[valid_mask, label_col] = np.where(distances > msc_threshold, -1, 1)
                line_fittings[well] = well_line_fit
                
    return new_features, line_fittings

def run_msc_pipeline(exp_label, p_value, ref_curves, msc_plot_path, dataset_names, kinetic_features, dataset_curves, Y_well, msc_features_list):
    os.makedirs(msc_plot_path, exist_ok=True)
    results_dfs = []
    
    for i, (name, feats_df, curves) in enumerate(zip(dataset_names, kinetic_features, dataset_curves)):
        msc_feats = msc_features_list[i]
        clean_title = name.replace("_", " ").title()
        print(f"  -> Running MSC [{exp_label}] for {clean_title}...")
        
        # 1. Detection
        new_feats_df, line_fits = detect_msc(feats_df, Y_well, msc_feats, p_value, exp_label)
        results_dfs.append(new_feats_df)
        
        # 2. HTML Visualization
        html = init_html_report(
            title=f"MSC Outliers: {clean_title}", 
            subtitle=f"Experiment: {exp_label} | P-Value: {p_value} | Features: {msc_feats}"
        )
        
        unique_wells = np.unique(Y_well)
        for well in unique_wells:
            well_mask = (Y_well == well)
            is_outlier = (new_feats_df.loc[well_mask, f"msc_label_{exp_label}"] == -1).fillna(False).values
            
            # Subplots for this specific well
            fig = plt.figure(figsize=(15, 4))
            
            # Reference Curves
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.plot(ref_curves[well_mask][~is_outlier].T, c="blue", alpha=0.2, rasterized=True)
            ax1.plot(ref_curves[well_mask][is_outlier].T, c="red", alpha=0.5, rasterized=True)
            ax1.set_title(f"Well {well} (Reference)")
            
            # Current Curves
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.plot(curves[well_mask][~is_outlier].T, c="blue", alpha=0.2, rasterized=True)
            ax2.plot(curves[well_mask][is_outlier].T, c="red", alpha=0.5, rasterized=True)
            ax2.set_title(f"Well {well} (Current)")
            
            # 3D Feature Space
            ax3 = fig.add_subplot(1, 3, 3, projection='3d')
            X = feats_df.loc[well_mask, msc_feats[0]].values
            Y = feats_df.loc[well_mask, msc_feats[1]].values
            Z = feats_df.loc[well_mask, msc_feats[2]].values
            
            # Draw the fitted line with Z-order
            well_line_fit = line_fits.get(well)
            if well_line_fit is not None and len(well_line_fit.get('projections', [])) > 0:
                projections = well_line_fit['projections']
                p0 = well_line_fit['mean']
                v = well_line_fit['direction']
                
                t_min, t_max = np.min(projections), np.max(projections)
                margin = (t_max - t_min) * 0.1
                line_start = p0 + (t_min - margin) * v
                line_end = p0 + (t_max + margin) * v
                
                # Draw the fitted line (zorder=10 forces it to the front)
                ax3.plot([line_start[0], line_end[0]], 
                         [line_start[1], line_end[1]], 
                         [line_start[2], line_end[2]], 
                         color='black', linewidth=3.0, zorder=10, label="Fitted Line")
            
            # Draw the scatter points (zorder=1 & 2 keeps them behind the line)
            ax3.scatter(X[~is_outlier], Y[~is_outlier], Z[~is_outlier], c="blue", s=20, alpha=0.5, zorder=1, label="Normal")
            ax3.scatter(X[is_outlier], Y[is_outlier], Z[is_outlier], c="red", s=40, marker='x', zorder=2, label="Outlier")
            
            ax3.set_title("Feature Space")
            ax3.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8)
            
            plt.tight_layout()
            img_b64 = fig_to_base64(fig)
            html += f"<div style='background: white; padding: 10px; border-radius: 8px; width: 45%;'><img src='data:image/png;base64,{img_b64}' width='100%'></div>"
            
        html += "</div></body></html>"
        
        save_path = os.path.join(msc_plot_path, f"{name}_{exp_label}.html")
        with open(save_path, "w") as f: f.write(html)
        
        interactive_path = os.path.join(msc_plot_path, f"{name}_{exp_label}_INTERACTIVE.html")
        is_outlier_array = (new_feats_df[f"msc_label_{exp_label}"] == -1).fillna(False).values
        
        build_interactive_msc_html(
            save_path=interactive_path,
            title=clean_title,
            msc_feats=msc_feats,
            Y_well=Y_well,
            X_feats=feats_df[msc_feats[0]].values,
            Y_feats=feats_df[msc_feats[1]].values,
            Z_feats=feats_df[msc_feats[2]].values,
            is_outlier=is_outlier_array,
            curves=curves,
            ref_curves=ref_curves,
            line_fits=line_fits
        )
        
        gc.collect()
        
    return results_dfs