import os
import logging
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from kneed import KneeLocator
from outlier_utils import init_html_report, fig_to_base64
from model_utils import set_global_determinism

tf.get_logger().setLevel(logging.ERROR)

def build_cnn_autoencoder(timesteps):
    """Builds a 1D Convolutional Autoencoder for fast, shape-aware anomaly detection."""
    inputs = layers.Input(shape=(timesteps, 1))
    
    # Encoder (Compresses the curve while learning local shapes like peaks and slopes)
    x = layers.Conv1D(filters=32, kernel_size=7, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
    x = layers.Conv1D(filters=16, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
    
    # Decoder (Reconstructs the curve from the compressed bottleneck)
    x = layers.Conv1D(filters=16, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(filters=32, kernel_size=7, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(size=2)(x)
    
    # Output Layer (Reconstructs back to a single fluorescence value per timestep)
    decoded = layers.Conv1D(filters=1, kernel_size=3, activation='linear', padding='same')(x)
    
    # Safety slice: In rare cases where timesteps aren't perfectly divisible by 4, 
    # this forces the output to strictly match the input length.
    decoded = decoded[:, :timesteps, :]
    
    autoencoder = models.Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder


def run_cnn_autoencoder_per_well_pipeline(dataset_names, dataset_curves, Y_well, ref_curves, ae_plot_path, threshold_percentiles=["elbow", 90, 95], epochs=60, batch_size=128, save_plot=True):
    os.makedirs(ae_plot_path, exist_ok=True)
    results_dfs = []
    
    unique_wells = np.unique(Y_well)

    for i, (name, curves) in enumerate(zip(dataset_names, dataset_curves)):
        clean_title = name.replace("_", " ").title()
        print(f"  -> Running Per-Well CNN Autoencoder Filtering for {clean_title}...")

        full_mse = np.full(len(Y_well), -1.0)
        
        # Initialize dictionary with CNN specific labels
        results_filter = {"cnn_ae_mse": full_mse}
        for pct in threshold_percentiles:
            results_filter[f"cnn_ae_label_{pct}"] = np.full(len(Y_well), -1)
            
        elbow_data = {}
        
        # --- TRAIN AND EVALUATE PER WELL ---
        for well in unique_wells:
            well_mask = (Y_well == well)
            well_indices = np.where(well_mask)[0]
            well_curves = curves[well_mask]
            
            invalid_mask = np.isnan(well_curves).any(axis=1) | np.isinf(well_curves).any(axis=1)
            valid_mask = ~invalid_mask
            X_valid = well_curves[valid_mask]
            valid_indices = well_indices[valid_mask]
            
            if len(X_valid) < 5:
                print(f"     [!] Skipping Well {well}: Not enough valid curves ({len(X_valid)}).")
                continue
                
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X_valid)
            
            # Reshape for CNN: (samples, timesteps) -> (samples, timesteps, 1 channel)
            timesteps = X_scaled.shape[1]
            X_scaled_3d = X_scaled.reshape((X_scaled.shape[0], timesteps, 1))
            
            set_global_determinism(0)

            autoencoder = build_cnn_autoencoder(timesteps)
            
            # Early stopping prevents over-training on the small per-well datasets
            early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
            
            autoencoder.fit(
                X_scaled_3d, X_scaled_3d, 
                epochs=epochs, 
                batch_size=batch_size, 
                shuffle=True, 
                callbacks=[early_stop],
                verbose=0
            )
            
            X_reconstructed_3d = autoencoder.predict(X_scaled_3d, verbose=0)
            X_reconstructed = X_reconstructed_3d.reshape(X_scaled.shape)
            mse = np.mean(np.power(X_scaled - X_reconstructed, 2), axis=1)
            
            full_mse[valid_indices] = mse
            
            sorted_mse = np.sort(mse)
            indices_arr = np.arange(len(sorted_mse))
            
            for pct in threshold_percentiles:
                label_col = f"cnn_ae_label_{pct}"
                
                if str(pct).lower() == "elbow":
                    kneedle = KneeLocator(indices_arr, sorted_mse, curve="convex", direction="increasing")
                    if kneedle.knee is not None:
                        cutoff_threshold = sorted_mse[kneedle.knee]
                        knee_idx = kneedle.knee
                    else:
                        cutoff_threshold = np.percentile(mse, 95)
                        knee_idx = None
                        
                    elbow_data[well] = {
                        "indices": indices_arr, 
                        "sorted_mse": sorted_mse, 
                        "knee_idx": knee_idx, 
                        "threshold": cutoff_threshold
                    }
                else:
                    cutoff_threshold = np.percentile(mse, float(pct))
                
                keep_mask = mse <= cutoff_threshold
                global_keep_indices = valid_indices[keep_mask]
                results_filter[label_col][global_keep_indices] = 1
                
            tf.keras.backend.clear_session()
            gc.collect()

        # Remove the raw MSE from the final feature output
        filtered_dict = {k: v for k, v in results_filter.items() if k != "cnn_ae_mse"}
        new_features = pd.DataFrame(filtered_dict)
        results_dfs.append(new_features)
        
        # --- PLOTTING BLOCK ---
        if save_plot:
            for pct in threshold_percentiles:
                label_col = f"cnn_ae_label_{pct}"
                if str(pct).lower() == "elbow":
                    sub_text = "Per-Well CNN | Threshold dynamically set using the Knee/Elbow Method"
                else:
                    sub_text = f"Per-Well CNN | Kept bottom {pct}% of MSE reconstruction errors per Well"
                    
                html = init_html_report(title=f"Per-Well CNN Outliers: {clean_title}", subtitle=sub_text)
                
                for well in unique_wells:
                    well_mask = (Y_well == well)
                    is_outlier = (new_features.loc[well_mask, label_col] == -1).fillna(False).values
                    
                    fig_curves, axes = plt.subplots(2, 2, figsize=(10, 8))
                    
                    if len(ref_curves[well_mask][~is_outlier]) > 0:
                        axes[0,0].plot(ref_curves[well_mask][~is_outlier].T, c="blue", alpha=0.3, rasterized=True)
                    axes[0,0].set_title(f"Well {well} - Ref Normal")
                    
                    if len(ref_curves[well_mask][is_outlier]) > 0:
                        axes[0,1].plot(ref_curves[well_mask][is_outlier].T, c="red", alpha=0.5, rasterized=True)
                    axes[0,1].set_title(f"Well {well} - Ref Outliers")
                    
                    if len(curves[well_mask][~is_outlier]) > 0:
                        axes[1,0].plot(curves[well_mask][~is_outlier].T, c="green", alpha=0.3, rasterized=True)
                    axes[1,0].set_title(f"Curr Normal")
                    
                    if len(curves[well_mask][is_outlier]) > 0:
                        axes[1,1].plot(curves[well_mask][is_outlier].T, c="red", alpha=0.5, rasterized=True)
                    axes[1,1].set_title(f"Curr Outliers")
                    
                    plt.tight_layout()
                    curves_b64 = fig_to_base64(fig_curves)
                    
                    elbow_html = ""
                    if str(pct).lower() == "elbow" and well in elbow_data:
                        fig_elbow, ax_elbow = plt.subplots(figsize=(8, 4))
                        ed = elbow_data[well]
                        
                        ax_elbow.plot(ed["indices"], ed["sorted_mse"], label="Sorted MSE", color="#2980b9", linewidth=2)
                        if ed["knee_idx"] is not None:
                            ax_elbow.axvline(ed["knee_idx"], color="#e74c3c", linestyle="--", linewidth=2, 
                                             label=f"Elbow Cutoff (MSE={ed['threshold']:.4f})")
                            ax_elbow.fill_between(ed["indices"][ed["knee_idx"]:], ed["sorted_mse"][ed["knee_idx"]:], 
                                                  color="#e74c3c", alpha=0.2, label="Rejected Curves")
                            
                        ax_elbow.set_title(f"Well {well} - Specialized CNN Error Distribution", fontweight="bold")
                        ax_elbow.set_xlabel("Curves (Sorted by Lowest to Highest Error)")
                        ax_elbow.set_ylabel("Reconstruction Error (MSE)")
                        ax_elbow.grid(alpha=0.3)
                        ax_elbow.legend()
                        
                        plt.tight_layout()
                        elbow_b64 = fig_to_base64(fig_elbow)
                        elbow_html = f"<div style='margin-top: 15px;'><img src='data:image/png;base64,{elbow_b64}' width='100%'></div>"
                    
                    html += f"""
                    <div style='background: white; padding: 15px; border-radius: 8px; width: 45%; min-width: 500px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                        <img src='data:image/png;base64,{curves_b64}' width='100%'>
                        {elbow_html}
                    </div>
                    """
                    
                html += "</div></body></html>"
                save_path = os.path.join(ae_plot_path, f"{name}_per_well_cnn_ae_{pct}.html")
                with open(save_path, "w") as f: 
                    f.write(html)
                    
        tf.keras.backend.clear_session()
        gc.collect()
    
    return results_dfs