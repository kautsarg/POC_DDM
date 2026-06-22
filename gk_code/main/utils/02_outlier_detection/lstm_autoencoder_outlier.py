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

def build_lstm_autoencoder(timesteps):
    inputs = layers.Input(shape=(timesteps, 1))
    x = layers.LSTM(32, return_sequences=True, name="lstm_ae_enc1")(inputs)
    x = layers.Dropout(0.1)(x)
    # Bottleneck — explicitly named so the encoder half can be sliced out and
    # reused as a pretrained feature extractor (see save_encoder_dir below /
    # model_utils.py's "lstm_ae_clf" model, which loads this layer's output).
    x = layers.LSTM(16, return_sequences=False, name="lstm_ae_enc2")(x)

    x = layers.RepeatVector(timesteps)(x)

    x = layers.LSTM(16, return_sequences=True, name="lstm_ae_dec1")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.LSTM(32, return_sequences=True, name="lstm_ae_dec2")(x)

    decoded = layers.TimeDistributed(layers.Dense(1))(x)
    autoencoder = models.Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder


def _save_encoder(autoencoder, scaler, save_encoder_dir, dataset_name):
    """Slice out the encoder half (Input -> ... -> lstm_ae_enc2 bottleneck) and save
    it standalone, sharing weights with the just-trained autoencoder. Also saves the
    fitted MinMaxScaler — the encoder was trained on scaled curves, so a later
    classifier (model_utils.py's "lstm_ae_clf", fed by 03_main_training.py) must
    apply this same transform to its raw curves before feeding the encoder, or the
    pretrained weights will see out-of-distribution input."""
    import joblib
    os.makedirs(save_encoder_dir, exist_ok=True)
    encoder = models.Model(
        inputs=autoencoder.input,
        outputs=autoencoder.get_layer("lstm_ae_enc2").output,
    )
    encoder_path = os.path.join(save_encoder_dir, f"lstm_ae_encoder_{dataset_name}.keras")
    scaler_path = os.path.join(save_encoder_dir, f"lstm_ae_scaler_{dataset_name}.joblib")
    encoder.save(encoder_path)
    joblib.dump(scaler, scaler_path)
    print(f"     [XAI] Saved pretrained LSTM-AE encoder -> {encoder_path}")

def run_lstm_autoencoder_pipeline(dataset_names, dataset_curves, Y_well, ref_curves, ae_plot_path, threshold_percentiles=["elbow", 90, 95], epochs=60, batch_size=128, save_plot=True, downsample_factor=1, per_well=True, save_encoder_dir=None):
    os.makedirs(ae_plot_path, exist_ok=True)
    results_dfs = []
    unique_wells = np.unique(Y_well)

    prefix = "lstm_ae"
    method_str = "pw" if per_well else "glb"
    ds_str = f"ds{downsample_factor}"
    base_label = f"{prefix}_{method_str}_{ds_str}"

    for i, (name, curves) in enumerate(zip(dataset_names, dataset_curves)):
        clean_title = name.replace("_", " ").title()
        print(f"  -> Running {method_str.upper()} LSTM Autoencoder (DS:{downsample_factor}x) for {clean_title}...")

        full_mse = np.full(len(Y_well), -1.0)
        results_filter = {f"{base_label}_mse": full_mse}
        for pct in threshold_percentiles:
            results_filter[f"{base_label}_label_{pct}"] = np.full(len(Y_well), -1)
            
        elbow_data = {}
        
        # --- GLOBAL TRAINING LOGIC ---
        if not per_well:
            invalid_mask = np.isnan(curves).any(axis=1) | np.isinf(curves).any(axis=1)
            valid_mask = ~invalid_mask
            X_valid = curves[valid_mask]
            valid_indices = np.where(valid_mask)[0]
            
            if len(X_valid) >= 5:
                X_downsampled = X_valid[:, ::downsample_factor]
                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(X_downsampled)
                
                timesteps = X_scaled.shape[1]
                X_scaled_3d = X_scaled.reshape((X_scaled.shape[0], timesteps, 1))
                
                set_global_determinism(0)

                autoencoder = build_lstm_autoencoder(timesteps)
                early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
                
                autoencoder.fit(X_scaled_3d, X_scaled_3d, epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[early_stop], verbose=0)

                X_reconstructed_3d = autoencoder.predict(X_scaled_3d, verbose=0)
                X_reconstructed = X_reconstructed_3d.reshape(X_scaled.shape)
                mse = np.mean(np.power(X_scaled - X_reconstructed, 2), axis=1)
                full_mse[valid_indices] = mse

                if save_encoder_dir is not None:
                    _save_encoder(autoencoder, scaler, save_encoder_dir, name)

                tf.keras.backend.clear_session()
                gc.collect()

        # --- EVALUATE AND THRESHOLD PER WELL ---
        for well in unique_wells:
            well_mask = (Y_well == well)
            well_indices = np.where(well_mask)[0]
            
            # --- PER WELL TRAINING LOGIC ---
            if per_well:
                well_curves = curves[well_mask]
                invalid_mask = np.isnan(well_curves).any(axis=1) | np.isinf(well_curves).any(axis=1)
                valid_mask = ~invalid_mask
                X_valid = well_curves[valid_mask]
                valid_indices = well_indices[valid_mask]
                
                if len(X_valid) >= 5:
                    X_downsampled = X_valid[:, ::downsample_factor]
                    scaler = MinMaxScaler()
                    X_scaled = scaler.fit_transform(X_downsampled)
                    
                    timesteps = X_scaled.shape[1]
                    X_scaled_3d = X_scaled.reshape((X_scaled.shape[0], timesteps, 1))
                    
                    set_global_determinism(0)
            
                    autoencoder = build_lstm_autoencoder(timesteps)
                    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
                    
                    autoencoder.fit(X_scaled_3d, X_scaled_3d, epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[early_stop], verbose=0)
                    
                    X_reconstructed_3d = autoencoder.predict(X_scaled_3d, verbose=0)
                    X_reconstructed = X_reconstructed_3d.reshape(X_scaled.shape)
                    mse = np.mean(np.power(X_scaled - X_reconstructed, 2), axis=1)
                    full_mse[valid_indices] = mse
                    
                    tf.keras.backend.clear_session()
                    gc.collect()

            # --- THRESHOLDING ---
            well_mse = full_mse[well_mask]
            valid_well_mse = well_mse[well_mse != -1.0]
            
            if len(valid_well_mse) > 0:
                sorted_mse = np.sort(valid_well_mse)
                indices_arr = np.arange(len(sorted_mse))
                
                for pct in threshold_percentiles:
                    label_col = f"{base_label}_label_{pct}"
                    
                    if str(pct).lower() == "elbow":
                        kneedle = KneeLocator(indices_arr, sorted_mse, curve="convex", direction="increasing")
                        if kneedle.knee is not None:
                            cutoff_threshold = sorted_mse[kneedle.knee]
                            knee_idx = kneedle.knee
                        else:
                            cutoff_threshold = np.percentile(valid_well_mse, 95)
                            knee_idx = None
                            
                        elbow_data[well] = {
                            "indices": indices_arr, 
                            "sorted_mse": sorted_mse, 
                            "knee_idx": knee_idx, 
                            "threshold": cutoff_threshold
                        }
                    else:
                        cutoff_threshold = np.percentile(valid_well_mse, float(pct))
                    
                    keep_mask = (well_mse <= cutoff_threshold) & (well_mse != -1.0)
                    global_keep_indices = well_indices[keep_mask]
                    results_filter[label_col][global_keep_indices] = 1

        filtered_dict = {k: v for k, v in results_filter.items() if k != f"{base_label}_mse"}
        new_features = pd.DataFrame(filtered_dict)
        results_dfs.append(new_features)
        
        # --- PLOTTING BLOCK ---
        if save_plot:
            for pct in threshold_percentiles:
                label_col = f"{base_label}_label_{pct}"
                method_name = "Per-Well" if per_well else "Global"
                
                if str(pct).lower() == "elbow":
                    sub_text = f"[{method_name} LSTM | DS: {downsample_factor}x] Threshold dynamically set using Knee/Elbow Method"
                else:
                    sub_text = f"[{method_name} LSTM | DS: {downsample_factor}x] Kept bottom {pct}% of MSE reconstruction errors"
                    
                html = init_html_report(title=f"LSTM Autoencoder Outliers: {clean_title}", subtitle=sub_text)
                
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
                            
                        ax_elbow.set_title(f"Well {well} - {method_name} LSTM Error Distribution", fontweight="bold")
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
                save_path = os.path.join(ae_plot_path, f"{name}_{base_label}_{pct}.html")
                with open(save_path, "w") as f: 
                    f.write(html)
                    
        tf.keras.backend.clear_session()
        gc.collect()
    
    return results_dfs