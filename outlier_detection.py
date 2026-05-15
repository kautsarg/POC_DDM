import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from kneed import KneeLocator
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

# Direct imports from Titan library
from titan.Experiment import Experiment
from titan.load_and_preprocessing import titan_load_and_preprocessing

# ==========================================
# 1. MODEL ARCHITECTURES
# ==========================================

def build_cnn_autoencoder(curves):
    """ 1D-CNN Autoencoder Architecture """
    inputs = layers.Input(shape=(curves, 1))
    
    # Encoder
    x = layers.Conv1D(32, 7, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Conv1D(16, 5, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2, padding='same')(x)
    
    # Decoder
    x = layers.Conv1D(16, 5, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(32, 7, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    
    decoded = layers.Conv1D(1, 3, activation='linear', padding='same')(x)
    decoded = decoded[:, :curves, :]
    
    model = models.Model(inputs, decoded)
    model.compile(optimizer='adam', loss='mse')
    return model

def build_lstm_autoencoder(curves):
    """ LSTM Autoencoder Architecture """
    inputs = layers.Input(shape=(curves, 1))
    
    # Encoder
    x = layers.LSTM(32, return_sequences=True)(inputs)
    x = layers.Dropout(0.1)(x)
    x = layers.LSTM(16, return_sequences=False)(x)
    
    # Bridge
    x = layers.RepeatVector(curves)(x)
    
    # Decoder
    x = layers.LSTM(16, return_sequences=True)(x)
    x = layers.Dropout(0.1)(x)
    x = layers.LSTM(32, return_sequences=True)(x)
    
    decoded = layers.TimeDistributed(layers.Dense(1))(x)
    
    model = models.Model(inputs, decoded)
    model.compile(optimizer='adam', loss='mse')
    return model

# ==========================================
# 2. DATA LOADING & PROCESSING
# ==========================================

def load_and_reconstruct(exp_path, n_wells=10, n_a_type="v04"):
    """
    Loads experiment and reconstructs data into flattened arrays.
    """
    print(f"\n[1/4] Loading experiment data from: {exp_path.name}")
    start_time = time.time()
    
    exp = titan_load_and_preprocessing(
        exp_path, 
        n_wells=n_wells, 
        start_type="temperature", 
        n_a_type=n_a_type,
        print_status=False
    )
    
    curves_data = None
    well_ids = []

    print(f"      -> Reconstructing curves for {len(exp.wells_list)} wells...")
    for well in exp.wells_list:
        temp_x = well.well_2d_bs_active.copy()
        temp_x = np.swapaxes(temp_x, 0, 1)

        if curves_data is None:
            curves_data = temp_x
        else:
            curves_data = np.vstack((curves_data, temp_x))
        
        well_ids.append(temp_x.shape[0])

    X_time = exp.wells_list[0].time
    Y_well = np.array([label for label, count in enumerate(well_ids) for _ in range(count)])
    
    elapsed = time.time() - start_time
    print(f"      -> Success. Total curves: {len(Y_well)}. ({elapsed:.2f}s)")
    return X_time, curves_data, Y_well

# ==========================================
# 3. OUTLIER DETECTION PIPELINE
# ==========================================

def run_outlier_detection(X_time, curves, Y_well, model_type="cnn", epochs=50, per_well=True):
    """
    Trains autoencoder(s) and returns outlier boolean mask.
    
    Parameters:
    - per_well: If True, trains a separate model for each well. 
                If False, trains one global model for the entire dataset.
    """
    print(f"\n[2/4] Initializing Outlier Detection ({model_type.upper()})")
    print(f"      -> Mode: {'Per-Well' if per_well else 'Global'}")
    
    is_outlier = np.zeros(len(Y_well), dtype=bool)
    unique_wells = np.unique(Y_well)
    num_timesteps = curves.shape[1]

    # Helper function to run detection on a specific subset of data
    def process_segment(segment_curves, segment_indices):
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(segment_curves)
        X_train = scaled.reshape((scaled.shape[0], scaled.shape[1], 1))

        # Model Selection
        if model_type.lower() == "lstm":
            model = build_lstm_autoencoder(num_timesteps)
        else:
            model = build_cnn_autoencoder(num_timesteps)
        
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.fit(X_train, X_train, epochs=epochs, batch_size=64, verbose=0, callbacks=[early_stop])
        
        # Error Calculation
        reconstructed = model.predict(X_train, verbose=0).reshape(scaled.shape)
        mse = np.mean(np.power(scaled - reconstructed, 2), axis=1)
        
        # Thresholding
        sorted_mse = np.sort(mse)
        kneedle = KneeLocator(np.arange(len(sorted_mse)), sorted_mse, curve="convex", direction="increasing")
        threshold = sorted_mse[kneedle.knee] if kneedle.knee else np.percentile(mse, 95)
        
        # Mark global mask
        is_outlier[segment_indices[mse > threshold]] = True
        return np.sum(mse > threshold)

    start_proc = time.time()

    if per_well: # Per Well Outlier Detection
        for well in unique_wells:
            mask = (Y_well == well)
            well_indices = np.where(mask)[0]
            if len(well_indices) >= 5:
                count = process_segment(curves[mask], well_indices)
                print(f"      -> Well {well}: {count} outliers found.")
    else: # Whole Chip Data Outlier Detection
        all_indices = np.arange(len(Y_well))
        count = process_segment(curves, all_indices)
        print(f"      -> Global Model: {count} outliers found across all wells.")

    print(f"      -> Process complete. ({time.time() - start_proc:.2f}s)")
    return is_outlier

def plot_outlier_results(time_axis, curves, wells, is_outlier, save_dir="plots"):
    print(f"\n[4/4] Saving plots to '{save_dir}' directory...")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"      -> Created directory: {save_dir}")

    unique_wells = np.unique(wells)
    for well in unique_wells:
        mask = (wells == well)
        plt.figure(figsize=(10, 4))
        
        inliers = curves[mask & ~is_outlier]
        if len(inliers) > 0:
            plt.plot(time_axis, inliers.T, color='blue', alpha=0.1)
            
        outliers = curves[mask & is_outlier]
        if len(outliers) > 0:
            plt.plot(time_axis, outliers.T, color='red', alpha=0.6)
            
        plt.title(f"Well {well}: Blue=Inlier ({len(inliers)}), Red=Outlier ({len(outliers)})")
        plt.xlabel("Time")
        plt.ylabel("Signal")
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f"well_{well}_outliers_well.png")
        plt.savefig(save_path)
        plt.close()
        print(f"      -> Saved: {save_path}")

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    EXP_PATH = Path("/Users/kautsarg/Documents/Final Project/Run Data/trial test data/D20250808_E00_C00_F4500KHz_U_Sample_7")
    
    if EXP_PATH.exists():
        global_start = time.time()
        
        # 1. Load Data
        time_data, curve_data, well_labels = load_and_reconstruct(EXP_PATH)
        
        # 2. Detect Outliers
        outlier_mask = run_outlier_detection(curve_data, well_labels, model_type="cnn", per_well=True)
        
        # 3. Save Results
        print(f"\nSummary: Detected {np.sum(outlier_mask)} total outliers.")
        # Pass a specific folder name to save the plots
        plot_outlier_results(time_data, curve_data, well_labels, outlier_mask, save_dir=Path(EXP_PATH, "outlier_plots"))
        
        total_elapsed = time.time() - global_start
        print(f"\nPipeline finished in {total_elapsed:.2f}s.")
    else:
        print(f"Error: Path {EXP_PATH} does not exist.")