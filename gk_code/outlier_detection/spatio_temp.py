import os
import sys
import joblib
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

import config

# ====================================================================
# HELPER FUNCTIONS
# ====================================================================
def get_exp_paths(exp_folder):
    """Scans the directory and returns a sorted list of valid experiment paths."""
    return sorted([
        Path(exp_folder, name) for name in os.listdir(exp_folder) 
        if os.path.isdir(os.path.join(exp_folder, name)) and name != ".DS_Store"
    ])

def calculate_elbow_threshold(errors):
    """
    Calculates the point of maximum curvature (the elbow) on the sorted error curve
    using perpendicular geometric distance.
    """
    sorted_errors = np.sort(errors)
    n_points = len(sorted_errors)
    
    # Create coordinates for the curve points
    points = np.column_stack((np.arange(n_points), sorted_errors))
    
    # Line vector from first point to last point
    p1 = points[0]
    p2 = points[-1]
    line_vec = p2 - p1
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    
    # Vector from p1 to all points
    vec_from_first = points - p1
    
    # Calculate scalar and vector projections
    scalar_proj = np.dot(vec_from_first, line_vec_norm)
    vec_proj = np.outer(scalar_proj, line_vec_norm)
    
    # The elbow is the point furthest from the straight line
    perp_dist = np.linalg.norm(vec_from_first - vec_proj, axis=1)
    elbow_idx = np.argmax(perp_dist)
    
    return sorted_errors[elbow_idx]

def build_convlstm_ae(time_steps, rows, cols):
    """Constructs the Spatio-Temporal Autoencoder."""
    model = models.Sequential([
        layers.Input(shape=(time_steps, rows, cols, 1)),
        
        # Encoder
        layers.ConvLSTM2D(filters=16, kernel_size=(3, 3), padding='same', return_sequences=True),
        layers.BatchNormalization(),
        
        # Bottleneck
        layers.ConvLSTM2D(filters=8, kernel_size=(3, 3), padding='same', return_sequences=True),
        layers.BatchNormalization(),
        
        # Decoder
        layers.ConvLSTM2D(filters=16, kernel_size=(3, 3), padding='same', return_sequences=True),
        layers.BatchNormalization(),
        
        # Output Layer
        layers.ConvLSTM2D(filters=1, kernel_size=(3, 3), padding='same', return_sequences=True, activation='linear')
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model


# ====================================================================
# MAIN PIPELINE LOGIC
# ====================================================================
def run_convlstm_pipeline(exp_path, ds_factor=5):
    train_data_path = os.path.join(exp_path, config.TRAINING_DATA_PATH)

    if not os.path.exists(train_data_path):
        print(f"[SKIP] Unified state not found at: {train_data_path}")
        return

    print(f"\n{'='*60}\nPROCESSING: {exp_path.name}\n{'='*60}")
    print(f"[*] Loading data from: {train_data_path}")
    
    pipeline_state = joblib.load(train_data_path)

    timestamps = pipeline_state["timestamps"]
    ori_idx = list(pipeline_state["dataset_name"]).index("ori_curves")
    ori_curves = pipeline_state["dataset"][ori_idx]
    features_df = pipeline_state["kinetic_features"][ori_idx]
    Y_well = pipeline_state["Y_well"]

    # Spatial metadata
    row_idx = pipeline_state["metadata_df"]["pixel_row_idx"].values
    col_idx = pipeline_state["metadata_df"]["pixel_col_idx"].values

    max_row = 49
    max_col = 81
    
    unique_wells = np.unique(Y_well)
    num_wells = len(unique_wells)
    well_to_idx = {w: i for i, w in enumerate(unique_wells)}

    # ----------------------------------------------------------------
    # 2. SPATIAL RECONSTRUCTION & DOWNSAMPLING
    # ----------------------------------------------------------------
    print(f"[*] Reconstructing spatial grids for {num_wells} wells...")

    # Scale curves to [0, 1] for stable Autoencoder training
    min_val = np.min(ori_curves)
    max_val = np.max(ori_curves)
    scaled_curves = (ori_curves - min_val) / (max_val - min_val + 1e-9)

    # Downsample time dimension to prevent VRAM OOM error
    timestamps_ds = timestamps[::ds_factor]
    scaled_curves_ds = scaled_curves[:, ::ds_factor]
    T_ds = len(timestamps_ds)
    print(f"  -> Downsampled time steps: {len(timestamps)} -> {T_ds} (Factor: {ds_factor})")

    # Initialize blank video tensor
    X_grid = np.zeros((num_wells, T_ds, max_row, max_col, 1), dtype=np.float32)

    # Scatter the downsampled 1D curves into exact hardware coordinates
    for i in range(len(scaled_curves_ds)):
        w_idx = well_to_idx[Y_well[i]]
        r = row_idx[i]
        c = col_idx[i]
        X_grid[w_idx, :, r, c, 0] = scaled_curves_ds[i]

    print(f"  -> Final Grid shape: {X_grid.shape}")

    # ----------------------------------------------------------------
    # 3. BUILD CONVLSTM AUTOENCODER
    # ----------------------------------------------------------------
    print("[*] Building ConvLSTM Autoencoder...")
    tf.keras.backend.clear_session()
    autoencoder = build_convlstm_ae(T_ds, max_row, max_col)

    # ----------------------------------------------------------------
    # 4. TRAIN AND INFER
    # ----------------------------------------------------------------
    print("[*] Training ConvLSTM...")
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    autoencoder.fit(
        X_grid, X_grid,
        epochs=50,
        batch_size=3, # Minimum batch size for 4D spatial tensors
        # callbacks=[early_stop],
        verbose=1
    )

    print("[*] Predicting and extracting per-pixel anomalies...")
    X_pred = autoencoder.predict(X_grid, batch_size=1)

    # ----------------------------------------------------------------
    # 5. ERROR EXTRACTION & THRESHOLDING
    # ----------------------------------------------------------------
    mse_errors = np.zeros(len(ori_curves))

    for i in range(len(ori_curves)):
        w_idx = well_to_idx[Y_well[i]]
        r = row_idx[i]
        c = col_idx[i]
        
        true_curve = X_grid[w_idx, :, r, c, 0]
        pred_curve = X_pred[w_idx, :, r, c, 0]
        mse_errors[i] = np.mean((true_curve - pred_curve) ** 2)

    threshold_95 = np.percentile(mse_errors, 95)
    outlier_labels_95 = (mse_errors > threshold_95).astype(int)

    elbow_threshold = calculate_elbow_threshold(mse_errors)
    outlier_labels_elbow = (mse_errors > elbow_threshold).astype(int)

    print(f"  -> Threshold (95%): {threshold_95:.7f} | Flagged {np.sum(outlier_labels_95)} outliers")
    print(f"  -> Threshold (Elbow): {elbow_threshold:.7f} | Flagged {np.sum(outlier_labels_elbow)} outliers")

    # ----------------------------------------------------------------
    # 6. UPDATE AND SAVE JOBLIB
    # ----------------------------------------------------------------
    print("[*] Saving results back to pipeline state...")
    
    # Use dynamic names based on the downsample factor
    features_df[f"convlstm_ae_ds{ds_factor}_mse"] = mse_errors
    features_df[f"convlstm_ae_ds{ds_factor}_label_95"] = outlier_labels_95
    features_df[f"convlstm_ae_ds{ds_factor}_label_elbow"] = outlier_labels_elbow

    pipeline_state["kinetic_features"][ori_idx] = features_df
    joblib.dump(pipeline_state, train_data_path, compress=3)
    
    print(f"[✓] Successfully updated {config.TRAINING_DATA_PATH} for {exp_path.name}!\n")


# ====================================================================
# ARGPARSE EXECUTION (ARRAY JOB ENTRY POINT)
# ====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ConvLSTM Outlier Detection Pipeline")
    parser.add_argument("--task_id", type=int, default=0, help="Array Job ID")
    parser.add_argument("--exp_folder", type=str, default=config.DEFAULT_EXP_FOLDER)
    parser.add_argument("--ds_factor", type=int, default=5, help="Temporal downsample factor to prevent OOM")
    args = parser.parse_args()

    exp_paths = get_exp_paths(args.exp_folder)

    # Safe array indexing
    if args.task_id >= len(exp_paths):
        print(f"[!] Task ID {args.task_id} is out of bounds for {len(exp_paths)} folders. Exiting.")
        sys.exit(0)

    # Execute only the assigned chunk for this node
    target_exp_path = exp_paths[args.task_id]
    
    run_convlstm_pipeline(target_exp_path, ds_factor=args.ds_factor)