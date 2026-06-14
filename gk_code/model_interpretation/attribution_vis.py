import os
import math
import joblib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
import sys

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, "../outlier_detection")
import config
from model_utils import set_global_determinism

# Shared colour-blind-safe colormap for well-index class labels (0..N_WELLS-1).
WELL_CMAP = ListedColormap(config.WELL_COLORS)

# ====================================================================
# HELPER FUNCTIONS: THE ULTIMATE COMBINATION
# ====================================================================
def get_top_dims_from_weights(model, top_k=25):
    """
    Ranks the latent dimensions based on the absolute weights of the final Dense layer.
    Assumes the model ends in a single Dense classification layer.
    """
    final_dense_layer = model.layers[-1]
    W, b = final_dense_layer.get_weights()
    
    # Calculate global importance: Sum of absolute weights across all classes
    global_importance = np.sum(np.abs(W), axis=1)
    top_dims = np.argsort(global_importance)[::-1][:top_k]
    
    return top_dims, global_importance

def targeted_latent_attribution(latent_model, x_batch, target_dims):
    """
    Calculates the time-step attribution ONLY for explicitly targeted dimensions.
    """
    x_tf = tf.convert_to_tensor(x_batch, dtype=tf.float32)
    attributions = {}
    
    for dim in target_dims:
        with tf.GradientTape() as tape:
            tape.watch(x_tf)
            z_dim = latent_model(x_tf)[:, dim]
            target = tf.reduce_mean(z_dim)
            
        grads = tape.gradient(target, x_tf).numpy()        
        time_imp = np.mean(np.abs(grads), axis=(0, 2))     
        attributions[dim] = time_imp
        
    return attributions


# ====================================================================
# MODULE 1: DATA & MODEL LOADING (NO TRAINING)
# ====================================================================
def prepare_dataset(exp_path, filter_key):
    """Loads dataset, applies outlier mask, and returns train/test splits."""
    data_path = exp_path / config.TRAINING_DATA_PATH
    if not data_path.exists():
        return None

    data = joblib.load(data_path)
    encoder = LabelEncoder()
    y_full = encoder.fit_transform(data["Y_well"])

    features_df = data["kinetic_features"][0]
    mask = (features_df[filter_key] == 1).fillna(False).values
    
    X = data["dataset"][0][mask].astype(np.float32)[..., None]
    y = y_full[mask]
    timestamps = data["timestamps"]
    dataset_name = data["dataset_name"][0]

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_idx, test_idx = next(splitter.split(X, y))
    
    return {
        "X_train": X[train_idx], "y_train": y[train_idx],
        "X_test": X[test_idx], "y_test": y[test_idx],
        "X_full": X, "y_full": y,
        "timestamps": timestamps, "dataset_name": dataset_name
    }

def load_saved_models(model_dir, filter_key):
    """Loads natively saved Keras models from the dataset's interpretation directory."""
    models = {}
    model_names = ['cnn', 'bigru', 'transformer']
    
    for name in model_names:
        model_path = model_dir / f"{name}_{filter_key}_model.keras"
        if model_path.exists():
            models[name] = tf.keras.models.load_model(model_path)
        else:
            pass # Suppressed warning for cleaner console if you only trained CNNs
            
    return models


# ====================================================================
# MODULE 2: PER-DATASET VISUALIZATIONS
# ====================================================================
def plot_latent_pca(models, X, y, dataset_name, save_path):
    if not models: return
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1: axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        latent_model = tf.keras.Model(model.input, model.layers[-2].output)
        z = latent_model.predict(X, verbose=0)
        z_2d = PCA(n_components=2, random_state=0).fit_transform(z) if z.shape[1] > 2 else z

        ax.scatter(z_2d[:, 0], z_2d[:, 1], c=y, cmap=WELL_CMAP, vmin=0, vmax=config.N_WELLS - 1, alpha=0.7, s=15, edgecolors='none')
        ax.set_title(f"{name.upper()} Latent Embeddings", fontsize=12, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"Latent Space Clustering: {dataset_name}", fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def plot_cnn_kernels(cnn_model, X, timestamps, dataset_name, save_path):
    conv_layers = [l for l in cnn_model.layers if isinstance(l, tf.keras.layers.Conv1D) or 'conv' in l.name.lower()]
    if not conv_layers: return

    layer = conv_layers[0]
    kernels = layer.get_weights()[0]
    k_size, in_ch, n_filters = kernels.shape
    kern_mean = kernels.mean(axis=1)

    orig_curve = np.squeeze(np.mean(X, axis=0))
    if orig_curve.ndim > 1: orig_curve = orig_curve.mean(axis=-1)
    t = timestamps if len(timestamps) == orig_curve.shape[0] else np.arange(orig_curve.shape[0])

    cols = 4
    rows = math.ceil(n_filters / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3))
    axes = axes.flatten()

    for i in range(n_filters):
        ax1 = axes[i]
        kern = kern_mean[:, i]
        filtered = np.convolve(orig_curve, kern, mode='valid')
        len_f = filtered.shape[0]
        k_center = k_size // 2
        
        if len(t) == orig_curve.shape[0]:
            start = max(0, k_center)
            end = start + len_f
            if end > len(t): start = max(0, len(t) - len_f); end = start + len_f
            t_f = t[start:end]
        else:
            t_f = np.arange(len_f)

        ax1.plot(t, orig_curve, color='k', lw=1.2)
        ax1.set_ylabel('Original', color='k', fontsize=9)
        ax1.tick_params(axis='y', labelcolor='k', labelsize=8)
        
        ax2 = ax1.twinx()
        ax2.plot(t_f, filtered, color='C1', lw=1.2)
        ax2.set_ylabel('Filtered', color='C1', fontsize=9)
        ax2.tick_params(axis='y', labelcolor='C1', labelsize=8)

        ax1.set_title(f'Filter {i}', fontsize=11, fontweight='bold')
        axin = ax1.inset_axes([0.65, 0.65, 0.30, 0.25])
        axin.plot(np.arange(k_size), kern, color='C2', lw=1.5)
        axin.set_title('Kernel', fontsize=7, pad=2)
        axin.set_xticks([]); axin.set_yticks([])

    for j in range(n_filters, len(axes)): axes[j].axis('off')

    fig.suptitle(f"Layer '{layer.name}' Kernels | {dataset_name}", fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def plot_ultimate_attributions(models, X, y, timestamps, dataset_name, save_path, top_k=25):
    if not models: return
    
    # --- 1. Batch Selection ---
    batch_size = min(512, len(X))
    rng = np.random.default_rng(42)
    rep_indices = []
    
    if y is not None and len(np.unique(y)) > 1:
        classes = np.unique(y)
        per_class = max(1, batch_size // len(classes))
        for c in classes:
            idxs = np.where(y == c)[0]
            if len(idxs) > 0:
                rep_indices.extend(rng.choice(idxs, size=min(per_class, len(idxs)), replace=False).tolist())
        if len(rep_indices) < batch_size:
            rem = np.setdiff1d(np.arange(len(X)), np.array(rep_indices, dtype=int))
            rep_indices.extend(rng.choice(rem, size=min(batch_size - len(rep_indices), len(rem)), replace=False).tolist())
    else:
        rep_indices = np.arange(batch_size).tolist()

    X_batch = X[np.array(rep_indices[:batch_size], dtype=int)]
    t = timestamps if len(timestamps) == X_batch.shape[1] else np.arange(X_batch.shape[1])

    mean_curve = np.squeeze(np.mean(X_batch, axis=0))
    std_curve = np.squeeze(np.std(X_batch, axis=0))
    if mean_curve.ndim > 1: 
        mean_curve = mean_curve.mean(axis=-1)
        std_curve = std_curve.mean(axis=-1)

    # --- 2. Dynamic Grid Calculation ---
    cols = min(5, top_k)  # Cap columns at 5
    rows_per_model = math.ceil(top_k / cols)
    total_rows = len(models) * rows_per_model

    fig, axes = plt.subplots(total_rows, cols, figsize=(4.5 * cols, 3.5 * total_rows), sharex=True)
    
    # Safely cast axes to a 2D array depending on dimensions
    if total_rows == 1 and cols == 1: 
        axes = np.array([[axes]])
    elif total_rows == 1: 
        axes = axes[np.newaxis, :]
    elif cols == 1: 
        axes = axes[:, np.newaxis]

    # --- 3. Plotting Loop ---
    for m_idx, (model_name, model) in enumerate(models.items()):
        top_dims, importances = get_top_dims_from_weights(model, top_k=top_k)
        latent_model = tf.keras.Model(model.input, model.layers[-2].output)
        attributions = targeted_latent_attribution(latent_model, X_batch, top_dims)
        
        for k_idx, dim in enumerate(top_dims):
            # Map the linear dimension index (k_idx) to a 2D grid position
            r_idx = (m_idx * rows_per_model) + (k_idx // cols)
            c_idx = k_idx % cols
            
            ax1 = axes[r_idx, c_idx]
            ax1.plot(t, mean_curve, color='k', lw=1.3, alpha=0.7)
            ax1.fill_between(t, mean_curve - std_curve, mean_curve + std_curve, color='k', alpha=0.10)
            ax1.tick_params(axis='y', labelcolor='k', labelsize=8)
            
            # Label left-most plots
            if c_idx == 0: 
                ax1.set_ylabel(f"{model_name.upper()}\nMean Curve", color='k', fontweight='bold')

            ax2 = ax1.twinx()
            ax2.plot(t, attributions[dim], color='C1', lw=1.5)
            ax2.tick_params(axis='y', labelcolor='C1', labelsize=8)
            
            # Label right-most plots (or the last plot in a partially filled row)
            if c_idx == cols - 1 or k_idx == len(top_dims) - 1: 
                ax2.set_ylabel('Attribution', color='C1', fontweight='bold')
                
            ax1.set_title(f'Top Dim #{dim}\nWeight Importance: {importances[dim]:.2f}', fontsize=11)

        # Hide empty subplots if top_k is not a perfect multiple of `cols`
        for k_idx in range(len(top_dims), rows_per_model * cols):
            r_idx = (m_idx * rows_per_model) + (k_idx // cols)
            c_idx = k_idx % cols
            axes[r_idx, c_idx].axis('off')

    fig.suptitle(f"Ultimate Latent Attribution (Dense Rank) | Dataset: {dataset_name}", fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_latent_saliency_heatmap(models, X, timestamps, dataset_name, save_path):
    """
    Plots a stacked figure for each model: 
    Top: The mean input curve across the batch.
    Bottom: A heatmap where the Y-axis represents ALL latent dimensions 
    (sorted by Dense layer weight importance) and the X-axis represents input timestamps.
    """
    if not models: return
    
    # 1. Select a consistent batch
    batch_size = min(512, len(X))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=batch_size, replace=False)
    X_batch = X[idx]
    
    t = timestamps if len(timestamps) == X_batch.shape[1] else np.arange(X_batch.shape[1])
    
    # Calculate the mean and standard deviation of the input signal to plot the curve
    mean_curve = np.squeeze(np.mean(X_batch, axis=0))
    std_curve = np.squeeze(np.std(X_batch, axis=0))
    if mean_curve.ndim > 1: 
        mean_curve = mean_curve.mean(axis=-1)
        std_curve = std_curve.mean(axis=-1)
    
    # Create subplots: 2 rows per model (1 for curve, 1 for heatmap)
    fig, axes = plt.subplots(
        nrows=len(models) * 2, 
        ncols=1, 
        figsize=(10, 6 * len(models)), 
        gridspec_kw={'height_ratios': [1, 3] * len(models)}
    )
    
    # Ensure axes is a 1D array even if there's only 1 model
    if len(models) * 2 == 2:
        axes = np.array(axes)

    for m_idx, (model_name, model) in enumerate(models.items()):
        # Assign the specific axes for this model
        ax_curve = axes[m_idx * 2]
        ax_heat = axes[m_idx * 2 + 1]
        
        # Extract the latent model to find the total number of dimensions
        latent_model = tf.keras.Model(model.input, model.layers[-2].output)
        # num_latent_dims = latent_model.output_shape[-1]
    
        num_latent_dims = 100
        
        # 2. Get ALL dimensions ranked by dense weights
        ranked_dims, importances = get_top_dims_from_weights(model, top_k=num_latent_dims)
        
        # 3. Calculate attribution for all ranked dimensions
        attributions = targeted_latent_attribution(latent_model, X_batch, ranked_dims)
        
        # 4. Build the Heatmap Matrix
        heatmap_matrix = np.zeros((num_latent_dims, len(t)))
        
        for rank_idx, dim in enumerate(ranked_dims):
            heatmap_matrix[rank_idx, :] = attributions[dim]
        
        # # Normalise globally across the matrix for better colour scaling
        # max_val = np.max(heatmap_matrix)
        # if max_val > 0:
        #     heatmap_matrix = heatmap_matrix / max_val

        # 5a. Plot the Input Curve (Top Plot)
        ax_curve.plot(t, mean_curve, color='black', lw=1.5)
        ax_curve.fill_between(t, mean_curve - std_curve, mean_curve + std_curve, color='gray', alpha=0.3)
        ax_curve.set_title(f"{model_name.upper()} - Saliency Map by Ranked Latent Dimension", fontsize=12, fontweight='bold')
        ax_curve.set_ylabel("Input Signal", fontsize=10, fontweight='bold')
        
        # Align x-axis limits strictly to the data bounds
        ax_curve.set_xlim(t[0], t[-1])
        ax_curve.margins(x=0) # Double ensure no padding is added
        plt.setp(ax_curve.get_xticklabels(), visible=False)
        ax_curve.grid(True, linestyle='--', alpha=0.5)

        # 5b. Plot the heatmap (Bottom Plot)
        im = ax_heat.imshow(
            heatmap_matrix, 
            aspect='auto', 
            cmap='inferno', 
            extent=[t[0], t[-1], num_latent_dims, 0], # 0 (Rank 1) at the top
            interpolation='nearest'
        )
        
        ax_heat.set_xlabel("Time", fontsize=10, fontweight='bold')
        ax_heat.set_ylabel("Latent Rank\n(Top = Most Important)", fontsize=10, fontweight='bold')
        
        # --- THE ALIGNMENT FIX ---
        # 1. Allocate a tiny slice of space on the right of the heatmap for the real colorbar
        divider_heat = make_axes_locatable(ax_heat)
        cax_heat = divider_heat.append_axes("right", size="3%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax_heat)
        cbar.set_label("Saliency Attribution", rotation=270, labelpad=15, fontweight='bold')
        
        # 2. Allocate the exact same tiny slice of space on the right of the curve... and make it invisible!
        divider_curve = make_axes_locatable(ax_curve)
        cax_curve = divider_curve.append_axes("right", size="3%", pad=0.1)
        cax_curve.axis('off')
        # -------------------------

    fig.suptitle(f"Global Latent Saliency Heatmap | {dataset_name}", fontsize=16, fontweight='bold', y=1.02)
    
    # Use hspace to control the gap between the curve and the heatmap
    plt.subplots_adjust(hspace=0.15)
    
    # Removed fig.tight_layout() as it sometimes fights with make_axes_locatable
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

# ====================================================================
# MODULE 3: GLOBAL VISUALIZATION
# ====================================================================
def plot_global_cnn_kernels(all_kernels, dataset_names, global_vis_dir):
    """Plots a massive grid of all layer 1 CNN kernels across all datasets."""
    if not all_kernels:
        print("[!] No kernels found to plot globally.")
        return

    n_rows = len(all_kernels)
    n_cols = all_kernels[0].shape[1]  
    k_size = all_kernels[0].shape[0]  

    print(f"\n[*] Building Global {n_rows}x{n_cols} CNN Kernel Grid Figure...")
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5), sharex=True)
    if n_rows == 1: axes = np.expand_dims(axes, axis=0)

    for row_idx in range(n_rows):
        kern_mean = all_kernels[row_idx]
        y_min, y_max = kern_mean.min(), kern_mean.max()
        padding = (y_max - y_min) * 0.1
        
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            ax.plot(np.arange(k_size), kern_mean[:, col_idx], color='C2', lw=2)
            
            ax.set_ylim(y_min - padding, y_max + padding)
            ax.set_xticks([])
            
            if col_idx != 0:
                ax.set_yticks([])
            else:
                ax.tick_params(axis='y', labelsize=8)
                clean_name = dataset_names[row_idx].replace("_", " ").title()
                ax.set_ylabel(clean_name, rotation=90, size=10, fontweight='bold', labelpad=10)
                
            if row_idx == 0:
                ax.set_title(f'F{col_idx}', fontsize=12, fontweight='bold')

    fig.suptitle("Layer 1 CNN Kernels Across All Datasets", fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = global_vis_dir / "global_cnn_layer1_kernels.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[✓] Saved global kernel grid to: {save_path}")


# ====================================================================
# MODULE 4: PIPELINE ORCHESTRATOR
# ====================================================================
def run_interpretation_pipeline(exp_folder_path=config.DEFAULT_EXP_FOLDER, filter_key='amf_label_amf_important'):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    set_global_determinism(0)

    exp_folder = Path(exp_folder_path)
    
    # Centralized directory for all output visualisations
    global_vis_dir = config.get_viz_dir(exp_folder, "model_interpretation")
    global_vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Grab dataset folders (excluding the new centralized vis folder)
    exp_paths = sorted([p for p in exp_folder.iterdir() if p.is_dir() and p.name not in ['.DS_Store', 'model_interpretation']])
    
    # Trackers for the global CNN grid plot
    global_kernels = []
    global_dataset_names = []
    
    for exp_path in exp_paths:
        print(f"\n{'='*70}\n[*] PROCESSING DATASET: {exp_path.name}\n{'='*70}")
        
        # Directory where the pre-trained models are physically stored
        model_dir = exp_path / "model_interpretation"
        
        # Step 1: Load Data
        data_dict = prepare_dataset(exp_path, filter_key)
        if not data_dict:
            print(f"  [-] Skipping {exp_path.name}: Data not found.")
            continue
        print(f"  -> Data Loaded | X: {data_dict['X_full'].shape}, y: {data_dict['y_full'].shape}")

        # Step 2: Load Pre-Trained Models
        print("  -> Loading Pre-Trained Models...")
        models = load_saved_models(model_dir, filter_key)
        
        if not models:
            print(f"  [-] Skipping visualisations for {exp_path.name}: No saved models found.")
            continue

        # Step 3: Per-Dataset Centralized Visualizations
        print(f"  -> Generating Visualizations into {global_vis_dir.name}/ ...")
        
        plot_latent_pca(
            models=models, 
            X=data_dict["X_full"], y=data_dict["y_full"], 
            dataset_name=data_dict["dataset_name"], 
            save_path=global_vis_dir / f"01_latent_space_pca_{exp_path.name}.png"
        )
        
        if 'cnn' in models:
            plot_cnn_kernels(
                cnn_model=models['cnn'], 
                X=data_dict["X_full"], timestamps=data_dict["timestamps"], 
                dataset_name=data_dict["dataset_name"], 
                save_path=global_vis_dir / f"02_cnn_kernels_{exp_path.name}.png"
            )
            
            # Extract CNN kernel parameters for the Global Plot later
            conv_layers = [l for l in models['cnn'].layers if isinstance(l, tf.keras.layers.Conv1D) or 'conv' in l.name.lower()]
            if conv_layers:
                kern_mean = conv_layers[0].get_weights()[0].mean(axis=1)
                global_kernels.append(kern_mean)
                global_dataset_names.append(data_dict["dataset_name"])
            
        plot_ultimate_attributions(
            models=models, 
            X=data_dict["X_full"], y=data_dict["y_full"], timestamps=data_dict["timestamps"], 
            dataset_name=data_dict["dataset_name"], 
            save_path=global_vis_dir / f"03_ultimate_time_attribution_{exp_path.name}.png",
            top_k=25
        )
        
        # --- NEW HEATMAP VISUALISATION IMPLEMENTED HERE ---
        plot_latent_saliency_heatmap(
            models=models, 
            X=data_dict["X_full"], timestamps=data_dict["timestamps"], 
            # dataset_name=data_dict["dataset_name"], 
            dataset_name=exp_path.name,
            save_path=global_vis_dir / f"04_saliency_heatmap_{exp_path.name}.png"
        )

        print(f"  [✓] Processed {exp_path.name}")
        tf.keras.backend.clear_session()

    # Step 4: Generate the Aggregate Global Plot
    plot_global_cnn_kernels(global_kernels, global_dataset_names, global_vis_dir)

if __name__ == "__main__":
    run_interpretation_pipeline(exp_folder_path=config.LAB_EXP_FOLDER, filter_key=None)