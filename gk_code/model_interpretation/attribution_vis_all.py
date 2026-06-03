import os
import math
import joblib
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import sys

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import mutual_info_classif
from sklearn.manifold import TSNE

sys.path.insert(0, "../outlier_detection")
import config
from model_utils import set_global_determinism

# ====================================================================
# HELPER FUNCTIONS: THE ULTIMATE COMBINATION
# ====================================================================
def get_top_dims_from_weights(model, top_k=25):
    """Ranks the latent dimensions based on the absolute weights of the final Dense layer."""
    final_dense_layer = model.layers[-1]
    W, b = final_dense_layer.get_weights()
    
    global_importance = np.sum(np.abs(W), axis=1)
    actual_top_k = min(top_k, len(global_importance))
    top_dims = np.argsort(global_importance)[::-1][:actual_top_k]
    
    return top_dims, global_importance

def targeted_latent_attribution(latent_model, x_inputs, target_dims):
    """Calculates time-step attribution ONLY for the curve input, watching routing."""
    x_tf_curve = tf.convert_to_tensor(x_inputs[0], dtype=tf.float32)
    
    if len(x_inputs) > 1:
        x_tf_man = tf.convert_to_tensor(x_inputs[1], dtype=tf.float32)
        model_inputs = [x_tf_curve, x_tf_man]
    else:
        model_inputs = x_tf_curve

    attributions = {}
    for dim in target_dims:
        with tf.GradientTape() as tape:
            tape.watch(x_tf_curve)
            z_dim = latent_model(model_inputs)[:, dim]
            target = tf.reduce_mean(z_dim)
            
        grads = tape.gradient(target, x_tf_curve).numpy()        
        time_imp = np.mean(np.abs(grads), axis=(0, 2))     
        attributions[dim] = time_imp
        
    return attributions


# ====================================================================
# MODULE 1: DATA & MODEL LOADING 
# ====================================================================
def prepare_dataset(exp_path, filter_key):
    """Loads dataset, extracts cached features from the LOCAL joblib, and returns splits."""
    data_path = exp_path / config.TRAINING_DATA_PATH
    if not data_path.exists():
        return None

    data = joblib.load(data_path)
    encoder = LabelEncoder()
    y_full = encoder.fit_transform(data["Y_well"])

    features_df = data["kinetic_features"][0]
    if filter_key is None:
        mask = np.ones(len(y_full), dtype=bool)
    else:
        mask = (features_df[filter_key] == 1).fillna(False).values
    
    X = data["dataset"][0][mask].astype(np.float32)[..., None]
    y = y_full[mask]
    features_df_masked = features_df[mask]
    timestamps = data["timestamps"]
    dataset_name = data["dataset_name"][0]

    # Use 0.1 to perfectly match the training split and scaler distributions
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    train_idx, test_idx = next(splitter.split(X, y))

    # --- Extract Top 10 Manual Features for LF Models (LOCALIZED JOBLIB FIX) ---
    joblib_path = exp_path / "model_interpretation" / "model_interpretation.joblib"
    saved_top_10 = None
    
    if joblib_path.exists():
        local_pkg = joblib.load(joblib_path)
        if local_pkg and "top_10_features" in local_pkg:
            saved_top_10 = local_pkg["top_10_features"].get(str(filter_key))

    if saved_top_10:
        top_10_features = saved_top_10
    else:
        # Fallback (Calculated strictly on train_idx to prevent leakage)
        X_candidates_train = features_df_masked.iloc[train_idx][config.LD_FEATURES].values
        X_candidates_train = np.nan_to_num(X_candidates_train, nan=0.0, posinf=0.0, neginf=0.0)
        mi_scores = mutual_info_classif(X_candidates_train, y[train_idx], random_state=0)
        top_10_idx = np.argsort(mi_scores)[-10:][::-1]
        top_10_features = [config.LD_FEATURES[i] for i in top_10_idx]

    X_man = features_df_masked[top_10_features].values
    X_man = np.nan_to_num(X_man, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Standardize Manual Features based on the training fold
    scaler = StandardScaler()
    X_man_scaled = np.empty_like(X_man)
    X_man_scaled[train_idx] = scaler.fit_transform(X_man[train_idx])
    X_man_scaled[test_idx] = scaler.transform(X_man[test_idx])

    return {
        "X_full": X, "X_man_full": X_man_scaled, "y_full": y,
        "timestamps": timestamps, "dataset_name": dataset_name,
        "top_10_features": top_10_features
    }

def load_saved_models(model_dir, filter_key, expected_seq_len):
    """Loads models and strictly checks shape to prevent ValueError crashes."""
    models = {}
    model_names = [
        'cnn', 'bigru', 'transformer',
        'cnn_lf', 'bigru_lf', 'transformer_lf',
        'cnn_gru_dual', 'cnn_trans_dual', 'cnn_transformer_dual'
    ]
    
    for name in model_names:
        model_path = model_dir / f"{name}_{filter_key}_model.keras"
        if model_path.exists():
            try:
                model = tf.keras.models.load_model(model_path)
                model_seq_len = model.input_shape[0][1] if isinstance(model.input_shape, list) else model.input_shape[1]
                if model_seq_len != expected_seq_len:
                    print(f"     [!] Skipping {name}: Model expects {model_seq_len} steps, but dataset has {expected_seq_len} steps. (Stale model)")
                    continue
                models[name] = model
            except Exception as e:
                print(f"     [!] Failed to load {name}: {str(e)}")
            
    return models


# ====================================================================
# MODULE 2: PER-DATASET VISUALIZATIONS
# ====================================================================
def plot_latent_pca(models, X, X_man, y, dataset_name, save_path):
    if not models: return
    
    import math
    
    num_models = len(models)
    cols = min(3, num_models)
    rows = math.ceil(num_models / cols)

    # Adjust figure size dynamically based on the grid
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    
    # Flatten axes array for easy iteration (and handle single plot edge case)
    if num_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    scatter = None
    for ax, (name, model) in zip(axes, models.items()):
        latent_model = tf.keras.Model(model.input, model.layers[-2].output)
        model_inputs = [X, X_man] if name.endswith('_lf') else X

        z = latent_model.predict(model_inputs, verbose=0)
        z_2d = PCA(n_components=2, random_state=0).fit_transform(z) if z.shape[1] > 2 else z

        # Save the scatter plot object so we can extract its legend elements later
        scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=y, cmap='tab10', alpha=0.7, s=15, edgecolors='none')
        ax.set_title(f"{name.upper()}", fontsize=12, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

    # Hide any unused subplots (e.g., if you have 8 models, the 9th grid spot is empty)
    for i in range(num_models, len(axes)):
        axes[i].axis('off')

    # Add a single shared legend for the entire figure
    if scatter is not None:
        fig.legend(
            *scatter.legend_elements(), 
            title="Classes", 
            loc="center left", 
            bbox_to_anchor=(1.02, 0.5), # Pushes the legend just outside the right edge of the grid
            fontsize=10, 
            title_fontsize=12
        )

    fig.suptitle(f"Latent Space Clustering: {dataset_name}", fontsize=16, fontweight='bold', y=1.02)
    
    # Adjust layout to prevent title overlap, leaving room for the legend
    plt.tight_layout()
    
    # bbox_inches='tight' ensures the external legend isn't cut off when saving
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def plot_latent_tsne(models, X, X_man, y, dataset_name, save_path, max_samples=2000):
    if not models: return
    
    import math
    from sklearn.manifold import TSNE
    
    # --- SUB-SAMPLING FOR t-SNE PERFORMANCE ---
    # t-SNE is extremely slow on 50k+ samples. Subsampling to ~2000 keeps it fast and accurate.
    if len(X) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X_batch = X[idx]
        X_man_batch = X_man[idx] if X_man is not None else None
        y_batch = y[idx]
    else:
        X_batch = X
        X_man_batch = X_man
        y_batch = y

    num_models = len(models)
    cols = min(3, num_models)
    rows = math.ceil(num_models / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    
    if num_models == 1: axes = [axes]
    else: axes = axes.flatten()

    scatter = None
    for ax, (name, model) in zip(axes, models.items()):
        latent_model = tf.keras.Model(model.input, model.layers[-2].output)
        model_inputs = [X_batch, X_man_batch] if name.endswith('_lf') else [X_batch]

        z = latent_model.predict(model_inputs, verbose=0)
        
        # Calculate t-SNE (Using PCA initialization for stability and speed)
        if z.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=0, init='pca', learning_rate='auto')
            z_2d = tsne.fit_transform(z)
        else:
            z_2d = z

        scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=y_batch, cmap='tab10', alpha=0.7, s=15, edgecolors='none')
        ax.set_title(f"{name.upper()}", fontsize=12, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

    for i in range(num_models, len(axes)):
        axes[i].axis('off')

    if scatter is not None:
        fig.legend(
            *scatter.legend_elements(), 
            title="Classes", 
            loc="center left", 
            bbox_to_anchor=(1.02, 0.5), 
            fontsize=10, 
            title_fontsize=12
        )

    fig.suptitle(f"Latent Space t-SNE: {dataset_name}", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def plot_ultimate_attributions(models, X, X_man, y, timestamps, dataset_name, save_path, top_k=25):
    if not models: return
    
    batch_size = min(512, len(X))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=batch_size, replace=False)
    
    X_batch = X[idx]
    X_man_batch = X_man[idx]
    t = timestamps if len(timestamps) == X_batch.shape[1] else np.arange(X_batch.shape[1])

    mean_curve = np.squeeze(np.mean(X_batch, axis=0))
    std_curve = np.squeeze(np.std(X_batch, axis=0))
    if mean_curve.ndim > 1: 
        mean_curve = mean_curve.mean(axis=-1)
        std_curve = std_curve.mean(axis=-1)

    cols = min(5, top_k)  
    rows_per_model = math.ceil(top_k / cols)
    total_rows = len(models) * rows_per_model

    fig, axes = plt.subplots(total_rows, cols, figsize=(4.5 * cols, 3.5 * total_rows), sharex=True)
    if total_rows == 1 and cols == 1: axes = np.array([[axes]])
    elif total_rows == 1: axes = axes[np.newaxis, :]
    elif cols == 1: axes = axes[:, np.newaxis]

    for m_idx, (model_name, model) in enumerate(models.items()):
        top_dims, importances = get_top_dims_from_weights(model, top_k=top_k)
        latent_model = tf.keras.Model(model.input, model.layers[-2].output)
        
        model_inputs = [X_batch, X_man_batch] if model_name.endswith('_lf') else [X_batch]
        attributions = targeted_latent_attribution(latent_model, model_inputs, top_dims)
        
        for k_idx, dim in enumerate(top_dims):
            r_idx = (m_idx * rows_per_model) + (k_idx // cols)
            c_idx = k_idx % cols
            
            ax1 = axes[r_idx, c_idx]
            ax1.plot(t, mean_curve, color='k', lw=1.3, alpha=0.7)
            ax1.fill_between(t, mean_curve - std_curve, mean_curve + std_curve, color='k', alpha=0.10)
            ax1.tick_params(axis='y', labelcolor='k', labelsize=8)
            ax1.grid(True, color='grey', alpha=0.3, linestyle='--')
            
            if c_idx == 0: ax1.set_ylabel(f"{model_name.upper()}\nMean Curve", color='k', fontweight='bold')

            ax2 = ax1.twinx()
            ax2.plot(t, attributions[dim], color='C1', lw=1.5)
            ax2.tick_params(axis='y', labelcolor='C1', labelsize=8)
            
            if c_idx == cols - 1 or k_idx == len(top_dims) - 1: 
                ax2.set_ylabel('Attribution', color='C1', fontweight='bold')
                
            ax1.set_title(f'Top Dim #{dim}\nWeight Imp: {importances[dim]:.2f}', fontsize=11, pad=10)

        for k_idx in range(len(top_dims), rows_per_model * cols):
            r_idx = (m_idx * rows_per_model) + (k_idx // cols)
            c_idx = k_idx % cols
            axes[r_idx, c_idx].axis('off')

    fig.suptitle(f"Ultimate Latent Attribution | {dataset_name}", fontsize=16, fontweight='bold', y=1.01)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def plot_latent_saliency_heatmap(models, X, X_man, timestamps, top_10_features, dataset_name, base_save_path):
    """Generates independent heatmaps for Base, LF, and Dual architectures with PERFECT alignment."""
    if not models: return
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable 
    
    batch_size = min(512, len(X))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=batch_size, replace=False)
    X_batch = X[idx]
    X_man_batch = X_man[idx]
    
    t = timestamps if len(timestamps) == X_batch.shape[1] else np.arange(X_batch.shape[1])
    
    # Calculate perfect boundaries for imshow extent to prevent half-pixel shifting
    t_step = t[1] - t[0] if len(t) > 1 else 1
    t_start = t[0] - (t_step / 2)
    t_end = t[-1] + (t_step / 2)
    
    mean_curve = np.squeeze(np.mean(X_batch, axis=0))
    std_curve = np.squeeze(np.std(X_batch, axis=0))
    if mean_curve.ndim > 1: 
        mean_curve = mean_curve.mean(axis=-1)
        std_curve = std_curve.mean(axis=-1)

    for model_name, model in models.items():
        save_path = str(base_save_path).replace('.png', f'_{model_name}.png')
        
        is_lf = model_name.endswith('_lf')
        is_dual = model_name.endswith('_dual')
        
        x_tf_curve = tf.convert_to_tensor(X_batch, dtype=tf.float32)
        x_tf_man = tf.convert_to_tensor(X_man_batch, dtype=tf.float32)
        
        if is_lf:
            latent_model = tf.keras.Model(model.input, model.layers[-2].output)
            num_latent_dims = latent_model.output_shape[-1]
            ranked_dims, _ = get_top_dims_from_weights(model, top_k=min(100, num_latent_dims))
            
            heatmap_curve, heatmap_man = [], []
            for dim in ranked_dims:
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(x_tf_curve)
                    tape.watch(x_tf_man)
                    z_dim = latent_model([x_tf_curve, x_tf_man])[:, dim]
                    target = tf.reduce_mean(z_dim)
                
                grad_c = tape.gradient(target, x_tf_curve).numpy()
                grad_m = tape.gradient(target, x_tf_man).numpy()
                heatmap_curve.append(np.mean(np.abs(grad_c), axis=(0, 2)))
                heatmap_man.append(np.mean(np.abs(grad_m), axis=0))
                del tape
                
            fig, (ax_curve, ax_heat_c, ax_heat_m) = plt.subplots(3, 1, figsize=(9, 12), gridspec_kw={'height_ratios': [1, 2.5, 2.5]})
            ax_heat_c.sharex(ax_curve) 
            
            # 1. Mean Curve
            ax_curve.plot(t, mean_curve, color='black', lw=1.5)
            ax_curve.fill_between(t, mean_curve - std_curve, mean_curve + std_curve, color='gray', alpha=0.3)
            ax_curve.set_title(f"{model_name.upper()} - Mean Input Signal", fontsize=12, fontweight='bold', pad=15)
            ax_curve.set_xlim(t_start, t_end)
            ax_curve.grid(True, color='grey', alpha=0.3, linestyle='--')
            plt.setp(ax_curve.get_xticklabels(), visible=False)
            
            # 2. Heatmap 1: Curve
            im_c = ax_heat_c.imshow(np.array(heatmap_curve), aspect='auto', cmap='inferno', extent=[t_start, t_end, len(ranked_dims), 0], interpolation='nearest')
            ax_heat_c.set_title("Saliency: Time-Series Curves", fontsize=11, fontweight='bold', pad=10)
            ax_heat_c.set_ylabel("Latent Rank", fontsize=10)
            ax_heat_c.grid(True, color='grey', alpha=0.3, linestyle='--')

            # 3. Heatmap 2: Manual Features
            im_m = ax_heat_m.imshow(np.array(heatmap_man), aspect='auto', cmap='inferno', extent=[0, 10, len(ranked_dims), 0], interpolation='nearest')
            ax_heat_m.set_title("Saliency: Top 10 Features", fontsize=11, fontweight='bold', pad=10)
            ax_heat_m.set_ylabel("Latent Rank", fontsize=10)
            ax_heat_m.set_xticks(np.arange(10) + 0.5)
            ax_heat_m.set_xticklabels(top_10_features, rotation=45, ha='right', fontsize=9, fontweight='bold')
            ax_heat_m.grid(True, color='grey', alpha=0.3, linestyle='--')

            # --- PERFECT ALIGNMENT ---
            cax_curve = make_axes_locatable(ax_curve).append_axes("right", size="3%", pad=0.1)
            cax_curve.axis('off')
            
            cax_c = make_axes_locatable(ax_heat_c).append_axes("right", size="3%", pad=0.1)
            fig.colorbar(im_c, cax=cax_c).set_label("Attribution", rotation=270, labelpad=15)
            
            cax_m = make_axes_locatable(ax_heat_m).append_axes("right", size="3%", pad=0.1)
            fig.colorbar(im_m, cax=cax_m).set_label("Attribution", rotation=270, labelpad=15)
            
            fig.align_ylabels([ax_curve, ax_heat_c, ax_heat_m]) # Locks left spines
            # ---------------------

            fig.suptitle(f"{dataset_name} | {model_name.upper()} Saliency", fontsize=14, fontweight='bold', y=0.98)
            plt.subplots_adjust(hspace=0.25)
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)

        elif is_dual:
            concat_layer = [l for l in model.layers if isinstance(l, tf.keras.layers.Concatenate)][0]
            branch_cnn_model = tf.keras.Model(model.input, concat_layer.input[0])
            branch_rnn_model = tf.keras.Model(model.input, concat_layer.input[1])
            
            cnn_dims, rnn_dims = branch_cnn_model.output_shape[-1], branch_rnn_model.output_shape[-1]
            heatmap_cnn, heatmap_rnn = [], []
            
            for dim in range(cnn_dims):
                with tf.GradientTape() as tape:
                    tape.watch(x_tf_curve)
                    target = tf.reduce_mean(branch_cnn_model(x_tf_curve)[:, dim])
                heatmap_cnn.append(np.mean(np.abs(tape.gradient(target, x_tf_curve).numpy()), axis=(0, 2)))
                
            for dim in range(rnn_dims):
                with tf.GradientTape() as tape:
                    tape.watch(x_tf_curve)
                    target = tf.reduce_mean(branch_rnn_model(x_tf_curve)[:, dim])
                heatmap_rnn.append(np.mean(np.abs(tape.gradient(target, x_tf_curve).numpy()), axis=(0, 2)))
                
            heatmap_cnn = np.array(heatmap_cnn)
            heatmap_rnn = np.array(heatmap_rnn)
            cnn_order = np.argsort(np.sum(heatmap_cnn, axis=1))[::-1]
            rnn_order = np.argsort(np.sum(heatmap_rnn, axis=1))[::-1]

            fig, (ax_curve, ax_heat_c, ax_heat_r) = plt.subplots(3, 1, figsize=(9, 12), gridspec_kw={'height_ratios': [1, 2.5, 2.5]}, sharex=True)
            
            # 1. Mean Curve
            ax_curve.plot(t, mean_curve, color='black', lw=1.5)
            ax_curve.fill_between(t, mean_curve - std_curve, mean_curve + std_curve, color='gray', alpha=0.3)
            ax_curve.set_title(f"{model_name.upper()} - Mean Input Signal", fontsize=12, fontweight='bold', pad=15)
            ax_curve.set_xlim(t_start, t_end)
            ax_curve.grid(True, color='grey', alpha=0.3, linestyle='--')
            plt.setp(ax_curve.get_xticklabels(), visible=False)
            
            # 2. CNN Heatmap
            im_c = ax_heat_c.imshow(heatmap_cnn[cnn_order], aspect='auto', cmap='inferno', extent=[t_start, t_end, cnn_dims, 0], interpolation='nearest')
            ax_heat_c.set_title("Saliency: CNN Branch (Ranked by Magnitude)", fontsize=11, fontweight='bold', pad=10)
            ax_heat_c.set_ylabel("Latent Rank", fontsize=10)
            ax_heat_c.grid(True, color='grey', alpha=0.3, linestyle='--')

            # 3. RNN/Transformer Heatmap
            im_r = ax_heat_r.imshow(heatmap_rnn[rnn_order], aspect='auto', cmap='inferno', extent=[t_start, t_end, rnn_dims, 0], interpolation='nearest')
            branch_name = "Transformer" if "trans" in model_name else "GRU"
            ax_heat_r.set_title(f"Saliency: {branch_name} Branch (Ranked by Magnitude)", fontsize=11, fontweight='bold', pad=10)
            ax_heat_r.set_ylabel("Latent Rank", fontsize=10)
            ax_heat_r.set_xlabel("Time", fontsize=10, fontweight='bold')
            ax_heat_r.grid(True, color='grey', alpha=0.3, linestyle='--')

            # --- PERFECT ALIGNMENT ---
            cax_curve = make_axes_locatable(ax_curve).append_axes("right", size="3%", pad=0.1)
            cax_curve.axis('off')
            
            cax_c = make_axes_locatable(ax_heat_c).append_axes("right", size="3%", pad=0.1)
            fig.colorbar(im_c, cax=cax_c).set_label("Attribution", rotation=270, labelpad=15)
            
            cax_r = make_axes_locatable(ax_heat_r).append_axes("right", size="3%", pad=0.1)
            fig.colorbar(im_r, cax=cax_r).set_label("Attribution", rotation=270, labelpad=15)
            
            fig.align_ylabels([ax_curve, ax_heat_c, ax_heat_r]) # Locks left spines
            # ---------------------

            fig.suptitle(f"{dataset_name} | {model_name.upper()} Saliency", fontsize=14, fontweight='bold', y=0.98)
            plt.subplots_adjust(hspace=0.25)
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)

        else:
            # BASE MODELS
            latent_model = tf.keras.Model(model.input, model.layers[-2].output)
            num_latent_dims = latent_model.output_shape[-1]
            ranked_dims, _ = get_top_dims_from_weights(model, top_k=min(100, num_latent_dims))
            
            heatmap_curve = []
            for dim in ranked_dims:
                with tf.GradientTape() as tape:
                    tape.watch(x_tf_curve)
                    z_dim = latent_model(x_tf_curve)[:, dim]
                    target = tf.reduce_mean(z_dim)
                heatmap_curve.append(np.mean(np.abs(tape.gradient(target, x_tf_curve).numpy()), axis=(0, 2)))
                
            fig, (ax_curve, ax_heat) = plt.subplots(2, 1, figsize=(9, 8), gridspec_kw={'height_ratios': [1, 3]}, sharex=True)
            
            ax_curve.plot(t, mean_curve, color='black', lw=1.5)
            ax_curve.fill_between(t, mean_curve - std_curve, mean_curve + std_curve, color='gray', alpha=0.3)
            ax_curve.set_title(f"{model_name.upper()} - Mean Input Signal", fontsize=12, fontweight='bold', pad=15)
            ax_curve.set_xlim(t_start, t_end)
            ax_curve.grid(True, color='grey', alpha=0.3, linestyle='--')
            plt.setp(ax_curve.get_xticklabels(), visible=False)
            
            im = ax_heat.imshow(np.array(heatmap_curve), aspect='auto', cmap='inferno', extent=[t_start, t_end, len(ranked_dims), 0], interpolation='nearest')
            ax_heat.set_title("Saliency: Time-Series Curves", fontsize=11, fontweight='bold', pad=10)
            ax_heat.set_ylabel("Latent Rank", fontsize=10)
            ax_heat.set_xlabel("Time", fontsize=10, fontweight='bold')
            ax_heat.grid(True, color='grey', alpha=0.3, linestyle='--')
            
            # --- PERFECT ALIGNMENT ---
            cax_curve = make_axes_locatable(ax_curve).append_axes("right", size="3%", pad=0.1)
            cax_curve.axis('off')

            cax_heat = make_axes_locatable(ax_heat).append_axes("right", size="3%", pad=0.1)
            fig.colorbar(im, cax=cax_heat).set_label("Attribution", rotation=270, labelpad=15)
            
            fig.align_ylabels([ax_curve, ax_heat]) # Locks left spines
            # ---------------------

            fig.suptitle(f"{dataset_name} | {model_name.upper()} Saliency", fontsize=14, fontweight='bold', y=0.98)
            plt.subplots_adjust(hspace=0.15)
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)


# ====================================================================
# MODULE 4: PIPELINE ORCHESTRATOR
# ====================================================================
def run_interpretation_pipeline(exp_folder_path=config.DEFAULT_EXP_FOLDER, filter_key='amf_label_amf_important'):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    set_global_determinism(0)

    exp_folder = Path(exp_folder_path)
    global_vis_dir = exp_folder / "model_interpretation"
    global_vis_dir.mkdir(parents=True, exist_ok=True)
    
    exp_paths = sorted([p for p in exp_folder.iterdir() if p.is_dir() and p.name not in ['.DS_Store', 'model_interpretation']])
    
    for exp_path in exp_paths:
        print(f"\n{'='*70}\n[*] PROCESSING DATASET: {exp_path.name}\n{'='*70}")
        model_dir = exp_path / "model_interpretation"
        
        data_dict = prepare_dataset(exp_path, filter_key)
        if not data_dict:
            print(f"  [-] Skipping {exp_path.name}: Data not found.")
            continue
        print(f"  -> Data Loaded | X: {data_dict['X_full'].shape}, y: {data_dict['y_full'].shape}")

        print("  -> Loading Pre-Trained Models...")
        expected_seq_len = data_dict['X_full'].shape[1]
        models = load_saved_models(model_dir, str(filter_key), expected_seq_len)
        
        if not models:
            print(f"  [-] Skipping visualisations for {exp_path.name}: No compatible saved models found.")
            continue

        print(f"  -> Generating Visualizations into {global_vis_dir.name}/ ...")
        
        plot_latent_pca(
            models=models, 
            X=data_dict["X_full"], X_man=data_dict["X_man_full"], y=data_dict["y_full"], 
            dataset_name=data_dict["dataset_name"], 
            save_path=global_vis_dir / f"01_latent_space_pca_{exp_path.name}.png"
        )
        
        plot_latent_tsne(
            models=models, 
            X=data_dict["X_full"], X_man=data_dict["X_man_full"], y=data_dict["y_full"], 
            dataset_name=data_dict["dataset_name"], 
            save_path=global_vis_dir / f"02_latent_space_tsne_{exp_path.name}.png"
        )

        # plot_ultimate_attributions(
        #     models=models, 
        #     X=data_dict["X_full"], X_man=data_dict["X_man_full"], 
        #     y=data_dict["y_full"], timestamps=data_dict["timestamps"], 
        #     dataset_name=data_dict["dataset_name"], 
        #     save_path=global_vis_dir / f"03_ultimate_time_attribution_{exp_path.name}.png",
        #     top_k=25
        # )
        
        plot_latent_saliency_heatmap(
            models=models, 
            X=data_dict["X_full"], X_man=data_dict["X_man_full"], 
            timestamps=data_dict["timestamps"], 
            top_10_features=data_dict["top_10_features"],
            dataset_name=exp_path.name,
            base_save_path=global_vis_dir / f"04_saliency_heatmap_{exp_path.name}.png"
        )

        print(f"  [✓] Processed {exp_path.name}")
        tf.keras.backend.clear_session()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAI Visualization Pipeline for DDM Models")
    parser.add_argument("--exp_folder", type=str, default=config.DEFAULT_EXP_FOLDER, help="Path to experiment datasets")

    args = parser.parse_args()
    exp_folder = Path(args.exp_folder)
    print(exp_folder)
    run_interpretation_pipeline(exp_folder_path=exp_folder, filter_key=None)