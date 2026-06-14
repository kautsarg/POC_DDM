import os
import math
import joblib
import argparse
import numpy as np
import scipy.stats
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
import sys

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import mutual_info_classif
from sklearn.manifold import TSNE
from mpl_toolkits.axes_grid1 import make_axes_locatable 

sys.path.insert(0, "..")
sys.path.insert(0, "../outlier_detection")
import config
from model_utils import set_global_determinism

# Shared colour-blind-safe colormap for well-index class labels (0..N_WELLS-1).
WELL_CMAP = ListedColormap(config.WELL_COLORS)


# ====================================================================
# MODULE 1: ORIGINAL DATA & MODEL LOADING 
# ====================================================================
def prepare_dataset(exp_path, filter_key):
    """Loads dataset, extracts cached features from the LOCAL joblib, and returns splits."""
    data_path = exp_path / config.TRAINING_DATA_PATH
    if not data_path.exists():
        return None

    data = joblib.load(data_path)
    
    Y_well = data["Y_well"]
    if hasattr(config, "LABEL_MAPPINGS") and exp_path.name in config.LABEL_MAPPINGS:
        print(f"  [*] Applying custom target label mapping for experiment: {exp_path.name}")
        mapping = config.LABEL_MAPPINGS[exp_path.name]
        Y_well = [mapping.get(w, w) for w in Y_well]
    else:
        print(f"  [*] No custom mapping found for {exp_path.name}. Retaining default well labels.")

    encoder = LabelEncoder()
    y_full = encoder.fit_transform(Y_well)

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

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    train_idx, test_idx = next(splitter.split(X, y))

    joblib_path = exp_path / "model_interpretation" / "model_interpretation.joblib"
    saved_top_10 = None
    
    if joblib_path.exists():
        local_pkg = joblib.load(joblib_path)
        if local_pkg and "top_10_features" in local_pkg:
            saved_top_10 = local_pkg["top_10_features"].get(str(filter_key))

    if saved_top_10:
        top_10_features = saved_top_10
    else:
        X_candidates_train = features_df_masked.iloc[train_idx][config.LD_FEATURES].values
        X_candidates_train = np.nan_to_num(X_candidates_train, nan=0.0, posinf=0.0, neginf=0.0)
        mi_scores = mutual_info_classif(X_candidates_train, y[train_idx], random_state=0)
        top_10_idx = np.argsort(mi_scores)[-10:][::-1]
        top_10_features = [config.LD_FEATURES[i] for i in top_10_idx]

    X_man = features_df_masked[top_10_features].values
    X_man = np.nan_to_num(X_man, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    scaler = StandardScaler()
    X_man_scaled = np.empty_like(X_man)
    X_man_scaled[train_idx] = scaler.fit_transform(X_man[train_idx])
    X_man_scaled[test_idx] = scaler.transform(X_man[test_idx])
    
    # Extract temporal timestamps for features if available in the dataframe
    T_man = None
    time_cols = [f"{feat}_time_idx" for feat in top_10_features]
    if all(col in features_df_masked.columns for col in time_cols):
        T_man = features_df_masked[time_cols].values

    return {
        "X_full": X, "X_man_full": X_man_scaled, "y_full": y,
        "timestamps": timestamps, "dataset_name": dataset_name,
        "top_10_features": top_10_features, "T_man_full": T_man
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
                    continue
                models[name] = model
            except Exception as e:
                print(f"     [!] Failed to load {name}: {str(e)}")
            
    return models


# ====================================================================
# MODULE 2: ORIGINAL LATENT CLUSTERING & HELPERS
# ====================================================================
def plot_latent_pca(models, X, X_man, y, dataset_name, save_path):
    if not models: return
    num_models = len(models)
    cols = min(3, num_models)
    rows = math.ceil(num_models / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if num_models == 1: axes = [axes]
    else: axes = axes.flatten()

    scatter = None
    for ax, (name, model) in zip(axes, models.items()):
        latent_model = tf.keras.Model(model.input, model.layers[-2].output)
        model_inputs = [X, X_man] if name.endswith('_lf') else X

        z = latent_model.predict(model_inputs, verbose=0)
        z_2d = PCA(n_components=2, random_state=0).fit_transform(z) if z.shape[1] > 2 else z

        scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=y, cmap=WELL_CMAP, vmin=0, vmax=config.N_WELLS - 1, alpha=0.7, s=15, edgecolors='none')
        ax.set_title(f"{name.upper()}", fontsize=12, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

    for i in range(num_models, len(axes)): axes[i].axis('off')

    if scatter is not None:
        fig.legend(*scatter.legend_elements(), title="Classes", loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10, title_fontsize=12)

    fig.suptitle(f"Latent Space Clustering: {dataset_name}", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def plot_latent_tsne(models, X, X_man, y, dataset_name, save_path, max_samples=2000):
    if not models: return
    
    if len(X) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X_batch, y_batch = X[idx], y[idx]
        X_man_batch = X_man[idx] if X_man is not None else None
    else:
        X_batch, X_man_batch, y_batch = X, X_man, y

    num_models = len(models)
    cols = min(3, num_models)
    rows = math.ceil(num_models / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if num_models == 1: axes = [axes]
    else: axes = axes.flatten()

    scatter = None
    for ax, (name, model) in zip(axes, models.items()):
        latent_model = tf.keras.Model(model.input, model.layers[-2].output)
        model_inputs = [X_batch, X_man_batch] if name.endswith('_lf') else X_batch

        z = latent_model.predict(model_inputs, verbose=0)
        if z.shape[1] > 2:
            z_2d = TSNE(n_components=2, random_state=0, init='pca', learning_rate='auto').fit_transform(z)
        else:
            z_2d = z

        scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=y_batch, cmap=WELL_CMAP, vmin=0, vmax=config.N_WELLS - 1, alpha=0.7, s=15, edgecolors='none')
        ax.set_title(f"{name.upper()}", fontsize=12, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

    for i in range(num_models, len(axes)): axes[i].axis('off')

    if scatter is not None:
        fig.legend(*scatter.legend_elements(), title="Classes", loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10, title_fontsize=12)

    fig.suptitle(f"Latent Space t-SNE: {dataset_name}", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ====================================================================
# MODULE 3: ARCHITECTURE & MATH EXTRACTORS
# ====================================================================
# User's Original Helpers
def get_top_dims_from_weights(model, top_k=25):
    final_dense_layer = model.layers[-1]
    W, b = final_dense_layer.get_weights()
    global_importance = np.sum(np.abs(W), axis=1)
    actual_top_k = min(top_k, len(global_importance))
    return np.argsort(global_importance)[::-1][:actual_top_k], global_importance

def targeted_latent_attribution(latent_model, x_inputs, target_dims):
    x_tf_curve = tf.convert_to_tensor(x_inputs[0], dtype=tf.float32)
    model_inputs = [x_tf_curve, tf.convert_to_tensor(x_inputs[1], dtype=tf.float32)] if len(x_inputs) > 1 else x_tf_curve
    attributions = {}
    for dim in target_dims:
        with tf.GradientTape() as tape:
            tape.watch(x_tf_curve)
            target = tf.reduce_mean(latent_model(model_inputs)[:, dim])
        grads = tape.gradient(target, x_tf_curve).numpy()        
        attributions[dim] = np.mean(np.abs(grads), axis=(0, 2))     
    return attributions

# Safe Layer Extraction
def find_bidirectional_recurrent_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Bidirectional):
            if isinstance(layer.forward_layer, (tf.keras.layers.GRU, tf.keras.layers.LSTM, tf.keras.layers.SimpleRNN)):
                return layer
    return None

def find_flatten_layer(model):
    flatten_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Flatten)]
    return flatten_layers[-1] if flatten_layers else None

def find_transformer_block_output(model):
    layer_norm_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.LayerNormalization)]
    return layer_norm_layers[-1] if layer_norm_layers else None

def normalize_heatmap(matrix, method='row'):
    if len(matrix) == 0: return matrix
    if method == 'row':
        mins = np.min(matrix, axis=1, keepdims=True)
        maxs = np.max(matrix, axis=1, keepdims=True)
        val_range = maxs - mins
        val_range[val_range == 0] = 1e-10 
        return (matrix - mins) / val_range
    return matrix

def rank_latents(dy_dz_np):
    importance = np.mean(np.abs(dy_dz_np), axis=0)
    imp_shape  = importance.shape
    imp_flat   = importance.flatten()
    order      = np.argsort(imp_flat)[::-1][:min(100, len(imp_flat))]
    return order, imp_shape

def compute_latent_saliency_batch(extractor, x_tf, order, imp_shape, input_tensor):
    """Computes dZ/dX but preserves the Batch dimension. Returns shape (Batch, Time)."""
    is_3d = len(imp_shape) == 2   
    saliency_maps = []
    for flat_dim in order:
        with tf.GradientTape() as tape:
            tape.watch(x_tf)
            z = extractor(input_tensor)
            if is_3d:
                row, col = np.unravel_index(int(flat_dim), imp_shape)
                target = z[:, row, col] 
            else:
                target = z[:, int(flat_dim)]
                
        grad = tape.gradient(target, x_tf).numpy()
        # Safely squeeze only the last dimension if it's the channel dimension
        if grad.ndim == 3 and grad.shape[-1] == 1:
            grad = grad.squeeze(axis=-1)
        saliency_maps.append(np.abs(grad)) 
    return saliency_maps

# ====================================================================
# MODULE 4: THE OPTIMIZED ARTIFACT EXTRACTOR 
# ====================================================================
def extract_xai_artifacts(models, X_batch, X_man_batch):
    """Runs all TF Gradient operations once and saves the matrices for all plots."""
    artifacts = {}
    x_tf_curve = tf.convert_to_tensor(X_batch, dtype=tf.float32)
    x_tf_man = tf.convert_to_tensor(X_man_batch, dtype=tf.float32)
    
    for model_name, model in models.items():
        print(f"    [+] Computing Gradients: {model_name}")
        is_lf = model_name.endswith('_lf')
        is_dual = model_name.endswith('_dual')
        
        recurrent_layer = find_bidirectional_recurrent_layer(model)
        transformer_layer = find_transformer_block_output(model)
        flatten_layer = find_flatten_layer(model)
        
        # --------------------------------------------------------
        # LATE FUSION MODELS
        if is_lf:
            concat_layer = next(l for l in model.layers if isinstance(l, tf.keras.layers.Concatenate))
            
            if recurrent_layer: extractor_curve = tf.keras.Model(inputs=model.input[0], outputs=recurrent_layer.output)
            elif transformer_layer: extractor_curve = tf.keras.Model(inputs=model.input[0], outputs=transformer_layer.output)
            elif flatten_layer: extractor_curve = tf.keras.Model(inputs=model.input[0], outputs=flatten_layer.output)
            else: extractor_curve = tf.keras.Model(inputs=model.input[0], outputs=concat_layer.input[0])
            
            extractor_man = tf.keras.Model(inputs=model.input[1], outputs=concat_layer.input[1])
            head_model = tf.keras.Model(inputs=[extractor_curve.output, extractor_man.output], outputs=model.output)
            
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x_tf_curve)
                tape.watch(x_tf_man)
                
                z_curve = extractor_curve(x_tf_curve)
                z_man = extractor_man(x_tf_man)
                tape.watch(z_curve)
                tape.watch(z_man)
                
                target_master = tf.reduce_max(head_model([z_curve, z_man]), axis=1)
                
            dy_dz_curve = tape.gradient(target_master, z_curve).numpy()
            dy_dz_man = tape.gradient(target_master, z_man).numpy()
            
            curve_order, curve_imp_shape = rank_latents(dy_dz_curve)
            man_order, man_imp_shape = rank_latents(dy_dz_man)
            
            raw_saliency_curve = compute_latent_saliency_batch(extractor_curve, x_tf_curve, curve_order, curve_imp_shape, x_tf_curve)
            raw_saliency_man = compute_latent_saliency_batch(extractor_man, x_tf_man, man_order, man_imp_shape, x_tf_man)
            
            master_sal = np.mean(np.abs(tape.gradient(target_master, x_tf_curve).numpy()), axis=(0, 2))
            del tape
            
            artifacts[model_name] = {
                "is_type": "lf", "master_saliency": master_sal,
                "z_curve": z_curve.numpy(), "curve_order": curve_order, "curve_imp_shape": curve_imp_shape, "raw_saliency_curve": raw_saliency_curve,
                "z_man": z_man.numpy(), "man_order": man_order, "man_imp_shape": man_imp_shape, "raw_saliency_man": raw_saliency_man
            }

        # --------------------------------------------------------
        # DUAL MODELS
        elif is_dual:
            concat_layer = next(l for l in model.layers if isinstance(l, tf.keras.layers.Concatenate))
            
            if flatten_layer: extractor_cnn = tf.keras.Model(inputs=model.input, outputs=flatten_layer.output)
            else: extractor_cnn = tf.keras.Model(inputs=model.input, outputs=concat_layer.input[0])
            
            if recurrent_layer: extractor_rnn = tf.keras.Model(inputs=model.input, outputs=recurrent_layer.output)
            elif transformer_layer: extractor_rnn = tf.keras.Model(inputs=model.input, outputs=transformer_layer.output)
            else: extractor_rnn = tf.keras.Model(inputs=model.input, outputs=concat_layer.input[1])
            
            head_model = tf.keras.Model(inputs=[extractor_cnn.output, extractor_rnn.output], outputs=model.output)
            
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x_tf_curve)
                
                z_cnn = extractor_cnn(x_tf_curve)
                z_rnn = extractor_rnn(x_tf_curve)
                tape.watch(z_cnn)
                tape.watch(z_rnn)
                
                target_master = tf.reduce_max(head_model([z_cnn, z_rnn]), axis=1)
                
            dy_dz_cnn = tape.gradient(target_master, z_cnn).numpy()
            dy_dz_rnn = tape.gradient(target_master, z_rnn).numpy()
            
            cnn_order, cnn_imp_shape = rank_latents(dy_dz_cnn)
            rnn_order, rnn_imp_shape = rank_latents(dy_dz_rnn)
            
            raw_saliency_cnn = compute_latent_saliency_batch(extractor_cnn, x_tf_curve, cnn_order, cnn_imp_shape, x_tf_curve)
            raw_saliency_rnn = compute_latent_saliency_batch(extractor_rnn, x_tf_curve, rnn_order, rnn_imp_shape, x_tf_curve)
            
            master_sal = np.mean(np.abs(tape.gradient(target_master, x_tf_curve).numpy()), axis=(0, 2))
            del tape
            
            artifacts[model_name] = {
                "is_type": "dual", "master_saliency": master_sal,
                "z_curve": z_cnn.numpy(), "curve_order": cnn_order, "curve_imp_shape": cnn_imp_shape, "raw_saliency_curve": raw_saliency_cnn,
                "z_rnn": z_rnn.numpy(), "rnn_order": rnn_order, "rnn_imp_shape": rnn_imp_shape, "raw_saliency_rnn": raw_saliency_rnn
            }

        # --------------------------------------------------------
        # BASE MODELS
        else:
            if recurrent_layer: extractor = tf.keras.Model(inputs=model.input, outputs=recurrent_layer.output)
            elif transformer_layer: extractor = tf.keras.Model(inputs=model.input, outputs=transformer_layer.output)
            elif flatten_layer: extractor = tf.keras.Model(inputs=model.input, outputs=flatten_layer.output)
            else: extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
            
            head_model = tf.keras.Model(inputs=extractor.output, outputs=model.output)
            
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x_tf_curve)
                z_curve = extractor(x_tf_curve)
                tape.watch(z_curve)
                target_master = tf.reduce_max(head_model(z_curve), axis=1)
                
            dy_dz = tape.gradient(target_master, z_curve).numpy()
            curve_order, curve_imp_shape = rank_latents(dy_dz)
            
            raw_saliency_curve = compute_latent_saliency_batch(extractor, x_tf_curve, curve_order, curve_imp_shape, x_tf_curve)
            master_sal = np.mean(np.abs(tape.gradient(target_master, x_tf_curve).numpy()), axis=(0, 2))
            del tape
            
            artifacts[model_name] = {
                "is_type": "base", "master_saliency": master_sal,
                "z_curve": z_curve.numpy(), "curve_order": curve_order, "curve_imp_shape": curve_imp_shape, "raw_saliency_curve": raw_saliency_curve
            }
            
    return artifacts


# ====================================================================
# MODULE 5: YOUR RESTORED VISUALIZATION FUNCTIONS (Optimized)
# ====================================================================
def plot_latent_saliency_heatmap(artifacts, model_name, timestamps, top_10_features, mean_curve, std_curve, dataset_name, base_save_path, normalize=False):
    """Restores your exact 3-subplot layout for LF/Dual architectures using the optimized artifact dict."""
    art = artifacts.get(model_name)
    if not art: return

    save_path = str(base_save_path).replace('.png', f'_{model_name}.png')
    t = timestamps if len(timestamps) == len(mean_curve) else np.arange(len(mean_curve))
    t_step = t[1] - t[0] if len(t) > 1 else 1
    t_start, t_end = t[0] - (t_step / 2), t[-1] + (t_step / 2)

    master_sal = art["master_saliency"]
    vmin, vmax = (0, 1) if normalize else (None, None)

    if art["is_type"] == "lf":
        heatmap_curve = np.array([np.mean(m, axis=0) for m in art["raw_saliency_curve"]])
        heatmap_man = np.array([np.mean(m, axis=0) for m in art["raw_saliency_man"]])
        if normalize:
            heatmap_curve = normalize_heatmap(heatmap_curve, method='row')
            heatmap_man = normalize_heatmap(heatmap_man, method='row')

        fig, (ax_curve, ax_heat_c, ax_heat_m) = plt.subplots(3, 1, figsize=(9, 12), gridspec_kw={'height_ratios': [1, 2.5, 2.5]})
        ax_heat_c.sharex(ax_curve)

        ax_curve.plot(t, mean_curve, color='black', lw=1.5, label='Mean Input')
        ax_curve.fill_between(t, mean_curve - std_curve, mean_curve + std_curve, color='gray', alpha=0.3)
        # ax_curve_twin = ax_curve.twinx()
        # ax_curve_twin.plot(t, master_sal, color='red', lw=1.5, linestyle='--', alpha=0.8, label='Master Output Saliency')
        ax_curve.set_title(f"{model_name.upper()} - Final Output Decision vs Mean Signal", fontsize=12, fontweight='bold', pad=15)
        ax_curve.set_xlim(t_start, t_end)
        ax_curve.grid(True, color='grey', alpha=0.3, linestyle='--')
        # ax_curve_twin.tick_params(axis='y', labelcolor='red')
        plt.setp(ax_curve.get_xticklabels(), visible=False)

        im_c = ax_heat_c.imshow(heatmap_curve, aspect='auto', cmap='inferno', vmin=vmin, vmax=vmax, extent=[t_start, t_end, len(heatmap_curve), 0], interpolation='nearest')
        ax_heat_c.set_title(f"Curve Branch Saliency (Top {len(heatmap_curve)} Dims, Ranked by dY/dZ)", fontsize=11, fontweight='bold', pad=10)
        ax_heat_c.set_ylabel("Rank (1 = Highest Impact)", fontsize=10)
        ax_heat_c.grid(True, color='grey', alpha=0.3, linestyle='--')

        im_m = ax_heat_m.imshow(heatmap_man, aspect='auto', cmap='inferno', vmin=vmin, vmax=vmax, extent=[0, 10, len(heatmap_man), 0], interpolation='nearest')
        ax_heat_m.set_title("Manual Features Saliency (Ranked by dY/dZ)", fontsize=11, fontweight='bold', pad=10)
        ax_heat_m.set_ylabel("Rank", fontsize=10)
        ax_heat_m.set_xticks(np.arange(10) + 0.5)
        ax_heat_m.set_xticklabels(top_10_features, rotation=45, ha='right', fontsize=9, fontweight='bold')
        ax_heat_m.grid(True, color='grey', alpha=0.3, linestyle='--')

        cax_curve = make_axes_locatable(ax_curve).append_axes("right", size="3%", pad=0.1); cax_curve.axis('off')
        fig.colorbar(im_c, cax=make_axes_locatable(ax_heat_c).append_axes("right", size="3%", pad=0.1)).set_label("Normalized Attrib", rotation=270, labelpad=15)
        fig.colorbar(im_m, cax=make_axes_locatable(ax_heat_m).append_axes("right", size="3%", pad=0.1)).set_label("Normalized Attrib", rotation=270, labelpad=15)
        fig.align_ylabels([ax_curve, ax_heat_c, ax_heat_m]) 

    elif art["is_type"] == "dual":
        heatmap_cnn = np.array([np.mean(m, axis=0) for m in art["raw_saliency_curve"]])
        heatmap_rnn = np.array([np.mean(m, axis=0) for m in art["raw_saliency_rnn"]])
        if normalize:
            heatmap_cnn = normalize_heatmap(heatmap_cnn, method='row')
            heatmap_rnn = normalize_heatmap(heatmap_rnn, method='row')

        fig, (ax_curve, ax_heat_c, ax_heat_r) = plt.subplots(3, 1, figsize=(9, 12), gridspec_kw={'height_ratios': [1, 2.5, 2.5]}, sharex=True)
        
        ax_curve.plot(t, mean_curve, color='black', lw=1.5, label='Mean Input')
        ax_curve.fill_between(t, mean_curve - std_curve, mean_curve + std_curve, color='gray', alpha=0.3)
        # ax_curve_twin = ax_curve.twinx()
        # ax_curve_twin.plot(t, master_sal, color='red', lw=1.5, linestyle='--', alpha=0.8, label='Master Output Saliency')
        ax_curve.set_title(f"{model_name.upper()} - Final Output Decision vs Mean Signal", fontsize=12, fontweight='bold', pad=15)
        ax_curve.set_xlim(t_start, t_end)
        ax_curve.grid(True, color='grey', alpha=0.3, linestyle='--')
        # ax_curve_twin.tick_params(axis='y', labelcolor='red')
        plt.setp(ax_curve.get_xticklabels(), visible=False)
        
        im_c = ax_heat_c.imshow(heatmap_cnn, aspect='auto', cmap='inferno', vmin=vmin, vmax=vmax, extent=[t_start, t_end, len(heatmap_cnn), 0], interpolation='nearest')
        ax_heat_c.set_title(f"CNN Branch Saliency (Top {len(heatmap_cnn)} Dims, Ranked by dY/dZ)", fontsize=11, fontweight='bold', pad=10)
        ax_heat_c.set_ylabel("Rank (1 = Highest Impact)", fontsize=10)
        ax_heat_c.grid(True, color='grey', alpha=0.3, linestyle='--')

        im_r = ax_heat_r.imshow(heatmap_rnn, aspect='auto', cmap='inferno', vmin=vmin, vmax=vmax, extent=[t_start, t_end, len(heatmap_rnn), 0], interpolation='nearest')
        branch_name = "Transformer" if "trans" in model_name else "GRU"
        ax_heat_r.set_title(f"{branch_name} Branch Saliency (Top {len(heatmap_rnn)} Dims, Ranked by dY/dZ)", fontsize=11, fontweight='bold', pad=10)
        ax_heat_r.set_ylabel("Rank", fontsize=10)
        ax_heat_r.set_xlabel("Time", fontsize=10, fontweight='bold')
        ax_heat_r.grid(True, color='grey', alpha=0.3, linestyle='--')

        cax_curve = make_axes_locatable(ax_curve).append_axes("right", size="3%", pad=0.1); cax_curve.axis('off')
        fig.colorbar(im_c, cax=make_axes_locatable(ax_heat_c).append_axes("right", size="3%", pad=0.1)).set_label("Normalized Attrib", rotation=270, labelpad=15)
        fig.colorbar(im_r, cax=make_axes_locatable(ax_heat_r).append_axes("right", size="3%", pad=0.1)).set_label("Normalized Attrib", rotation=270, labelpad=15)
        fig.align_ylabels([ax_curve, ax_heat_c, ax_heat_r]) 

    else:
        heatmap_curve = np.array([np.mean(m, axis=0) for m in art["raw_saliency_curve"]])
        if normalize:
            heatmap_curve = normalize_heatmap(heatmap_curve, method='row')

        fig, (ax_curve, ax_heat) = plt.subplots(2, 1, figsize=(9, 8), gridspec_kw={'height_ratios': [1, 3]}, sharex=True)
        
        ax_curve.plot(t, mean_curve, color='black', lw=1.5, label='Mean Input')
        ax_curve.fill_between(t, mean_curve - std_curve, mean_curve + std_curve, color='gray', alpha=0.3)
        # ax_curve_twin = ax_curve.twinx()
        # ax_curve_twin.plot(t, master_sal, color='red', lw=1.5, linestyle='--', alpha=0.8, label='Master Output Saliency')
        ax_curve.set_title(f"{model_name.upper()} - Final Output Decision vs Mean Signal", fontsize=12, fontweight='bold', pad=15)
        ax_curve.set_xlim(t_start, t_end)
        ax_curve.grid(True, color='grey', alpha=0.3, linestyle='--')
        # ax_curve_twin.tick_params(axis='y', labelcolor='red')
        plt.setp(ax_curve.get_xticklabels(), visible=False)
        
        im = ax_heat.imshow(heatmap_curve, aspect='auto', cmap='inferno', vmin=vmin, vmax=vmax, extent=[t_start, t_end, len(heatmap_curve), 0], interpolation='nearest')
        ax_heat.set_title(f"dZ/dX Spatial Focus (Top {len(heatmap_curve)} Dims, Ranked by dY/dZ)", fontsize=11, fontweight='bold', pad=10)
        ax_heat.set_ylabel("Rank (1 = Highest Impact)", fontsize=10)
        ax_heat.set_xlabel("Time", fontsize=10, fontweight='bold')
        
        cax_curve = make_axes_locatable(ax_curve).append_axes("right", size="3%", pad=0.1); cax_curve.axis('off')
        fig.colorbar(im, cax=make_axes_locatable(ax_heat).append_axes("right", size="3%", pad=0.1)).set_label("Normalized Attrib", rotation=270, labelpad=15)
        fig.align_ylabels([ax_curve, ax_heat]) 

    fig.suptitle(f"{dataset_name} | {model_name.upper()} Causal Saliency", fontsize=14, fontweight='bold', y=0.98)
    plt.subplots_adjust(hspace=0.25 if art["is_type"] != "base" else 0.15)
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def plot_concept_alignment_matrix(artifacts, model_name, X_man_batch, top_10_features, dataset_name, base_save_path):
    art = artifacts.get(model_name)
    if not art: return
    
    save_path = str(base_save_path).replace('.png', f'_concept_alignment_{model_name}.png')
    TOP_K = min(15, len(art["curve_order"]))
    
    z_np = art["z_curve"]
    order = art["curve_order"][:TOP_K]
    imp_shape = art["curve_imp_shape"]
    
    z_traces = []
    is_3d = len(imp_shape) == 2
    for flat_idx in order:
        if is_3d:
            row, col = np.unravel_index(int(flat_idx), imp_shape)
            z_traces.append(z_np[:, row, col])
        else:
            z_traces.append(z_np[:, int(flat_idx)])
            
    z_traces = np.array(z_traces) # (TOP_K, Batch)
    num_feats = X_man_batch.shape[1]
    correlation_matrix = np.zeros((TOP_K, num_feats))
    
    for i in range(TOP_K):
        for j in range(num_feats):
            corr, _ = scipy.stats.spearmanr(z_traces[i], X_man_batch[:, j])
            correlation_matrix[i, j] = 0.0 if np.isnan(corr) else corr
            
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(np.abs(correlation_matrix), cmap='Blues', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(num_feats))
    ax.set_yticks(np.arange(TOP_K))
    ax.set_xticklabels(top_10_features, rotation=45, ha="right", fontsize=10, fontweight='bold')
    ax.set_yticklabels([f"Latent Dim {d}" for d in order], fontsize=10, fontweight='bold')
    
    ax.set_title(f"Concept Alignment: {model_name.upper()}\nTop Causal Latent Dims vs. Manual Features", fontsize=14, fontweight='bold', pad=15)
    for i in range(TOP_K):
        for j in range(num_feats):
            val = correlation_matrix[i, j]
            text_color = "white" if np.abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    fig.colorbar(im, cax=cax).set_label("Absolute Spearman Correlation ($|\\rho|$)", rotation=270, labelpad=15, fontweight='bold')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def plot_temporal_alignment_matrix(artifacts, model_name, T_man_batch, top_10_features, dataset_name, base_save_path):
    if T_man_batch is None: 
        return # Skips gracefully if the user doesn't pass T_man
        
    art = artifacts.get(model_name)
    if not art: return
    
    save_path = str(base_save_path).replace('.png', f'_temporal_alignment_{model_name}.png')
    TOP_K = min(15, len(art["curve_order"]))
    raw_maps = art["raw_saliency_curve"][:TOP_K] 
    order = art["curve_order"][:TOP_K]
    
    num_feats = T_man_batch.shape[1]
    temporal_error_matrix = np.zeros((TOP_K, num_feats))
    
    for idx, batch_saliency in enumerate(raw_maps):
        t_peaks = np.argmax(batch_saliency, axis=1) # Shape: (Batch,)
        
        for j in range(num_feats):
            time_distances = np.abs(t_peaks - T_man_batch[:, j])
            temporal_error_matrix[idx, j] = np.mean(time_distances)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(temporal_error_matrix, cmap='viridis_r', aspect='auto', vmin=0, vmax=100) 
    
    ax.set_xticks(np.arange(num_feats))
    ax.set_yticks(np.arange(TOP_K))
    ax.set_xticklabels(top_10_features, rotation=45, ha="right", fontsize=10, fontweight='bold')
    ax.set_yticklabels([f"Latent Dim {d}" for d in order], fontsize=10, fontweight='bold')
    
    ax.set_title(f"Temporal Grounding: {model_name.upper()}\nDistance between Saliency Peak and Manual Feature Timestamp", fontsize=14, fontweight='bold', pad=15)
    for i in range(TOP_K):
        for j in range(num_feats):
            val = temporal_error_matrix[i, j]
            text_color = "white" if val < 30 else "black" 
            ax.text(j, i, f"{val:.1f} steps", ha="center", va="center", color=text_color, fontsize=8)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    fig.colorbar(im, cax=cax).set_label("Mean Temporal Error (Time-Steps)", rotation=270, labelpad=15, fontweight='bold')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ====================================================================
# MODULE 6: LATENT → FEATURE MAPPING
# ====================================================================
#
# All SCALAR outputs of extract_kinetic_parameters_original are used.
# Array outputs (5p_sigmoid_fitted, 5p_sigmoid_fitted_dydx) and pure
# index artifacts (ct_idx, ct_idx_ori) are excluded.
#
# Three complementary scores are fused per (latent, feature) pair:
#   1. |Spearman ρ|   – rank-correlation of latent activation vs feature value
#   2. MI (normed)    – mutual information, catches non-linear links
#   3. Cosine sim     – alignment of mean saliency profile vs finite-difference
#                       feature-sensitivity profile over time
#
# Weights: Spearman 0.35 · MI 0.25 · Cosine 0.40
#
# Visualisation layout (one figure per model):
#   - TOP_K rows, one per ranked latent
#   - Each row = [mini curve panel | horizontal score bar across all features]
#     · Mini panel: mean input curve (black) + ±1σ band (grey) +
#                   latent saliency profile (red fill, right y-axis)
#     · Score bar:  colour-coded combined score for every feature,
#                   best assignment annotated with ★ and feature name

# Scalar keys returned by extract_kinetic_parameters_original
# (arrays and index artifacts excluded)
_ALL_KINETIC_SCALAR_KEYS = [
    'Fm', 'Fb', 'Sc', 'Cs', 'As',
    'xms', 'xs', 'xe', 'xp1', 'xp2', 'TH',
    'y_xms', 'y_xs', 'y_xe', 'y_xp1', 'y_xp2', 'amplitude',
    'dy_xms', 'dy_xp1', 'dy_xp2', 'd2y_xp1', 'd2y_xp2',
    'threshold_distance', 'first_half_distance', 'second_half_distance',
    'distance_asymmetry_index', 'peak_shifting_distance',
    'A1', 'A2', 'area_asymmetry_index', 'peak_asymmetry_index',
    'Ct', 'Cy0', 'F_max', 'Ct_ori', 'Cy0_ori', 'F_max_ori',
    'log_F0', 'F0',
    'Send', 'Send_abs', 'Send_fit', 'Send_fit_abs',
    'baseline_mean', 'baseline_std', 'baseline_slope',
    'plateau_mean', 'plateau_std', 'plateau_slope',
    't10', 't50', 't90', 'rise_time_10_90', 'rise_time_20_80', 'lag_time',
    'snr_peak', 'snr_xms',
    'auc', 'auc_norm',
    'max_accel', 'min_accel', 'accel_fwhm',
    'fit_rmse', 'fit_r2',
    'overshoot_index',
]


def _extract_all_kinetic_features_batch(X_batch, timestamps):
    """
    Run extract_kinetic_parameters_original on every sample in X_batch.

    Returns
    -------
    feat_matrix : np.ndarray  (Batch, n_scalar_features)
    feat_names  : list of str  – matches columns of feat_matrix
    """
    from sigmoid_fitting import extract_kinetic_parameters_original

    X_sq = np.squeeze(X_batch, axis=-1)          # (Batch, T)
    t    = np.asarray(timestamps, dtype=float)
    T    = X_sq.shape[0]
    if len(t) != X_sq.shape[1]:
        t = np.linspace(0, X_sq.shape[1] - 1, X_sq.shape[1])

    rows = []
    for s in range(T):
        try:
            d = extract_kinetic_parameters_original(t, X_sq[s])
        except Exception:
            d = {}
        row = [float(d.get(k, np.nan)) for k in _ALL_KINETIC_SCALAR_KEYS]
        rows.append(row)

    feat_matrix = np.array(rows, dtype=np.float32)  # (Batch, n_feats)
    return feat_matrix, list(_ALL_KINETIC_SCALAR_KEYS)


def _compute_feature_sensitivity_profiles_all(X_batch, timestamps, feat_names,
                                               delta_frac=0.05, n_sub=48):
    """
    Finite-difference sensitivity profile for every feature in feat_names.

    For each time-step t, perturb X[s, t] by +δ and measure |Δfeature| / δ.
    Uses a small random subset of n_sub curves for speed.

    Returns
    -------
    sensitivity : np.ndarray  (n_feats, T)  – each row normalised to [0, 1]
    """
    from sigmoid_fitting import extract_kinetic_parameters_original

    X_sq = np.squeeze(X_batch, axis=-1)
    t    = np.asarray(timestamps, dtype=float)
    if len(t) != X_sq.shape[1]:
        t = np.linspace(0, X_sq.shape[1] - 1, X_sq.shape[1])

    rng    = np.random.default_rng(7)
    n_sub  = min(n_sub, X_sq.shape[0])
    sub    = rng.choice(X_sq.shape[0], size=n_sub, replace=False)
    X_sub  = X_sq[sub]

    n_feats = len(feat_names)
    T_len   = X_sq.shape[1]
    delta   = delta_frac * (X_sub.max() - X_sub.min() + 1e-8)
    sensitivity = np.zeros((n_feats, T_len), dtype=np.float32)

    for t_idx in range(T_len):
        X_pert = X_sub.copy()
        X_pert[:, t_idx] += delta
        diffs = np.zeros((n_sub, n_feats), dtype=np.float32)
        for s in range(n_sub):
            try:
                base = extract_kinetic_parameters_original(t, X_sub[s])
                pert = extract_kinetic_parameters_original(t, X_pert[s])
                for fi, fn in enumerate(feat_names):
                    bv = float(base.get(fn, np.nan))
                    pv = float(pert.get(fn, np.nan))
                    if np.isfinite(bv) and np.isfinite(pv):
                        diffs[s, fi] = abs(pv - bv) / delta
            except Exception:
                pass
        sensitivity[:, t_idx] = np.mean(diffs, axis=0)

    for fi in range(n_feats):
        mx = sensitivity[fi].max()
        sensitivity[fi] = sensitivity[fi] / mx if mx > 0 else np.ones(T_len) / T_len

    return sensitivity   # (n_feats, T)


def _cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0


def compute_kinetic_feature_cache(X_batch, timestamps):
    """
    Pre-compute the two data-side artefacts that are the same for all models:
      - feat_matrix      : (Batch, n_feats) – kinetic feature values per sample
      - feat_sensitivity : (n_feats, T)     – finite-diff sensitivity profiles
      - feat_names       : list[str]

    Call once per dataset and pass the result into plot_latent_feature_mapping.
    """
    print("    [+] Extracting kinetic features for all samples ...")
    feat_matrix, feat_names = _extract_all_kinetic_features_batch(X_batch, timestamps)

    print("    [+] Computing feature sensitivity profiles (finite-diff) ...")
    feat_sensitivity = _compute_feature_sensitivity_profiles_all(
        X_batch, timestamps, feat_names
    )
    return feat_matrix, feat_sensitivity, feat_names


def plot_latent_feature_mapping(
    artifacts, model_name,
    X_batch, timestamps,
    feat_matrix, feat_sensitivity, feat_names,
    mean_curve, std_curve,
    dataset_name, base_save_path,
    TOP_K=10,
    w_spearman=0.35, w_mi=0.25, w_cosine=0.40,
):
    """
    For each of the TOP_K most-important latent dimensions, draw one row:
      LEFT  – mini curve panel: mean input curve + ±1σ + latent saliency overlay
      RIGHT – horizontal score bar: combined mapping score vs every kinetic feature

    Parameters
    ----------
    artifacts        : dict from extract_xai_artifacts
    model_name       : str
    X_batch          : (Batch, T, 1) raw curves
    timestamps       : 1-D array length T
    feat_matrix      : (Batch, n_feats)  from compute_kinetic_feature_cache
    feat_sensitivity : (n_feats, T)      from compute_kinetic_feature_cache
    feat_names       : list[str]         from compute_kinetic_feature_cache
    mean_curve       : (T,)
    std_curve        : (T,)
    dataset_name     : str
    base_save_path   : Path or str
    TOP_K            : rows to show (default 10)
    w_*              : score weights, must sum to 1
    """
    from sklearn.feature_selection import mutual_info_regression

    art = artifacts.get(model_name)
    if not art:
        return

    save_path = str(base_save_path).replace('.png', f'_latent_mapping_{model_name}.png')

    t = np.asarray(timestamps, dtype=float)
    T = len(mean_curve)
    if len(t) != T:
        t = np.linspace(0, T - 1, T)
    t_step  = t[1] - t[0] if T > 1 else 1
    t_start = t[0]  - t_step / 2
    t_end   = t[-1] + t_step / 2

    n_feats = len(feat_names)
    TOP_K   = min(TOP_K, len(art["curve_order"]))

    # ------------------------------------------------------------------
    # 1. Gather latent activation traces  (TOP_K, Batch)
    # ------------------------------------------------------------------
    z_np      = art["z_curve"]
    order     = art["curve_order"][:TOP_K]
    imp_shape = art["curve_imp_shape"]
    is_3d     = len(imp_shape) == 2

    z_traces = []
    for flat_idx in order:
        if is_3d:
            r, c = np.unravel_index(int(flat_idx), imp_shape)
            z_traces.append(z_np[:, r, c])
        else:
            z_traces.append(z_np[:, int(flat_idx)])
    z_traces = np.array(z_traces)   # (TOP_K, Batch)

    # ------------------------------------------------------------------
    # 2. Mean saliency profiles  (TOP_K, T)
    # ------------------------------------------------------------------
    raw_maps     = art["raw_saliency_curve"][:TOP_K]
    sal_profiles = np.array([np.mean(m, axis=0) for m in raw_maps])  # (TOP_K, T)
    for i in range(TOP_K):
        mx = sal_profiles[i].max()
        if mx > 0:
            sal_profiles[i] /= mx

    # ------------------------------------------------------------------
    # 3. Score matrices  (TOP_K, n_feats)
    # ------------------------------------------------------------------
    spearman_m = np.zeros((TOP_K, n_feats))
    mi_m       = np.zeros((TOP_K, n_feats))
    cosine_m   = np.zeros((TOP_K, n_feats))

    # Mask columns that are entirely NaN across the batch
    valid_col = np.array([
        np.sum(np.isfinite(feat_matrix[:, j])) >= 5
        for j in range(n_feats)
    ])

    for i in range(TOP_K):
        z_i = z_traces[i]
        for j in range(n_feats):
            if not valid_col[j]:
                continue
            f_j = feat_matrix[:, j]
            mask = np.isfinite(f_j)
            if mask.sum() < 5:
                continue

            corr, _ = scipy.stats.spearmanr(z_i[mask], f_j[mask])
            spearman_m[i, j] = 0.0 if np.isnan(corr) else abs(corr)

            try:
                mi = mutual_info_regression(
                    z_i[mask].reshape(-1, 1), f_j[mask], random_state=0
                )[0]
            except Exception:
                mi = 0.0
            mi_m[i, j] = mi

            cosine_m[i, j] = max(0.0, _cosine_sim(sal_profiles[i], feat_sensitivity[j]))

    mi_max = mi_m.max()
    if mi_max > 0:
        mi_m /= mi_max

    combined = w_spearman * spearman_m + w_mi * mi_m + w_cosine * cosine_m

    best_feat_idx = np.argmax(combined, axis=1)   # (TOP_K,)
    best_score    = combined[np.arange(TOP_K), best_feat_idx]

    # ------------------------------------------------------------------
    # 4. Build figure:  TOP_K rows × 2 columns
    #    col 0 (width 1): mini curve + saliency overlay
    #    col 1 (width 4): horizontal score bar across all features
    # ------------------------------------------------------------------
    row_h   = 1.6          # inches per row
    fig_h   = TOP_K * row_h + 1.8   # +title space
    fig_w   = 22
    bar_w_ratio = 5        # right panel is 5× wider than the mini curve

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    fig.suptitle(
        f"{dataset_name} | {model_name.upper()} — Latent → Feature Mapping\n"
        f"Score = {w_spearman:.0%}·|Spearman| + {w_mi:.0%}·MI + {w_cosine:.0%}·Cos(saliency, sensitivity)",
        fontsize=12, fontweight='bold', y=1.0
    )

    # GridSpec: TOP_K rows, 2 cols
    gs = fig.add_gridspec(
        TOP_K, 2,
        width_ratios=[1, bar_w_ratio],
        hspace=0.15,
        wspace=0.04,
        left=0.04, right=0.97,
        top=0.94, bottom=0.08
    )

    # Shared colour limits for the score bars (all rows use the same cmap)
    cmap_bar = plt.cm.YlOrRd

    # Colour-coded x-tick labels for the score bar based on feature group
    _GROUP_COLOURS = {
        'timing':     '#1a6faf',   # blue   – time-point features
        'shape':      '#c45c00',   # orange – curve shape / asymmetry
        'amplitude':  '#2e8b57',   # green  – amplitude / y-value
        'derivative': '#8b2ec4',   # purple – derivative-based
        'baseline':   '#6e6e6e',   # grey   – baseline / plateau
        'integral':   '#bf8000',   # gold   – area / integral
        'fit':        '#b03060',   # rose   – sigmoid fit params
        'other':      '#333333',
    }
    _FEAT_GROUP = {
        'xms': 'timing', 'xs': 'timing', 'xe': 'timing',
        'xp1': 'timing', 'xp2': 'timing',
        'Ct': 'timing',  'Ct_ori': 'timing',
        't10': 'timing', 't50': 'timing', 't90': 'timing',
        'rise_time_10_90': 'timing', 'rise_time_20_80': 'timing',
        'lag_time': 'timing', 'Cy0': 'timing', 'Cy0_ori': 'timing',
        'threshold_distance': 'shape', 'first_half_distance': 'shape',
        'second_half_distance': 'shape', 'distance_asymmetry_index': 'shape',
        'peak_shifting_distance': 'shape', 'area_asymmetry_index': 'shape',
        'peak_asymmetry_index': 'shape', 'accel_fwhm': 'shape',
        'distance_asymmetry_index': 'shape',
        'y_xms': 'amplitude', 'y_xs': 'amplitude', 'y_xe': 'amplitude',
        'y_xp1': 'amplitude', 'y_xp2': 'amplitude', 'amplitude': 'amplitude',
        'F_max': 'amplitude', 'F_max_ori': 'amplitude',
        'F0': 'amplitude', 'log_F0': 'amplitude',
        'baseline_mean': 'baseline', 'baseline_std': 'baseline',
        'baseline_slope': 'baseline', 'plateau_mean': 'baseline',
        'plateau_std': 'baseline', 'plateau_slope': 'baseline',
        'snr_peak': 'baseline', 'snr_xms': 'baseline',
        'dy_xms': 'derivative', 'dy_xp1': 'derivative', 'dy_xp2': 'derivative',
        'd2y_xp1': 'derivative', 'd2y_xp2': 'derivative',
        'Send': 'derivative', 'Send_abs': 'derivative',
        'Send_fit': 'derivative', 'Send_fit_abs': 'derivative',
        'max_accel': 'derivative', 'min_accel': 'derivative',
        'TH': 'derivative',
        'A1': 'integral', 'A2': 'integral',
        'auc': 'integral', 'auc_norm': 'integral',
        'Fm': 'fit', 'Fb': 'fit', 'Sc': 'fit', 'Cs': 'fit', 'As': 'fit',
        'fit_rmse': 'fit', 'fit_r2': 'fit', 'overshoot_index': 'fit',
    }
    tick_colours = [
        _GROUP_COLOURS.get(_FEAT_GROUP.get(fn, 'other'), '#333333')
        for fn in feat_names
    ]

    x_positions = np.arange(n_feats)

    for i in range(TOP_K):
        flat_idx   = order[i]
        sal_prof   = sal_profiles[i]        # (T,)
        scores_row = combined[i]            # (n_feats,)
        best_j     = best_feat_idx[i]
        score_best = best_score[i]

        # --- Left panel: mini curve + saliency ---
        ax_curve = fig.add_subplot(gs[i, 0])
        ax_curve.fill_between(t, mean_curve - std_curve, mean_curve + std_curve,
                               color='#b0b0b0', alpha=0.4)
        ax_curve.plot(t, mean_curve, color='black', lw=1.0)
        ax_curve.set_xlim(t_start, t_end)
        ax_curve.set_yticks([])
        ax_curve.tick_params(axis='x', labelsize=6)
        ax_curve.spines[['top', 'right']].set_visible(False)

        # Saliency overlay on twin axis
        ax_sal = ax_curve.twinx()
        ax_sal.fill_between(t, 0, sal_prof, color='#e84040', alpha=0.45, lw=0)
        ax_sal.plot(t, sal_prof, color='#e84040', lw=0.8)
        ax_sal.set_ylim(0, sal_prof.max() * 2.0 if sal_prof.max() > 0 else 1)
        ax_sal.set_yticks([])
        ax_sal.spines[['top', 'right']].set_visible(False)

        # Y-label: latent dim + rank
        ax_curve.set_ylabel(
            f"Rank {i+1}\n(dim {flat_idx})",
            fontsize=7, fontweight='bold', rotation=0,
            labelpad=38, va='center'
        )
        if i < TOP_K - 1:
            plt.setp(ax_curve.get_xticklabels(), visible=False)
        else:
            ax_curve.set_xlabel("Time", fontsize=7)

        # --- Right panel: horizontal score bar ---
        ax_bar = fig.add_subplot(gs[i, 1])

        # Draw each feature as a vertical bar coloured by score
        bar_colours = [cmap_bar(s) for s in scores_row]
        ax_bar.bar(x_positions, scores_row, color=bar_colours,
                   width=0.85, linewidth=0)

        # Highlight best assignment
        ax_bar.bar(best_j, scores_row[best_j], color=cmap_bar(scores_row[best_j]),
                   width=0.85, linewidth=1.5, edgecolor='#222222')
        ax_bar.text(
            best_j, scores_row[best_j] + 0.02,
            f"★ {feat_names[best_j]}\n({score_best:.2f})",
            ha='center', va='bottom', fontsize=6.5, fontweight='bold',
            color='#222222',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='#888888', alpha=0.85, linewidth=0.7)
        )

        ax_bar.set_xlim(-0.5, n_feats - 0.5)
        ax_bar.set_ylim(0, 1.15)
        ax_bar.set_yticks([0, 0.5, 1.0])
        ax_bar.tick_params(axis='y', labelsize=6)
        ax_bar.spines[['top', 'right']].set_visible(False)
        ax_bar.axhline(0.5, color='#aaaaaa', lw=0.5, linestyle='--')

        if i < TOP_K - 1:
            ax_bar.set_xticks([])
        else:
            # Only the bottom row gets x-tick labels
            ax_bar.set_xticks(x_positions)
            ax_bar.set_xticklabels(feat_names, rotation=60, ha='right',
                                    fontsize=6.5)
            for tick, col in zip(ax_bar.get_xticklabels(), tick_colours):
                tick.set_color(col)

    # Colour legend for feature groups
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=col, label=grp.capitalize())
        for grp, col in _GROUP_COLOURS.items()
        if grp != 'other'
    ]
    fig.legend(
        handles=legend_handles,
        title="Feature group", title_fontsize=8,
        fontsize=7, loc='lower right',
        bbox_to_anchor=(0.98, 0.0),
        ncol=4, framealpha=0.9
    )

    # Shared score-bar colour bar on the right edge
    sm = plt.cm.ScalarMappable(cmap=cmap_bar, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.975, 0.10, 0.008, 0.80])
    fig.colorbar(sm, cax=cbar_ax).set_label(
        "Combined Mapping Score", rotation=270, labelpad=12, fontsize=8
    )

    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"     [✓] Saved latent mapping: {save_path}")


# ====================================================================
# MODULE 7: PIPELINE ORCHESTRATOR
# ====================================================================
def run_interpretation_pipeline(exp_folder_path=config.DEFAULT_EXP_FOLDER, filter_key=None, normalize=None, force_rerun=False):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    set_global_determinism(0)

    exp_folder = Path(exp_folder_path)
    global_vis_dir = config.get_viz_dir(exp_folder, "model_interpretation")
    global_vis_dir.mkdir(parents=True, exist_ok=True)
    
    exp_paths = sorted([p for p in exp_folder.iterdir() if p.is_dir() and p.name not in ['.DS_Store', 'model_interpretation']])
    
    for exp_path in exp_paths:
        print(f"\n{'='*70}\n[*] PROCESSING DATASET: {exp_path.name}\n{'='*70}")
        model_dir = exp_path / "model_interpretation"
        
        data_dict = prepare_dataset(exp_path, filter_key)
        if not data_dict: continue
        print(f"  -> Data Loaded | X: {data_dict['X_full'].shape}, y: {data_dict['y_full'].shape}")

        print("  -> Loading Pre-Trained Models...")
        models = load_saved_models(model_dir, str(filter_key), data_dict['X_full'].shape[1])
        if not models:
            print(f"  [-] Skipping visualisations for {exp_path.name}: No compatible saved models found.")
            continue

        pca_path = global_vis_dir / f"01_latent_space_pca_{exp_path.name}.png"
        tsne_path = global_vis_dir / f"02_latent_space_tsne_{exp_path.name}.png"
        saliency_base = global_vis_dir / f"04_saliency_{exp_path.name}.png"
        expected_outputs = [pca_path, tsne_path] + [
            Path(str(saliency_base).replace('.png', f'_{model_name}.png')) for model_name in models
        ]
        if not force_rerun and all(p.exists() for p in expected_outputs):
            print(f"  [-] Skipping {exp_path.name}: outputs already exist (use --force_rerun to regenerate).")
            continue

        batch_size = min(512, len(data_dict['X_full']))
        rng = np.random.default_rng(42)
        idx = rng.choice(len(data_dict['X_full']), size=batch_size, replace=False)
        
        X_batch = data_dict['X_full'][idx]
        X_man_batch = data_dict['X_man_full'][idx]
        T_man_batch = data_dict['T_man_full'][idx] if data_dict['T_man_full'] is not None else None
        y_batch = data_dict['y_full'][idx]
        
        mean_curve = np.squeeze(np.mean(X_batch, axis=0))
        std_curve = np.squeeze(np.std(X_batch, axis=0))
        if mean_curve.ndim > 1: 
            mean_curve = mean_curve.mean(axis=-1)
            std_curve = std_curve.mean(axis=-1)

        print(f"  -> Generating Central XAI Artifacts...")
        artifacts = extract_xai_artifacts(models, X_batch, X_man_batch)

        print(f"  -> Generating Visualizations into {global_vis_dir.name}/ ...")
        
        # 1. Base Plots (Original Logic Unchanged)
        plot_latent_pca(models, data_dict["X_full"], data_dict["X_man_full"], data_dict["y_full"], data_dict["dataset_name"], global_vis_dir / f"01_latent_space_pca_{exp_path.name}.png")
        plot_latent_tsne(models, data_dict["X_full"], data_dict["X_man_full"], data_dict["y_full"], data_dict["dataset_name"], global_vis_dir / f"02_latent_space_tsne_{exp_path.name}.png")

        # 2. Advanced Heatmaps (Optimized)
        for model_name in models.keys():
            plot_latent_saliency_heatmap(artifacts, model_name, data_dict["timestamps"], data_dict["top_10_features"], mean_curve, std_curve, exp_path.name, global_vis_dir / f"04_saliency_{exp_path.name}.png", normalize=normalize)
            # plot_concept_alignment_matrix(artifacts, model_name, X_man_batch, data_dict["top_10_features"], exp_path.name, global_vis_dir / f"05_concept_{exp_path.name}.png")
            # plot_temporal_alignment_matrix(artifacts, model_name, T_man_batch, data_dict["top_10_features"], exp_path.name, global_vis_dir / f"06_temporal_{exp_path.name}.png")

        # 3. Latent → Feature mapping (new)
        # compute_kinetic_feature_cache runs extract_kinetic_parameters_original on
        # every sample and builds the finite-diff sensitivity profiles.  It is
        # dataset-level (independent of the model) so we compute it once and reuse.
        # print(f"  -> Computing kinetic feature cache for {exp_path.name} ...")
        # feat_matrix, feat_sensitivity, feat_names = compute_kinetic_feature_cache(
        #     X_batch, data_dict["timestamps"]
        # )
        # for model_name in models.keys():
        #     plot_latent_feature_mapping(
        #         artifacts, model_name,
        #         X_batch, data_dict["timestamps"],
        #         feat_matrix, feat_sensitivity, feat_names,
        #         mean_curve, std_curve,
        #         exp_path.name,
        #         global_vis_dir / f"07_latent_mapping_{exp_path.name}.png",
        #     )

        # print(f"  [✓] Processed {exp_path.name}")
        # tf.keras.backend.clear_session()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAI Visualization Pipeline for DDM Models")
    parser.add_argument("--exp_folder", type=str, default=config.DEFAULT_EXP_FOLDER, help="Path to experiment datasets")
    parser.add_argument("--normalize", action="store_true", help="Whether to normalize the saliency maps")
    parser.add_argument("--force_rerun", action="store_true", help="Rerun and overwrite outputs even if they already exist")

    args = parser.parse_args()
    exp_folder = Path(args.exp_folder)
    normalize = args.normalize

    print(exp_folder)
    run_interpretation_pipeline(exp_folder_path=exp_folder, filter_key=None, normalize=normalize, force_rerun=args.force_rerun)