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
from mpl_toolkits.axes_grid1 import make_axes_locatable 

sys.path.insert(0, "../outlier_detection")
import config
from model_utils import set_global_determinism


# ====================================================================
# MODULE 1: DATA & MODEL LOADING 
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
                    continue
                models[name] = model
            except Exception as e:
                print(f"     [!] Failed to load {name}: {str(e)}")
            
    return models


# ====================================================================
# MODULE 2: LATENT CLUSTERING
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

        scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=y, cmap='tab10', alpha=0.7, s=15, edgecolors='none')
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

        scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=y_batch, cmap='tab10', alpha=0.7, s=15, edgecolors='none')
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
# MODULE 3: CAUSAL (OUTPUT-IMPACT) SALIENCY MAPPING
# ====================================================================
def plot_latent_saliency_heatmap(models, X, X_man, timestamps, top_10_features, dataset_name, base_save_path):
    """Generates heatmaps ranked strictly by dY/dZ (Causal Output Impact)."""
    if not models: return
    
    batch_size = min(512, len(X))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=batch_size, replace=False)
    X_batch, X_man_batch = X[idx], X_man[idx]
    
    t = timestamps if len(timestamps) == X_batch.shape[1] else np.arange(X_batch.shape[1])
    t_step = t[1] - t[0] if len(t) > 1 else 1
    t_start, t_end = t[0] - (t_step / 2), t[-1] + (t_step / 2)
    
    mean_curve = np.squeeze(np.mean(X_batch, axis=0))
    std_curve = np.squeeze(np.std(X_batch, axis=0))
    if mean_curve.ndim > 1: 
        mean_curve = mean_curve.mean(axis=-1)
        std_curve = std_curve.mean(axis=-1)

    def normalize_heatmap(matrix, method='row'):
        if len(matrix) == 0: return matrix
        if method == 'row':
            mins = np.min(matrix, axis=1, keepdims=True)
            maxs = np.max(matrix, axis=1, keepdims=True)
            val_range = maxs - mins
            val_range[val_range == 0] = 1e-10 
            return (matrix - mins) / val_range
        return matrix

    for model_name, model in models.items():
        save_path = str(base_save_path).replace('.png', f'_{model_name}.png')
        is_lf = model_name.endswith('_lf')
        is_dual = model_name.endswith('_dual')
        
        x_tf_curve = tf.convert_to_tensor(X_batch, dtype=tf.float32)
        x_tf_man = tf.convert_to_tensor(X_man_batch, dtype=tf.float32)

        # ------------------------------------------------------------------
        # LATE FUSION MODELS
        # ------------------------------------------------------------------
        if is_lf:
            curve_input = model.get_layer("curve_input")
            man_input = model.get_layer("features_input")
            concat_layer = next(l for l in model.layers if isinstance(l, tf.keras.layers.Concatenate))
            spatial_layer = next((l for l in model.layers if isinstance(l, (tf.keras.layers.Flatten, tf.keras.layers.GlobalAveragePooling1D))), None)
            
            # 1. Extractors (Input -> Latent)
            extractor_curve = tf.keras.Model(inputs=curve_input.input, outputs=spatial_layer.output if spatial_layer else concat_layer.input[0])
            extractor_man = tf.keras.Model(inputs=man_input.input, outputs=concat_layer.input[1])
            
            # 2. Classification Head (Latent -> Output)
            head_model = tf.keras.Model(inputs=[extractor_curve.output, extractor_man.output], outputs=model.output)
            
            # 3. Calculate Master Saliency for reference
            with tf.GradientTape() as tape_m:
                tape_m.watch(x_tf_curve)
                target_master_full = tf.reduce_max(model([x_tf_curve, x_tf_man]), axis=1)
            master_saliency_c = np.mean(np.abs(tape_m.gradient(target_master_full, x_tf_curve).numpy()), axis=(0, 2))
            
            # 4. Calculate dy/dz (Causal Importance)
            z_curve = extractor_curve(x_tf_curve)
            z_man = extractor_man(x_tf_man)
            
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(z_curve)
                tape.watch(z_man)
                target_master = tf.reduce_max(head_model([z_curve, z_man]), axis=1)
                
            dy_dz_curve = tape.gradient(target_master, z_curve).numpy()
            dy_dz_man = tape.gradient(target_master, z_man).numpy()
            del tape
            
            # 5. Rank and Cap
            importance_c = np.mean(np.abs(dy_dz_curve), axis=0)
            importance_m = np.mean(np.abs(dy_dz_man), axis=0)
            
            curve_order = np.argsort(importance_c)[::-1][:min(100, len(importance_c))]
            man_order = np.argsort(importance_m)[::-1]
            
            # 6. Calculate spatial heatmaps ONLY for Top 100
            heatmap_curve = []
            for dim in curve_order:
                with tf.GradientTape() as tape:
                    tape.watch(x_tf_curve)
                    target_dim = tf.reduce_mean(extractor_curve(x_tf_curve)[:, dim])
                grad_c = tape.gradient(target_dim, x_tf_curve).numpy()
                heatmap_curve.append(np.mean(np.abs(grad_c), axis=(0, 2)))
                
            heatmap_man = []
            for dim in man_order:
                with tf.GradientTape() as tape:
                    tape.watch(x_tf_man)
                    target_dim = tf.reduce_mean(extractor_man(x_tf_man)[:, dim])
                grad_m = tape.gradient(target_dim, x_tf_man).numpy()
                heatmap_man.append(np.mean(np.abs(grad_m), axis=0))
                
            heatmap_curve = normalize_heatmap(np.array(heatmap_curve), method='row')
            heatmap_man = normalize_heatmap(np.array(heatmap_man), method='row')

            # --- PLOTTING ---
            fig, (ax_curve, ax_heat_c, ax_heat_m) = plt.subplots(3, 1, figsize=(9, 12), gridspec_kw={'height_ratios': [1, 2.5, 2.5]})
            ax_heat_c.sharex(ax_curve) 
            
            ax_curve.plot(t, mean_curve, color='black', lw=1.5, label='Mean Input')
            ax_curve.fill_between(t, mean_curve - std_curve, mean_curve + std_curve, color='gray', alpha=0.3)
            ax_curve_twin = ax_curve.twinx()
            ax_curve_twin.plot(t, master_saliency_c, color='red', lw=1.5, linestyle='--', alpha=0.8, label='Master Output Saliency')
            
            ax_curve.set_title(f"{model_name.upper()} - Final Output Decision vs Mean Signal", fontsize=12, fontweight='bold', pad=15)
            ax_curve.set_xlim(t_start, t_end)
            ax_curve.grid(True, color='grey', alpha=0.3, linestyle='--')
            ax_curve_twin.tick_params(axis='y', labelcolor='red')
            plt.setp(ax_curve.get_xticklabels(), visible=False)
            
            im_c = ax_heat_c.imshow(heatmap_curve, aspect='auto', cmap='inferno', vmin=0, vmax=1, extent=[t_start, t_end, len(curve_order), 0], interpolation='nearest')
            ax_heat_c.set_title(f"Curve Branch Ranked by dY/dZ Impact (Top {len(curve_order)} Dims)", fontsize=11, fontweight='bold', pad=10)
            ax_heat_c.set_ylabel("Rank (1 = Highest Output Impact)", fontsize=10)
            ax_heat_c.grid(True, color='grey', alpha=0.3, linestyle='--')

            im_m = ax_heat_m.imshow(heatmap_man, aspect='auto', cmap='inferno', vmin=0, vmax=1, extent=[0, 10, len(man_order), 0], interpolation='nearest')
            ax_heat_m.set_title("Manual Features Ranked by dY/dZ Impact", fontsize=11, fontweight='bold', pad=10)
            ax_heat_m.set_ylabel("Rank", fontsize=10)
            ax_heat_m.set_xticks(np.arange(10) + 0.5)
            ax_heat_m.set_xticklabels(top_10_features, rotation=45, ha='right', fontsize=9, fontweight='bold')
            ax_heat_m.grid(True, color='grey', alpha=0.3, linestyle='--')

            cax_curve = make_axes_locatable(ax_curve).append_axes("right", size="3%", pad=0.1)
            cax_curve.axis('off')
            cax_c = make_axes_locatable(ax_heat_c).append_axes("right", size="3%", pad=0.1)
            fig.colorbar(im_c, cax=cax_c).set_label("Normalized Attrib", rotation=270, labelpad=15)
            cax_m = make_axes_locatable(ax_heat_m).append_axes("right", size="3%", pad=0.1)
            fig.colorbar(im_m, cax=cax_m).set_label("Normalized Attrib", rotation=270, labelpad=15)
            
            fig.align_ylabels([ax_curve, ax_heat_c, ax_heat_m]) 
            fig.suptitle(f"{dataset_name} | {model_name.upper()} Causal Impact Saliency", fontsize=14, fontweight='bold', y=0.98)
            plt.subplots_adjust(hspace=0.25)
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)

        # ------------------------------------------------------------------
        # DUAL MODELS
        # ------------------------------------------------------------------
        elif is_dual:
            concat_layer = next(l for l in model.layers if isinstance(l, tf.keras.layers.Concatenate))
            spatial_layer_cnn = next((l for l in model.layers if isinstance(l, (tf.keras.layers.Flatten, tf.keras.layers.GlobalAveragePooling1D))), None)
            
            extractor_cnn = tf.keras.Model(inputs=model.input, outputs=spatial_layer_cnn.output if spatial_layer_cnn else concat_layer.input[0])
            extractor_rnn = tf.keras.Model(inputs=model.input, outputs=concat_layer.input[1])
            
            head_model = tf.keras.Model(inputs=[extractor_cnn.output, extractor_rnn.output], outputs=model.output)
            
            with tf.GradientTape() as tape_m:
                tape_m.watch(x_tf_curve)
                target_master_full = tf.reduce_max(model(x_tf_curve), axis=1)
            master_saliency = np.mean(np.abs(tape_m.gradient(target_master_full, x_tf_curve).numpy()), axis=(0, 2))
            
            z_cnn = extractor_cnn(x_tf_curve)
            z_rnn = extractor_rnn(x_tf_curve)
            
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(z_cnn)
                tape.watch(z_rnn)
                target_master = tf.reduce_max(head_model([z_cnn, z_rnn]), axis=1)
                
            dy_dz_cnn = tape.gradient(target_master, z_cnn).numpy()
            dy_dz_rnn = tape.gradient(target_master, z_rnn).numpy()
            del tape
            
            importance_cnn = np.mean(np.abs(dy_dz_cnn), axis=0)
            importance_rnn = np.mean(np.abs(dy_dz_rnn), axis=0)
            
            cnn_order = np.argsort(importance_cnn)[::-1][:min(100, len(importance_cnn))]
            rnn_order = np.argsort(importance_rnn)[::-1][:min(100, len(importance_rnn))]

            heatmap_cnn = []
            for dim in cnn_order:
                with tf.GradientTape() as tape:
                    tape.watch(x_tf_curve)
                    target = tf.reduce_mean(extractor_cnn(x_tf_curve)[:, dim])
                heatmap_cnn.append(np.mean(np.abs(tape.gradient(target, x_tf_curve).numpy()), axis=(0, 2)))
                
            heatmap_rnn = []
            for dim in rnn_order:
                with tf.GradientTape() as tape:
                    tape.watch(x_tf_curve)
                    target = tf.reduce_mean(extractor_rnn(x_tf_curve)[:, dim])
                heatmap_rnn.append(np.mean(np.abs(tape.gradient(target, x_tf_curve).numpy()), axis=(0, 2)))

            heatmap_cnn = normalize_heatmap(np.array(heatmap_cnn), method='row')
            heatmap_rnn = normalize_heatmap(np.array(heatmap_rnn), method='row')

            fig, (ax_curve, ax_heat_c, ax_heat_r) = plt.subplots(3, 1, figsize=(9, 12), gridspec_kw={'height_ratios': [1, 2.5, 2.5]}, sharex=True)
            
            ax_curve.plot(t, mean_curve, color='black', lw=1.5, label='Mean Input')
            ax_curve.fill_between(t, mean_curve - std_curve, mean_curve + std_curve, color='gray', alpha=0.3)
            ax_curve_twin = ax_curve.twinx()
            ax_curve_twin.plot(t, master_saliency, color='red', lw=1.5, linestyle='--', alpha=0.8, label='Master Output Saliency')
            
            ax_curve.set_title(f"{model_name.upper()} - Final Output Decision vs Mean Signal", fontsize=12, fontweight='bold', pad=15)
            ax_curve.set_xlim(t_start, t_end)
            ax_curve.grid(True, color='grey', alpha=0.3, linestyle='--')
            ax_curve_twin.tick_params(axis='y', labelcolor='red')
            plt.setp(ax_curve.get_xticklabels(), visible=False)
            
            im_c = ax_heat_c.imshow(heatmap_cnn, aspect='auto', cmap='inferno', vmin=0, vmax=1, extent=[t_start, t_end, len(cnn_order), 0], interpolation='nearest')
            ax_heat_c.set_title(f"CNN Branch Ranked by dY/dZ Impact (Top {len(cnn_order)} Dims)", fontsize=11, fontweight='bold', pad=10)
            ax_heat_c.set_ylabel("Rank (1 = Highest Impact)", fontsize=10)
            ax_heat_c.grid(True, color='grey', alpha=0.3, linestyle='--')

            im_r = ax_heat_r.imshow(heatmap_rnn, aspect='auto', cmap='inferno', vmin=0, vmax=1, extent=[t_start, t_end, len(rnn_order), 0], interpolation='nearest')
            branch_name = "Transformer" if "trans" in model_name else "GRU"
            ax_heat_r.set_title(f"{branch_name} Branch Ranked by dY/dZ Impact (Top {len(rnn_order)} Dims)", fontsize=11, fontweight='bold', pad=10)
            ax_heat_r.set_ylabel("Rank", fontsize=10)
            ax_heat_r.set_xlabel("Time", fontsize=10, fontweight='bold')
            ax_heat_r.grid(True, color='grey', alpha=0.3, linestyle='--')

            cax_curve = make_axes_locatable(ax_curve).append_axes("right", size="3%", pad=0.1)
            cax_curve.axis('off')
            cax_c = make_axes_locatable(ax_heat_c).append_axes("right", size="3%", pad=0.1)
            fig.colorbar(im_c, cax=cax_c).set_label("Normalized Attrib", rotation=270, labelpad=15)
            cax_r = make_axes_locatable(ax_heat_r).append_axes("right", size="3%", pad=0.1)
            fig.colorbar(im_r, cax=cax_r).set_label("Normalized Attrib", rotation=270, labelpad=15)
            
            fig.align_ylabels([ax_curve, ax_heat_c, ax_heat_r]) 
            fig.suptitle(f"{dataset_name} | {model_name.upper()} Causal Impact Saliency", fontsize=14, fontweight='bold', y=0.98)
            plt.subplots_adjust(hspace=0.25)
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)

        # ------------------------------------------------------------------
        # BASE MODELS
        # ------------------------------------------------------------------
        else:
            spatial_layer = next((l for l in model.layers if isinstance(l, (tf.keras.layers.Flatten, tf.keras.layers.GlobalAveragePooling1D))), None)
            
            extractor = tf.keras.Model(inputs=model.input, outputs=spatial_layer.output if spatial_layer else model.layers[-2].output)
            head_model = tf.keras.Model(inputs=extractor.output, outputs=model.output)
            
            with tf.GradientTape() as tape_m:
                tape_m.watch(x_tf_curve)
                target_master_full = tf.reduce_max(model(x_tf_curve), axis=1)
            master_saliency = np.mean(np.abs(tape_m.gradient(target_master_full, x_tf_curve).numpy()), axis=(0, 2))
            
            z_curve = extractor(x_tf_curve)
            
            with tf.GradientTape() as tape:
                tape.watch(z_curve)
                target_master = tf.reduce_max(head_model(z_curve), axis=1)
                
            dy_dz = tape.gradient(target_master, z_curve).numpy()
            importance = np.mean(np.abs(dy_dz), axis=0)
            
            curve_order = np.argsort(importance)[::-1][:min(100, len(importance))]
            
            heatmap_curve = []
            for dim in curve_order:
                with tf.GradientTape() as tape:
                    tape.watch(x_tf_curve)
                    target = tf.reduce_mean(extractor(x_tf_curve)[:, dim])
                heatmap_curve.append(np.mean(np.abs(tape.gradient(target, x_tf_curve).numpy()), axis=(0, 2)))
                
            heatmap_curve = normalize_heatmap(np.array(heatmap_curve), method='row')
            
            fig, (ax_curve, ax_heat) = plt.subplots(2, 1, figsize=(9, 8), gridspec_kw={'height_ratios': [1, 3]}, sharex=True)
            
            ax_curve.plot(t, mean_curve, color='black', lw=1.5, label='Mean Input')
            ax_curve.fill_between(t, mean_curve - std_curve, mean_curve + std_curve, color='gray', alpha=0.3)
            ax_curve_twin = ax_curve.twinx()
            ax_curve_twin.plot(t, master_saliency, color='red', lw=1.5, linestyle='--', alpha=0.8, label='Master Output Saliency')
            
            ax_curve.set_title(f"{model_name.upper()} - Final Output Decision vs Mean Signal", fontsize=12, fontweight='bold', pad=15)
            ax_curve.set_xlim(t_start, t_end)
            ax_curve.grid(True, color='grey', alpha=0.3, linestyle='--')
            ax_curve_twin.tick_params(axis='y', labelcolor='red')
            plt.setp(ax_curve.get_xticklabels(), visible=False)
            
            im = ax_heat.imshow(heatmap_curve, aspect='auto', cmap='inferno', vmin=0, vmax=1, extent=[t_start, t_end, len(curve_order), 0], interpolation='nearest')
            
            layer_name_str = "Spatial/Flatten Dims" if spatial_layer else "Latent Dims"
            ax_heat.set_title(f"Saliency Ranked by dY/dZ Impact (Top {len(curve_order)} {layer_name_str})", fontsize=11, fontweight='bold', pad=10)
            ax_heat.set_ylabel("Rank (1 = Highest Impact)", fontsize=10)
            ax_heat.set_xlabel("Time", fontsize=10, fontweight='bold')
            ax_heat.grid(True, color='grey', alpha=0.3, linestyle='--')
            
            cax_curve = make_axes_locatable(ax_curve).append_axes("right", size="3%", pad=0.1)
            cax_curve.axis('off')
            cax_heat = make_axes_locatable(ax_heat).append_axes("right", size="3%", pad=0.1)
            fig.colorbar(im, cax=cax_heat).set_label("Normalized Attrib", rotation=270, labelpad=15)
            
            fig.align_ylabels([ax_curve, ax_heat]) 
            fig.suptitle(f"{dataset_name} | {model_name.upper()} Causal Impact Saliency", fontsize=14, fontweight='bold', y=0.98)
            plt.subplots_adjust(hspace=0.15)
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)


# ====================================================================
# MODULE 4: PIPELINE ORCHESTRATOR
# ====================================================================
def run_interpretation_pipeline(exp_folder_path=config.DEFAULT_EXP_FOLDER, filter_key=None):
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