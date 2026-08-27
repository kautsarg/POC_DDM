import numpy as np
import scipy.stats
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.feature_selection import mutual_info_regression

import config


# ============================================================
# Dual-branch latent saliency
# ============================================================

def find_flatten_layer(model):
    flatten_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Flatten)]
    return flatten_layers[-1] if flatten_layers else None


def find_bidirectional_recurrent_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Bidirectional) and isinstance(
            layer.forward_layer, (tf.keras.layers.GRU, tf.keras.layers.LSTM, tf.keras.layers.SimpleRNN)
        ):
            return layer
    return None


def find_weighted_recon_layer(model):
    return next((l for l in model.layers if type(l).__name__ == '_WeightedRecon'), None)


def _is_neighbor_stack_model(model):
    return (not isinstance(model.input, (list, tuple))
            and hasattr(model.input, 'name')
            and 'neighbor_stack' in model.input.name)


def normalize_heatmap(matrix, method='row'):
    if len(matrix) == 0:
        return matrix
    if method == 'row':
        mins = np.min(matrix, axis=1, keepdims=True)
        maxs = np.max(matrix, axis=1, keepdims=True)
        val_range = maxs - mins
        val_range[val_range == 0] = 1e-10
        return (matrix - mins) / val_range
    return matrix


def _rank_latents(dy_dz, n_dims):
    importance = np.mean(np.abs(dy_dz), axis=0)
    return np.argsort(importance)[::-1][:min(n_dims, len(importance))]


def _latent_saliency_batch(extractor, x_tf, order):
    saliency_maps = []
    for dim in order:
        with tf.GradientTape() as tape:
            tape.watch(x_tf)
            z = extractor(x_tf)
            target = z[:, int(dim)]
        grad = tape.gradient(target, x_tf).numpy()
        if grad.ndim == 3 and grad.shape[-1] == 1:
            grad = grad.squeeze(axis=-1)
        saliency_maps.append(np.abs(grad))
    return saliency_maps


_NS_CHUNK = 64


def _extract_dual_saliency_neighbor_stack(model, X_batch, n_dims):
    recon_layer = find_weighted_recon_layer(model)
    flatten_layer = find_flatten_layer(model)
    recurrent_layer = find_bidirectional_recurrent_layer(model)
    if recon_layer is None or flatten_layer is None or recurrent_layer is None:
        raise ValueError('Model does not look like create_cnn_gru_dual_attn_recon.')

    k_plus_1 = model.input.shape[1]
    x_center = tf.constant(X_batch[:, :, 0], dtype=tf.float32)
    x_stack = tf.stack([x_center] * k_plus_1, axis=1)

    recon_t = recon_layer.output
    recon_model = tf.keras.Model(model.input, recon_t)
    m_outs = model.output if isinstance(model.output, (list, tuple)) else [model.output]
    combined_model = tf.keras.Model(recon_t, [flatten_layer.output, recurrent_layer.output] + list(m_outs))

    recon_parts, z_cnn_parts, z_rnn_parts = [], [], []
    dy_dz_cnn_parts, dy_dz_rnn_parts = [], []
    for start in range(0, x_stack.shape[0], _NS_CHUNK):
        x_chunk = x_stack[start:start + _NS_CHUNK]
        recon_chunk = recon_model(x_chunk)
        recon_parts.append(recon_chunk)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(recon_chunk)
            z_cnn_c, z_rnn_c, *rest = combined_model(recon_chunk)
            target = tf.reduce_max(rest[0], axis=1)
        dy_dz_cnn_parts.append(tape.gradient(target, z_cnn_c).numpy())
        dy_dz_rnn_parts.append(tape.gradient(target, z_rnn_c).numpy())
        z_cnn_parts.append(z_cnn_c.numpy())
        z_rnn_parts.append(z_rnn_c.numpy())
        del tape

    recon_val = tf.concat(recon_parts, axis=0)
    z_cnn = np.concatenate(z_cnn_parts, axis=0)
    z_rnn = np.concatenate(z_rnn_parts, axis=0)
    dy_dz_cnn = np.concatenate(dy_dz_cnn_parts, axis=0)
    dy_dz_rnn = np.concatenate(dy_dz_rnn_parts, axis=0)

    cnn_order = _rank_latents(dy_dz_cnn, n_dims)
    rnn_order = _rank_latents(dy_dz_rnn, n_dims)

    extractor_cnn = tf.keras.Model(recon_t, flatten_layer.output)
    extractor_rnn = tf.keras.Model(recon_t, recurrent_layer.output)

    def _sal(extractor, order):
        maps = []
        for dim in order:
            chunks = []
            for start in range(0, recon_val.shape[0], _NS_CHUNK):
                r_chunk = recon_val[start:start + _NS_CHUNK]
                with tf.GradientTape() as tape:
                    tape.watch(r_chunk)
                    z = extractor(r_chunk)
                    target = z[:, int(dim)]
                grad = tape.gradient(target, r_chunk).numpy()
                chunks.append(np.abs(grad)[:, 0, :])
            maps.append(np.concatenate(chunks, axis=0))
        return maps

    return {
        'raw_saliency_curve': _sal(extractor_cnn, cnn_order),
        'raw_saliency_rnn': _sal(extractor_rnn, rnn_order),
        'is_type': 'dual',
        'z_curve': z_cnn, 'curve_order': cnn_order, 'curve_imp_shape': dy_dz_cnn.shape[1:],
        'z_rnn': z_rnn, 'rnn_order': rnn_order, 'rnn_imp_shape': dy_dz_rnn.shape[1:],
    }


def extract_dual_saliency(model, X_batch, n_dims=100):
    if _is_neighbor_stack_model(model):
        return _extract_dual_saliency_neighbor_stack(model, X_batch, n_dims)

    flatten_layer = find_flatten_layer(model)
    recurrent_layer = find_bidirectional_recurrent_layer(model)
    if flatten_layer is None or recurrent_layer is None:
        raise ValueError('Model does not look like create_cnn_gru_dual (missing Flatten or Bidirectional GRU layer).')

    extractor_cnn = tf.keras.Model(inputs=model.input, outputs=flatten_layer.output)
    extractor_rnn = tf.keras.Model(inputs=model.input, outputs=recurrent_layer.output)
    head_model = tf.keras.Model(inputs=[extractor_cnn.output, extractor_rnn.output], outputs=model.output)

    x_tf = tf.convert_to_tensor(X_batch, dtype=tf.float32)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x_tf)
        z_cnn = extractor_cnn(x_tf)
        z_rnn = extractor_rnn(x_tf)
        tape.watch(z_cnn)
        tape.watch(z_rnn)
        target_master = tf.reduce_max(head_model([z_cnn, z_rnn]), axis=1)

    dy_dz_cnn = tape.gradient(target_master, z_cnn).numpy()
    dy_dz_rnn = tape.gradient(target_master, z_rnn).numpy()
    del tape

    cnn_order = _rank_latents(dy_dz_cnn, n_dims)
    rnn_order = _rank_latents(dy_dz_rnn, n_dims)

    return {
        'raw_saliency_curve': _latent_saliency_batch(extractor_cnn, x_tf, cnn_order),
        'raw_saliency_rnn': _latent_saliency_batch(extractor_rnn, x_tf, rnn_order),
        'is_type': 'dual',
        'z_curve': z_cnn.numpy(), 'curve_order': cnn_order, 'curve_imp_shape': dy_dz_cnn.shape[1:],
        'z_rnn': z_rnn.numpy(), 'rnn_order': rnn_order, 'rnn_imp_shape': dy_dz_rnn.shape[1:],
    }


def plot_per_label_saliency_heatmap(
    art, X_batch, y_true, y_pred, class_idx, class_name, timestamps,
    n_dims=25, save_path=None, show_std=True,
):
    """
    View A (normalised) only: samples where y_true==class_idx AND y_pred==class_idx.
    Curve (mean +/- std) -> CNN latent heatmap (row-normalised) -> GRU latent
    heatmap (row-normalised) -> 1D summary (collapsed row-normalised heatmaps).
    """
    T = len(timestamps)
    t_step = (timestamps[-1] - timestamps[0]) / (T - 1) if T > 1 else 1.0
    t_start = timestamps[0] - t_step / 2
    t_end = timestamps[-1] + t_step / 2

    mask = (y_true == class_idx) & (y_pred == class_idx)
    n_correct = int(mask.sum())
    n_total = int((y_true == class_idx).sum())
    if n_correct == 0:
        print(f'    [skip] {class_name}: no correctly-classified samples in batch')
        return False

    n_cnn = min(n_dims, len(art['raw_saliency_curve']))
    n_rnn = min(n_dims, len(art['raw_saliency_rnn']))
    hm_cnn = normalize_heatmap(np.array([m[mask].mean(0) for m in art['raw_saliency_curve'][:n_cnn]]), method='row')
    hm_rnn = normalize_heatmap(np.array([m[mask].mean(0) for m in art['raw_saliency_rnn'][:n_rnn]]), method='row')

    def _norm1d(x):
        return (x - x.min()) / (np.ptp(x) + 1e-12)

    sal_1d = _norm1d(hm_cnn.mean(0) + hm_rnn.mean(0))

    mean_c = X_batch[mask, :, 0].mean(0)
    std_c = X_batch[mask, :, 0].std(0)

    fig = plt.figure(figsize=(9, 8.5), facecolor='white')
    gs = gridspec.GridSpec(4, 1, height_ratios=[1.6, 2.6, 2.6, 1.2], hspace=0.32, figure=fig)

    ax0 = fig.add_subplot(gs[0])
    ax0.plot(timestamps, mean_c, color='#16213E', lw=1.7)
    if show_std:
        ax0.fill_between(timestamps, mean_c - std_c, mean_c + std_c, color='gray', alpha=0.22, label=f'\xb11σ  N={n_correct}')
        ax0.legend(fontsize=8, loc='upper left', framealpha=0.8)
    ax0.set_title(f'{class_name} Amplification Curve', fontsize=10, fontweight='bold')
    ax0.set_xlim(t_start, t_end)
    ax0.grid(True, color='grey', alpha=0.22, ls='--')
    ax0.tick_params(labelsize=7)
    plt.setp(ax0.get_xticklabels(), visible=False)

    def _heatmap(ax_idx, hm, title, ref_ax):
        ax = fig.add_subplot(gs[ax_idx], sharex=ref_ax)
        im = ax.imshow(hm, aspect='auto', cmap='inferno', vmin=0, vmax=1,
                        extent=[t_start, t_end, len(hm), 0], interpolation='nearest')
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_ylabel('Latent rank', fontsize=8)
        ax.grid(True, color='grey', alpha=0.17, ls='--')
        ax.tick_params(labelsize=7)
        cax = make_axes_locatable(ax).append_axes('right', size='3%', pad=0.04)
        fig.colorbar(im, cax=cax).set_label('|∂Z/∂X| (row-norm)', rotation=270, labelpad=10, fontsize=7.5)
        plt.setp(ax.get_xticklabels(), visible=False)
        return ax

    _heatmap(1, hm_cnn, f'CNN latent (top {n_cnn})', ax0)
    _heatmap(2, hm_rnn, f'GRU latent (top {n_rnn})', ax0)

    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    ax3b = ax3.twinx()
    ax3.plot(timestamps, mean_c, color='#16213E', lw=1.2, alpha=0.75)
    ax3b.fill_between(timestamps, 0, sal_1d, color='#1565C0', alpha=0.38)
    ax3b.plot(timestamps, sal_1d, color='#1565C0', lw=1.0)
    ax3b.set_ylim(0, 1.1)
    ax3b.set_yticks([])
    ax3.set_xlabel('Cycle', fontsize=9)
    ax3.set_ylabel('Fluor.', fontsize=8, color='#16213E')
    ax3.set_title('Aggregated Latent Saliency (CNN + GRU)', fontsize=9, fontweight='bold')
    ax3.grid(True, color='grey', alpha=0.20, ls='--')
    ax3.tick_params(labelsize=7)

    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return True


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
    from sigmoid_fitting import extract_kinetic_parameters_original
    from threadpoolctl import threadpool_limits

    X_sq = np.squeeze(X_batch, axis=-1)
    t = np.asarray(timestamps, dtype=float)
    N = X_sq.shape[0]
    if len(t) != X_sq.shape[1]:
        t = np.linspace(0, X_sq.shape[1] - 1, X_sq.shape[1])

    rows = []
    with threadpool_limits(limits=1):
        for s in range(N):
            try:
                d = extract_kinetic_parameters_original(t, X_sq[s])
            except Exception:
                d = {}
            rows.append([float(d.get(k, np.nan)) for k in _ALL_KINETIC_SCALAR_KEYS])

    return np.array(rows, dtype=np.float32), list(_ALL_KINETIC_SCALAR_KEYS)


def _compute_feature_sensitivity_profiles_all(X_batch, timestamps, feat_names,
                                               delta_frac=0.05, n_sub=48):
    from sigmoid_fitting import extract_kinetic_parameters_original
    from threadpoolctl import threadpool_limits

    X_sq = np.squeeze(X_batch, axis=-1)
    t = np.asarray(timestamps, dtype=float)
    if len(t) != X_sq.shape[1]:
        t = np.linspace(0, X_sq.shape[1] - 1, X_sq.shape[1])

    rng = np.random.default_rng(7)
    n_sub = min(n_sub, X_sq.shape[0])
    sub = rng.choice(X_sq.shape[0], size=n_sub, replace=False)
    X_sub = X_sq[sub]

    n_feats = len(feat_names)
    T_len = X_sq.shape[1]
    delta = delta_frac * (X_sub.max() - X_sub.min() + 1e-8)
    sensitivity = np.zeros((n_feats, T_len), dtype=np.float32)

    with threadpool_limits(limits=1):
        for s in range(n_sub):
            try:
                base = extract_kinetic_parameters_original(t, X_sub[s])
            except Exception:
                base = {}
            base_vals = [float(base.get(fn, np.nan)) for fn in feat_names]

            for t_idx in range(T_len):
                x_pert = X_sub[s].copy()
                x_pert[t_idx] += delta
                try:
                    pert = extract_kinetic_parameters_original(t, x_pert)
                except Exception:
                    pert = {}
                for fi, fn in enumerate(feat_names):
                    bv = base_vals[fi]
                    pv = float(pert.get(fn, np.nan))
                    if np.isfinite(bv) and np.isfinite(pv):
                        sensitivity[fi, t_idx] += abs(pv - bv) / delta

    sensitivity /= n_sub

    for fi in range(n_feats):
        mx = sensitivity[fi].max()
        sensitivity[fi] = sensitivity[fi] / mx if mx > 0 else np.ones(T_len) / T_len

    return sensitivity


def _cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0


def compute_kinetic_feature_cache(X_batch, timestamps):
    print("    [+] Extracting kinetic features for all samples ...")
    feat_matrix, feat_names = _extract_all_kinetic_features_batch(X_batch, timestamps)

    print("    [+] Computing feature sensitivity profiles (finite-diff) ...")
    feat_sensitivity = _compute_feature_sensitivity_profiles_all(X_batch, timestamps, feat_names)
    return feat_matrix, feat_sensitivity, feat_names


def plot_latent_feature_mapping(
    art, model_name, timestamps,
    feat_matrix, feat_sensitivity, feat_names,
    mean_curve, dataset_name, save_path,
    TOP_N=5, w_spearman=0.35, w_mi=0.25, w_cosine=0.40,
    sample_mask=None,
):
    """sample_mask: optional boolean array over the batch -- restricts z_traces,
    saliency profiles, and feat_matrix rows to this subset. feat_sensitivity is
    left as-is: it's a property of the feature extractor, computed once per dataset."""
    if sample_mask is not None:
        if int(np.sum(sample_mask)) == 0:
            print(f'    [skip] {dataset_name}/{model_name}: no samples in mask')
            return
        feat_matrix = feat_matrix[sample_mask]

    t = np.asarray(timestamps, dtype=float)
    T = len(mean_curve)
    if len(t) != T:
        t = np.linspace(0, T - 1, T)

    main_features = [f for f in config.XAI_KINETIC_FEATURE_GROUP.keys() if f in feat_names]
    main_idx = [feat_names.index(f) for f in main_features]
    feat_matrix = feat_matrix[:, main_idx]
    feat_sensitivity = feat_sensitivity[main_idx]
    feat_names = main_features
    n_feats = len(feat_names)

    if art["is_type"] == "dual":
        rnn_label = "Transformer" if "trans" in model_name else "GRU"
        branch_sections = [
            ("CNN",     art["curve_order"], art["curve_imp_shape"], art["z_curve"], art["raw_saliency_curve"]),
            (rnn_label, art["rnn_order"],   art["rnn_imp_shape"],   art["z_rnn"],   art["raw_saliency_rnn"]),
        ]
    else:
        branch_sections = [
            (None, art["curve_order"], art["curve_imp_shape"], art["z_curve"], art["raw_saliency_curve"]),
        ]

    valid_col = np.array([np.sum(np.isfinite(feat_matrix[:, j])) >= 5 for j in range(n_feats)])

    sections = []
    for label, order, imp_shape, z_np, raw_maps in branch_sections:
        full_k = len(order)
        is_3d = len(imp_shape) == 2

        z_np_use = z_np[sample_mask] if sample_mask is not None else z_np
        raw_maps_use = [m[sample_mask] if sample_mask is not None else m for m in raw_maps]

        z_traces = []
        for flat_idx in order:
            if is_3d:
                r, c = np.unravel_index(int(flat_idx), imp_shape)
                z_traces.append(z_np_use[:, r, c])
            else:
                z_traces.append(z_np_use[:, int(flat_idx)])
        z_traces = np.array(z_traces)

        sal_profiles = np.array([np.mean(m, axis=0) for m in raw_maps_use])
        for i in range(full_k):
            mx = sal_profiles[i].max()
            if mx > 0:
                sal_profiles[i] /= mx

        spearman_m = np.zeros((full_k, n_feats))
        mi_m = np.zeros((full_k, n_feats))
        cosine_m = np.zeros((full_k, n_feats))

        for i in range(full_k):
            z_i = z_traces[i]
            for j in range(n_feats):
                if not valid_col[j]:
                    continue
                f_j = feat_matrix[:, j]
                mask = np.isfinite(f_j)
                if mask.sum() < 5:
                    continue

                cosine_m[i, j] = max(0.0, _cosine_sim(sal_profiles[i], feat_sensitivity[j]))

                z_sub, f_sub = z_i[mask], f_j[mask]
                if np.ptp(z_sub) == 0 or np.ptp(f_sub) == 0:
                    continue

                corr, _ = scipy.stats.spearmanr(z_sub, f_sub)
                spearman_m[i, j] = 0.0 if np.isnan(corr) else abs(corr)

                try:
                    mi = mutual_info_regression(z_sub.reshape(-1, 1), f_sub, random_state=0)[0]
                except Exception:
                    mi = 0.0
                mi_m[i, j] = mi

        mi_max = mi_m.max()
        if mi_max > 0:
            mi_m /= mi_max

        combined = w_spearman * spearman_m + w_mi * mi_m + w_cosine * cosine_m
        best_feat_idx_full = np.argmax(combined, axis=1)

        selected, seen_feats, true_ranks = [], set(), []
        for i in range(full_k):
            feat = int(best_feat_idx_full[i])
            if feat in seen_feats:
                continue
            seen_feats.add(feat)
            selected.append(i)
            true_ranks.append(i + 1)
            if len(selected) >= TOP_N:
                break

        k = len(selected)
        sections.append({
            "label": label, "order": order[selected], "k": k, "true_ranks": true_ranks,
            "sal_profiles": sal_profiles[selected], "combined": combined[selected],
            "spearman": spearman_m[selected], "mi": mi_m[selected], "cosine": cosine_m[selected],
            "best_feat_idx": best_feat_idx_full[selected], "best_score": combined[selected, best_feat_idx_full[selected]],
        })

    render_latent_feature_mapping_figure(
        sections, mean_curve, t, feat_names,
        model_name, dataset_name, save_path,
        w_spearman=w_spearman, w_mi=w_mi, w_cosine=w_cosine,
        feat_matrix=feat_matrix,
    )


_GROUP_COLOURS = {
    'timing':     '#1a6faf',
    'shape':      '#c45c00',
    'amplitude':  '#2e8b57',
    'derivative': '#8b2ec4',
    'baseline':   '#6e6e6e',
    'integral':   '#bf8000',
    'fit':        '#b03060',
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


def render_latent_feature_mapping_figure(
    sections, mean_curve, timestamps, feat_names,
    model_name, dataset_name, save_path,
    w_spearman=0.35, w_mi=0.25, w_cosine=0.40,
    feat_matrix=None, metric="combined",
):
    if metric not in ("combined", "spearman", "mi", "cosine"):
        raise ValueError(f"metric must be one of 'combined'/'spearman'/'mi'/'cosine', got {metric!r}")
    t = np.asarray(timestamps, dtype=float)
    T = len(mean_curve)
    if len(t) != T:
        t = np.linspace(0, T - 1, T)
    t_step = t[1] - t[0] if T > 1 else 1
    t_start = t[0] - t_step / 2
    t_end = t[-1] + t_step / 2
    n_feats = len(feat_names)

    total_rows = sum(sec["k"] for sec in sections)

    row_h = 1.6
    fig_h = total_rows * row_h + 1.8
    fig_w = 18.3
    bar_w_ratio = 2.5

    target = dataset_name.rsplit('|', 1)[-1].strip() if '|' in dataset_name else dataset_name
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    fig.suptitle(f"{target} Kinetic Feature Correlation", fontsize=17, fontweight='bold', y=1.0)

    gs = fig.add_gridspec(
        total_rows, 2,
        width_ratios=[1, bar_w_ratio],
        hspace=0.45, wspace=0.04,
        left=0.04, right=0.97, top=0.94, bottom=0.08
    )

    cmap_bar = plt.cm.YlOrRd
    x_positions = np.arange(n_feats)

    global_row = 0
    for sec in sections:
        label = sec["label"]
        order = sec["order"]
        k = sec["k"]
        true_ranks = sec["true_ranks"]
        sal_profiles = sec["sal_profiles"]
        metric_scores = sec[metric]

        if metric == "combined":
            best_js = [int(np.argmax(metric_scores[i])) for i in range(k)]
        else:
            used, best_js = set(), []
            for i in range(k):
                ranked = np.argsort(-metric_scores[i])
                chosen = next((int(j) for j in ranked if j not in used), int(ranked[0]))
                used.add(chosen)
                best_js.append(chosen)

        for i in range(k):
            flat_idx = order[i]
            sal_prof = sal_profiles[i]
            scores_row = metric_scores[i]
            best_j = best_js[i]
            score_best = scores_row[best_j]
            row_max = scores_row.max()
            scores_row_disp = scores_row / row_max if row_max > 0 else scores_row

            ax_curve = fig.add_subplot(gs[global_row, 0])
            ax_curve.plot(t, mean_curve, color='black', lw=1.0)
            ax_curve.set_xlim(t_start, t_end)
            ax_curve.set_yticks([])
            ax_curve.tick_params(axis='x', labelsize=9)
            ax_curve.spines[['top', 'right']].set_visible(False)

            ax_sal = ax_curve.twinx()
            ax_sal.fill_between(t, 0, sal_prof, color='#e84040', alpha=0.45, lw=0)
            ax_sal.plot(t, sal_prof, color='#e84040', lw=0.8)
            ax_sal.set_ylim(0, sal_prof.max() * 2.0 if sal_prof.max() > 0 else 1)
            ax_sal.set_yticks([])
            ax_sal.spines[['top', 'right']].set_visible(False)

            best_name = feat_names[best_j]
            if feat_matrix is not None and _FEAT_GROUP.get(best_name) == 'timing':
                t_vals = feat_matrix[:, best_j]
                t_vals = t_vals[np.isfinite(t_vals)]
                if len(t_vals) > 0:
                    t_mark = np.median(t_vals)
                    if t_start <= t_mark <= t_end:
                        ax_curve.axvline(t_mark, color=_GROUP_COLOURS['timing'],
                                          lw=1.2, linestyle=':', zorder=5)

            if i == 0 and label is not None:
                ax_curve.set_title(f"{label} Branch", fontsize=14,
                                    fontweight='bold', loc='left', pad=10)

            ax_curve.set_ylabel(
                f"Rank {true_ranks[i]}\n(dim {flat_idx})",
                fontsize=10, fontweight='bold', rotation=0,
                labelpad=44, va='center'
            )
            if global_row < total_rows - 1:
                plt.setp(ax_curve.get_xticklabels(), visible=False)
            else:
                ax_curve.set_xlabel("Time", fontsize=10)

            ax_bar = fig.add_subplot(gs[global_row, 1])
            bar_colours = [cmap_bar(s) for s in scores_row_disp]
            ax_bar.bar(x_positions, scores_row_disp, color=bar_colours, width=0.85, linewidth=0)
            ax_bar.bar(best_j, scores_row_disp[best_j], color=cmap_bar(scores_row_disp[best_j]),
                       width=0.85, linewidth=1.5, edgecolor='#222222')
            ax_bar.text(
                best_j, scores_row_disp[best_j] + 0.02,
                f"★ {feat_names[best_j]}\n({score_best:.2f})",
                ha='center', va='bottom', fontsize=9.5, fontweight='bold',
                color='#222222',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='#888888', alpha=0.85, linewidth=0.7)
            )

            ax_bar.set_xlim(-0.5, n_feats - 0.5)
            ax_bar.set_ylim(0, 1.34)
            ax_bar.set_yticks([0, 0.5, 1.0])
            ax_bar.tick_params(axis='y', labelsize=9)
            ax_bar.spines[['top', 'right']].set_visible(False)
            ax_bar.axhline(0.5, color='#aaaaaa', lw=0.5, linestyle='--')

            if global_row < total_rows - 1:
                ax_bar.set_xticks([])
            else:
                ax_bar.set_xticks(x_positions)
                ax_bar.set_xticklabels(feat_names, rotation=60, ha='right', fontsize=13)

            global_row += 1

    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"     [✓] Saved latent mapping: {save_path}")
