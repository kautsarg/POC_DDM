import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
    # d(latent dim)/d(X) per top-ranked latent dim -> (N, T) per dim
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


def extract_dual_saliency(model, X_batch, n_dims=100):
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
    ax0.set_title(f'{class_name}  |  View A (normalised)  |  correct N={n_correct}/{n_total}', fontsize=10, fontweight='bold')
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
    ax3.set_title('Σ|∂Z/∂X| (row-norm, collapsed)', fontsize=9, fontweight='bold')
    ax3.grid(True, color='grey', alpha=0.20, ls='--')
    ax3.tick_params(labelsize=7)

    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return True
