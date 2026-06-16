"""
xai_gated_utils.py  —  XAI Utilities for Gated Dual-Branch Models
===================================================================
Analysis and visualisation functions for the eight gated fusion models in
model_utils_gated.py.  All functions are pure computation/visualisation:
no file I/O, no pipeline logic — designed to be imported by
07_attribution_vis_all.py when ready.

Organisation
------------
Section 1 — Gate XAI        : extract/visualise gate values (gate models)
Section 2 — Branch Embedding : PCA / t-SNE of CNN and other-branch embeddings
Section 3 — FiLM XAI        : extract/visualise γ,β modulation params (FiLM models)
Section 4 — Co-Attention XAI : extract/visualise attention weights (co-attn models)
Section 5 — Gate Importance  : discriminability analysis from gate values
Section 6 — Aggregate entry  : get_xai_outputs() — single call for 07 integration

Layer naming conventions (from model_utils_gated.py)
-----------------------------------------------------
  fuse_gate       gate values ∈ [0,1]^32  (gate models)
  fuse_mha1       CNN→other cross-attention  (co-attn models)
  fuse_mha2       other→CNN cross-attention  (co-attn models)
  fuse_q1/q2      query inputs to fuse_mha1/mha2
  fuse_kv1/kv2    key/value inputs to fuse_mha1/mha2
  fuse_gamma1/2   FiLM scale params  (FiLM models)
  fuse_beta1/2    FiLM shift params  (FiLM models)
  cnn_emb         CNN branch embedding  (all models)
  gru_emb         GRU branch embedding  (CNN+GRU variants)
  trans_emb       Transformer branch embedding  (CNN+Trans variants)

Usage
-----
    from utils.model_training.xai_gated_utils import get_xai_outputs
    outputs = get_xai_outputs(model, X, y, class_names, model_type="gate")
    outputs["gate_heatmap"].savefig("gate_heatmap.png")
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False


# ============================================================
# INTERNAL HELPER
# ============================================================

def _make_extractor(model, layer_name):
    """Build a Keras Model that outputs one named layer's activations."""
    output_tensor = model.get_layer(layer_name).output
    return tf.keras.Model(inputs=model.input, outputs=output_tensor)


# ============================================================
# SECTION 1 — GATE XAI  (gate fusion models)
# ============================================================
# Extractable layer: {pfx}_gate  →  gate ∈ [0,1]^32 per sample
# gate ≈ 1 → CNN dominates that embedding dim
# gate ≈ 0 → other branch (GRU/Trans) dominates that embedding dim


def extract_gate_values(model, X, batch_size=64, pfx="fuse"):
    """
    Extract per-sample gate values from a gate-fusion model.

    Parameters
    ----------
    model      : compiled Keras model (create_cnn_{gru,trans}_gate_model)
    X          : np.ndarray  (N, T, 1)
    batch_size : int
    pfx        : str — layer name prefix used when building the model

    Returns
    -------
    gate_values : np.ndarray  (N, 32)
        Values in [0, 1].  gate ≈ 1 → CNN-dominant; gate ≈ 0 → other-dominant.
    """
    extractor = _make_extractor(model, f"{pfx}_gate")
    return extractor.predict(X, batch_size=batch_size, verbose=0)


def gate_stats_per_class(gate_values, y, class_names=None):
    """
    Compute per-class mean gate values.

    Parameters
    ----------
    gate_values : np.ndarray  (N, 32)
    y           : np.ndarray  (N,) — integer class labels
    class_names : list[str] | None

    Returns
    -------
    dict mapping class_name → mean_gate (np.ndarray  (32,))
    Also includes '_overall' key for the global mean.
    """
    classes = np.unique(y)
    if class_names is None:
        class_names = [str(c) for c in classes]
    stats = {name: gate_values[y == cls].mean(axis=0)
             for cls, name in zip(classes, class_names)}
    stats['_overall'] = gate_values.mean(axis=0)
    return stats


def plot_gate_heatmap(gate_stats, figsize=(10, 4), title="Gate Values per Class",
                      save_path=None):
    """
    Heatmap of mean gate values: rows = classes, cols = embedding dims.

    Colour encodes CNN dominance (warm = CNN-dominant, cool = other-dominant).
    Cells are annotated with values for direct reading.

    Parameters
    ----------
    gate_stats : dict from gate_stats_per_class
    figsize    : tuple
    title      : str
    save_path  : str | None

    Returns
    -------
    matplotlib.figure.Figure
    """
    class_names = [k for k in gate_stats if k != '_overall']
    matrix = np.array([gate_stats[c] for c in class_names])  # (n_classes, 32)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap='RdYlBu_r', aspect='auto')
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Class")
    ax.set_title(title)

    for i in range(len(class_names)):
        for j in range(matrix.shape[1]):
            color = 'black' if 0.3 < matrix[i, j] < 0.7 else 'white'
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha='center', va='center',
                    fontsize=5, color=color)

    plt.colorbar(im, ax=ax, label="Gate value  (1=CNN, 0=other)")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
    return fig


def plot_gate_distribution(gate_values, y, class_names=None, n_dims=5,
                           save_path=None):
    """
    Box plots: gate value distribution per class for the top N most discriminative dims.

    Dims are ranked by inter-class variance of mean gate values.

    Parameters
    ----------
    gate_values : np.ndarray  (N, 32)
    y           : np.ndarray  (N,)
    class_names : list[str] | None
    n_dims      : int — number of most-variable dims to show (max 8)
    save_path   : str | None

    Returns
    -------
    matplotlib.figure.Figure
    """
    classes = np.unique(y)
    if class_names is None:
        class_names = [str(c) for c in classes]

    class_means = np.array([gate_values[y == c].mean(axis=0) for c in classes])
    dim_variance = class_means.var(axis=0)
    top_dims = np.argsort(dim_variance)[-n_dims:][::-1]

    colors = cm.tab10(np.linspace(0, 1, len(classes)))
    fig, axes = plt.subplots(1, n_dims, figsize=(3 * n_dims, 4), sharey=True)
    if n_dims == 1:
        axes = [axes]

    for ax, dim in zip(axes, top_dims):
        data_per_class = [gate_values[y == c, dim] for c in classes]
        bp = ax.boxplot(data_per_class, patch_artist=True, tick_labels=class_names)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(f"Dim {dim}\n(ivar={dim_variance[dim]:.3f})")
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='balanced')
        ax.tick_params(axis='x', rotation=30)

    axes[0].set_ylabel("Gate value  (1=CNN, 0=other)")
    fig.suptitle("Gate Distribution per Class (top discriminative dims)", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
    return fig


# ============================================================
# SECTION 2 — BRANCH EMBEDDING XAI  (all gated models)
# ============================================================
# Layers: cnn_emb, gru_emb / trans_emb


def extract_branch_embeddings(model, X, cnn_layer="cnn_emb", other_layer="gru_emb",
                              batch_size=64):
    """
    Extract CNN and other-branch embeddings for latent-space analysis.

    Parameters
    ----------
    model       : compiled Keras gated model
    X           : np.ndarray  (N, T, 1)
    cnn_layer   : str — layer name for CNN embedding (default "cnn_emb")
    other_layer : str — "gru_emb" for GRU variants, "trans_emb" for Transformer
    batch_size  : int

    Returns
    -------
    dict:
        'cnn'   : np.ndarray  (N, 32)
        'other' : np.ndarray  (N, 32)
    """
    return {
        'cnn':   _make_extractor(model, cnn_layer).predict(X, batch_size=batch_size, verbose=0),
        'other': _make_extractor(model, other_layer).predict(X, batch_size=batch_size, verbose=0),
    }


def plot_branch_pca(branch_embs, y, class_names=None, figsize=(12, 5),
                    save_path=None):
    """
    Side-by-side PCA scatter: CNN branch (left) vs other branch (right).

    Both plots use the same class→colour mapping for direct visual comparison.

    Parameters
    ----------
    branch_embs : dict from extract_branch_embeddings
    y           : np.ndarray  (N,)
    class_names : list[str] | None
    figsize     : tuple
    save_path   : str | None

    Returns
    -------
    matplotlib.figure.Figure
    """
    from sklearn.decomposition import PCA

    classes = np.unique(y)
    if class_names is None:
        class_names = [str(c) for c in classes]
    colors = cm.tab10(np.linspace(0, 1, len(classes)))

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, (key, label) in zip(axes, [('cnn', 'CNN branch'), ('other', 'Other branch')]):
        pca = PCA(n_components=2)
        coords = pca.fit_transform(branch_embs[key])
        var_exp = pca.explained_variance_ratio_
        for cls, (name, color) in enumerate(zip(class_names, colors)):
            mask = y == cls
            ax.scatter(coords[mask, 0], coords[mask, 1], c=[color], label=name,
                       alpha=0.6, s=20, linewidths=0)
        ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)")
        ax.set_title(label)
        ax.legend(fontsize=8)

    fig.suptitle("Branch Embedding PCA")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
    return fig


def plot_branch_tsne(branch_embs, y, class_names=None, perplexity=30,
                     figsize=(12, 5), save_path=None):
    """
    Side-by-side t-SNE scatter: CNN branch (left) vs other branch (right).

    Slower than PCA but better at exposing non-linear cluster structure.

    Parameters
    ----------
    branch_embs : dict from extract_branch_embeddings
    y           : np.ndarray  (N,)
    class_names : list[str] | None
    perplexity  : int — t-SNE perplexity, recommend ≤ n_samples / 5
    figsize     : tuple
    save_path   : str | None

    Returns
    -------
    matplotlib.figure.Figure
    """
    from sklearn.manifold import TSNE

    classes = np.unique(y)
    if class_names is None:
        class_names = [str(c) for c in classes]
    colors = cm.tab10(np.linspace(0, 1, len(classes)))

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, (key, label) in zip(axes, [('cnn', 'CNN branch'), ('other', 'Other branch')]):
        coords = TSNE(n_components=2, perplexity=perplexity,
                      random_state=42).fit_transform(branch_embs[key])
        for cls, (name, color) in enumerate(zip(class_names, colors)):
            mask = y == cls
            ax.scatter(coords[mask, 0], coords[mask, 1], c=[color], label=name,
                       alpha=0.6, s=20, linewidths=0)
        ax.set_title(f"{label}  (t-SNE, perp={perplexity})")
        ax.legend(fontsize=8)

    fig.suptitle("Branch Embedding t-SNE")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
    return fig


# ============================================================
# SECTION 3 — FiLM MODULATION XAI  (FiLM fusion models)
# ============================================================
# Layers: {pfx}_gamma1, {pfx}_beta1, {pfx}_gamma2, {pfx}_beta2


def extract_film_params(model, X, pfx="fuse", batch_size=64):
    """
    Extract FiLM modulation parameters γ and β for both conditioning stages.

    Stage 1: CNN affinely modulates the raw other-branch embedding.
    Stage 2: CNN affinely modulates a Dense-transformed version of stage-1 output.

    Parameters
    ----------
    model      : compiled Keras model (create_cnn_{gru,trans}_film_model)
    X          : np.ndarray  (N, T, 1)
    pfx        : str — layer prefix (default "fuse")
    batch_size : int

    Returns
    -------
    dict:
        'gamma1' : (N, 32) — stage-1 scale (predicted by CNN from its embedding)
        'beta1'  : (N, 32) — stage-1 shift
        'gamma2' : (N, 32) — stage-2 scale
        'beta2'  : (N, 32) — stage-2 shift
    """
    layers = {
        'gamma1': f"{pfx}_gamma1",
        'beta1':  f"{pfx}_beta1",
        'gamma2': f"{pfx}_gamma2",
        'beta2':  f"{pfx}_beta2",
    }
    return {key: _make_extractor(model, lname).predict(X, batch_size=batch_size, verbose=0)
            for key, lname in layers.items()}


def plot_film_modulation(film_params, y, class_names=None, figsize=(14, 8),
                         save_path=None):
    """
    2×2 heatmap grid: per-class mean γ1, β1, γ2, β2 over embedding dims.

    Rows = classes, cols = embedding dims (32).
    γ ≈ 1 means 'keep this dim unchanged'; β ≈ 0 means 'no shift applied'.
    Deviations from these baselines reveal which dims CNN actively modulates
    and whether modulation differs across classes.

    Parameters
    ----------
    film_params : dict from extract_film_params
    y           : np.ndarray  (N,)
    class_names : list[str] | None
    figsize     : tuple
    save_path   : str | None

    Returns
    -------
    matplotlib.figure.Figure
    """
    classes = np.unique(y)
    if class_names is None:
        class_names = [str(c) for c in classes]

    panels = [
        ('gamma1', 'Stage-1  γ  (scale)'),
        ('beta1',  'Stage-1  β  (shift)'),
        ('gamma2', 'Stage-2  γ  (scale)'),
        ('beta2',  'Stage-2  β  (shift)'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for ax, (key, label) in zip(axes.flatten(), panels):
        matrix = np.array([film_params[key][y == c].mean(axis=0) for c in classes])
        vabs = max(abs(matrix.min()), abs(matrix.max()), 1e-6)
        im = ax.imshow(matrix, vmin=-vabs, vmax=vabs, cmap='coolwarm', aspect='auto')
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Embedding Dim")
        ax.set_title(label)
        plt.colorbar(im, ax=ax)

    fig.suptitle("FiLM Modulation Parameters per Class  (CNN conditioning other branch)")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
    return fig


# ============================================================
# SECTION 4 — CO-ATTENTION XAI  (co-attention fusion models)
# ============================================================
# Layers: fuse_q1, fuse_kv1 → fuse_mha1  (CNN→other direction)
#         fuse_q2, fuse_kv2 → fuse_mha2  (other→CNN direction)


def build_coattn_weight_extractor(model, pfx="fuse", direction=1):
    """
    Build an extractor model that returns attention weight scores from a co-attention model.

    This calls the existing MHA layer a second time with return_attention_scores=True.
    Keras functional API supports layer reuse: the second call shares the layer's
    trained weights while producing new symbolic outputs (attention scores).

    Parameters
    ----------
    model     : compiled Keras co-attention model
    pfx       : str — layer prefix (default "fuse")
    direction : int — 1 = CNN→other  (CNN_emb queries other_seq)
                      2 = other→CNN  (other_emb queries cnn_seq)

    Returns
    -------
    tf.keras.Model — same input as `model`, outputs attention weights
        shape: (batch, num_heads, 1, T_seq)
        T_seq = key/value sequence length for that direction.

    Notes
    -----
    Use only in inference mode (training=False).  Attention scores are computed
    before the internal dropout of MHA, giving stable visualisations.
    """
    mha_layer = model.get_layer(f"{pfx}_mha{direction}")
    q_tensor  = model.get_layer(f"{pfx}_q{direction}").output   # (B, 1, emb_dim)
    kv_tensor = model.get_layer(f"{pfx}_kv{direction}").output  # (B, T, emb_dim)

    _, attn_weights = mha_layer(
        query=q_tensor, key=kv_tensor, value=kv_tensor,
        return_attention_scores=True,
    )
    # attn_weights: (batch, num_heads, 1, T_seq)
    return tf.keras.Model(inputs=model.input, outputs=attn_weights)


def extract_attention_weights(model, X, pfx="fuse", direction=1, batch_size=64):
    """
    Extract cross-attention weights from a co-attention model.

    Parameters
    ----------
    model     : compiled Keras co-attention model
    X         : np.ndarray  (N, T, 1)
    pfx       : str — layer prefix (default "fuse")
    direction : int — 1 = CNN→other (what in other_seq does CNN attend to?)
                      2 = other→CNN (what in cnn_seq does other branch attend to?)
    batch_size: int

    Returns
    -------
    attn_weights : np.ndarray  (N, num_heads, 1, T_seq)
    """
    extractor = build_coattn_weight_extractor(model, pfx=pfx, direction=direction)
    return extractor.predict(X, batch_size=batch_size, verbose=0)


def plot_attention_heatmap(attn_weights, y, class_names=None, timestamps=None,
                           figsize=None, save_path=None):
    """
    Filled area plot of mean attention over time, per class and per head.

    Rows = classes, cols = heads.  Reveals which temporal regions the cross-attention
    mechanism focuses on for each class when attending to the other branch's sequence.

    Parameters
    ----------
    attn_weights : np.ndarray  (N, num_heads, 1, T_seq)
    y            : np.ndarray  (N,)
    class_names  : list[str] | None
    timestamps   : np.ndarray | None — x-axis values (e.g. actual time grid)
    figsize      : tuple | None — auto-computed if None
    save_path    : str | None

    Returns
    -------
    matplotlib.figure.Figure
    """
    attn = attn_weights[:, :, 0, :]  # (N, num_heads, T_seq)
    num_heads = attn.shape[1]
    classes   = np.unique(y)
    if class_names is None:
        class_names = [str(c) for c in classes]
    n_classes = len(classes)

    if figsize is None:
        figsize = (max(10, attn.shape[2] // 4), 3 * ((n_classes * num_heads + 1) // 2))

    fig, axes = plt.subplots(n_classes, num_heads, figsize=figsize, squeeze=False)
    for row, (cls, cname) in enumerate(zip(classes, class_names)):
        mask = y == cls
        class_attn = attn[mask]  # (n_cls, num_heads, T_seq)
        for col in range(num_heads):
            ax = axes[row][col]
            mean_attn = class_attn[:, col, :].mean(axis=0)  # (T_seq,)
            std_attn  = class_attn[:, col, :].std(axis=0)
            x = timestamps if timestamps is not None else np.arange(len(mean_attn))
            ax.fill_between(x, mean_attn - std_attn, mean_attn + std_attn,
                            alpha=0.2, color='steelblue')
            ax.fill_between(x, mean_attn, alpha=0.4, color='steelblue')
            ax.plot(x, mean_attn, color='steelblue', linewidth=1.5)
            ax.set_ylim(0, None)
            ax.set_title(f"{cname} | Head {col + 1}", fontsize=9)
            if row == n_classes - 1:
                ax.set_xlabel("Timestep" if timestamps is None else "Time")
            if col == 0:
                ax.set_ylabel("Attention weight")

    fig.suptitle("Mean Cross-Attention over Time  (per class × head)")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
    return fig


# ============================================================
# SECTION 5 — GATE IMPORTANCE  (model-agnostic analysis of gate values)
# ============================================================


def compute_gate_cnn_dominance(gate_values, y=None, class_names=None):
    """
    Summarise how much the CNN branch dominates over the other branch.

    overall_cnn_dominance = mean gate value across all samples and dims.
    0.5 = perfectly balanced; > 0.5 = CNN-dominant; < 0.5 = other-dominant.

    Parameters
    ----------
    gate_values : np.ndarray  (N, 32)
    y           : np.ndarray | None
    class_names : list[str] | None

    Returns
    -------
    dict:
        'overall_cnn_dominance' : float
        'per_dim_overall'       : np.ndarray  (32,)
        'per_class'             : dict {class_name: float}  — if y provided
    """
    result = {
        'overall_cnn_dominance': float(gate_values.mean()),
        'per_dim_overall':       gate_values.mean(axis=0),
    }
    if y is not None:
        classes = np.unique(y)
        if class_names is None:
            class_names = [str(c) for c in classes]
        result['per_class'] = {
            name: float(gate_values[y == cls].mean())
            for cls, name in zip(classes, class_names)
        }
    return result


def rank_gate_dims(gate_values, y, top_k=10):
    """
    Rank embedding dimensions by their discriminative power from gate values.

    Score = inter-class variance of mean gate values.
    High-scoring dims are the ones where different classes consistently
    use different CNN-vs-other blending ratios — most interpretable dims.

    Parameters
    ----------
    gate_values : np.ndarray  (N, 32)
    y           : np.ndarray  (N,)
    top_k       : int

    Returns
    -------
    dict:
        'dim_indices'  : np.ndarray  (top_k,) — ranked by discriminability
        'dim_scores'   : np.ndarray  (top_k,) — inter-class variance scores
        'class_means'  : np.ndarray  (n_classes, 32) — per-class mean per dim
    """
    classes = np.unique(y)
    class_means = np.array([gate_values[y == c].mean(axis=0) for c in classes])
    scores   = class_means.var(axis=0)
    top_dims = np.argsort(scores)[-top_k:][::-1]
    return {
        'dim_indices': top_dims,
        'dim_scores':  scores[top_dims],
        'class_means': class_means,
    }


# ============================================================
# SECTION 6 — AGGREGATE ENTRY POINT  (for 07 integration)
# ============================================================


def get_xai_outputs(model, X, y, class_names=None, model_type="gate",
                    timestamps=None, pfx="fuse", batch_size=64):
    """
    Aggregate all XAI outputs for a gated model in one call.

    This is the intended entry point when integrating with 07_attribution_vis_all.py.
    It runs the appropriate analysis path based on model_type and returns all
    figures and arrays needed to build the report section for that model.

    Parameters
    ----------
    model      : compiled Keras gated model
    X          : np.ndarray  (N, T, 1)
    y          : np.ndarray  (N,) — integer labels
    class_names: list[str] | None — if None, derived from unique(y)
    model_type : str — "gate" | "hadamard" | "coattn" | "film"
    timestamps : np.ndarray | None — for attention plots x-axis
    pfx        : str — layer name prefix (default "fuse")
    batch_size : int

    Returns
    -------
    dict — keys depend on model_type:

        Always present:
            'branch_embs'    : {'cnn': (N,32), 'other': (N,32)}
            'branch_pca_fig' : Figure — side-by-side PCA scatter

        model_type == "gate":
            'gate_values'    : (N, 32)
            'gate_stats'     : {class_name: (32,) mean gate}
            'gate_heatmap'   : Figure
            'gate_dist_fig'  : Figure
            'dominance'      : dict from compute_gate_cnn_dominance
            'gate_dim_ranks' : dict from rank_gate_dims

        model_type == "film":
            'film_params'    : {'gamma1','beta1','gamma2','beta2': (N,32)}
            'film_fig'       : Figure

        model_type == "coattn":
            'attn1'          : (N, heads, 1, T_seq)  — CNN→other
            'attn2'          : (N, heads, 1, T_seq)  — other→CNN
            'attn1_fig'      : Figure
            'attn2_fig'      : Figure

        model_type == "hadamard":
            (branch embeddings only — no additional fusion-specific XAI)
    """
    # Detect which branch was used from layer names
    other_layer = "gru_emb"
    try:
        model.get_layer("gru_emb")
    except ValueError:
        other_layer = "trans_emb"

    result = {}

    branch_embs = extract_branch_embeddings(
        model, X,
        cnn_layer="cnn_emb", other_layer=other_layer,
        batch_size=batch_size,
    )
    result['branch_embs']    = branch_embs
    result['branch_pca_fig'] = plot_branch_pca(branch_embs, y, class_names=class_names)

    if model_type == "gate":
        gate_vals = extract_gate_values(model, X, batch_size=batch_size, pfx=pfx)
        gate_stats = gate_stats_per_class(gate_vals, y, class_names=class_names)
        result['gate_values']    = gate_vals
        result['gate_stats']     = gate_stats
        result['gate_heatmap']   = plot_gate_heatmap(gate_stats)
        result['gate_dist_fig']  = plot_gate_distribution(gate_vals, y, class_names=class_names)
        result['dominance']      = compute_gate_cnn_dominance(gate_vals, y, class_names=class_names)
        result['gate_dim_ranks'] = rank_gate_dims(gate_vals, y)

    elif model_type == "film":
        film_params = extract_film_params(model, X, pfx=pfx, batch_size=batch_size)
        result['film_params'] = film_params
        result['film_fig']    = plot_film_modulation(film_params, y, class_names=class_names)

    elif model_type == "coattn":
        attn1 = extract_attention_weights(model, X, pfx=pfx, direction=1, batch_size=batch_size)
        attn2 = extract_attention_weights(model, X, pfx=pfx, direction=2, batch_size=batch_size)
        result['attn1']     = attn1
        result['attn2']     = attn2
        result['attn1_fig'] = plot_attention_heatmap(
            attn1, y, class_names=class_names, timestamps=timestamps)
        result['attn2_fig'] = plot_attention_heatmap(
            attn2, y, class_names=class_names, timestamps=timestamps)

    # model_type == "hadamard" → branch embeddings only (no fusion-specific XAI layer)

    return result


# ============================================================
# __main__ — quick smoke test
# ============================================================

if __name__ == "__main__":
    print("xai_gated_utils.py — XAI utilities for gated dual-branch models")
    print()

    public_api = [
        ("Section 1: Gate XAI",
         ["extract_gate_values", "gate_stats_per_class",
          "plot_gate_heatmap", "plot_gate_distribution"]),
        ("Section 2: Branch Embedding",
         ["extract_branch_embeddings", "plot_branch_pca", "plot_branch_tsne"]),
        ("Section 3: FiLM XAI",
         ["extract_film_params", "plot_film_modulation"]),
        ("Section 4: Co-Attention XAI",
         ["build_coattn_weight_extractor", "extract_attention_weights",
          "plot_attention_heatmap"]),
        ("Section 5: Gate Importance",
         ["compute_gate_cnn_dominance", "rank_gate_dims"]),
        ("Section 6: Aggregate entry",
         ["get_xai_outputs"]),
    ]
    for section, fns in public_api:
        print(f"  {section}")
        for fn in fns:
            print(f"    - {fn}")

    print()
    print("Usage:")
    print("  from utils.model_training.xai_gated_utils import get_xai_outputs")
    print("  outputs = get_xai_outputs(model, X, y, class_names, model_type='gate')")
    print()

    # Minimal integration test (requires model_utils_gated and TF)
    try:
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        from model_utils_gated import create_cnn_gru_gate_model, create_cnn_gru_film_model
        from model_utils_gated import create_cnn_gru_crossattn_model

        N, T, n_classes = 60, 100, 3
        rng = np.random.default_rng(0)
        X_test = rng.standard_normal((N, T, 1)).astype(np.float32)
        y_test = rng.integers(0, n_classes, size=N)
        cnames = ["A", "B", "C"]

        print("Running smoke tests (N=60, T=100, n_classes=3)...")

        # Gate model
        m_gate = create_cnn_gru_gate_model(T, n_classes)
        out = get_xai_outputs(m_gate, X_test, y_test, cnames, model_type="gate")
        assert out['gate_values'].shape == (N, 32), "gate shape mismatch"
        print("  [PASS] gate model: gate_values, gate_heatmap, gate_dist_fig, branch_pca_fig")

        # FiLM model
        m_film = create_cnn_gru_film_model(T, n_classes)
        out = get_xai_outputs(m_film, X_test, y_test, cnames, model_type="film")
        assert out['film_params']['gamma1'].shape == (N, 32), "film gamma1 shape mismatch"
        print("  [PASS] film model: film_params (gamma1/beta1/gamma2/beta2), film_fig")

        # Co-attention model
        m_coattn = create_cnn_gru_crossattn_model(T, n_classes)
        out = get_xai_outputs(m_coattn, X_test, y_test, cnames, model_type="coattn")
        assert out['attn1'].shape[0] == N, "attn1 batch size mismatch"
        print("  [PASS] coattn model: attn1/attn2, attn1_fig/attn2_fig")

        print()
        print("All smoke tests passed.")
        plt.close('all')

    except Exception as exc:
        print(f"  [SKIP] Smoke test skipped: {exc}")
