import os
# ====================================================================
# SUPPRESS TENSORFLOW C++ WARNINGS (Must be before TF import)
# ====================================================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=INFO, 1=WARN, 2=ERROR, 3=FATAL

import gc
import time
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import joblib
import tensorflow as tf
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
tf.get_logger().setLevel('ERROR')

from scikeras.wrappers import KerasClassifier

import model_utils_gated

# ====================================================================
# GPU SETUP & VERIFICATION
# ====================================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"\n[*] SUCCESS: TensorFlow is utilizing the GPU -> {gpus[0].name}\n")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
else:
    print("\n[!] WARNING: No GPU found. TensorFlow will run on the CPU.")
    print("    Ensure you have installed: pip install tensorflow-macos tensorflow-metal\n")

# ====================================================================
# GLOBAL DETERMINISM SETUP
# ====================================================================
def set_global_determinism(seed=0, strict=True):
    """RNG seeding always happens. `strict=True` (default) additionally forces
    TensorFlow off cuDNN's fast non-deterministic RNN kernels for bit-exact reruns —
    this is the single biggest cost for GRU/LSTM/Transformer training. Pass
    `strict=False` (currently only exposed via 04_cross_dataset_training.py's
    --fast_mode) to keep seeding but allow cuDNN's fast path."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if strict:
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        try:
            tf.config.experimental.enable_op_determinism()
        except AttributeError:
            pass


# ====================================================================
# CURVE RESAMPLING (for combining datasets with different timestamp grids)
# ====================================================================
class CurveResampler:
    """
    Resamples curves onto a single common time grid via linear interpolation.

    The grid spans [0, duration], where `duration` is the shortest curve
    duration seen during `fit` (so every fitted curve covers the full grid
    and no extrapolation is needed). Curves shorter than the grid (e.g. at
    inference time) are flat-extrapolated, since `np.interp` clips to the
    boundary value outside the source range.
    """

    def __init__(self, t_grid):
        self.t_grid = np.asarray(t_grid, dtype=float)

    @classmethod
    def fit(cls, timestamps_list, n_points=None):
        """timestamps_list: list of 1D arrays, each starting at t=0 (i.e. t - t[0])."""
        duration = min(t[-1] for t in timestamps_list)
        if n_points is None:
            n_points = max(int(np.sum(t <= duration)) for t in timestamps_list)
        return cls(np.linspace(0, duration, n_points))

    def transform(self, t_raw, curves):
        """curves: (N, T) array sharing timestamps t_raw -> (N, len(t_grid))."""
        t_zeroed = np.asarray(t_raw, dtype=float) - t_raw[0]
        return np.array([np.interp(self.t_grid, t_zeroed, c) for c in curves])

# ====================================================================
# DUAL MODEL
# ====================================================================

def _build_cnn_gru_dual_branches(input_curve):
    """Dual-branch CNN (Local) + BiGRU (Global) feature extractor.

    Takes a Keras tensor of shape (T, 1) and returns the pre-output fused embedding.
    Factored out of create_cnn_gru_dual_model so the neighbor-reconstruction variants
    (create_cnn_gru_dual_attn_recon_model) can feed a *different* curve tensor — e.g.
    one reconstructed from a pixel + its spatial neighbours — into this exact same
    downstream architecture, rather than duplicating it.
    """
    # 1. Local Feature Branch (CNN)
    c = tf.keras.layers.Conv1D(16, 5, activation='relu')(input_curve)
    c = tf.keras.layers.Conv1D(8, 3, activation='relu')(c)
    c = tf.keras.layers.Flatten()(c)
    cnn_emb = tf.keras.layers.Dense(32, activation='relu')(c)

    # 2. Global Feature Branch (BiGRU)
    g = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, return_sequences=True))(input_curve)
    g = tf.keras.layers.LayerNormalization()(g)
    g = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(16))(g)
    g = tf.keras.layers.Dropout(0.2)(g)
    gru_emb = tf.keras.layers.Dense(32, activation='relu')(g)

    # 3. Fusion
    merged = tf.keras.layers.Concatenate()([cnn_emb, gru_emb])
    z = tf.keras.layers.Dense(64, activation='relu')(merged)
    z = tf.keras.layers.Dropout(0.2)(z)
    return z


def create_cnn_gru_dual_model(input_size_curve, output_size):
    """
    Dual-branch architecture combining CNN (Local) and BiGRU (Global)
    using only the raw curve as input.
    """
    input_curve = tf.keras.layers.Input(shape=(input_size_curve, 1), name="curve_input")
    z = _build_cnn_gru_dual_branches(input_curve)
    outputs = tf.keras.layers.Dense(output_size, activation='softmax')(z)

    # Compile
    model = tf.keras.models.Model(inputs=input_curve, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# ====================================================================
# SPATIAL NEIGHBOR RECONSTRUCTION (cnn_gru_dual_cosine_recon / cnn_gru_dual_attn_recon)
# ====================================================================
# Two variants of an idea inspired by the 03b_gnn_spatial_training.py GNN: instead of
# message-passing on an explicit graph, reconstruct ONE denoised curve per pixel from
# itself + its k spatial neighbours (within the same well -- cross-well pixels are
# different reactions/labels entirely, see 03b's well_ids fallback to Y_well), and feed
# that single curve into the *unmodified* cnn_gru_dual architecture. The two variants
# differ only in how the reconstruction weights are produced:
#   - cosine_recon : weights = softmax(cosine similarity to the pixel's own curve).
#                    Fixed, no learned parameters -- a deterministic preprocessing step.
#   - attn_recon   : weights = learned attention (own curve's embedding as query, all
#                    k+1 embeddings as keys), trained end-to-end with the classifier.

def build_neighbor_curve_stack(curves, coords, well_ids, k):
    """
    For each row i, finds its k nearest neighbours by pixel coords *within the same
    well* (mirrors 03b_gnn_spatial_training.py's build_knn_edge_index). Returns
    (N, k+1, T): the pixel's own curve at index 0, followed by its k nearest
    neighbours' curves (nearest-first).

    Wells with fewer than k other pixels pad by cycling through whatever neighbours
    exist (or repeating the pixel itself, for a well of size 1) so every row gets a
    fixed-size (k+1, T) stack -- needed for a dense tensor rather than a per-well
    variable-length structure.
    """
    n, t = curves.shape
    stack = np.empty((n, k + 1, t), dtype=curves.dtype)

    for well in np.unique(well_ids):
        well_idx = np.where(well_ids == well)[0]
        well_coords = coords[well_idx]
        well_curves = curves[well_idx]
        n_well = len(well_idx)

        k_actual = min(k, n_well - 1)
        if k_actual > 0:
            nbrs = NearestNeighbors(n_neighbors=k_actual + 1).fit(well_coords)
            _, neighbor_pos = nbrs.kneighbors(well_coords)  # (n_well, k_actual+1), includes self

        for local_i, global_i in enumerate(well_idx):
            stack[global_i, 0] = well_curves[local_i]

            if k_actual > 0:
                chosen = neighbor_pos[local_i]
                chosen = chosen[chosen != local_i][:k]
            else:
                chosen = np.array([], dtype=int)

            if len(chosen) < k:
                if len(chosen) == 0:
                    chosen = np.full(k, local_i, dtype=int)          # well of size 1: repeat self
                else:
                    reps = int(np.ceil(k / len(chosen)))
                    chosen = np.tile(chosen, reps)[:k]

            stack[global_i, 1:] = well_curves[chosen]

    return stack


def reconstruct_curves_cosine(stack):
    """
    stack: (N, k+1, T), own curve at index 0. Reconstruction weights are the
    softmax of the cosine similarity between each row's own curve and every curve in
    its stack (including itself) -- no learned parameters, purely a function of the
    raw curve values.
    """
    own = stack[:, 0:1, :]
    own_norm = own / (np.linalg.norm(own, axis=-1, keepdims=True) + 1e-8)
    stack_norm = stack / (np.linalg.norm(stack, axis=-1, keepdims=True) + 1e-8)
    sims = np.sum(own_norm * stack_norm, axis=-1)                    # (N, k+1)

    sims = sims - sims.max(axis=1, keepdims=True)                    # numerical stability
    weights = np.exp(sims)
    weights /= weights.sum(axis=1, keepdims=True)
    return np.sum(weights[:, :, None] * stack, axis=1)               # (N, T)


def create_cnn_gru_dual_attn_recon_model(k_plus_1, input_size_curve, output_size, attn_dim=16):
    """
    Learnable-attention counterpart to reconstruct_curves_cosine: a small shared
    per-curve encoder produces an embedding for each of the k+1 curves in the stack;
    the pixel's own embedding is the query, all k+1 embeddings are keys, and the
    resulting softmax attention weights are applied to the *raw curves* (not the
    embeddings) to produce one reconstructed (T,) curve. That curve then flows into
    _build_cnn_gru_dual_branches -- the exact same downstream architecture
    create_cnn_gru_dual_model uses -- so the only difference from the baseline is
    which curve the classifier sees, and the whole thing (encoder + attention +
    classifier) trains end-to-end via ordinary model.fit.
    """
    stack_input = tf.keras.layers.Input(shape=(k_plus_1, input_size_curve), name="neighbor_stack_input")

    per_curve_encoder = tf.keras.Sequential([
        tf.keras.layers.Reshape((input_size_curve, 1)),
        tf.keras.layers.Conv1D(16, 5, activation='relu', padding='same'),
        tf.keras.layers.Conv1D(8, 3, activation='relu', padding='same'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(attn_dim, activation='relu'),
    ], name="per_curve_encoder")
    embeddings = tf.keras.layers.TimeDistributed(per_curve_encoder)(stack_input)   # (N, k+1, attn_dim)

    query = tf.keras.layers.Lambda(lambda x: x[:, 0:1, :])(embeddings)             # (N, 1, attn_dim)
    scores = tf.keras.layers.Lambda(
        lambda t: tf.matmul(t[0], t[1], transpose_b=True) / tf.sqrt(tf.cast(attn_dim, tf.float32))
    )([query, embeddings])                                                        # (N, 1, k+1)
    attn_weights = tf.keras.layers.Softmax(axis=-1, name="attn_weights")(scores)  # (N, 1, k+1)

    reconstructed = tf.keras.layers.Lambda(
        lambda t: tf.matmul(t[0], t[1])                                          # (N,1,k+1) @ (N,k+1,T) -> (N,1,T)
    )([attn_weights, stack_input])
    reconstructed = tf.keras.layers.Reshape((input_size_curve, 1))(reconstructed)  # (N, T, 1)

    z = _build_cnn_gru_dual_branches(reconstructed)
    outputs = tf.keras.layers.Dense(output_size, activation='softmax')(z)

    model = tf.keras.models.Model(inputs=stack_input, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def create_cnn_transformer_dual_model(input_size_curve, output_size, head_size=32, num_heads=2, ff_dim=32, num_blocks=2, dropout=0.1):
    """
    Dual-branch architecture combining CNN (Local) and Transformer (Global) 
    using only the raw curve as input.
    """
    input_curve = tf.keras.layers.Input(shape=(input_size_curve, 1), name="curve_input")
    
    # 1. Local Feature Branch (CNN)
    c = tf.keras.layers.Conv1D(16, 5, activation='relu')(input_curve)
    c = tf.keras.layers.Conv1D(8, 3, activation='relu')(c)
    c = tf.keras.layers.Flatten()(c)
    cnn_emb = tf.keras.layers.Dense(32, activation='relu')(c)
    
    # 2. Global Feature Branch (Transformer)
    t = tf.keras.layers.Conv1D(filters=head_size, kernel_size=5, strides=2, padding="same", activation="relu")(input_curve)
    t = tf.keras.layers.MaxPooling1D(pool_size=2, padding="same")(t)
    
    new_seq_len = t.shape[1] 
    positions = tf.range(start=0, limit=new_seq_len, delta=1)
    pos_embedding = tf.keras.layers.Embedding(input_dim=new_seq_len, output_dim=head_size)(positions)
    t = t + pos_embedding 
    
    for _ in range(num_blocks):
        attn_output = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(t, t)
        attn_output = tf.keras.layers.Dropout(dropout)(attn_output)
        t = tf.keras.layers.LayerNormalization(epsilon=1e-6)(t + attn_output)

        ffn_output = tf.keras.layers.Dense(ff_dim, activation="relu")(t)
        ffn_output = tf.keras.layers.Dropout(dropout)(ffn_output)
        ffn_output = tf.keras.layers.Dense(head_size)(ffn_output) 
        t = tf.keras.layers.LayerNormalization(epsilon=1e-6)(t + ffn_output)

    t = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_last")(t)
    trans_emb = tf.keras.layers.Dense(32, activation="relu")(t)
    
    # 3. Fusion & Output
    merged = tf.keras.layers.Concatenate()([cnn_emb, trans_emb])
    z = tf.keras.layers.Dense(64, activation='relu')(merged)
    z = tf.keras.layers.Dropout(0.2)(z)
    outputs = tf.keras.layers.Dense(output_size, activation='softmax')(z)

    # Compile
    model = tf.keras.models.Model(inputs=input_curve, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ====================================================================
# LATE FUSION NEURAL NETWORKS (Multi-Input)
# ====================================================================
def create_cnn_lf_model(input_size_curve, input_size_features, output_size):
    # 1. Raw Curve Branch
    input_curve = tf.keras.layers.Input(shape=(input_size_curve, 1), name="curve_input")
    x = tf.keras.layers.Conv1D(16, 5, activation='relu')(input_curve)
    x = tf.keras.layers.Conv1D(8, 3, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    curve_emb = tf.keras.layers.Dense(32, activation='relu')(x)
    
    # 2. Manual Features Branch
    input_features = tf.keras.layers.Input(shape=(input_size_features,), name="features_input")
    feat_emb = tf.keras.layers.Dense(32, activation='relu')(input_features)
    
    # 3. Fusion & Output
    merged = tf.keras.layers.Concatenate()([curve_emb, feat_emb])
    z = tf.keras.layers.Dense(32, activation='relu')(merged)
    z = tf.keras.layers.Dropout(0.2)(z)
    outputs = tf.keras.layers.Dense(output_size, activation='softmax')(z)
    
    model = tf.keras.models.Model(inputs=[input_curve, input_features], outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_lstm_lf_model(input_size_curve, input_size_features, output_size):
    input_curve = tf.keras.layers.Input(shape=(input_size_curve, 1), name="curve_input")
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))(input_curve)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    curve_emb = tf.keras.layers.Dense(32, activation='relu')(x)
    
    input_features = tf.keras.layers.Input(shape=(input_size_features,), name="features_input")
    feat_emb = tf.keras.layers.Dense(32, activation='relu')(input_features)
    
    merged = tf.keras.layers.Concatenate()([curve_emb, feat_emb])
    z = tf.keras.layers.Dense(32, activation='relu')(merged)
    z = tf.keras.layers.Dropout(0.2)(z)
    outputs = tf.keras.layers.Dense(output_size, activation='softmax')(z)
    
    model = tf.keras.models.Model(inputs=[input_curve, input_features], outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_gru_lf_model(input_size_curve, input_size_features, output_size):
    # 1. Raw Curve Branch (GRU)
    input_curve = tf.keras.layers.Input(shape=(input_size_curve, 1), name="curve_input")
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, return_sequences=True))(input_curve)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(16))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    curve_emb = tf.keras.layers.Dense(32, activation='relu')(x)
    
    # 2. Manual Features Branch
    input_features = tf.keras.layers.Input(shape=(input_size_features,), name="features_input")
    feat_emb = tf.keras.layers.Dense(32, activation='relu')(input_features)
    
    # 3. Fusion & Output
    merged = tf.keras.layers.Concatenate()([curve_emb, feat_emb])
    z = tf.keras.layers.Dense(32, activation='relu')(merged)
    z = tf.keras.layers.Dropout(0.2)(z)
    outputs = tf.keras.layers.Dense(output_size, activation='softmax')(z)
    
    # Compile
    model = tf.keras.models.Model(inputs=[input_curve, input_features], outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_transformer_lf_model(input_size_curve, input_size_features, output_size, head_size=32, num_heads=2, ff_dim=32, num_blocks=2, dropout=0.1):
    input_curve = tf.keras.layers.Input(shape=(input_size_curve, 1), name="curve_input")
    
    x = tf.keras.layers.Conv1D(filters=head_size, kernel_size=5, strides=2, padding="same", activation="relu")(input_curve)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, padding="same")(x)
    
    new_seq_len = x.shape[1] 
    positions = tf.range(start=0, limit=new_seq_len, delta=1)
    pos_embedding = tf.keras.layers.Embedding(input_dim=new_seq_len, output_dim=head_size)(positions)
    x = x + pos_embedding 
    
    for _ in range(num_blocks):
        attn_output = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
        attn_output = tf.keras.layers.Dropout(dropout)(attn_output)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

        ffn_output = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
        ffn_output = tf.keras.layers.Dropout(dropout)(ffn_output)
        ffn_output = tf.keras.layers.Dense(head_size)(ffn_output) 
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

    x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    curve_emb = tf.keras.layers.Dense(32, activation="relu")(x)
    
    input_features = tf.keras.layers.Input(shape=(input_size_features,), name="features_input")
    feat_emb = tf.keras.layers.Dense(32, activation='relu')(input_features)
    
    merged = tf.keras.layers.Concatenate()([curve_emb, feat_emb])
    z = tf.keras.layers.Dense(32, activation='relu')(merged)
    z = tf.keras.layers.Dropout(0.2)(z)
    outputs = tf.keras.layers.Dense(output_size, activation='softmax')(z)

    model = tf.keras.models.Model(inputs=[input_curve, input_features], outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ====================================================================
# NEURAL NETWORK SETUP
# ====================================================================
class KerasModelWrapper(KerasClassifier):
    pass

# 1. 1D CNN
def create_cnn_model(input_size, output_size, kernel_size_1=5, kernel_size_2=3): 
    inputs = tf.keras.layers.Input(shape=(input_size, 1))
    x = tf.keras.layers.Conv1D(16, kernel_size_1, activation='relu')(inputs)
    x = tf.keras.layers.Conv1D(8, kernel_size_2, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(output_size, activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# 2. LSTM (Bidirectional + Gradient Clipping)
def create_lstm_model(input_size, output_size):
    inputs = tf.keras.layers.Input(shape=(input_size, 1))
    # Wrapped in Bidirectional
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(output_size, activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    # Added clipnorm to prevent exploding gradients
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 2b. LSTM-AE pretrained classifier — reuses the encoder half of the global LSTM
# autoencoder trained in 02_outlier_detection_pipeline.py (outlier detection) as a
# frozen feature extractor, with only a small new Dense head trained for
# classification. One backbone serving both outlier-scoring and classification,
# instead of two separately-trained LSTM stacks.
def load_lstm_ae_clf_model(pretrained_encoder_path, pretrained_scaler_path, input_size, output_size):
    """Loads the encoder + its fitted MinMaxScaler saved by
    lstm_autoencoder_outlier.py's _save_encoder(), freezes the encoder, and stacks a
    classification head on top. Returns (model, scaler) — the scaler must be applied
    to curves before feeding them to `model`, since the encoder was trained on scaled
    (not raw) curves. Returns (None, None) if the files are missing or the encoder's
    expected sequence length doesn't match `input_size` (e.g. a different curve_type
    or AE_DOWNSAMPLE_FACTOR than what the encoder was trained on)."""
    if not (os.path.exists(pretrained_encoder_path) and os.path.exists(pretrained_scaler_path)):
        return None, None

    encoder = tf.keras.models.load_model(pretrained_encoder_path, compile=False)
    encoder_seq_len = encoder.input_shape[1]
    if encoder_seq_len != input_size:
        print(f"     [!] lstm_ae_clf: encoder expects {encoder_seq_len} timesteps, "
              f"got {input_size}. Skipping (encoder/data mismatch).")
        return None, None

    scaler = joblib.load(pretrained_scaler_path)
    encoder.trainable = False  # freeze — only the new Dense head is trained

    # Explicit names: the loaded encoder already has an auto-named "dropout" layer
    # (from its own architecture) — an unnamed new Dropout here would collide with it
    # when both end up in the same combined functional graph.
    x = tf.keras.layers.Dense(32, activation='relu', name="lstm_ae_clf_head_dense1")(encoder.output)
    x = tf.keras.layers.Dropout(0.2, name="lstm_ae_clf_head_dropout")(x)
    out = tf.keras.layers.Dense(output_size, activation='softmax', name="lstm_ae_clf_head_out")(x)
    model = tf.keras.models.Model(inputs=encoder.input, outputs=out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model, scaler

# 3. GRU (Bidirectional + Gradient Clipping)
def create_gru_model(input_size, output_size):
    inputs = tf.keras.layers.Input(shape=(input_size, 1))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, return_sequences=True))(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(16))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(output_size, activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# 4. Simple RNN (Bidirectional + Gradient Clipping)
def create_rnn_model(input_size, output_size):
    inputs = tf.keras.layers.Input(shape=(input_size, 1))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(32, return_sequences=True))(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(16))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(output_size, activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# 5. Transformer
def create_transformer_model(input_size, output_size, head_size=32, num_heads=2, ff_dim=32, num_blocks=2, dropout=0.1):
    inputs = tf.keras.layers.Input(shape=(input_size, 1))
    
    # --- THE DOWNSAMPLING STEM ---
    # Shrinks sequence from 600 -> ~150 while projecting to 32 features
    x = tf.keras.layers.Conv1D(filters=head_size, kernel_size=5, strides=2, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, padding="same")(x)
    
    # Calculate the new sequence length mathematically for the Positional Embedding
    new_seq_len = x.shape[1] 
    
    # Positional Embedding
    positions = tf.range(start=0, limit=x.shape[1], delta=1)
    pos_embedding = tf.keras.layers.Embedding(input_dim=new_seq_len, output_dim=head_size)(positions)
    x = x + pos_embedding 
    
    # --- STANDARD TRANSFORMER BLOCKS ---
    for _ in range(num_blocks):
        attn_output = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
        attn_output = tf.keras.layers.Dropout(dropout)(attn_output)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

        ffn_output = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
        ffn_output = tf.keras.layers.Dropout(dropout)(ffn_output)
        ffn_output = tf.keras.layers.Dense(head_size)(ffn_output) 
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

    x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(output_size, activation="softmax")(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ====================================================================
# MODULE 1: MODEL EVALUATION FUNCTION (WITH PROBABILITIES)
# ====================================================================
def _remap_global_splits(global_splits, mask, valid_mask=None):
    """
    Convert (train_idx, test_idx) pairs defined over the FULL pre-filter index
    space (0..N-1, matching X_curves/y_encoded as passed in) into positional
    indices over the array after `mask` (and optionally `valid_mask`, applied
    on top of `mask`) has been used to slice it.
    """
    kept_global = np.where(mask)[0]
    if valid_mask is not None:
        kept_global = kept_global[valid_mask]
    pos_lookup = {g: i for i, g in enumerate(kept_global)}

    remapped = []
    for train_idx, test_idx in global_splits:
        local_train = np.array([pos_lookup[g] for g in train_idx if g in pos_lookup], dtype=int)
        local_test = np.array([pos_lookup[g] for g in test_idx if g in pos_lookup], dtype=int)
        remapped.append((local_train, local_test))
    return remapped


# Maps evaluate_outlier_filters internal model keys to the canonical names used
# when saving .keras files for 07_attribution_vis_all.
_XAI_SAVE_NAME = {
    'cnn': 'cnn',
    'gru': 'bigru',
    'transformer': 'transformer',
    'cnn_lf': 'cnn_lf',
    'gru_lf': 'bigru_lf',
    'trans_lf': 'transformer_lf',
    'cnn_gru_dual': 'cnn_gru_dual',
    'cnn_trans_dual': 'cnn_trans_dual',
    'lstm_ae_clf': 'lstm_ae_clf',
    **{name: name for name in model_utils_gated._ALL_FACTORIES},
}


def evaluate_outlier_filters(
    X_curves, features_df, y_encoded, outlier_filters, dataset_name, mode_name,
    cached_results=None, models=["cnn", "cnn_lf"], n_splits=1,
    checkpoint_fn=None, KFS=None, rerun_models=[], cv_splits=None,
    save_model_dir=None, save_model_curve_type="ori_curve",
    pretrained_encoder_path=None, pretrained_scaler_path=None,
    coords=None, well_ids=None, k_neighbors=8,
):
    """Train and evaluate models across outlier filters.

    When save_model_dir is set, the trained Keras model from the first fold of
    the None (baseline) filter is saved to disk so attribution_vis_all can load
    it without a separate run.

    pretrained_encoder_path/pretrained_scaler_path: only used by the "lstm_ae_clf"
    model — paths to the encoder/scaler saved by 02_outlier_detection_pipeline.py's
    global LSTM autoencoder for this exact (experiment, curve_type). If either is
    None/missing, "lstm_ae_clf" is skipped for every filter (no from-scratch
    fallback — the model only makes sense paired with its pretrained backbone).

    coords/well_ids: (N, 2) pixel [row, col] and (N,) well id per sample, aligned to
    X_curves. Only used by "cnn_gru_dual_cosine_recon"/"cnn_gru_dual_attn_recon" (see
    build_neighbor_curve_stack) to find each pixel's spatial neighbours within its own
    well. If either is None, both models are skipped (with a warning) regardless of
    whether they're in `models` — every other model is unaffected.
    """
    if save_model_dir is not None:
        Path(save_model_dir).mkdir(parents=True, exist_ok=True)

    X_FFI_full = X_curves[:, [-1]]

    # Safely extract manual features if KFS is provided
    if KFS is not None:
        X_manual_full = features_df[KFS].values
        # Replace inf/-inf/NaN generated by feature extraction edge cases
        X_manual_full = np.nan_to_num(X_manual_full, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        X_manual_full = None
    
    results_dict = cached_results.copy() if cached_results is not None else {}
    total_filters = len(outlier_filters)
    models = [m.lower() for m in models]

    model_key_map = {
        "cnn": ("y_preds_AC_", "y_probs_AC_", "classes_AC_"),
        "lstm": ("y_preds_AC_lstm_", "y_probs_AC_lstm_", "classes_AC_lstm_"),
        "gru": ("y_preds_AC_gru_", "y_probs_AC_gru_", "classes_AC_gru_"),
        "rnn": ("y_preds_AC_rnn_", "y_probs_AC_rnn_", "classes_AC_rnn_"),
        "transformer": ("y_preds_AC_trans_", "y_probs_AC_trans_", "classes_AC_trans_"),
        "rf": ("y_preds_AC_rf_", "y_probs_AC_rf_", "classes_AC_rf_"),
        "knn": ("y_preds_AC_kNN_", "y_probs_AC_kNN_", "classes_AC_kNN_"),
        "ffi": ("y_preds_FFI_", "y_probs_FFI_", "classes_FFI_"),
        "cnn_lf": ("y_preds_AC_cnn_lf_", "y_probs_AC_cnn_lf_", "classes_AC_cnn_lf_"),
        "lstm_lf": ("y_preds_AC_lstm_lf_", "y_probs_AC_lstm_lf_", "classes_AC_lstm_lf_"),
        "trans_lf": ("y_preds_AC_trans_lf_", "y_probs_AC_trans_lf_", "classes_AC_trans_lf_"),
        "gru_lf": ("y_preds_AC_gru_lf_", "y_probs_AC_gru_lf_", "classes_AC_gru_lf_"),
        "cnn_gru_dual": ("y_preds_AC_cnn_gru_dual_", "y_probs_AC_cnn_gru_dual_", "classes_AC_cnn_gru_dual_"),
        "cnn_trans_dual": ("y_preds_AC_cnn_trans_dual_", "y_probs_AC_cnn_trans_dual_", "classes_AC_cnn_trans_dual_"),
        "lstm_ae_clf": ("y_preds_AC_lstm_ae_clf_", "y_probs_AC_lstm_ae_clf_", "classes_AC_lstm_ae_clf_"),
        "cnn_gru_dual_cosine_recon": ("y_preds_AC_cnn_gru_dual_cosine_recon_", "y_probs_AC_cnn_gru_dual_cosine_recon_", "classes_AC_cnn_gru_dual_cosine_recon_"),
        "cnn_gru_dual_attn_recon": ("y_preds_AC_cnn_gru_dual_attn_recon_", "y_probs_AC_cnn_gru_dual_attn_recon_", "classes_AC_cnn_gru_dual_attn_recon_"),
        # 8 gated dual-branch fusion models (model_utils_gated.py).
        **{name: (f"y_preds_AC_{name}_", f"y_probs_AC_{name}_", f"classes_AC_{name}_")
           for name in model_utils_gated._ALL_FACTORIES},
    }

    model_print_map = {
        "cnn": "CNN (ACA)", "lstm": "LSTM (ACA)", "gru": "GRU (ACA)",
        "rnn": "RNN (ACA)", "transformer": "Trans (ACA)", "rf": "RF (ACA)",
        "knn": "KNN (ACA)", "ffi": "LR (FFI)",
        "cnn_lf": "CNN LF", "lstm_lf": "LSTM LF", "trans_lf": "Trans LF", "gru_lf": "GRU LF",
        "cnn_gru_dual": "CNN+GRU Dual", "cnn_trans_dual": "CNN+Tr Dual",
        "lstm_ae_clf": "LSTM-AE Clf",
        "cnn_gru_dual_cosine_recon": "CNN+GRU CosRecon", "cnn_gru_dual_attn_recon": "CNN+GRU AttnRecon",
        "cnn_gru_gate": "CNN+GRU Gate", "cnn_gru_hadamard": "CNN+GRU Hadamard",
        "cnn_gru_crossattn": "CNN+GRU CoAttn", "cnn_gru_film": "CNN+GRU FiLM",
        "cnn_trans_gate": "CNN+Tr Gate", "cnn_trans_hadamard": "CNN+Tr Hadamard",
        "cnn_trans_crossattn": "CNN+Tr CoAttn", "cnn_trans_film": "CNN+Tr FiLM",
    }

    _SPATIAL_RECON_MODELS = ("cnn_gru_dual_cosine_recon", "cnn_gru_dual_attn_recon")

    for idx, f in enumerate(outlier_filters):
        filter_name = f if f else 'None (Baseline)'
        filter_pct = ((idx + 1) / total_filters) * 100
        print(f"  -> Testing Filter [{idx+1}/{total_filters} | {filter_pct:.1f}%]: {filter_name}")
        
        res_entry = results_dict.get(f, {})
        
        if f is None:
            mask = np.ones(len(y_encoded), dtype=bool)
        elif f in features_df.columns:
            mask = (features_df[f] == 1).fillna(False).values
        else:
            print(f"     [Warning] {f} not found in dataset. Skipping.")
            continue

        X_AC = np.nan_to_num(X_curves[mask], nan=0.0, posinf=0.0, neginf=0.0)
        X_FFI = np.nan_to_num(X_FFI_full[mask], nan=0.0, posinf=0.0, neginf=0.0)
        y_true = y_encoded[mask]

        X_manual = X_manual_full[mask] if X_manual_full is not None else None
        coords_m = coords[mask] if coords is not None else None
        well_ids_m = well_ids[mask] if well_ids is not None else None

        unique_classes, class_counts = np.unique(y_true, return_counts=True)
        rare_classes = unique_classes[class_counts < 2]

        valid_class_mask = None
        if len(rare_classes) > 0:
            valid_class_mask = ~np.isin(y_true, rare_classes)
            X_AC = X_AC[valid_class_mask]
            X_FFI = X_FFI[valid_class_mask]
            y_true = y_true[valid_class_mask]
            if X_manual is not None:
                X_manual = X_manual[valid_class_mask]
            if coords_m is not None:
                coords_m = coords_m[valid_class_mask]
                well_ids_m = well_ids_m[valid_class_mask]

        n_classes = len(np.unique(y_true))


        # # THISSS ############################################################
        # # Curve Normalisations
        # curve_mins = np.min(X_AC, axis=1, keepdims=True)
        # curve_maxs = np.max(X_AC, axis=1, keepdims=True)
        # X_AC = (X_AC - curve_mins) / (curve_maxs - curve_mins + 1e-8)
        ######################################################################

        if n_classes < 2 or len(y_true) < 2 * n_classes:
            print(f"     [Warning] Insufficient classes or samples. Skipping.")
            continue

        if cv_splits is not None:
            splits = _remap_global_splits(cv_splits, mask, valid_class_mask)
            splits = [(tr, te) for tr, te in splits if len(tr) > 0 and len(te) > 0]
            if not splits:
                print(f"     [Warning] No samples remain for this filter under the given CV splits. Skipping.")
                continue
        else:
            calculated_test_size = max(int(len(y_true) * 0.10), n_classes)

            if n_splits == 1:
                splitter = StratifiedShuffleSplit(n_splits=1, test_size=calculated_test_size, random_state=0)
            else:
                min_class_count = np.min(class_counts[~np.isin(unique_classes, rare_classes)])
                actual_splits = min(n_splits, min_class_count)
                splitter = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=0)

            splits = list(splitter.split(X_AC, y_true))
        
        if "y_trues_" not in res_entry:
            res_entry["y_trues_"] = [y_true[test_index] for _, test_index in splits]
            res_entry["mask_count"] = int(np.sum(mask))

        # --- Spatial neighbour reconstruction setup ---
        # The actual (N, k+1, T) neighbour stack is built LAZILY, inside the per-model
        # loop below, right before the first of these two models that needs it -- NOT
        # eagerly here. It's k+1x the size of the raw curve array (float32, still cast
        # down from the source curves' likely float64) and for large combined pools
        # (e.g. 04_cross_dataset_training.py's LOFO pools) this is multiple GB; building
        # it up front meant it sat in memory for the *entire* filter, including while
        # training cnn/gru/transformer/etc, which don't use it at all -- a real OOM risk
        # that's now avoided by building it only when reached, and freeing it (`_free_spatial_recon`
        # below) once both spatial models for this filter are done.
        _wanted_recon = [m for m in models if m in _SPATIAL_RECON_MODELS]
        _recon_unavailable = bool(_wanted_recon) and (coords_m is None or well_ids_m is None)
        if _recon_unavailable:
            print(f"     [SKIP] {', '.join(_wanted_recon)}: no coords/well_ids provided "
                  f"(pass coords=/well_ids= to evaluate_outlier_filters). Skipping for this filter.")
        X_AC_cosine_recon, X_AC_stack = None, None
        _recon_models_left = set(_wanted_recon)  # shrinks as each is reached below

        def _free_spatial_recon():
            nonlocal X_AC_cosine_recon, X_AC_stack
            X_AC_cosine_recon, X_AC_stack = None, None
            gc.collect()

        for m in models:
            if m not in model_key_map: continue
            if "lf" in m and X_manual is None:
                print(f"     [Error] Model {m} requires KFS features, but KFS was not provided.")
                continue
            if m == "lstm_ae_clf":
                _has_files = (pretrained_encoder_path and pretrained_scaler_path
                             and os.path.exists(pretrained_encoder_path)
                             and os.path.exists(pretrained_scaler_path))
                if not _has_files:
                    print(f"     [SKIP] lstm_ae_clf: no pretrained encoder found at "
                          f"{pretrained_encoder_path} (run 02's global LSTM autoencoder first). "
                          f"Not falling back to training a fresh backbone.")
                    continue
                # One-time shape check (timesteps is the same across all folds for this
                # filter) so a mismatch skips the whole model cleanly, rather than
                # surfacing as repeated per-fold failures inside the training loop below.
                _probe = tf.keras.models.load_model(pretrained_encoder_path, compile=False)
                _encoder_seq_len = _probe.input_shape[1]
                del _probe
                tf.keras.backend.clear_session()
                if _encoder_seq_len != X_AC.shape[1]:
                    print(f"     [SKIP] lstm_ae_clf: encoder expects {_encoder_seq_len} timesteps, "
                          f"this filter's curves have {X_AC.shape[1]}. Skipping.")
                    continue

            if m in _SPATIAL_RECON_MODELS and _recon_unavailable:
                continue  # already warned above (no coords/well_ids)

            preds_key, probs_key, classes_key = model_key_map[m]
            print_name = f"{model_print_map[m]:<11}"

            # --- CHECK CACHE ---
            if (preds_key in res_entry) and (m not in rerun_models):
                fold_accs = [accuracy_score(yt, yp) for yt, yp in zip(res_entry["y_trues_"], res_entry[preds_key])]
                acc = np.mean(fold_accs) * 100
                std = np.std(fold_accs) * 100

                print(f"     [CACHE HIT] {m.upper()} cached result found. Skipping training.")
                print(f"     [+] {mode_name}-{dataset_name}-{filter_name[:30]} | {print_name} | {acc:5.2f}% ± {std:5.2f}% | Duration: Cached")
                if m in _recon_models_left:
                    _recon_models_left.discard(m)
                    if not _recon_models_left:
                        _free_spatial_recon()
                continue

            # Lazily build the (N, k+1, T) neighbour stack the first time it's actually
            # needed for training (not on a cache hit, see above) -- see the comment where
            # _recon_models_left is defined for why this isn't done eagerly.
            if m in _SPATIAL_RECON_MODELS and X_AC_cosine_recon is None and X_AC_stack is None:
                neighbor_stack = build_neighbor_curve_stack(
                    X_AC.astype(np.float32, copy=False), coords_m, well_ids_m, k=k_neighbors)
                if "cnn_gru_dual_cosine_recon" in _wanted_recon:
                    X_AC_cosine_recon = reconstruct_curves_cosine(neighbor_stack)
                if "cnn_gru_dual_attn_recon" in _wanted_recon:
                    X_AC_stack = neighbor_stack
                else:
                    del neighbor_stack

            # --- TRAIN NEW MODEL ---
            preds, probs, classes_list = [], [], []
            start_time = time.perf_counter()
            _lstm_ae_failed = False

            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                if m == 'ffi':
                    X_train_curve, X_test_curve = X_FFI[train_idx], X_FFI[test_idx]
                elif m == 'cnn_gru_dual_cosine_recon':
                    X_train_curve, X_test_curve = X_AC_cosine_recon[train_idx], X_AC_cosine_recon[test_idx]
                elif m == 'cnn_gru_dual_attn_recon':
                    # (n, k+1, T) -- same axis-0 indexing as every other model's (n, T) curve
                    # array, just with an extra trailing "neighbour" dimension along for the ride.
                    X_train_curve, X_test_curve = X_AC_stack[train_idx], X_AC_stack[test_idx]
                else:
                    X_train_curve, X_test_curve = X_AC[train_idx], X_AC[test_idx]
                y_train = y_true[train_idx]

                # Setup Manual Features (Scaled strictly on train fold)
                if X_manual is not None:
                    scaler = StandardScaler()
                    X_train_man = scaler.fit_transform(X_manual[train_idx])
                    X_test_man = scaler.transform(X_manual[test_idx])

                # Whether to save this model for XAI use (first fold only, all filters)
                _do_xai_save = (save_model_dir is not None and fold_idx == 0
                                and m in _XAI_SAVE_NAME)

                # Stratified validation split for EarlyStopping/ReduceLROnPlateau, shared
                # across all Keras branches below (rf/knn/ffi don't use it — not epoch-based).
                # Avoids Keras's validation_split, which takes a trailing slice of the array
                # (not stratified) and could miss whole classes depending on row ordering.
                # Falls back to no validation split if a class is too sparse in this fold.
                _val_split_ok = False
                if m not in ("rf", "knn", "ffi"):
                    try:
                        _tr_sub, _val_sub = train_test_split(
                            np.arange(len(y_train)), test_size=0.1, stratify=y_train, random_state=0)
                        y_train_fit = y_train[_tr_sub]
                        y_val = y_train[_val_sub]
                        X_train_curve_fit = X_train_curve[_tr_sub]
                        X_val_curve = X_train_curve[_val_sub]
                        if X_manual is not None:
                            X_train_man_fit = X_train_man[_tr_sub]
                            X_val_man = X_train_man[_val_sub]
                        _val_split_ok = True
                    except ValueError:
                        pass
                if not _val_split_ok:
                    y_train_fit = y_train
                    X_train_curve_fit = X_train_curve
                    if X_manual is not None:
                        X_train_man_fit = X_train_man

                # EarlyStopping patience > ReduceLROnPlateau patience so LR gets a chance
                # to drop before training stops; LR scheduling reduces seed-to-seed variance
                # by preventing different seeds from getting stuck at a poor LR for the whole run.
                _fit_callbacks = [
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=30, min_lr=1e-5),
                ] if _val_split_ok else []

                # Train Standard vs. Late Fusion models
                if m in ["cnn_lf", "lstm_lf", "trans_lf", "gru_lf"]:
                    tf.keras.backend.clear_session()
                    if m == "cnn_lf":
                        model = create_cnn_lf_model(X_train_curve.shape[1], X_train_man.shape[1], n_classes)
                        epochs = 1000
                    elif m == "lstm_lf":
                        model = create_lstm_lf_model(X_train_curve.shape[1], X_train_man.shape[1], n_classes)
                        epochs = 500
                    elif m == "trans_lf":
                        model = create_transformer_lf_model(X_train_curve.shape[1], X_train_man.shape[1], n_classes)
                        epochs = 500
                    elif m == "gru_lf":
                        model = create_gru_lf_model(X_train_curve.shape[1], X_train_man.shape[1], n_classes)
                        epochs = 500

                    if _val_split_ok:
                        model.fit([X_train_curve_fit, X_train_man_fit], y_train_fit,
                                 validation_data=([X_val_curve, X_val_man], y_val),
                                 epochs=epochs, batch_size=512, shuffle=True, verbose=0,
                                 callbacks=_fit_callbacks)
                    else:
                        model.fit([X_train_curve, X_train_man], y_train, epochs=epochs, batch_size=512, shuffle=True, verbose=0)

                    if _do_xai_save:
                        _xai_path = Path(save_model_dir) / f"{_XAI_SAVE_NAME[m]}_{f}_{save_model_curve_type}_model.keras"
                        model.save(_xai_path)
                        print(f"     [XAI] Saved {_XAI_SAVE_NAME[m]} -> {_xai_path}")

                    prob = model.predict([X_test_curve, X_test_man], verbose=0)
                    pred = np.argmax(prob, axis=1)
                    cls = np.unique(y_encoded)

                    preds.append(pred)
                    probs.append(prob)
                    classes_list.append(cls)

                elif m in ["cnn_gru_dual", "cnn_trans_dual", "cnn_gru_dual_cosine_recon"]:
                    tf.keras.backend.clear_session()

                    if m == "cnn_gru_dual":
                        model = create_cnn_gru_dual_model(X_train_curve.shape[1], n_classes)
                        epochs = 500
                    elif m == "cnn_trans_dual":
                        model = create_cnn_transformer_dual_model(X_train_curve.shape[1], n_classes)
                        epochs = 500
                    elif m == "cnn_gru_dual_cosine_recon":
                        # Same architecture as cnn_gru_dual -- only the input curve differs
                        # (X_train_curve here is the cosine-similarity-reconstructed curve,
                        # not the raw per-pixel one; see reconstruct_curves_cosine above).
                        model = create_cnn_gru_dual_model(X_train_curve.shape[1], n_classes)
                        epochs = 500

                    # Notice we only pass X_train_curve here, not a list of inputs!
                    if _val_split_ok:
                        model.fit(X_train_curve_fit, y_train_fit,
                                 validation_data=(X_val_curve, y_val),
                                 epochs=epochs, batch_size=512, shuffle=True, verbose=0,
                                 callbacks=_fit_callbacks)
                    else:
                        model.fit(X_train_curve, y_train, epochs=epochs, batch_size=512, shuffle=True, verbose=0)

                    if _do_xai_save:
                        _xai_path = Path(save_model_dir) / f"{_XAI_SAVE_NAME[m]}_{f}_{save_model_curve_type}_model.keras"
                        model.save(_xai_path)
                        print(f"     [XAI] Saved {_XAI_SAVE_NAME[m]} -> {_xai_path}")

                    prob = model.predict(X_test_curve, verbose=0)
                    pred = np.argmax(prob, axis=1)
                    cls = np.unique(y_encoded)

                    preds.append(pred)
                    probs.append(prob)
                    classes_list.append(cls)

                elif m == "cnn_gru_dual_attn_recon":
                    # X_train_curve/X_test_curve here are (n, k+1, T) neighbour stacks, not
                    # (n, T) curves -- create_cnn_gru_dual_attn_recon_model takes that stack
                    # directly and learns the reconstruction + classifier jointly. No XAI
                    # save (07_attribution_vis_all.py's saliency pipeline assumes a plain
                    # (T,) curve input; out of scope here).
                    tf.keras.backend.clear_session()
                    model = create_cnn_gru_dual_attn_recon_model(
                        X_train_curve.shape[1], X_train_curve.shape[2], n_classes)
                    epochs = 500

                    if _val_split_ok:
                        model.fit(X_train_curve_fit, y_train_fit,
                                 validation_data=(X_val_curve, y_val),
                                 epochs=epochs, batch_size=512, shuffle=True, verbose=0,
                                 callbacks=_fit_callbacks)
                    else:
                        model.fit(X_train_curve, y_train, epochs=epochs, batch_size=512, shuffle=True, verbose=0)

                    prob = model.predict(X_test_curve, verbose=0)
                    pred = np.argmax(prob, axis=1)
                    cls = np.unique(y_encoded)

                    preds.append(pred)
                    probs.append(prob)
                    classes_list.append(cls)

                elif m in model_utils_gated._ALL_FACTORIES:
                    # 8 gated CNN+(GRU|Transformer) dual-branch fusion models — same
                    # single-curve-input, no-manual-features shape as cnn_gru_dual /
                    # cnn_trans_dual above, just dispatched through the factory dict
                    # instead of named functions (model_utils_gated.py).
                    tf.keras.backend.clear_session()

                    model = model_utils_gated._ALL_FACTORIES[m](X_train_curve.shape[1], n_classes)
                    epochs = 500

                    if _val_split_ok:
                        model.fit(X_train_curve_fit, y_train_fit,
                                 validation_data=(X_val_curve, y_val),
                                 epochs=epochs, batch_size=512, shuffle=True, verbose=0,
                                 callbacks=_fit_callbacks)
                    else:
                        model.fit(X_train_curve, y_train, epochs=epochs, batch_size=512, shuffle=True, verbose=0)

                    if _do_xai_save:
                        _xai_path = Path(save_model_dir) / f"{_XAI_SAVE_NAME[m]}_{f}_{save_model_curve_type}_model.keras"
                        model.save(_xai_path)
                        print(f"     [XAI] Saved {_XAI_SAVE_NAME[m]} -> {_xai_path}")

                    prob = model.predict(X_test_curve, verbose=0)
                    pred = np.argmax(prob, axis=1)
                    cls = np.unique(y_encoded)

                    preds.append(pred)
                    probs.append(prob)
                    classes_list.append(cls)

                elif m == "lstm_ae_clf":
                    tf.keras.backend.clear_session()

                    # Shape compatibility already verified once before the fold loop
                    # above, so this load is expected to succeed here — but pretrained
                    # files can still fail to load (corrupt/incompatible save format,
                    # version-mismatched scaler pickle, etc.), so guard it anyway rather
                    # than crashing the whole run. Named ae_scaler (not `scaler`) to
                    # avoid shadowing the StandardScaler already assigned to `scaler`
                    # above for X_manual, in case anything downstream ever needs to
                    # refer back to it within the same fold.
                    try:
                        model, ae_scaler = load_lstm_ae_clf_model(
                            pretrained_encoder_path, pretrained_scaler_path,
                            X_train_curve.shape[1], n_classes)
                    except Exception as e:
                        model, ae_scaler = None, None
                        print(f"     [SKIP] lstm_ae_clf: failed to load pretrained encoder/scaler ({e}).")

                    if model is None or ae_scaler is None:
                        _lstm_ae_failed = True
                        break
                    epochs = 500

                    # Encoder was trained on MinMax-scaled curves (see _save_encoder in
                    # lstm_autoencoder_outlier.py) — apply the same fitted scaler here so
                    # the frozen pretrained weights see in-distribution input. ae_scaler
                    # expects 2D (n_samples, n_timesteps); squeeze/restore the channel dim.
                    def _scale(x):
                        return ae_scaler.transform(x.squeeze(-1))[..., None]

                    X_train_curve_scaled = _scale(X_train_curve_fit if _val_split_ok else X_train_curve)
                    X_test_curve_scaled = _scale(X_test_curve)

                    if _val_split_ok:
                        model.fit(X_train_curve_scaled, y_train_fit,
                                 validation_data=(_scale(X_val_curve), y_val),
                                 epochs=epochs, batch_size=512, shuffle=True, verbose=0,
                                 callbacks=_fit_callbacks)
                    else:
                        model.fit(X_train_curve_scaled, y_train, epochs=epochs, batch_size=512, shuffle=True, verbose=0)

                    if _do_xai_save:
                        _xai_path = Path(save_model_dir) / f"{_XAI_SAVE_NAME[m]}_{f}_{save_model_curve_type}_model.keras"
                        model.save(_xai_path)
                        print(f"     [XAI] Saved {_XAI_SAVE_NAME[m]} -> {_xai_path}")

                    prob = model.predict(X_test_curve_scaled, verbose=0)
                    pred = np.argmax(prob, axis=1)
                    cls = np.unique(y_encoded)

                    preds.append(pred)
                    probs.append(prob)
                    classes_list.append(cls)

                else:
                    # Standard 1D Models (scikeras/sklearn)
                    if m == "cnn": clf = KerasModelWrapper(model=create_cnn_model, model__input_size=X_train_curve.shape[1], model__output_size=n_classes, epochs=1000, batch_size=512, shuffle=True, verbose=False, random_state=0)
                    elif m == "lstm": clf = KerasModelWrapper(model=create_lstm_model, model__input_size=X_train_curve.shape[1], model__output_size=n_classes, epochs=500, batch_size=512, shuffle=True, verbose=False, random_state=0)
                    elif m == "gru": clf = KerasModelWrapper(model=create_gru_model, model__input_size=X_train_curve.shape[1], model__output_size=n_classes, epochs=500, batch_size=512, shuffle=True, verbose=False, random_state=0)
                    elif m == "rnn": clf = KerasModelWrapper(model=create_rnn_model, model__input_size=X_train_curve.shape[1], model__output_size=n_classes, epochs=500, batch_size=512, shuffle=True, verbose=False, random_state=0)
                    elif m == "transformer": clf = KerasModelWrapper(model=create_transformer_model, model__input_size=X_train_curve.shape[1], model__output_size=n_classes, epochs=500, batch_size=512, shuffle=True, verbose=False, random_state=0)
                    elif m == "rf": clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
                    elif m == "knn": clf = KNeighborsClassifier(n_neighbors=10)
                    elif m == "ffi": clf = LogisticRegression(max_iter=1000)
                    else:
                        raise ValueError(f"Model '{m}' is not properly defined in the training loop.")

                    if _val_split_ok:
                        clf.fit(X_train_curve_fit, y_train_fit,
                               validation_data=(X_val_curve, y_val),
                               callbacks=_fit_callbacks)
                    else:
                        clf.fit(X_train_curve, y_train)

                    if _do_xai_save and m in ['cnn', 'lstm', 'gru', 'rnn', 'transformer']:
                        _xai_path = Path(save_model_dir) / f"{_XAI_SAVE_NAME[m]}_{f}_{save_model_curve_type}_model.keras"
                        clf.model_.save(_xai_path)
                        print(f"     [XAI] Saved {_XAI_SAVE_NAME[m]} -> {_xai_path}")

                    preds.append(clf.predict(X_test_curve))
                    probs.append(clf.predict_proba(X_test_curve))
                    classes_list.append(clf.classes_)

                    if m in ["cnn", "lstm", "gru", "rnn", "transformer"]:
                        tf.keras.backend.clear_session()

            if _lstm_ae_failed:
                continue

            end_time = time.perf_counter()
            duration = end_time - start_time
            formatted_time = time.strftime("%H:%M:%S", time.gmtime(int(duration)))
            
            res_entry[preds_key] = preds
            res_entry[probs_key] = probs
            res_entry[classes_key] = classes_list
            
            results_dict[f] = res_entry
            if checkpoint_fn is not None:
                checkpoint_fn(results_dict)
                
            fold_accs = [accuracy_score(yt, yp) for yt, yp in zip(res_entry["y_trues_"], preds)]
            acc = np.mean(fold_accs) * 100
            std = np.std(fold_accs) * 100

            print(f"     [+] {mode_name}-{dataset_name}-{filter_name[:30]} | {print_name} | {acc:5.2f}% ± {std:5.2f}% | Duration: {formatted_time}")

            if m in _recon_models_left:
                _recon_models_left.discard(m)
                if not _recon_models_left:
                    _free_spatial_recon()

        results_dict[f] = res_entry

    return results_dict


# ====================================================================
# MODULE 2: VISUALIZATION FUNCTIONS
# ====================================================================
def plot_ml_results(results_dict, outlier_filters, dataset_name, mode_name, total_count, save_prefix=None):
    # Local import to avoid a circular import (config imports from this module).
    import config

    present_filters = [f for f in outlier_filters if f in results_dict]
    filter_labels = [str(f) if f is not None else "No Filter" for f in present_filters]

    # Reuse the shared colour palette so each outlier filter keeps the same
    # colour across every bar chart in the report.
    filter_palette = config.get_palette(present_filters, config.FILTER_COLORS)
    colors = [filter_palette[f] for f in present_filters]

    # Dynamically determine which models were evaluated
    sample_res = next((results_dict[f] for f in outlier_filters if f in results_dict), None)
    
    method_info = []
    if sample_res:
        if 'y_preds_FFI_' in sample_res:
            method_info.append(('Logistic Regression (FFI)', 'y_preds_FFI_'))
        if 'y_preds_AC_kNN_' in sample_res:
            method_info.append(('kNN (ACA)', 'y_preds_AC_kNN_'))
        if 'y_preds_AC_rf_' in sample_res:
            method_info.append(('Random Forest (ACA)', 'y_preds_AC_rf_'))
        if 'y_preds_AC_trans_' in sample_res:
            method_info.append(('Transformer (ACA)', 'y_preds_AC_trans_'))
        if 'y_preds_AC_rnn_' in sample_res:
            method_info.append(('Simple RNN (ACA)', 'y_preds_AC_rnn_'))
        if 'y_preds_AC_gru_' in sample_res:
            method_info.append(('Gated Recurrent Unit (ACA)', 'y_preds_AC_gru_'))
        if 'y_preds_AC_lstm_' in sample_res:
            method_info.append(('Long Short-Term Memory (ACA)', 'y_preds_AC_lstm_'))
        if 'y_preds_AC_' in sample_res:
            method_info.append(('Convolutional Neural Network (ACA)', 'y_preds_AC_'))
        if 'y_preds_AC_cnn_lf_' in sample_res:
            method_info.append(('CNN Late Fusion', 'y_preds_AC_cnn_lf_'))
        if 'y_preds_AC_lstm_lf_' in sample_res:
            method_info.append(('LSTM Late Fusion', 'y_preds_AC_lstm_lf_'))
        if 'y_preds_AC_gru_lf_' in sample_res:
            method_info.append(('GRU Late Fusion', 'y_preds_AC_gru_lf_'))
        if 'y_preds_AC_trans_lf_' in sample_res:
            method_info.append(('Transformer Late Fusion', 'y_preds_AC_trans_lf_'))
        if 'y_preds_AC_cnn_gru_dual_' in sample_res:
            method_info.append(('CNN + GRU Dual', 'y_preds_AC_cnn_gru_dual_'))
        if 'y_preds_AC_cnn_trans_dual_' in sample_res:
            method_info.append(('CNN + Transformer Dual', 'y_preds_AC_cnn_trans_dual_'))
        if 'y_preds_AC_lstm_ae_clf_' in sample_res:
            method_info.append(('LSTM-AE Pretrained Classifier', 'y_preds_AC_lstm_ae_clf_'))
        for _gated_name, _gated_print in [
            ("cnn_gru_gate", "CNN+GRU Gate"), ("cnn_gru_hadamard", "CNN+GRU Hadamard"),
            ("cnn_gru_crossattn", "CNN+GRU CoAttn"), ("cnn_gru_film", "CNN+GRU FiLM"),
            ("cnn_trans_gate", "CNN+Tr Gate"), ("cnn_trans_hadamard", "CNN+Tr Hadamard"),
            ("cnn_trans_crossattn", "CNN+Tr CoAttn"), ("cnn_trans_film", "CNN+Tr FiLM"),
        ]:
            _key = f'y_preds_AC_{_gated_name}_'
            if _key in sample_res:
                method_info.append((_gated_print, _key))
        if 'y_preds_AC_gnn_gat_' in sample_res:
            method_info.append(('GNN-GAT (Spatial)', 'y_preds_AC_gnn_gat_'))
        if 'y_preds_AC_gnn_gcn_' in sample_res:
            method_info.append(('GNN-GCN (Spatial)', 'y_preds_AC_gnn_gcn_'))
        if 'y_preds_AC_cnn_gru_dual_cosine_recon_' in sample_res:
            method_info.append(('CNN+GRU Dual (Cosine Recon)', 'y_preds_AC_cnn_gru_dual_cosine_recon_'))
        if 'y_preds_AC_cnn_gru_dual_attn_recon_' in sample_res:
            method_info.append(('CNN+GRU Dual (Attn Recon)', 'y_preds_AC_cnn_gru_dual_attn_recon_'))

    if not method_info:
        print(f"  [Warning] No model data found in results dict to plot for {dataset_name}.")
        return []

    # --- 1. PLOT ACCURACIES ---
    # Adjust figure height based on the number of models actually plotted
    fig_acc, axes = plt.subplots(len(method_info), 1, figsize=(14, 6 * len(method_info)))
    if len(method_info) == 1:
        axes = [axes]
    
    for ax, (title, m_key) in zip(axes, method_info):
        means, stds = [], []
        base_mean = 0

        for f in outlier_filters:
            if f not in results_dict: continue
            res = results_dict[f]
            
            # Check if this model completed for this filter to avoid KeyErrors
            if m_key not in res:
                means.append(0)
                stds.append(0)
                continue

            fold_accs = [accuracy_score(yt, yp) * 100 for yt, yp in zip(res['y_trues_'], res[m_key])]
            if len(fold_accs) > 0:
                m_val = np.mean(fold_accs)
                s_val = np.std(fold_accs)
            else:
                m_val = 0.0
                s_val = 0.0

            means.append(m_val)
            stds.append(s_val)
            
            if f is None:
                base_mean = m_val

        bar_labels = [f'{m:.1f}%' for m in means]
        bars = ax.bar(filter_labels, means, yerr=stds, color=colors, edgecolor='black', alpha=0.8, capsize=5)
        
        ax.axhline(y=base_mean, color='red', linestyle='--', linewidth=2, label=f'Baseline ({base_mean:.1f}%)')
        ax.bar_label(bars, labels=bar_labels, padding=5, fontsize=10, fontweight='bold')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0, 115) 
        ax.set_xticks(range(len(filter_labels)))
        ax.set_xticklabels(filter_labels, rotation=15, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.legend(loc='upper right')

    fig_acc.suptitle(f"Model Accuracies | {mode_name}: {dataset_name}", fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_prefix:
        acc_path = f"{save_prefix}_accuracies.png"
        fig_acc.savefig(acc_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig_acc)

    # --- 2. PLOT DATA COMPOSITION ---
    n_normals = [results_dict[f]['mask_count'] for f in outlier_filters if f in results_dict]
    n_outliers = [total_count - n for n in n_normals]

    fig_comp, ax_comp = plt.subplots(figsize=(12, 6))
    ax_comp.bar(filter_labels, n_normals, color=colors, edgecolor='black', alpha=0.8, label='Normal')
    ax_comp.bar(filter_labels, n_outliers, bottom=n_normals, color='#ffcccc', edgecolor='black', alpha=0.6, label='Outlier')

    for i in range(len(filter_labels)):
        if n_normals[i] > 0:
            ax_comp.text(i, n_normals[i]/2, f'{(n_normals[i]/total_count)*100:.1f}%', ha='center', color='white', fontweight='bold')
        if n_outliers[i] > 0:
            ax_comp.text(i, n_normals[i] + (n_outliers[i]/2), f'{(n_outliers[i]/total_count)*100:.1f}%', ha='center', color='darkred', fontweight='bold')

    ax_comp.set_title(f"Data Composition | {mode_name}: {dataset_name}", fontsize=14, fontweight='bold')
    ax_comp.set_ylabel("Number of Samples")
    ax_comp.set_xticks(range(len(filter_labels)))
    ax_comp.set_xticklabels(filter_labels, rotation=15, ha='right')
    plt.tight_layout()
    
    if save_prefix:
        comp_path = f"{save_prefix}_composition.png"
        fig_comp.savefig(comp_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig_comp)
    
   # --- 3. CONSOLE LEADERBOARD PRINT ---
    print(f"\n  🏆 Top Combinations for {mode_name}: {dataset_name}")
    print("  " + "-"*105)
    
    all_results = []
    for title, m_key in method_info:
        for f in outlier_filters:
            if f not in results_dict: continue
            res = results_dict[f]
            if m_key not in res: continue
            
            fold_accs = [accuracy_score(yt, yp) * 100 for yt, yp in zip(res['y_trues_'], res[m_key])]
    
            if len(fold_accs) > 0:
                mean_acc = np.mean(fold_accs)
                std_acc = np.std(fold_accs)
                filt_name = str(f) if f is not None else "Baseline (None)"
                all_results.append((mean_acc, std_acc, dataset_name, mode_name, title, filt_name))
            
    all_results.sort(key=lambda x: x[0], reverse=True)
    
    for i, (acc, std_acc, d_name, m_name, method, filt) in enumerate(all_results):
        print(f"  {i+1:2d}. {acc:6.2f}% ± {std_acc:5.2f}% | Data: {d_name[:15]:<15} | Model: {method[:20]:<20} | Filter: {filt[:30]}")
    print("  " + "-"*105 + "\n")
    
    return all_results