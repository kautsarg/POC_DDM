import tensorflow as tf

def _build_cnn_gru_dual_branches_mtl(input_tensor, return_branches=False):
    """CNN+GRU dual branches; GRU embedding is 64-dim (vs 32 in model_utils) giving 96-dim fused output."""
    c = tf.keras.layers.Conv1D(16, 5, activation='relu')(input_tensor)
    c = tf.keras.layers.Conv1D(8, 3, activation='relu')(c)
    c = tf.keras.layers.Flatten()(c)
    cnn_emb = tf.keras.layers.Dense(32, activation='relu')(c)

    g = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, return_sequences=True))(input_tensor)
    g = tf.keras.layers.LayerNormalization()(g)
    g = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(16))(g)
    g = tf.keras.layers.Dropout(0.2)(g)
    gru_emb = tf.keras.layers.Dense(64, activation='relu')(g)

    merged = tf.keras.layers.Concatenate()([cnn_emb, gru_emb])
    z = tf.keras.layers.Dense(96, activation='relu')(merged)
    z = tf.keras.layers.Dropout(0.2)(z)
    if return_branches:
        return cnn_emb, gru_emb, z  # (32-dim, 64-dim, 96-dim fused)
    return z


@tf.keras.utils.register_keras_serializable(package="model_utils_mtl")
class _QuerySlice(tf.keras.layers.Layer):
    """Extracts the first timestep (index 0) as the query: (N, k+1, D) → (N, 1, D)."""
    def call(self, x):
        return x[:, 0:1, :]


@tf.keras.utils.register_keras_serializable(package="model_utils_mtl")
class _AttnScores(tf.keras.layers.Layer):
    """Scaled dot-product scores: Q @ K^T / sqrt(d). Needs attn_dim in config for reload."""
    def __init__(self, attn_dim, **kwargs):
        super().__init__(**kwargs)
        self.attn_dim = attn_dim

    def call(self, inputs):
        q, k = inputs
        return tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(self.attn_dim, tf.float32))

    def get_config(self):
        return {**super().get_config(), "attn_dim": self.attn_dim}


@tf.keras.utils.register_keras_serializable(package="model_utils_mtl")
class _WeightedRecon(tf.keras.layers.Layer):
    """Attention-weighted reconstruction: weights @ stack → (N, 1, T)."""
    def call(self, inputs):
        weights, stack = inputs
        return tf.matmul(weights, stack)


def _build_cnn_gru_dual_attn_recon_embedding_mtl(stack_input, T, attn_dim=16, return_branches=False):
    """Shared attention-weighted reconstruction → CNN+GRU embedding.
    stack_input: Keras tensor (N, k+1, T). Returns 96-dim embedding for _supcon_wrap/_dann_wrap.
    When return_branches=True, returns (cnn_emb, gru_emb, fused) for branch SupCon/DANN factories.
    """
    per_curve_encoder = tf.keras.Sequential([
        tf.keras.layers.Reshape((T, 1)),
        tf.keras.layers.Conv1D(16, 5, activation='relu', padding='same'),
        tf.keras.layers.Conv1D(8, 3, activation='relu', padding='same'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(attn_dim, activation='relu'),
    ], name='per_curve_encoder')
    embeddings = tf.keras.layers.TimeDistributed(per_curve_encoder)(stack_input)  # (N, k+1, attn_dim)

    query = _QuerySlice()(embeddings)                                              # (N, 1, attn_dim)
    scores = _AttnScores(attn_dim)([query, embeddings])                           # (N, 1, k+1)
    attn_weights = tf.keras.layers.Softmax(axis=-1, name='attn_weights')(scores)  # (N, 1, k+1)

    reconstructed = _WeightedRecon()([attn_weights, stack_input])                 # (N, 1, T)
    reconstructed = tf.keras.layers.Reshape((T, 1))(reconstructed)               # (N, T, 1)
    return _build_cnn_gru_dual_branches_mtl(reconstructed, return_branches=return_branches)
