import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

REG_SENTINEL = -1.0

MTL_MODEL_KEYS = [
    'cnn_mtl', 'lstm_mtl', 'gru_mtl', 'rnn_mtl', 'transformer_mtl',
    'cnn_gru_dual_mtl', 'cnn_trans_dual_mtl',
    'cnn_gru_dual_cosine_recon_mtl', 'cnn_gru_dual_attn_recon_mtl',
]


# ====================================================================
# KENDALL UNCERTAINTY-WEIGHTED MTL MODEL
# ====================================================================

@tf.keras.utils.register_keras_serializable(package='mtl')
class MTLModel(tf.keras.Model):
    """Shared-backbone model with classification + regression heads.

    Loss = exp(-s_cls)*CE + s_cls + exp(-s_reg)*MSE_masked + s_reg
    where s_cls, s_reg are learnable log-variance scalars (Kendall 2018).
    Regression targets encoded as REG_SENTINEL are masked out of the MSE.
    """

    def __init__(self, *args, reg_sentinel=REG_SENTINEL, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_sentinel = reg_sentinel
        self.log_var_cls = self.add_weight(
            'log_var_cls', shape=(), initializer='zeros', trainable=True)
        self.log_var_reg = self.add_weight(
            'log_var_reg', shape=(), initializer='zeros', trainable=True)

    def _compute_loss(self, y_cls, y_reg, cls_out, reg_out):
        ce = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(y_cls, cls_out))
        mask = tf.cast(tf.not_equal(y_reg, self.reg_sentinel), tf.float32)
        mse = (tf.reduce_sum(mask * tf.square(y_reg - reg_out[:, 0]))
               / (tf.reduce_sum(mask) + 1e-8))
        loss = (tf.exp(-self.log_var_cls) * ce + self.log_var_cls
                + tf.exp(-self.log_var_reg) * mse + self.log_var_reg)
        return loss, ce, mse

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            cls_out, reg_out = self(x, training=True)
            loss, ce, mse = self._compute_loss(
                y_dict['cls_out'], y_dict['reg_out'], cls_out, reg_out)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y_dict['cls_out'], cls_out)
        return ({m.name: m.result() for m in self.metrics}
                | {'loss': loss, 'cls_ce': ce, 'reg_mse': mse})

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        cls_out, reg_out = self(x, training=False)
        loss, ce, mse = self._compute_loss(
            y_dict['cls_out'], y_dict['reg_out'], cls_out, reg_out)
        self.compiled_metrics.update_state(y_dict['cls_out'], cls_out)
        return ({m.name: m.result() for m in self.metrics}
                | {'loss': loss, 'cls_ce': ce, 'reg_mse': mse})

    def get_config(self):
        config = super().get_config()
        config['reg_sentinel'] = self.reg_sentinel
        return config

    @classmethod
    def from_config(cls, config):
        reg_sentinel = config.pop('reg_sentinel', REG_SENTINEL)
        model = super().from_config(config)
        model.reg_sentinel = reg_sentinel
        return model


# ====================================================================
# CONCENTRATION NORMALISATION HELPER
# ====================================================================

def _normalize_concentration(conc_array):
    """Fit StandardScaler on non-sentinel values; transform full array.
    Sentinel entries are preserved as REG_SENTINEL (not scaled).
    Returns (scaled_array, fitted_scaler).
    """
    arr = conc_array.astype(float).copy()
    valid_mask = arr != REG_SENTINEL
    scaler = StandardScaler()
    if valid_mask.sum() > 0:
        arr[valid_mask] = scaler.fit_transform(arr[valid_mask].reshape(-1, 1)).ravel()
    return arr, scaler


def _inverse_normalize_concentration(scaled_array, scaler, sentinel=REG_SENTINEL):
    """Inverse-transform non-sentinel values using a previously fitted scaler."""
    arr = scaled_array.astype(float).copy()
    valid_mask = arr != sentinel
    if valid_mask.sum() > 0 and hasattr(scaler, 'mean_'):
        arr[valid_mask] = scaler.inverse_transform(
            arr[valid_mask].reshape(-1, 1)).ravel()
    return arr


# ====================================================================
# MTL MODEL FACTORIES
# ====================================================================
# Each factory mirrors its classification counterpart exactly up to the
# final Dense softmax, then attaches two heads and wraps in MTLModel.

def _mtl_wrap(inputs, embedding, n_classes):
    """Attach dual heads to a shared embedding and wrap in MTLModel."""
    cls_out = tf.keras.layers.Dense(n_classes, activation='softmax', name='cls_out')(embedding)
    reg_out = tf.keras.layers.Dense(1, activation='linear', name='reg_out')(embedding)
    return MTLModel(inputs=inputs, outputs=[cls_out, reg_out])


# --- 1. CNN MTL ---
def create_cnn_mtl_model(T, n_classes):
    inputs = tf.keras.layers.Input(shape=(T, 1))
    x = tf.keras.layers.Conv1D(16, 5, activation='relu')(inputs)
    x = tf.keras.layers.Conv1D(8, 3, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    embedding = tf.keras.layers.Dense(64, activation='relu')(x)
    embedding = tf.keras.layers.Dropout(0.2)(embedding)
    return _mtl_wrap(inputs, embedding, n_classes)


# --- 2. LSTM MTL ---
def create_lstm_mtl_model(T, n_classes):
    inputs = tf.keras.layers.Input(shape=(T, 1))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16))(x)
    embedding = tf.keras.layers.Dropout(0.2)(x)
    return _mtl_wrap(inputs, embedding, n_classes)


# --- 3. GRU MTL ---
def create_gru_mtl_model(T, n_classes):
    inputs = tf.keras.layers.Input(shape=(T, 1))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, return_sequences=True))(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(16))(x)
    embedding = tf.keras.layers.Dropout(0.2)(x)
    return _mtl_wrap(inputs, embedding, n_classes)


# --- 4. RNN MTL ---
def create_rnn_mtl_model(T, n_classes):
    inputs = tf.keras.layers.Input(shape=(T, 1))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(32, return_sequences=True))(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(16))(x)
    embedding = tf.keras.layers.Dropout(0.2)(x)
    return _mtl_wrap(inputs, embedding, n_classes)


# --- 5. Transformer MTL ---
def create_transformer_mtl_model(T, n_classes, head_size=32, num_heads=2, ff_dim=32, num_blocks=2, dropout=0.1):
    inputs = tf.keras.layers.Input(shape=(T, 1))
    x = tf.keras.layers.Conv1D(filters=head_size, kernel_size=5, strides=2, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, padding='same')(x)
    new_seq_len = x.shape[1]
    positions = tf.range(start=0, limit=new_seq_len, delta=1)
    pos_embedding = tf.keras.layers.Embedding(input_dim=new_seq_len, output_dim=head_size)(positions)
    x = x + pos_embedding
    for _ in range(num_blocks):
        attn = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
        attn = tf.keras.layers.Dropout(dropout)(attn)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn)
        ffn = tf.keras.layers.Dense(ff_dim, activation='relu')(x)
        ffn = tf.keras.layers.Dropout(dropout)(ffn)
        ffn = tf.keras.layers.Dense(head_size)(ffn)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    embedding = tf.keras.layers.Dense(64, activation='relu')(x)
    embedding = tf.keras.layers.Dropout(0.2)(embedding)
    return _mtl_wrap(inputs, embedding, n_classes)


# --- 6. CNN+GRU Dual MTL (also reused for cnn_gru_dual_cosine_recon_mtl) ---
def _build_cnn_gru_dual_branches_mtl(input_tensor):
    """Identical to model_utils._build_cnn_gru_dual_branches but self-contained."""
    c = tf.keras.layers.Conv1D(16, 5, activation='relu')(input_tensor)
    c = tf.keras.layers.Conv1D(8, 3, activation='relu')(c)
    c = tf.keras.layers.Flatten()(c)
    cnn_emb = tf.keras.layers.Dense(32, activation='relu')(c)

    g = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, return_sequences=True))(input_tensor)
    g = tf.keras.layers.LayerNormalization()(g)
    g = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(16))(g)
    g = tf.keras.layers.Dropout(0.2)(g)
    gru_emb = tf.keras.layers.Dense(32, activation='relu')(g)

    merged = tf.keras.layers.Concatenate()([cnn_emb, gru_emb])
    z = tf.keras.layers.Dense(64, activation='relu')(merged)
    z = tf.keras.layers.Dropout(0.2)(z)
    return z


def create_cnn_gru_dual_mtl_model(T, n_classes):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    embedding = _build_cnn_gru_dual_branches_mtl(inputs)
    return _mtl_wrap(inputs, embedding, n_classes)


# --- 7. CNN+Transformer Dual MTL ---
def create_cnn_trans_dual_mtl_model(T, n_classes, head_size=32, num_heads=2, ff_dim=32, num_blocks=2, dropout=0.1):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')

    c = tf.keras.layers.Conv1D(16, 5, activation='relu')(inputs)
    c = tf.keras.layers.Conv1D(8, 3, activation='relu')(c)
    c = tf.keras.layers.Flatten()(c)
    cnn_emb = tf.keras.layers.Dense(32, activation='relu')(c)

    t = tf.keras.layers.Conv1D(filters=head_size, kernel_size=5, strides=2, padding='same', activation='relu')(inputs)
    t = tf.keras.layers.MaxPooling1D(pool_size=2, padding='same')(t)
    new_seq_len = t.shape[1]
    positions = tf.range(start=0, limit=new_seq_len, delta=1)
    pos_embedding = tf.keras.layers.Embedding(input_dim=new_seq_len, output_dim=head_size)(positions)
    t = t + pos_embedding
    for _ in range(num_blocks):
        attn = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(t, t)
        attn = tf.keras.layers.Dropout(dropout)(attn)
        t = tf.keras.layers.LayerNormalization(epsilon=1e-6)(t + attn)
        ffn = tf.keras.layers.Dense(ff_dim, activation='relu')(t)
        ffn = tf.keras.layers.Dropout(dropout)(ffn)
        ffn = tf.keras.layers.Dense(head_size)(ffn)
        t = tf.keras.layers.LayerNormalization(epsilon=1e-6)(t + ffn)
    t = tf.keras.layers.GlobalAveragePooling1D()(t)
    trans_emb = tf.keras.layers.Dense(32, activation='relu')(t)

    merged = tf.keras.layers.Concatenate()([cnn_emb, trans_emb])
    embedding = tf.keras.layers.Dense(64, activation='relu')(merged)
    embedding = tf.keras.layers.Dropout(0.2)(embedding)
    return _mtl_wrap(inputs, embedding, n_classes)


# --- 8. CNN+GRU Attn-Recon MTL ---
def create_cnn_gru_dual_attn_recon_mtl_model(k_plus_1, T, n_classes, attn_dim=16):
    """Same as create_cnn_gru_dual_attn_recon_model but with dual heads."""
    stack_input = tf.keras.layers.Input(shape=(k_plus_1, T), name='neighbor_stack_input')

    per_curve_encoder = tf.keras.Sequential([
        tf.keras.layers.Reshape((T, 1)),
        tf.keras.layers.Conv1D(16, 5, activation='relu', padding='same'),
        tf.keras.layers.Conv1D(8, 3, activation='relu', padding='same'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(attn_dim, activation='relu'),
    ], name='per_curve_encoder')
    embeddings = tf.keras.layers.TimeDistributed(per_curve_encoder)(stack_input)  # (N, k+1, attn_dim)

    query = tf.keras.layers.Lambda(lambda x: x[:, 0:1, :])(embeddings)            # (N, 1, attn_dim)
    scores = tf.keras.layers.Lambda(
        lambda t: tf.matmul(t[0], t[1], transpose_b=True) / tf.sqrt(tf.cast(attn_dim, tf.float32))
    )([query, embeddings])                                                         # (N, 1, k+1)
    attn_weights = tf.keras.layers.Softmax(axis=-1, name='attn_weights')(scores)  # (N, 1, k+1)

    reconstructed = tf.keras.layers.Lambda(
        lambda t: tf.matmul(t[0], t[1])                                           # (N,1,T)
    )([attn_weights, stack_input])
    reconstructed = tf.keras.layers.Reshape((T, 1))(reconstructed)               # (N, T, 1)

    embedding = _build_cnn_gru_dual_branches_mtl(reconstructed)
    return _mtl_wrap(stack_input, embedding, n_classes)
