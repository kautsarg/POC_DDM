import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import model_utils_gated

REG_SENTINEL = -1.0

MTL_MODEL_KEYS = [
    'cnn_mtl', 'lstm_mtl', 'gru_mtl', 'rnn_mtl', 'transformer_mtl',
    'cnn_gru_dual_mtl', 'cnn_trans_dual_mtl',
    'cnn_gru_dual_cosine_recon_mtl', 'cnn_gru_dual_attn_recon_mtl',
    'cnn_lf_mtl', 'gru_lf_mtl', 'trans_lf_mtl', 'lstm_lf_mtl',
    'cnn_lstm_dual_mtl',
    'cnn_gru_gate_mtl', 'cnn_gru_hadamard_mtl', 'cnn_gru_crossattn_mtl', 'cnn_gru_film_mtl',
    'cnn_trans_gate_mtl', 'cnn_trans_hadamard_mtl', 'cnn_trans_crossattn_mtl', 'cnn_trans_film_mtl',
]


# ====================================================================
# KENDALL UNCERTAINTY-WEIGHTED MTL MODEL
# ====================================================================

@tf.keras.utils.register_keras_serializable(package='mtl')
class MTLModel(tf.keras.Model):
    """Shared-backbone model with classification + regression heads.

    Active loss — Kendall (2018) uncertainty weighting (2 learnable scalars):
        L = exp(-s_cls)*CE + s_cls + exp(-s_reg)*MSE_masked + s_reg

    UW-SO alternative (commented out below) — single trainable temperature T:
        Analytical weights w_i ∝ 1/L_i (stop_gradient); only global scale T is learned.
        L = exp(-log_T) * (w_cls*CE + w_reg*MSE_masked) + log_T
        Ref: "Investigating Uncertainty Weighting for MTL: Insights and Analytical Alternative"
        To switch: comment out the Kendall blocks and uncomment the UW-SO blocks.
    """

    def __init__(self, *args, reg_sentinel=REG_SENTINEL, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_sentinel = reg_sentinel
        # ── Kendall (2018): 2 learnable log-variance scalars ─────────────────
        # self.log_var_cls = self.add_weight(
        #     name='log_var_cls', shape=(), initializer='zeros', trainable=True)
        # self.log_var_reg = self.add_weight(
        #     name='log_var_reg', shape=(), initializer='zeros', trainable=True)
        # ── UW-SO: single temperature scalar (replace the two above) ─────────
        self.log_T = self.add_weight(
            name='log_T', shape=(), initializer='zeros', trainable=True)

    def _compute_loss(self, y_cls, y_reg, cls_out, reg_out):
        ce = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(y_cls, cls_out))
        mask = tf.cast(tf.not_equal(y_reg, self.reg_sentinel), tf.float32)
        n_valid = tf.reduce_sum(mask)
        mse = (tf.reduce_sum(mask * tf.square(y_reg - reg_out[:, 0]))
               / (n_valid + 1e-8))

        # ── Kendall (2018) ────────────────────────────────────────────────────
        # loss = (tf.exp(-self.log_var_cls) * ce + self.log_var_cls
        #         + tf.exp(-self.log_var_reg) * mse + self.log_var_reg)

        # ── UW-SO: analytical inverse-loss weights, single temperature scalar ──
        eps = 1e-8
        ce_sg  = tf.stop_gradient(ce)
        mse_sg = tf.stop_gradient(mse)
        inv_ce  = 1.0 / (ce_sg + eps)
        inv_mse = tf.cond(n_valid > 0,
                          lambda: 1.0 / (mse_sg + eps),
                          lambda: tf.constant(0.0))
        Z      = inv_ce + inv_mse + eps
        w_cls  = inv_ce  / Z
        w_reg  = inv_mse / Z
        weighted = w_cls * ce + w_reg * mse
        loss = tf.exp(-self.log_T) * weighted + self.log_T

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
                | {'loss': loss, 'cls_ce': ce, 'reg_mse': mse, 'log_T': self.log_T})

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        cls_out, reg_out = self(x, training=False)
        loss, ce, mse = self._compute_loss(
            y_dict['cls_out'], y_dict['reg_out'], cls_out, reg_out)
        self.compiled_metrics.update_state(y_dict['cls_out'], cls_out)
        return ({m.name: m.result() for m in self.metrics}
                | {'loss': loss, 'cls_ce': ce, 'reg_mse': mse, 'log_T': self.log_T})

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
    """Log10-transform then StandardScaler on non-sentinel values.

    Pipeline: raw → log10 → z-score.  Sentinel entries are preserved unchanged.
    Concentrations of 0 must already be encoded as REG_SENTINEL by the caller
    (log10(0) is undefined; they are negative controls and masked from regression loss).
    Returns (scaled_array, fitted_scaler).
    """
    arr = conc_array.astype(float).copy()
    valid_mask = arr != REG_SENTINEL
    scaler = StandardScaler()
    if valid_mask.sum() > 0:
        log_vals = np.log10(arr[valid_mask])   # safe: caller sentinels 0s
        arr[valid_mask] = scaler.fit_transform(log_vals.reshape(-1, 1)).ravel()
    return arr, scaler


def _inverse_normalize_concentration(scaled_array, scaler, sentinel=REG_SENTINEL):
    """Inverse of _normalize_concentration: inverse z-score then 10^x → original scale."""
    arr = scaled_array.astype(float).copy()
    valid_mask = arr != sentinel
    if valid_mask.sum() > 0 and hasattr(scaler, 'mean_'):
        log_back = scaler.inverse_transform(arr[valid_mask].reshape(-1, 1)).ravel()
        arr[valid_mask] = np.power(10.0, log_back)
    return arr


# ====================================================================
# MTL MODEL FACTORIES
# ====================================================================
# Each factory mirrors its classification counterpart exactly up to the
# final Dense softmax, then attaches two heads and wraps in MTLModel.

def _mtl_wrap(inputs, embedding, n_classes):
    """Attach dual heads to a shared embedding and wrap in MTLModel.

    Proportional Dense(16) towers (embedding_dim // 4 for 64-dim backbones) give
    gradient isolation without the overfitting caused by the previous Dense(32) towers.
    """
    cls_feat = tf.keras.layers.Dense(16, activation='relu',    name='cls_feat')(embedding)
    cls_out  = tf.keras.layers.Dense(n_classes, activation='softmax', name='cls_out')(cls_feat)
    reg_feat = tf.keras.layers.Dense(16, activation='relu',    name='reg_feat')(embedding)
    reg_h    = tf.keras.layers.Dense(8,  activation='relu',    name='reg_hidden')(reg_feat)
    reg_out  = tf.keras.layers.Dense(1,  activation='linear',  name='reg_out')(reg_h)
    return MTLModel(inputs=inputs, outputs=[cls_out, reg_out])


# ======================================================================
# BACKBONE BUILDERS (reused by MTL factories and model_utils_supcon.py)
# ======================================================================

def _build_cnn_backbone_mtl(inputs):
    """Conv1D stack + Dense(64) embedding. Shared by CNN MTL and CNN SupCon."""
    x = tf.keras.layers.Conv1D(16, 5, activation='relu')(inputs)
    x = tf.keras.layers.Conv1D(8, 3, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    return tf.keras.layers.Dropout(0.2)(x)


def _build_gru_backbone_mtl(inputs):
    """BiGRU stack + Dense(64) embedding. Shared by GRU MTL and GRU SupCon."""
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, return_sequences=True))(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(16))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    return tf.keras.layers.Dense(64, activation='relu')(x)


def _build_transformer_backbone_mtl(inputs, head_size=32, num_heads=2, ff_dim=32, num_blocks=2, dropout=0.1):
    """Transformer blocks + Dense(64) embedding. Shared by Transformer MTL and SupCon."""
    x = tf.keras.layers.Conv1D(filters=head_size, kernel_size=5, strides=2,
                                padding='same', activation='relu')(inputs)
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
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    return tf.keras.layers.Dropout(dropout)(x)


def _build_cnn_trans_dual_branches_mtl(inputs, head_size=32, num_heads=2, ff_dim=32, num_blocks=2, dropout=0.1,
                                        return_branches=False):
    """CNN+Transformer dual backbone (96-dim). Shared by CNN+Trans MTL and SupCon."""
    c = tf.keras.layers.Conv1D(16, 5, activation='relu')(inputs)
    c = tf.keras.layers.Conv1D(8, 3, activation='relu')(c)
    c = tf.keras.layers.Flatten()(c)
    cnn_emb = tf.keras.layers.Dense(32, activation='relu')(c)
    t = tf.keras.layers.Conv1D(filters=head_size, kernel_size=5, strides=2,
                                padding='same', activation='relu')(inputs)
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
    trans_emb = tf.keras.layers.Dense(64, activation='relu')(t)
    merged = tf.keras.layers.Concatenate()([cnn_emb, trans_emb])
    x = tf.keras.layers.Dense(96, activation='relu')(merged)
    z = tf.keras.layers.Dropout(0.2)(x)
    if return_branches:
        return cnn_emb, trans_emb, z  # (32-dim, 64-dim, 96-dim fused)
    return z


# ======================================================================
# FACTORY FUNCTIONS
# ======================================================================

# --- 1. CNN MTL ---
def create_cnn_mtl_model(T, n_classes):
    inputs = tf.keras.layers.Input(shape=(T, 1))
    return _mtl_wrap(inputs, _build_cnn_backbone_mtl(inputs), n_classes)


# --- 2. LSTM MTL ---
def create_lstm_mtl_model(T, n_classes):
    inputs = tf.keras.layers.Input(shape=(T, 1))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    embedding = tf.keras.layers.Dense(64, activation='relu')(x)
    return _mtl_wrap(inputs, embedding, n_classes)


# --- 3. GRU MTL ---
def create_gru_mtl_model(T, n_classes):
    inputs = tf.keras.layers.Input(shape=(T, 1))
    return _mtl_wrap(inputs, _build_gru_backbone_mtl(inputs), n_classes)


# --- 4. RNN MTL ---
def create_rnn_mtl_model(T, n_classes):
    inputs = tf.keras.layers.Input(shape=(T, 1))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(32, return_sequences=True))(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(16))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    embedding = tf.keras.layers.Dense(64, activation='relu')(x)
    return _mtl_wrap(inputs, embedding, n_classes)


# --- 5. Transformer MTL ---
def create_transformer_mtl_model(T, n_classes, head_size=32, num_heads=2, ff_dim=32, num_blocks=2, dropout=0.1):
    inputs = tf.keras.layers.Input(shape=(T, 1))
    return _mtl_wrap(inputs, _build_transformer_backbone_mtl(inputs, head_size, num_heads, ff_dim, num_blocks, dropout), n_classes)


# --- 6. CNN+GRU Dual MTL (also reused for cnn_gru_dual_cosine_recon_mtl) ---
def _build_cnn_gru_dual_branches_mtl(input_tensor, return_branches=False):
    """Identical to model_utils._build_cnn_gru_dual_branches but self-contained."""
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


def create_cnn_gru_dual_mtl_model(T, n_classes):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    embedding = _build_cnn_gru_dual_branches_mtl(inputs)
    return _mtl_wrap(inputs, embedding, n_classes)


# --- 7. CNN+Transformer Dual MTL ---
def create_cnn_trans_dual_mtl_model(T, n_classes, head_size=32, num_heads=2, ff_dim=32, num_blocks=2, dropout=0.1):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    return _mtl_wrap(inputs, _build_cnn_trans_dual_branches_mtl(inputs, head_size, num_heads, ff_dim, num_blocks, dropout), n_classes)


# --- 8. CNN+GRU Attn-Recon MTL ---
def _build_cnn_gru_dual_attn_recon_embedding_mtl(stack_input, T, attn_dim=16, return_branches=False):
    """Shared attention-weighted reconstruction → CNN+GRU embedding.
    stack_input: Keras tensor (N, k+1, T). Returns 96-dim embedding for _mtl_wrap/_supcon_wrap.
    When return_branches=True, returns (cnn_emb, gru_emb, fused) for branch SupCon factories.
    """
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
        lambda t: tf.matmul(t[0], t[1])                                           # (N, 1, T)
    )([attn_weights, stack_input])
    reconstructed = tf.keras.layers.Reshape((T, 1))(reconstructed)               # (N, T, 1)
    return _build_cnn_gru_dual_branches_mtl(reconstructed, return_branches=return_branches)


def create_cnn_gru_dual_attn_recon_mtl_model(k_plus_1, T, n_classes, attn_dim=16):
    stack_input = tf.keras.layers.Input(shape=(k_plus_1, T), name='neighbor_stack_input')
    return _mtl_wrap(stack_input,
                     _build_cnn_gru_dual_attn_recon_embedding_mtl(stack_input, T, attn_dim),
                     n_classes)


# ====================================================================
# LATE FUSION MTL MODEL FACTORIES
# ====================================================================
# Each mirrors its non-MTL LF counterpart exactly up to the final softmax,
# then attaches dual heads via _mtl_wrap([curve_in, feat_in], embedding, n_classes).

# --- 9. CNN Late Fusion MTL ---
def create_cnn_lf_mtl_model(T, n_features, n_classes):
    input_curve = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    x = tf.keras.layers.Conv1D(16, 5, activation='relu')(input_curve)
    x = tf.keras.layers.Conv1D(8, 3, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    curve_emb = tf.keras.layers.Dense(32, activation='relu')(x)

    input_features = tf.keras.layers.Input(shape=(n_features,), name='features_input')
    feat_emb = tf.keras.layers.Dense(32, activation='relu')(input_features)

    merged = tf.keras.layers.Concatenate()([curve_emb, feat_emb])
    z = tf.keras.layers.Dense(64, activation='relu')(merged)
    embedding = tf.keras.layers.Dropout(0.2)(z)
    return _mtl_wrap([input_curve, input_features], embedding, n_classes)


# --- 10. GRU Late Fusion MTL ---
def create_gru_lf_mtl_model(T, n_features, n_classes):
    input_curve = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, return_sequences=True))(input_curve)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(16))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    curve_emb = tf.keras.layers.Dense(32, activation='relu')(x)

    input_features = tf.keras.layers.Input(shape=(n_features,), name='features_input')
    feat_emb = tf.keras.layers.Dense(32, activation='relu')(input_features)

    merged = tf.keras.layers.Concatenate()([curve_emb, feat_emb])
    z = tf.keras.layers.Dense(64, activation='relu')(merged)
    embedding = tf.keras.layers.Dropout(0.2)(z)
    return _mtl_wrap([input_curve, input_features], embedding, n_classes)


# --- 11. Transformer Late Fusion MTL ---
def create_transformer_lf_mtl_model(T, n_features, n_classes, head_size=32, num_heads=2, ff_dim=32, num_blocks=2, dropout=0.1):
    input_curve = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    x = tf.keras.layers.Conv1D(filters=head_size, kernel_size=5, strides=2, padding='same', activation='relu')(input_curve)
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
    curve_emb = tf.keras.layers.Dense(32, activation='relu')(x)

    input_features = tf.keras.layers.Input(shape=(n_features,), name='features_input')
    feat_emb = tf.keras.layers.Dense(32, activation='relu')(input_features)

    merged = tf.keras.layers.Concatenate()([curve_emb, feat_emb])
    z = tf.keras.layers.Dense(64, activation='relu')(merged)
    embedding = tf.keras.layers.Dropout(0.2)(z)
    return _mtl_wrap([input_curve, input_features], embedding, n_classes)


# --- 12. LSTM Late Fusion MTL ---
def create_lstm_lf_mtl_model(T, n_features, n_classes):
    input_curve = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))(input_curve)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    curve_emb = tf.keras.layers.Dense(32, activation='relu')(x)

    input_features = tf.keras.layers.Input(shape=(n_features,), name='features_input')
    feat_emb = tf.keras.layers.Dense(32, activation='relu')(input_features)

    merged_lf = tf.keras.layers.Concatenate()([curve_emb, feat_emb])
    z = tf.keras.layers.Dense(64, activation='relu')(merged_lf)
    embedding = tf.keras.layers.Dropout(0.2)(z)
    return _mtl_wrap([input_curve, input_features], embedding, n_classes)


# ====================================================================
# CNN+LSTM DUAL MTL
# ====================================================================

def _build_cnn_lstm_dual_branches_mtl(input_tensor):
    """Dual-branch CNN + BiLSTM, self-contained twin of model_utils._build_cnn_lstm_dual_branches."""
    c = tf.keras.layers.Conv1D(16, 5, activation='relu')(input_tensor)
    c = tf.keras.layers.Conv1D(8, 3, activation='relu')(c)
    c = tf.keras.layers.Flatten()(c)
    cnn_emb = tf.keras.layers.Dense(32, activation='relu')(c)

    l = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))(input_tensor)
    l = tf.keras.layers.LayerNormalization()(l)
    l = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16))(l)
    l = tf.keras.layers.Dropout(0.2)(l)
    lstm_emb = tf.keras.layers.Dense(64, activation='relu')(l)

    merged_dual = tf.keras.layers.Concatenate()([cnn_emb, lstm_emb])
    z = tf.keras.layers.Dense(96, activation='relu')(merged_dual)
    z = tf.keras.layers.Dropout(0.2)(z)
    return z


def create_cnn_lstm_dual_mtl_model(T, n_classes):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    embedding = _build_cnn_lstm_dual_branches_mtl(inputs)
    return _mtl_wrap(inputs, embedding, n_classes)


# ====================================================================
# GATED FUSION MTL FACTORIES  (mirror model_utils_gated._ALL_FACTORIES)
# ====================================================================
# Each reuses the shared branch/fusion helpers from model_utils_gated but
# replaces _head(merged, output_size) with Dense(64,relu)+Dropout → _mtl_wrap.

def _gated_mtl_head(inp, merged, n_classes):
    """Dense(64,relu) → Dropout → dual heads → MTLModel (replaces _head for MTL)."""
    z = tf.keras.layers.Dense(64, activation='relu', name='head_d1')(merged)
    z = tf.keras.layers.Dropout(0.2, name='head_drop')(z)
    return _mtl_wrap(inp, z, n_classes)


def create_cnn_gru_gate_mtl_model(T, n_classes):
    inp = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    _, cnn_emb = model_utils_gated._cnn_branch(inp, pfx='cnn')
    gru_emb    = model_utils_gated._gru_branch(inp, pfx='gru')
    merged     = model_utils_gated._fuse_gate(cnn_emb, gru_emb, pfx='fuse')
    return _gated_mtl_head(inp, merged, n_classes)


def create_cnn_gru_hadamard_mtl_model(T, n_classes):
    inp = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    _, cnn_emb = model_utils_gated._cnn_branch(inp, pfx='cnn')
    gru_emb    = model_utils_gated._gru_branch(inp, pfx='gru')
    merged     = model_utils_gated._fuse_hadamard(cnn_emb, gru_emb, pfx='fuse')
    return _gated_mtl_head(inp, merged, n_classes)


def create_cnn_gru_crossattn_mtl_model(T, n_classes, num_heads=2):
    inp = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    cnn_seq, cnn_emb = model_utils_gated._cnn_branch(inp, pfx='cnn')
    gru_seq, gru_emb = model_utils_gated._gru_branch(inp, return_seq=True, pfx='gru')
    merged = model_utils_gated._fuse_coattn(cnn_seq, cnn_emb, gru_seq, gru_emb,
                                            num_heads=num_heads, pfx='fuse')
    return _gated_mtl_head(inp, merged, n_classes)


def create_cnn_gru_film_mtl_model(T, n_classes):
    inp = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    _, cnn_emb = model_utils_gated._cnn_branch(inp, pfx='cnn')
    gru_emb    = model_utils_gated._gru_branch(inp, pfx='gru')
    merged     = model_utils_gated._fuse_film(cnn_emb, gru_emb, pfx='fuse')
    return _gated_mtl_head(inp, merged, n_classes)


def create_cnn_trans_gate_mtl_model(T, n_classes,
                                    head_size=32, num_heads=2, ff_dim=32, num_blocks=2, dropout=0.1):
    inp = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    _, cnn_emb = model_utils_gated._cnn_branch(inp, pfx='cnn')
    trans_emb  = model_utils_gated._trans_branch(inp, head_size, num_heads, ff_dim,
                                                 num_blocks, dropout, pfx='trans')
    merged     = model_utils_gated._fuse_gate(cnn_emb, trans_emb, pfx='fuse')
    return _gated_mtl_head(inp, merged, n_classes)


def create_cnn_trans_hadamard_mtl_model(T, n_classes,
                                        head_size=32, num_heads=2, ff_dim=32, num_blocks=2, dropout=0.1):
    inp = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    _, cnn_emb = model_utils_gated._cnn_branch(inp, pfx='cnn')
    trans_emb  = model_utils_gated._trans_branch(inp, head_size, num_heads, ff_dim,
                                                 num_blocks, dropout, pfx='trans')
    merged     = model_utils_gated._fuse_hadamard(cnn_emb, trans_emb, pfx='fuse')
    return _gated_mtl_head(inp, merged, n_classes)


def create_cnn_trans_crossattn_mtl_model(T, n_classes,
                                         head_size=32, num_heads=2, ff_dim=32, num_blocks=2, dropout=0.1):
    inp = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    cnn_seq, cnn_emb     = model_utils_gated._cnn_branch(inp, pfx='cnn')
    trans_seq, trans_emb = model_utils_gated._trans_branch(inp, head_size, num_heads, ff_dim,
                                                           num_blocks, dropout,
                                                           return_seq=True, pfx='trans')
    merged = model_utils_gated._fuse_coattn(cnn_seq, cnn_emb, trans_seq, trans_emb,
                                            num_heads=num_heads, pfx='fuse')
    return _gated_mtl_head(inp, merged, n_classes)


def create_cnn_trans_film_mtl_model(T, n_classes,
                                    head_size=32, num_heads=2, ff_dim=32, num_blocks=2, dropout=0.1):
    inp = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    _, cnn_emb = model_utils_gated._cnn_branch(inp, pfx='cnn')
    trans_emb  = model_utils_gated._trans_branch(inp, head_size, num_heads, ff_dim,
                                                 num_blocks, dropout, pfx='trans')
    merged     = model_utils_gated._fuse_film(cnn_emb, trans_emb, pfx='fuse')
    return _gated_mtl_head(inp, merged, n_classes)


_ALL_GATED_MTL_FACTORIES = {
    'cnn_gru_gate_mtl':        create_cnn_gru_gate_mtl_model,
    'cnn_gru_hadamard_mtl':    create_cnn_gru_hadamard_mtl_model,
    'cnn_gru_crossattn_mtl':   create_cnn_gru_crossattn_mtl_model,
    'cnn_gru_film_mtl':        create_cnn_gru_film_mtl_model,
    'cnn_trans_gate_mtl':      create_cnn_trans_gate_mtl_model,
    'cnn_trans_hadamard_mtl':  create_cnn_trans_hadamard_mtl_model,
    'cnn_trans_crossattn_mtl': create_cnn_trans_crossattn_mtl_model,
    'cnn_trans_film_mtl':      create_cnn_trans_film_mtl_model,
}
