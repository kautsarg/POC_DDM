import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

REG_SENTINEL = -1.0


@tf.keras.utils.register_keras_serializable(package='mtl')
class MTLModel(tf.keras.Model):

    def __init__(self, *args, reg_sentinel=REG_SENTINEL, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_sentinel = reg_sentinel
        self._T_raw = self.add_weight(
            name='T_raw', shape=(), trainable=True,
            initializer=tf.keras.initializers.Constant(float(np.log(np.e - 1.0))))

    @property
    def T(self):
        return tf.nn.softplus(self._T_raw)

    def _compute_loss(self, y_cls, y_reg, cls_out, reg_out):
        ce = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(y_cls, cls_out))
        mask = tf.cast(tf.not_equal(y_reg, self.reg_sentinel), tf.float32)
        n_valid = tf.reduce_sum(mask)
        has_reg = n_valid > 0
        mse = (tf.reduce_sum(mask * tf.square(y_reg - reg_out[:, 0]))
               / (n_valid + 1e-8))

        eps = 1e-8
        T = self.T
        a_cls = 1.0 / (tf.stop_gradient(ce) + eps)
        a_reg = 1.0 / (tf.stop_gradient(mse) + eps)

        logit_cls = a_cls / T
        logit_reg = a_reg / T
        m = tf.maximum(logit_cls, logit_reg)
        exp_cls = tf.exp(logit_cls - m)
        exp_reg = tf.exp(logit_reg - m)
        w_reg = tf.where(has_reg, exp_reg / (exp_cls + exp_reg), tf.zeros_like(exp_reg))
        w_cls = 1.0 - w_reg

        weighted = w_cls * ce + w_reg * mse
        loss = weighted + 1.0 / T
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
                | {'loss': loss, 'cls_ce': ce, 'reg_mse': mse, 'T': self.T})

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        cls_out, reg_out = self(x, training=False)
        loss, ce, mse = self._compute_loss(
            y_dict['cls_out'], y_dict['reg_out'], cls_out, reg_out)
        self.compiled_metrics.update_state(y_dict['cls_out'], cls_out)
        return ({m.name: m.result() for m in self.metrics}
                | {'loss': loss, 'cls_ce': ce, 'reg_mse': mse, 'T': self.T})

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


def _normalize_concentration(conc_array):
    """log10 then StandardScaler on non-sentinel values. Concentration 0 must already
    be encoded as REG_SENTINEL by the caller (log10(0) undefined)."""
    arr = conc_array.astype(float).copy()
    valid_mask = arr != REG_SENTINEL
    scaler = StandardScaler()
    if valid_mask.sum() > 0:
        log_vals = np.log10(arr[valid_mask])
        arr[valid_mask] = scaler.fit_transform(log_vals.reshape(-1, 1)).ravel()
    return arr, scaler


def _inverse_normalize_concentration(scaled_array, scaler, sentinel=REG_SENTINEL):
    arr = scaled_array.astype(float).copy()
    valid_mask = arr != sentinel
    if valid_mask.sum() > 0 and hasattr(scaler, 'mean_'):
        log_back = scaler.inverse_transform(arr[valid_mask].reshape(-1, 1)).ravel()
        arr[valid_mask] = np.power(10.0, log_back)
    return arr


def _mtl_wrap(inputs, embedding, n_classes):
    cls_feat = tf.keras.layers.Dense(16, activation='relu',    name='cls_feat')(embedding)
    cls_out  = tf.keras.layers.Dense(n_classes, activation='softmax', name='cls_out')(cls_feat)
    reg_feat = tf.keras.layers.Dense(16, activation='relu',    name='reg_feat')(embedding)
    reg_h    = tf.keras.layers.Dense(8,  activation='relu',    name='reg_hidden')(reg_feat)
    reg_out  = tf.keras.layers.Dense(1,  activation='linear',  name='reg_out')(reg_h)
    return MTLModel(inputs=inputs, outputs=[cls_out, reg_out])


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


def create_cnn_gru_dual_attn_recon_mtl_model(k_plus_1, T, n_classes, attn_dim=16):
    stack_input = tf.keras.layers.Input(shape=(k_plus_1, T), name='neighbor_stack_input')
    return _mtl_wrap(stack_input,
                     _build_cnn_gru_dual_attn_recon_embedding_mtl(stack_input, T, attn_dim),
                     n_classes)
