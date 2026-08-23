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

    Active loss — UW-SO (Kirchdorfer et al. 2026), Strategy 3 (gradient-learned T):
        w_k = softmax(a_k / T), a_k = 1 / sg[L_k]        (Eq. 17)
        L   = w_cls*CE + w_reg*MSE_masked + 1/T           (Eq. 24)
        T = softplus(T_raw) keeps the temperature positive; T_raw is learned.
        Ref: "Investigating Uncertainty Weighting for Multi-Task Learning:
        Insights and Analytical Alternative" (Kirchdorfer et al., IJCV 2026)

    Kendall (2018) 2-sigma uncertainty weighting is scaffolded below but inactive.
    """

    def __init__(self, *args, reg_sentinel=REG_SENTINEL, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_sentinel = reg_sentinel
        # ── Kendall (2018): 2 learnable log-variance scalars (inactive) ──────
        # self.log_var_cls = self.add_weight(
        #     name='log_var_cls', shape=(), initializer='zeros', trainable=True)
        # self.log_var_reg = self.add_weight(
        #     name='log_var_reg', shape=(), initializer='zeros', trainable=True)
        # ── UW-SO temperature T, softplus-reparameterized to stay positive ───
        # softplus(ln(e-1)) == 1.0, matching the paper's default T init of 1.
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

        # ── Kendall (2018) ────────────────────────────────────────────────────
        # loss = (tf.exp(-self.log_var_cls) * ce + self.log_var_cls
        #         + tf.exp(-self.log_var_reg) * mse + self.log_var_reg)

        # ── UW-SO (Kirchdorfer et al. 2026), Strategy 3: tempered softmax over
        # analytically-optimal inverse-loss weights a_k = 1/sg[L_k] (Eq. 17),
        # plus a 1/T regularizer against temperature collapse (Eq. 24). ───────
        eps = 1e-8
        T = self.T
        a_cls = 1.0 / (tf.stop_gradient(ce) + eps)
        a_reg = 1.0 / (tf.stop_gradient(mse) + eps)

        logit_cls = a_cls / T
        logit_reg = a_reg / T
        m = tf.maximum(logit_cls, logit_reg)
        exp_cls = tf.exp(logit_cls - m)
        exp_reg = tf.exp(logit_reg - m)
        # No valid regression target this batch: force w_reg=0 (mse would
        # otherwise collapse toward 0, making a_reg explode and wrongly
        # dominate the softmax).
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


# ====================================================================
# PHASE-DECOUPLED MTL (CURRICULUM LEARNING)
# ====================================================================

_HEAD_LAYERS = frozenset({'cls_feat', 'cls_out', 'reg_feat', 'reg_hidden', 'reg_out'})


def _freeze_backbone(model):
    """Freeze all non-head layers, advance curriculum_phase to 1, recompile.

    compile() resets train_function to None; make_train_function(force=True) rebuilds it
    so the running fit() loop doesn't call None(iterator) on the next batch/epoch.
    """
    for layer in model.layers:
        if layer.name not in _HEAD_LAYERS:
            layer.trainable = False
    model.curriculum_phase.assign(1)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        jit_compile=False)
    model.make_train_function(force=True)


@tf.keras.utils.register_keras_serializable(package='mtl')
class CurriculumMTLModel(MTLModel):
    """Phase-decoupled MTL: phase 0 = regression only, phase 1 = classification only."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.curriculum_phase = tf.Variable(0, trainable=False, dtype=tf.int32)

    def _compute_loss(self, y_cls, y_reg, cls_out, reg_out):
        ce = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(y_cls, cls_out))
        mask = tf.cast(tf.not_equal(y_reg, self.reg_sentinel), tf.float32)
        n_valid = tf.reduce_sum(mask)
        mse = (tf.reduce_sum(mask * tf.square(y_reg - reg_out[:, 0]))
               / (n_valid + 1e-8))
        loss = tf.cond(tf.equal(self.curriculum_phase, 0), lambda: mse, lambda: ce)
        return loss, ce, mse

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            cls_out, reg_out = self(x, training=True)
            loss, ce, mse = self._compute_loss(
                y_dict['cls_out'], y_dict['reg_out'], cls_out, reg_out)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            (g, v) for g, v in zip(grads, self.trainable_variables) if g is not None)
        self._loss_tracker.update_state(loss)
        return {'loss': loss, 'cls_ce': ce, 'reg_mse': mse}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        cls_out, reg_out = self(x, training=False)
        loss, ce, mse = self._compute_loss(
            y_dict['cls_out'], y_dict['reg_out'], cls_out, reg_out)
        self._loss_tracker.update_state(loss)
        return {'loss': loss, 'cls_ce': ce, 'reg_mse': mse}

    def get_config(self):
        return super().get_config()


class AutoPhaseTransitionCallback(tf.keras.callbacks.Callback):
    """Transition from regression-only to cls-only when val_reg_mse plateaus.

    Waits min_phase1_epochs, then triggers on patience consecutive non-improving epochs.
    Resets EarlyStopping/ReduceLROnPlateau state at transition so Phase 2 is tracked fresh.
    """

    def __init__(self, min_phase1_epochs=30, patience=10,
                 early_stop_cb=None, rlrp_cb=None):
        super().__init__()
        self.min_phase1_epochs = min_phase1_epochs
        self.patience = patience
        self.early_stop_cb = early_stop_cb
        self.rlrp_cb = rlrp_cb
        self._best = float('inf')
        self._wait = 0
        self._transitioned = False

    def on_epoch_end(self, epoch, logs=None):
        if self._transitioned:
            return
        mse = (logs or {}).get('val_reg_mse')
        if mse is None or epoch < self.min_phase1_epochs:
            return
        if mse < self._best:
            self._best = mse
            self._wait = 0
        else:
            self._wait += 1
        if self._wait >= self.patience:
            self._transitioned = True
            _freeze_backbone(self.model)
            if self.early_stop_cb is not None:
                self.early_stop_cb.best = float('inf')
            if self.rlrp_cb is not None:
                self.rlrp_cb.best = float('inf')
            print(f'\n[CL] Phase 1→2 auto-transition at epoch {epoch + 1}')


class FixedPhaseTransitionCallback(tf.keras.callbacks.Callback):
    """Transition from regression-only to cls-only at a fixed epoch."""

    def __init__(self, phase1_epochs, early_stop_cb=None, rlrp_cb=None):
        super().__init__()
        self.phase1_epochs = phase1_epochs
        self.early_stop_cb = early_stop_cb
        self.rlrp_cb = rlrp_cb
        self._transitioned = False

    def on_epoch_begin(self, epoch, logs=None):
        if not self._transitioned and epoch == self.phase1_epochs:
            self._transitioned = True
            _freeze_backbone(self.model)
            if self.early_stop_cb is not None:
                self.early_stop_cb.best = float('inf')
            if self.rlrp_cb is not None:
                self.rlrp_cb.best = float('inf')
            print(f'\n[CL] Phase 1→2 fixed transition at epoch {epoch}')


def _cl_mtl_wrap(inputs, embedding, n_classes):
    """Same as _mtl_wrap but returns CurriculumMTLModel."""
    cls_feat = tf.keras.layers.Dense(16, activation='relu',    name='cls_feat')(embedding)
    cls_out  = tf.keras.layers.Dense(n_classes, activation='softmax', name='cls_out')(cls_feat)
    reg_feat = tf.keras.layers.Dense(16, activation='relu',    name='reg_feat')(embedding)
    reg_h    = tf.keras.layers.Dense(8,  activation='relu',    name='reg_hidden')(reg_feat)
    reg_out  = tf.keras.layers.Dense(1,  activation='linear',  name='reg_out')(reg_h)
    return CurriculumMTLModel(inputs=inputs, outputs=[cls_out, reg_out])


def create_cnn_gru_dual_cl_mtl_model(T, n_classes):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    return _cl_mtl_wrap(inputs, _build_cnn_gru_dual_branches_mtl(inputs), n_classes)


def create_cnn_trans_dual_cl_mtl_model(T, n_classes, head_size=32, num_heads=2,
                                        ff_dim=32, num_blocks=2, dropout=0.1):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    return _cl_mtl_wrap(inputs,
                        _build_cnn_trans_dual_branches_mtl(inputs, head_size, num_heads,
                                                            ff_dim, num_blocks, dropout),
                        n_classes)


CL_MTL_MODEL_KEYS = ['cnn_gru_dual_cl_mtl', 'cnn_trans_dual_cl_mtl']


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


def create_cnn_gru_dual_mtl_model(T, n_classes):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    embedding = _build_cnn_gru_dual_branches_mtl(inputs)
    return _mtl_wrap(inputs, embedding, n_classes)


# --- 7. CNN+Transformer Dual MTL ---
def create_cnn_trans_dual_mtl_model(T, n_classes, head_size=32, num_heads=2, ff_dim=32, num_blocks=2, dropout=0.1):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    return _mtl_wrap(inputs, _build_cnn_trans_dual_branches_mtl(inputs, head_size, num_heads, ff_dim, num_blocks, dropout), n_classes)


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


# --- 8b. CNN+GRU GAT-Recon MTL (mirrors attn_recon embedding builder above, swapping
# the per-curve CNN encoder for gnn_gat's single shared linear projection, dim=T) ---
def _build_cnn_gru_dual_gat_recon_embedding_mtl(stack_input, T, return_branches=False):
    """stack_input: Keras tensor (N, k+1, T). Returns 96-dim embedding for _dann_wrap/
    _coral_wrap/_branch3_supcon_*_wrap. When return_branches=True, returns
    (cnn_emb, gru_emb, fused) for the branch-3 SupCon/DANN/CORAL factories."""
    shared_proj = tf.keras.layers.Dense(T, activation=None, name='gat_shared_proj')
    Wh = tf.keras.layers.TimeDistributed(shared_proj, name='Wh')(stack_input)      # (N, k+1, T)

    query = _QuerySlice()(Wh)                                                      # (N, 1, T)
    scores = _AttnScores(T)([query, Wh])                                          # (N, 1, k+1)
    attn_weights = tf.keras.layers.Softmax(axis=-1, name='gat_attn_weights')(scores)

    reconstructed = _WeightedRecon()([attn_weights, Wh])                          # (N, 1, T)
    reconstructed = tf.keras.layers.Reshape((T, 1))(reconstructed)               # (N, T, 1)
    return _build_cnn_gru_dual_branches_mtl(reconstructed, return_branches=return_branches)


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
