import tensorflow as tf

from model_utils_mtl import (_build_cnn_gru_dual_branches_mtl,
                              _build_cnn_gru_dual_attn_recon_embedding_mtl)
from model_utils_supcon import _proj_head, SUPCON_TEMP, supcon_loss


def coral_loss(embedding, chip_id, n_chips):
    """Deep CORAL (Sun & Saenko 2016): penalizes each chip's embedding covariance
    vs. the batch-pooled covariance. n_chips is the training pool's chip count
    (small, fixed -- LOFO folds have 3-4), so this unrolls into n_chips static
    graph ops rather than a dynamic loop. Chips absent (or near-absent) from a
    given batch contribute 0, not NaN."""
    embedding = tf.cast(embedding, tf.float32)
    chip_id = tf.reshape(chip_id, [-1])
    d = tf.cast(tf.shape(embedding)[1], tf.float32)

    def _cov(x):
        mean = tf.reduce_mean(x, axis=0, keepdims=True)
        centered = x - mean
        n = tf.maximum(tf.cast(tf.shape(x)[0], tf.float32) - 1.0, 1.0)
        return tf.matmul(centered, centered, transpose_a=True) / n

    pooled_cov = _cov(embedding)
    total = tf.constant(0.0, dtype=tf.float32)
    for c in range(n_chips):
        mask = tf.equal(chip_id, c)
        n_c = tf.reduce_sum(tf.cast(mask, tf.float32))
        chip_embed = tf.boolean_mask(embedding, mask)
        contrib = tf.cond(
            n_c > 1.0,
            lambda chip_embed=chip_embed: tf.reduce_sum(tf.square(_cov(chip_embed) - pooled_cov)),
            lambda: tf.constant(0.0, dtype=tf.float32),
        )
        total += contrib
    return total / (4.0 * d * d * tf.cast(n_chips, tf.float32))


def _coral_wrap(inputs, embedding, n_classes, n_chips=1, coral_weight=1.0):
    cls_feat  = tf.keras.layers.Dense(16, activation='relu', name='cls_feat')(embedding)
    cls_out   = tf.keras.layers.Dense(n_classes, activation='softmax', name='cls_out')(cls_feat)
    embed_out = tf.keras.layers.Activation('linear', name='embed_out')(embedding)
    return CORALModel(inputs=inputs, outputs=[cls_out, embed_out],
                      n_chips=n_chips, coral_weight=coral_weight)


@tf.keras.utils.register_keras_serializable(package='model_utils_coral')
class CORALModel(tf.keras.Model):
    """CE + coral_weight*coral_loss(embedding, chip_id) -- a non-adversarial
    alternative to DANN's chip-invariance pressure. No GRL, no lambda schedule --
    coral_weight is fixed, since CORAL's loss has no analogous "too early = collapse"
    failure mode."""
    def __init__(self, *args, n_chips=1, coral_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_chips = n_chips
        self.coral_weight = coral_weight

    def _compute_loss(self, y_cls, chip_id, cls_out, embed_out):
        ce = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_cls, cls_out))
        coral = coral_loss(embed_out, chip_id, self.n_chips)
        loss = ce + self.coral_weight * coral
        return loss, ce, coral

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            cls_out, embed_out = self(x, training=True)
            loss, ce, coral = self._compute_loss(y_dict['cls_out'], y_dict['chip_id'], cls_out, embed_out)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y_dict['cls_out'], cls_out)
        return ({m.name: m.result() for m in self.metrics}
                | {'loss': loss, 'cls_ce': ce, 'coral': coral})

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        cls_out, embed_out = self(x, training=False)
        loss, ce, coral = self._compute_loss(y_dict['cls_out'], y_dict['chip_id'], cls_out, embed_out)
        self.compiled_metrics.update_state(y_dict['cls_out'], cls_out)
        return ({m.name: m.result() for m in self.metrics}
                | {'loss': loss, 'cls_ce': ce, 'coral': coral})


def _branch3_supcon_coral_wrap(inputs, cnn_emb, seq_emb, fused, n_classes, n_chips=1, coral_weight=1.0):
    cls_feat   = tf.keras.layers.Dense(16, activation='relu', name='cls_feat')(fused)
    cls_out    = tf.keras.layers.Dense(n_classes, activation='softmax', name='cls_out')(cls_feat)
    cnn_proj   = _proj_head(cnn_emb, 'cnn')
    seq_proj   = _proj_head(seq_emb, 'seq')
    fused_proj = _proj_head(fused,   'fused')
    embed_out  = tf.keras.layers.Activation('linear', name='embed_out')(fused)
    return SupConBranch3CORALModel(
        inputs=inputs, outputs=[cls_out, cnn_proj, seq_proj, fused_proj, embed_out],
        n_chips=n_chips, coral_weight=coral_weight)


@tf.keras.utils.register_keras_serializable(package='model_utils_coral')
class SupConBranch3CORALModel(tf.keras.Model):
    """(1-3*supcon_lambda_each)*CE + supcon_lambda_each*(3 SC losses) + coral_weight*coral_loss."""
    def __init__(self, *args, supcon_temp=SUPCON_TEMP, supcon_lambda_each=0.1,
                 n_chips=1, coral_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.supcon_temp = supcon_temp
        self.supcon_lambda_each = supcon_lambda_each
        self.n_chips = n_chips
        self.coral_weight = coral_weight

    def _compute_loss(self, y_cls, chip_id, cls_out, cnn_proj, seq_proj, fused_proj, embed_out):
        ce = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_cls, cls_out))
        sc = (supcon_loss(cnn_proj, y_cls, self.supcon_temp)
              + supcon_loss(seq_proj, y_cls, self.supcon_temp)
              + supcon_loss(fused_proj, y_cls, self.supcon_temp))
        coral = coral_loss(embed_out, chip_id, self.n_chips)
        loss = ((1.0 - 3 * self.supcon_lambda_each) * ce
                + self.supcon_lambda_each * sc
                + self.coral_weight * coral)
        return loss, ce, sc, coral

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            cls_out, cnn_proj, seq_proj, fused_proj, embed_out = self(x, training=True)
            loss, ce, sc, coral = self._compute_loss(
                y_dict['cls_out'], y_dict['chip_id'], cls_out, cnn_proj, seq_proj, fused_proj, embed_out)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y_dict['cls_out'], cls_out)
        return ({m.name: m.result() for m in self.metrics}
                | {'loss': loss, 'cls_ce': ce, 'supcon': sc, 'coral': coral})

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        cls_out, cnn_proj, seq_proj, fused_proj, embed_out = self(x, training=False)
        loss, ce, sc, coral = self._compute_loss(
            y_dict['cls_out'], y_dict['chip_id'], cls_out, cnn_proj, seq_proj, fused_proj, embed_out)
        self.compiled_metrics.update_state(y_dict['cls_out'], cls_out)
        return ({m.name: m.result() for m in self.metrics}
                | {'loss': loss, 'cls_ce': ce, 'supcon': sc, 'coral': coral})


def create_cnn_gru_dual_coral_model(T, n_classes, n_chips):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    return _coral_wrap(inputs, _build_cnn_gru_dual_branches_mtl(inputs), n_classes, n_chips=n_chips)


def create_cnn_gru_dual_attn_recon_coral_model(k_plus_1, T, n_classes, n_chips, attn_dim=16):
    stack_input = tf.keras.layers.Input(shape=(k_plus_1, T), name='neighbor_stack_input')
    embedding = _build_cnn_gru_dual_attn_recon_embedding_mtl(stack_input, T, attn_dim)
    return _coral_wrap(stack_input, embedding, n_classes, n_chips=n_chips)


def create_cnn_gru_dual_supcon3_coral_model(T, n_classes, n_chips):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    cnn_emb, seq_emb, fused = _build_cnn_gru_dual_branches_mtl(inputs, return_branches=True)
    return _branch3_supcon_coral_wrap(inputs, cnn_emb, seq_emb, fused, n_classes, n_chips=n_chips)


def create_cnn_gru_dual_attn_recon_supcon3_coral_model(k_plus_1, T, n_classes, n_chips, attn_dim=16):
    stack_input = tf.keras.layers.Input(shape=(k_plus_1, T), name='neighbor_stack_input')
    cnn_emb, seq_emb, fused = _build_cnn_gru_dual_attn_recon_embedding_mtl(
        stack_input, T, attn_dim, return_branches=True)
    return _branch3_supcon_coral_wrap(stack_input, cnn_emb, seq_emb, fused, n_classes, n_chips=n_chips)


CORAL_MODEL_KEYS = (
    'cnn_gru_dual_coral', 'cnn_gru_dual_attn_recon_coral',
    'cnn_gru_dual_supcon3_coral', 'cnn_gru_dual_attn_recon_supcon3_coral',
)
