import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from model_utils_mtl import (
    _build_cnn_gru_dual_branches_mtl,
    _build_cnn_gru_dual_attn_recon_embedding_mtl,
)

SUPCON_TEMP   = 0.1
SUPCON_LAMBDA = 0.2

SUPCON_MODEL_KEYS = [
    'cnn_gru_dual_supcon', 'cnn_gru_dual_attn_recon_supcon',
]

BRANCH_SUPCON3_MODEL_KEYS = [
    'cnn_gru_dual_supcon3', 'cnn_gru_dual_attn_recon_supcon3',
]

ALL_SUPCON_KEYS = SUPCON_MODEL_KEYS + BRANCH_SUPCON3_MODEL_KEYS


# ======================================================================
# LOSS
# ======================================================================

def supcon_loss(embeddings, labels, temp=SUPCON_TEMP):
    """Supervised contrastive loss (Khosla et al. 2020).
    embeddings: (N, D), L2-normalized. labels: (N,), integer class indices.
    """
    N = tf.shape(embeddings)[0]
    sim = tf.matmul(embeddings, embeddings, transpose_b=True) / temp  # (N, N)

    labels_col = tf.cast(tf.expand_dims(labels, 1), tf.int32)
    labels_row = tf.cast(tf.expand_dims(labels, 0), tf.int32)
    same_label = tf.equal(labels_col, labels_row)
    not_self   = tf.logical_not(tf.eye(N, dtype=tf.bool))
    pos_mask   = tf.cast(tf.logical_and(same_label, not_self), tf.float32)
    neg_mask   = tf.cast(not_self, tf.float32)

    sim_max   = tf.reduce_max(sim, axis=1, keepdims=True)
    exp_sim   = tf.exp(sim - sim_max)
    log_denom = tf.math.log(tf.reduce_sum(exp_sim * neg_mask, axis=1, keepdims=True) + 1e-8)
    log_prob  = (sim - sim_max) - log_denom  # numerically stable

    n_pos           = tf.reduce_sum(pos_mask, axis=1)
    has_pos         = tf.cast(n_pos > 0, tf.float32)
    loss_per_anchor = -tf.reduce_sum(log_prob * pos_mask, axis=1) / (n_pos + 1e-8)
    return tf.reduce_mean(loss_per_anchor * has_pos)


# ======================================================================
# MODEL CLASSES
# ======================================================================

@tf.keras.utils.register_keras_serializable(package='supcon')
class SupConModel(tf.keras.Model):
    """ST + SupCon: CE + supervised contrastive loss on projection head."""
    def __init__(self, *args, supcon_temp=SUPCON_TEMP, supcon_lambda=SUPCON_LAMBDA, **kwargs):
        super().__init__(*args, **kwargs)
        self.supcon_temp   = supcon_temp
        self.supcon_lambda = supcon_lambda

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        y_cls = y_dict['cls_out']
        with tf.GradientTape() as tape:
            cls_out, proj_norm = self(x, training=True)
            ce   = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(y_cls, cls_out))
            sc   = supcon_loss(proj_norm, y_cls, self.supcon_temp)
            loss = (1.0 - self.supcon_lambda) * ce + self.supcon_lambda * sc
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y_cls, cls_out)
        return {m.name: m.result() for m in self.metrics} | {'loss': loss, 'cls_ce': ce, 'supcon': sc}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        y_cls = y_dict['cls_out']
        cls_out, proj_norm = self(x, training=False)
        ce   = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(y_cls, cls_out))
        sc   = supcon_loss(proj_norm, y_cls, self.supcon_temp)
        loss = (1.0 - self.supcon_lambda) * ce + self.supcon_lambda * sc
        self.compiled_metrics.update_state(y_cls, cls_out)
        return {m.name: m.result() for m in self.metrics} | {'loss': loss, 'cls_ce': ce, 'supcon': sc}


# ======================================================================
# TIER-1 (FUSED) WRAP HELPER
# ======================================================================

def _supcon_wrap(inputs, embedding, n_classes,
                 supcon_temp=SUPCON_TEMP, supcon_lambda=SUPCON_LAMBDA):
    """Attach cls head + projection head to embedding, return SupConModel."""
    cls_feat  = tf.keras.layers.Dense(16, activation='relu',  name='cls_feat')(embedding)
    cls_out   = tf.keras.layers.Dense(n_classes, activation='softmax', name='cls_out')(cls_feat)
    proj      = tf.keras.layers.Dense(64, activation='relu',  name='proj_hidden')(embedding)
    proj_norm = tf.keras.layers.UnitNormalization(axis=1, name='proj')(proj)
    return SupConModel(inputs=inputs, outputs=[cls_out, proj_norm],
                       supcon_temp=supcon_temp, supcon_lambda=supcon_lambda)


# ======================================================================
# TIER-1 (FUSED) FACTORY FUNCTIONS (reuse MTL backbone builders)
# ======================================================================

def create_cnn_gru_dual_supcon_model(T, n_classes):
    inputs = tf.keras.layers.Input(shape=(T, 1))
    return _supcon_wrap(inputs, _build_cnn_gru_dual_branches_mtl(inputs), n_classes)


def create_cnn_gru_dual_attn_recon_supcon_model(k_plus_1, T, n_classes, attn_dim=16):
    stack_input = tf.keras.layers.Input(shape=(k_plus_1, T), name='neighbor_stack_input')
    return _supcon_wrap(stack_input,
                        _build_cnn_gru_dual_attn_recon_embedding_mtl(stack_input, T, attn_dim),
                        n_classes)


# ======================================================================
# TIER-3 (BRANCH) MODEL CLASS
# ======================================================================

@tf.keras.utils.register_keras_serializable(package='supcon')
class SupConBranch3STModel(tf.keras.Model):
    """v3 ST: 0.7*CE + 0.1*(L_SC_cnn + L_SC_seq + L_SC_fused). 4 outputs."""
    def __init__(self, *args, supcon_temp=SUPCON_TEMP, supcon_lambda_each=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.supcon_temp        = supcon_temp
        self.supcon_lambda_each = supcon_lambda_each  # 0.1 per head; CE = 0.7

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        y_cls = y_dict['cls_out']
        with tf.GradientTape() as tape:
            cls_out, cnn_proj, seq_proj, fused_proj = self(x, training=True)
            ce  = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_cls, cls_out))
            sc  = (supcon_loss(cnn_proj,   y_cls, self.supcon_temp)
                   + supcon_loss(seq_proj,   y_cls, self.supcon_temp)
                   + supcon_loss(fused_proj, y_cls, self.supcon_temp))
            loss = (1.0 - 3 * self.supcon_lambda_each) * ce + self.supcon_lambda_each * sc
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y_cls, cls_out)
        return {m.name: m.result() for m in self.metrics} | {'loss': loss, 'cls_ce': ce, 'supcon': sc}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        y_cls = y_dict['cls_out']
        cls_out, cnn_proj, seq_proj, fused_proj = self(x, training=False)
        ce  = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_cls, cls_out))
        sc  = (supcon_loss(cnn_proj,   y_cls, self.supcon_temp)
               + supcon_loss(seq_proj,   y_cls, self.supcon_temp)
               + supcon_loss(fused_proj, y_cls, self.supcon_temp))
        loss = (1.0 - 3 * self.supcon_lambda_each) * ce + self.supcon_lambda_each * sc
        self.compiled_metrics.update_state(y_cls, cls_out)
        return {m.name: m.result() for m in self.metrics} | {'loss': loss, 'cls_ce': ce, 'supcon': sc}


# ======================================================================
# TIER-3 (BRANCH) WRAP HELPER
# ======================================================================

def _proj_head(embedding, name_prefix):
    """Shared projection head: Dense(64,relu) → L2-normalize."""
    h = tf.keras.layers.Dense(64, activation='relu', name=f'{name_prefix}_proj_hidden')(embedding)
    return tf.keras.layers.UnitNormalization(axis=1, name=f'{name_prefix}_proj')(h)


def _branch3_supcon_wrap(inputs, cnn_emb, seq_emb, fused, n_classes):
    """3 projection heads (cnn + seq + fused). Outputs: [cls_out, cnn_proj, seq_proj, fused_proj]."""
    cls_feat   = tf.keras.layers.Dense(16, activation='relu', name='cls_feat')(fused)
    cls_out    = tf.keras.layers.Dense(n_classes, activation='softmax', name='cls_out')(cls_feat)
    cnn_proj   = _proj_head(cnn_emb, 'cnn')
    seq_proj   = _proj_head(seq_emb, 'seq')
    fused_proj = _proj_head(fused,   'fused')
    return SupConBranch3STModel(inputs=inputs, outputs=[cls_out, cnn_proj, seq_proj, fused_proj])


# ======================================================================
# TIER-3 (BRANCH) FACTORY FUNCTIONS
# ======================================================================

def create_cnn_gru_dual_supcon3_model(T, n_classes):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    cnn_emb, gru_emb, fused = _build_cnn_gru_dual_branches_mtl(inputs, return_branches=True)
    return _branch3_supcon_wrap(inputs, cnn_emb, gru_emb, fused, n_classes)


def create_cnn_gru_dual_attn_recon_supcon3_model(k_plus_1, T, n_classes, attn_dim=16):
    stack_input = tf.keras.layers.Input(shape=(k_plus_1, T), name='neighbor_stack_input')
    cnn_emb, gru_emb, fused = _build_cnn_gru_dual_attn_recon_embedding_mtl(
        stack_input, T, attn_dim, return_branches=True)
    return _branch3_supcon_wrap(stack_input, cnn_emb, gru_emb, fused, n_classes)
