import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from model_utils_mtl import (
    MTLModel, REG_SENTINEL,
    _build_cnn_backbone_mtl, _build_gru_backbone_mtl,
    _build_transformer_backbone_mtl,
    _build_cnn_gru_dual_branches_mtl, _build_cnn_trans_dual_branches_mtl,
    _build_cnn_gru_dual_attn_recon_embedding_mtl,
)

SUPCON_TEMP   = 0.1
SUPCON_LAMBDA = 0.2

# ST + SupCon (classification + contrastive; no regression)
SUPCON_MODEL_KEYS = [
    'cnn_supcon', 'gru_supcon', 'transformer_supcon',
    'cnn_gru_dual_supcon', 'cnn_trans_dual_supcon',
    'cnn_gru_dual_cosine_recon_supcon', 'cnn_gru_dual_attn_recon_supcon',
]

# MTL + SupCon (classification + regression + contrastive)
SUPCON_MTL_MODEL_KEYS = [
    'cnn_supcon_mtl', 'gru_supcon_mtl', 'transformer_supcon_mtl',
    'cnn_gru_dual_supcon_mtl', 'cnn_trans_dual_supcon_mtl',
    'cnn_gru_dual_cosine_recon_supcon_mtl', 'cnn_gru_dual_attn_recon_supcon_mtl',
]

ALL_SUPCON_KEYS = SUPCON_MODEL_KEYS + SUPCON_MTL_MODEL_KEYS


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


class SupConMTLModel(MTLModel):
    """MTL + SupCon: UW-SO (CE + MSE) + supervised contrastive loss."""
    def __init__(self, *args, supcon_temp=SUPCON_TEMP, supcon_lambda=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.supcon_temp   = supcon_temp
        self.supcon_lambda = supcon_lambda  # smaller default: MTL already has 2 losses

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            cls_out, reg_out, proj_norm = self(x, training=True)
            mtl_loss, ce, mse = self._compute_loss(
                y_dict['cls_out'], y_dict['reg_out'], cls_out, reg_out)
            sc   = supcon_loss(proj_norm, y_dict['cls_out'], self.supcon_temp)
            loss = mtl_loss + self.supcon_lambda * sc
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y_dict['cls_out'], cls_out)
        return ({m.name: m.result() for m in self.metrics}
                | {'loss': loss, 'cls_ce': ce, 'reg_mse': mse, 'supcon': sc, 'log_T': self.log_T})

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        cls_out, reg_out, proj_norm = self(x, training=False)
        mtl_loss, ce, mse = self._compute_loss(
            y_dict['cls_out'], y_dict['reg_out'], cls_out, reg_out)
        sc   = supcon_loss(proj_norm, y_dict['cls_out'], self.supcon_temp)
        loss = mtl_loss + self.supcon_lambda * sc
        self.compiled_metrics.update_state(y_dict['cls_out'], cls_out)
        return ({m.name: m.result() for m in self.metrics}
                | {'loss': loss, 'cls_ce': ce, 'reg_mse': mse, 'supcon': sc, 'log_T': self.log_T})


# ======================================================================
# WRAP HELPERS
# ======================================================================

def _supcon_wrap(inputs, embedding, n_classes,
                 supcon_temp=SUPCON_TEMP, supcon_lambda=SUPCON_LAMBDA):
    """Attach cls head + projection head to embedding, return SupConModel."""
    cls_feat  = tf.keras.layers.Dense(16, activation='relu',  name='cls_feat')(embedding)
    cls_out   = tf.keras.layers.Dense(n_classes, activation='softmax', name='cls_out')(cls_feat)
    proj      = tf.keras.layers.Dense(64, activation='relu',  name='proj_hidden')(embedding)
    proj_norm = tf.keras.layers.Lambda(
        lambda z: tf.math.l2_normalize(z, axis=1), name='proj')(proj)
    return SupConModel(inputs=inputs, outputs=[cls_out, proj_norm],
                       supcon_temp=supcon_temp, supcon_lambda=supcon_lambda)


def _supcon_mtl_wrap(inputs, embedding, n_classes,
                     supcon_temp=SUPCON_TEMP, supcon_lambda=0.1):
    """Attach cls + reg heads + projection head, return SupConMTLModel."""
    cls_feat  = tf.keras.layers.Dense(16, activation='relu',   name='cls_feat')(embedding)
    cls_out   = tf.keras.layers.Dense(n_classes, activation='softmax', name='cls_out')(cls_feat)
    reg_feat  = tf.keras.layers.Dense(16, activation='relu',   name='reg_feat')(embedding)
    reg_h     = tf.keras.layers.Dense(8,  activation='relu',   name='reg_hidden')(reg_feat)
    reg_out   = tf.keras.layers.Dense(1,  activation='linear', name='reg_out')(reg_h)
    proj      = tf.keras.layers.Dense(64, activation='relu',   name='proj_hidden')(embedding)
    proj_norm = tf.keras.layers.Lambda(
        lambda z: tf.math.l2_normalize(z, axis=1), name='proj')(proj)
    return SupConMTLModel(inputs=inputs, outputs=[cls_out, reg_out, proj_norm],
                          supcon_temp=supcon_temp, supcon_lambda=supcon_lambda)


# ======================================================================
# ST + SUPCON FACTORY FUNCTIONS (all reuse MTL backbone builders)
# ======================================================================

def create_cnn_supcon_model(T, n_classes):
    inputs = tf.keras.layers.Input(shape=(T, 1))
    return _supcon_wrap(inputs, _build_cnn_backbone_mtl(inputs), n_classes)


def create_gru_supcon_model(T, n_classes):
    inputs = tf.keras.layers.Input(shape=(T, 1))
    return _supcon_wrap(inputs, _build_gru_backbone_mtl(inputs), n_classes)


def create_transformer_supcon_model(T, n_classes, head_size=32, num_heads=2,
                                    ff_dim=32, num_blocks=2, dropout=0.1):
    inputs = tf.keras.layers.Input(shape=(T, 1))
    return _supcon_wrap(
        inputs,
        _build_transformer_backbone_mtl(inputs, head_size, num_heads, ff_dim, num_blocks, dropout),
        n_classes)


def create_cnn_gru_dual_supcon_model(T, n_classes):
    inputs = tf.keras.layers.Input(shape=(T, 1))
    return _supcon_wrap(inputs, _build_cnn_gru_dual_branches_mtl(inputs), n_classes)


def create_cnn_trans_dual_supcon_model(T, n_classes, head_size=32, num_heads=2,
                                       ff_dim=32, num_blocks=2, dropout=0.1):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    return _supcon_wrap(
        inputs,
        _build_cnn_trans_dual_branches_mtl(inputs, head_size, num_heads, ff_dim, num_blocks, dropout),
        n_classes)


# ======================================================================
# MTL + SUPCON FACTORY FUNCTIONS (same backbones → SupConMTLModel)
# ======================================================================

def create_cnn_supcon_mtl_model(T, n_classes):
    inputs = tf.keras.layers.Input(shape=(T, 1))
    return _supcon_mtl_wrap(inputs, _build_cnn_backbone_mtl(inputs), n_classes)


def create_gru_supcon_mtl_model(T, n_classes):
    inputs = tf.keras.layers.Input(shape=(T, 1))
    return _supcon_mtl_wrap(inputs, _build_gru_backbone_mtl(inputs), n_classes)


def create_transformer_supcon_mtl_model(T, n_classes, head_size=32, num_heads=2,
                                        ff_dim=32, num_blocks=2, dropout=0.1):
    inputs = tf.keras.layers.Input(shape=(T, 1))
    return _supcon_mtl_wrap(
        inputs,
        _build_transformer_backbone_mtl(inputs, head_size, num_heads, ff_dim, num_blocks, dropout),
        n_classes)


def create_cnn_gru_dual_supcon_mtl_model(T, n_classes):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    return _supcon_mtl_wrap(inputs, _build_cnn_gru_dual_branches_mtl(inputs), n_classes)


def create_cnn_trans_dual_supcon_mtl_model(T, n_classes, head_size=32, num_heads=2,
                                            ff_dim=32, num_blocks=2, dropout=0.1):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    return _supcon_mtl_wrap(
        inputs,
        _build_cnn_trans_dual_branches_mtl(inputs, head_size, num_heads, ff_dim, num_blocks, dropout),
        n_classes)


# ======================================================================
# SPATIAL RECONSTRUCTION + SUPCON FACTORIES
# Input for cosine_recon variants: (T, 1) scalar curve (post-reconstruction).
# Input for attn_recon variants:   (k+1, T) neighbor stack.
# ======================================================================

def create_cnn_gru_dual_cosine_recon_supcon_model(T, n_classes):
    inputs = tf.keras.layers.Input(shape=(T, 1))
    return _supcon_wrap(inputs, _build_cnn_gru_dual_branches_mtl(inputs), n_classes)


def create_cnn_gru_dual_attn_recon_supcon_model(k_plus_1, T, n_classes, attn_dim=16):
    stack_input = tf.keras.layers.Input(shape=(k_plus_1, T), name='neighbor_stack_input')
    return _supcon_wrap(stack_input,
                        _build_cnn_gru_dual_attn_recon_embedding_mtl(stack_input, T, attn_dim),
                        n_classes)


def create_cnn_gru_dual_cosine_recon_supcon_mtl_model(T, n_classes):
    inputs = tf.keras.layers.Input(shape=(T, 1))
    return _supcon_mtl_wrap(inputs, _build_cnn_gru_dual_branches_mtl(inputs), n_classes)


def create_cnn_gru_dual_attn_recon_supcon_mtl_model(k_plus_1, T, n_classes, attn_dim=16):
    stack_input = tf.keras.layers.Input(shape=(k_plus_1, T), name='neighbor_stack_input')
    return _supcon_mtl_wrap(stack_input,
                            _build_cnn_gru_dual_attn_recon_embedding_mtl(stack_input, T, attn_dim),
                            n_classes)
