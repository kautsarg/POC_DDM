"""Multi-label model utilities for the multiplex pipeline.

Nothing in main/ is modified. This module imports from main/ and
provides thin wrappers / subclasses for multi-label classification.
"""
import os
import gc
import sys
import time
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score, hamming_loss, f1_score

# ── locate main/ and its sub-dirs ──────────────────────────────────────────
# This file lives at:  main/multiplex/utils/model_training/model_utils_multilabel.py
#                                 (3)    (2)       (1)                (0)
_THIS              = Path(__file__).resolve()
_MULTIPLEX_DIR     = _THIS.parent.parent.parent   # main/multiplex/
_MAIN_DIR          = _MULTIPLEX_DIR.parent        # main/
_MAIN_UTILS        = _MAIN_DIR / "utils"
_MAIN_MODEL_UTILS  = _MAIN_UTILS / "model_training"

for _p in [str(_MAIN_MODEL_UTILS), str(_MAIN_UTILS), str(_MAIN_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_utils_mtl import (
    MTLModel, REG_SENTINEL,
    _normalize_concentration, _inverse_normalize_concentration,
    _build_cnn_backbone_mtl, _build_gru_backbone_mtl, _build_transformer_backbone_mtl,
    _build_cnn_gru_dual_branches_mtl, _build_cnn_trans_dual_branches_mtl,
)
from model_utils_supcon import SUPCON_TEMP, SUPCON_LAMBDA, _proj_head
from model_utils_rcfd import _build_rcfd_backbone
from model_utils import (
    create_cnn_model, create_gru_model, create_transformer_model,
    create_cnn_gru_dual_model, create_cnn_transformer_dual_model,
)
from safe_io import safe_keras_save


# ======================================================================
# ENCODING HELPERS
# ======================================================================

def encode_multilabel_for_training(labels_list, all_targets):
    """Binarize a list-of-lists label set.

    Returns:
        y_binary    (N, n_targets) int8 binary indicator matrix
        y_combo_int (N,) int -- encoded combination string for StratifiedKFold
        mlb         fitted MultiLabelBinarizer
        combo_enc   fitted LabelEncoder on combination strings
    """
    mlb = MultiLabelBinarizer(classes=sorted(all_targets))
    y_binary = mlb.fit_transform(labels_list).astype(np.int8)

    combo_strings = ['_'.join(sorted(lbl)) for lbl in labels_list]
    combo_enc = LabelEncoder()
    y_combo_int = combo_enc.fit_transform(combo_strings)
    return y_binary, y_combo_int, mlb, combo_enc


# ======================================================================
# CONTINUOUS JACCARD POSITIVE-PAIR WEIGHT MATRIX
# ======================================================================

def _jaccard_weight_matrix(y_binary):
    """Pairwise Jaccard similarity for SupCon weighting: (B, B) float in [0,1].

    Diagonal zeroed (no self-pairs). Identical label vectors -> 1.0,
    partial overlap -> fractional, disjoint -> 0.0.
    """
    y            = tf.cast(y_binary, tf.float32)
    intersection = tf.matmul(y, y, transpose_b=True)
    row_sums     = tf.reduce_sum(y, axis=1)
    union        = tf.expand_dims(row_sums, 1) + tf.expand_dims(row_sums, 0) - intersection
    return (intersection / (union + 1e-8)) * (1.0 - tf.eye(tf.shape(y)[0]))


def _supcon_loss_jaccard(embeddings, pos_mask, temp=SUPCON_TEMP):
    """Weighted SupCon loss using continuous Jaccard pos_mask.

    embeddings : (N, D) L2-normalised
    pos_mask   : (N, N) float [0, 1], diagonal=0
    """
    not_self  = 1.0 - tf.eye(tf.shape(embeddings)[0])
    sim       = tf.matmul(embeddings, embeddings, transpose_b=True) / temp
    sim_max   = tf.reduce_max(sim, axis=1, keepdims=True)
    exp_sim   = tf.exp(sim - sim_max)
    log_denom = tf.math.log(tf.reduce_sum(exp_sim * not_self, axis=1, keepdims=True) + 1e-8)
    log_prob  = (sim - sim_max) - log_denom
    pos_sum   = tf.reduce_sum(pos_mask, axis=1)
    has_pos   = tf.cast(pos_sum > 0, tf.float32)
    per_anchor = -tf.reduce_sum(log_prob * pos_mask, axis=1) / (pos_sum + 1e-8)
    return tf.reduce_mean(per_anchor * has_pos)


# ======================================================================
# STANDARD MULTI-LABEL WRAPPER (functional models, no custom train_step)
# ======================================================================

@tf.keras.utils.register_keras_serializable(package='ml_standard')
class _StandardMultiLabelModel(tf.keras.Model):
    """Wraps a functional model with multi-label BCE train/test steps."""

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        y_bin = tf.cast(y_dict['cls_out'], tf.float32)
        with tf.GradientTape() as tape:
            cls_out = self(x, training=True)
            loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_bin, cls_out))
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return {'loss': loss}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        y_bin = tf.cast(y_dict['cls_out'], tf.float32)
        cls_out = self(x, training=False)
        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_bin, cls_out))
        return {'loss': loss}


def adapt_for_multilabel(model, n_targets):
    """Replace last softmax Dense with sigmoid Dense(n_targets).

    Returns a new _StandardMultiLabelModel; the original is unchanged.
    """
    for layer in reversed(model.layers):
        if (hasattr(layer, 'activation')
                and layer.activation.__name__ == 'softmax'):
            new_out = tf.keras.layers.Dense(
                n_targets, activation='sigmoid', name='cls_out')(layer.input)
            return _StandardMultiLabelModel(inputs=model.inputs, outputs=new_out)
    raise ValueError("No softmax output layer found to replace")


# ======================================================================
# MULTI-LABEL SUPCON MODEL CLASSES (ST -- no regression head)
# ======================================================================

@tf.keras.utils.register_keras_serializable(package='ml_supcon')
class MultiLabelSupConModel(tf.keras.Model):
    """SC1 multi-label ST: BCE + continuous-Jaccard contrastive loss."""
    def __init__(self, *args, supcon_temp=SUPCON_TEMP, supcon_lambda=SUPCON_LAMBDA, **kwargs):
        super().__init__(*args, **kwargs)
        self.supcon_temp   = supcon_temp
        self.supcon_lambda = supcon_lambda

    def _step(self, x, y_dict, training):
        y_bin    = tf.cast(y_dict['cls_out'], tf.float32)
        cls_out, proj_norm = self(x, training=training)
        bce      = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_bin, cls_out))
        pos_mask = _jaccard_weight_matrix(y_bin)
        sc       = _supcon_loss_jaccard(proj_norm, pos_mask, self.supcon_temp)
        loss     = (1.0 - self.supcon_lambda) * bce + self.supcon_lambda * sc
        return loss, bce, sc

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            loss, bce, sc = self._step(x, y_dict, training=True)
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return {'loss': loss, 'cls_bce': bce, 'supcon': sc}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        loss, bce, sc = self._step(x, y_dict, training=False)
        return {'loss': loss, 'cls_bce': bce, 'supcon': sc}


@tf.keras.utils.register_keras_serializable(package='ml_supcon')
class MultiLabelSupConBranch2STModel(tf.keras.Model):
    """SC2 multi-label ST: BCE + Jaccard SupCon on CNN + seq branches."""
    def __init__(self, *args, supcon_temp=SUPCON_TEMP, supcon_lambda_each=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.supcon_temp        = supcon_temp
        self.supcon_lambda_each = supcon_lambda_each

    def _step(self, x, y_dict, training):
        y_bin    = tf.cast(y_dict['cls_out'], tf.float32)
        cls_out, cnn_proj, seq_proj = self(x, training=training)
        bce      = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_bin, cls_out))
        pos_mask = _jaccard_weight_matrix(y_bin)
        sc       = (_supcon_loss_jaccard(cnn_proj, pos_mask, self.supcon_temp)
                    + _supcon_loss_jaccard(seq_proj, pos_mask, self.supcon_temp))
        loss     = (1.0 - 2 * self.supcon_lambda_each) * bce + self.supcon_lambda_each * sc
        return loss, bce, sc

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            loss, bce, sc = self._step(x, y_dict, training=True)
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return {'loss': loss, 'cls_bce': bce, 'supcon': sc}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        loss, bce, sc = self._step(x, y_dict, training=False)
        return {'loss': loss, 'cls_bce': bce, 'supcon': sc}


@tf.keras.utils.register_keras_serializable(package='ml_supcon')
class MultiLabelSupConBranch3STModel(tf.keras.Model):
    """SC3 multi-label ST: BCE + Jaccard SupCon on CNN + seq + fused branches."""
    def __init__(self, *args, supcon_temp=SUPCON_TEMP, supcon_lambda_each=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.supcon_temp        = supcon_temp
        self.supcon_lambda_each = supcon_lambda_each

    def _step(self, x, y_dict, training):
        y_bin    = tf.cast(y_dict['cls_out'], tf.float32)
        cls_out, cnn_proj, seq_proj, fused_proj = self(x, training=training)
        bce      = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_bin, cls_out))
        pos_mask = _jaccard_weight_matrix(y_bin)
        sc       = (_supcon_loss_jaccard(cnn_proj,   pos_mask, self.supcon_temp)
                    + _supcon_loss_jaccard(seq_proj,   pos_mask, self.supcon_temp)
                    + _supcon_loss_jaccard(fused_proj, pos_mask, self.supcon_temp))
        loss     = (1.0 - 3 * self.supcon_lambda_each) * bce + self.supcon_lambda_each * sc
        return loss, bce, sc

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            loss, bce, sc = self._step(x, y_dict, training=True)
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return {'loss': loss, 'cls_bce': bce, 'supcon': sc}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        loss, bce, sc = self._step(x, y_dict, training=False)
        return {'loss': loss, 'cls_bce': bce, 'supcon': sc}


# ======================================================================
# MULTI-LABEL MTL BASE (BCE + MSE regression, UW-SO weighting)
# ======================================================================

@tf.keras.utils.register_keras_serializable(package='ml_rcfd')
class MultiLabelMTLModel(MTLModel):
    """MTL base with binary_crossentropy cls loss (multi-label sigmoid head)."""

    def _compute_loss(self, y_binary, y_reg, cls_out, reg_out):
        bce     = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
            tf.cast(y_binary, tf.float32), cls_out))
        mask    = tf.cast(tf.not_equal(y_reg, self.reg_sentinel), tf.float32)
        n_valid = tf.reduce_sum(mask)
        mse     = (tf.reduce_sum(mask * tf.square(y_reg - reg_out[:, 0]))
                   / (n_valid + 1e-8))

        eps     = 1e-8
        inv_bce = 1.0 / (tf.stop_gradient(bce) + eps)
        inv_mse = tf.cond(n_valid > 0,
                          lambda: 1.0 / (tf.stop_gradient(mse) + eps),
                          lambda: tf.constant(0.0))
        Z       = inv_bce + inv_mse + eps
        loss    = tf.exp(-self.log_T) * (inv_bce / Z * bce + inv_mse / Z * mse) + self.log_T
        return loss, bce, mse

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            cls_out, reg_out = self(x, training=True)
            loss, bce, mse = self._compute_loss(
                y_dict['cls_out'], y_dict['reg_out'], cls_out, reg_out)
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return {'loss': loss, 'cls_bce': bce, 'reg_mse': mse, 'log_T': self.log_T}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        cls_out, reg_out = self(x, training=False)
        loss, bce, mse = self._compute_loss(
            y_dict['cls_out'], y_dict['reg_out'], cls_out, reg_out)
        return {'loss': loss, 'cls_bce': bce, 'reg_mse': mse, 'log_T': self.log_T}


# ======================================================================
# MULTI-LABEL RCFD MODEL CLASSES
# ======================================================================

@tf.keras.utils.register_keras_serializable(package='ml_rcfd')
class MultiLabelRCFDModel(MultiLabelMTLModel):
    """Base RCFD multi-label: UW-SO(BCE+MSE). sigmoid cls head."""
    pass


@tf.keras.utils.register_keras_serializable(package='ml_rcfd_sc1')
class MultiLabelRCFDSupConMTLModel(MultiLabelMTLModel):
    """RCFD SC1 multi-label: UW-SO(BCE+MSE) + Jaccard SupCon on z_cond."""
    def __init__(self, *args, supcon_temp=SUPCON_TEMP, supcon_lambda=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.supcon_temp   = supcon_temp
        self.supcon_lambda = supcon_lambda

    def _sc_step(self, x, y_dict, training):
        y_bin = tf.cast(y_dict['cls_out'], tf.float32)
        cls_out, reg_out, proj_norm = self(x, training=training)
        mtl_loss, bce, mse = self._compute_loss(y_bin, y_dict['reg_out'], cls_out, reg_out)
        pos_mask = _jaccard_weight_matrix(y_bin)
        sc = _supcon_loss_jaccard(proj_norm, pos_mask, self.supcon_temp)
        return mtl_loss + self.supcon_lambda * sc, bce, mse, sc

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            loss, bce, mse, sc = self._sc_step(x, y_dict, training=True)
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return {'loss': loss, 'cls_bce': bce, 'reg_mse': mse, 'supcon': sc, 'log_T': self.log_T}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        loss, bce, mse, sc = self._sc_step(x, y_dict, training=False)
        return {'loss': loss, 'cls_bce': bce, 'reg_mse': mse, 'supcon': sc, 'log_T': self.log_T}


@tf.keras.utils.register_keras_serializable(package='ml_rcfd_sc2')
class MultiLabelRCFDBranch2MTLModel(MultiLabelMTLModel):
    """RCFD SC2 multi-label: UW-SO(BCE+MSE) + Jaccard SupCon on CNN+seq branches."""
    def __init__(self, *args, supcon_temp=SUPCON_TEMP, supcon_lambda_each=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.supcon_temp        = supcon_temp
        self.supcon_lambda_each = supcon_lambda_each

    def _sc_step(self, x, y_dict, training):
        y_bin = tf.cast(y_dict['cls_out'], tf.float32)
        cls_out, reg_out, cnn_proj, seq_proj = self(x, training=training)
        mtl_loss, bce, mse = self._compute_loss(y_bin, y_dict['reg_out'], cls_out, reg_out)
        pos_mask = _jaccard_weight_matrix(y_bin)
        sc = (_supcon_loss_jaccard(cnn_proj, pos_mask, self.supcon_temp)
              + _supcon_loss_jaccard(seq_proj, pos_mask, self.supcon_temp))
        return mtl_loss + self.supcon_lambda_each * sc, bce, mse, sc

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            loss, bce, mse, sc = self._sc_step(x, y_dict, training=True)
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return {'loss': loss, 'cls_bce': bce, 'reg_mse': mse, 'supcon': sc, 'log_T': self.log_T}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        loss, bce, mse, sc = self._sc_step(x, y_dict, training=False)
        return {'loss': loss, 'cls_bce': bce, 'reg_mse': mse, 'supcon': sc, 'log_T': self.log_T}


@tf.keras.utils.register_keras_serializable(package='ml_rcfd_sc3')
class MultiLabelRCFDBranch3MTLModel(MultiLabelMTLModel):
    """RCFD SC3 multi-label: UW-SO(BCE+MSE) + Jaccard SupCon on all 3 branches."""
    def __init__(self, *args, supcon_temp=SUPCON_TEMP, supcon_lambda_each=0.033, **kwargs):
        super().__init__(*args, **kwargs)
        self.supcon_temp        = supcon_temp
        self.supcon_lambda_each = supcon_lambda_each

    def _sc_step(self, x, y_dict, training):
        y_bin = tf.cast(y_dict['cls_out'], tf.float32)
        cls_out, reg_out, cnn_proj, seq_proj, fused_proj = self(x, training=training)
        mtl_loss, bce, mse = self._compute_loss(y_bin, y_dict['reg_out'], cls_out, reg_out)
        pos_mask = _jaccard_weight_matrix(y_bin)
        sc = (_supcon_loss_jaccard(cnn_proj,   pos_mask, self.supcon_temp)
              + _supcon_loss_jaccard(seq_proj,   pos_mask, self.supcon_temp)
              + _supcon_loss_jaccard(fused_proj, pos_mask, self.supcon_temp))
        return mtl_loss + self.supcon_lambda_each * sc, bce, mse, sc

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            loss, bce, mse, sc = self._sc_step(x, y_dict, training=True)
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return {'loss': loss, 'cls_bce': bce, 'reg_mse': mse, 'supcon': sc, 'log_T': self.log_T}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        loss, bce, mse, sc = self._sc_step(x, y_dict, training=False)
        return {'loss': loss, 'cls_bce': bce, 'reg_mse': mse, 'supcon': sc, 'log_T': self.log_T}


# ======================================================================
# MODEL FACTORY FUNCTIONS
# ======================================================================

def _ml_cls_head(z, n_targets):
    h = tf.keras.layers.Dense(16, activation='relu', name='cls_feat')(z)
    return tf.keras.layers.Dense(n_targets, activation='sigmoid', name='cls_out')(h)


def create_ml_cnn_gru_dual_model(T, n_targets):
    """Standard dual-branch model with sigmoid multi-label head."""
    base = create_cnn_gru_dual_model(T, n_targets)
    return adapt_for_multilabel(base, n_targets)


def create_ml_cnn_gru_dual_supcon_model(T, n_targets):
    """Dual-branch + SC1 (Jaccard) multi-label ST."""
    inputs    = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    embedding = _build_cnn_gru_dual_branches_mtl(inputs)
    cls_out   = _ml_cls_head(embedding, n_targets)
    proj_norm = _proj_head(embedding, 'fused')
    return MultiLabelSupConModel(inputs=inputs, outputs=[cls_out, proj_norm])


def create_ml_cnn_gru_dual_supcon2_model(T, n_targets):
    """Dual-branch + SC2 (Jaccard on CNN+seq) multi-label ST."""
    inputs               = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    cnn_emb, seq_emb, z = _build_cnn_gru_dual_branches_mtl(inputs, return_branches=True)
    cls_out              = _ml_cls_head(z, n_targets)
    cnn_proj             = _proj_head(cnn_emb, 'cnn')
    seq_proj             = _proj_head(seq_emb, 'seq')
    return MultiLabelSupConBranch2STModel(inputs=inputs, outputs=[cls_out, cnn_proj, seq_proj])


def create_ml_cnn_gru_dual_supcon3_model(T, n_targets):
    """Dual-branch + SC3 (Jaccard on CNN+seq+fused) multi-label ST."""
    inputs               = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    cnn_emb, seq_emb, z = _build_cnn_gru_dual_branches_mtl(inputs, return_branches=True)
    cls_out              = _ml_cls_head(z, n_targets)
    cnn_proj             = _proj_head(cnn_emb, 'cnn')
    seq_proj             = _proj_head(seq_emb, 'seq')
    fused_proj           = _proj_head(z, 'fused')
    return MultiLabelSupConBranch3STModel(inputs=inputs, outputs=[cls_out, cnn_proj, seq_proj, fused_proj])


def create_ml_gru_rcfd_cgd_model(T, n_targets):
    inputs          = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z_cond, reg_out = _build_rcfd_backbone(inputs, 'gru', 'cgd')
    cls_out         = _ml_cls_head(z_cond, n_targets)
    return MultiLabelRCFDModel(inputs=inputs, outputs=[cls_out, reg_out])


def create_ml_gru_rcfd_cgd_supcon_mtl_model(T, n_targets):
    inputs          = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z_cond, reg_out = _build_rcfd_backbone(inputs, 'gru', 'cgd')
    cls_out         = _ml_cls_head(z_cond, n_targets)
    proj_norm       = _proj_head(z_cond, 'fused')
    return MultiLabelRCFDSupConMTLModel(inputs=inputs, outputs=[cls_out, reg_out, proj_norm])


def create_ml_gru_rcfd_cgd_supcon2_mtl_model(T, n_targets):
    inputs                            = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z_cond, reg_out, cnn_emb, seq_emb = _build_rcfd_backbone(inputs, 'gru', 'cgd', return_branches=True)
    cls_out                           = _ml_cls_head(z_cond, n_targets)
    cnn_proj                          = _proj_head(cnn_emb, 'cnn')
    seq_proj                          = _proj_head(seq_emb, 'seq')
    return MultiLabelRCFDBranch2MTLModel(inputs=inputs, outputs=[cls_out, reg_out, cnn_proj, seq_proj])


def create_ml_gru_rcfd_cgd_supcon3_mtl_model(T, n_targets):
    inputs                            = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z_cond, reg_out, cnn_emb, seq_emb = _build_rcfd_backbone(inputs, 'gru', 'cgd', return_branches=True)
    cls_out                           = _ml_cls_head(z_cond, n_targets)
    cnn_proj                          = _proj_head(cnn_emb, 'cnn')
    seq_proj                          = _proj_head(seq_emb, 'seq')
    fused_proj                        = _proj_head(z_cond, 'fused')
    return MultiLabelRCFDBranch3MTLModel(inputs=inputs, outputs=[cls_out, reg_out, cnn_proj, seq_proj, fused_proj])


# -- gru_rcfd_ctd (GRU early encoder + CNN+Trans dual) ----------------------

def create_ml_gru_rcfd_ctd_model(T, n_targets):
    inputs          = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z_cond, reg_out = _build_rcfd_backbone(inputs, 'gru', 'ctd')
    cls_out         = _ml_cls_head(z_cond, n_targets)
    return MultiLabelRCFDModel(inputs=inputs, outputs=[cls_out, reg_out])


def create_ml_gru_rcfd_ctd_supcon_mtl_model(T, n_targets):
    inputs          = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z_cond, reg_out = _build_rcfd_backbone(inputs, 'gru', 'ctd')
    cls_out         = _ml_cls_head(z_cond, n_targets)
    proj_norm       = _proj_head(z_cond, 'fused')
    return MultiLabelRCFDSupConMTLModel(inputs=inputs, outputs=[cls_out, reg_out, proj_norm])


def create_ml_gru_rcfd_ctd_supcon2_mtl_model(T, n_targets):
    inputs                            = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z_cond, reg_out, cnn_emb, seq_emb = _build_rcfd_backbone(inputs, 'gru', 'ctd', return_branches=True)
    cls_out                           = _ml_cls_head(z_cond, n_targets)
    cnn_proj                          = _proj_head(cnn_emb, 'cnn')
    seq_proj                          = _proj_head(seq_emb, 'seq')
    return MultiLabelRCFDBranch2MTLModel(inputs=inputs, outputs=[cls_out, reg_out, cnn_proj, seq_proj])


def create_ml_gru_rcfd_ctd_supcon3_mtl_model(T, n_targets):
    inputs                            = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z_cond, reg_out, cnn_emb, seq_emb = _build_rcfd_backbone(inputs, 'gru', 'ctd', return_branches=True)
    cls_out                           = _ml_cls_head(z_cond, n_targets)
    cnn_proj                          = _proj_head(cnn_emb, 'cnn')
    seq_proj                          = _proj_head(seq_emb, 'seq')
    fused_proj                        = _proj_head(z_cond, 'fused')
    return MultiLabelRCFDBranch3MTLModel(inputs=inputs, outputs=[cls_out, reg_out, cnn_proj, seq_proj, fused_proj])


# -- trans_rcfd_cgd (Transformer early encoder + CNN+GRU dual) ---------------

def create_ml_trans_rcfd_cgd_model(T, n_targets):
    inputs          = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z_cond, reg_out = _build_rcfd_backbone(inputs, 'transformer', 'cgd')
    cls_out         = _ml_cls_head(z_cond, n_targets)
    return MultiLabelRCFDModel(inputs=inputs, outputs=[cls_out, reg_out])


def create_ml_trans_rcfd_cgd_supcon_mtl_model(T, n_targets):
    inputs          = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z_cond, reg_out = _build_rcfd_backbone(inputs, 'transformer', 'cgd')
    cls_out         = _ml_cls_head(z_cond, n_targets)
    proj_norm       = _proj_head(z_cond, 'fused')
    return MultiLabelRCFDSupConMTLModel(inputs=inputs, outputs=[cls_out, reg_out, proj_norm])


def create_ml_trans_rcfd_cgd_supcon2_mtl_model(T, n_targets):
    inputs                            = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z_cond, reg_out, cnn_emb, seq_emb = _build_rcfd_backbone(inputs, 'transformer', 'cgd', return_branches=True)
    cls_out                           = _ml_cls_head(z_cond, n_targets)
    cnn_proj                          = _proj_head(cnn_emb, 'cnn')
    seq_proj                          = _proj_head(seq_emb, 'seq')
    return MultiLabelRCFDBranch2MTLModel(inputs=inputs, outputs=[cls_out, reg_out, cnn_proj, seq_proj])


def create_ml_trans_rcfd_cgd_supcon3_mtl_model(T, n_targets):
    inputs                            = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z_cond, reg_out, cnn_emb, seq_emb = _build_rcfd_backbone(inputs, 'transformer', 'cgd', return_branches=True)
    cls_out                           = _ml_cls_head(z_cond, n_targets)
    cnn_proj                          = _proj_head(cnn_emb, 'cnn')
    seq_proj                          = _proj_head(seq_emb, 'seq')
    fused_proj                        = _proj_head(z_cond, 'fused')
    return MultiLabelRCFDBranch3MTLModel(inputs=inputs, outputs=[cls_out, reg_out, cnn_proj, seq_proj, fused_proj])


# -- trans_rcfd_ctd (Transformer early encoder + CNN+Trans dual) -------------

def create_ml_trans_rcfd_ctd_model(T, n_targets):
    inputs          = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z_cond, reg_out = _build_rcfd_backbone(inputs, 'transformer', 'ctd')
    cls_out         = _ml_cls_head(z_cond, n_targets)
    return MultiLabelRCFDModel(inputs=inputs, outputs=[cls_out, reg_out])


def create_ml_trans_rcfd_ctd_supcon_mtl_model(T, n_targets):
    inputs          = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z_cond, reg_out = _build_rcfd_backbone(inputs, 'transformer', 'ctd')
    cls_out         = _ml_cls_head(z_cond, n_targets)
    proj_norm       = _proj_head(z_cond, 'fused')
    return MultiLabelRCFDSupConMTLModel(inputs=inputs, outputs=[cls_out, reg_out, proj_norm])


def create_ml_trans_rcfd_ctd_supcon2_mtl_model(T, n_targets):
    inputs                            = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z_cond, reg_out, cnn_emb, seq_emb = _build_rcfd_backbone(inputs, 'transformer', 'ctd', return_branches=True)
    cls_out                           = _ml_cls_head(z_cond, n_targets)
    cnn_proj                          = _proj_head(cnn_emb, 'cnn')
    seq_proj                          = _proj_head(seq_emb, 'seq')
    return MultiLabelRCFDBranch2MTLModel(inputs=inputs, outputs=[cls_out, reg_out, cnn_proj, seq_proj])


def create_ml_trans_rcfd_ctd_supcon3_mtl_model(T, n_targets):
    inputs                            = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z_cond, reg_out, cnn_emb, seq_emb = _build_rcfd_backbone(inputs, 'transformer', 'ctd', return_branches=True)
    cls_out                           = _ml_cls_head(z_cond, n_targets)
    cnn_proj                          = _proj_head(cnn_emb, 'cnn')
    seq_proj                          = _proj_head(seq_emb, 'seq')
    fused_proj                        = _proj_head(z_cond, 'fused')
    return MultiLabelRCFDBranch3MTLModel(inputs=inputs, outputs=[cls_out, reg_out, cnn_proj, seq_proj, fused_proj])


# -- single-branch standard ST -----------------------------------------------

def create_ml_cnn_model(T, n_targets):
    """Standard CNN with sigmoid multi-label head (same arch as main/create_cnn_model)."""
    return adapt_for_multilabel(create_cnn_model(T, n_targets), n_targets)


def create_ml_gru_model(T, n_targets):
    """Standard GRU with sigmoid multi-label head (same arch as main/create_gru_model)."""
    return adapt_for_multilabel(create_gru_model(T, n_targets), n_targets)


def create_ml_transformer_model(T, n_targets):
    """Standard Transformer with sigmoid multi-label head."""
    return adapt_for_multilabel(create_transformer_model(T, n_targets), n_targets)


def create_ml_cnn_trans_dual_model(T, n_targets):
    """Standard CNN+Transformer dual with sigmoid multi-label head."""
    return adapt_for_multilabel(create_cnn_transformer_dual_model(T, n_targets), n_targets)


# -- single-branch SC1 -------------------------------------------------------

def create_ml_cnn_supcon_model(T, n_targets):
    """CNN backbone + SC1 (Jaccard) multi-label ST."""
    inputs    = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    embedding = _build_cnn_backbone_mtl(inputs)
    cls_out   = _ml_cls_head(embedding, n_targets)
    proj_norm = _proj_head(embedding, 'fused')
    return MultiLabelSupConModel(inputs=inputs, outputs=[cls_out, proj_norm])


def create_ml_gru_supcon_model(T, n_targets):
    """GRU backbone + SC1 (Jaccard) multi-label ST."""
    inputs    = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    embedding = _build_gru_backbone_mtl(inputs)
    cls_out   = _ml_cls_head(embedding, n_targets)
    proj_norm = _proj_head(embedding, 'fused')
    return MultiLabelSupConModel(inputs=inputs, outputs=[cls_out, proj_norm])


def create_ml_transformer_supcon_model(T, n_targets):
    """Transformer backbone + SC1 (Jaccard) multi-label ST."""
    inputs    = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    embedding = _build_transformer_backbone_mtl(inputs)
    cls_out   = _ml_cls_head(embedding, n_targets)
    proj_norm = _proj_head(embedding, 'fused')
    return MultiLabelSupConModel(inputs=inputs, outputs=[cls_out, proj_norm])


# -- CNN+Trans dual SC1/2/3 --------------------------------------------------

def create_ml_cnn_trans_dual_supcon_model(T, n_targets):
    """CNN+Trans dual fused embedding + SC1 (Jaccard) multi-label ST."""
    inputs    = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    embedding = _build_cnn_trans_dual_branches_mtl(inputs)
    cls_out   = _ml_cls_head(embedding, n_targets)
    proj_norm = _proj_head(embedding, 'fused')
    return MultiLabelSupConModel(inputs=inputs, outputs=[cls_out, proj_norm])


def create_ml_cnn_trans_dual_supcon2_model(T, n_targets):
    """CNN+Trans dual + SC2 (Jaccard on CNN+Trans branches) multi-label ST."""
    inputs               = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    cnn_emb, seq_emb, z = _build_cnn_trans_dual_branches_mtl(inputs, return_branches=True)
    cls_out              = _ml_cls_head(z, n_targets)
    cnn_proj             = _proj_head(cnn_emb, 'cnn')
    seq_proj             = _proj_head(seq_emb, 'seq')
    return MultiLabelSupConBranch2STModel(inputs=inputs, outputs=[cls_out, cnn_proj, seq_proj])


def create_ml_cnn_trans_dual_supcon3_model(T, n_targets):
    """CNN+Trans dual + SC3 (Jaccard on CNN+Trans+fused) multi-label ST."""
    inputs               = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    cnn_emb, seq_emb, z = _build_cnn_trans_dual_branches_mtl(inputs, return_branches=True)
    cls_out              = _ml_cls_head(z, n_targets)
    cnn_proj             = _proj_head(cnn_emb, 'cnn')
    seq_proj             = _proj_head(seq_emb, 'seq')
    fused_proj           = _proj_head(z, 'fused')
    return MultiLabelSupConBranch3STModel(inputs=inputs, outputs=[cls_out, cnn_proj, seq_proj, fused_proj])


# -- factory dispatch --------------------------------------------------------

ML_FACTORIES = {
    # Single-branch standard ST
    'cnn':                            create_ml_cnn_model,
    'gru':                            create_ml_gru_model,
    'transformer':                    create_ml_transformer_model,
    # Single-branch SC1
    'cnn_supcon':                     create_ml_cnn_supcon_model,
    'gru_supcon':                     create_ml_gru_supcon_model,
    'transformer_supcon':             create_ml_transformer_supcon_model,
    # Dual CNN+GRU
    'cnn_gru_dual':                   create_ml_cnn_gru_dual_model,
    'cnn_gru_dual_supcon':            create_ml_cnn_gru_dual_supcon_model,
    'cnn_gru_dual_supcon2':           create_ml_cnn_gru_dual_supcon2_model,
    'cnn_gru_dual_supcon3':           create_ml_cnn_gru_dual_supcon3_model,
    # Dual CNN+Trans
    'cnn_trans_dual':                 create_ml_cnn_trans_dual_model,
    'cnn_trans_dual_supcon':          create_ml_cnn_trans_dual_supcon_model,
    'cnn_trans_dual_supcon2':         create_ml_cnn_trans_dual_supcon2_model,
    'cnn_trans_dual_supcon3':         create_ml_cnn_trans_dual_supcon3_model,
    # RCFD — GRU early encoder, CNN+GRU dual
    'gru_rcfd_cgd':                   create_ml_gru_rcfd_cgd_model,
    'gru_rcfd_cgd_supcon_mtl':        create_ml_gru_rcfd_cgd_supcon_mtl_model,
    'gru_rcfd_cgd_supcon2_mtl':       create_ml_gru_rcfd_cgd_supcon2_mtl_model,
    'gru_rcfd_cgd_supcon3_mtl':       create_ml_gru_rcfd_cgd_supcon3_mtl_model,
    # RCFD — GRU early encoder, CNN+Trans dual
    'gru_rcfd_ctd':                   create_ml_gru_rcfd_ctd_model,
    'gru_rcfd_ctd_supcon_mtl':        create_ml_gru_rcfd_ctd_supcon_mtl_model,
    'gru_rcfd_ctd_supcon2_mtl':       create_ml_gru_rcfd_ctd_supcon2_mtl_model,
    'gru_rcfd_ctd_supcon3_mtl':       create_ml_gru_rcfd_ctd_supcon3_mtl_model,
    # RCFD — Transformer early encoder, CNN+GRU dual
    'trans_rcfd_cgd':                 create_ml_trans_rcfd_cgd_model,
    'trans_rcfd_cgd_supcon_mtl':      create_ml_trans_rcfd_cgd_supcon_mtl_model,
    'trans_rcfd_cgd_supcon2_mtl':     create_ml_trans_rcfd_cgd_supcon2_mtl_model,
    'trans_rcfd_cgd_supcon3_mtl':     create_ml_trans_rcfd_cgd_supcon3_mtl_model,
    # RCFD — Transformer early encoder, CNN+Trans dual
    'trans_rcfd_ctd':                 create_ml_trans_rcfd_ctd_model,
    'trans_rcfd_ctd_supcon_mtl':      create_ml_trans_rcfd_ctd_supcon_mtl_model,
    'trans_rcfd_ctd_supcon2_mtl':     create_ml_trans_rcfd_ctd_supcon2_mtl_model,
    'trans_rcfd_ctd_supcon3_mtl':     create_ml_trans_rcfd_ctd_supcon3_mtl_model,
}

# Models that carry a concentration regression output
_RCFD_ML_KEYS = frozenset({
    'gru_rcfd_cgd',  'gru_rcfd_cgd_supcon_mtl',  'gru_rcfd_cgd_supcon2_mtl',  'gru_rcfd_cgd_supcon3_mtl',
    'gru_rcfd_ctd',  'gru_rcfd_ctd_supcon_mtl',  'gru_rcfd_ctd_supcon2_mtl',  'gru_rcfd_ctd_supcon3_mtl',
    'trans_rcfd_cgd', 'trans_rcfd_cgd_supcon_mtl', 'trans_rcfd_cgd_supcon2_mtl', 'trans_rcfd_cgd_supcon3_mtl',
    'trans_rcfd_ctd', 'trans_rcfd_ctd_supcon_mtl', 'trans_rcfd_ctd_supcon2_mtl', 'trans_rcfd_ctd_supcon3_mtl',
})

# Models that output a projection head in addition to cls_out
_SUPCON_ML_KEYS = frozenset({
    'cnn_supcon', 'gru_supcon', 'transformer_supcon',
    'cnn_gru_dual_supcon', 'cnn_gru_dual_supcon2', 'cnn_gru_dual_supcon3',
    'cnn_trans_dual_supcon', 'cnn_trans_dual_supcon2', 'cnn_trans_dual_supcon3',
    'gru_rcfd_cgd_supcon_mtl',  'gru_rcfd_cgd_supcon2_mtl',  'gru_rcfd_cgd_supcon3_mtl',
    'gru_rcfd_ctd_supcon_mtl',  'gru_rcfd_ctd_supcon2_mtl',  'gru_rcfd_ctd_supcon3_mtl',
    'trans_rcfd_cgd_supcon_mtl', 'trans_rcfd_cgd_supcon2_mtl', 'trans_rcfd_cgd_supcon3_mtl',
    'trans_rcfd_ctd_supcon_mtl', 'trans_rcfd_ctd_supcon2_mtl', 'trans_rcfd_ctd_supcon3_mtl',
})

# Canonical key maps — consumed by 03_main_training.py and print_ml_results_summary
ML_MODEL_KEY_MAP = {
    m: (f'y_preds_{m}_', f'y_probs_{m}_', f'classes_{m}_')
    for m in ML_FACTORIES
}

ML_MODEL_PRINT_MAP = {
    'cnn':                       'CNN',
    'gru':                       'GRU',
    'transformer':               'Transformer',
    'cnn_supcon':                'CNN SC1',
    'gru_supcon':                'GRU SC1',
    'transformer_supcon':        'Trans SC1',
    'cnn_gru_dual':              'CNN+GRU',
    'cnn_gru_dual_supcon':       'CNN+GRU SC1',
    'cnn_gru_dual_supcon2':      'CNN+GRU SC2',
    'cnn_gru_dual_supcon3':      'CNN+GRU SC3',
    'cnn_trans_dual':            'CNN+Trans',
    'cnn_trans_dual_supcon':     'CNN+Trans SC1',
    'cnn_trans_dual_supcon2':    'CNN+Trans SC2',
    'cnn_trans_dual_supcon3':    'CNN+Trans SC3',
    'gru_rcfd_cgd':               'GRU RCFD CGD',
    'gru_rcfd_cgd_supcon_mtl':    'GRU RCFD CGD SC1',
    'gru_rcfd_cgd_supcon2_mtl':   'GRU RCFD CGD SC2',
    'gru_rcfd_cgd_supcon3_mtl':   'GRU RCFD CGD SC3',
    'gru_rcfd_ctd':               'GRU RCFD CTD',
    'gru_rcfd_ctd_supcon_mtl':    'GRU RCFD CTD SC1',
    'gru_rcfd_ctd_supcon2_mtl':   'GRU RCFD CTD SC2',
    'gru_rcfd_ctd_supcon3_mtl':   'GRU RCFD CTD SC3',
    'trans_rcfd_cgd':             'Trans RCFD CGD',
    'trans_rcfd_cgd_supcon_mtl':  'Trans RCFD CGD SC1',
    'trans_rcfd_cgd_supcon2_mtl': 'Trans RCFD CGD SC2',
    'trans_rcfd_cgd_supcon3_mtl': 'Trans RCFD CGD SC3',
    'trans_rcfd_ctd':             'Trans RCFD CTD',
    'trans_rcfd_ctd_supcon_mtl':  'Trans RCFD CTD SC1',
    'trans_rcfd_ctd_supcon2_mtl': 'Trans RCFD CTD SC2',
    'trans_rcfd_ctd_supcon3_mtl': 'Trans RCFD CTD SC3',
}


# ======================================================================
# METRICS
# ======================================================================

def multilabel_metrics(y_true_binary, y_pred_binary, target_names):
    """Compute standard multi-label metrics. Returns a dict."""
    yt = np.asarray(y_true_binary)
    yp = np.asarray(y_pred_binary)
    return {
        'exact_acc':    accuracy_score(yt, yp),
        'hamming':      hamming_loss(yt, yp),
        'f1_samples':   f1_score(yt, yp, average='samples', zero_division=0),
        'f1_macro':     f1_score(yt, yp, average='macro',   zero_division=0),
        'f1_micro':     f1_score(yt, yp, average='micro',   zero_division=0),
        'f1_per_label': {t: float(v) for t, v in zip(
            target_names, f1_score(yt, yp, average=None, zero_division=0))},
    }


# ======================================================================
# MAIN TRAINING LOOP
# ======================================================================

def evaluate_outlier_filters_ml(
    X_curves,
    features_df,
    y_binary,
    y_combo_int,
    all_targets,
    outlier_filters,
    dataset_name,
    mode_name,
    ml_model_key_map,
    ml_model_print_map,
    cached_results=None,
    models=None,
    n_splits=1,
    checkpoint_fn=None,
    save_model_dir=None,
    y_concentration=None,
    threshold=0.5,
    rerun_models=None,
):
    """Train and evaluate multi-label models across outlier filters.

    Parameters
    ----------
    X_curves           : (N, T) float
    features_df        : DataFrame -- may contain outlier filter columns
    y_binary           : (N, n_targets) int8 binary indicator matrix
    y_combo_int        : (N,) int -- combination index for StratifiedKFold
    all_targets        : list[str] e.g. ['KPC', 'NDM', 'VIM']
    outlier_filters    : list of filter column names (None = no filter)
    ml_model_key_map   : {model_name: (preds_key, probs_key, classes_key)}
    ml_model_print_map : {model_name: display_str}
    y_concentration    : (N,) float or None -- RCFD regression head target
    threshold          : float -- sigmoid threshold for binary prediction
    rerun_models       : list[str] or None -- restrict retraining to these keys
    """
    if models is None:
        models = list(ML_FACTORIES.keys())

    if save_model_dir is not None:
        Path(save_model_dir).mkdir(parents=True, exist_ok=True)

    results_dict = cached_results.copy() if cached_results is not None else {}
    rerun_models = set(rerun_models) if rerun_models else set()
    n_targets    = len(all_targets)

    for idx, f in enumerate(outlier_filters):
        filter_name = f if f else 'None (Baseline)'
        print(f"  -> Filter [{idx+1}/{len(outlier_filters)}]: {filter_name}")

        res_entry = results_dict.get(f, {})

        if f is None:
            mask = np.ones(len(y_combo_int), dtype=bool)
        elif f in features_df.columns:
            mask = features_df[f].fillna(False).astype(bool).values
        else:
            print(f"     [Warning] {f} not found in dataset. Skipping.")
            continue

        X_f      = np.nan_to_num(X_curves[mask], nan=0.0, posinf=0.0, neginf=0.0)
        y_bin_f  = y_binary[mask]
        y_comb_f = y_combo_int[mask]
        y_conc_f = y_concentration[mask] if y_concentration is not None else None

        # Drop rare combinations that cannot be stratified
        counts      = np.bincount(y_comb_f)
        rare_combos = np.where(counts < 2)[0]
        if len(rare_combos):
            keep     = ~np.isin(y_comb_f, rare_combos)
            X_f      = X_f[keep]
            y_bin_f  = y_bin_f[keep]
            y_comb_f = y_comb_f[keep]
            if y_conc_f is not None:
                y_conc_f = y_conc_f[keep]

        if len(y_comb_f) < 4:
            print(f"     [Warning] Too few samples after rare-class filter. Skipping.")
            continue

        current_mask_count = int(np.sum(mask))
        if res_entry.get('mask_count') not in (None, current_mask_count):
            print(f"     [Warning] Stale cache (count changed). Resetting for this filter.")
            res_entry = {}

        n_total   = len(y_comb_f)
        test_size = max(int(n_total * 0.10), len(np.unique(y_comb_f)))
        min_cnt   = int(np.min(counts[counts > 0]))

        if n_splits == 1:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
        else:
            splitter = StratifiedKFold(n_splits=min(n_splits, min_cnt), shuffle=True, random_state=0)

        splits = list(splitter.split(X_f, y_comb_f))

        if 'y_trues_' not in res_entry:
            res_entry['y_trues_'] = [y_bin_f[te] for _, te in splits]

        # Purge stale per-model caches when fold sizes changed
        expected_lens = [len(a) for a in res_entry['y_trues_']]
        stale = [k for k, v in res_entry.items()
                 if k.startswith('y_preds_') and isinstance(v, list)
                 and [len(a) for a in v] != expected_lens]
        for k in stale:
            base = k[len('y_preds_'):]
            for prefix in ('y_preds_', 'y_probs_', 'classes_'):
                res_entry.pop(prefix + base, None)

        res_entry['mask_count']   = current_mask_count
        res_entry['y_true_count'] = n_total

        for m in models:
            if m not in ml_model_key_map:
                print(f"     [SKIP] {m}: not in ml_model_key_map")
                continue
            if m not in ML_FACTORIES:
                print(f"     [SKIP] {m}: no factory function")
                continue

            preds_key, probs_key, classes_key = ml_model_key_map[m]

            if preds_key in res_entry and m not in rerun_models:
                accs = [accuracy_score(yt, yp)
                        for yt, yp in zip(res_entry['y_trues_'], res_entry[preds_key])]
                print(f"     [CACHE] {m} | "
                      f"exact_acc={np.mean(accs)*100:.2f}%+-{np.std(accs)*100:.2f}%")
                continue

            is_rcfd = m in _RCFD_ML_KEYS

            preds_folds, probs_folds, classes_folds = [], [], []
            reg_preds_folds, reg_trues_folds = [], []
            t0 = time.perf_counter()

            for fold_idx, (tr, te) in enumerate(splits):
                X_train, X_test   = X_f[tr], X_f[te]
                yb_train, yb_test = y_bin_f[tr], y_bin_f[te]

                # Val split for early stopping
                try:
                    tr_sub, val_sub = train_test_split(
                        np.arange(len(yb_train)), test_size=0.1,
                        stratify=y_comb_f[tr], random_state=0)
                    X_tr, X_val   = X_train[tr_sub], X_train[val_sub]
                    yb_tr, yb_val = yb_train[tr_sub], yb_train[val_sub]
                    _has_val      = True
                except ValueError:
                    X_tr, yb_tr = X_train, yb_train
                    _has_val    = False

                T_steps = X_train.shape[1]
                tf.keras.backend.clear_session()
                model = ML_FACTORIES[m](T_steps, n_targets)
                model.compile(optimizer=tf.keras.optimizers.Adam(0.001, clipnorm=1.0))

                if is_rcfd:
                    conc_tr_raw  = (y_conc_f[tr] if y_conc_f is not None
                                    else np.full(len(tr), REG_SENTINEL))
                    conc_te_raw  = (y_conc_f[te] if y_conc_f is not None
                                    else np.full(len(te), REG_SENTINEL))
                    conc_tr_sc, _scaler = _normalize_concentration(conc_tr_raw)
                    conc_te_sc   = conc_te_raw.copy()
                    _valid_te    = conc_te_raw != REG_SENTINEL
                    if _valid_te.sum() > 0 and hasattr(_scaler, 'mean_'):
                        conc_te_sc[_valid_te] = _scaler.transform(
                            conc_te_raw[_valid_te].reshape(-1, 1)).ravel()

                    if _has_val:
                        y_tr_d  = {'cls_out': yb_tr,    'reg_out': conc_tr_sc[tr_sub]}
                        y_val_d = {'cls_out': yb_val,   'reg_out': conc_tr_sc[val_sub]}
                    else:
                        y_tr_d  = {'cls_out': yb_train, 'reg_out': conc_tr_sc}
                        y_val_d = None
                else:
                    if _has_val:
                        y_tr_d  = {'cls_out': yb_tr}
                        y_val_d = {'cls_out': yb_val}
                    else:
                        y_tr_d  = {'cls_out': yb_train}
                        y_val_d = None

                cbs = []
                if _has_val:
                    cbs = [
                        tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss', patience=100, restore_best_weights=True),
                        tf.keras.callbacks.ReduceLROnPlateau(
                            monitor='val_loss', factor=0.5, patience=30, min_lr=1e-5),
                    ]

                fit_kw = dict(epochs=500, batch_size=512, shuffle=True, verbose=0, callbacks=cbs)
                if _has_val:
                    model.fit(X_tr, y_tr_d, validation_data=(X_val, y_val_d), **fit_kw)
                else:
                    model.fit(X_train, y_tr_d, **fit_kw)

                if save_model_dir is not None and fold_idx == 0:
                    safe_keras_save(model,
                                    Path(save_model_dir) / f"{m}_{f}_{mode_name}.keras")

                raw_out  = model.predict(X_test, verbose=0)
                cls_prob = raw_out[0] if isinstance(raw_out, (list, tuple)) else raw_out
                cls_pred = (cls_prob >= threshold).astype(np.int8)

                preds_folds.append(cls_pred)
                probs_folds.append(cls_prob)
                classes_folds.append(np.array(all_targets))

                if is_rcfd and isinstance(raw_out, (list, tuple)):
                    reg_pred_orig = _inverse_normalize_concentration(raw_out[1][:, 0], _scaler)
                    reg_preds_folds.append(reg_pred_orig)
                    reg_trues_folds.append(conc_te_raw)

                tf.keras.backend.clear_session()
                gc.collect()

            res_entry[preds_key]   = preds_folds
            res_entry[probs_key]   = probs_folds
            res_entry[classes_key] = classes_folds
            if is_rcfd and reg_preds_folds:
                res_entry[f'y_reg_preds_{m}_'] = reg_preds_folds
                res_entry[f'y_reg_trues_{m}_'] = reg_trues_folds

            duration = time.perf_counter() - t0
            accs = [accuracy_score(yt, yp) for yt, yp in zip(res_entry['y_trues_'], preds_folds)]
            hls  = [hamming_loss(yt, yp)   for yt, yp in zip(res_entry['y_trues_'], preds_folds)]
            f1s  = [f1_score(yt, yp, average='samples', zero_division=0)
                    for yt, yp in zip(res_entry['y_trues_'], preds_folds)]
            print(f"     [+] {mode_name}/{dataset_name}/{filter_name[:25]} | "
                  f"{ml_model_print_map.get(m, m):<20} | "
                  f"exact={np.mean(accs)*100:.2f}% | "
                  f"hamming={np.mean(hls):.4f} | "
                  f"f1_samp={np.mean(f1s):.4f} | {duration:.1f}s")

            if checkpoint_fn is not None:
                checkpoint_fn(results_dict)   # mirrors main/model_utils.py: pass full dict

        results_dict[f] = res_entry

    return results_dict


# ======================================================================
# CONSOLE SUMMARY (full HTML report is in 06_model_prediction_report.py)
# ======================================================================

def print_ml_results_summary(results_dict, outlier_filters, dataset_name, mode_name,
                              ml_model_key_map, ml_model_print_map):
    """Print a text summary of multi-label results per filter."""
    print(f"\n[Results] {mode_name} -- {dataset_name}")
    for f in outlier_filters:
        res = results_dict.get(f)
        if not res or 'y_trues_' not in res:
            continue
        filter_name = f if f else 'None (Baseline)'
        for m, (preds_key, _, _) in ml_model_key_map.items():
            if preds_key not in res:
                continue
            accs = [accuracy_score(yt, yp)
                    for yt, yp in zip(res['y_trues_'], res[preds_key])]
            hls  = [hamming_loss(yt, yp)
                    for yt, yp in zip(res['y_trues_'], res[preds_key])]
            print(f"  {filter_name:30} | {ml_model_print_map.get(m, m):20} | "
                  f"exact={np.mean(accs)*100:.2f}% | hamming={np.mean(hls):.4f}")
