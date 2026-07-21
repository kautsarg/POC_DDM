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
from model_utils_source_sep import build_source_sep_encoder, build_source_sep_classifier

_SS_D_SHARED = 16
_SS_D_TARGET  = 10


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
    """Single-output sigmoid multi-label model; uses standard Keras BCE training."""
    pass


def adapt_for_multilabel(model, n_targets):
    """Replace last softmax Dense with sigmoid Dense(n_targets).

    Returns a new _StandardMultiLabelModel; the original is unchanged.
    """
    for layer in reversed(model.layers):
        if (hasattr(layer, 'activation')
                and layer.activation.__name__ == 'softmax'):
            new_out = tf.keras.layers.Dense(
                n_targets, activation='sigmoid', name='cls_out')(layer.input)
            return _StandardMultiLabelModel(inputs=model.inputs, outputs=new_out,
                                            name='standard_ml_model')
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


@tf.keras.utils.register_keras_serializable(package='ml_cross_attn')
class MultiLabelAuxDetModel(tf.keras.Model):
    """Cross-attn v2 with per-block auxiliary BCE losses (v2_auxdet).

    Outputs: [cls_out, aux0, aux1, ...]. lambda_aux weight per aux term.
    cls_prob = raw_out[0] at inference (handled by existing training loop).
    """
    def __init__(self, *args, lambda_aux=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_aux = lambda_aux

    def _step(self, x, y_dict, training):
        y_bin   = tf.cast(y_dict['cls_out'], tf.float32)
        outputs = self(x, training=training)
        loss    = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_bin, outputs[0]))
        for aux in outputs[1:]:
            loss = loss + self.lambda_aux * tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(y_bin, aux))
        return loss

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            loss = self._step(x, y_dict, training=True)
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return {'loss': loss}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        return {'loss': self._step(x, y_dict, training=False)}


@tf.keras.utils.register_keras_serializable(package='ml_cross_attn')
class MultiLabelQueryConModel(tf.keras.Model):
    """Cross-attn v2 with per-label query contrastive loss (QuerCon).

    Outputs: [cls_out, queries_norm] where queries_norm: (batch, n_targets, query_dim).
    For each label j, SupCon is applied on queries_norm[:,j,:] keyed by y[:,j].
    """
    def __init__(self, *args, supcon_temp=SUPCON_TEMP, lambda_qc=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.supcon_temp = supcon_temp
        self.lambda_qc   = lambda_qc

    def _quercon_loss(self, queries_norm, y_bin):
        n_j   = queries_norm.shape[1] or 3
        total = tf.constant(0.0)
        for j in range(n_j):
            emb_j = queries_norm[:, j, :]
            y_j   = tf.cast(y_bin[:, j], tf.float32)
            pos   = (tf.expand_dims(y_j, 1) * tf.expand_dims(y_j, 0)
                     * (1.0 - tf.eye(tf.shape(emb_j)[0])))
            total = total + _supcon_loss_jaccard(emb_j, pos, self.supcon_temp)
        return total / tf.cast(n_j, tf.float32)

    def _step(self, x, y_dict, training):
        y_bin           = tf.cast(y_dict['cls_out'], tf.float32)
        cls_out, q_norm = self(x, training=training)
        bce  = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_bin, cls_out))
        qc   = self._quercon_loss(q_norm, y_bin)
        loss = (1.0 - self.lambda_qc) * bce + self.lambda_qc * qc
        return loss, bce, qc

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            loss, bce, qc = self._step(x, y_dict, training=True)
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return {'loss': loss, 'cls_bce': bce, 'quercon': qc}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        loss, bce, qc = self._step(x, y_dict, training=False)
        return {'loss': loss, 'cls_bce': bce, 'quercon': qc}


@tf.keras.utils.register_keras_serializable(package='ml_cross_attn')
class MultiLabelQueryConSC1Model(MultiLabelQueryConModel):
    """QuerCon + backbone Jaccard SupCon (SC1) combined. Outputs [cls_out, queries_norm, proj_norm]."""
    def __init__(self, *args, supcon_lambda=SUPCON_LAMBDA, **kwargs):
        super().__init__(*args, **kwargs)
        self.supcon_lambda = supcon_lambda

    def _step(self, x, y_dict, training):
        y_bin                      = tf.cast(y_dict['cls_out'], tf.float32)
        cls_out, q_norm, proj_norm = self(x, training=training)
        bce      = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_bin, cls_out))
        pos_mask = _jaccard_weight_matrix(y_bin)
        sc       = _supcon_loss_jaccard(proj_norm, pos_mask, self.supcon_temp)
        qc       = self._quercon_loss(q_norm, y_bin)
        loss     = ((1.0 - self.supcon_lambda - self.lambda_qc) * bce
                    + self.supcon_lambda * sc + self.lambda_qc * qc)
        return loss, bce, sc, qc

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            loss, bce, sc, qc = self._step(x, y_dict, training=True)
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return {'loss': loss, 'cls_bce': bce, 'supcon': sc, 'quercon': qc}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        loss, bce, sc, qc = self._step(x, y_dict, training=False)
        return {'loss': loss, 'cls_bce': bce, 'supcon': sc, 'quercon': qc}


@tf.keras.utils.register_keras_serializable(package='ml_cross_attn')
class MultiLabelAuxDetSC1Model(MultiLabelAuxDetModel):
    """AuxDet + backbone Jaccard SupCon SC1. Outputs [cls_out, aux..., proj_norm]."""
    def __init__(self, *args, supcon_temp=SUPCON_TEMP, supcon_lambda=SUPCON_LAMBDA, **kwargs):
        super().__init__(*args, **kwargs)
        self.supcon_temp   = supcon_temp
        self.supcon_lambda = supcon_lambda

    def _step(self, x, y_dict, training):
        y_bin    = tf.cast(y_dict['cls_out'], tf.float32)
        outputs  = self(x, training=training)
        cls_out, aux_outs, proj_norm = outputs[0], outputs[1:-1], outputs[-1]
        l_det    = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_bin, cls_out))
        for aux in aux_outs:
            l_det = l_det + self.lambda_aux * tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(y_bin, aux))
        pos_mask = _jaccard_weight_matrix(y_bin)
        sc       = _supcon_loss_jaccard(proj_norm, pos_mask, self.supcon_temp)
        loss     = (1.0 - self.supcon_lambda) * l_det + self.supcon_lambda * sc
        return loss, l_det, sc

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            loss, l_det, sc = self._step(x, y_dict, training=True)
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return {'loss': loss, 'det_bce': l_det, 'supcon': sc}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        loss, l_det, sc = self._step(x, y_dict, training=False)
        return {'loss': loss, 'det_bce': l_det, 'supcon': sc}


@tf.keras.utils.register_keras_serializable(package='ml_cross_attn')
class MultiLabelAuxDetSC2Model(MultiLabelAuxDetModel):
    """AuxDet + branch SupCon SC2. Outputs [cls_out, aux..., cnn_proj, seq_proj]."""
    def __init__(self, *args, supcon_temp=SUPCON_TEMP, supcon_lambda_each=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.supcon_temp        = supcon_temp
        self.supcon_lambda_each = supcon_lambda_each

    def _step(self, x, y_dict, training):
        y_bin    = tf.cast(y_dict['cls_out'], tf.float32)
        outputs  = self(x, training=training)
        cls_out, aux_outs = outputs[0], outputs[1:-2]
        cnn_proj, seq_proj = outputs[-2], outputs[-1]
        l_det    = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_bin, cls_out))
        for aux in aux_outs:
            l_det = l_det + self.lambda_aux * tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(y_bin, aux))
        pos_mask = _jaccard_weight_matrix(y_bin)
        sc       = (_supcon_loss_jaccard(cnn_proj, pos_mask, self.supcon_temp)
                    + _supcon_loss_jaccard(seq_proj, pos_mask, self.supcon_temp))
        loss     = ((1.0 - 2 * self.supcon_lambda_each) * l_det
                    + self.supcon_lambda_each * sc)
        return loss, l_det, sc

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            loss, l_det, sc = self._step(x, y_dict, training=True)
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return {'loss': loss, 'det_bce': l_det, 'supcon': sc}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        loss, l_det, sc = self._step(x, y_dict, training=False)
        return {'loss': loss, 'det_bce': l_det, 'supcon': sc}


@tf.keras.utils.register_keras_serializable(package='ml_cross_attn')
class MultiLabelAuxDetSC3Model(MultiLabelAuxDetModel):
    """AuxDet + three-branch SupCon SC3. Outputs [cls_out, aux..., cnn_proj, seq_proj, fused_proj]."""
    def __init__(self, *args, supcon_temp=SUPCON_TEMP, supcon_lambda_each=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.supcon_temp        = supcon_temp
        self.supcon_lambda_each = supcon_lambda_each

    def _step(self, x, y_dict, training):
        y_bin    = tf.cast(y_dict['cls_out'], tf.float32)
        outputs  = self(x, training=training)
        cls_out, aux_outs = outputs[0], outputs[1:-3]
        cnn_proj, seq_proj, fused_proj = outputs[-3], outputs[-2], outputs[-1]
        l_det    = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_bin, cls_out))
        for aux in aux_outs:
            l_det = l_det + self.lambda_aux * tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(y_bin, aux))
        pos_mask = _jaccard_weight_matrix(y_bin)
        sc       = (_supcon_loss_jaccard(cnn_proj,   pos_mask, self.supcon_temp)
                    + _supcon_loss_jaccard(seq_proj,   pos_mask, self.supcon_temp)
                    + _supcon_loss_jaccard(fused_proj, pos_mask, self.supcon_temp))
        loss     = ((1.0 - 3 * self.supcon_lambda_each) * l_det
                    + self.supcon_lambda_each * sc)
        return loss, l_det, sc

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            loss, l_det, sc = self._step(x, y_dict, training=True)
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return {'loss': loss, 'det_bce': l_det, 'supcon': sc}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        loss, l_det, sc = self._step(x, y_dict, training=False)
        return {'loss': loss, 'det_bce': l_det, 'supcon': sc}


@tf.keras.utils.register_keras_serializable(package='ml_cross_attn')
class MultiLabelQueryConSC2Model(MultiLabelQueryConModel):
    """QuerCon + branch SupCon SC2. Outputs [cls_out, queries_norm, cnn_proj, seq_proj]."""
    def __init__(self, *args, supcon_temp=SUPCON_TEMP, supcon_lambda_each=0.1, **kwargs):
        super().__init__(*args, supcon_temp=supcon_temp, **kwargs)
        self.supcon_lambda_each = supcon_lambda_each

    def _step(self, x, y_dict, training):
        y_bin    = tf.cast(y_dict['cls_out'], tf.float32)
        outputs  = self(x, training=training)
        cls_out, q_norm, cnn_proj, seq_proj = outputs
        bce      = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_bin, cls_out))
        qc       = self._quercon_loss(q_norm, y_bin)
        pos_mask = _jaccard_weight_matrix(y_bin)
        sc       = (_supcon_loss_jaccard(cnn_proj, pos_mask, self.supcon_temp)
                    + _supcon_loss_jaccard(seq_proj, pos_mask, self.supcon_temp))
        loss     = ((1.0 - self.lambda_qc - 2 * self.supcon_lambda_each) * bce
                    + self.lambda_qc * qc + self.supcon_lambda_each * sc)
        return loss, bce, sc, qc

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            loss, bce, sc, qc = self._step(x, y_dict, training=True)
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return {'loss': loss, 'cls_bce': bce, 'supcon': sc, 'quercon': qc}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        loss, bce, sc, qc = self._step(x, y_dict, training=False)
        return {'loss': loss, 'cls_bce': bce, 'supcon': sc, 'quercon': qc}


@tf.keras.utils.register_keras_serializable(package='ml_cross_attn')
class MultiLabelQueryConSC3Model(MultiLabelQueryConModel):
    """QuerCon + three-branch SupCon SC3. Outputs [cls_out, queries_norm, cnn_proj, seq_proj, fused_proj]."""
    def __init__(self, *args, supcon_temp=SUPCON_TEMP, supcon_lambda_each=0.1, **kwargs):
        super().__init__(*args, supcon_temp=supcon_temp, **kwargs)
        self.supcon_lambda_each = supcon_lambda_each

    def _step(self, x, y_dict, training):
        y_bin    = tf.cast(y_dict['cls_out'], tf.float32)
        outputs  = self(x, training=training)
        cls_out, q_norm, cnn_proj, seq_proj, fused_proj = outputs
        bce      = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_bin, cls_out))
        qc       = self._quercon_loss(q_norm, y_bin)
        pos_mask = _jaccard_weight_matrix(y_bin)
        sc       = (_supcon_loss_jaccard(cnn_proj,   pos_mask, self.supcon_temp)
                    + _supcon_loss_jaccard(seq_proj,   pos_mask, self.supcon_temp)
                    + _supcon_loss_jaccard(fused_proj, pos_mask, self.supcon_temp))
        loss     = ((1.0 - self.lambda_qc - 3 * self.supcon_lambda_each) * bce
                    + self.lambda_qc * qc + self.supcon_lambda_each * sc)
        return loss, bce, sc, qc

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            loss, bce, sc, qc = self._step(x, y_dict, training=True)
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return {'loss': loss, 'cls_bce': bce, 'supcon': sc, 'quercon': qc}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        loss, bce, sc, qc = self._step(x, y_dict, training=False)
        return {'loss': loss, 'cls_bce': bce, 'supcon': sc, 'quercon': qc}


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
# CRF MODEL CLASSES
# ======================================================================

def compute_crf_state_weights(y_binary, n_targets=3):
    """Inverse-frequency weights for 2^n_targets states to address label-combo imbalance."""
    powers  = np.array([2**j for j in range(n_targets)], dtype=np.int32)
    y_idx   = (y_binary.astype(np.int32) @ powers)
    counts  = np.bincount(y_idx, minlength=2**n_targets).astype(np.float32)
    counts  = np.where(counts == 0, 1.0, counts)
    weights = 1.0 / counts
    return (weights / weights.sum() * len(weights)).tolist()


# ── Option A: Full-state MRF ─────────────────────────────────────────────────

@tf.keras.utils.register_keras_serializable(package='ml_crf_mrf')
class MultiLabelCRFMRFModel(tf.keras.Model):
    """Option A base — Full-state MRF: 2^n_targets joint states, NLL via logsumexp.

    INVARIANT: call() returns raw (batch, n_states) logits, or (logits, proj...) for SC subclasses.
    SC subclasses override only _get_crf_output(); predict_binary/predict_marginals use it.
    """
    def __init__(self, *args, n_targets=3, state_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_targets    = n_targets
        self.n_states     = 2 ** n_targets
        self.decode_mat   = tf.constant(
            [[int((i >> j) & 1) for j in range(n_targets)]
             for i in range(2 ** n_targets)], dtype=tf.float32)   # (n_states, n_targets)
        self.powers       = tf.constant([2 ** j for j in range(n_targets)], dtype=tf.int32)
        self.state_weights = (tf.constant(state_weights, dtype=tf.float32)
                              if state_weights is not None else None)

    def _get_crf_output(self, x):
        """Return (batch, n_states) logits. SC subclasses override to unpack tuple."""
        return self(x, training=False)

    def _nll(self, state_logits, y_dict):
        """NLL loss given pre-extracted state logits."""
        y_bin      = tf.cast(y_dict['cls_out'] if isinstance(y_dict, dict) else y_dict, tf.int32)
        y_idx      = tf.reduce_sum(y_bin * self.powers, axis=-1)
        log_Z      = tf.reduce_logsumexp(state_logits, axis=-1)
        per_sample = log_Z - tf.gather(state_logits, y_idx, batch_dims=1)
        if self.state_weights is not None:
            w = tf.gather(self.state_weights, y_idx)
            return tf.reduce_sum(w * per_sample) / tf.reduce_sum(w)
        return tf.reduce_mean(per_sample)

    def _step(self, x, y_dict, training):
        return self._nll(self(x, training=training), y_dict)

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            loss = self._step(x, y_dict, training=True)
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return {'loss': loss}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        return {'loss': self._step(x, y_dict, training=False)}

    def predict_binary(self, x):
        """Argmax over joint states → (batch, n_targets) int32."""
        logits     = self._get_crf_output(x)
        pred_state = tf.argmax(logits, axis=-1)
        return tf.gather(tf.cast(self.decode_mat, tf.int32), pred_state).numpy()

    def predict_marginals(self, x):
        """Softmax marginals → (batch, n_targets) float32 in [0,1]."""
        state_probs = tf.nn.softmax(self._get_crf_output(x), axis=-1)
        return (state_probs @ self.decode_mat).numpy()


@tf.keras.utils.register_keras_serializable(package='ml_crf_mrf_sc1')
class MultiLabelCRFMRFSC1Model(MultiLabelCRFMRFModel):
    """CRF-MRF + fused-embedding Jaccard SupCon (SC1). Loss: NLL + λ·SC."""
    def __init__(self, *args, supcon_temp=SUPCON_TEMP, supcon_lambda=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.supcon_temp   = supcon_temp
        self.supcon_lambda = supcon_lambda

    def _get_crf_output(self, x):
        return self(x, training=False)[0]

    def _step(self, x, y_dict, training):
        y_bin = tf.cast(y_dict['cls_out'], tf.float32)
        state_logits, proj_norm = self(x, training=training)
        nll      = self._nll(state_logits, y_dict)
        pos_mask = _jaccard_weight_matrix(y_bin)
        sc       = _supcon_loss_jaccard(proj_norm, pos_mask, self.supcon_temp)
        return nll + self.supcon_lambda * sc, nll, sc

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            loss, nll, sc = self._step(x, y_dict, training=True)
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return {'loss': loss, 'crf_nll': nll, 'supcon': sc}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        loss, nll, sc = self._step(x, y_dict, training=False)
        return {'loss': loss, 'crf_nll': nll, 'supcon': sc}


@tf.keras.utils.register_keras_serializable(package='ml_crf_mrf_sc2')
class MultiLabelCRFMRFSC2Model(MultiLabelCRFMRFModel):
    """CRF-MRF + CNN+seq branch Jaccard SupCon (SC2). Loss: NLL + λ·(SC_cnn + SC_seq)."""
    def __init__(self, *args, supcon_temp=SUPCON_TEMP, supcon_lambda_each=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.supcon_temp        = supcon_temp
        self.supcon_lambda_each = supcon_lambda_each

    def _get_crf_output(self, x):
        return self(x, training=False)[0]

    def _step(self, x, y_dict, training):
        y_bin = tf.cast(y_dict['cls_out'], tf.float32)
        state_logits, cnn_proj, seq_proj = self(x, training=training)
        nll      = self._nll(state_logits, y_dict)
        pos_mask = _jaccard_weight_matrix(y_bin)
        sc       = (_supcon_loss_jaccard(cnn_proj, pos_mask, self.supcon_temp)
                    + _supcon_loss_jaccard(seq_proj, pos_mask, self.supcon_temp))
        return nll + self.supcon_lambda_each * sc, nll, sc

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            loss, nll, sc = self._step(x, y_dict, training=True)
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return {'loss': loss, 'crf_nll': nll, 'supcon': sc}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        loss, nll, sc = self._step(x, y_dict, training=False)
        return {'loss': loss, 'crf_nll': nll, 'supcon': sc}


@tf.keras.utils.register_keras_serializable(package='ml_crf_mrf_sc3')
class MultiLabelCRFMRFSC3Model(MultiLabelCRFMRFModel):
    """CRF-MRF + CNN+seq+fused Jaccard SupCon (SC3). Loss: NLL + λ·(SC_cnn + SC_seq + SC_fused)."""
    def __init__(self, *args, supcon_temp=SUPCON_TEMP, supcon_lambda_each=0.033, **kwargs):
        super().__init__(*args, **kwargs)
        self.supcon_temp        = supcon_temp
        self.supcon_lambda_each = supcon_lambda_each

    def _get_crf_output(self, x):
        return self(x, training=False)[0]

    def _step(self, x, y_dict, training):
        y_bin = tf.cast(y_dict['cls_out'], tf.float32)
        state_logits, cnn_proj, seq_proj, fused_proj = self(x, training=training)
        nll      = self._nll(state_logits, y_dict)
        pos_mask = _jaccard_weight_matrix(y_bin)
        sc       = (_supcon_loss_jaccard(cnn_proj,   pos_mask, self.supcon_temp)
                    + _supcon_loss_jaccard(seq_proj,   pos_mask, self.supcon_temp)
                    + _supcon_loss_jaccard(fused_proj, pos_mask, self.supcon_temp))
        return nll + self.supcon_lambda_each * sc, nll, sc

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            loss, nll, sc = self._step(x, y_dict, training=True)
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return {'loss': loss, 'crf_nll': nll, 'supcon': sc}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        loss, nll, sc = self._step(x, y_dict, training=False)
        return {'loss': loss, 'crf_nll': nll, 'supcon': sc}


# ── Option B: Linear-chain CRF ───────────────────────────────────────────────

@tf.keras.utils.register_keras_serializable(package='ml_crf_chain')
class MultiLabelCRFChainModel(tf.keras.Model):
    """Option B base — Linear-chain CRF: n_targets positions × 2 states, shared (2,2) transition.

    INVARIANT: call() returns emission logits (batch, n_targets, 2), or (emit, proj...) for SC.
    SC subclasses override only _get_emit(); predict_binary/predict_marginals use it.
    Label order is fixed (e.g. KPC=0, NDM=1, VIM=2).
    """
    def __init__(self, *args, n_targets=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_targets  = n_targets
        self.transition = self.add_weight(
            name='crf_transition', shape=(2, 2),
            initializer='zeros', trainable=True)  # transition[from_state, to_state]

    def _get_emit(self, x):
        """Return (batch, n_targets, 2) emissions. SC subclasses override to unpack tuple."""
        return self(x, training=False)

    def _log_forward(self, emit):
        """Forward algorithm. emit: (batch, n_targets, 2). Returns (log_Z, alphas list)."""
        alphas = [emit[:, 0, :]]                                      # (batch, 2) at position 0
        for t in range(1, self.n_targets):
            a_exp  = alphas[-1][:, :, tf.newaxis]                    # (batch, from, 1)
            scores = a_exp + self.transition[tf.newaxis, :, :]       # (batch, from, to)
            alphas.append(
                tf.reduce_logsumexp(scores, axis=1) + emit[:, t, :])  # (batch, to)
        log_Z = tf.reduce_logsumexp(alphas[-1], axis=-1)              # (batch,)
        return log_Z, alphas

    def _true_score(self, emit, y_int):
        """Score of ground-truth label sequence. y_int: (batch, n_targets) int32 {0,1}."""
        oh    = tf.one_hot(y_int, 2, dtype=tf.float32)               # (batch, n_targets, 2)
        e_sum = tf.reduce_sum(emit * oh, axis=[1, 2])                 # (batch,)
        t_sum = tf.zeros_like(e_sum)
        for t in range(self.n_targets - 1):
            idx   = tf.stack([y_int[:, t], y_int[:, t + 1]], axis=1)  # (batch, 2)
            t_sum = t_sum + tf.gather_nd(self.transition, idx)
        return e_sum + t_sum

    def _nll(self, emit, y_dict):
        """Forward-algorithm NLL given pre-extracted emission logits."""
        y_int = tf.cast(y_dict['cls_out'] if isinstance(y_dict, dict) else y_dict, tf.int32)
        log_Z, _ = self._log_forward(emit)
        return tf.reduce_mean(log_Z - self._true_score(emit, y_int))

    def _step(self, x, y_dict, training):
        return self._nll(self(x, training=training), y_dict)

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            loss = self._step(x, y_dict, training=True)
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return {'loss': loss}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        return {'loss': self._step(x, y_dict, training=False)}

    def predict_binary(self, x):
        """Viterbi decoding → (batch, n_targets) int32."""
        emit   = self._get_emit(x)
        vit    = emit[:, 0, :]                                        # (batch, 2)
        btrack = []
        for t in range(1, self.n_targets):
            scores = vit[:, :, tf.newaxis] + self.transition[tf.newaxis, :, :]  # (batch, from, to)
            btrack.append(tf.argmax(scores, axis=1, output_type=tf.int32))
            vit = tf.reduce_max(scores, axis=1) + emit[:, t, :]
        pred = [tf.argmax(vit, axis=-1, output_type=tf.int32)]
        for bp in reversed(btrack):
            batch_idx = tf.range(tf.shape(pred[-1])[0])
            pred.append(tf.gather_nd(bp, tf.stack([batch_idx, pred[-1]], axis=1)))
        return tf.stack(list(reversed(pred)), axis=1).numpy()

    def predict_marginals(self, x):
        """Forward-backward marginals → (batch, n_targets) float32 in [0,1]."""
        emit          = self._get_emit(x)
        log_Z, alphas = self._log_forward(emit)
        betas = [tf.zeros_like(alphas[-1])]                           # log 1 at final position
        for t in range(self.n_targets - 2, -1, -1):
            b_exp  = betas[0][:, tf.newaxis, :]                      # (batch, 1, to)
            e_exp  = emit[:, t + 1, tf.newaxis, :]                   # (batch, 1, to)
            scores = self.transition[tf.newaxis, :, :] + e_exp + b_exp  # (batch, from, to)
            betas.insert(0, tf.reduce_logsumexp(scores, axis=2))
        marginals = [tf.exp(alphas[t][:, 1] + betas[t][:, 1] - log_Z)
                     for t in range(self.n_targets)]
        return tf.stack(marginals, axis=1).numpy()


@tf.keras.utils.register_keras_serializable(package='ml_crf_chain_sc1')
class MultiLabelCRFChainSC1Model(MultiLabelCRFChainModel):
    """CRF-chain + fused-embedding Jaccard SupCon (SC1). Loss: NLL + λ·SC."""
    def __init__(self, *args, supcon_temp=SUPCON_TEMP, supcon_lambda=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.supcon_temp   = supcon_temp
        self.supcon_lambda = supcon_lambda

    def _get_emit(self, x):
        return self(x, training=False)[0]

    def _step(self, x, y_dict, training):
        y_bin = tf.cast(y_dict['cls_out'], tf.float32)
        emit, proj_norm = self(x, training=training)
        nll      = self._nll(emit, y_dict)
        pos_mask = _jaccard_weight_matrix(y_bin)
        sc       = _supcon_loss_jaccard(proj_norm, pos_mask, self.supcon_temp)
        return nll + self.supcon_lambda * sc, nll, sc

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            loss, nll, sc = self._step(x, y_dict, training=True)
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return {'loss': loss, 'crf_nll': nll, 'supcon': sc}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        loss, nll, sc = self._step(x, y_dict, training=False)
        return {'loss': loss, 'crf_nll': nll, 'supcon': sc}


@tf.keras.utils.register_keras_serializable(package='ml_crf_chain_sc2')
class MultiLabelCRFChainSC2Model(MultiLabelCRFChainModel):
    """CRF-chain + CNN+seq branch Jaccard SupCon (SC2). Loss: NLL + λ·(SC_cnn + SC_seq)."""
    def __init__(self, *args, supcon_temp=SUPCON_TEMP, supcon_lambda_each=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.supcon_temp        = supcon_temp
        self.supcon_lambda_each = supcon_lambda_each

    def _get_emit(self, x):
        return self(x, training=False)[0]

    def _step(self, x, y_dict, training):
        y_bin = tf.cast(y_dict['cls_out'], tf.float32)
        emit, cnn_proj, seq_proj = self(x, training=training)
        nll      = self._nll(emit, y_dict)
        pos_mask = _jaccard_weight_matrix(y_bin)
        sc       = (_supcon_loss_jaccard(cnn_proj, pos_mask, self.supcon_temp)
                    + _supcon_loss_jaccard(seq_proj, pos_mask, self.supcon_temp))
        return nll + self.supcon_lambda_each * sc, nll, sc

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            loss, nll, sc = self._step(x, y_dict, training=True)
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return {'loss': loss, 'crf_nll': nll, 'supcon': sc}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        loss, nll, sc = self._step(x, y_dict, training=False)
        return {'loss': loss, 'crf_nll': nll, 'supcon': sc}


@tf.keras.utils.register_keras_serializable(package='ml_crf_chain_sc3')
class MultiLabelCRFChainSC3Model(MultiLabelCRFChainModel):
    """CRF-chain + CNN+seq+fused Jaccard SupCon (SC3). Loss: NLL + λ·(SC_cnn + SC_seq + SC_fused)."""
    def __init__(self, *args, supcon_temp=SUPCON_TEMP, supcon_lambda_each=0.033, **kwargs):
        super().__init__(*args, **kwargs)
        self.supcon_temp        = supcon_temp
        self.supcon_lambda_each = supcon_lambda_each

    def _get_emit(self, x):
        return self(x, training=False)[0]

    def _step(self, x, y_dict, training):
        y_bin = tf.cast(y_dict['cls_out'], tf.float32)
        emit, cnn_proj, seq_proj, fused_proj = self(x, training=training)
        nll      = self._nll(emit, y_dict)
        pos_mask = _jaccard_weight_matrix(y_bin)
        sc       = (_supcon_loss_jaccard(cnn_proj,   pos_mask, self.supcon_temp)
                    + _supcon_loss_jaccard(seq_proj,   pos_mask, self.supcon_temp)
                    + _supcon_loss_jaccard(fused_proj, pos_mask, self.supcon_temp))
        return nll + self.supcon_lambda_each * sc, nll, sc

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            loss, nll, sc = self._step(x, y_dict, training=True)
        self.optimizer.apply_gradients(
            zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return {'loss': loss, 'crf_nll': nll, 'supcon': sc}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        loss, nll, sc = self._step(x, y_dict, training=False)
        return {'loss': loss, 'crf_nll': nll, 'supcon': sc}


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


# ======================================================================
# LABEL QUERY CROSS-ATTENTION HEAD
# ======================================================================

@tf.keras.utils.register_keras_serializable(package='ml_cross_attn')
class LabelQueryEmbedding(tf.keras.layers.Layer):
    """Learnable label query embeddings; batch dim extracted from ref_tensor."""
    def __init__(self, n_targets, query_dim, **kwargs):
        super().__init__(**kwargs)
        self.n_targets = n_targets
        self.query_dim = query_dim

    def build(self, input_shape):
        self.query_emb = self.add_weight(
            shape=(self.n_targets, self.query_dim),
            initializer='glorot_uniform',
            trainable=True,
            name='query_emb',
        )

    def call(self, ref_tensor):
        batch = tf.shape(ref_tensor)[0]
        q = tf.expand_dims(self.query_emb, 0)
        return tf.tile(q, [batch, 1, 1])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'n_targets': self.n_targets, 'query_dim': self.query_dim})
        return cfg


class CRFFactoredHead(tf.keras.layers.Layer):
    """Per-label emit Dense(1) + fixed decode_mat + learned state bias → (batch, n_states).

    logits[b,s] = Σ_j emit[b,j] * decode_mat[s,j]  +  state_bias[s]
    where decode_mat[s,j] = bit j of state s (0 or 1).
    """
    def __init__(self, n_targets, **kwargs):
        super().__init__(**kwargs)
        self.n_targets  = n_targets
        self.n_states   = 2 ** n_targets
        self.emit_dense = tf.keras.layers.Dense(1, name='crf_emit')

    def build(self, input_shape):
        self.state_bias = self.add_weight(
            shape=(self.n_states,), initializer='zeros', trainable=True, name='state_bias')
        self.decode_mat = tf.constant(
            [[int((s >> j) & 1) for j in range(self.n_targets)]
             for s in range(self.n_states)], dtype=tf.float32)   # (n_states, n_targets)
        super().build(input_shape)

    def call(self, queries):
        emit   = tf.squeeze(self.emit_dense(queries), axis=-1)   # (batch, n_targets)
        logits = tf.einsum('bj,sj->bs', emit, self.decode_mat)   # (batch, n_states)
        return logits + self.state_bias

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'n_targets': self.n_targets})
        return cfg


class CRFBilinearHead(tf.keras.layers.Layer):
    """Per-label-per-state bilinear CRF head.

    logits[b,s] = Σ_j  dot(queries[b,j], state_embs[s,j])
                = einsum('bjd,sjd->bs', queries, state_embs)
    state_embs: (n_states, n_targets, query_dim) trainable.
    """
    def __init__(self, n_targets, query_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.n_targets = n_targets
        self.n_states  = 2 ** n_targets
        self.query_dim = query_dim

    def build(self, input_shape):
        self.state_embs = self.add_weight(
            shape=(self.n_states, self.n_targets, self.query_dim),
            initializer='glorot_uniform', trainable=True, name='state_embs')
        super().build(input_shape)

    def call(self, queries):
        return tf.einsum('bjd,sjd->bs', queries, self.state_embs)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'n_targets': self.n_targets, 'query_dim': self.query_dim})
        return cfg


def _ml_cross_attn_head(kv_seq, ref_tensor, n_targets, query_dim=64, num_heads=4):
    """Label query cross-attention + inter-label self-attention cls head.

    kv_seq:     (batch, T', C) pre-pooling 3D sequence from backbone
    ref_tensor: any tensor used only to extract batch size
    returns:    cls_out (batch, n_targets) sigmoid
    """
    queries = LabelQueryEmbedding(n_targets, query_dim, name='label_q_emb')(ref_tensor)
    kv      = tf.keras.layers.Dense(query_dim, name='ca_kv_proj')(kv_seq)
    ca      = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=query_dim // num_heads,
        name='cross_attn')(query=queries, key=kv, value=kv)
    ca      = tf.keras.layers.LayerNormalization(name='ca_ln')(queries + ca)
    sa      = tf.keras.layers.MultiHeadAttention(
        num_heads=2, key_dim=query_dim // 2, name='inter_label_sa')(ca, ca)
    sa      = tf.keras.layers.LayerNormalization(name='sa_ln')(ca + sa)
    logits  = tf.keras.layers.Dense(1, name='label_logit')(sa)          # (batch, n_t, 1)
    cls_out = tf.keras.layers.Activation('sigmoid', name='cls_out')(
        tf.keras.layers.Reshape((n_targets,), name='ca_reshape')(logits))
    return cls_out


def _build_gru_dual_seq_backbone(inputs, return_branches=False):
    """CNN+GRU dual backbone exposing GRU sequence before pooling for cross-attention.

    Mirrors _build_cnn_gru_dual_branches_mtl with 'ca_' layer name prefix.
    gru_seq: (batch, T, 64) first BiGRU output (kv_seq for cross-attn).
    Returns (z, gru_seq) or (z, gru_seq, cnn_emb, gru_emb).
    """
    c       = tf.keras.layers.Conv1D(16, 5, activation='relu', name='ca_cnn_conv1')(inputs)
    c       = tf.keras.layers.Conv1D(8,  3, activation='relu', name='ca_cnn_conv2')(c)
    c       = tf.keras.layers.Flatten(name='ca_cnn_flat')(c)
    cnn_emb = tf.keras.layers.Dense(32, activation='relu', name='ca_cnn_emb')(c)

    g       = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(32, return_sequences=True), name='ca_bigru1')(inputs)
    gru_seq = g   # (batch, T, 64) — kv_seq, captured before LayerNorm
    g       = tf.keras.layers.LayerNormalization(name='ca_gru_ln')(g)
    g       = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(16), name='ca_bigru2')(g)
    g       = tf.keras.layers.Dropout(0.2, name='ca_gru_drop')(g)
    gru_emb = tf.keras.layers.Dense(64, activation='relu', name='ca_gru_emb')(g)

    merged  = tf.keras.layers.Concatenate(name='ca_cgd_merge')([cnn_emb, gru_emb])
    z       = tf.keras.layers.Dense(96, activation='relu', name='ca_cgd_fused')(merged)
    z       = tf.keras.layers.Dropout(0.2, name='ca_cgd_drop')(z)

    if return_branches:
        return z, gru_seq, cnn_emb, gru_emb
    return z, gru_seq


def _build_trans_dual_seq_backbone(inputs, head_size=32, num_heads=2, ff_dim=32,
                                    num_blocks=2, dropout=0.1, return_branches=False):
    """CNN+Trans dual backbone exposing transformer sequence before pooling for cross-attention.

    Mirrors _build_cnn_trans_dual_branches_mtl with 'ca_' layer name prefix.
    trans_seq: (batch, T', head_size) post-transformer-blocks output (kv_seq).
    Returns (z, trans_seq) or (z, trans_seq, cnn_emb, trans_emb).
    """
    c       = tf.keras.layers.Conv1D(16, 5, activation='relu', name='ca_cnn_conv1')(inputs)
    c       = tf.keras.layers.Conv1D(8,  3, activation='relu', name='ca_cnn_conv2')(c)
    c       = tf.keras.layers.Flatten(name='ca_cnn_flat')(c)
    cnn_emb = tf.keras.layers.Dense(32, activation='relu', name='ca_cnn_emb')(c)

    t       = tf.keras.layers.Conv1D(head_size, 5, strides=2, padding='same',
                                      activation='relu', name='ca_tr_conv')(inputs)
    t       = tf.keras.layers.MaxPooling1D(pool_size=2, padding='same', name='ca_tr_pool')(t)
    new_seq_len = t.shape[1]
    positions   = tf.range(start=0, limit=new_seq_len, delta=1)
    pos_emb     = tf.keras.layers.Embedding(
        new_seq_len, head_size, name='ca_tr_posemb')(positions)
    t = t + pos_emb
    for i in range(num_blocks):
        attn = tf.keras.layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout,
            name=f'ca_tr_mha{i}')(t, t)
        attn = tf.keras.layers.Dropout(dropout, name=f'ca_tr_attn_drop{i}')(attn)
        t    = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name=f'ca_tr_ln1_{i}')(t + attn)
        ffn  = tf.keras.layers.Dense(ff_dim, activation='relu', name=f'ca_tr_ff1_{i}')(t)
        ffn  = tf.keras.layers.Dropout(dropout, name=f'ca_tr_ff_drop{i}')(ffn)
        ffn  = tf.keras.layers.Dense(head_size, name=f'ca_tr_ff2_{i}')(ffn)
        t    = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name=f'ca_tr_ln2_{i}')(t + ffn)
    trans_seq  = t   # (batch, T', head_size) — kv_seq

    t          = tf.keras.layers.GlobalAveragePooling1D(name='ca_tr_gap')(t)
    trans_emb  = tf.keras.layers.Dense(64, activation='relu', name='ca_tr_emb')(t)

    merged     = tf.keras.layers.Concatenate(name='ca_ctd_merge')([cnn_emb, trans_emb])
    z          = tf.keras.layers.Dense(96, activation='relu', name='ca_ctd_fused')(merged)
    z          = tf.keras.layers.Dropout(0.2, name='ca_ctd_drop')(z)

    if return_branches:
        return z, trans_seq, cnn_emb, trans_emb
    return z, trans_seq


# ======================================================================
# DEEP CROSS-ATTENTION v2: BACKBONE HELPERS + HEAD FUNCTIONS
# Layer prefix 'ca2_' throughout to avoid name collision with 'ca_' models.
# clear_session() is called before each model build in the training loop,
# so cross-session naming conflicts do not occur in practice.
# ======================================================================

def _build_gru_dual_deep_seq_backbone(inputs, return_branches=False):
    """CNN+GRU dual with deeper kv_seq: second BiGRU return_sequences=True + learned pos-enc.

    deep_gru_seq: (batch, T, 32) after second BiGRU + positional encoding.
    Returns (z, deep_gru_seq) or (z, deep_gru_seq, cnn_emb, gru_emb).
    """
    c       = tf.keras.layers.Conv1D(16, 5, activation='relu', name='ca2_cnn_conv1')(inputs)
    c       = tf.keras.layers.Conv1D(8,  3, activation='relu', name='ca2_cnn_conv2')(c)
    c       = tf.keras.layers.Flatten(name='ca2_cnn_flat')(c)
    cnn_emb = tf.keras.layers.Dense(32, activation='relu', name='ca2_cnn_emb')(c)

    g       = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(32, return_sequences=True), name='ca2_bigru1')(inputs)
    g       = tf.keras.layers.LayerNormalization(name='ca2_gru_ln')(g)
    g       = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(16, return_sequences=True), name='ca2_bigru2')(g)

    T            = inputs.shape[1]
    pos_emb      = tf.keras.layers.Embedding(T, 32, name='ca2_pos_emb')(tf.range(T))
    deep_gru_seq = g + pos_emb   # (batch, T, 32)

    gru_pool = tf.keras.layers.GlobalAveragePooling1D(name='ca2_bigru2_pool')(deep_gru_seq)
    gru_pool = tf.keras.layers.Dropout(0.2, name='ca2_gru_drop')(gru_pool)
    gru_emb  = tf.keras.layers.Dense(64, activation='relu', name='ca2_gru_emb')(gru_pool)

    merged   = tf.keras.layers.Concatenate(name='ca2_cgd_merge')([cnn_emb, gru_emb])
    z        = tf.keras.layers.Dense(96, activation='relu', name='ca2_cgd_fused')(merged)
    z        = tf.keras.layers.Dropout(0.2, name='ca2_cgd_drop')(z)

    if return_branches:
        return z, deep_gru_seq, cnn_emb, gru_emb
    return z, deep_gru_seq


def _build_trans_dual_deep_seq_backbone(inputs, head_size=32, num_heads=2, ff_dim=32,
                                         num_blocks=2, dropout=0.1, return_branches=False):
    """CNN+Trans dual backbone with 'ca2_' prefix (mirrors _build_trans_dual_seq_backbone).

    Used for cnn_trans_dual_cross_attn_v2. trans_seq: (batch, T', head_size).
    """
    c       = tf.keras.layers.Conv1D(16, 5, activation='relu', name='ca2_cnn_conv1')(inputs)
    c       = tf.keras.layers.Conv1D(8,  3, activation='relu', name='ca2_cnn_conv2')(c)
    c       = tf.keras.layers.Flatten(name='ca2_cnn_flat')(c)
    cnn_emb = tf.keras.layers.Dense(32, activation='relu', name='ca2_cnn_emb')(c)

    t           = tf.keras.layers.Conv1D(head_size, 5, strides=2, padding='same',
                                          activation='relu', name='ca2_tr_conv')(inputs)
    t           = tf.keras.layers.MaxPooling1D(pool_size=2, padding='same',
                                                name='ca2_tr_pool')(t)
    new_seq_len = t.shape[1]
    pos_emb     = tf.keras.layers.Embedding(
        new_seq_len, head_size, name='ca2_tr_posemb')(tf.range(new_seq_len))
    t = t + pos_emb
    for i in range(num_blocks):
        attn = tf.keras.layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout,
            name=f'ca2_tr_mha{i}')(t, t)
        attn = tf.keras.layers.Dropout(dropout, name=f'ca2_tr_attn_drop{i}')(attn)
        t    = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name=f'ca2_tr_ln1_{i}')(t + attn)
        ffn  = tf.keras.layers.Dense(ff_dim, activation='relu', name=f'ca2_tr_ff1_{i}')(t)
        ffn  = tf.keras.layers.Dropout(dropout, name=f'ca2_tr_ff_drop{i}')(ffn)
        ffn  = tf.keras.layers.Dense(head_size, name=f'ca2_tr_ff2_{i}')(ffn)
        t    = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name=f'ca2_tr_ln2_{i}')(t + ffn)
    trans_seq  = t

    t          = tf.keras.layers.GlobalAveragePooling1D(name='ca2_tr_gap')(t)
    trans_emb  = tf.keras.layers.Dense(64, activation='relu', name='ca2_tr_emb')(t)
    merged     = tf.keras.layers.Concatenate(name='ca2_ctd_merge')([cnn_emb, trans_emb])
    z          = tf.keras.layers.Dense(96, activation='relu', name='ca2_ctd_fused')(merged)
    z          = tf.keras.layers.Dropout(0.2, name='ca2_ctd_drop')(z)

    if return_branches:
        return z, trans_seq, cnn_emb, trans_emb
    return z, trans_seq


def _ml_cross_attn_deep_head(kv_seq, ref_tensor, n_targets, query_dim=64, num_heads=4, n_ca_blocks=2):
    """Stacked cross-attn + inter-label SA + FFN blocks with 'ca2_' prefix.

    Returns cls_out (batch, n_targets) sigmoid.
    """
    queries = LabelQueryEmbedding(n_targets, query_dim, name='ca2_label_q_emb')(ref_tensor)
    kv      = tf.keras.layers.Dense(query_dim, name='ca2_kv_proj')(kv_seq)
    for i in range(n_ca_blocks):
        ca      = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=query_dim // num_heads,
            name=f'ca2_cross_attn_{i}')(query=queries, key=kv, value=kv)
        queries = tf.keras.layers.LayerNormalization(name=f'ca2_ca_ln_{i}')(queries + ca)
        sa      = tf.keras.layers.MultiHeadAttention(
            num_heads=2, key_dim=query_dim // 2,
            name=f'ca2_inter_sa_{i}')(queries, queries)
        queries = tf.keras.layers.LayerNormalization(name=f'ca2_sa_ln_{i}')(queries + sa)
        ffn     = tf.keras.layers.Dense(
            query_dim * 2, activation='relu', name=f'ca2_ffn1_{i}')(queries)
        ffn     = tf.keras.layers.Dense(query_dim, name=f'ca2_ffn2_{i}')(ffn)
        queries = tf.keras.layers.LayerNormalization(name=f'ca2_ffn_ln_{i}')(queries + ffn)
    logits  = tf.keras.layers.Dense(1, name='ca2_label_logit')(queries)
    cls_out = tf.keras.layers.Activation('sigmoid', name='cls_out')(
        tf.keras.layers.Reshape((n_targets,), name='ca2_reshape')(logits))
    return cls_out


def _ml_cross_attn_deep_head_aux(kv_seq, ref_tensor, n_targets, query_dim=64, num_heads=4, n_ca_blocks=2):
    """Deep head with per-block auxiliary BCE outputs (tapped after CA, before SA).

    Returns [cls_out, aux0, aux1, ...] — for MultiLabelAuxDetModel.
    """
    queries  = LabelQueryEmbedding(n_targets, query_dim, name='ca2_label_q_emb')(ref_tensor)
    kv       = tf.keras.layers.Dense(query_dim, name='ca2_kv_proj')(kv_seq)
    aux_outs = []
    for i in range(n_ca_blocks):
        ca      = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=query_dim // num_heads,
            name=f'ca2_cross_attn_{i}')(query=queries, key=kv, value=kv)
        queries = tf.keras.layers.LayerNormalization(name=f'ca2_ca_ln_{i}')(queries + ca)
        aux_l   = tf.keras.layers.Dense(1, name=f'ca2_aux_logit_{i}')(queries)
        aux_out = tf.keras.layers.Activation('sigmoid', name=f'ca2_aux_{i}')(
            tf.keras.layers.Reshape((n_targets,), name=f'ca2_aux_reshape_{i}')(aux_l))
        aux_outs.append(aux_out)
        sa      = tf.keras.layers.MultiHeadAttention(
            num_heads=2, key_dim=query_dim // 2,
            name=f'ca2_inter_sa_{i}')(queries, queries)
        queries = tf.keras.layers.LayerNormalization(name=f'ca2_sa_ln_{i}')(queries + sa)
        ffn     = tf.keras.layers.Dense(
            query_dim * 2, activation='relu', name=f'ca2_ffn1_{i}')(queries)
        ffn     = tf.keras.layers.Dense(query_dim, name=f'ca2_ffn2_{i}')(ffn)
        queries = tf.keras.layers.LayerNormalization(name=f'ca2_ffn_ln_{i}')(queries + ffn)
    logits  = tf.keras.layers.Dense(1, name='ca2_label_logit')(queries)
    cls_out = tf.keras.layers.Activation('sigmoid', name='cls_out')(
        tf.keras.layers.Reshape((n_targets,), name='ca2_reshape')(logits))
    return [cls_out] + aux_outs


def _ml_cross_attn_deep_head_quercon(kv_seq, ref_tensor, n_targets, query_dim=64, num_heads=4, n_ca_blocks=2):
    """Deep head exposing L2-normalised pre-logit query vectors for QuerCon loss.

    Returns (cls_out, queries_norm) where queries_norm: (batch, n_targets, query_dim).
    """
    queries = LabelQueryEmbedding(n_targets, query_dim, name='ca2_label_q_emb')(ref_tensor)
    kv      = tf.keras.layers.Dense(query_dim, name='ca2_kv_proj')(kv_seq)
    for i in range(n_ca_blocks):
        ca      = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=query_dim // num_heads,
            name=f'ca2_cross_attn_{i}')(query=queries, key=kv, value=kv)
        queries = tf.keras.layers.LayerNormalization(name=f'ca2_ca_ln_{i}')(queries + ca)
        sa      = tf.keras.layers.MultiHeadAttention(
            num_heads=2, key_dim=query_dim // 2,
            name=f'ca2_inter_sa_{i}')(queries, queries)
        queries = tf.keras.layers.LayerNormalization(name=f'ca2_sa_ln_{i}')(queries + sa)
        ffn     = tf.keras.layers.Dense(
            query_dim * 2, activation='relu', name=f'ca2_ffn1_{i}')(queries)
        ffn     = tf.keras.layers.Dense(query_dim, name=f'ca2_ffn2_{i}')(ffn)
        queries = tf.keras.layers.LayerNormalization(name=f'ca2_ffn_ln_{i}')(queries + ffn)
    queries_norm = tf.keras.layers.Lambda(
        lambda q: tf.math.l2_normalize(q, axis=-1), name='ca2_queries_norm')(queries)
    logits  = tf.keras.layers.Dense(1, name='ca2_label_logit')(queries)
    cls_out = tf.keras.layers.Activation('sigmoid', name='cls_out')(
        tf.keras.layers.Reshape((n_targets,), name='ca2_reshape')(logits))
    return cls_out, queries_norm


def _ml_cross_attn_deep_head_crf_flat(kv_seq, ref_tensor, n_targets,
                                       query_dim=64, num_heads=4, n_ca_blocks=2):
    """Cross-attn deep head + CRF flat projection: Flatten → Dense(2^n_targets) state logits.

    Identical CA stack to _ml_cross_attn_deep_head; final Dense(1)→sigmoid replaced by
    Flatten → Dense(n_states) → raw logits for MultiLabelCRFMRFModel NLL.
    """
    queries = LabelQueryEmbedding(n_targets, query_dim, name='ca2_label_q_emb')(ref_tensor)
    kv      = tf.keras.layers.Dense(query_dim, name='ca2_kv_proj')(kv_seq)
    for i in range(n_ca_blocks):
        ca      = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=query_dim // num_heads,
            name=f'ca2_cross_attn_{i}')(query=queries, key=kv, value=kv)
        queries = tf.keras.layers.LayerNormalization(name=f'ca2_ca_ln_{i}')(queries + ca)
        sa      = tf.keras.layers.MultiHeadAttention(
            num_heads=2, key_dim=query_dim // 2,
            name=f'ca2_inter_sa_{i}')(queries, queries)
        queries = tf.keras.layers.LayerNormalization(name=f'ca2_sa_ln_{i}')(queries + sa)
        ffn     = tf.keras.layers.Dense(
            query_dim * 2, activation='relu', name=f'ca2_ffn1_{i}')(queries)
        ffn     = tf.keras.layers.Dense(query_dim, name=f'ca2_ffn2_{i}')(ffn)
        queries = tf.keras.layers.LayerNormalization(name=f'ca2_ffn_ln_{i}')(queries + ffn)
    flat         = tf.keras.layers.Flatten(name='crf_flat')(queries)
    state_logits = tf.keras.layers.Dense(2 ** n_targets, name='crf_logits')(flat)
    return state_logits


def _ml_cross_attn_deep_head_crf_factored(kv_seq, ref_tensor, n_targets,
                                           query_dim=64, num_heads=4, n_ca_blocks=2):
    """Cross-attn deep head + CRF factored: per-label emit + decode_mat + state_bias.

    logits[b,s] = Σ_j emit[b,j] * decode_mat[s,j] + state_bias[s]  (CRFFactoredHead)
    """
    queries = LabelQueryEmbedding(n_targets, query_dim, name='ca2_label_q_emb')(ref_tensor)
    kv      = tf.keras.layers.Dense(query_dim, name='ca2_kv_proj')(kv_seq)
    for i in range(n_ca_blocks):
        ca      = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=query_dim // num_heads,
            name=f'ca2_cross_attn_{i}')(query=queries, key=kv, value=kv)
        queries = tf.keras.layers.LayerNormalization(name=f'ca2_ca_ln_{i}')(queries + ca)
        sa      = tf.keras.layers.MultiHeadAttention(
            num_heads=2, key_dim=query_dim // 2,
            name=f'ca2_inter_sa_{i}')(queries, queries)
        queries = tf.keras.layers.LayerNormalization(name=f'ca2_sa_ln_{i}')(queries + sa)
        ffn     = tf.keras.layers.Dense(
            query_dim * 2, activation='relu', name=f'ca2_ffn1_{i}')(queries)
        ffn     = tf.keras.layers.Dense(query_dim, name=f'ca2_ffn2_{i}')(ffn)
        queries = tf.keras.layers.LayerNormalization(name=f'ca2_ffn_ln_{i}')(queries + ffn)
    state_logits = CRFFactoredHead(n_targets, name='crf_factored')(queries)
    return state_logits


def _ml_cross_attn_deep_head_crf_bilinear(kv_seq, ref_tensor, n_targets,
                                           query_dim=64, num_heads=4, n_ca_blocks=2):
    """Cross-attn deep head + CRF bilinear: einsum('bjd,sjd->bs', queries, state_embs).

    Per-label-per-state bilinear projection (CRFBilinearHead). No cross-label mixing
    in the CRF head — all label interaction stays in the inter-label SA blocks.
    """
    queries = LabelQueryEmbedding(n_targets, query_dim, name='ca2_label_q_emb')(ref_tensor)
    kv      = tf.keras.layers.Dense(query_dim, name='ca2_kv_proj')(kv_seq)
    for i in range(n_ca_blocks):
        ca      = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=query_dim // num_heads,
            name=f'ca2_cross_attn_{i}')(query=queries, key=kv, value=kv)
        queries = tf.keras.layers.LayerNormalization(name=f'ca2_ca_ln_{i}')(queries + ca)
        sa      = tf.keras.layers.MultiHeadAttention(
            num_heads=2, key_dim=query_dim // 2,
            name=f'ca2_inter_sa_{i}')(queries, queries)
        queries = tf.keras.layers.LayerNormalization(name=f'ca2_sa_ln_{i}')(queries + sa)
        ffn     = tf.keras.layers.Dense(
            query_dim * 2, activation='relu', name=f'ca2_ffn1_{i}')(queries)
        ffn     = tf.keras.layers.Dense(query_dim, name=f'ca2_ffn2_{i}')(ffn)
        queries = tf.keras.layers.LayerNormalization(name=f'ca2_ffn_ln_{i}')(queries + ffn)
    state_logits = CRFBilinearHead(n_targets, query_dim, name='crf_bilinear')(queries)
    return state_logits


# -- CNN+GRU dual cross-attn SC0-3 -------------------------------------------

def create_ml_cnn_gru_dual_cross_attn_model(T, n_targets):
    inputs    = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq = _build_gru_dual_seq_backbone(inputs)
    cls_out   = _ml_cross_attn_head(kv_seq, inputs, n_targets)
    return _StandardMultiLabelModel(inputs=inputs, outputs=cls_out,
                                    name='cnn_gru_dual_cross_attn')


def create_ml_cnn_gru_dual_cross_attn_supcon_model(T, n_targets):
    inputs    = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq = _build_gru_dual_seq_backbone(inputs)
    cls_out   = _ml_cross_attn_head(kv_seq, inputs, n_targets)
    proj_norm = _proj_head(z, 'fused')
    return MultiLabelSupConModel(inputs=inputs, outputs=[cls_out, proj_norm])


def create_ml_cnn_gru_dual_cross_attn_supcon2_model(T, n_targets):
    inputs                       = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq, cnn_emb, gru_emb = _build_gru_dual_seq_backbone(inputs, return_branches=True)
    cls_out  = _ml_cross_attn_head(kv_seq, inputs, n_targets)
    cnn_proj = _proj_head(cnn_emb, 'cnn')
    seq_proj = _proj_head(gru_emb, 'seq')
    return MultiLabelSupConBranch2STModel(inputs=inputs, outputs=[cls_out, cnn_proj, seq_proj])


def create_ml_cnn_gru_dual_cross_attn_supcon3_model(T, n_targets):
    inputs                       = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq, cnn_emb, gru_emb = _build_gru_dual_seq_backbone(inputs, return_branches=True)
    cls_out    = _ml_cross_attn_head(kv_seq, inputs, n_targets)
    cnn_proj   = _proj_head(cnn_emb, 'cnn')
    seq_proj   = _proj_head(gru_emb, 'seq')
    fused_proj = _proj_head(z, 'fused')
    return MultiLabelSupConBranch3STModel(
        inputs=inputs, outputs=[cls_out, cnn_proj, seq_proj, fused_proj])


# -- CNN+Trans dual cross-attn SC0-3 -----------------------------------------

def create_ml_cnn_trans_dual_cross_attn_model(T, n_targets):
    inputs       = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, trans_seq = _build_trans_dual_seq_backbone(inputs)
    cls_out      = _ml_cross_attn_head(trans_seq, inputs, n_targets)
    return _StandardMultiLabelModel(inputs=inputs, outputs=cls_out,
                                    name='cnn_trans_dual_cross_attn')


def create_ml_cnn_trans_dual_cross_attn_supcon_model(T, n_targets):
    inputs       = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, trans_seq = _build_trans_dual_seq_backbone(inputs)
    cls_out      = _ml_cross_attn_head(trans_seq, inputs, n_targets)
    proj_norm    = _proj_head(z, 'fused')
    return MultiLabelSupConModel(inputs=inputs, outputs=[cls_out, proj_norm])


def create_ml_cnn_trans_dual_cross_attn_supcon2_model(T, n_targets):
    inputs                          = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, trans_seq, cnn_emb, tr_emb  = _build_trans_dual_seq_backbone(inputs, return_branches=True)
    cls_out  = _ml_cross_attn_head(trans_seq, inputs, n_targets)
    cnn_proj = _proj_head(cnn_emb, 'cnn')
    seq_proj = _proj_head(tr_emb, 'seq')
    return MultiLabelSupConBranch2STModel(inputs=inputs, outputs=[cls_out, cnn_proj, seq_proj])


def create_ml_cnn_trans_dual_cross_attn_supcon3_model(T, n_targets):
    inputs                          = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, trans_seq, cnn_emb, tr_emb  = _build_trans_dual_seq_backbone(inputs, return_branches=True)
    cls_out    = _ml_cross_attn_head(trans_seq, inputs, n_targets)
    cnn_proj   = _proj_head(cnn_emb, 'cnn')
    seq_proj   = _proj_head(tr_emb, 'seq')
    fused_proj = _proj_head(z, 'fused')
    return MultiLabelSupConBranch3STModel(
        inputs=inputs, outputs=[cls_out, cnn_proj, seq_proj, fused_proj])


# -- CNN+GRU dual cross-attn DeepKV SC0-3 -----------------------------------
# Shallow head (_ml_cross_attn_head) + deep backbone (_build_gru_dual_deep_seq_backbone)

def create_ml_cnn_gru_dual_cross_attn_deepkv_model(T, n_targets):
    inputs    = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq = _build_gru_dual_deep_seq_backbone(inputs)
    cls_out   = _ml_cross_attn_head(kv_seq, inputs, n_targets)
    return _StandardMultiLabelModel(inputs=inputs, outputs=cls_out,
                                    name='cnn_gru_dual_cross_attn_deepkv')

def create_ml_cnn_gru_dual_cross_attn_deepkv_supcon_model(T, n_targets):
    inputs    = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq = _build_gru_dual_deep_seq_backbone(inputs)
    cls_out   = _ml_cross_attn_head(kv_seq, inputs, n_targets)
    proj_norm = _proj_head(z, 'fused')
    return MultiLabelSupConModel(inputs=inputs, outputs=[cls_out, proj_norm])

def create_ml_cnn_gru_dual_cross_attn_deepkv_supcon2_model(T, n_targets):
    inputs                       = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq, cnn_emb, gru_emb = _build_gru_dual_deep_seq_backbone(inputs, return_branches=True)
    cls_out  = _ml_cross_attn_head(kv_seq, inputs, n_targets)
    cnn_proj = _proj_head(cnn_emb, 'cnn')
    seq_proj = _proj_head(gru_emb, 'seq')
    return MultiLabelSupConBranch2STModel(inputs=inputs, outputs=[cls_out, cnn_proj, seq_proj])

def create_ml_cnn_gru_dual_cross_attn_deepkv_supcon3_model(T, n_targets):
    inputs                       = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq, cnn_emb, gru_emb = _build_gru_dual_deep_seq_backbone(inputs, return_branches=True)
    cls_out    = _ml_cross_attn_head(kv_seq, inputs, n_targets)
    cnn_proj   = _proj_head(cnn_emb, 'cnn')
    seq_proj   = _proj_head(gru_emb, 'seq')
    fused_proj = _proj_head(z, 'fused')
    return MultiLabelSupConBranch3STModel(
        inputs=inputs, outputs=[cls_out, cnn_proj, seq_proj, fused_proj])


# -- CNN+GRU dual cross-attn DeepHead SC0-3 ----------------------------------
# Deep head (_ml_cross_attn_deep_head) + shallow backbone (_build_gru_dual_seq_backbone)

def create_ml_cnn_gru_dual_cross_attn_deephead_model(T, n_targets):
    inputs    = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq = _build_gru_dual_seq_backbone(inputs)
    cls_out   = _ml_cross_attn_deep_head(kv_seq, inputs, n_targets)
    return _StandardMultiLabelModel(inputs=inputs, outputs=cls_out,
                                    name='cnn_gru_dual_cross_attn_deephead')

def create_ml_cnn_gru_dual_cross_attn_deephead_supcon_model(T, n_targets):
    inputs    = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq = _build_gru_dual_seq_backbone(inputs)
    cls_out   = _ml_cross_attn_deep_head(kv_seq, inputs, n_targets)
    proj_norm = _proj_head(z, 'fused')
    return MultiLabelSupConModel(inputs=inputs, outputs=[cls_out, proj_norm])

def create_ml_cnn_gru_dual_cross_attn_deephead_supcon2_model(T, n_targets):
    inputs                       = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq, cnn_emb, gru_emb = _build_gru_dual_seq_backbone(inputs, return_branches=True)
    cls_out  = _ml_cross_attn_deep_head(kv_seq, inputs, n_targets)
    cnn_proj = _proj_head(cnn_emb, 'cnn')
    seq_proj = _proj_head(gru_emb, 'seq')
    return MultiLabelSupConBranch2STModel(inputs=inputs, outputs=[cls_out, cnn_proj, seq_proj])

def create_ml_cnn_gru_dual_cross_attn_deephead_supcon3_model(T, n_targets):
    inputs                       = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq, cnn_emb, gru_emb = _build_gru_dual_seq_backbone(inputs, return_branches=True)
    cls_out    = _ml_cross_attn_deep_head(kv_seq, inputs, n_targets)
    cnn_proj   = _proj_head(cnn_emb, 'cnn')
    seq_proj   = _proj_head(gru_emb, 'seq')
    fused_proj = _proj_head(z, 'fused')
    return MultiLabelSupConBranch3STModel(
        inputs=inputs, outputs=[cls_out, cnn_proj, seq_proj, fused_proj])


# -- CNN+GRU dual cross-attn V2 SC0-3 ----------------------------------------
# Deep head + deep backbone (combined best-of-both)

def create_ml_cnn_gru_dual_cross_attn_v2_model(T, n_targets):
    inputs    = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq = _build_gru_dual_deep_seq_backbone(inputs)
    cls_out   = _ml_cross_attn_deep_head(kv_seq, inputs, n_targets)
    return _StandardMultiLabelModel(inputs=inputs, outputs=cls_out,
                                    name='cnn_gru_dual_cross_attn_v2')

def create_ml_cnn_gru_dual_cross_attn_v2_supcon_model(T, n_targets):
    inputs    = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq = _build_gru_dual_deep_seq_backbone(inputs)
    cls_out   = _ml_cross_attn_deep_head(kv_seq, inputs, n_targets)
    proj_norm = _proj_head(z, 'fused')
    return MultiLabelSupConModel(inputs=inputs, outputs=[cls_out, proj_norm])

def create_ml_cnn_gru_dual_cross_attn_v2_supcon2_model(T, n_targets):
    inputs                       = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq, cnn_emb, gru_emb = _build_gru_dual_deep_seq_backbone(inputs, return_branches=True)
    cls_out  = _ml_cross_attn_deep_head(kv_seq, inputs, n_targets)
    cnn_proj = _proj_head(cnn_emb, 'cnn')
    seq_proj = _proj_head(gru_emb, 'seq')
    return MultiLabelSupConBranch2STModel(inputs=inputs, outputs=[cls_out, cnn_proj, seq_proj])

def create_ml_cnn_gru_dual_cross_attn_v2_supcon3_model(T, n_targets):
    inputs                       = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq, cnn_emb, gru_emb = _build_gru_dual_deep_seq_backbone(inputs, return_branches=True)
    cls_out    = _ml_cross_attn_deep_head(kv_seq, inputs, n_targets)
    cnn_proj   = _proj_head(cnn_emb, 'cnn')
    seq_proj   = _proj_head(gru_emb, 'seq')
    fused_proj = _proj_head(z, 'fused')
    return MultiLabelSupConBranch3STModel(
        inputs=inputs, outputs=[cls_out, cnn_proj, seq_proj, fused_proj])


# -- CNN+Trans dual cross-attn V2 SC0-3 --------------------------------------

def create_ml_cnn_trans_dual_cross_attn_v2_model(T, n_targets):
    inputs         = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, trans_seq   = _build_trans_dual_deep_seq_backbone(inputs)
    cls_out        = _ml_cross_attn_deep_head(trans_seq, inputs, n_targets)
    return _StandardMultiLabelModel(inputs=inputs, outputs=cls_out,
                                    name='cnn_trans_dual_cross_attn_v2')

def create_ml_cnn_trans_dual_cross_attn_v2_supcon_model(T, n_targets):
    inputs         = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, trans_seq   = _build_trans_dual_deep_seq_backbone(inputs)
    cls_out        = _ml_cross_attn_deep_head(trans_seq, inputs, n_targets)
    proj_norm      = _proj_head(z, 'fused')
    return MultiLabelSupConModel(inputs=inputs, outputs=[cls_out, proj_norm])

def create_ml_cnn_trans_dual_cross_attn_v2_supcon2_model(T, n_targets):
    inputs                         = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, trans_seq, cnn_emb, tr_emb  = _build_trans_dual_deep_seq_backbone(inputs, return_branches=True)
    cls_out  = _ml_cross_attn_deep_head(trans_seq, inputs, n_targets)
    cnn_proj = _proj_head(cnn_emb, 'cnn')
    seq_proj = _proj_head(tr_emb, 'seq')
    return MultiLabelSupConBranch2STModel(inputs=inputs, outputs=[cls_out, cnn_proj, seq_proj])

def create_ml_cnn_trans_dual_cross_attn_v2_supcon3_model(T, n_targets):
    inputs                         = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, trans_seq, cnn_emb, tr_emb  = _build_trans_dual_deep_seq_backbone(inputs, return_branches=True)
    cls_out    = _ml_cross_attn_deep_head(trans_seq, inputs, n_targets)
    cnn_proj   = _proj_head(cnn_emb, 'cnn')
    seq_proj   = _proj_head(tr_emb, 'seq')
    fused_proj = _proj_head(z, 'fused')
    return MultiLabelSupConBranch3STModel(
        inputs=inputs, outputs=[cls_out, cnn_proj, seq_proj, fused_proj])


# -- CNN+GRU V2 AuxDet SC0-3 -------------------------------------------------
# Deep backbone + deep head with per-block aux BCE; aux_outs count = n_ca_blocks (default 2)
# Output convention: [cls_out, aux0, aux1, ...projections...]

def create_ml_cnn_gru_dual_cross_attn_v2_auxdet_model(T, n_targets):
    inputs      = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq   = _build_gru_dual_deep_seq_backbone(inputs)
    outputs     = _ml_cross_attn_deep_head_aux(kv_seq, inputs, n_targets)
    return MultiLabelAuxDetModel(inputs=inputs, outputs=outputs)

def create_ml_cnn_gru_dual_cross_attn_v2_auxdet_supcon_model(T, n_targets):
    inputs    = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq = _build_gru_dual_deep_seq_backbone(inputs)
    aux_outs  = _ml_cross_attn_deep_head_aux(kv_seq, inputs, n_targets)  # [cls_out, aux0, aux1]
    proj_norm = _proj_head(z, 'fused')
    return MultiLabelAuxDetSC1Model(inputs=inputs, outputs=aux_outs + [proj_norm])

def create_ml_cnn_gru_dual_cross_attn_v2_auxdet_supcon2_model(T, n_targets):
    inputs                       = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq, cnn_emb, gru_emb = _build_gru_dual_deep_seq_backbone(inputs, return_branches=True)
    aux_outs = _ml_cross_attn_deep_head_aux(kv_seq, inputs, n_targets)
    cnn_proj = _proj_head(cnn_emb, 'cnn')
    seq_proj = _proj_head(gru_emb, 'seq')
    return MultiLabelAuxDetSC2Model(inputs=inputs, outputs=aux_outs + [cnn_proj, seq_proj])

def create_ml_cnn_gru_dual_cross_attn_v2_auxdet_supcon3_model(T, n_targets):
    inputs                       = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq, cnn_emb, gru_emb = _build_gru_dual_deep_seq_backbone(inputs, return_branches=True)
    aux_outs   = _ml_cross_attn_deep_head_aux(kv_seq, inputs, n_targets)
    cnn_proj   = _proj_head(cnn_emb, 'cnn')
    seq_proj   = _proj_head(gru_emb, 'seq')
    fused_proj = _proj_head(z, 'fused')
    return MultiLabelAuxDetSC3Model(inputs=inputs, outputs=aux_outs + [cnn_proj, seq_proj, fused_proj])


# -- CNN+GRU V2 QuerCon SC0-3 ------------------------------------------------
# Deep backbone + quercon head; outputs [cls_out, queries_norm, ...projections...]

def create_ml_cnn_gru_dual_cross_attn_v2_quercon_model(T, n_targets):
    inputs             = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq          = _build_gru_dual_deep_seq_backbone(inputs)
    cls_out, q_norm    = _ml_cross_attn_deep_head_quercon(kv_seq, inputs, n_targets)
    return MultiLabelQueryConModel(inputs=inputs, outputs=[cls_out, q_norm])

def create_ml_cnn_gru_dual_cross_attn_v2_quercon_supcon_model(T, n_targets):
    inputs          = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq       = _build_gru_dual_deep_seq_backbone(inputs)
    cls_out, q_norm = _ml_cross_attn_deep_head_quercon(kv_seq, inputs, n_targets)
    proj_norm       = _proj_head(z, 'fused')
    return MultiLabelQueryConSC1Model(inputs=inputs, outputs=[cls_out, q_norm, proj_norm])

def create_ml_cnn_gru_dual_cross_attn_v2_quercon_supcon2_model(T, n_targets):
    inputs                       = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq, cnn_emb, gru_emb = _build_gru_dual_deep_seq_backbone(inputs, return_branches=True)
    cls_out, q_norm = _ml_cross_attn_deep_head_quercon(kv_seq, inputs, n_targets)
    cnn_proj        = _proj_head(cnn_emb, 'cnn')
    seq_proj        = _proj_head(gru_emb, 'seq')
    return MultiLabelQueryConSC2Model(inputs=inputs, outputs=[cls_out, q_norm, cnn_proj, seq_proj])

def create_ml_cnn_gru_dual_cross_attn_v2_quercon_supcon3_model(T, n_targets):
    inputs                       = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq, cnn_emb, gru_emb = _build_gru_dual_deep_seq_backbone(inputs, return_branches=True)
    cls_out, q_norm = _ml_cross_attn_deep_head_quercon(kv_seq, inputs, n_targets)
    cnn_proj        = _proj_head(cnn_emb, 'cnn')
    seq_proj        = _proj_head(gru_emb, 'seq')
    fused_proj      = _proj_head(z, 'fused')
    return MultiLabelQueryConSC3Model(
        inputs=inputs, outputs=[cls_out, q_norm, cnn_proj, seq_proj, fused_proj])



# ── CAttn-V2 + CRF-MRF factories: flat / factored / bilinear × SC0-3 × CGD+CTD ──

# -- CGD (CNN+GRU dual) × flat SC0-3 ----------------------------------------

def create_ml_cnn_gru_dual_cross_attn_v2_crf_flat_model(T, n_targets):
    inputs    = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq = _build_gru_dual_deep_seq_backbone(inputs)
    logits    = _ml_cross_attn_deep_head_crf_flat(kv_seq, inputs, n_targets)
    return MultiLabelCRFMRFModel(inputs=inputs, outputs=logits, n_targets=n_targets)

def create_ml_cnn_gru_dual_cross_attn_v2_crf_flat_supcon_model(T, n_targets):
    inputs    = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq = _build_gru_dual_deep_seq_backbone(inputs)
    logits    = _ml_cross_attn_deep_head_crf_flat(kv_seq, inputs, n_targets)
    proj_norm = _proj_head(z, 'fused')
    return MultiLabelCRFMRFSC1Model(inputs=inputs, outputs=[logits, proj_norm], n_targets=n_targets)

def create_ml_cnn_gru_dual_cross_attn_v2_crf_flat_supcon2_model(T, n_targets):
    inputs                       = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq, cnn_emb, gru_emb = _build_gru_dual_deep_seq_backbone(inputs, return_branches=True)
    logits   = _ml_cross_attn_deep_head_crf_flat(kv_seq, inputs, n_targets)
    cnn_proj = _proj_head(cnn_emb, 'cnn')
    seq_proj = _proj_head(gru_emb, 'seq')
    return MultiLabelCRFMRFSC2Model(inputs=inputs, outputs=[logits, cnn_proj, seq_proj], n_targets=n_targets)

def create_ml_cnn_gru_dual_cross_attn_v2_crf_flat_supcon3_model(T, n_targets):
    inputs                       = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq, cnn_emb, gru_emb = _build_gru_dual_deep_seq_backbone(inputs, return_branches=True)
    logits     = _ml_cross_attn_deep_head_crf_flat(kv_seq, inputs, n_targets)
    cnn_proj   = _proj_head(cnn_emb, 'cnn')
    seq_proj   = _proj_head(gru_emb, 'seq')
    fused_proj = _proj_head(z, 'fused')
    return MultiLabelCRFMRFSC3Model(inputs=inputs, outputs=[logits, cnn_proj, seq_proj, fused_proj], n_targets=n_targets)


# -- CGD × factored SC0-3 ----------------------------------------------------

def create_ml_cnn_gru_dual_cross_attn_v2_crf_factored_model(T, n_targets):
    inputs    = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq = _build_gru_dual_deep_seq_backbone(inputs)
    logits    = _ml_cross_attn_deep_head_crf_factored(kv_seq, inputs, n_targets)
    return MultiLabelCRFMRFModel(inputs=inputs, outputs=logits, n_targets=n_targets)

def create_ml_cnn_gru_dual_cross_attn_v2_crf_factored_supcon_model(T, n_targets):
    inputs    = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq = _build_gru_dual_deep_seq_backbone(inputs)
    logits    = _ml_cross_attn_deep_head_crf_factored(kv_seq, inputs, n_targets)
    proj_norm = _proj_head(z, 'fused')
    return MultiLabelCRFMRFSC1Model(inputs=inputs, outputs=[logits, proj_norm], n_targets=n_targets)

def create_ml_cnn_gru_dual_cross_attn_v2_crf_factored_supcon2_model(T, n_targets):
    inputs                       = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq, cnn_emb, gru_emb = _build_gru_dual_deep_seq_backbone(inputs, return_branches=True)
    logits   = _ml_cross_attn_deep_head_crf_factored(kv_seq, inputs, n_targets)
    cnn_proj = _proj_head(cnn_emb, 'cnn')
    seq_proj = _proj_head(gru_emb, 'seq')
    return MultiLabelCRFMRFSC2Model(inputs=inputs, outputs=[logits, cnn_proj, seq_proj], n_targets=n_targets)

def create_ml_cnn_gru_dual_cross_attn_v2_crf_factored_supcon3_model(T, n_targets):
    inputs                       = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq, cnn_emb, gru_emb = _build_gru_dual_deep_seq_backbone(inputs, return_branches=True)
    logits     = _ml_cross_attn_deep_head_crf_factored(kv_seq, inputs, n_targets)
    cnn_proj   = _proj_head(cnn_emb, 'cnn')
    seq_proj   = _proj_head(gru_emb, 'seq')
    fused_proj = _proj_head(z, 'fused')
    return MultiLabelCRFMRFSC3Model(inputs=inputs, outputs=[logits, cnn_proj, seq_proj, fused_proj], n_targets=n_targets)


# -- CGD × bilinear SC0-3 ----------------------------------------------------

def create_ml_cnn_gru_dual_cross_attn_v2_crf_bilinear_model(T, n_targets):
    inputs    = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq = _build_gru_dual_deep_seq_backbone(inputs)
    logits    = _ml_cross_attn_deep_head_crf_bilinear(kv_seq, inputs, n_targets)
    return MultiLabelCRFMRFModel(inputs=inputs, outputs=logits, n_targets=n_targets)

def create_ml_cnn_gru_dual_cross_attn_v2_crf_bilinear_supcon_model(T, n_targets):
    inputs    = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq = _build_gru_dual_deep_seq_backbone(inputs)
    logits    = _ml_cross_attn_deep_head_crf_bilinear(kv_seq, inputs, n_targets)
    proj_norm = _proj_head(z, 'fused')
    return MultiLabelCRFMRFSC1Model(inputs=inputs, outputs=[logits, proj_norm], n_targets=n_targets)

def create_ml_cnn_gru_dual_cross_attn_v2_crf_bilinear_supcon2_model(T, n_targets):
    inputs                       = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq, cnn_emb, gru_emb = _build_gru_dual_deep_seq_backbone(inputs, return_branches=True)
    logits   = _ml_cross_attn_deep_head_crf_bilinear(kv_seq, inputs, n_targets)
    cnn_proj = _proj_head(cnn_emb, 'cnn')
    seq_proj = _proj_head(gru_emb, 'seq')
    return MultiLabelCRFMRFSC2Model(inputs=inputs, outputs=[logits, cnn_proj, seq_proj], n_targets=n_targets)

def create_ml_cnn_gru_dual_cross_attn_v2_crf_bilinear_supcon3_model(T, n_targets):
    inputs                       = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, kv_seq, cnn_emb, gru_emb = _build_gru_dual_deep_seq_backbone(inputs, return_branches=True)
    logits     = _ml_cross_attn_deep_head_crf_bilinear(kv_seq, inputs, n_targets)
    cnn_proj   = _proj_head(cnn_emb, 'cnn')
    seq_proj   = _proj_head(gru_emb, 'seq')
    fused_proj = _proj_head(z, 'fused')
    return MultiLabelCRFMRFSC3Model(inputs=inputs, outputs=[logits, cnn_proj, seq_proj, fused_proj], n_targets=n_targets)


# -- CTD (CNN+Trans dual) × flat SC0-3 ---------------------------------------

def create_ml_cnn_trans_dual_cross_attn_v2_crf_flat_model(T, n_targets):
    inputs       = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, trans_seq = _build_trans_dual_deep_seq_backbone(inputs)
    logits       = _ml_cross_attn_deep_head_crf_flat(trans_seq, inputs, n_targets)
    return MultiLabelCRFMRFModel(inputs=inputs, outputs=logits, n_targets=n_targets)

def create_ml_cnn_trans_dual_cross_attn_v2_crf_flat_supcon_model(T, n_targets):
    inputs       = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, trans_seq = _build_trans_dual_deep_seq_backbone(inputs)
    logits       = _ml_cross_attn_deep_head_crf_flat(trans_seq, inputs, n_targets)
    proj_norm    = _proj_head(z, 'fused')
    return MultiLabelCRFMRFSC1Model(inputs=inputs, outputs=[logits, proj_norm], n_targets=n_targets)

def create_ml_cnn_trans_dual_cross_attn_v2_crf_flat_supcon2_model(T, n_targets):
    inputs                         = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, trans_seq, cnn_emb, tr_emb  = _build_trans_dual_deep_seq_backbone(inputs, return_branches=True)
    logits   = _ml_cross_attn_deep_head_crf_flat(trans_seq, inputs, n_targets)
    cnn_proj = _proj_head(cnn_emb, 'cnn')
    seq_proj = _proj_head(tr_emb,  'seq')
    return MultiLabelCRFMRFSC2Model(inputs=inputs, outputs=[logits, cnn_proj, seq_proj], n_targets=n_targets)

def create_ml_cnn_trans_dual_cross_attn_v2_crf_flat_supcon3_model(T, n_targets):
    inputs                         = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, trans_seq, cnn_emb, tr_emb  = _build_trans_dual_deep_seq_backbone(inputs, return_branches=True)
    logits     = _ml_cross_attn_deep_head_crf_flat(trans_seq, inputs, n_targets)
    cnn_proj   = _proj_head(cnn_emb, 'cnn')
    seq_proj   = _proj_head(tr_emb,  'seq')
    fused_proj = _proj_head(z, 'fused')
    return MultiLabelCRFMRFSC3Model(inputs=inputs, outputs=[logits, cnn_proj, seq_proj, fused_proj], n_targets=n_targets)


# -- CTD × factored SC0-3 ----------------------------------------------------

def create_ml_cnn_trans_dual_cross_attn_v2_crf_factored_model(T, n_targets):
    inputs       = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, trans_seq = _build_trans_dual_deep_seq_backbone(inputs)
    logits       = _ml_cross_attn_deep_head_crf_factored(trans_seq, inputs, n_targets)
    return MultiLabelCRFMRFModel(inputs=inputs, outputs=logits, n_targets=n_targets)

def create_ml_cnn_trans_dual_cross_attn_v2_crf_factored_supcon_model(T, n_targets):
    inputs       = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, trans_seq = _build_trans_dual_deep_seq_backbone(inputs)
    logits       = _ml_cross_attn_deep_head_crf_factored(trans_seq, inputs, n_targets)
    proj_norm    = _proj_head(z, 'fused')
    return MultiLabelCRFMRFSC1Model(inputs=inputs, outputs=[logits, proj_norm], n_targets=n_targets)

def create_ml_cnn_trans_dual_cross_attn_v2_crf_factored_supcon2_model(T, n_targets):
    inputs                         = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, trans_seq, cnn_emb, tr_emb  = _build_trans_dual_deep_seq_backbone(inputs, return_branches=True)
    logits   = _ml_cross_attn_deep_head_crf_factored(trans_seq, inputs, n_targets)
    cnn_proj = _proj_head(cnn_emb, 'cnn')
    seq_proj = _proj_head(tr_emb,  'seq')
    return MultiLabelCRFMRFSC2Model(inputs=inputs, outputs=[logits, cnn_proj, seq_proj], n_targets=n_targets)

def create_ml_cnn_trans_dual_cross_attn_v2_crf_factored_supcon3_model(T, n_targets):
    inputs                         = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, trans_seq, cnn_emb, tr_emb  = _build_trans_dual_deep_seq_backbone(inputs, return_branches=True)
    logits     = _ml_cross_attn_deep_head_crf_factored(trans_seq, inputs, n_targets)
    cnn_proj   = _proj_head(cnn_emb, 'cnn')
    seq_proj   = _proj_head(tr_emb,  'seq')
    fused_proj = _proj_head(z, 'fused')
    return MultiLabelCRFMRFSC3Model(inputs=inputs, outputs=[logits, cnn_proj, seq_proj, fused_proj], n_targets=n_targets)


# -- CTD × bilinear SC0-3 ----------------------------------------------------

def create_ml_cnn_trans_dual_cross_attn_v2_crf_bilinear_model(T, n_targets):
    inputs       = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, trans_seq = _build_trans_dual_deep_seq_backbone(inputs)
    logits       = _ml_cross_attn_deep_head_crf_bilinear(trans_seq, inputs, n_targets)
    return MultiLabelCRFMRFModel(inputs=inputs, outputs=logits, n_targets=n_targets)

def create_ml_cnn_trans_dual_cross_attn_v2_crf_bilinear_supcon_model(T, n_targets):
    inputs       = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, trans_seq = _build_trans_dual_deep_seq_backbone(inputs)
    logits       = _ml_cross_attn_deep_head_crf_bilinear(trans_seq, inputs, n_targets)
    proj_norm    = _proj_head(z, 'fused')
    return MultiLabelCRFMRFSC1Model(inputs=inputs, outputs=[logits, proj_norm], n_targets=n_targets)

def create_ml_cnn_trans_dual_cross_attn_v2_crf_bilinear_supcon2_model(T, n_targets):
    inputs                         = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, trans_seq, cnn_emb, tr_emb  = _build_trans_dual_deep_seq_backbone(inputs, return_branches=True)
    logits   = _ml_cross_attn_deep_head_crf_bilinear(trans_seq, inputs, n_targets)
    cnn_proj = _proj_head(cnn_emb, 'cnn')
    seq_proj = _proj_head(tr_emb,  'seq')
    return MultiLabelCRFMRFSC2Model(inputs=inputs, outputs=[logits, cnn_proj, seq_proj], n_targets=n_targets)

def create_ml_cnn_trans_dual_cross_attn_v2_crf_bilinear_supcon3_model(T, n_targets):
    inputs                         = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z, trans_seq, cnn_emb, tr_emb  = _build_trans_dual_deep_seq_backbone(inputs, return_branches=True)
    logits     = _ml_cross_attn_deep_head_crf_bilinear(trans_seq, inputs, n_targets)
    cnn_proj   = _proj_head(cnn_emb, 'cnn')
    seq_proj   = _proj_head(tr_emb,  'seq')
    fused_proj = _proj_head(z, 'fused')
    return MultiLabelCRFMRFSC3Model(inputs=inputs, outputs=[logits, cnn_proj, seq_proj, fused_proj], n_targets=n_targets)


# ── CRF-MRF factories (Option A: full-state 2^n_targets NLL) ────────────────

def create_ml_cnn_gru_dual_crf_mrf_model(T, n_targets):
    """CNN+GRU dual + full-state MRF SC0."""
    inputs       = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z            = _build_cnn_gru_dual_branches_mtl(inputs)
    state_logits = tf.keras.layers.Dense(2 ** n_targets, name='crf_logits')(z)
    return MultiLabelCRFMRFModel(inputs=inputs, outputs=state_logits, n_targets=n_targets)


def create_ml_cnn_gru_dual_crf_mrf_supcon_model(T, n_targets):
    """CNN+GRU dual + full-state MRF SC1 (fused SupCon)."""
    inputs       = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z            = _build_cnn_gru_dual_branches_mtl(inputs)
    state_logits = tf.keras.layers.Dense(2 ** n_targets, name='crf_logits')(z)
    proj_norm    = _proj_head(z, 'fused')
    return MultiLabelCRFMRFSC1Model(inputs=inputs, outputs=[state_logits, proj_norm], n_targets=n_targets)


def create_ml_cnn_gru_dual_crf_mrf_supcon2_model(T, n_targets):
    """CNN+GRU dual + full-state MRF SC2 (CNN+seq SupCon)."""
    inputs               = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    cnn_emb, seq_emb, z = _build_cnn_gru_dual_branches_mtl(inputs, return_branches=True)
    state_logits         = tf.keras.layers.Dense(2 ** n_targets, name='crf_logits')(z)
    cnn_proj             = _proj_head(cnn_emb, 'cnn')
    seq_proj             = _proj_head(seq_emb, 'seq')
    return MultiLabelCRFMRFSC2Model(inputs=inputs, outputs=[state_logits, cnn_proj, seq_proj], n_targets=n_targets)


def create_ml_cnn_gru_dual_crf_mrf_supcon3_model(T, n_targets):
    """CNN+GRU dual + full-state MRF SC3 (CNN+seq+fused SupCon)."""
    inputs               = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    cnn_emb, seq_emb, z = _build_cnn_gru_dual_branches_mtl(inputs, return_branches=True)
    state_logits         = tf.keras.layers.Dense(2 ** n_targets, name='crf_logits')(z)
    cnn_proj             = _proj_head(cnn_emb, 'cnn')
    seq_proj             = _proj_head(seq_emb, 'seq')
    fused_proj           = _proj_head(z, 'fused')
    return MultiLabelCRFMRFSC3Model(inputs=inputs, outputs=[state_logits, cnn_proj, seq_proj, fused_proj], n_targets=n_targets)


def create_ml_cnn_trans_dual_crf_mrf_model(T, n_targets):
    """CNN+Trans dual + full-state MRF SC0."""
    inputs       = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z            = _build_cnn_trans_dual_branches_mtl(inputs)
    state_logits = tf.keras.layers.Dense(2 ** n_targets, name='crf_logits')(z)
    return MultiLabelCRFMRFModel(inputs=inputs, outputs=state_logits, n_targets=n_targets)


def create_ml_cnn_trans_dual_crf_mrf_supcon_model(T, n_targets):
    """CNN+Trans dual + full-state MRF SC1 (fused SupCon)."""
    inputs       = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z            = _build_cnn_trans_dual_branches_mtl(inputs)
    state_logits = tf.keras.layers.Dense(2 ** n_targets, name='crf_logits')(z)
    proj_norm    = _proj_head(z, 'fused')
    return MultiLabelCRFMRFSC1Model(inputs=inputs, outputs=[state_logits, proj_norm], n_targets=n_targets)


def create_ml_cnn_trans_dual_crf_mrf_supcon2_model(T, n_targets):
    """CNN+Trans dual + full-state MRF SC2 (CNN+seq SupCon)."""
    inputs               = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    cnn_emb, seq_emb, z = _build_cnn_trans_dual_branches_mtl(inputs, return_branches=True)
    state_logits         = tf.keras.layers.Dense(2 ** n_targets, name='crf_logits')(z)
    cnn_proj             = _proj_head(cnn_emb, 'cnn')
    seq_proj             = _proj_head(seq_emb, 'seq')
    return MultiLabelCRFMRFSC2Model(inputs=inputs, outputs=[state_logits, cnn_proj, seq_proj], n_targets=n_targets)


def create_ml_cnn_trans_dual_crf_mrf_supcon3_model(T, n_targets):
    """CNN+Trans dual + full-state MRF SC3 (CNN+seq+fused SupCon)."""
    inputs               = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    cnn_emb, seq_emb, z = _build_cnn_trans_dual_branches_mtl(inputs, return_branches=True)
    state_logits         = tf.keras.layers.Dense(2 ** n_targets, name='crf_logits')(z)
    cnn_proj             = _proj_head(cnn_emb, 'cnn')
    seq_proj             = _proj_head(seq_emb, 'seq')
    fused_proj           = _proj_head(z, 'fused')
    return MultiLabelCRFMRFSC3Model(inputs=inputs, outputs=[state_logits, cnn_proj, seq_proj, fused_proj], n_targets=n_targets)


# ── CRF-chain factories (Option B: linear-chain, Viterbi/forward-backward) ──

def create_ml_cnn_gru_dual_crf_chain_model(T, n_targets):
    """CNN+GRU dual + linear-chain CRF SC0."""
    inputs    = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z         = _build_cnn_gru_dual_branches_mtl(inputs)
    emit_flat = tf.keras.layers.Dense(n_targets * 2, name='crf_emit')(z)
    emit      = tf.keras.layers.Reshape((n_targets, 2), name='crf_emit_r')(emit_flat)
    return MultiLabelCRFChainModel(inputs=inputs, outputs=emit, n_targets=n_targets)


def create_ml_cnn_gru_dual_crf_chain_supcon_model(T, n_targets):
    """CNN+GRU dual + linear-chain CRF SC1 (fused SupCon)."""
    inputs    = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z         = _build_cnn_gru_dual_branches_mtl(inputs)
    emit_flat = tf.keras.layers.Dense(n_targets * 2, name='crf_emit')(z)
    emit      = tf.keras.layers.Reshape((n_targets, 2), name='crf_emit_r')(emit_flat)
    proj_norm = _proj_head(z, 'fused')
    return MultiLabelCRFChainSC1Model(inputs=inputs, outputs=[emit, proj_norm], n_targets=n_targets)


def create_ml_cnn_gru_dual_crf_chain_supcon2_model(T, n_targets):
    """CNN+GRU dual + linear-chain CRF SC2 (CNN+seq SupCon)."""
    inputs               = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    cnn_emb, seq_emb, z = _build_cnn_gru_dual_branches_mtl(inputs, return_branches=True)
    emit_flat            = tf.keras.layers.Dense(n_targets * 2, name='crf_emit')(z)
    emit                 = tf.keras.layers.Reshape((n_targets, 2), name='crf_emit_r')(emit_flat)
    cnn_proj             = _proj_head(cnn_emb, 'cnn')
    seq_proj             = _proj_head(seq_emb, 'seq')
    return MultiLabelCRFChainSC2Model(inputs=inputs, outputs=[emit, cnn_proj, seq_proj], n_targets=n_targets)


def create_ml_cnn_gru_dual_crf_chain_supcon3_model(T, n_targets):
    """CNN+GRU dual + linear-chain CRF SC3 (CNN+seq+fused SupCon)."""
    inputs               = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    cnn_emb, seq_emb, z = _build_cnn_gru_dual_branches_mtl(inputs, return_branches=True)
    emit_flat            = tf.keras.layers.Dense(n_targets * 2, name='crf_emit')(z)
    emit                 = tf.keras.layers.Reshape((n_targets, 2), name='crf_emit_r')(emit_flat)
    cnn_proj             = _proj_head(cnn_emb, 'cnn')
    seq_proj             = _proj_head(seq_emb, 'seq')
    fused_proj           = _proj_head(z, 'fused')
    return MultiLabelCRFChainSC3Model(inputs=inputs, outputs=[emit, cnn_proj, seq_proj, fused_proj], n_targets=n_targets)


def create_ml_cnn_trans_dual_crf_chain_model(T, n_targets):
    """CNN+Trans dual + linear-chain CRF SC0."""
    inputs    = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z         = _build_cnn_trans_dual_branches_mtl(inputs)
    emit_flat = tf.keras.layers.Dense(n_targets * 2, name='crf_emit')(z)
    emit      = tf.keras.layers.Reshape((n_targets, 2), name='crf_emit_r')(emit_flat)
    return MultiLabelCRFChainModel(inputs=inputs, outputs=emit, n_targets=n_targets)


def create_ml_cnn_trans_dual_crf_chain_supcon_model(T, n_targets):
    """CNN+Trans dual + linear-chain CRF SC1 (fused SupCon)."""
    inputs    = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z         = _build_cnn_trans_dual_branches_mtl(inputs)
    emit_flat = tf.keras.layers.Dense(n_targets * 2, name='crf_emit')(z)
    emit      = tf.keras.layers.Reshape((n_targets, 2), name='crf_emit_r')(emit_flat)
    proj_norm = _proj_head(z, 'fused')
    return MultiLabelCRFChainSC1Model(inputs=inputs, outputs=[emit, proj_norm], n_targets=n_targets)


def create_ml_cnn_trans_dual_crf_chain_supcon2_model(T, n_targets):
    """CNN+Trans dual + linear-chain CRF SC2 (CNN+seq SupCon)."""
    inputs               = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    cnn_emb, seq_emb, z = _build_cnn_trans_dual_branches_mtl(inputs, return_branches=True)
    emit_flat            = tf.keras.layers.Dense(n_targets * 2, name='crf_emit')(z)
    emit                 = tf.keras.layers.Reshape((n_targets, 2), name='crf_emit_r')(emit_flat)
    cnn_proj             = _proj_head(cnn_emb, 'cnn')
    seq_proj             = _proj_head(seq_emb, 'seq')
    return MultiLabelCRFChainSC2Model(inputs=inputs, outputs=[emit, cnn_proj, seq_proj], n_targets=n_targets)


def create_ml_cnn_trans_dual_crf_chain_supcon3_model(T, n_targets):
    """CNN+Trans dual + linear-chain CRF SC3 (CNN+seq+fused SupCon)."""
    inputs               = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    cnn_emb, seq_emb, z = _build_cnn_trans_dual_branches_mtl(inputs, return_branches=True)
    emit_flat            = tf.keras.layers.Dense(n_targets * 2, name='crf_emit')(z)
    emit                 = tf.keras.layers.Reshape((n_targets, 2), name='crf_emit_r')(emit_flat)
    cnn_proj             = _proj_head(cnn_emb, 'cnn')
    seq_proj             = _proj_head(seq_emb, 'seq')
    fused_proj           = _proj_head(z, 'fused')
    return MultiLabelCRFChainSC3Model(inputs=inputs, outputs=[emit, cnn_proj, seq_proj, fused_proj], n_targets=n_targets)


# -- source separation factories -----------------------------------------------

def create_ml_cnn_gru_source_sep_model(T, n_targets):
    encoder = build_source_sep_encoder(T, _SS_D_SHARED, _SS_D_TARGET, n_targets)
    cls = build_source_sep_classifier(encoder, _SS_D_SHARED, _SS_D_TARGET, n_targets)
    return _StandardMultiLabelModel(inputs=cls.inputs, outputs=cls.outputs,
                                    name='source_sep_classifier')


def create_ml_cnn_gru_source_sep_supcon_model(T, n_targets):
    encoder = build_source_sep_encoder(T, _SS_D_SHARED, _SS_D_TARGET, n_targets)
    cls = build_source_sep_classifier(encoder, _SS_D_SHARED, _SS_D_TARGET, n_targets)
    return _StandardMultiLabelModel(inputs=cls.inputs, outputs=cls.outputs,
                                    name='source_sep_classifier')


def create_ml_cnn_gru_source_sep_crf_model(T, n_targets):
    """Source sep encoder + full-state MRF (SC0).

    Replaces independent per-target sigmoid heads with a joint 2^n_targets NLL
    loss that captures label co-occurrence structure.  Phase 2+3 still applies:
    Phase 2 freezes the encoder and trains only the CRF logit layer; Phase 3
    unfreezes the encoder for end-to-end fine-tuning.
    """
    encoder      = build_source_sep_encoder(T, _SS_D_SHARED, _SS_D_TARGET, n_targets)
    inputs       = tf.keras.layers.Input(shape=(T, 1), name='cls_input')
    z            = encoder(inputs)
    state_logits = tf.keras.layers.Dense(2 ** n_targets, name='crf_logits')(z)
    return MultiLabelCRFMRFModel(inputs=inputs, outputs=state_logits, n_targets=n_targets)


def create_ml_cnn_gru_source_sep_crf_supcon_model(T, n_targets):
    """Source sep encoder + full-state MRF (SC1 — reserved for SupCon encoder variant)."""
    encoder      = build_source_sep_encoder(T, _SS_D_SHARED, _SS_D_TARGET, n_targets)
    inputs       = tf.keras.layers.Input(shape=(T, 1), name='cls_input')
    z            = encoder(inputs)
    state_logits = tf.keras.layers.Dense(2 ** n_targets, name='crf_logits')(z)
    return MultiLabelCRFMRFModel(inputs=inputs, outputs=state_logits, n_targets=n_targets)


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
    # Dual CNN+GRU cross-attn head: base + SC1/2/3
    'cnn_gru_dual_cross_attn':          create_ml_cnn_gru_dual_cross_attn_model,
    'cnn_gru_dual_cross_attn_supcon':   create_ml_cnn_gru_dual_cross_attn_supcon_model,
    'cnn_gru_dual_cross_attn_supcon2':  create_ml_cnn_gru_dual_cross_attn_supcon2_model,
    'cnn_gru_dual_cross_attn_supcon3':  create_ml_cnn_gru_dual_cross_attn_supcon3_model,
    # Dual CNN+Trans cross-attn head: base + SC1/2/3
    'cnn_trans_dual_cross_attn':          create_ml_cnn_trans_dual_cross_attn_model,
    'cnn_trans_dual_cross_attn_supcon':   create_ml_cnn_trans_dual_cross_attn_supcon_model,
    'cnn_trans_dual_cross_attn_supcon2':  create_ml_cnn_trans_dual_cross_attn_supcon2_model,
    'cnn_trans_dual_cross_attn_supcon3':  create_ml_cnn_trans_dual_cross_attn_supcon3_model,
    # Cross-attn v2 ablation — DeepKV SC0-3
    'cnn_gru_dual_cross_attn_deepkv':          create_ml_cnn_gru_dual_cross_attn_deepkv_model,
    'cnn_gru_dual_cross_attn_deepkv_supcon':   create_ml_cnn_gru_dual_cross_attn_deepkv_supcon_model,
    'cnn_gru_dual_cross_attn_deepkv_supcon2':  create_ml_cnn_gru_dual_cross_attn_deepkv_supcon2_model,
    'cnn_gru_dual_cross_attn_deepkv_supcon3':  create_ml_cnn_gru_dual_cross_attn_deepkv_supcon3_model,
    # Cross-attn v2 ablation — DeepHead SC0-3
    'cnn_gru_dual_cross_attn_deephead':          create_ml_cnn_gru_dual_cross_attn_deephead_model,
    'cnn_gru_dual_cross_attn_deephead_supcon':   create_ml_cnn_gru_dual_cross_attn_deephead_supcon_model,
    'cnn_gru_dual_cross_attn_deephead_supcon2':  create_ml_cnn_gru_dual_cross_attn_deephead_supcon2_model,
    'cnn_gru_dual_cross_attn_deephead_supcon3':  create_ml_cnn_gru_dual_cross_attn_deephead_supcon3_model,
    # Cross-attn v2 — CGD combined SC0-3
    'cnn_gru_dual_cross_attn_v2':          create_ml_cnn_gru_dual_cross_attn_v2_model,
    'cnn_gru_dual_cross_attn_v2_supcon':   create_ml_cnn_gru_dual_cross_attn_v2_supcon_model,
    'cnn_gru_dual_cross_attn_v2_supcon2':  create_ml_cnn_gru_dual_cross_attn_v2_supcon2_model,
    'cnn_gru_dual_cross_attn_v2_supcon3':  create_ml_cnn_gru_dual_cross_attn_v2_supcon3_model,
    # Cross-attn v2 — CTD SC0-3
    'cnn_trans_dual_cross_attn_v2':          create_ml_cnn_trans_dual_cross_attn_v2_model,
    'cnn_trans_dual_cross_attn_v2_supcon':   create_ml_cnn_trans_dual_cross_attn_v2_supcon_model,
    'cnn_trans_dual_cross_attn_v2_supcon2':  create_ml_cnn_trans_dual_cross_attn_v2_supcon2_model,
    'cnn_trans_dual_cross_attn_v2_supcon3':  create_ml_cnn_trans_dual_cross_attn_v2_supcon3_model,
    # Cross-attn v2 AuxDet SC0-3
    'cnn_gru_dual_cross_attn_v2_auxdet':          create_ml_cnn_gru_dual_cross_attn_v2_auxdet_model,
    'cnn_gru_dual_cross_attn_v2_auxdet_supcon':   create_ml_cnn_gru_dual_cross_attn_v2_auxdet_supcon_model,
    'cnn_gru_dual_cross_attn_v2_auxdet_supcon2':  create_ml_cnn_gru_dual_cross_attn_v2_auxdet_supcon2_model,
    'cnn_gru_dual_cross_attn_v2_auxdet_supcon3':  create_ml_cnn_gru_dual_cross_attn_v2_auxdet_supcon3_model,
    # Cross-attn v2 QuerCon (QuerCon + backbone SC0/1/2/3)
    'cnn_gru_dual_cross_attn_v2_quercon':          create_ml_cnn_gru_dual_cross_attn_v2_quercon_model,
    'cnn_gru_dual_cross_attn_v2_quercon_supcon':   create_ml_cnn_gru_dual_cross_attn_v2_quercon_supcon_model,
    'cnn_gru_dual_cross_attn_v2_quercon_supcon2':  create_ml_cnn_gru_dual_cross_attn_v2_quercon_supcon2_model,
    'cnn_gru_dual_cross_attn_v2_quercon_supcon3':  create_ml_cnn_gru_dual_cross_attn_v2_quercon_supcon3_model,
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
    # CRF-MRF (Option A: full-state 8-class NLL) SC0-3
    'cnn_gru_dual_crf_mrf':              create_ml_cnn_gru_dual_crf_mrf_model,
    'cnn_gru_dual_crf_mrf_supcon':       create_ml_cnn_gru_dual_crf_mrf_supcon_model,
    'cnn_gru_dual_crf_mrf_supcon2':      create_ml_cnn_gru_dual_crf_mrf_supcon2_model,
    'cnn_gru_dual_crf_mrf_supcon3':      create_ml_cnn_gru_dual_crf_mrf_supcon3_model,
    'cnn_trans_dual_crf_mrf':            create_ml_cnn_trans_dual_crf_mrf_model,
    'cnn_trans_dual_crf_mrf_supcon':     create_ml_cnn_trans_dual_crf_mrf_supcon_model,
    'cnn_trans_dual_crf_mrf_supcon2':    create_ml_cnn_trans_dual_crf_mrf_supcon2_model,
    'cnn_trans_dual_crf_mrf_supcon3':    create_ml_cnn_trans_dual_crf_mrf_supcon3_model,
    # CRF-chain (Option B: linear-chain with shared (2,2) transition) SC0-3
    'cnn_gru_dual_crf_chain':            create_ml_cnn_gru_dual_crf_chain_model,
    'cnn_gru_dual_crf_chain_supcon':     create_ml_cnn_gru_dual_crf_chain_supcon_model,
    'cnn_gru_dual_crf_chain_supcon2':    create_ml_cnn_gru_dual_crf_chain_supcon2_model,
    'cnn_gru_dual_crf_chain_supcon3':    create_ml_cnn_gru_dual_crf_chain_supcon3_model,
    'cnn_trans_dual_crf_chain':          create_ml_cnn_trans_dual_crf_chain_model,
    'cnn_trans_dual_crf_chain_supcon':   create_ml_cnn_trans_dual_crf_chain_supcon_model,
    'cnn_trans_dual_crf_chain_supcon2':  create_ml_cnn_trans_dual_crf_chain_supcon2_model,
    'cnn_trans_dual_crf_chain_supcon3':  create_ml_cnn_trans_dual_crf_chain_supcon3_model,
    # Source separation pretrained classifier SC0-1
    'cnn_gru_source_sep':               create_ml_cnn_gru_source_sep_model,
    'cnn_gru_source_sep_supcon':        create_ml_cnn_gru_source_sep_supcon_model,
    'cnn_gru_source_sep_crf':           create_ml_cnn_gru_source_sep_crf_model,
    'cnn_gru_source_sep_crf_supcon':    create_ml_cnn_gru_source_sep_crf_supcon_model,
    # CAttn-V2 + CRF-MRF: flat / factored / bilinear × CGD SC0-3
    'cnn_gru_dual_cross_attn_v2_crf_flat':            create_ml_cnn_gru_dual_cross_attn_v2_crf_flat_model,
    'cnn_gru_dual_cross_attn_v2_crf_flat_supcon':     create_ml_cnn_gru_dual_cross_attn_v2_crf_flat_supcon_model,
    'cnn_gru_dual_cross_attn_v2_crf_flat_supcon2':    create_ml_cnn_gru_dual_cross_attn_v2_crf_flat_supcon2_model,
    'cnn_gru_dual_cross_attn_v2_crf_flat_supcon3':    create_ml_cnn_gru_dual_cross_attn_v2_crf_flat_supcon3_model,
    'cnn_gru_dual_cross_attn_v2_crf_factored':        create_ml_cnn_gru_dual_cross_attn_v2_crf_factored_model,
    'cnn_gru_dual_cross_attn_v2_crf_factored_supcon': create_ml_cnn_gru_dual_cross_attn_v2_crf_factored_supcon_model,
    'cnn_gru_dual_cross_attn_v2_crf_factored_supcon2':create_ml_cnn_gru_dual_cross_attn_v2_crf_factored_supcon2_model,
    'cnn_gru_dual_cross_attn_v2_crf_factored_supcon3':create_ml_cnn_gru_dual_cross_attn_v2_crf_factored_supcon3_model,
    'cnn_gru_dual_cross_attn_v2_crf_bilinear':        create_ml_cnn_gru_dual_cross_attn_v2_crf_bilinear_model,
    'cnn_gru_dual_cross_attn_v2_crf_bilinear_supcon': create_ml_cnn_gru_dual_cross_attn_v2_crf_bilinear_supcon_model,
    'cnn_gru_dual_cross_attn_v2_crf_bilinear_supcon2':create_ml_cnn_gru_dual_cross_attn_v2_crf_bilinear_supcon2_model,
    'cnn_gru_dual_cross_attn_v2_crf_bilinear_supcon3':create_ml_cnn_gru_dual_cross_attn_v2_crf_bilinear_supcon3_model,
    # CAttn-V2 + CRF-MRF: flat / factored / bilinear × CTD SC0-3
    'cnn_trans_dual_cross_attn_v2_crf_flat':            create_ml_cnn_trans_dual_cross_attn_v2_crf_flat_model,
    'cnn_trans_dual_cross_attn_v2_crf_flat_supcon':     create_ml_cnn_trans_dual_cross_attn_v2_crf_flat_supcon_model,
    'cnn_trans_dual_cross_attn_v2_crf_flat_supcon2':    create_ml_cnn_trans_dual_cross_attn_v2_crf_flat_supcon2_model,
    'cnn_trans_dual_cross_attn_v2_crf_flat_supcon3':    create_ml_cnn_trans_dual_cross_attn_v2_crf_flat_supcon3_model,
    'cnn_trans_dual_cross_attn_v2_crf_factored':        create_ml_cnn_trans_dual_cross_attn_v2_crf_factored_model,
    'cnn_trans_dual_cross_attn_v2_crf_factored_supcon': create_ml_cnn_trans_dual_cross_attn_v2_crf_factored_supcon_model,
    'cnn_trans_dual_cross_attn_v2_crf_factored_supcon2':create_ml_cnn_trans_dual_cross_attn_v2_crf_factored_supcon2_model,
    'cnn_trans_dual_cross_attn_v2_crf_factored_supcon3':create_ml_cnn_trans_dual_cross_attn_v2_crf_factored_supcon3_model,
    'cnn_trans_dual_cross_attn_v2_crf_bilinear':        create_ml_cnn_trans_dual_cross_attn_v2_crf_bilinear_model,
    'cnn_trans_dual_cross_attn_v2_crf_bilinear_supcon': create_ml_cnn_trans_dual_cross_attn_v2_crf_bilinear_supcon_model,
    'cnn_trans_dual_cross_attn_v2_crf_bilinear_supcon2':create_ml_cnn_trans_dual_cross_attn_v2_crf_bilinear_supcon2_model,
    'cnn_trans_dual_cross_attn_v2_crf_bilinear_supcon3':create_ml_cnn_trans_dual_cross_attn_v2_crf_bilinear_supcon3_model,
}

# Models that carry a concentration regression output
_RCFD_ML_KEYS = frozenset({
    'gru_rcfd_cgd',  'gru_rcfd_cgd_supcon_mtl',  'gru_rcfd_cgd_supcon2_mtl',  'gru_rcfd_cgd_supcon3_mtl',
    'gru_rcfd_ctd',  'gru_rcfd_ctd_supcon_mtl',  'gru_rcfd_ctd_supcon2_mtl',  'gru_rcfd_ctd_supcon3_mtl',
    'trans_rcfd_cgd', 'trans_rcfd_cgd_supcon_mtl', 'trans_rcfd_cgd_supcon2_mtl', 'trans_rcfd_cgd_supcon3_mtl',
    'trans_rcfd_ctd', 'trans_rcfd_ctd_supcon_mtl', 'trans_rcfd_ctd_supcon2_mtl', 'trans_rcfd_ctd_supcon3_mtl',
})

# Source separation models — evaluate loop loads pretrained encoder weights before training
_SS_ML_KEYS = frozenset({
    'cnn_gru_source_sep', 'cnn_gru_source_sep_supcon',
    'cnn_gru_source_sep_crf', 'cnn_gru_source_sep_crf_supcon',
})

# Models using CRF output — evaluate loop calls predict_marginals/predict_binary instead of model.predict
_CRF_ML_KEYS = frozenset({
    'cnn_gru_dual_crf_mrf',   'cnn_gru_dual_crf_mrf_supcon',   'cnn_gru_dual_crf_mrf_supcon2',   'cnn_gru_dual_crf_mrf_supcon3',
    'cnn_trans_dual_crf_mrf', 'cnn_trans_dual_crf_mrf_supcon', 'cnn_trans_dual_crf_mrf_supcon2', 'cnn_trans_dual_crf_mrf_supcon3',
    'cnn_gru_dual_crf_chain',   'cnn_gru_dual_crf_chain_supcon',   'cnn_gru_dual_crf_chain_supcon2',   'cnn_gru_dual_crf_chain_supcon3',
    'cnn_trans_dual_crf_chain', 'cnn_trans_dual_crf_chain_supcon', 'cnn_trans_dual_crf_chain_supcon2', 'cnn_trans_dual_crf_chain_supcon3',
    # Source sep + CRF-MRF: pretrained encoder + joint-state structured output
    'cnn_gru_source_sep_crf', 'cnn_gru_source_sep_crf_supcon',
    # CAttn-V2 + CRF-MRF: flat/factored/bilinear × CGD SC0-3
    'cnn_gru_dual_cross_attn_v2_crf_flat',    'cnn_gru_dual_cross_attn_v2_crf_flat_supcon',    'cnn_gru_dual_cross_attn_v2_crf_flat_supcon2',    'cnn_gru_dual_cross_attn_v2_crf_flat_supcon3',
    'cnn_gru_dual_cross_attn_v2_crf_factored','cnn_gru_dual_cross_attn_v2_crf_factored_supcon','cnn_gru_dual_cross_attn_v2_crf_factored_supcon2','cnn_gru_dual_cross_attn_v2_crf_factored_supcon3',
    'cnn_gru_dual_cross_attn_v2_crf_bilinear','cnn_gru_dual_cross_attn_v2_crf_bilinear_supcon','cnn_gru_dual_cross_attn_v2_crf_bilinear_supcon2','cnn_gru_dual_cross_attn_v2_crf_bilinear_supcon3',
    # CAttn-V2 + CRF-MRF: flat/factored/bilinear × CTD SC0-3
    'cnn_trans_dual_cross_attn_v2_crf_flat',    'cnn_trans_dual_cross_attn_v2_crf_flat_supcon',    'cnn_trans_dual_cross_attn_v2_crf_flat_supcon2',    'cnn_trans_dual_cross_attn_v2_crf_flat_supcon3',
    'cnn_trans_dual_cross_attn_v2_crf_factored','cnn_trans_dual_cross_attn_v2_crf_factored_supcon','cnn_trans_dual_cross_attn_v2_crf_factored_supcon2','cnn_trans_dual_cross_attn_v2_crf_factored_supcon3',
    'cnn_trans_dual_cross_attn_v2_crf_bilinear','cnn_trans_dual_cross_attn_v2_crf_bilinear_supcon','cnn_trans_dual_cross_attn_v2_crf_bilinear_supcon2','cnn_trans_dual_cross_attn_v2_crf_bilinear_supcon3',
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
    # Cross-attn SC1/2/3 (SC0 base is not a SupCon key)
    'cnn_gru_dual_cross_attn_supcon',   'cnn_gru_dual_cross_attn_supcon2',   'cnn_gru_dual_cross_attn_supcon3',
    'cnn_trans_dual_cross_attn_supcon', 'cnn_trans_dual_cross_attn_supcon2', 'cnn_trans_dual_cross_attn_supcon3',
    # Cross-attn v2 ablation SC1/2/3
    'cnn_gru_dual_cross_attn_deepkv_supcon',  'cnn_gru_dual_cross_attn_deepkv_supcon2',  'cnn_gru_dual_cross_attn_deepkv_supcon3',
    'cnn_gru_dual_cross_attn_deephead_supcon','cnn_gru_dual_cross_attn_deephead_supcon2','cnn_gru_dual_cross_attn_deephead_supcon3',
    'cnn_gru_dual_cross_attn_v2_supcon',      'cnn_gru_dual_cross_attn_v2_supcon2',      'cnn_gru_dual_cross_attn_v2_supcon3',
    'cnn_trans_dual_cross_attn_v2_supcon',    'cnn_trans_dual_cross_attn_v2_supcon2',    'cnn_trans_dual_cross_attn_v2_supcon3',
    # AuxDet SC1/2/3
    'cnn_gru_dual_cross_attn_v2_auxdet_supcon', 'cnn_gru_dual_cross_attn_v2_auxdet_supcon2', 'cnn_gru_dual_cross_attn_v2_auxdet_supcon3',
    # QuerCon (all 4 — QuerCon itself is a contrastive loss so all variants are "supcon-like")
    'cnn_gru_dual_cross_attn_v2_quercon', 'cnn_gru_dual_cross_attn_v2_quercon_supcon',
    'cnn_gru_dual_cross_attn_v2_quercon_supcon2', 'cnn_gru_dual_cross_attn_v2_quercon_supcon3',
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
    'cnn_gru_dual_cross_attn':           'CNN+GRU CAttn SC0',
    'cnn_gru_dual_cross_attn_supcon':    'CNN+GRU CAttn SC1',
    'cnn_gru_dual_cross_attn_supcon2':   'CNN+GRU CAttn SC2',
    'cnn_gru_dual_cross_attn_supcon3':   'CNN+GRU CAttn SC3',
    'cnn_trans_dual_cross_attn':          'CNN+Tr CAttn SC0',
    'cnn_trans_dual_cross_attn_supcon':   'CNN+Tr CAttn SC1',
    'cnn_trans_dual_cross_attn_supcon2':  'CNN+Tr CAttn SC2',
    'cnn_trans_dual_cross_attn_supcon3':  'CNN+Tr CAttn SC3',
    # v2 ablation
    'cnn_gru_dual_cross_attn_deepkv':          'CNN+GRU CAttn-DKV SC0',
    'cnn_gru_dual_cross_attn_deepkv_supcon':   'CNN+GRU CAttn-DKV SC1',
    'cnn_gru_dual_cross_attn_deepkv_supcon2':  'CNN+GRU CAttn-DKV SC2',
    'cnn_gru_dual_cross_attn_deepkv_supcon3':  'CNN+GRU CAttn-DKV SC3',
    'cnn_gru_dual_cross_attn_deephead':          'CNN+GRU CAttn-DHD SC0',
    'cnn_gru_dual_cross_attn_deephead_supcon':   'CNN+GRU CAttn-DHD SC1',
    'cnn_gru_dual_cross_attn_deephead_supcon2':  'CNN+GRU CAttn-DHD SC2',
    'cnn_gru_dual_cross_attn_deephead_supcon3':  'CNN+GRU CAttn-DHD SC3',
    'cnn_gru_dual_cross_attn_v2':          'CNN+GRU CAttn-V2 SC0',
    'cnn_gru_dual_cross_attn_v2_supcon':   'CNN+GRU CAttn-V2 SC1',
    'cnn_gru_dual_cross_attn_v2_supcon2':  'CNN+GRU CAttn-V2 SC2',
    'cnn_gru_dual_cross_attn_v2_supcon3':  'CNN+GRU CAttn-V2 SC3',
    'cnn_trans_dual_cross_attn_v2':          'CNN+Tr CAttn-V2 SC0',
    'cnn_trans_dual_cross_attn_v2_supcon':   'CNN+Tr CAttn-V2 SC1',
    'cnn_trans_dual_cross_attn_v2_supcon2':  'CNN+Tr CAttn-V2 SC2',
    'cnn_trans_dual_cross_attn_v2_supcon3':  'CNN+Tr CAttn-V2 SC3',
    # v2 AuxDet
    'cnn_gru_dual_cross_attn_v2_auxdet':          'CNN+GRU CAttn-V2 AuxDet SC0',
    'cnn_gru_dual_cross_attn_v2_auxdet_supcon':   'CNN+GRU CAttn-V2 AuxDet SC1',
    'cnn_gru_dual_cross_attn_v2_auxdet_supcon2':  'CNN+GRU CAttn-V2 AuxDet SC2',
    'cnn_gru_dual_cross_attn_v2_auxdet_supcon3':  'CNN+GRU CAttn-V2 AuxDet SC3',
    # v2 QuerCon
    'cnn_gru_dual_cross_attn_v2_quercon':          'CNN+GRU CAttn-V2 QuerCon',
    'cnn_gru_dual_cross_attn_v2_quercon_supcon':   'CNN+GRU CAttn-V2 QuerCon SC1',
    'cnn_gru_dual_cross_attn_v2_quercon_supcon2':  'CNN+GRU CAttn-V2 QuerCon SC2',
    'cnn_gru_dual_cross_attn_v2_quercon_supcon3':  'CNN+GRU CAttn-V2 QuerCon SC3',
    # CRF-MRF SC0-3
    'cnn_gru_dual_crf_mrf':           'CNN+GRU CRF-MRF SC0',
    'cnn_gru_dual_crf_mrf_supcon':    'CNN+GRU CRF-MRF SC1',
    'cnn_gru_dual_crf_mrf_supcon2':   'CNN+GRU CRF-MRF SC2',
    'cnn_gru_dual_crf_mrf_supcon3':   'CNN+GRU CRF-MRF SC3',
    'cnn_trans_dual_crf_mrf':         'CNN+Tr CRF-MRF SC0',
    'cnn_trans_dual_crf_mrf_supcon':  'CNN+Tr CRF-MRF SC1',
    'cnn_trans_dual_crf_mrf_supcon2': 'CNN+Tr CRF-MRF SC2',
    'cnn_trans_dual_crf_mrf_supcon3': 'CNN+Tr CRF-MRF SC3',
    # CRF-chain SC0-3
    'cnn_gru_dual_crf_chain':           'CNN+GRU CRF-chain SC0',
    'cnn_gru_dual_crf_chain_supcon':    'CNN+GRU CRF-chain SC1',
    'cnn_gru_dual_crf_chain_supcon2':   'CNN+GRU CRF-chain SC2',
    'cnn_gru_dual_crf_chain_supcon3':   'CNN+GRU CRF-chain SC3',
    'cnn_trans_dual_crf_chain':         'CNN+Tr CRF-chain SC0',
    'cnn_trans_dual_crf_chain_supcon':  'CNN+Tr CRF-chain SC1',
    'cnn_trans_dual_crf_chain_supcon2': 'CNN+Tr CRF-chain SC2',
    'cnn_trans_dual_crf_chain_supcon3': 'CNN+Tr CRF-chain SC3',
    # Source separation — independent sigmoid heads (SC0-1)
    'cnn_gru_source_sep':              'SrcSep SC0',
    'cnn_gru_source_sep_supcon':       'SrcSep SC1',
    # Source separation — joint-state CRF-MRF output (SC0-1)
    'cnn_gru_source_sep_crf':          'SrcSep CRF SC0',
    'cnn_gru_source_sep_crf_supcon':   'SrcSep CRF SC1',
    # CAttn-V2 + CRF-MRF flat (CGD SC0-3)
    'cnn_gru_dual_cross_attn_v2_crf_flat':          'CNN+GRU CAttn-V2 CRF-flat SC0',
    'cnn_gru_dual_cross_attn_v2_crf_flat_supcon':   'CNN+GRU CAttn-V2 CRF-flat SC1',
    'cnn_gru_dual_cross_attn_v2_crf_flat_supcon2':  'CNN+GRU CAttn-V2 CRF-flat SC2',
    'cnn_gru_dual_cross_attn_v2_crf_flat_supcon3':  'CNN+GRU CAttn-V2 CRF-flat SC3',
    # CAttn-V2 + CRF-MRF factored (CGD SC0-3)
    'cnn_gru_dual_cross_attn_v2_crf_factored':          'CNN+GRU CAttn-V2 CRF-factored SC0',
    'cnn_gru_dual_cross_attn_v2_crf_factored_supcon':   'CNN+GRU CAttn-V2 CRF-factored SC1',
    'cnn_gru_dual_cross_attn_v2_crf_factored_supcon2':  'CNN+GRU CAttn-V2 CRF-factored SC2',
    'cnn_gru_dual_cross_attn_v2_crf_factored_supcon3':  'CNN+GRU CAttn-V2 CRF-factored SC3',
    # CAttn-V2 + CRF-MRF bilinear (CGD SC0-3)
    'cnn_gru_dual_cross_attn_v2_crf_bilinear':          'CNN+GRU CAttn-V2 CRF-bilinear SC0',
    'cnn_gru_dual_cross_attn_v2_crf_bilinear_supcon':   'CNN+GRU CAttn-V2 CRF-bilinear SC1',
    'cnn_gru_dual_cross_attn_v2_crf_bilinear_supcon2':  'CNN+GRU CAttn-V2 CRF-bilinear SC2',
    'cnn_gru_dual_cross_attn_v2_crf_bilinear_supcon3':  'CNN+GRU CAttn-V2 CRF-bilinear SC3',
    # CAttn-V2 + CRF-MRF flat (CTD SC0-3)
    'cnn_trans_dual_cross_attn_v2_crf_flat':          'CNN+Tr CAttn-V2 CRF-flat SC0',
    'cnn_trans_dual_cross_attn_v2_crf_flat_supcon':   'CNN+Tr CAttn-V2 CRF-flat SC1',
    'cnn_trans_dual_cross_attn_v2_crf_flat_supcon2':  'CNN+Tr CAttn-V2 CRF-flat SC2',
    'cnn_trans_dual_cross_attn_v2_crf_flat_supcon3':  'CNN+Tr CAttn-V2 CRF-flat SC3',
    # CAttn-V2 + CRF-MRF factored (CTD SC0-3)
    'cnn_trans_dual_cross_attn_v2_crf_factored':          'CNN+Tr CAttn-V2 CRF-factored SC0',
    'cnn_trans_dual_cross_attn_v2_crf_factored_supcon':   'CNN+Tr CAttn-V2 CRF-factored SC1',
    'cnn_trans_dual_cross_attn_v2_crf_factored_supcon2':  'CNN+Tr CAttn-V2 CRF-factored SC2',
    'cnn_trans_dual_cross_attn_v2_crf_factored_supcon3':  'CNN+Tr CAttn-V2 CRF-factored SC3',
    # CAttn-V2 + CRF-MRF bilinear (CTD SC0-3)
    'cnn_trans_dual_cross_attn_v2_crf_bilinear':          'CNN+Tr CAttn-V2 CRF-bilinear SC0',
    'cnn_trans_dual_cross_attn_v2_crf_bilinear_supcon':   'CNN+Tr CAttn-V2 CRF-bilinear SC1',
    'cnn_trans_dual_cross_attn_v2_crf_bilinear_supcon2':  'CNN+Tr CAttn-V2 CRF-bilinear SC2',
    'cnn_trans_dual_cross_attn_v2_crf_bilinear_supcon3':  'CNN+Tr CAttn-V2 CRF-bilinear SC3',
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
    encoder_weights_path=None,
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

        res_entry = results_dict.setdefault(f, {})

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
                hls  = [hamming_loss(yt, yp)
                        for yt, yp in zip(res_entry['y_trues_'], res_entry[preds_key])]
                print(f"     [CACHE] {m} | "
                      f"exact_acc={np.mean(accs)*100:.2f}%+-{np.std(accs)*100:.2f}% | "
                      f"hamming={np.mean(hls):.4f}")
                continue

            is_rcfd = m in _RCFD_ML_KEYS
            is_crf  = m in _CRF_ML_KEYS

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
                _is_standard   = isinstance(model, _StandardMultiLabelModel)
                _is_crf_sc0    = is_crf and len(model.outputs) == 1
                # AuxDet SC0 (base class only): test_step returns single {'loss'} metric →
                # Keras 3 val_loss=0.0 bug. SC1/SC2/SC3 return 3 metrics → tracked fine.
                _is_auxdet_sc0 = type(model) is MultiLabelAuxDetModel
                _enc = None
                if (m in _SS_ML_KEYS
                        and encoder_weights_path
                        and os.path.exists(encoder_weights_path)):
                    _enc = model.get_layer('source_sep_encoder')
                    _enc.load_weights(encoder_weights_path)
                    _enc.trainable = False  # Phase 2: frozen encoder
                if _is_standard:
                    model.compile(optimizer=tf.keras.optimizers.Adam(0.001, clipnorm=1.0),
                                  loss='binary_crossentropy')
                else:
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
                    _plain_y = _is_standard or _is_crf_sc0
                    if _has_val:
                        y_tr_d  = yb_tr    if _plain_y else {'cls_out': yb_tr}
                        y_val_d = yb_val   if _plain_y else {'cls_out': yb_val}
                    else:
                        y_tr_d  = yb_train if _plain_y else {'cls_out': yb_train}
                        y_val_d = None

                # Keras 3 val_loss=0.0 bug: test_step returning only {'loss': scalar}
                # is not tracked at epoch level → EarlyStopping fires at epoch=patience
                # with near-init weights. Affected: CRF SC0, AuxDet SC0.
                # Fix: skip val-based callbacks and train on full training fold.
                _skip_val_cbs = _is_crf_sc0 or _is_auxdet_sc0
                _use_val_cbs  = _has_val and not _skip_val_cbs
                cbs = []
                if _use_val_cbs:
                    cbs = [
                        tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss', patience=100, restore_best_weights=True),
                        tf.keras.callbacks.ReduceLROnPlateau(
                            monitor='val_loss', factor=0.5, patience=30, min_lr=1e-5),
                    ]

                fit_kw = dict(epochs=500, batch_size=512, shuffle=True, verbose=0, callbacks=cbs)
                if _use_val_cbs:
                    model.fit(X_tr, y_tr_d, validation_data=(X_val, y_val_d), **fit_kw)
                else:
                    # When _has_val=True, y_tr_d is the split subset (yb_tr); we need
                    # the full-fold y to match X_train. Build it explicitly per model type.
                    if _is_crf_sc0:
                        _full_y = yb_train
                    elif _is_auxdet_sc0:
                        _full_y = {'cls_out': yb_train}
                    else:
                        _full_y = y_tr_d  # already full-fold when _has_val=False
                    model.fit(X_train, _full_y, **fit_kw)

                # Phase 3: unfreeze encoder, fine-tune end-to-end at low LR
                if _enc is not None:
                    _enc.trainable = True
                    if _is_standard:
                        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5, clipnorm=1.0),
                                      loss='binary_crossentropy')
                    else:
                        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5, clipnorm=1.0))
                    p3_use_val = _has_val and not _skip_val_cbs
                    p3_cbs = ([tf.keras.callbacks.EarlyStopping(
                                   monitor='val_loss', patience=30,
                                   restore_best_weights=True)]
                              if p3_use_val else [])
                    p3_kw = dict(epochs=100, batch_size=512, shuffle=True,
                                 verbose=0, callbacks=p3_cbs)
                    if p3_use_val:
                        model.fit(X_tr, y_tr_d,
                                  validation_data=(X_val, y_val_d), **p3_kw)
                    else:
                        if _is_crf_sc0:
                            _p3_full_y = yb_train
                        elif _is_auxdet_sc0:
                            _p3_full_y = {'cls_out': yb_train}
                        else:
                            _p3_full_y = y_tr_d
                        model.fit(X_train, _p3_full_y, **p3_kw)

                if save_model_dir is not None and fold_idx == 0:
                    safe_keras_save(model,
                                    Path(save_model_dir) / f"{m}_{f}_{mode_name}.keras")

                raw_out = None
                if is_crf:
                    cls_prob = model.predict_marginals(X_test)
                    cls_pred = model.predict_binary(X_test).astype(np.int8)
                else:
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
            rmse_str = ""
            if reg_preds_folds and reg_trues_folds:
                rmse_str = _ml_mean_rmse(reg_preds_folds, reg_trues_folds)
            hh, mm, ss = int(duration)//3600, int(duration)%3600//60, int(duration)%60
            print(f"     [+] {mode_name}/{dataset_name}/{filter_name[:25]} | "
                  f"{ml_model_print_map.get(m, m):<20} | "
                  f"exact={np.mean(accs)*100:.2f}% | "
                  f"hamming={np.mean(hls):.4f} | "
                  f"f1_samp={np.mean(f1s):.4f}"
                  + (f" | rmse={rmse_str}" if rmse_str else "")
                  + f" | {hh:02d}:{mm:02d}:{ss:02d}")

            if checkpoint_fn is not None:
                checkpoint_fn(results_dict)   # mirrors main/model_utils.py: pass full dict

        results_dict[f] = res_entry

    return results_dict


# ======================================================================
# CONSOLE SUMMARY (full HTML report is in 06_model_prediction_report.py)
# ======================================================================

def _ml_mean_rmse(reg_preds_folds, reg_trues_folds):
    """Mean RMSE across folds, ignoring REG_SENTINEL entries. Returns '' if no valid data."""
    rmses = []
    for rp, rt in zip(reg_preds_folds, reg_trues_folds):
        rp, rt = np.asarray(rp), np.asarray(rt)
        valid = rt != REG_SENTINEL
        if valid.any():
            rmses.append(np.sqrt(np.mean((rp[valid] - rt[valid]) ** 2)))
    return f"{np.mean(rmses):.4f}" if rmses else ""


def print_ml_results_summary(results_dict, outlier_filters, dataset_name, mode_name,
                              ml_model_key_map, ml_model_print_map):
    """Ranked leaderboard of multi-label results, sorted by exact-match accuracy."""
    rows = []
    for f in outlier_filters:
        res = results_dict.get(f)
        if not res or 'y_trues_' not in res:
            continue
        filter_name = str(f) if f is not None else 'None (Baseline)'
        for m, (preds_key, _, _) in ml_model_key_map.items():
            if preds_key not in res:
                continue
            accs = [accuracy_score(yt, yp)
                    for yt, yp in zip(res['y_trues_'], res[preds_key])]
            hls  = [hamming_loss(yt, yp)
                    for yt, yp in zip(res['y_trues_'], res[preds_key])]
            f1s  = [f1_score(yt, yp, average='samples', zero_division=0)
                    for yt, yp in zip(res['y_trues_'], res[preds_key])]
            rmse_str = _ml_mean_rmse(
                res.get(f'y_reg_preds_{m}_', []),
                res.get(f'y_reg_trues_{m}_', []),
            )
            rows.append((np.mean(accs) * 100, np.std(accs) * 100,
                         np.mean(hls), np.mean(f1s), rmse_str,
                         ml_model_print_map.get(m, m), filter_name))

    rows.sort(key=lambda x: x[0], reverse=True)

    has_rmse = any(r[4] for r in rows)
    W = 120 if has_rmse else 105
    print(f"\n  \U0001f3c6 Leaderboard [{mode_name}] {dataset_name}")
    print("  " + "-" * W)
    for i, (acc, std, hl, f1, rmse_str, method, filt) in enumerate(rows):
        rmse_col = f" | rmse={rmse_str:<10}" if has_rmse else ""
        print(f"  {i+1:2d}.  {acc:6.2f}% ± {std:5.2f}%"
              f" | hamming={hl:.4f} | f1_samp={f1:.4f}"
              + rmse_col
              + f" | Model: {method[:25]:<25} | Filter: {filt[:30]}")
    print("  " + "-" * W + "\n")
