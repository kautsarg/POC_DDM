"""RC-POC (03e): 4-family × 4 SC-level = 16 models.

Research question: Does concentration help classification (RCFD) or vice versa (CCFD-POC)?

1.a cgd_rc_poc       — Pure classification baseline (CGD, no regression)
1.b mtl_cgd_rc_poc   — Joint cls+reg (CGD MTL)
1.c gru_rcfd_cgd_rc_poc  — RCFD: GRU early reg → FiLM → CGD cls (same as model_utils_rcfd)
1.d gru_ccfd_cgd_rc_poc  — CCFD-POC: FROZEN cls (1.a) → FiLM → trainable CGD reg backbone

Training order: Pass 1 trains 1.a/1.b/1.c (saves 1.a .keras); Pass 2 trains 1.d
(loads frozen 1.a .keras per fold). Fully standalone — no changes to any other file.

Usage:
    python 03e_rc_poc_training.py --task_id 0 --exp_folder /path/to/LAB_DDM_paper
"""
import os
import sys
import gc
import argparse
import time
import joblib
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score

sys.path.insert(0, 'utils')
sys.path.insert(0, 'utils/model_training')

from safe_io import safe_joblib_dump, safe_keras_save
from pipeline_utils import get_exp_paths, check_task_id
from model_utils import set_global_determinism
from model_utils_mtl import (
    MTLModel, REG_SENTINEL,
    _normalize_concentration, _inverse_normalize_concentration,
    _build_cnn_gru_dual_branches_mtl,
)
from model_utils_rcfd import _apply_film_scalar, _build_rcfd_backbone
from model_utils_supcon import (
    SupConModel, SupConMTLModel,
    SupConBranch2STModel, SupConBranch2MTLModel,
    SupConBranch3STModel, SupConBranch3MTLModel,
    _proj_head, supcon_loss,
    SUPCON_TEMP,
)
import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


# ======================================================================
# MODEL CLASSES — 16 distinct Keras packages for safe serialization
# ======================================================================

# --- 1.a: CLS-only (CE loss, no regression head) ---

@tf.keras.utils.register_keras_serializable(package='rc_poc_cls')
class RCPocClsModel(tf.keras.Model):
    """Pure classification (CE only). Single output: cls_out."""
    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        y_cls = y_dict['cls_out']
        with tf.GradientTape() as tape:
            cls_out = self(x, training=True)
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(y_cls, cls_out))
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y_cls, cls_out)
        return {m.name: m.result() for m in self.metrics} | {'loss': loss}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        y_cls = y_dict['cls_out']
        cls_out = self(x, training=False)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(y_cls, cls_out))
        self.compiled_metrics.update_state(y_cls, cls_out)
        return {m.name: m.result() for m in self.metrics} | {'loss': loss}


@tf.keras.utils.register_keras_serializable(package='rc_poc_cls_sc1')
class RCPocClsSupConModel(SupConModel):
    """CLS SC1: CE + SupCon on fused. Outputs: [cls_out, proj]."""
    pass


@tf.keras.utils.register_keras_serializable(package='rc_poc_cls_sc2')
class RCPocClsBranch2Model(SupConBranch2STModel):
    """CLS SC2: CE + SupCon on cnn+seq. Outputs: [cls_out, cnn_proj, seq_proj]."""
    pass


@tf.keras.utils.register_keras_serializable(package='rc_poc_cls_sc3')
class RCPocClsBranch3Model(SupConBranch3STModel):
    """CLS SC3: CE + SupCon on cnn+seq+fused. Outputs: [cls_out, cnn_proj, seq_proj, fused_proj]."""
    pass


# --- 1.b: MTL (cls + reg, UW-SO) ---

@tf.keras.utils.register_keras_serializable(package='rc_poc_mtl')
class RCPocMTLModel(MTLModel):
    """MTL SC0. Outputs: [cls_out, reg_out]."""
    pass


@tf.keras.utils.register_keras_serializable(package='rc_poc_mtl_sc1')
class RCPocMTLSupConModel(SupConMTLModel):
    """MTL SC1. Outputs: [cls_out, reg_out, proj]."""
    pass


@tf.keras.utils.register_keras_serializable(package='rc_poc_mtl_sc2')
class RCPocMTLBranch2Model(SupConBranch2MTLModel):
    """MTL SC2. Outputs: [cls_out, reg_out, cnn_proj, seq_proj]."""
    pass


@tf.keras.utils.register_keras_serializable(package='rc_poc_mtl_sc3')
class RCPocMTLBranch3Model(SupConBranch3MTLModel):
    """MTL SC3. Outputs: [cls_out, reg_out, cnn_proj, seq_proj, fused_proj]."""
    pass


# --- 1.c: RCFD (GRU early encoder → FiLM → CGD cls backbone) ---

@tf.keras.utils.register_keras_serializable(package='rc_poc_rcfd')
class RCPocRCFDModel(MTLModel):
    """RCFD SC0. Outputs: [cls_out, reg_out]."""
    pass


@tf.keras.utils.register_keras_serializable(package='rc_poc_rcfd_sc1')
class RCPocRCFDSupConModel(SupConMTLModel):
    """RCFD SC1. Outputs: [cls_out, reg_out, proj]."""
    pass


@tf.keras.utils.register_keras_serializable(package='rc_poc_rcfd_sc2')
class RCPocRCFDBranch2Model(SupConBranch2MTLModel):
    """RCFD SC2. Outputs: [cls_out, reg_out, cnn_proj, seq_proj]."""
    pass


@tf.keras.utils.register_keras_serializable(package='rc_poc_rcfd_sc3')
class RCPocRCFDBranch3Model(SupConBranch3MTLModel):
    """RCFD SC3. Outputs: [cls_out, reg_out, cnn_proj, seq_proj, fused_proj]."""
    pass


# --- 1.d: CCFD-POC (frozen cls backbone → FiLM → trainable CGD reg) ---

@tf.keras.utils.register_keras_serializable(package='rc_poc_ccfd')
class RCPocCCFDModel(tf.keras.Model):
    """CCFD-POC SC0: MSE-only on reg_out. Single output: reg_out."""
    def __init__(self, *args, reg_sentinel=REG_SENTINEL, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_sentinel = reg_sentinel

    def _mse_loss(self, y_reg, reg_out):
        mask = tf.cast(tf.not_equal(y_reg, self.reg_sentinel), tf.float32)
        n_valid = tf.reduce_sum(mask)
        return tf.reduce_sum(mask * tf.square(y_reg - reg_out[:, 0])) / (n_valid + 1e-8)

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            raw = self(x, training=True)
            reg_out = raw[0] if isinstance(raw, (list, tuple)) else raw
            loss = self._mse_loss(y_dict['reg_out'], reg_out)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {'loss': loss}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        raw = self(x, training=False)
        reg_out = raw[0] if isinstance(raw, (list, tuple)) else raw
        loss = self._mse_loss(y_dict['reg_out'], reg_out)
        return {'loss': loss}

    def get_config(self):
        cfg = super().get_config()
        cfg['reg_sentinel'] = self.reg_sentinel
        return cfg

    @classmethod
    def from_config(cls, config):
        reg_sentinel = config.pop('reg_sentinel', REG_SENTINEL)
        m = super().from_config(config)
        m.reg_sentinel = reg_sentinel
        return m


@tf.keras.utils.register_keras_serializable(package='rc_poc_ccfd_sc1')
class RCPocCCFDSupConModel(RCPocCCFDModel):
    """CCFD-POC SC1: MSE + SupCon on z_cond. Outputs: [reg_out, fused_proj]."""
    def __init__(self, *args, supcon_temp=SUPCON_TEMP, supcon_lambda=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.supcon_temp   = supcon_temp
        self.supcon_lambda = supcon_lambda

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            reg_out, proj_norm = self(x, training=True)
            mse  = self._mse_loss(y_dict['reg_out'], reg_out)
            sc   = supcon_loss(proj_norm, y_dict['cls_out'], self.supcon_temp)
            loss = mse + self.supcon_lambda * sc
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {'loss': loss, 'reg_mse': mse, 'supcon': sc}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        reg_out, proj_norm = self(x, training=False)
        mse  = self._mse_loss(y_dict['reg_out'], reg_out)
        sc   = supcon_loss(proj_norm, y_dict['cls_out'], self.supcon_temp)
        loss = mse + self.supcon_lambda * sc
        return {'loss': loss, 'reg_mse': mse, 'supcon': sc}

    def get_config(self):
        cfg = super().get_config()
        cfg['supcon_temp']   = self.supcon_temp
        cfg['supcon_lambda'] = self.supcon_lambda
        return cfg


@tf.keras.utils.register_keras_serializable(package='rc_poc_ccfd_sc2')
class RCPocCCFDBranch2Model(RCPocCCFDModel):
    """CCFD-POC SC2: MSE + SupCon on cnn+seq. Outputs: [reg_out, cnn_proj, seq_proj]."""
    def __init__(self, *args, supcon_temp=SUPCON_TEMP, supcon_lambda_each=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.supcon_temp        = supcon_temp
        self.supcon_lambda_each = supcon_lambda_each

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            reg_out, cnn_proj, seq_proj = self(x, training=True)
            mse  = self._mse_loss(y_dict['reg_out'], reg_out)
            sc   = (supcon_loss(cnn_proj, y_dict['cls_out'], self.supcon_temp)
                    + supcon_loss(seq_proj, y_dict['cls_out'], self.supcon_temp))
            loss = mse + self.supcon_lambda_each * sc
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {'loss': loss, 'reg_mse': mse, 'supcon': sc}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        reg_out, cnn_proj, seq_proj = self(x, training=False)
        mse  = self._mse_loss(y_dict['reg_out'], reg_out)
        sc   = (supcon_loss(cnn_proj, y_dict['cls_out'], self.supcon_temp)
                + supcon_loss(seq_proj, y_dict['cls_out'], self.supcon_temp))
        loss = mse + self.supcon_lambda_each * sc
        return {'loss': loss, 'reg_mse': mse, 'supcon': sc}

    def get_config(self):
        cfg = super().get_config()
        cfg['supcon_temp']        = self.supcon_temp
        cfg['supcon_lambda_each'] = self.supcon_lambda_each
        return cfg


@tf.keras.utils.register_keras_serializable(package='rc_poc_ccfd_sc3')
class RCPocCCFDBranch3Model(RCPocCCFDModel):
    """CCFD-POC SC3: MSE + SupCon on cnn+seq+fused. Outputs: [reg_out, cnn_proj, seq_proj, fused_proj]."""
    def __init__(self, *args, supcon_temp=SUPCON_TEMP, supcon_lambda_each=0.033, **kwargs):
        super().__init__(*args, **kwargs)
        self.supcon_temp        = supcon_temp
        self.supcon_lambda_each = supcon_lambda_each

    def train_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            reg_out, cnn_proj, seq_proj, fused_proj = self(x, training=True)
            mse  = self._mse_loss(y_dict['reg_out'], reg_out)
            sc   = (supcon_loss(cnn_proj,   y_dict['cls_out'], self.supcon_temp)
                    + supcon_loss(seq_proj,   y_dict['cls_out'], self.supcon_temp)
                    + supcon_loss(fused_proj, y_dict['cls_out'], self.supcon_temp))
            loss = mse + self.supcon_lambda_each * sc
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {'loss': loss, 'reg_mse': mse, 'supcon': sc}

    def test_step(self, data):
        x, y_dict, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        reg_out, cnn_proj, seq_proj, fused_proj = self(x, training=False)
        mse  = self._mse_loss(y_dict['reg_out'], reg_out)
        sc   = (supcon_loss(cnn_proj,   y_dict['cls_out'], self.supcon_temp)
                + supcon_loss(seq_proj,   y_dict['cls_out'], self.supcon_temp)
                + supcon_loss(fused_proj, y_dict['cls_out'], self.supcon_temp))
        loss = mse + self.supcon_lambda_each * sc
        return {'loss': loss, 'reg_mse': mse, 'supcon': sc}

    def get_config(self):
        cfg = super().get_config()
        cfg['supcon_temp']        = self.supcon_temp
        cfg['supcon_lambda_each'] = self.supcon_lambda_each
        return cfg


# ======================================================================
# BUILD FUNCTIONS
# ======================================================================

def _build_cgd_branches(inputs):
    """64-dim non-MTL backbone — mirrors model_utils._build_cnn_gru_dual_branches exactly."""
    c = tf.keras.layers.Conv1D(16, 5, activation='relu')(inputs)
    c = tf.keras.layers.Conv1D(8, 3, activation='relu')(c)
    c = tf.keras.layers.Flatten()(c)
    cnn_emb = tf.keras.layers.Dense(32, activation='relu')(c)
    g = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, return_sequences=True))(inputs)
    g = tf.keras.layers.LayerNormalization()(g)
    g = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(16))(g)
    g = tf.keras.layers.Dropout(0.2)(g)
    gru_emb = tf.keras.layers.Dense(32, activation='relu')(g)
    merged = tf.keras.layers.Concatenate()([cnn_emb, gru_emb])
    z = tf.keras.layers.Dense(64, activation='relu')(merged)
    return tf.keras.layers.Dropout(0.2)(z)


# --- 1.a CLS ---

def _build_cls_poc_model(T, n_classes):
    inputs  = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z       = _build_cgd_branches(inputs)
    cls_out = tf.keras.layers.Dense(n_classes, activation='softmax', name='cls_out')(z)
    return RCPocClsModel(inputs=inputs, outputs=cls_out)


def _build_cls_poc_supcon_model(T, n_classes):
    inputs  = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z       = _build_cnn_gru_dual_branches_mtl(inputs)
    cls_out = tf.keras.layers.Dense(
        n_classes, activation='softmax', name='cls_out')(
        tf.keras.layers.Dense(16, activation='relu', name='cls_feat')(z))
    proj_norm = tf.keras.layers.UnitNormalization(axis=1, name='proj')(
        tf.keras.layers.Dense(64, activation='relu', name='proj_hidden')(z))
    return RCPocClsSupConModel(inputs=inputs, outputs=[cls_out, proj_norm])


def _build_cls_poc_supcon2_model(T, n_classes):
    inputs              = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    cnn_emb, gru_emb, z = _build_cnn_gru_dual_branches_mtl(inputs, return_branches=True)
    cls_out  = tf.keras.layers.Dense(
        n_classes, activation='softmax', name='cls_out')(
        tf.keras.layers.Dense(16, activation='relu', name='cls_feat')(z))
    cnn_proj = _proj_head(cnn_emb, 'cnn')
    seq_proj = _proj_head(gru_emb, 'seq')
    return RCPocClsBranch2Model(inputs=inputs, outputs=[cls_out, cnn_proj, seq_proj])


def _build_cls_poc_supcon3_model(T, n_classes):
    inputs               = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    cnn_emb, gru_emb, z  = _build_cnn_gru_dual_branches_mtl(inputs, return_branches=True)
    cls_out    = tf.keras.layers.Dense(
        n_classes, activation='softmax', name='cls_out')(
        tf.keras.layers.Dense(16, activation='relu', name='cls_feat')(z))
    cnn_proj   = _proj_head(cnn_emb, 'cnn')
    seq_proj   = _proj_head(gru_emb, 'seq')
    fused_proj = _proj_head(z, 'fused')
    return RCPocClsBranch3Model(inputs=inputs, outputs=[cls_out, cnn_proj, seq_proj, fused_proj])


# --- 1.b MTL ---

def _build_mtl_poc_model(T, n_classes):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z      = _build_cnn_gru_dual_branches_mtl(inputs)
    cls_out = tf.keras.layers.Dense(
        n_classes, activation='softmax', name='cls_out')(
        tf.keras.layers.Dense(16, activation='relu', name='cls_feat')(z))
    reg_out = tf.keras.layers.Dense(1, activation='linear', name='reg_out')(
        tf.keras.layers.Dense(8, activation='relu', name='reg_hidden')(
        tf.keras.layers.Dense(16, activation='relu', name='reg_feat')(z)))
    return RCPocMTLModel(inputs=inputs, outputs=[cls_out, reg_out])


def _build_mtl_poc_supcon_model(T, n_classes):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z      = _build_cnn_gru_dual_branches_mtl(inputs)
    cls_out = tf.keras.layers.Dense(
        n_classes, activation='softmax', name='cls_out')(
        tf.keras.layers.Dense(16, activation='relu', name='cls_feat')(z))
    reg_out = tf.keras.layers.Dense(1, activation='linear', name='reg_out')(
        tf.keras.layers.Dense(8, activation='relu', name='reg_hidden')(
        tf.keras.layers.Dense(16, activation='relu', name='reg_feat')(z)))
    proj_norm = tf.keras.layers.UnitNormalization(axis=1, name='proj')(
        tf.keras.layers.Dense(64, activation='relu', name='proj_hidden')(z))
    return RCPocMTLSupConModel(inputs=inputs, outputs=[cls_out, reg_out, proj_norm])


def _build_mtl_poc_supcon2_model(T, n_classes):
    inputs               = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    cnn_emb, gru_emb, z  = _build_cnn_gru_dual_branches_mtl(inputs, return_branches=True)
    cls_out  = tf.keras.layers.Dense(
        n_classes, activation='softmax', name='cls_out')(
        tf.keras.layers.Dense(16, activation='relu', name='cls_feat')(z))
    reg_out  = tf.keras.layers.Dense(1, activation='linear', name='reg_out')(
        tf.keras.layers.Dense(8, activation='relu', name='reg_hidden')(
        tf.keras.layers.Dense(16, activation='relu', name='reg_feat')(z)))
    cnn_proj = _proj_head(cnn_emb, 'cnn')
    seq_proj = _proj_head(gru_emb, 'seq')
    return RCPocMTLBranch2Model(inputs=inputs, outputs=[cls_out, reg_out, cnn_proj, seq_proj])


def _build_mtl_poc_supcon3_model(T, n_classes):
    inputs               = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    cnn_emb, gru_emb, z  = _build_cnn_gru_dual_branches_mtl(inputs, return_branches=True)
    cls_out    = tf.keras.layers.Dense(
        n_classes, activation='softmax', name='cls_out')(
        tf.keras.layers.Dense(16, activation='relu', name='cls_feat')(z))
    reg_out    = tf.keras.layers.Dense(1, activation='linear', name='reg_out')(
        tf.keras.layers.Dense(8, activation='relu', name='reg_hidden')(
        tf.keras.layers.Dense(16, activation='relu', name='reg_feat')(z)))
    cnn_proj   = _proj_head(cnn_emb, 'cnn')
    seq_proj   = _proj_head(gru_emb, 'seq')
    fused_proj = _proj_head(z, 'fused')
    return RCPocMTLBranch3Model(inputs=inputs,
                                outputs=[cls_out, reg_out, cnn_proj, seq_proj, fused_proj])


# --- 1.c RCFD (GRU early encoder + CGD backbone) ---

def _build_rcfd_poc_model(T, n_classes):
    inputs         = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z_cond, reg_out = _build_rcfd_backbone(inputs, 'gru', 'cgd')
    cls_out         = tf.keras.layers.Dense(
        n_classes, activation='softmax', name='cls_out')(
        tf.keras.layers.Dense(16, activation='relu', name='cls_feat')(z_cond))
    return RCPocRCFDModel(inputs=inputs, outputs=[cls_out, reg_out])


def _build_rcfd_poc_supcon_model(T, n_classes):
    inputs          = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z_cond, reg_out = _build_rcfd_backbone(inputs, 'gru', 'cgd')
    cls_out         = tf.keras.layers.Dense(
        n_classes, activation='softmax', name='cls_out')(
        tf.keras.layers.Dense(16, activation='relu', name='cls_feat')(z_cond))
    proj_norm       = _proj_head(z_cond, 'fused')
    return RCPocRCFDSupConModel(inputs=inputs, outputs=[cls_out, reg_out, proj_norm])


def _build_rcfd_poc_supcon2_model(T, n_classes):
    inputs                          = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z_cond, reg_out, cnn_emb, seq_emb = _build_rcfd_backbone(
        inputs, 'gru', 'cgd', return_branches=True)
    cls_out  = tf.keras.layers.Dense(
        n_classes, activation='softmax', name='cls_out')(
        tf.keras.layers.Dense(16, activation='relu', name='cls_feat')(z_cond))
    cnn_proj = _proj_head(cnn_emb, 'cnn')
    seq_proj = _proj_head(seq_emb, 'seq')
    return RCPocRCFDBranch2Model(inputs=inputs,
                                 outputs=[cls_out, reg_out, cnn_proj, seq_proj])


def _build_rcfd_poc_supcon3_model(T, n_classes):
    inputs                          = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z_cond, reg_out, cnn_emb, seq_emb = _build_rcfd_backbone(
        inputs, 'gru', 'cgd', return_branches=True)
    cls_out    = tf.keras.layers.Dense(
        n_classes, activation='softmax', name='cls_out')(
        tf.keras.layers.Dense(16, activation='relu', name='cls_feat')(z_cond))
    cnn_proj   = _proj_head(cnn_emb, 'cnn')
    seq_proj   = _proj_head(seq_emb, 'seq')
    fused_proj = _proj_head(z_cond, 'fused')
    return RCPocRCFDBranch3Model(inputs=inputs,
                                 outputs=[cls_out, reg_out, cnn_proj, seq_proj, fused_proj])


# --- 1.d CCFD-POC (frozen cls → FiLM → trainable reg backbone) ---
# Each function takes (T, n_classes, frozen_cls_model) as args.

def _build_ccfd_poc_model(T, n_classes, frozen_cls_model):
    inputs    = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    raw_cls   = frozen_cls_model(inputs, training=False)
    cls_probs = raw_cls[0] if isinstance(raw_cls, (list, tuple)) else raw_cls
    z_raw     = _build_cnn_gru_dual_branches_mtl(inputs)
    z_cond    = tf.keras.layers.Dropout(0.2, name='cfilm_poc_drop')(
        _apply_film_scalar(cls_probs, z_raw, emb_dim=96, name_prefix='cfilm_poc'))
    reg_out   = tf.keras.layers.Dense(1, activation='linear', name='reg_out')(
        tf.keras.layers.Dense(8, activation='relu', name='reg_hidden')(
        tf.keras.layers.Dense(16, activation='relu', name='reg_feat')(z_cond)))
    return RCPocCCFDModel(inputs=inputs, outputs=reg_out)


def _build_ccfd_poc_supcon_model(T, n_classes, frozen_cls_model):
    inputs    = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    raw_cls   = frozen_cls_model(inputs, training=False)
    cls_probs = raw_cls[0] if isinstance(raw_cls, (list, tuple)) else raw_cls
    z_raw     = _build_cnn_gru_dual_branches_mtl(inputs)
    z_cond    = tf.keras.layers.Dropout(0.2, name='cfilm_poc_drop')(
        _apply_film_scalar(cls_probs, z_raw, emb_dim=96, name_prefix='cfilm_poc'))
    reg_out   = tf.keras.layers.Dense(1, activation='linear', name='reg_out')(
        tf.keras.layers.Dense(8, activation='relu', name='reg_hidden')(
        tf.keras.layers.Dense(16, activation='relu', name='reg_feat')(z_cond)))
    fused_proj = _proj_head(z_cond, 'reg_fused')
    return RCPocCCFDSupConModel(inputs=inputs, outputs=[reg_out, fused_proj])


def _build_ccfd_poc_supcon2_model(T, n_classes, frozen_cls_model):
    inputs               = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    raw_cls              = frozen_cls_model(inputs, training=False)
    cls_probs            = raw_cls[0] if isinstance(raw_cls, (list, tuple)) else raw_cls
    cnn_emb, gru_emb, z_raw = _build_cnn_gru_dual_branches_mtl(inputs, return_branches=True)
    z_cond    = tf.keras.layers.Dropout(0.2, name='cfilm_poc_drop')(
        _apply_film_scalar(cls_probs, z_raw, emb_dim=96, name_prefix='cfilm_poc'))
    reg_out   = tf.keras.layers.Dense(1, activation='linear', name='reg_out')(
        tf.keras.layers.Dense(8, activation='relu', name='reg_hidden')(
        tf.keras.layers.Dense(16, activation='relu', name='reg_feat')(z_cond)))
    cnn_proj  = _proj_head(cnn_emb, 'reg_cnn')
    seq_proj  = _proj_head(gru_emb, 'reg_seq')
    return RCPocCCFDBranch2Model(inputs=inputs, outputs=[reg_out, cnn_proj, seq_proj])


def _build_ccfd_poc_supcon3_model(T, n_classes, frozen_cls_model):
    inputs               = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    raw_cls              = frozen_cls_model(inputs, training=False)
    cls_probs            = raw_cls[0] if isinstance(raw_cls, (list, tuple)) else raw_cls
    cnn_emb, gru_emb, z_raw = _build_cnn_gru_dual_branches_mtl(inputs, return_branches=True)
    z_cond     = tf.keras.layers.Dropout(0.2, name='cfilm_poc_drop')(
        _apply_film_scalar(cls_probs, z_raw, emb_dim=96, name_prefix='cfilm_poc'))
    reg_out    = tf.keras.layers.Dense(1, activation='linear', name='reg_out')(
        tf.keras.layers.Dense(8, activation='relu', name='reg_hidden')(
        tf.keras.layers.Dense(16, activation='relu', name='reg_feat')(z_cond)))
    cnn_proj   = _proj_head(cnn_emb, 'reg_cnn')
    seq_proj   = _proj_head(gru_emb, 'reg_seq')
    fused_proj = _proj_head(z_cond, 'reg_fused')
    return RCPocCCFDBranch3Model(inputs=inputs,
                                 outputs=[reg_out, cnn_proj, seq_proj, fused_proj])


# ======================================================================
# KEY LISTS + FACTORY MAPS
# ======================================================================

CLS_KEYS = ['cgd_rc_poc', 'cgd_rc_poc_sc1', 'cgd_rc_poc_sc2', 'cgd_rc_poc_sc3']
MTL_KEYS = ['mtl_cgd_rc_poc', 'mtl_cgd_rc_poc_sc1', 'mtl_cgd_rc_poc_sc2', 'mtl_cgd_rc_poc_sc3']
RCFD_KEYS = [
    'gru_rcfd_cgd_rc_poc', 'gru_rcfd_cgd_rc_poc_sc1',
    'gru_rcfd_cgd_rc_poc_sc2', 'gru_rcfd_cgd_rc_poc_sc3',
]
CCFD_POC_KEYS = [
    'gru_ccfd_cgd_rc_poc', 'gru_ccfd_cgd_rc_poc_sc1',
    'gru_ccfd_cgd_rc_poc_sc2', 'gru_ccfd_cgd_rc_poc_sc3',
]
PASS1_KEYS      = CLS_KEYS + MTL_KEYS + RCFD_KEYS
ALL_RC_POC_KEYS = PASS1_KEYS + CCFD_POC_KEYS  # 16 total

_PASS1_FACTORIES = {
    'cgd_rc_poc':              _build_cls_poc_model,
    'cgd_rc_poc_sc1':          _build_cls_poc_supcon_model,
    'cgd_rc_poc_sc2':          _build_cls_poc_supcon2_model,
    'cgd_rc_poc_sc3':          _build_cls_poc_supcon3_model,
    'mtl_cgd_rc_poc':          _build_mtl_poc_model,
    'mtl_cgd_rc_poc_sc1':      _build_mtl_poc_supcon_model,
    'mtl_cgd_rc_poc_sc2':      _build_mtl_poc_supcon2_model,
    'mtl_cgd_rc_poc_sc3':      _build_mtl_poc_supcon3_model,
    'gru_rcfd_cgd_rc_poc':     _build_rcfd_poc_model,
    'gru_rcfd_cgd_rc_poc_sc1': _build_rcfd_poc_supcon_model,
    'gru_rcfd_cgd_rc_poc_sc2': _build_rcfd_poc_supcon2_model,
    'gru_rcfd_cgd_rc_poc_sc3': _build_rcfd_poc_supcon3_model,
}

_CCFD_POC_BUILD_FNS = {
    'gru_ccfd_cgd_rc_poc':     _build_ccfd_poc_model,
    'gru_ccfd_cgd_rc_poc_sc1': _build_ccfd_poc_supcon_model,
    'gru_ccfd_cgd_rc_poc_sc2': _build_ccfd_poc_supcon2_model,
    'gru_ccfd_cgd_rc_poc_sc3': _build_ccfd_poc_supcon3_model,
}

_CCFD_POC_BACKBONE_KEY = {
    'gru_ccfd_cgd_rc_poc':     'cgd_rc_poc',
    'gru_ccfd_cgd_rc_poc_sc1': 'cgd_rc_poc_sc1',
    'gru_ccfd_cgd_rc_poc_sc2': 'cgd_rc_poc_sc2',
    'gru_ccfd_cgd_rc_poc_sc3': 'cgd_rc_poc_sc3',
}

_CLS_ONLY_KEYS = set(CLS_KEYS)
_BOTH_KEYS     = set(MTL_KEYS + RCFD_KEYS)


# ======================================================================
# KEY / PRINT MAPS
# ======================================================================

def _mk(key):
    return (
        f'y_preds_AC_{key}_', f'y_probs_AC_{key}_', f'classes_AC_{key}_',
        f'y_reg_preds_{key}_', f'y_reg_trues_{key}_',
    )


_RC_POC_KEY_MAP = {m: _mk(m) for m in ALL_RC_POC_KEYS}

_RC_POC_PRINT_MAP = {
    'cgd_rc_poc':               'CGD (RC-POC)',
    'cgd_rc_poc_sc1':           'CGD SC1 (RC-POC)',
    'cgd_rc_poc_sc2':           'CGD SC2 (RC-POC)',
    'cgd_rc_poc_sc3':           'CGD SC3 (RC-POC)',
    'mtl_cgd_rc_poc':           'MTL CGD (RC-POC)',
    'mtl_cgd_rc_poc_sc1':       'MTL CGD SC1 (RC-POC)',
    'mtl_cgd_rc_poc_sc2':       'MTL CGD SC2 (RC-POC)',
    'mtl_cgd_rc_poc_sc3':       'MTL CGD SC3 (RC-POC)',
    'gru_rcfd_cgd_rc_poc':      'RCFD GRU CGD (RC-POC)',
    'gru_rcfd_cgd_rc_poc_sc1':  'RCFD GRU CGD SC1 (RC-POC)',
    'gru_rcfd_cgd_rc_poc_sc2':  'RCFD GRU CGD SC2 (RC-POC)',
    'gru_rcfd_cgd_rc_poc_sc3':  'RCFD GRU CGD SC3 (RC-POC)',
    'gru_ccfd_cgd_rc_poc':      'CCFD-POC GRU CGD',
    'gru_ccfd_cgd_rc_poc_sc1':  'CCFD-POC GRU CGD SC1',
    'gru_ccfd_cgd_rc_poc_sc2':  'CCFD-POC GRU CGD SC2',
    'gru_ccfd_cgd_rc_poc_sc3':  'CCFD-POC GRU CGD SC3',
}

# All result keys (for force-rerun isolation); only rc_poc keys cleared
_ALL_RC_POC_RESULT_KEYS = set()
for _, (_pk, _prk, _ck, _rpk, _rtk) in _RC_POC_KEY_MAP.items():
    _ALL_RC_POC_RESULT_KEYS.update([_pk, _prk, _ck, _rpk, _rtk])


# ======================================================================
# DATA HELPERS
# ======================================================================

def _load_training_data(exp_path):
    data_path = os.path.join(exp_path, config.TRAINING_DATA_PATH)
    if not os.path.exists(data_path):
        print(f'  -> [SKIP] {data_path} not found. Run 01b + 02 first.')
        sys.exit(0)
    return joblib.load(data_path)


def _load_or_init_results(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            print(f'  -> [WARNING] Results file corrupt ({e}), starting fresh.')
    return {}


def _load_concentration(training_data, n_samples):
    raw = training_data.get('concentration', None)
    if raw is not None:
        _arr = np.asarray(raw, dtype=object)
        _float = np.array(
            [float(v) if v is not None else np.nan for v in _arr], dtype=float)
        y_conc = np.where(np.isnan(_float) | (_float == 0.0), REG_SENTINEL, _float)
    else:
        y_conc = np.full(n_samples, REG_SENTINEL, dtype=float)
        print('  [!] No concentration in training data — sentinel FiLM for all samples.')
    n_valid = int((y_conc != REG_SENTINEL).sum())
    print(f'  [03e] Concentration: {n_valid}/{len(y_conc)} valid non-sentinel samples.')
    return y_conc


def _filter_datasets(dataset_name, dataset, kinetic_features):
    out_n, out_d, out_f = [], [], []
    for n, d, f in zip(dataset_name, dataset, kinetic_features):
        if not n.startswith('avg_') and not n.startswith('original_fitted_stretched'):
            out_n.append(n); out_d.append(d); out_f.append(f)
    return out_n, out_d, out_f


def _fmt_hms(s):
    h, r = divmod(int(s), 3600)
    m, s = divmod(r, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def _rmse_str(reg_trues_folds, reg_preds_folds):
    rmses = []
    for rt, rp in zip(reg_trues_folds, reg_preds_folds):
        rt, rp = np.asarray(rt, float), np.asarray(rp, float)
        v = (rt != REG_SENTINEL) & np.isfinite(rt) & np.isfinite(rp)
        if v.sum() >= 2:
            rmses.append(np.sqrt(np.mean((rt[v] - rp[v]) ** 2)))
    return f'rmse={np.mean(rmses):.4f}' if rmses else 'rmse=N/A'


def _make_splits(X_f, y_f, n_splits):
    unique_cls, cls_counts = np.unique(y_f, return_counts=True)
    n_cls = len(unique_cls)
    test_size = max(int(len(y_f) * 0.10), n_cls)
    if n_splits == 1:
        return list(StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=0).split(X_f, y_f))
    actual = min(n_splits, int(np.min(cls_counts)))
    return list(StratifiedKFold(n_splits=actual, shuffle=True, random_state=0).split(X_f, y_f))


def _fold_data(X_f, y_f, conc_f, tr_idx):
    conc_train_scaled, scaler = _normalize_concentration(conc_f[tr_idx])
    try:
        tr_sub, val_sub = train_test_split(
            np.arange(len(tr_idx)), test_size=0.1,
            stratify=y_f[tr_idx], random_state=0)
        X_tr, X_val = X_f[tr_idx][tr_sub], X_f[tr_idx][val_sub]
        y_tr, y_val = y_f[tr_idx][tr_sub], y_f[tr_idx][val_sub]
        c_tr = conc_train_scaled[tr_sub]
        c_val = conc_train_scaled[val_sub]
        val_data = (X_val, {'cls_out': y_val, 'reg_out': c_val})
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=100,
                restore_best_weights=True, verbose=0),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=30, min_lr=1e-5),
        ]
    except ValueError:
        X_tr, y_tr, c_tr = X_f[tr_idx], y_f[tr_idx], conc_train_scaled
        val_data, callbacks = None, []
    fit_y = {'cls_out': y_tr, 'reg_out': c_tr}
    return X_tr, y_tr, c_tr, fit_y, val_data, callbacks, scaler


# ======================================================================
# TRAINING LOOP
# ======================================================================

def run_03e_training(
    X_curves, features_df, y_encoded, y_concentration,
    encoder_classes, n_splits,
    cached_results, force_rerun,
    results_file_path, all_ml_results, clean_title, mode_key,
    model_save_dir, curve_type='ori_curve',
):
    results = cached_results.copy()

    if force_rerun:
        n_cleared = 0
        for re in results.values():
            if isinstance(re, dict):
                for rk in list(re.keys()):
                    if rk in _ALL_RC_POC_RESULT_KEYS:
                        del re[rk]
                        n_cleared += 1
        if n_cleared:
            print(f'  -> [FORCE RERUN] Cleared {n_cleared} rc_poc cached key(s).')

    f = None  # filter: None only (design decision 3.4)
    filter_name = 'None (Baseline)'
    _filter_str  = 'None'
    print(f'\n  -> Filter: {filter_name}')

    mask = np.ones(len(y_encoded), dtype=bool)
    X_f    = np.nan_to_num(X_curves[mask], nan=0.0, posinf=0.0, neginf=0.0)
    y_f    = y_encoded[mask]
    conc_f = y_concentration[mask]

    unique_cls, cls_counts = np.unique(y_f, return_counts=True)
    rare = unique_cls[cls_counts < 2]
    if len(rare) > 0:
        vcm = ~np.isin(y_f, rare)
        X_f, y_f, conc_f = X_f[vcm], y_f[vcm], conc_f[vcm]
        unique_cls, cls_counts = np.unique(y_f, return_counts=True)

    n_cls = len(unique_cls)
    if n_cls < 2 or len(y_f) < 2 * n_cls:
        print('     [Warning] Insufficient classes or samples. Skipping.')
        return results

    splits = _make_splits(X_f, y_f, n_splits)

    res_entry = results.get(f, {})
    current_mask_count = int(np.sum(mask))
    if res_entry.get('mask_count') not in (None, current_mask_count):
        print('     [Warning] Stale cache. Discarding for this filter.')
        res_entry = {}
    if 'y_trues_' not in res_entry:
        res_entry['y_trues_'] = [y_f[te] for _, te in splits]
    res_entry['mask_count']   = current_mask_count
    res_entry['y_true_count'] = len(y_f)

    T = X_f.shape[1]

    # ------------------------------------------------------------------ #
    # Pass 1: CLS / MTL / RCFD
    # ------------------------------------------------------------------ #
    for m in PASS1_KEYS:
        preds_key, probs_key, classes_key, reg_preds_key, reg_trues_key = _RC_POC_KEY_MAP[m]
        has_reg = m in _BOTH_KEYS

        primary_key = preds_key  # all pass-1 models have cls output

        if primary_key in res_entry:
            rstr = _rmse_str(res_entry.get(reg_trues_key, []),
                             res_entry.get(reg_preds_key, [])) if has_reg else 'no reg'
            cached_accs = [accuracy_score(yt, yp)
                           for yt, yp in zip(res_entry['y_trues_'], res_entry[preds_key])]
            print(f'     [CACHE] {_RC_POC_PRINT_MAP[m]:<28} | '
                  f'acc={np.mean(cached_accs)*100:.2f}%±{np.std(cached_accs)*100:.2f}% | '
                  f'{rstr}')
            continue

        preds_folds, probs_folds, classes_folds = [], [], []
        reg_preds_folds, reg_trues_folds = [], []
        t0 = time.perf_counter()

        for fold_idx, (tr_idx, te_idx) in enumerate(splits):
            X_tr, _, _, fit_y, val_data, callbacks, scaler = _fold_data(
                X_f, y_f, conc_f, tr_idx)

            tf.keras.backend.clear_session()
            model = _PASS1_FACTORIES[m](T, n_cls)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
                jit_compile=False)
            model.fit(X_tr, fit_y,
                      validation_data=val_data, callbacks=callbacks,
                      epochs=500, batch_size=512, shuffle=True, verbose=0)

            # save at fold 0; CLS keras is required by CCFD-POC pass 2
            if fold_idx == 0:
                _keras_path = model_save_dir / (
                    f'{m}_{_filter_str}_{curve_type}_{n_splits}fold_model.keras')
                safe_keras_save(model, _keras_path)

            raw_out   = model.predict(X_f[te_idx], verbose=0)
            cls_probs = raw_out[0] if isinstance(raw_out, (list, tuple)) else raw_out
            pred      = np.argmax(cls_probs, axis=1)

            preds_folds.append(pred)
            probs_folds.append(cls_probs)
            classes_folds.append(encoder_classes)

            if has_reg:
                reg_scaled = raw_out[1]
                reg_orig   = _inverse_normalize_concentration(reg_scaled[:, 0], scaler)
                reg_preds_folds.append(reg_orig)
                reg_trues_folds.append(conc_f[te_idx])

            gc.collect()

        duration = time.perf_counter() - t0
        accs = [accuracy_score(yt, yp)
                for yt, yp in zip(res_entry['y_trues_'], preds_folds)]
        rstr = _rmse_str(reg_trues_folds, reg_preds_folds) if has_reg else 'no reg'

        res_entry[preds_key]   = preds_folds
        res_entry[probs_key]   = probs_folds
        res_entry[classes_key] = classes_folds
        if has_reg:
            res_entry[reg_preds_key] = reg_preds_folds
            res_entry[reg_trues_key] = reg_trues_folds

        results[f] = res_entry
        all_ml_results[clean_title][mode_key] = results
        safe_joblib_dump(all_ml_results, results_file_path, compress=3)

        print(f'     [+] {_RC_POC_PRINT_MAP[m]:<28} | '
              f'acc={np.mean(accs)*100:.2f}%±{np.std(accs)*100:.2f}% | '
              f'{rstr} | {_fmt_hms(duration)}')

    # ------------------------------------------------------------------ #
    # Pass 2: CCFD-POC (loads frozen CLS backbone per fold)
    # ------------------------------------------------------------------ #
    for m in CCFD_POC_KEYS:
        preds_key, probs_key, classes_key, reg_preds_key, reg_trues_key = _RC_POC_KEY_MAP[m]

        backbone_key    = _CCFD_POC_BACKBONE_KEY[m]
        cls_keras_path  = model_save_dir / (
            f'{backbone_key}_{_filter_str}_{curve_type}_{n_splits}fold_model.keras')

        if not cls_keras_path.exists():
            print(f'     [SKIP] {_RC_POC_PRINT_MAP[m]}: '
                  f'frozen backbone not found ({cls_keras_path.name})')
            continue

        if reg_preds_key in res_entry:
            rstr = _rmse_str(res_entry[reg_trues_key], res_entry[reg_preds_key])
            print(f'     [CACHE] {_RC_POC_PRINT_MAP[m]:<28} | reg-only | {rstr}')
            continue

        reg_preds_folds, reg_trues_folds = [], []
        t0 = time.perf_counter()

        for fold_idx, (tr_idx, te_idx) in enumerate(splits):
            X_tr, _, _, fit_y, val_data, callbacks, scaler = _fold_data(
                X_f, y_f, conc_f, tr_idx)

            tf.keras.backend.clear_session()
            # reload frozen backbone after clear_session() — it was invalidated
            frozen_cls_model = tf.keras.models.load_model(
                str(cls_keras_path), compile=False)
            frozen_cls_model.trainable = False

            model = _CCFD_POC_BUILD_FNS[m](T, n_cls, frozen_cls_model)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
                jit_compile=False)
            model.fit(X_tr, fit_y,
                      validation_data=val_data, callbacks=callbacks,
                      epochs=500, batch_size=512, shuffle=True, verbose=0)

            if fold_idx == 0:
                _keras_path = model_save_dir / (
                    f'{m}_{_filter_str}_{curve_type}_{n_splits}fold_model.keras')
                safe_keras_save(model, _keras_path)

            raw_out    = model.predict(X_f[te_idx], verbose=0)
            reg_scaled = raw_out[0] if isinstance(raw_out, (list, tuple)) else raw_out
            reg_orig   = _inverse_normalize_concentration(reg_scaled[:, 0], scaler)
            reg_preds_folds.append(reg_orig)
            reg_trues_folds.append(conc_f[te_idx])

            gc.collect()

        duration = time.perf_counter() - t0
        rstr = _rmse_str(reg_trues_folds, reg_preds_folds)

        res_entry[reg_preds_key] = reg_preds_folds
        res_entry[reg_trues_key] = reg_trues_folds

        results[f] = res_entry
        all_ml_results[clean_title][mode_key] = results
        safe_joblib_dump(all_ml_results, results_file_path, compress=3)

        print(f'     [+] {_RC_POC_PRINT_MAP[m]:<28} | reg-only | {rstr} | {_fmt_hms(duration)}')

    results[f] = res_entry
    return results


# ======================================================================
# LEADERBOARD
# ======================================================================

def _print_leaderboard(all_ml_results, clean_title, mode_key):
    mode_res = all_ml_results.get(clean_title, {}).get(mode_key, {})
    rows = []
    for f_key, res in mode_res.items():
        if not isinstance(res, dict) or 'y_trues_' not in res:
            continue
        filter_name = str(f_key) if f_key is not None else 'None (Baseline)'
        for key, val in res.items():
            if not (key.startswith('y_preds_AC_') and 'rc_poc' in key):
                continue
            try:
                accs = [accuracy_score(yt, yp)
                        for yt, yp in zip(res['y_trues_'], val)]
                lbl = _RC_POC_PRINT_MAP.get(
                    key.replace('y_preds_AC_', '').rstrip('_'), key)
                rows.append((np.mean(accs) * 100, np.std(accs) * 100, lbl, filter_name))
            except Exception:
                continue
    rows.sort(key=lambda x: x[0], reverse=True)
    W = 95
    print(f'\n  Leaderboard [RC-POC] {clean_title} / {mode_key}')
    print('  ' + '-' * W)
    for i, (acc, std, lbl, filt) in enumerate(rows):
        print(f'  {i+1:3d}.  {acc:6.2f}% +/- {std:5.2f}%'
              f'  |  {lbl:<36}  |  Filter: {filt}')
    print('  ' + '-' * W + '\n')


# ======================================================================
# MAIN
# ======================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='03e: RC-POC 4-family 16-model training')
    parser.add_argument('--task_id',     type=int, default=0)
    parser.add_argument('--exp_folder',  type=str, default=config.DEFAULT_EXP_FOLDER)
    parser.add_argument('--n_splits',    type=int, default=1)
    parser.add_argument('--mode',        type=str, default='native',
                        choices=['native', 'reference'])
    parser.add_argument('--curve_type',  type=str, default='ori_curve')
    parser.add_argument('--force_rerun', action='store_true')
    args = parser.parse_args()

    set_global_determinism(0)

    exp_paths = get_exp_paths(args.exp_folder)
    check_task_id(args.task_id, exp_paths)
    exp_path  = exp_paths[args.task_id]

    print(f"\n{'='*70}")
    print(f'[RUNNING] 03e_rc_poc_training.py  [RC-POC 16-model | mode={args.mode}]')
    print(f"{'='*70}\n")
    print(f"{'#'*80}\nSTARTING TRAINING FOR: {exp_path.name}\n{'#'*80}")

    rc_poc_dir     = Path(exp_path) / 'rc_poc'
    model_save_dir = rc_poc_dir / 'models'
    model_save_dir.mkdir(parents=True, exist_ok=True)

    joblib_name = (
        'rc_poc_results_5fold.joblib' if args.n_splits > 1
        else 'rc_poc_results.joblib')
    results_file_path = str(rc_poc_dir / joblib_name)

    training_data    = _load_training_data(exp_path)
    dataset_name     = training_data['dataset_name']
    dataset          = training_data['dataset']
    kinetic_features = training_data['kinetic_features']
    Y_well           = training_data['Y_well']

    y_concentration = _load_concentration(training_data, len(Y_well))

    label_mappings = config.get_label_mappings(exp_path)
    if exp_path.name in label_mappings:
        mapping = label_mappings[exp_path.name]
        Y_well  = [mapping.get(w, w) for w in Y_well]

    encoder = LabelEncoder()
    y_full  = encoder.fit_transform(Y_well)

    dataset_name, dataset, kinetic_features = _filter_datasets(
        dataset_name, dataset, kinetic_features)

    target_name    = config.CURVE_TYPE_ALIASES.get(args.curve_type, args.curve_type)
    all_ml_results = _load_or_init_results(results_file_path)

    for name, features_df, curves_2d in zip(dataset_name, kinetic_features, dataset):
        if name != target_name:
            continue

        clean_title = name.replace('_', ' ').title()
        mode_key    = 'Native' if args.mode == 'native' else 'Reference'
        X_curves    = curves_2d if args.mode == 'native' else dataset[0]

        if clean_title not in all_ml_results:
            all_ml_results[clean_title] = {}
        if mode_key not in all_ml_results[clean_title]:
            all_ml_results[clean_title][mode_key] = {}

        results = run_03e_training(
            X_curves        = X_curves,
            features_df     = features_df,
            y_encoded       = y_full,
            y_concentration = y_concentration,
            encoder_classes = encoder.classes_,
            n_splits        = args.n_splits,
            cached_results  = all_ml_results[clean_title][mode_key],
            force_rerun     = args.force_rerun,
            results_file_path = results_file_path,
            all_ml_results    = all_ml_results,
            clean_title       = clean_title,
            mode_key          = mode_key,
            model_save_dir    = model_save_dir,
            curve_type        = args.curve_type,
        )

        all_ml_results[clean_title][mode_key] = results
        safe_joblib_dump(all_ml_results, results_file_path, compress=3)
        print(f'\n  -> Final results saved to {results_file_path}')

        _print_leaderboard(all_ml_results, clean_title, mode_key)

        gc.collect()
