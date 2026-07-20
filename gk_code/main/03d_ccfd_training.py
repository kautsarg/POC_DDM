"""Dual-backbone RCFD + CCFD (standalone, 16 models).

Both new RCFD and new CCFD use two independent full dual backbones of the same type
(no more lightweight early encoder/classifier). The early branch is now a complete
CNN+GRU or CNN+Trans backbone, conditioning the FiLM of the main backbone.

New CCFD (8 models): dual-backbone early cls → stop_gradient → FiLM → dual-backbone reg
New RCFD (8 models): dual-backbone early reg → stop_gradient → FiLM → dual-backbone cls

Keys (no enc_type prefix — its absence signals dual-backbone early branch):
  ccfd_cgd, ccfd_ctd, ccfd_cgd_supcon_mtl, ... (8)
  rcfd_cgd, rcfd_ctd, rcfd_cgd_supcon_mtl, ... (8)

Initial RCFD (24 models, lightweight early encoder) lives in model_utils_rcfd.py unchanged.
Results coexist in classification_performances[_10fold].joblib under distinct keys.

Usage:
    python 03d_ccfd_training.py --task_id 0 --n_splits 5 --mode native
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
    _build_cnn_gru_dual_branches_mtl, _build_cnn_trans_dual_branches_mtl,
)
from model_utils_rcfd import _apply_film_scalar
from model_utils_supcon import (
    SupConMTLModel, SupConBranch2MTLModel, SupConBranch3MTLModel,
    _proj_head,
)
import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


# ======================================================================
# MODEL CLASSES
# ======================================================================

@tf.keras.utils.register_keras_serializable(package='ccfd')
class CCFDModel(MTLModel):
    """New CCFD SC0. Outputs: [cls_early, reg_out]."""
    pass


@tf.keras.utils.register_keras_serializable(package='ccfd_sc1')
class CCFDSupConMTLModel(SupConMTLModel):
    """New CCFD SC1. Outputs: [cls_early, reg_out, proj]."""
    pass


@tf.keras.utils.register_keras_serializable(package='ccfd_sc2')
class CCFDBranch2MTLModel(SupConBranch2MTLModel):
    """New CCFD SC2. Outputs: [cls_early, reg_out, cnn_proj, seq_proj]."""
    pass


@tf.keras.utils.register_keras_serializable(package='ccfd_sc3')
class CCFDBranch3MTLModel(SupConBranch3MTLModel):
    """New CCFD SC3. Outputs: [cls_early, reg_out, cnn_proj, seq_proj, fused_proj]."""
    pass


@tf.keras.utils.register_keras_serializable(package='new_rcfd')
class NewRCFDModel(MTLModel):
    """New RCFD SC0. Outputs: [cls_out, early_reg]."""
    pass


@tf.keras.utils.register_keras_serializable(package='new_rcfd_sc1')
class NewRCFDSupConMTLModel(SupConMTLModel):
    """New RCFD SC1. Outputs: [cls_out, early_reg, proj]."""
    pass


@tf.keras.utils.register_keras_serializable(package='new_rcfd_sc2')
class NewRCFDBranch2MTLModel(SupConBranch2MTLModel):
    """New RCFD SC2. Outputs: [cls_out, early_reg, cnn_proj, seq_proj]."""
    pass


@tf.keras.utils.register_keras_serializable(package='new_rcfd_sc3')
class NewRCFDBranch3MTLModel(SupConBranch3MTLModel):
    """New RCFD SC3. Outputs: [cls_out, early_reg, cnn_proj, seq_proj, fused_proj]."""
    pass


# ======================================================================
# HEAD HELPERS
# ======================================================================

def _ccfd_cls_head(z, n_classes):
    """Early cls head for CCFD (output of early dual backbone)."""
    h = tf.keras.layers.Dense(16, activation='relu', name='ce_feat')(z)
    h = tf.keras.layers.Dense(8,  activation='relu', name='ce_feat2')(h)
    return tf.keras.layers.Dense(n_classes, activation='softmax', name='cls_early')(h)


def _reg_head(z):
    """Main reg head for CCFD (FiLM-conditioned embedding → concentration)."""
    h = tf.keras.layers.Dense(16, activation='relu', name='reg_feat')(z)
    h = tf.keras.layers.Dense(8,  activation='relu', name='reg_feat2')(h)
    return tf.keras.layers.Dense(1, activation='linear', name='reg_out')(h)


def _cls_head(z, n_classes):
    """Main cls head for new RCFD (FiLM-conditioned embedding → class)."""
    h = tf.keras.layers.Dense(16, activation='relu', name='cls_feat')(z)
    h = tf.keras.layers.Dense(8,  activation='relu', name='cls_feat2')(h)
    return tf.keras.layers.Dense(n_classes, activation='softmax', name='cls_out')(h)


def _rcfd_reg_head(z):
    """Early reg head for new RCFD (output of early dual backbone → concentration)."""
    h = tf.keras.layers.Dense(16, activation='relu', name='re2_feat')(z)
    h = tf.keras.layers.Dense(8,  activation='relu', name='re2_feat2')(h)
    return tf.keras.layers.Dense(1, activation='linear', name='early_reg')(h)


# ======================================================================
# DUAL-BACKBONE DISPATCH + BACKBONE BUILDERS
# ======================================================================

def _dual_branch(inputs, bb_type, return_branches=False):
    """Dispatch to cgd or ctd dual backbone; all internal layers are anonymous."""
    if bb_type == 'cgd':
        return _build_cnn_gru_dual_branches_mtl(inputs, return_branches=return_branches)
    return _build_cnn_trans_dual_branches_mtl(inputs, return_branches=return_branches)


def _build_ccfd_backbone(inputs, bb_type, n_classes, return_branches=False):
    """Two independent dual backbones (early cls → FiLM → main reg).
    Calling _dual_branch twice is safe — all layers are anonymous so Keras
    auto-numbers them (conv1d, conv1d_1, ...) without conflicts.
    return_branches returns branches from the MAIN (FiLM-conditioned) backbone only.
    """
    z_early   = _dual_branch(inputs, bb_type)
    cls_early = _ccfd_cls_head(z_early, n_classes)

    if return_branches:
        cnn_m, seq_m, z_main = _dual_branch(inputs, bb_type, return_branches=True)
    else:
        z_main = _dual_branch(inputs, bb_type)

    z_cond = tf.keras.layers.Dropout(0.2, name='cfilm_drop')(
        _apply_film_scalar(cls_early, z_main, emb_dim=96, name_prefix='cfilm'))

    if return_branches:
        return z_cond, cls_early, cnn_m, seq_m
    return z_cond, cls_early


def _build_rcfd2_backbone(inputs, bb_type, return_branches=False):
    """Two independent dual backbones (early reg → FiLM → main cls).
    return_branches returns branches from the MAIN (FiLM-conditioned) backbone only.
    """
    z_early   = _dual_branch(inputs, bb_type)
    early_reg = _rcfd_reg_head(z_early)

    if return_branches:
        cnn_m, seq_m, z_main = _dual_branch(inputs, bb_type, return_branches=True)
    else:
        z_main = _dual_branch(inputs, bb_type)

    z_cond = tf.keras.layers.Dropout(0.2, name='film_drop')(
        _apply_film_scalar(early_reg, z_main, emb_dim=96, name_prefix='film'))

    if return_branches:
        return z_cond, early_reg, cnn_m, seq_m
    return z_cond, early_reg


# ======================================================================
# BUILD FUNCTIONS
# ======================================================================

# --- New CCFD (position 0 = cls_early, position 1 = reg_out)

def _build_ccfd_model(T, n_classes, bb_type):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z_cond, cls_early = _build_ccfd_backbone(inputs, bb_type, n_classes)
    return CCFDModel(inputs=inputs, outputs=[cls_early, _reg_head(z_cond)])


def _build_ccfd_supcon_model(T, n_classes, bb_type):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z_cond, cls_early = _build_ccfd_backbone(inputs, bb_type, n_classes)
    proj_norm = _proj_head(z_cond, 'fused')
    return CCFDSupConMTLModel(inputs=inputs, outputs=[cls_early, _reg_head(z_cond), proj_norm])


def _build_ccfd_supcon2_model(T, n_classes, bb_type):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z_cond, cls_early, cnn_m, seq_m = _build_ccfd_backbone(
        inputs, bb_type, n_classes, return_branches=True)
    cnn_proj = _proj_head(cnn_m, 'cnn')
    seq_proj = _proj_head(seq_m, 'seq')
    return CCFDBranch2MTLModel(inputs=inputs,
                               outputs=[cls_early, _reg_head(z_cond), cnn_proj, seq_proj])


def _build_ccfd_supcon3_model(T, n_classes, bb_type):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z_cond, cls_early, cnn_m, seq_m = _build_ccfd_backbone(
        inputs, bb_type, n_classes, return_branches=True)
    cnn_proj   = _proj_head(cnn_m, 'cnn')
    seq_proj   = _proj_head(seq_m, 'seq')
    fused_proj = _proj_head(z_cond, 'fused')
    return CCFDBranch3MTLModel(inputs=inputs,
                               outputs=[cls_early, _reg_head(z_cond), cnn_proj, seq_proj, fused_proj])


# --- New RCFD (position 0 = cls_out, position 1 = early_reg)

def _build_rcfd2_model(T, n_classes, bb_type):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z_cond, early_reg = _build_rcfd2_backbone(inputs, bb_type)
    return NewRCFDModel(inputs=inputs, outputs=[_cls_head(z_cond, n_classes), early_reg])


def _build_rcfd2_supcon_model(T, n_classes, bb_type):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z_cond, early_reg = _build_rcfd2_backbone(inputs, bb_type)
    cls_out   = _cls_head(z_cond, n_classes)
    proj_norm = _proj_head(z_cond, 'fused')
    return NewRCFDSupConMTLModel(inputs=inputs, outputs=[cls_out, early_reg, proj_norm])


def _build_rcfd2_supcon2_model(T, n_classes, bb_type):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z_cond, early_reg, cnn_m, seq_m = _build_rcfd2_backbone(
        inputs, bb_type, return_branches=True)
    cls_out  = _cls_head(z_cond, n_classes)
    cnn_proj = _proj_head(cnn_m, 'cnn')
    seq_proj = _proj_head(seq_m, 'seq')
    return NewRCFDBranch2MTLModel(inputs=inputs,
                                  outputs=[cls_out, early_reg, cnn_proj, seq_proj])


def _build_rcfd2_supcon3_model(T, n_classes, bb_type):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z_cond, early_reg, cnn_m, seq_m = _build_rcfd2_backbone(
        inputs, bb_type, return_branches=True)
    cls_out    = _cls_head(z_cond, n_classes)
    cnn_proj   = _proj_head(cnn_m, 'cnn')
    seq_proj   = _proj_head(seq_m, 'seq')
    fused_proj = _proj_head(z_cond, 'fused')
    return NewRCFDBranch3MTLModel(inputs=inputs,
                                  outputs=[cls_out, early_reg, cnn_proj, seq_proj, fused_proj])


# ======================================================================
# FACTORY FUNCTIONS (16 total)
# ======================================================================

def create_ccfd_cgd_model(T, n):            return _build_ccfd_model(T, n, 'cgd')
def create_ccfd_ctd_model(T, n):            return _build_ccfd_model(T, n, 'ctd')
def create_ccfd_cgd_supcon_mtl_model(T, n): return _build_ccfd_supcon_model(T, n, 'cgd')
def create_ccfd_ctd_supcon_mtl_model(T, n): return _build_ccfd_supcon_model(T, n, 'ctd')
def create_ccfd_cgd_supcon2_mtl_model(T, n): return _build_ccfd_supcon2_model(T, n, 'cgd')
def create_ccfd_ctd_supcon2_mtl_model(T, n): return _build_ccfd_supcon2_model(T, n, 'ctd')
def create_ccfd_cgd_supcon3_mtl_model(T, n): return _build_ccfd_supcon3_model(T, n, 'cgd')
def create_ccfd_ctd_supcon3_mtl_model(T, n): return _build_ccfd_supcon3_model(T, n, 'ctd')

def create_rcfd_cgd_model(T, n):            return _build_rcfd2_model(T, n, 'cgd')
def create_rcfd_ctd_model(T, n):            return _build_rcfd2_model(T, n, 'ctd')
def create_rcfd_cgd_supcon_mtl_model(T, n): return _build_rcfd2_supcon_model(T, n, 'cgd')
def create_rcfd_ctd_supcon_mtl_model(T, n): return _build_rcfd2_supcon_model(T, n, 'ctd')
def create_rcfd_cgd_supcon2_mtl_model(T, n): return _build_rcfd2_supcon2_model(T, n, 'cgd')
def create_rcfd_ctd_supcon2_mtl_model(T, n): return _build_rcfd2_supcon2_model(T, n, 'ctd')
def create_rcfd_cgd_supcon3_mtl_model(T, n): return _build_rcfd2_supcon3_model(T, n, 'cgd')
def create_rcfd_ctd_supcon3_mtl_model(T, n): return _build_rcfd2_supcon3_model(T, n, 'ctd')


# ======================================================================
# FACTORY DICTS + KEY LISTS
# ======================================================================

_CCFD_BASE_FACTORIES = {
    'ccfd_cgd': create_ccfd_cgd_model,
    'ccfd_ctd': create_ccfd_ctd_model,
}
_CCFD_SC1_FACTORIES = {
    'ccfd_cgd_supcon_mtl': create_ccfd_cgd_supcon_mtl_model,
    'ccfd_ctd_supcon_mtl': create_ccfd_ctd_supcon_mtl_model,
}
_CCFD_SC2_FACTORIES = {
    'ccfd_cgd_supcon2_mtl': create_ccfd_cgd_supcon2_mtl_model,
    'ccfd_ctd_supcon2_mtl': create_ccfd_ctd_supcon2_mtl_model,
}
_CCFD_SC3_FACTORIES = {
    'ccfd_cgd_supcon3_mtl': create_ccfd_cgd_supcon3_mtl_model,
    'ccfd_ctd_supcon3_mtl': create_ccfd_ctd_supcon3_mtl_model,
}
_CCFD_ALL_FACTORIES = {
    **_CCFD_BASE_FACTORIES, **_CCFD_SC1_FACTORIES,
    **_CCFD_SC2_FACTORIES,  **_CCFD_SC3_FACTORIES,
}

_RCFD2_BASE_FACTORIES = {
    'rcfd_cgd': create_rcfd_cgd_model,
    'rcfd_ctd': create_rcfd_ctd_model,
}
_RCFD2_SC1_FACTORIES = {
    'rcfd_cgd_supcon_mtl': create_rcfd_cgd_supcon_mtl_model,
    'rcfd_ctd_supcon_mtl': create_rcfd_ctd_supcon_mtl_model,
}
_RCFD2_SC2_FACTORIES = {
    'rcfd_cgd_supcon2_mtl': create_rcfd_cgd_supcon2_mtl_model,
    'rcfd_ctd_supcon2_mtl': create_rcfd_ctd_supcon2_mtl_model,
}
_RCFD2_SC3_FACTORIES = {
    'rcfd_cgd_supcon3_mtl': create_rcfd_cgd_supcon3_mtl_model,
    'rcfd_ctd_supcon3_mtl': create_rcfd_ctd_supcon3_mtl_model,
}
_RCFD2_ALL_FACTORIES = {
    **_RCFD2_BASE_FACTORIES, **_RCFD2_SC1_FACTORIES,
    **_RCFD2_SC2_FACTORIES,  **_RCFD2_SC3_FACTORIES,
}

CCFD_MODEL_KEYS             = list(_CCFD_BASE_FACTORIES)
CCFD_SUPCON_MTL_MODEL_KEYS  = list(_CCFD_SC1_FACTORIES)
CCFD_BRANCH2_MTL_MODEL_KEYS = list(_CCFD_SC2_FACTORIES)
CCFD_BRANCH3_MTL_MODEL_KEYS = list(_CCFD_SC3_FACTORIES)
ALL_CCFD_KEYS = (CCFD_MODEL_KEYS + CCFD_SUPCON_MTL_MODEL_KEYS
                 + CCFD_BRANCH2_MTL_MODEL_KEYS + CCFD_BRANCH3_MTL_MODEL_KEYS)

RCFD2_MODEL_KEYS             = list(_RCFD2_BASE_FACTORIES)
RCFD2_SUPCON_MTL_MODEL_KEYS  = list(_RCFD2_SC1_FACTORIES)
RCFD2_BRANCH2_MTL_MODEL_KEYS = list(_RCFD2_SC2_FACTORIES)
RCFD2_BRANCH3_MTL_MODEL_KEYS = list(_RCFD2_SC3_FACTORIES)
ALL_RCFD2_KEYS = (RCFD2_MODEL_KEYS + RCFD2_SUPCON_MTL_MODEL_KEYS
                  + RCFD2_BRANCH2_MTL_MODEL_KEYS + RCFD2_BRANCH3_MTL_MODEL_KEYS)

ALL_03D_KEYS = ALL_CCFD_KEYS + ALL_RCFD2_KEYS    # 16 total
_ALL_FACTORIES_03D = {**_CCFD_ALL_FACTORIES, **_RCFD2_ALL_FACTORIES}


# ======================================================================
# LOCAL KEY / PRINT MAPS
# ======================================================================

_CCFD_KEY_MAP = {
    'ccfd_cgd':              ('y_preds_AC_ccfd_cgd_',              'y_probs_AC_ccfd_cgd_',              'classes_AC_ccfd_cgd_'),
    'ccfd_ctd':              ('y_preds_AC_ccfd_ctd_',              'y_probs_AC_ccfd_ctd_',              'classes_AC_ccfd_ctd_'),
    'ccfd_cgd_supcon_mtl':   ('y_preds_AC_ccfd_cgd_supcon_mtl_',   'y_probs_AC_ccfd_cgd_supcon_mtl_',   'classes_AC_ccfd_cgd_supcon_mtl_'),
    'ccfd_ctd_supcon_mtl':   ('y_preds_AC_ccfd_ctd_supcon_mtl_',   'y_probs_AC_ccfd_ctd_supcon_mtl_',   'classes_AC_ccfd_ctd_supcon_mtl_'),
    'ccfd_cgd_supcon2_mtl':  ('y_preds_AC_ccfd_cgd_supcon2_mtl_',  'y_probs_AC_ccfd_cgd_supcon2_mtl_',  'classes_AC_ccfd_cgd_supcon2_mtl_'),
    'ccfd_ctd_supcon2_mtl':  ('y_preds_AC_ccfd_ctd_supcon2_mtl_',  'y_probs_AC_ccfd_ctd_supcon2_mtl_',  'classes_AC_ccfd_ctd_supcon2_mtl_'),
    'ccfd_cgd_supcon3_mtl':  ('y_preds_AC_ccfd_cgd_supcon3_mtl_',  'y_probs_AC_ccfd_cgd_supcon3_mtl_',  'classes_AC_ccfd_cgd_supcon3_mtl_'),
    'ccfd_ctd_supcon3_mtl':  ('y_preds_AC_ccfd_ctd_supcon3_mtl_',  'y_probs_AC_ccfd_ctd_supcon3_mtl_',  'classes_AC_ccfd_ctd_supcon3_mtl_'),
}

_RCFD2_KEY_MAP = {
    'rcfd_cgd':              ('y_preds_AC_rcfd_cgd_',              'y_probs_AC_rcfd_cgd_',              'classes_AC_rcfd_cgd_'),
    'rcfd_ctd':              ('y_preds_AC_rcfd_ctd_',              'y_probs_AC_rcfd_ctd_',              'classes_AC_rcfd_ctd_'),
    'rcfd_cgd_supcon_mtl':   ('y_preds_AC_rcfd_cgd_supcon_mtl_',   'y_probs_AC_rcfd_cgd_supcon_mtl_',   'classes_AC_rcfd_cgd_supcon_mtl_'),
    'rcfd_ctd_supcon_mtl':   ('y_preds_AC_rcfd_ctd_supcon_mtl_',   'y_probs_AC_rcfd_ctd_supcon_mtl_',   'classes_AC_rcfd_ctd_supcon_mtl_'),
    'rcfd_cgd_supcon2_mtl':  ('y_preds_AC_rcfd_cgd_supcon2_mtl_',  'y_probs_AC_rcfd_cgd_supcon2_mtl_',  'classes_AC_rcfd_cgd_supcon2_mtl_'),
    'rcfd_ctd_supcon2_mtl':  ('y_preds_AC_rcfd_ctd_supcon2_mtl_',  'y_probs_AC_rcfd_ctd_supcon2_mtl_',  'classes_AC_rcfd_ctd_supcon2_mtl_'),
    'rcfd_cgd_supcon3_mtl':  ('y_preds_AC_rcfd_cgd_supcon3_mtl_',  'y_probs_AC_rcfd_cgd_supcon3_mtl_',  'classes_AC_rcfd_cgd_supcon3_mtl_'),
    'rcfd_ctd_supcon3_mtl':  ('y_preds_AC_rcfd_ctd_supcon3_mtl_',  'y_probs_AC_rcfd_ctd_supcon3_mtl_',  'classes_AC_rcfd_ctd_supcon3_mtl_'),
}

_CCFD_PRINT_MAP = {
    'ccfd_cgd':             'CCFD CGD',     'ccfd_ctd':             'CCFD CTD',
    'ccfd_cgd_supcon_mtl':  'CCFD CGD SC1', 'ccfd_ctd_supcon_mtl':  'CCFD CTD SC1',
    'ccfd_cgd_supcon2_mtl': 'CCFD CGD SC2', 'ccfd_ctd_supcon2_mtl': 'CCFD CTD SC2',
    'ccfd_cgd_supcon3_mtl': 'CCFD CGD SC3', 'ccfd_ctd_supcon3_mtl': 'CCFD CTD SC3',
}

_RCFD2_PRINT_MAP = {
    'rcfd_cgd':             'New RCFD CGD',     'rcfd_ctd':             'New RCFD CTD',
    'rcfd_cgd_supcon_mtl':  'New RCFD CGD SC1', 'rcfd_ctd_supcon_mtl':  'New RCFD CTD SC1',
    'rcfd_cgd_supcon2_mtl': 'New RCFD CGD SC2', 'rcfd_ctd_supcon2_mtl': 'New RCFD CTD SC2',
    'rcfd_cgd_supcon3_mtl': 'New RCFD CGD SC3', 'rcfd_ctd_supcon3_mtl': 'New RCFD CTD SC3',
}

_ALL_KEY_MAP   = {**_CCFD_KEY_MAP,   **_RCFD2_KEY_MAP}
_ALL_PRINT_MAP = {**_CCFD_PRINT_MAP, **_RCFD2_PRINT_MAP}

# Force-rerun isolation: 16 models × 5 result keys = 80 keys
# All contain 'ccfd_' or 'rcfd_' WITHOUT an enc_type prefix — no overlap with initial RCFD.
_ALL_03D_RESULT_KEYS = set()
for _k, (_pk, _probk, _clsk) in _ALL_KEY_MAP.items():
    _ALL_03D_RESULT_KEYS.update([
        _pk, _probk, _clsk,
        f'y_reg_preds_{_k}_', f'y_reg_trues_{_k}_',
    ])


# ======================================================================
# DATA LOADING HELPERS
# ======================================================================

def _load_training_data(exp_path):
    data_path = os.path.join(exp_path, config.TRAINING_DATA_PATH)
    if not os.path.exists(data_path):
        print(f'  -> [SKIP] {data_path} not found. Run 01b + 02 first.')
        sys.exit(0)
    return joblib.load(data_path)


def _load_or_init_results(results_file_path):
    if os.path.exists(results_file_path):
        try:
            return joblib.load(results_file_path)
        except Exception as e:
            print(f'  -> [WARNING] Results file corrupt ({e}), starting fresh.')
    return {}


def _load_concentration(training_data, n_samples):
    raw_conc = training_data.get('concentration', None)
    if raw_conc is not None:
        _arr = np.asarray(raw_conc, dtype=object)
        _float_arr = np.array(
            [float(v) if v is not None else np.nan for v in _arr], dtype=float)
        y_conc = np.where(
            np.isnan(_float_arr) | (_float_arr == 0.0),
            REG_SENTINEL, _float_arr)
    else:
        y_conc = np.full(n_samples, REG_SENTINEL, dtype=float)
        print('  [!] No concentration in training data — sentinel FiLM for all samples.')
    n_valid = int((y_conc != REG_SENTINEL).sum())
    print(f'  [03d] Concentration: {n_valid}/{len(y_conc)} valid non-sentinel samples.')
    return y_conc


def _apply_conc_scaler(conc_array, scaler):
    arr = conc_array.astype(float).copy()
    valid = arr != REG_SENTINEL
    if valid.any() and hasattr(scaler, 'mean_'):
        arr[valid] = scaler.transform(
            np.log10(arr[valid]).reshape(-1, 1)
        ).ravel()
    return arr


def _filter_datasets(dataset_name, dataset, kinetic_features):
    out_names, out_data, out_feat = [], [], []
    for name, data, feat in zip(dataset_name, dataset, kinetic_features):
        if not name.startswith('avg_') and not name.startswith('original_fitted_stretched'):
            out_names.append(name)
            out_data.append(data)
            out_feat.append(feat)
    return out_names, out_data, out_feat


# ======================================================================
# TRAINING LOOP
# ======================================================================

def _fmt_hms(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def run_03d_training(
    X_curves, features_df, y_encoded, y_concentration,
    outlier_filters, encoder_classes, n_splits,
    cached_results, force_rerun,
    results_file_path, all_ml_results, clean_title, mode_key,
    model_save_dir=None, curve_type='ori_curve',
):
    results = cached_results.copy()

    if force_rerun:
        n_cleared = 0
        for res_entry in results.values():
            if isinstance(res_entry, dict):
                for rk in list(res_entry.keys()):
                    if rk in _ALL_03D_RESULT_KEYS:
                        del res_entry[rk]
                        n_cleared += 1
        if n_cleared:
            print(f'  -> [FORCE RERUN] Cleared {n_cleared} 03d cached key(s). '
                  f'Initial RCFD/standard results preserved.')

    for f in outlier_filters:
        filter_name = f if f else 'None (Baseline)'
        print(f'\n  -> Filter: {filter_name}')

        if f is None:
            mask = np.ones(len(y_encoded), dtype=bool)
        elif f in features_df.columns:
            mask = (features_df[f] == 1).fillna(False).values
        else:
            print(f'     [Warning] Filter column "{f}" not found. Skipping.')
            continue

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
            print(f'     [Warning] Insufficient classes or samples. Skipping.')
            continue

        test_size = max(int(len(y_f) * 0.10), n_cls)
        if n_splits == 1:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
        else:
            actual_splits = min(n_splits, int(np.min(cls_counts)))
            splitter = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=0)
        splits = list(splitter.split(X_f, y_f))

        res_entry = results.get(f, {})
        current_mask_count = int(np.sum(mask))
        if res_entry.get('mask_count') not in (None, current_mask_count):
            print(f'     [Warning] Stale cache (mask_count mismatch). Discarding for this filter.')
            res_entry = {}
        if 'y_trues_' not in res_entry:
            res_entry['y_trues_'] = [y_f[te] for _, te in splits]
        res_entry['mask_count']   = current_mask_count
        res_entry['y_true_count'] = len(y_f)

        T = X_f.shape[1]

        for m in ALL_03D_KEYS:
            preds_key, probs_key, classes_key = _ALL_KEY_MAP[m]
            reg_preds_key = f'y_reg_preds_{m}_'
            reg_trues_key = f'y_reg_trues_{m}_'

            if preds_key in res_entry:
                cached_accs = [accuracy_score(yt, yp)
                               for yt, yp in zip(res_entry['y_trues_'], res_entry[preds_key])]
                _rmses = []
                if reg_preds_key in res_entry and reg_trues_key in res_entry:
                    for _rt, _rp in zip(res_entry[reg_trues_key], res_entry[reg_preds_key]):
                        _rt, _rp = np.asarray(_rt, dtype=float), np.asarray(_rp, dtype=float)
                        _v = (_rt != REG_SENTINEL) & np.isfinite(_rt) & np.isfinite(_rp)
                        if _v.sum() >= 2:
                            _rmses.append(np.sqrt(np.mean((_rt[_v] - _rp[_v]) ** 2)))
                _rmse_str = f'rmse={np.mean(_rmses):.4f}' if _rmses else 'rmse=N/A'
                print(f'     [CACHE] {_ALL_PRINT_MAP[m]:<22} | '
                      f'acc={np.mean(cached_accs)*100:.2f}%±{np.std(cached_accs)*100:.2f}% | '
                      f'{_rmse_str}')
                continue

            preds_folds, probs_folds, classes_folds = [], [], []
            reg_preds_folds, reg_trues_folds = [], []
            t0 = time.perf_counter()

            for fold_idx, (tr_idx, te_idx) in enumerate(splits):
                conc_train_scaled, _scaler = _normalize_concentration(conc_f[tr_idx])

                try:
                    tr_sub, val_sub = train_test_split(
                        np.arange(len(tr_idx)),
                        test_size=0.1, stratify=y_f[tr_idx], random_state=0)
                    X_tr = X_f[tr_idx][tr_sub];  X_val = X_f[tr_idx][val_sub]
                    y_tr = y_f[tr_idx][tr_sub];  y_val = y_f[tr_idx][val_sub]
                    c_tr = conc_train_scaled[tr_sub]; c_val = conc_train_scaled[val_sub]
                    fit_y    = {'cls_out': y_tr,  'reg_out': c_tr}
                    val_y    = {'cls_out': y_val, 'reg_out': c_val}
                    val_data = (X_val, val_y)
                    es = tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss', patience=10,
                        restore_best_weights=True, verbose=0)
                    callbacks = [es]
                except ValueError:
                    X_tr, y_tr, c_tr = X_f[tr_idx], y_f[tr_idx], conc_train_scaled
                    fit_y    = {'cls_out': y_tr, 'reg_out': c_tr}
                    val_data = None
                    callbacks = []

                tf.keras.backend.clear_session()
                model = _ALL_FACTORIES_03D[m](T, n_cls)
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
                    jit_compile=False)
                model.fit(X_tr, fit_y,
                          validation_data=val_data,
                          callbacks=callbacks,
                          epochs=500, batch_size=512, shuffle=True, verbose=0)

                if fold_idx == 0 and model_save_dir is not None:
                    _filter_str = str(f) if f is not None else 'None'
                    _keras_path = (
                        model_save_dir / f'{m}_{_filter_str}_{curve_type}_model.keras')
                    safe_keras_save(model, _keras_path)

                # raw_out[0] = cls output (position 0), raw_out[1] = reg output (position 1)
                # This holds for both CCFD (cls_early, reg_out) and new RCFD (cls_out, early_reg).
                raw_out    = model.predict(X_f[te_idx], verbose=0)
                cls_probs  = raw_out[0]
                reg_scaled = raw_out[1]
                pred       = np.argmax(cls_probs, axis=1)
                reg_orig   = _inverse_normalize_concentration(reg_scaled[:, 0], _scaler)

                preds_folds.append(pred)
                probs_folds.append(cls_probs)
                classes_folds.append(encoder_classes)
                reg_preds_folds.append(reg_orig)
                reg_trues_folds.append(conc_f[te_idx])

                gc.collect()

            duration = time.perf_counter() - t0
            accs = [accuracy_score(yt, yp)
                    for yt, yp in zip(res_entry['y_trues_'], preds_folds)]

            _rmses = []
            for _rt, _rp in zip(reg_trues_folds, reg_preds_folds):
                _rt, _rp = np.asarray(_rt, dtype=float), np.asarray(_rp, dtype=float)
                _v = (_rt != REG_SENTINEL) & np.isfinite(_rt) & np.isfinite(_rp)
                if _v.sum() >= 2:
                    _rmses.append(np.sqrt(np.mean((_rt[_v] - _rp[_v]) ** 2)))
            _rmse_str = f'rmse={np.mean(_rmses):.4f}' if _rmses else 'rmse=N/A'

            res_entry[preds_key]     = preds_folds
            res_entry[probs_key]     = probs_folds
            res_entry[classes_key]   = classes_folds
            res_entry[reg_preds_key] = reg_preds_folds
            res_entry[reg_trues_key] = reg_trues_folds

            results[f] = res_entry
            all_ml_results[clean_title][mode_key] = results
            safe_joblib_dump(all_ml_results, results_file_path, compress=3)

            print(f'     [+] {_ALL_PRINT_MAP[m]:<22} | '
                  f'acc={np.mean(accs)*100:.2f}%±{np.std(accs)*100:.2f}% | '
                  f'{_rmse_str} | {_fmt_hms(duration)}')

        results[f] = res_entry

    return results


# ======================================================================
# LEADERBOARD
# ======================================================================

def _print_leaderboard(all_ml_results, clean_title, mode_key, outlier_filters):
    mode_res = all_ml_results.get(clean_title, {}).get(mode_key, {})

    # Build label map from known models (03d + config standard models)
    _label_map = {_ALL_KEY_MAP[m][0]: _ALL_PRINT_MAP[m] for m in ALL_03D_KEYS}
    try:
        for mk, (pk, _, _) in config.MODEL_KEY_MAP.items():
            _label_map.setdefault(pk, config.MODEL_PRINT_MAP.get(mk, mk))
    except (AttributeError, TypeError, ValueError):
        pass

    rows = []
    for f in outlier_filters:
        res = mode_res.get(f)
        if not isinstance(res, dict) or 'y_trues_' not in res:
            continue
        filter_name = str(f) if f is not None else 'None (Baseline)'
        for key, val in res.items():
            if not key.startswith('y_preds_AC_'):
                continue
            try:
                accs = [accuracy_score(yt, yp)
                        for yt, yp in zip(res['y_trues_'], val)]
                lbl = _label_map.get(key, key.replace('y_preds_AC_', '').rstrip('_'))
                rows.append((np.mean(accs) * 100, np.std(accs) * 100, lbl, filter_name))
            except Exception:
                continue

    rows.sort(key=lambda x: x[0], reverse=True)
    W = 95
    print(f'\n  Leaderboard [ALL results] {clean_title} / {mode_key}')
    print('  ' + '-' * W)
    for i, (acc, std, lbl, filt) in enumerate(rows):
        print(f'  {i+1:3d}.  {acc:6.2f}% +/- {std:5.2f}%'
              f'  |  {lbl:<32}  |  Filter: {filt}')
    print('  ' + '-' * W + '\n')


# ======================================================================
# MAIN
# ======================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='03d: Dual-backbone RCFD + CCFD (16 models)')
    parser.add_argument('--task_id',     type=int, default=0)
    parser.add_argument('--exp_folder',  type=str, default=config.DEFAULT_EXP_FOLDER)
    parser.add_argument('--n_splits',    type=int, default=1)
    parser.add_argument('--mode',        type=str, default='native',
                        choices=['native', 'reference'])
    parser.add_argument('--curve_type',  type=str, default='ori_curve')
    parser.add_argument('--force_rerun', action='store_true',
                        help='Clear only 03d keys and retrain. All other results untouched.')
    args = parser.parse_args()

    set_global_determinism(0)

    exp_paths = get_exp_paths(args.exp_folder)
    check_task_id(args.task_id, exp_paths)
    exp_path  = exp_paths[args.task_id]

    print(f"\n{'='*70}")
    print(f'[RUNNING] 03d_ccfd_training.py  [dual-bb RCFD+CCFD 16-model | mode={args.mode}]')
    print(f"{'='*70}\n")
    print(f"{'#'*80}\nSTARTING TRAINING FOR: {exp_path.name}\n{'#'*80}")

    results_file_path = os.path.join(
        exp_path,
        config.TRAINING_10FOLD_RESULT_PATH if args.n_splits > 1
        else config.TRAINING_RESULT_PATH,
    )

    training_data = _load_training_data(exp_path)

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

    target_name     = config.CURVE_TYPE_ALIASES.get(args.curve_type, args.curve_type)
    outlier_filters = [None]

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

        _model_save_dir = Path(exp_path) / 'model_interpretation'
        _model_save_dir.mkdir(parents=True, exist_ok=True)

        results = run_03d_training(
            X_curves        = X_curves,
            features_df     = features_df,
            y_encoded       = y_full,
            y_concentration = y_concentration,
            outlier_filters = outlier_filters,
            encoder_classes = encoder.classes_,
            n_splits        = args.n_splits,
            cached_results  = all_ml_results[clean_title][mode_key],
            force_rerun     = args.force_rerun,
            results_file_path  = results_file_path,
            all_ml_results     = all_ml_results,
            clean_title        = clean_title,
            mode_key           = mode_key,
            model_save_dir     = _model_save_dir,
            curve_type         = args.curve_type,
        )

        all_ml_results[clean_title][mode_key] = results
        safe_joblib_dump(all_ml_results, results_file_path, compress=3)
        print(f'\n  -> Final results saved to {results_file_path}')

        _print_leaderboard(all_ml_results, clean_title, mode_key, outlier_filters)

        gc.collect()
