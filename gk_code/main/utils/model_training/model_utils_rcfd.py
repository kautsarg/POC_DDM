import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from model_utils_mtl import (
    MTLModel, REG_SENTINEL,
    _build_cnn_gru_dual_branches_mtl, _build_cnn_trans_dual_branches_mtl,
    _build_cnn_gru_dual_attn_recon_embedding_mtl,
)
from model_utils_supcon import (
    SupConMTLModel, SupConBranch2MTLModel, SupConBranch3MTLModel,
    _proj_head, SUPCON_TEMP,
)


@tf.keras.utils.register_keras_serializable(package='rcfd')
class _StopGradient(tf.keras.layers.Layer):
    """Breaks the gradient tape: cls loss cannot update the early encoder via FiLM."""
    def call(self, x):
        return tf.stop_gradient(x)
    def compute_output_shape(self, input_shape):
        return input_shape


# ======================================================================
# MODEL CLASSES — pure subclasses for Keras serialization
# ======================================================================

@tf.keras.utils.register_keras_serializable(package='rcfd')
class RCFDModel(MTLModel):
    """Base RCFD: UW-SO(CE+MSE). Outputs: [cls_out, reg_out]."""
    pass


@tf.keras.utils.register_keras_serializable(package='rcfd_sc1')
class RCFDSupConMTLModel(SupConMTLModel):
    """RCFD + SupCon v1: UW-SO + supcon on z_cond. Outputs: [cls_out, reg_out, proj]."""
    pass


@tf.keras.utils.register_keras_serializable(package='rcfd_sc2')
class RCFDBranch2MTLModel(SupConBranch2MTLModel):
    """RCFD + Branch SupCon v2: supcon on CNN+seq branches. Outputs: [cls_out, reg_out, cnn_proj, seq_proj]."""
    pass


@tf.keras.utils.register_keras_serializable(package='rcfd_sc3')
class RCFDBranch3MTLModel(SupConBranch3MTLModel):
    """RCFD + Branch SupCon v3: supcon on branches+z_cond. Outputs: [cls_out, reg_out, cnn_proj, seq_proj, fused_proj]."""
    pass


# ======================================================================
# BUILDER HELPERS
# ======================================================================

def _build_early_encoder(inputs, enc_type):
    """Lightweight early regression encoder; returns (reg_emb, reg_out).
    reg_emb (batch, 32) exposed for future SupCon projection heads.
    """
    if enc_type == 'cnn':
        x = tf.keras.layers.Conv1D(16, 5, activation='relu', name='re_conv1')(inputs)
        x = tf.keras.layers.Conv1D(8,  3, activation='relu', name='re_conv2')(x)
        x = tf.keras.layers.GlobalAveragePooling1D(name='re_gap')(x)
    elif enc_type == 'gru':
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(16, return_sequences=False), name='re_bigru')(inputs)
    elif enc_type == 'transformer':
        x = tf.keras.layers.Conv1D(32, 5, strides=2, padding='same',
                                   activation='relu', name='re_tconv')(inputs)
        x = tf.keras.layers.MaxPooling1D(2, padding='same', name='re_tpool')(x)
        attn_out = tf.keras.layers.MultiHeadAttention(
            num_heads=2, key_dim=16, name='re_mha')(x, x)
        x = tf.keras.layers.LayerNormalization(name='re_ln1')(x + attn_out)
        ffn = tf.keras.layers.Dense(32, activation='relu', name='re_ffn')(x)
        x = tf.keras.layers.LayerNormalization(name='re_ln2')(x + ffn)
        x = tf.keras.layers.GlobalAveragePooling1D(name='re_gap')(x)
    else:
        raise ValueError(f"Unknown enc_type: {enc_type!r}")

    reg_emb = tf.keras.layers.Dense(32, activation='relu', name='re_emb')(x)
    h = tf.keras.layers.Dense(16, activation='relu', name='re_h1')(reg_emb)
    h = tf.keras.layers.Dense(8,  activation='relu', name='re_h2')(h)
    reg_out = tf.keras.layers.Dense(1, activation='linear', name='reg_out')(h)
    return reg_emb, reg_out


def _apply_film_scalar(c_pred, z_raw, emb_dim=96, name_prefix='film'):
    """Scalar-to-feature FiLM with three hardening measures:
    1. stop_gradient: cls loss cannot reshape the early encoder through FiLM.
    2. BatchNorm on c_pred: clips wild early-training activations; sentinel
       samples (near-zero after BN) produce near-identity FiLM.
    3. Identity init: gamma starts at 1, beta at 0 — epoch-0 is exact identity.
    """
    c_sg  = _StopGradient(name=f'{name_prefix}_sg')(c_pred)
    c_bn  = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_c_bn')(c_sg)
    gamma = tf.keras.layers.Dense(
        emb_dim, kernel_initializer='zeros', bias_initializer='ones',
        name=f'{name_prefix}_gamma')(c_bn)
    beta  = tf.keras.layers.Dense(
        emb_dim, kernel_initializer='zeros', bias_initializer='zeros',
        name=f'{name_prefix}_beta')(c_bn)
    scaled = tf.keras.layers.Multiply(name=f'{name_prefix}_scale')([gamma, z_raw])
    return tf.keras.layers.Add(name=f'{name_prefix}_cond')([scaled, beta])


def _build_rcfd_backbone(inputs, enc_type, dual_type, return_branches=False):
    """Shared RCFD backbone: early encoder → dual backbone → FiLM conditioning.
    Returns (z_cond, reg_out) or (z_cond, reg_out, cnn_emb, seq_emb).
    """
    _, reg_out = _build_early_encoder(inputs, enc_type)
    if dual_type == 'cgd':
        if return_branches:
            cnn_emb, seq_emb, z_raw = _build_cnn_gru_dual_branches_mtl(inputs, return_branches=True)
        else:
            z_raw = _build_cnn_gru_dual_branches_mtl(inputs)
    else:  # ctd
        if return_branches:
            cnn_emb, seq_emb, z_raw = _build_cnn_trans_dual_branches_mtl(inputs, return_branches=True)
        else:
            z_raw = _build_cnn_trans_dual_branches_mtl(inputs)
    z_cond = tf.keras.layers.Dropout(0.2, name='film_drop')(
        _apply_film_scalar(reg_out, z_raw, emb_dim=96))
    if return_branches:
        return z_cond, reg_out, cnn_emb, seq_emb
    return z_cond, reg_out


def _build_rcfd_attn_recon_backbone(stack_input, T, enc_type, attn_dim=16, return_branches=False):
    """RCFD backbone for the attn_recon architecture.

    The early (regression) encoder has no use for neighbour curves -- it only
    predicts this sample's own concentration, so it runs on just the center/
    query curve (stack_input[:, 0, :]), reused unmodified from _build_early_encoder.
    The classification backbone uses the full neighbour stack via the existing
    attention-reconstruction embedding, then FiLM-conditions it exactly like
    every other RCFD variant.
    Returns (z_cond, reg_out) or (z_cond, reg_out, cnn_emb, seq_emb).
    """
    center_curve = tf.keras.layers.Lambda(
        lambda x: x[:, 0, :], name='center_curve')(stack_input)
    center_curve = tf.keras.layers.Reshape((T, 1))(center_curve)
    _, reg_out = _build_early_encoder(center_curve, enc_type)

    if return_branches:
        cnn_emb, seq_emb, z_raw = _build_cnn_gru_dual_attn_recon_embedding_mtl(
            stack_input, T, attn_dim, return_branches=True)
    else:
        z_raw = _build_cnn_gru_dual_attn_recon_embedding_mtl(stack_input, T, attn_dim)

    z_cond = tf.keras.layers.Dropout(0.2, name='film_drop')(
        _apply_film_scalar(reg_out, z_raw, emb_dim=96))
    if return_branches:
        return z_cond, reg_out, cnn_emb, seq_emb
    return z_cond, reg_out


def _cls_head(z_cond, n_classes):
    h = tf.keras.layers.Dense(16, activation='relu', name='cls_feat')(z_cond)
    return tf.keras.layers.Dense(n_classes, activation='softmax', name='cls_out')(h)


# ======================================================================
# BUILD FUNCTIONS (one per variant)
# ======================================================================

def _build_rcfd_model(T, n_classes, enc_type, dual_type):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z_cond, reg_out = _build_rcfd_backbone(inputs, enc_type, dual_type)
    cls_out = _cls_head(z_cond, n_classes)
    return RCFDModel(inputs=inputs, outputs=[cls_out, reg_out])


def _build_rcfd_attn_recon_model(k_plus_1, T, n_classes, enc_type):
    stack_input = tf.keras.layers.Input(shape=(k_plus_1, T), name='neighbor_stack_input')
    z_cond, reg_out = _build_rcfd_attn_recon_backbone(stack_input, T, enc_type)
    cls_out = _cls_head(z_cond, n_classes)
    return RCFDModel(inputs=stack_input, outputs=[cls_out, reg_out])


def _build_rcfd_supcon_model(T, n_classes, enc_type, dual_type):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z_cond, reg_out = _build_rcfd_backbone(inputs, enc_type, dual_type)
    cls_out   = _cls_head(z_cond, n_classes)
    proj_norm = _proj_head(z_cond, 'fused')
    return RCFDSupConMTLModel(inputs=inputs, outputs=[cls_out, reg_out, proj_norm])


def _build_rcfd_supcon2_model(T, n_classes, enc_type, dual_type):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z_cond, reg_out, cnn_emb, seq_emb = _build_rcfd_backbone(
        inputs, enc_type, dual_type, return_branches=True)
    cls_out  = _cls_head(z_cond, n_classes)
    cnn_proj = _proj_head(cnn_emb, 'cnn')
    seq_proj = _proj_head(seq_emb, 'seq')
    return RCFDBranch2MTLModel(inputs=inputs, outputs=[cls_out, reg_out, cnn_proj, seq_proj])


def _build_rcfd_supcon3_model(T, n_classes, enc_type, dual_type):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z_cond, reg_out, cnn_emb, seq_emb = _build_rcfd_backbone(
        inputs, enc_type, dual_type, return_branches=True)
    cls_out    = _cls_head(z_cond, n_classes)
    cnn_proj   = _proj_head(cnn_emb, 'cnn')
    seq_proj   = _proj_head(seq_emb, 'seq')
    fused_proj = _proj_head(z_cond, 'fused')
    return RCFDBranch3MTLModel(inputs=inputs, outputs=[cls_out, reg_out, cnn_proj, seq_proj, fused_proj])


# ======================================================================
# FACTORY FUNCTIONS (27 total)
# ======================================================================

# --- Base (6)
def create_cnn_rcfd_cgd_model(T, n):   return _build_rcfd_model(T, n, 'cnn', 'cgd')
def create_cnn_rcfd_ctd_model(T, n):   return _build_rcfd_model(T, n, 'cnn', 'ctd')
def create_gru_rcfd_cgd_model(T, n):   return _build_rcfd_model(T, n, 'gru', 'cgd')
def create_gru_rcfd_ctd_model(T, n):   return _build_rcfd_model(T, n, 'gru', 'ctd')
def create_trans_rcfd_cgd_model(T, n): return _build_rcfd_model(T, n, 'transformer', 'cgd')
def create_trans_rcfd_ctd_model(T, n): return _build_rcfd_model(T, n, 'transformer', 'ctd')

# --- Base attn_recon (3) -- signature (k_plus_1, T, n), matching the other
# attn_recon factories (create_cnn_gru_dual_attn_recon_mtl_model, etc.)
def create_cnn_rcfd_attn_recon_model(k_plus_1, T, n):   return _build_rcfd_attn_recon_model(k_plus_1, T, n, 'cnn')
def create_gru_rcfd_attn_recon_model(k_plus_1, T, n):   return _build_rcfd_attn_recon_model(k_plus_1, T, n, 'gru')
def create_trans_rcfd_attn_recon_model(k_plus_1, T, n): return _build_rcfd_attn_recon_model(k_plus_1, T, n, 'transformer')

# --- SupCon v1 (6)
def create_cnn_rcfd_cgd_supcon_mtl_model(T, n):   return _build_rcfd_supcon_model(T, n, 'cnn', 'cgd')
def create_cnn_rcfd_ctd_supcon_mtl_model(T, n):   return _build_rcfd_supcon_model(T, n, 'cnn', 'ctd')
def create_gru_rcfd_cgd_supcon_mtl_model(T, n):   return _build_rcfd_supcon_model(T, n, 'gru', 'cgd')
def create_gru_rcfd_ctd_supcon_mtl_model(T, n):   return _build_rcfd_supcon_model(T, n, 'gru', 'ctd')
def create_trans_rcfd_cgd_supcon_mtl_model(T, n): return _build_rcfd_supcon_model(T, n, 'transformer', 'cgd')
def create_trans_rcfd_ctd_supcon_mtl_model(T, n): return _build_rcfd_supcon_model(T, n, 'transformer', 'ctd')

# --- SupCon v2 (6)
def create_cnn_rcfd_cgd_supcon2_mtl_model(T, n):   return _build_rcfd_supcon2_model(T, n, 'cnn', 'cgd')
def create_cnn_rcfd_ctd_supcon2_mtl_model(T, n):   return _build_rcfd_supcon2_model(T, n, 'cnn', 'ctd')
def create_gru_rcfd_cgd_supcon2_mtl_model(T, n):   return _build_rcfd_supcon2_model(T, n, 'gru', 'cgd')
def create_gru_rcfd_ctd_supcon2_mtl_model(T, n):   return _build_rcfd_supcon2_model(T, n, 'gru', 'ctd')
def create_trans_rcfd_cgd_supcon2_mtl_model(T, n): return _build_rcfd_supcon2_model(T, n, 'transformer', 'cgd')
def create_trans_rcfd_ctd_supcon2_mtl_model(T, n): return _build_rcfd_supcon2_model(T, n, 'transformer', 'ctd')

# --- SupCon v3 (6)
def create_cnn_rcfd_cgd_supcon3_mtl_model(T, n):   return _build_rcfd_supcon3_model(T, n, 'cnn', 'cgd')
def create_cnn_rcfd_ctd_supcon3_mtl_model(T, n):   return _build_rcfd_supcon3_model(T, n, 'cnn', 'ctd')
def create_gru_rcfd_cgd_supcon3_mtl_model(T, n):   return _build_rcfd_supcon3_model(T, n, 'gru', 'cgd')
def create_gru_rcfd_ctd_supcon3_mtl_model(T, n):   return _build_rcfd_supcon3_model(T, n, 'gru', 'ctd')
def create_trans_rcfd_cgd_supcon3_mtl_model(T, n): return _build_rcfd_supcon3_model(T, n, 'transformer', 'cgd')
def create_trans_rcfd_ctd_supcon3_mtl_model(T, n): return _build_rcfd_supcon3_model(T, n, 'transformer', 'ctd')


# ======================================================================
# FACTORY DICTS + KEY LISTS
# ======================================================================

_RCFD_BASE_FACTORIES = {
    'cnn_rcfd_cgd':   create_cnn_rcfd_cgd_model,
    'cnn_rcfd_ctd':   create_cnn_rcfd_ctd_model,
    'gru_rcfd_cgd':   create_gru_rcfd_cgd_model,
    'gru_rcfd_ctd':   create_gru_rcfd_ctd_model,
    'trans_rcfd_cgd': create_trans_rcfd_cgd_model,
    'trans_rcfd_ctd': create_trans_rcfd_ctd_model,
    'cnn_rcfd_attn_recon':   create_cnn_rcfd_attn_recon_model,
    'gru_rcfd_attn_recon':   create_gru_rcfd_attn_recon_model,
    'trans_rcfd_attn_recon': create_trans_rcfd_attn_recon_model,
}
# attn_recon factories take (k_plus_1, T, n) instead of (T, n) -- callers
# dispatching through _RCFD_ALL_FACTORIES need to check this set first.
RCFD_ATTN_RECON_MODEL_KEYS = (
    'cnn_rcfd_attn_recon', 'gru_rcfd_attn_recon', 'trans_rcfd_attn_recon',
)
_RCFD_SC1_FACTORIES = {
    'cnn_rcfd_cgd_supcon_mtl':   create_cnn_rcfd_cgd_supcon_mtl_model,
    'cnn_rcfd_ctd_supcon_mtl':   create_cnn_rcfd_ctd_supcon_mtl_model,
    'gru_rcfd_cgd_supcon_mtl':   create_gru_rcfd_cgd_supcon_mtl_model,
    'gru_rcfd_ctd_supcon_mtl':   create_gru_rcfd_ctd_supcon_mtl_model,
    'trans_rcfd_cgd_supcon_mtl': create_trans_rcfd_cgd_supcon_mtl_model,
    'trans_rcfd_ctd_supcon_mtl': create_trans_rcfd_ctd_supcon_mtl_model,
}
_RCFD_SC2_FACTORIES = {
    'cnn_rcfd_cgd_supcon2_mtl':   create_cnn_rcfd_cgd_supcon2_mtl_model,
    'cnn_rcfd_ctd_supcon2_mtl':   create_cnn_rcfd_ctd_supcon2_mtl_model,
    'gru_rcfd_cgd_supcon2_mtl':   create_gru_rcfd_cgd_supcon2_mtl_model,
    'gru_rcfd_ctd_supcon2_mtl':   create_gru_rcfd_ctd_supcon2_mtl_model,
    'trans_rcfd_cgd_supcon2_mtl': create_trans_rcfd_cgd_supcon2_mtl_model,
    'trans_rcfd_ctd_supcon2_mtl': create_trans_rcfd_ctd_supcon2_mtl_model,
}
_RCFD_SC3_FACTORIES = {
    'cnn_rcfd_cgd_supcon3_mtl':   create_cnn_rcfd_cgd_supcon3_mtl_model,
    'cnn_rcfd_ctd_supcon3_mtl':   create_cnn_rcfd_ctd_supcon3_mtl_model,
    'gru_rcfd_cgd_supcon3_mtl':   create_gru_rcfd_cgd_supcon3_mtl_model,
    'gru_rcfd_ctd_supcon3_mtl':   create_gru_rcfd_ctd_supcon3_mtl_model,
    'trans_rcfd_cgd_supcon3_mtl': create_trans_rcfd_cgd_supcon3_mtl_model,
    'trans_rcfd_ctd_supcon3_mtl': create_trans_rcfd_ctd_supcon3_mtl_model,
}
_RCFD_ALL_FACTORIES = {
    **_RCFD_BASE_FACTORIES,
    **_RCFD_SC1_FACTORIES,
    **_RCFD_SC2_FACTORIES,
    **_RCFD_SC3_FACTORIES,
}

RCFD_MODEL_KEYS             = list(_RCFD_BASE_FACTORIES)
RCFD_SUPCON_MTL_MODEL_KEYS  = list(_RCFD_SC1_FACTORIES)
RCFD_BRANCH2_MTL_MODEL_KEYS = list(_RCFD_SC2_FACTORIES)
RCFD_BRANCH3_MTL_MODEL_KEYS = list(_RCFD_SC3_FACTORIES)
ALL_RCFD_KEYS = (RCFD_MODEL_KEYS + RCFD_SUPCON_MTL_MODEL_KEYS
                 + RCFD_BRANCH2_MTL_MODEL_KEYS + RCFD_BRANCH3_MTL_MODEL_KEYS)
