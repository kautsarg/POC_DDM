"""CCGD arch-poc single-task model classes, backbone, and builders.

Shared between 03g_arch_poc_st_training.py (direct training loop) and
model_utils.py (evaluate_outlier_filters dispatch via 03_main_training.py).
"""
import tensorflow as tf
from model_utils_supcon import SupConModel, SupConBranch2STModel, SupConBranch3STModel, _proj_head


# ======================================================================
# MODEL CLASSES — 4 unique Keras packages for CCGD ST
# ======================================================================

@tf.keras.utils.register_keras_serializable(package='arch_poc_st_ccgd')
class ArchPocSTCCGDModel(tf.keras.Model):
    """CCGD SC0 ST: plain CE via compiled loss. Output: cls_out."""
    pass


@tf.keras.utils.register_keras_serializable(package='arch_poc_st_ccgd_sc1')
class ArchPocSTCCGDSC1Model(SupConModel):
    """CCGD SC1 ST: CE + SupCon on fused z. Outputs: [cls_out, proj]."""
    pass


@tf.keras.utils.register_keras_serializable(package='arch_poc_st_ccgd_sc2')
class ArchPocSTCCGDSC2Model(SupConBranch2STModel):
    """CCGD SC2 ST: CE + SupCon on cnn+cnngru. Outputs: [cls_out, cnn_proj, seq_proj]."""
    pass


@tf.keras.utils.register_keras_serializable(package='arch_poc_st_ccgd_sc3')
class ArchPocSTCCGDSC3Model(SupConBranch3STModel):
    """CCGD SC3 ST: CE + SupCon on cnn+cnngru+fused. Outputs: [cls_out, cnn_proj, seq_proj, fused_proj]."""
    pass


# ======================================================================
# BACKBONE
# ======================================================================

def _build_cnn_cnngru_dual_branches(inputs, return_branches=False):
    """CNN ‖ (CNN→GRU) dual: separate CNN weights per branch. z = 96-dim.

    SC branches (return_branches=True): cnn_emb (32-dim), cnngru_emb (64-dim).
    """
    a       = tf.keras.layers.Conv1D(16, 5, activation='relu')(inputs)
    a       = tf.keras.layers.Conv1D(8,  3, activation='relu')(a)
    a       = tf.keras.layers.Flatten()(a)
    cnn_emb = tf.keras.layers.Dense(32, activation='relu')(a)

    b          = tf.keras.layers.Conv1D(16, 5, activation='relu')(inputs)
    b_seq      = tf.keras.layers.Conv1D(8,  3, activation='relu')(b)
    b          = tf.keras.layers.Bidirectional(
                     tf.keras.layers.GRU(32, return_sequences=True))(b_seq)
    b          = tf.keras.layers.LayerNormalization()(b)
    b          = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(16))(b)
    b          = tf.keras.layers.Dropout(0.2)(b)
    cnngru_emb = tf.keras.layers.Dense(64, activation='relu')(b)

    merged = tf.keras.layers.Concatenate()([cnn_emb, cnngru_emb])
    z      = tf.keras.layers.Dropout(0.2)(
                 tf.keras.layers.Dense(96, activation='relu')(merged))
    if return_branches:
        return cnn_emb, cnngru_emb, z
    return z


def _heads(z, n_classes):
    return tf.keras.layers.Dense(
        n_classes, activation='softmax', name='cls_out')(
        tf.keras.layers.Dense(16, activation='relu', name='cls_feat')(z))


# ======================================================================
# BUILD FUNCTIONS
# ======================================================================

def _build_ccgd_st_poc(T, n_classes):
    inputs  = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z       = _build_cnn_cnngru_dual_branches(inputs)
    cls_out = _heads(z, n_classes)
    return ArchPocSTCCGDModel(inputs=inputs, outputs=cls_out)


def _build_ccgd_st_poc_sc1(T, n_classes):
    inputs    = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z         = _build_cnn_cnngru_dual_branches(inputs)
    cls_out   = _heads(z, n_classes)
    proj_norm = tf.keras.layers.UnitNormalization(axis=1, name='proj')(
        tf.keras.layers.Dense(64, activation='relu', name='proj_hidden')(z))
    return ArchPocSTCCGDSC1Model(inputs=inputs, outputs=[cls_out, proj_norm])


def _build_ccgd_st_poc_sc2(T, n_classes):
    inputs                 = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    cnn_emb, cnngru_emb, z = _build_cnn_cnngru_dual_branches(inputs, return_branches=True)
    cls_out  = _heads(z, n_classes)
    cnn_proj = _proj_head(cnn_emb,    'cnn')
    seq_proj = _proj_head(cnngru_emb, 'seq')
    return ArchPocSTCCGDSC2Model(inputs=inputs, outputs=[cls_out, cnn_proj, seq_proj])


def _build_ccgd_st_poc_sc3(T, n_classes):
    inputs                 = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    cnn_emb, cnngru_emb, z = _build_cnn_cnngru_dual_branches(inputs, return_branches=True)
    cls_out    = _heads(z, n_classes)
    cnn_proj   = _proj_head(cnn_emb,    'cnn')
    seq_proj   = _proj_head(cnngru_emb, 'seq')
    fused_proj = _proj_head(z,          'fused')
    return ArchPocSTCCGDSC3Model(inputs=inputs, outputs=[cls_out, cnn_proj, seq_proj, fused_proj])


# ======================================================================
# KEYS + FACTORY MAP
# ======================================================================

CCGD_ST_KEYS = ['ccgd_arch_poc_st', 'ccgd_arch_poc_st_sc1', 'ccgd_arch_poc_st_sc2', 'ccgd_arch_poc_st_sc3']
CCGD_ST_ALL_KEYS = set(CCGD_ST_KEYS)

_CCGD_FACTORIES = {
    'ccgd_arch_poc_st':      _build_ccgd_st_poc,
    'ccgd_arch_poc_st_sc1':  _build_ccgd_st_poc_sc1,
    'ccgd_arch_poc_st_sc2':  _build_ccgd_st_poc_sc2,
    'ccgd_arch_poc_st_sc3':  _build_ccgd_st_poc_sc3,
}
