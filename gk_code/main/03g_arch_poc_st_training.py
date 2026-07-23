"""Arch-PoC (03g): Same architectures as 03f, single-task (classification only).

Three architectures × 4 SC levels = 12 ST models:
  CGD  — CNN+GRU Dual
  CGS  — CNN+GRU Serial
  CCGD — CNN+(CNN→GRU) Dual

Usage:
  python 03g_arch_poc_st_training.py --task_id 0 --exp_folder /path/to/folder
  python 03g_arch_poc_st_training.py --task_id 0 --exp_folder /path --n_splits 5
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

_THIS     = Path(__file__).resolve()
_MAIN_DIR = _THIS.parent
sys.path.insert(0, str(_MAIN_DIR))
sys.path.insert(0, str(_MAIN_DIR / "utils"))
sys.path.insert(0, str(_MAIN_DIR / "utils" / "model_training"))

from safe_io import safe_joblib_dump, safe_keras_save
from pipeline_utils import get_exp_paths, check_task_id
from model_utils import set_global_determinism
from model_utils_mtl import _build_cnn_gru_dual_branches_mtl
from model_utils_supcon import (
    SupConModel, SupConBranch2STModel, SupConBranch3STModel,
    _proj_head,
)
import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


# ======================================================================
# MODEL CLASSES — 12 unique Keras packages
# ======================================================================

# --- CGD ---

@tf.keras.utils.register_keras_serializable(package='arch_poc_st_cgd')
class ArchPocSTCGDModel(tf.keras.Model):
    """CGD SC0 ST: plain CE via compiled loss. Output: cls_out."""
    pass


@tf.keras.utils.register_keras_serializable(package='arch_poc_st_cgd_sc1')
class ArchPocSTCGDSC1Model(SupConModel):
    """CGD SC1 ST: CE + SupCon on fused. Outputs: [cls_out, proj]."""
    pass


@tf.keras.utils.register_keras_serializable(package='arch_poc_st_cgd_sc2')
class ArchPocSTCGDSC2Model(SupConBranch2STModel):
    """CGD SC2 ST: CE + SupCon on cnn+seq. Outputs: [cls_out, cnn_proj, seq_proj]."""
    pass


@tf.keras.utils.register_keras_serializable(package='arch_poc_st_cgd_sc3')
class ArchPocSTCGDSC3Model(SupConBranch3STModel):
    """CGD SC3 ST: CE + SupCon on cnn+seq+fused. Outputs: [cls_out, cnn_proj, seq_proj, fused_proj]."""
    pass


# --- CGS ---

@tf.keras.utils.register_keras_serializable(package='arch_poc_st_cgs')
class ArchPocSTCGSModel(tf.keras.Model):
    """CGS SC0 ST: plain CE via compiled loss. Output: cls_out."""
    pass


@tf.keras.utils.register_keras_serializable(package='arch_poc_st_cgs_sc1')
class ArchPocSTCGSSC1Model(SupConModel):
    """CGS SC1 ST: CE + SupCon on z. Outputs: [cls_out, proj]."""
    pass


@tf.keras.utils.register_keras_serializable(package='arch_poc_st_cgs_sc2')
class ArchPocSTCGSSC2Model(SupConBranch2STModel):
    """CGS SC2 ST: CE + SupCon on cnn_tap+gru_raw. Outputs: [cls_out, cnn_proj, seq_proj]."""
    pass


@tf.keras.utils.register_keras_serializable(package='arch_poc_st_cgs_sc3')
class ArchPocSTCGSSC3Model(SupConBranch3STModel):
    """CGS SC3 ST: CE + SupCon on cnn_tap+gru_raw+z. Outputs: [cls_out, cnn_proj, seq_proj, fused_proj]."""
    pass


# --- CCGD ---

@tf.keras.utils.register_keras_serializable(package='arch_poc_st_ccgd')
class ArchPocSTCCGDModel(tf.keras.Model):
    """CCGD SC0 ST: plain CE via compiled loss. Output: cls_out."""
    pass


@tf.keras.utils.register_keras_serializable(package='arch_poc_st_ccgd_sc1')
class ArchPocSTCCGDSC1Model(SupConModel):
    """CCGD SC1 ST: CE + SupCon on fused. Outputs: [cls_out, proj]."""
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
# BACKBONE BUILDERS (copied from 03f — same dimensions)
# ======================================================================

def _build_cnn_gru_serial_branches(inputs, return_branches=False):
    """CNN → GRU serial: z = 64-dim.

    SC branches (return_branches=True): cnn_tap (8-dim), gru_raw (32-dim).
    """
    x       = tf.keras.layers.Conv1D(16, 5, activation='relu')(inputs)
    cnn_seq = tf.keras.layers.Conv1D(8,  3, activation='relu')(x)
    g       = tf.keras.layers.Bidirectional(
                  tf.keras.layers.GRU(32, return_sequences=True))(cnn_seq)
    g       = tf.keras.layers.LayerNormalization()(g)
    gru_raw = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(16))(g)
    g       = tf.keras.layers.Dropout(0.2)(gru_raw)
    z       = tf.keras.layers.Dropout(0.2)(
                  tf.keras.layers.Dense(64, activation='relu')(g))
    if return_branches:
        cnn_tap = tf.keras.layers.GlobalAveragePooling1D()(cnn_seq)
        return cnn_tap, gru_raw, z
    return z


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


# ======================================================================
# HEADS — cls only (no reg)
# ======================================================================

def _heads(z, n_classes):
    return tf.keras.layers.Dense(
        n_classes, activation='softmax', name='cls_out')(
        tf.keras.layers.Dense(16, activation='relu', name='cls_feat')(z))


# ======================================================================
# BUILD FUNCTIONS — CGD
# ======================================================================

def _build_cgd_st_poc(T, n_classes):
    inputs  = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z       = _build_cnn_gru_dual_branches_mtl(inputs)
    cls_out = _heads(z, n_classes)
    return ArchPocSTCGDModel(inputs=inputs, outputs=cls_out)


def _build_cgd_st_poc_sc1(T, n_classes):
    inputs    = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z         = _build_cnn_gru_dual_branches_mtl(inputs)
    cls_out   = _heads(z, n_classes)
    proj_norm = tf.keras.layers.UnitNormalization(axis=1, name='proj')(
        tf.keras.layers.Dense(64, activation='relu', name='proj_hidden')(z))
    return ArchPocSTCGDSC1Model(inputs=inputs, outputs=[cls_out, proj_norm])


def _build_cgd_st_poc_sc2(T, n_classes):
    inputs               = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    cnn_emb, gru_emb, z = _build_cnn_gru_dual_branches_mtl(inputs, return_branches=True)
    cls_out  = _heads(z, n_classes)
    cnn_proj = _proj_head(cnn_emb, 'cnn')
    seq_proj = _proj_head(gru_emb, 'seq')
    return ArchPocSTCGDSC2Model(inputs=inputs, outputs=[cls_out, cnn_proj, seq_proj])


def _build_cgd_st_poc_sc3(T, n_classes):
    inputs               = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    cnn_emb, gru_emb, z = _build_cnn_gru_dual_branches_mtl(inputs, return_branches=True)
    cls_out    = _heads(z, n_classes)
    cnn_proj   = _proj_head(cnn_emb, 'cnn')
    seq_proj   = _proj_head(gru_emb, 'seq')
    fused_proj = _proj_head(z, 'fused')
    return ArchPocSTCGDSC3Model(inputs=inputs, outputs=[cls_out, cnn_proj, seq_proj, fused_proj])


# ======================================================================
# BUILD FUNCTIONS — CGS
# ======================================================================

def _build_cgs_st_poc(T, n_classes):
    inputs  = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z       = _build_cnn_gru_serial_branches(inputs)
    cls_out = _heads(z, n_classes)
    return ArchPocSTCGSModel(inputs=inputs, outputs=cls_out)


def _build_cgs_st_poc_sc1(T, n_classes):
    inputs    = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z         = _build_cnn_gru_serial_branches(inputs)
    cls_out   = _heads(z, n_classes)
    proj_norm = tf.keras.layers.UnitNormalization(axis=1, name='proj')(
        tf.keras.layers.Dense(64, activation='relu', name='proj_hidden')(z))
    return ArchPocSTCGSSC1Model(inputs=inputs, outputs=[cls_out, proj_norm])


def _build_cgs_st_poc_sc2(T, n_classes):
    inputs              = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    cnn_tap, gru_raw, z = _build_cnn_gru_serial_branches(inputs, return_branches=True)
    cls_out  = _heads(z, n_classes)
    cnn_proj = _proj_head(cnn_tap, 'cnn')
    seq_proj = _proj_head(gru_raw, 'seq')
    return ArchPocSTCGSSC2Model(inputs=inputs, outputs=[cls_out, cnn_proj, seq_proj])


def _build_cgs_st_poc_sc3(T, n_classes):
    inputs              = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    cnn_tap, gru_raw, z = _build_cnn_gru_serial_branches(inputs, return_branches=True)
    cls_out    = _heads(z, n_classes)
    cnn_proj   = _proj_head(cnn_tap, 'cnn')
    seq_proj   = _proj_head(gru_raw, 'seq')
    fused_proj = _proj_head(z, 'fused')
    return ArchPocSTCGSSC3Model(inputs=inputs, outputs=[cls_out, cnn_proj, seq_proj, fused_proj])


# ======================================================================
# BUILD FUNCTIONS — CCGD
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
# KEY LISTS + FACTORY MAP
# ======================================================================

CGD_ST_KEYS  = ['cgd_arch_poc_st',  'cgd_arch_poc_st_sc1',  'cgd_arch_poc_st_sc2',  'cgd_arch_poc_st_sc3']
CGS_ST_KEYS  = ['cgs_arch_poc_st',  'cgs_arch_poc_st_sc1',  'cgs_arch_poc_st_sc2',  'cgs_arch_poc_st_sc3']
CCGD_ST_KEYS = ['ccgd_arch_poc_st', 'ccgd_arch_poc_st_sc1', 'ccgd_arch_poc_st_sc2', 'ccgd_arch_poc_st_sc3']
ALL_ARCH_POC_ST_KEYS = CGD_ST_KEYS + CGS_ST_KEYS + CCGD_ST_KEYS

SC0_KEYS = {CGD_ST_KEYS[0], CGS_ST_KEYS[0], CCGD_ST_KEYS[0]}

_FACTORIES = {
    'cgd_arch_poc_st':       _build_cgd_st_poc,
    'cgd_arch_poc_st_sc1':   _build_cgd_st_poc_sc1,
    'cgd_arch_poc_st_sc2':   _build_cgd_st_poc_sc2,
    'cgd_arch_poc_st_sc3':   _build_cgd_st_poc_sc3,
    'cgs_arch_poc_st':       _build_cgs_st_poc,
    'cgs_arch_poc_st_sc1':   _build_cgs_st_poc_sc1,
    'cgs_arch_poc_st_sc2':   _build_cgs_st_poc_sc2,
    'cgs_arch_poc_st_sc3':   _build_cgs_st_poc_sc3,
    'ccgd_arch_poc_st':      _build_ccgd_st_poc,
    'ccgd_arch_poc_st_sc1':  _build_ccgd_st_poc_sc1,
    'ccgd_arch_poc_st_sc2':  _build_ccgd_st_poc_sc2,
    'ccgd_arch_poc_st_sc3':  _build_ccgd_st_poc_sc3,
}


# ======================================================================
# KEY / PRINT MAPS
# ======================================================================

def _mk(key):
    return (f'y_preds_AC_{key}_', f'y_probs_AC_{key}_', f'classes_AC_{key}_')


_ARCH_POC_ST_KEY_MAP = {m: _mk(m) for m in ALL_ARCH_POC_ST_KEYS}

_ARCH_POC_ST_PRINT_MAP = {
    'cgd_arch_poc_st':       'CGD ST (Arch-PoC)',
    'cgd_arch_poc_st_sc1':   'CGD SC1 ST (Arch-PoC)',
    'cgd_arch_poc_st_sc2':   'CGD SC2 ST (Arch-PoC)',
    'cgd_arch_poc_st_sc3':   'CGD SC3 ST (Arch-PoC)',
    'cgs_arch_poc_st':       'CGS Serial ST (Arch-PoC)',
    'cgs_arch_poc_st_sc1':   'CGS Serial SC1 ST (Arch-PoC)',
    'cgs_arch_poc_st_sc2':   'CGS Serial SC2 ST (Arch-PoC)',
    'cgs_arch_poc_st_sc3':   'CGS Serial SC3 ST (Arch-PoC)',
    'ccgd_arch_poc_st':      'CCGD Mixed ST (Arch-PoC)',
    'ccgd_arch_poc_st_sc1':  'CCGD Mixed SC1 ST (Arch-PoC)',
    'ccgd_arch_poc_st_sc2':  'CCGD Mixed SC2 ST (Arch-PoC)',
    'ccgd_arch_poc_st_sc3':  'CCGD Mixed SC3 ST (Arch-PoC)',
}

_ALL_ARCH_POC_ST_RESULT_KEYS = set()
for _, (_pk, _prk, _ck) in _ARCH_POC_ST_KEY_MAP.items():
    _ALL_ARCH_POC_ST_RESULT_KEYS.update([_pk, _prk, _ck])


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


def _make_splits(X_f, y_f, n_splits):
    unique_cls, cls_counts = np.unique(y_f, return_counts=True)
    n_cls     = len(unique_cls)
    test_size = max(int(len(y_f) * 0.10), n_cls)
    if n_splits == 1:
        return list(StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=0).split(X_f, y_f))
    actual = min(n_splits, int(np.min(cls_counts)))
    return list(StratifiedKFold(n_splits=actual, shuffle=True, random_state=0).split(X_f, y_f))


def _fold_data(X_f, y_f, tr_idx, is_sc0):
    try:
        tr_sub, val_sub = train_test_split(
            np.arange(len(tr_idx)), test_size=0.1,
            stratify=y_f[tr_idx], random_state=0)
        X_tr, X_val = X_f[tr_idx][tr_sub], X_f[tr_idx][val_sub]
        y_tr, y_val = y_f[tr_idx][tr_sub], y_f[tr_idx][val_sub]
        if is_sc0:
            fit_y    = y_tr
            val_data = (X_val, y_val)
        else:
            fit_y    = {'cls_out': y_tr}
            val_data = (X_val, {'cls_out': y_val})
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=100,
                restore_best_weights=True, verbose=0),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=30, min_lr=1e-5),
        ]
    except ValueError:
        X_tr, y_tr = X_f[tr_idx], y_f[tr_idx]
        fit_y      = y_tr if is_sc0 else {'cls_out': y_tr}
        val_data, callbacks = None, []
    return X_tr, y_tr, fit_y, val_data, callbacks


# ======================================================================
# TRAINING LOOP
# ======================================================================

def run_03g_training(
    X_curves, features_df, y_encoded,
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
                    if rk in _ALL_ARCH_POC_ST_RESULT_KEYS:
                        del re[rk]
                        n_cleared += 1
        if n_cleared:
            print(f'  -> [FORCE RERUN] Cleared {n_cleared} arch_poc_st cached key(s).')

    f           = None
    _filter_str = 'None'
    print(f'\n  -> Filter: None (Baseline)')

    mask = np.ones(len(y_encoded), dtype=bool)
    X_f  = np.nan_to_num(X_curves[mask], nan=0.0, posinf=0.0, neginf=0.0)
    y_f  = y_encoded[mask]

    unique_cls, cls_counts = np.unique(y_f, return_counts=True)
    rare = unique_cls[cls_counts < 2]
    if len(rare) > 0:
        vcm = ~np.isin(y_f, rare)
        X_f, y_f = X_f[vcm], y_f[vcm]
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

    for m in ALL_ARCH_POC_ST_KEYS:
        preds_key, probs_key, classes_key = _ARCH_POC_ST_KEY_MAP[m]
        is_sc0 = m in SC0_KEYS

        if preds_key in res_entry:
            cached_accs = [accuracy_score(yt, yp)
                           for yt, yp in zip(res_entry['y_trues_'], res_entry[preds_key])]
            print(f'     [CACHE] {_ARCH_POC_ST_PRINT_MAP[m]:<36} | '
                  f'acc={np.mean(cached_accs)*100:.2f}%±{np.std(cached_accs)*100:.2f}%')
            continue

        preds_folds, probs_folds, classes_folds = [], [], []
        t0 = time.perf_counter()

        for fold_idx, (tr_idx, te_idx) in enumerate(splits):
            X_tr, y_tr, fit_y, val_data, callbacks = _fold_data(X_f, y_f, tr_idx, is_sc0)

            tf.keras.backend.clear_session()
            model = _FACTORIES[m](T, n_cls)

            _probe = model(X_tr[:1], training=False)
            if is_sc0:
                if not isinstance(_probe, tf.Tensor):
                    raise RuntimeError(
                        f'{m}: SC0 expected single Tensor, got {type(_probe).__name__}.')
            else:
                if not isinstance(_probe, (list, tuple)) or len(_probe) < 2:
                    raise RuntimeError(
                        f'{m}: expected ≥2 outputs, '
                        f'got {type(_probe).__name__} '
                        f'len={getattr(_probe, "__len__", lambda: "?")()}.')

            if is_sc0:
                # Built-in Keras loss — handles named output → y matching
                # correctly, avoiding the custom train_step dict-wrapping bug.
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
                    loss='sparse_categorical_crossentropy',
                    jit_compile=False)
            else:
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

            if is_sc0:
                cls_probs = model.predict(X_f[te_idx], verbose=0)
            else:
                raw_out = model.predict(X_f[te_idx], verbose=0)
                if not isinstance(raw_out, (list, tuple)):
                    raise RuntimeError(
                        f'{m}: model.predict() returned {type(raw_out).__name__} '
                        f'instead of list.')
                cls_probs = raw_out[0]

            pred = np.argmax(cls_probs, axis=1)
            preds_folds.append(pred)
            probs_folds.append(cls_probs)
            classes_folds.append(encoder_classes)

            gc.collect()

        duration = time.perf_counter() - t0
        accs = [accuracy_score(yt, yp)
                for yt, yp in zip(res_entry['y_trues_'], preds_folds)]

        res_entry[preds_key]   = preds_folds
        res_entry[probs_key]   = probs_folds
        res_entry[classes_key] = classes_folds

        results[f] = res_entry
        all_ml_results[clean_title][mode_key] = results
        safe_joblib_dump(all_ml_results, results_file_path, compress=3)

        print(f'     [+] {_ARCH_POC_ST_PRINT_MAP[m]:<36} | '
              f'acc={np.mean(accs)*100:.2f}%±{np.std(accs)*100:.2f}% | '
              f'{_fmt_hms(duration)}')

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
        for key, val in res.items():
            if not (key.startswith('y_preds_AC_') and 'arch_poc_st' in key):
                continue
            try:
                accs = [accuracy_score(yt, yp)
                        for yt, yp in zip(res['y_trues_'], val)]
                lbl = _ARCH_POC_ST_PRINT_MAP.get(
                    key.replace('y_preds_AC_', '').rstrip('_'), key)
                rows.append((np.mean(accs) * 100, np.std(accs) * 100, lbl))
            except Exception:
                continue
    rows.sort(key=lambda x: x[0], reverse=True)
    W = 80
    print(f'\n  Leaderboard [Arch-PoC ST] {clean_title} / {mode_key}')
    print('  ' + '-' * W)
    for i, (acc, std, lbl) in enumerate(rows):
        print(f'  {i+1:3d}.  {acc:6.2f}% +/- {std:5.2f}%  |  {lbl}')
    print('  ' + '-' * W + '\n')


# ======================================================================
# MAIN
# ======================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='03g: Arch-PoC 12-model single-task training')
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
    print(f'[RUNNING] 03g_arch_poc_st_training.py  [Arch-PoC ST 12-model]')
    print(f"{'='*70}\n")
    print(f"{'#'*80}\nSTARTING TRAINING FOR: {exp_path.name}\n{'#'*80}")

    arch_poc_dir   = Path(exp_path) / 'arch_poc'
    model_save_dir = arch_poc_dir / 'models'
    model_save_dir.mkdir(parents=True, exist_ok=True)

    joblib_name       = ('arch_poc_st_results_5fold.joblib' if args.n_splits > 1
                         else 'arch_poc_st_results.joblib')
    results_file_path = str(arch_poc_dir / joblib_name)

    training_data    = _load_training_data(exp_path)
    dataset_name     = training_data['dataset_name']
    dataset          = training_data['dataset']
    kinetic_features = training_data['kinetic_features']
    Y_well           = training_data['Y_well']

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

        results = run_03g_training(
            X_curves          = X_curves,
            features_df       = features_df,
            y_encoded         = y_full,
            encoder_classes   = encoder.classes_,
            n_splits          = args.n_splits,
            cached_results    = all_ml_results[clean_title][mode_key],
            force_rerun       = args.force_rerun,
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
