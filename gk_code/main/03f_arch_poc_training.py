"""Arch-PoC (03f): Dual vs Serial vs Mixed CNN+GRU architecture comparison.

Three architectures × 4 SC levels = 12 MTL models:
  CGD  — CNN+GRU Dual      (existing _build_cnn_gru_dual_branches_mtl)
  CGS  — CNN+GRU Serial    (new: CNN sequence → GRU → z)
  CCGD — CNN+(CNN→GRU) Dual (new: CNN branch ‖ serial CNN→GRU branch)

Research question: does routing CNN features through GRU (serial) beat parallel branches,
and does a mixed dual (CNN ‖ CNN→GRU) outperform either alone?

Usage:
  python 03f_arch_poc_training.py --task_id 0 --exp_folder /path/to/folder
  python 03f_arch_poc_training.py --task_id 0 --exp_folder /path --n_splits 5
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
from model_utils_mtl import (
    MTLModel, REG_SENTINEL,
    _normalize_concentration, _inverse_normalize_concentration,
    _build_cnn_gru_dual_branches_mtl,
)
from model_utils_supcon import (
    SupConMTLModel, SupConBranch2MTLModel, SupConBranch3MTLModel,
    _proj_head, supcon_loss, SUPCON_TEMP,
)
import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


# ======================================================================
# MODEL CLASSES — 12 unique Keras packages
# ======================================================================

# --- CGD (CNN+GRU Dual) ---

@tf.keras.utils.register_keras_serializable(package='arch_poc_cgd')
class ArchPocCGDModel(MTLModel):
    """CGD SC0: UW-SO(CE, MSE). Outputs: [cls_out, reg_out]."""
    pass


@tf.keras.utils.register_keras_serializable(package='arch_poc_cgd_sc1')
class ArchPocCGDSC1Model(SupConMTLModel):
    """CGD SC1: UW-SO + SupCon on fused. Outputs: [cls_out, reg_out, proj]."""
    pass


@tf.keras.utils.register_keras_serializable(package='arch_poc_cgd_sc2')
class ArchPocCGDSC2Model(SupConBranch2MTLModel):
    """CGD SC2: UW-SO + SupCon on cnn+seq. Outputs: [cls_out, reg_out, cnn_proj, seq_proj]."""
    pass


@tf.keras.utils.register_keras_serializable(package='arch_poc_cgd_sc3')
class ArchPocCGDSC3Model(SupConBranch3MTLModel):
    """CGD SC3: UW-SO + SupCon on cnn+seq+fused. Outputs: [cls_out, reg_out, cnn_proj, seq_proj, fused_proj]."""
    pass


# --- CGS (CNN+GRU Serial) ---

@tf.keras.utils.register_keras_serializable(package='arch_poc_cgs')
class ArchPocCGSModel(MTLModel):
    """CGS SC0: UW-SO(CE, MSE). Outputs: [cls_out, reg_out]."""
    pass


@tf.keras.utils.register_keras_serializable(package='arch_poc_cgs_sc1')
class ArchPocCGSSC1Model(SupConMTLModel):
    """CGS SC1: UW-SO + SupCon on z. Outputs: [cls_out, reg_out, proj]."""
    pass


@tf.keras.utils.register_keras_serializable(package='arch_poc_cgs_sc2')
class ArchPocCGSSC2Model(SupConBranch2MTLModel):
    """CGS SC2: UW-SO + SupCon on cnn_tap+gru_raw. Outputs: [cls_out, reg_out, cnn_proj, seq_proj]."""
    pass


@tf.keras.utils.register_keras_serializable(package='arch_poc_cgs_sc3')
class ArchPocCGSSC3Model(SupConBranch3MTLModel):
    """CGS SC3: UW-SO + SupCon on cnn_tap+gru_raw+z. Outputs: [cls_out, reg_out, cnn_proj, seq_proj, fused_proj]."""
    pass


# --- CCGD (CNN + CNN→GRU Dual) ---

@tf.keras.utils.register_keras_serializable(package='arch_poc_ccgd')
class ArchPocCCGDModel(MTLModel):
    """CCGD SC0: UW-SO(CE, MSE). Outputs: [cls_out, reg_out]."""
    pass


@tf.keras.utils.register_keras_serializable(package='arch_poc_ccgd_sc1')
class ArchPocCCGDSC1Model(SupConMTLModel):
    """CCGD SC1: UW-SO + SupCon on fused. Outputs: [cls_out, reg_out, proj]."""
    pass


@tf.keras.utils.register_keras_serializable(package='arch_poc_ccgd_sc2')
class ArchPocCCGDSC2Model(SupConBranch2MTLModel):
    """CCGD SC2: UW-SO + SupCon on cnn+cnngru. Outputs: [cls_out, reg_out, cnn_proj, seq_proj]."""
    pass


@tf.keras.utils.register_keras_serializable(package='arch_poc_ccgd_sc3')
class ArchPocCCGDSC3Model(SupConBranch3MTLModel):
    """CCGD SC3: UW-SO + SupCon on cnn+cnngru+fused. Outputs: [cls_out, reg_out, cnn_proj, seq_proj, fused_proj]."""
    pass


# ======================================================================
# BACKBONE BUILDERS
# ======================================================================

def _build_cnn_gru_serial_branches(inputs, return_branches=False):
    """CNN → GRU serial: CNN pre-processes signal; GRU models patterns over CNN features.

    z = 64-dim.
    SC branches (only built when return_branches=True):
      cnn_tap = GlobalAvgPool(CNN sequence) — 8-dim pooled CNN feature summary
      gru_raw = BiGRU(16) output — 32-dim, before Dense(64) projection
    """
    x       = tf.keras.layers.Conv1D(16, 5, activation='relu')(inputs)
    cnn_seq = tf.keras.layers.Conv1D(8,  3, activation='relu')(x)        # (T-6, 8)
    g       = tf.keras.layers.Bidirectional(
                  tf.keras.layers.GRU(32, return_sequences=True))(cnn_seq)
    g       = tf.keras.layers.LayerNormalization()(g)
    gru_raw = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(16))(g)  # (32,)
    g       = tf.keras.layers.Dropout(0.2)(gru_raw)
    z       = tf.keras.layers.Dropout(0.2)(
                  tf.keras.layers.Dense(64, activation='relu')(g))
    if return_branches:
        cnn_tap = tf.keras.layers.GlobalAveragePooling1D()(cnn_seq)      # (8,)
        return cnn_tap, gru_raw, z
    return z


def _build_cnn_cnngru_dual_branches(inputs, return_branches=False):
    """CNN ‖ (CNN→GRU) dual: parallel CNN branch + serial CNN→GRU branch.

    Separate CNN weights per branch. z = 96-dim.
    SC branches: cnn_emb (32-dim) and cnngru_emb (64-dim).
    """
    # Branch A — CNN only
    a       = tf.keras.layers.Conv1D(16, 5, activation='relu')(inputs)
    a       = tf.keras.layers.Conv1D(8,  3, activation='relu')(a)
    a       = tf.keras.layers.Flatten()(a)
    cnn_emb = tf.keras.layers.Dense(32, activation='relu')(a)             # (32,)

    # Branch B — CNN → GRU serial (independent CNN weights from Branch A)
    b       = tf.keras.layers.Conv1D(16, 5, activation='relu')(inputs)
    b_seq   = tf.keras.layers.Conv1D(8,  3, activation='relu')(b)         # (T-6, 8) sequence
    b       = tf.keras.layers.Bidirectional(
                  tf.keras.layers.GRU(32, return_sequences=True))(b_seq)
    b       = tf.keras.layers.LayerNormalization()(b)
    b       = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(16))(b)
    b       = tf.keras.layers.Dropout(0.2)(b)
    cnngru_emb = tf.keras.layers.Dense(64, activation='relu')(b)          # (64,)

    merged = tf.keras.layers.Concatenate()([cnn_emb, cnngru_emb])
    z      = tf.keras.layers.Dropout(0.2)(
                 tf.keras.layers.Dense(96, activation='relu')(merged))    # (96,)
    if return_branches:
        return cnn_emb, cnngru_emb, z
    return z


# ======================================================================
# BUILD FUNCTIONS — CGD
# ======================================================================

def _heads(z, n_classes):
    cls_out = tf.keras.layers.Dense(
        n_classes, activation='softmax', name='cls_out')(
        tf.keras.layers.Dense(16, activation='relu', name='cls_feat')(z))
    reg_out = tf.keras.layers.Dense(1, activation='linear', name='reg_out')(
        tf.keras.layers.Dense(8, activation='relu', name='reg_hidden')(
        tf.keras.layers.Dense(16, activation='relu', name='reg_feat')(z)))
    return cls_out, reg_out


def _build_cgd_poc(T, n_classes):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z      = _build_cnn_gru_dual_branches_mtl(inputs)
    cls_out, reg_out = _heads(z, n_classes)
    return ArchPocCGDModel(inputs=inputs, outputs=[cls_out, reg_out])


def _build_cgd_poc_sc1(T, n_classes):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z      = _build_cnn_gru_dual_branches_mtl(inputs)
    cls_out, reg_out = _heads(z, n_classes)
    proj_norm = tf.keras.layers.UnitNormalization(axis=1, name='proj')(
        tf.keras.layers.Dense(64, activation='relu', name='proj_hidden')(z))
    return ArchPocCGDSC1Model(inputs=inputs, outputs=[cls_out, reg_out, proj_norm])


def _build_cgd_poc_sc2(T, n_classes):
    inputs               = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    cnn_emb, gru_emb, z = _build_cnn_gru_dual_branches_mtl(inputs, return_branches=True)
    cls_out, reg_out     = _heads(z, n_classes)
    cnn_proj = _proj_head(cnn_emb, 'cnn')
    seq_proj = _proj_head(gru_emb, 'seq')
    return ArchPocCGDSC2Model(inputs=inputs, outputs=[cls_out, reg_out, cnn_proj, seq_proj])


def _build_cgd_poc_sc3(T, n_classes):
    inputs               = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    cnn_emb, gru_emb, z = _build_cnn_gru_dual_branches_mtl(inputs, return_branches=True)
    cls_out, reg_out     = _heads(z, n_classes)
    cnn_proj   = _proj_head(cnn_emb, 'cnn')
    seq_proj   = _proj_head(gru_emb, 'seq')
    fused_proj = _proj_head(z, 'fused')
    return ArchPocCGDSC3Model(inputs=inputs, outputs=[cls_out, reg_out, cnn_proj, seq_proj, fused_proj])


# ======================================================================
# BUILD FUNCTIONS — CGS
# ======================================================================

def _build_cgs_poc(T, n_classes):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z      = _build_cnn_gru_serial_branches(inputs)
    cls_out, reg_out = _heads(z, n_classes)
    return ArchPocCGSModel(inputs=inputs, outputs=[cls_out, reg_out])


def _build_cgs_poc_sc1(T, n_classes):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z      = _build_cnn_gru_serial_branches(inputs)
    cls_out, reg_out = _heads(z, n_classes)
    proj_norm = tf.keras.layers.UnitNormalization(axis=1, name='proj')(
        tf.keras.layers.Dense(64, activation='relu', name='proj_hidden')(z))
    return ArchPocCGSSC1Model(inputs=inputs, outputs=[cls_out, reg_out, proj_norm])


def _build_cgs_poc_sc2(T, n_classes):
    inputs                  = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    cnn_tap, gru_raw, z     = _build_cnn_gru_serial_branches(inputs, return_branches=True)
    cls_out, reg_out        = _heads(z, n_classes)
    cnn_proj = _proj_head(cnn_tap, 'cnn')
    seq_proj = _proj_head(gru_raw, 'seq')
    return ArchPocCGSSC2Model(inputs=inputs, outputs=[cls_out, reg_out, cnn_proj, seq_proj])


def _build_cgs_poc_sc3(T, n_classes):
    inputs              = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    cnn_tap, gru_raw, z = _build_cnn_gru_serial_branches(inputs, return_branches=True)
    cls_out, reg_out    = _heads(z, n_classes)
    cnn_proj   = _proj_head(cnn_tap, 'cnn')
    seq_proj   = _proj_head(gru_raw, 'seq')
    fused_proj = _proj_head(z, 'fused')
    return ArchPocCGSSC3Model(inputs=inputs, outputs=[cls_out, reg_out, cnn_proj, seq_proj, fused_proj])


# ======================================================================
# BUILD FUNCTIONS — CCGD
# ======================================================================

def _build_ccgd_poc(T, n_classes):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z      = _build_cnn_cnngru_dual_branches(inputs)
    cls_out, reg_out = _heads(z, n_classes)
    return ArchPocCCGDModel(inputs=inputs, outputs=[cls_out, reg_out])


def _build_ccgd_poc_sc1(T, n_classes):
    inputs = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z      = _build_cnn_cnngru_dual_branches(inputs)
    cls_out, reg_out = _heads(z, n_classes)
    proj_norm = tf.keras.layers.UnitNormalization(axis=1, name='proj')(
        tf.keras.layers.Dense(64, activation='relu', name='proj_hidden')(z))
    return ArchPocCCGDSC1Model(inputs=inputs, outputs=[cls_out, reg_out, proj_norm])


def _build_ccgd_poc_sc2(T, n_classes):
    inputs                   = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    cnn_emb, cnngru_emb, z   = _build_cnn_cnngru_dual_branches(inputs, return_branches=True)
    cls_out, reg_out         = _heads(z, n_classes)
    cnn_proj = _proj_head(cnn_emb,    'cnn')
    seq_proj = _proj_head(cnngru_emb, 'seq')
    return ArchPocCCGDSC2Model(inputs=inputs, outputs=[cls_out, reg_out, cnn_proj, seq_proj])


def _build_ccgd_poc_sc3(T, n_classes):
    inputs                 = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    cnn_emb, cnngru_emb, z = _build_cnn_cnngru_dual_branches(inputs, return_branches=True)
    cls_out, reg_out       = _heads(z, n_classes)
    cnn_proj   = _proj_head(cnn_emb,    'cnn')
    seq_proj   = _proj_head(cnngru_emb, 'seq')
    fused_proj = _proj_head(z,          'fused')
    return ArchPocCCGDSC3Model(inputs=inputs, outputs=[cls_out, reg_out, cnn_proj, seq_proj, fused_proj])


# ======================================================================
# KEY LISTS + FACTORY MAP
# ======================================================================

CGD_KEYS  = ['cgd_arch_poc',  'cgd_arch_poc_sc1',  'cgd_arch_poc_sc2',  'cgd_arch_poc_sc3']
CGS_KEYS  = ['cgs_arch_poc',  'cgs_arch_poc_sc1',  'cgs_arch_poc_sc2',  'cgs_arch_poc_sc3']
CCGD_KEYS = ['ccgd_arch_poc', 'ccgd_arch_poc_sc1', 'ccgd_arch_poc_sc2', 'ccgd_arch_poc_sc3']
ALL_ARCH_POC_KEYS = CGD_KEYS + CGS_KEYS + CCGD_KEYS  # 12 total

_FACTORIES = {
    'cgd_arch_poc':       _build_cgd_poc,
    'cgd_arch_poc_sc1':   _build_cgd_poc_sc1,
    'cgd_arch_poc_sc2':   _build_cgd_poc_sc2,
    'cgd_arch_poc_sc3':   _build_cgd_poc_sc3,
    'cgs_arch_poc':       _build_cgs_poc,
    'cgs_arch_poc_sc1':   _build_cgs_poc_sc1,
    'cgs_arch_poc_sc2':   _build_cgs_poc_sc2,
    'cgs_arch_poc_sc3':   _build_cgs_poc_sc3,
    'ccgd_arch_poc':      _build_ccgd_poc,
    'ccgd_arch_poc_sc1':  _build_ccgd_poc_sc1,
    'ccgd_arch_poc_sc2':  _build_ccgd_poc_sc2,
    'ccgd_arch_poc_sc3':  _build_ccgd_poc_sc3,
}


# ======================================================================
# KEY / PRINT MAPS
# ======================================================================

def _mk(key):
    return (
        f'y_preds_AC_{key}_', f'y_probs_AC_{key}_', f'classes_AC_{key}_',
        f'y_reg_preds_{key}_', f'y_reg_trues_{key}_',
    )


_ARCH_POC_KEY_MAP = {m: _mk(m) for m in ALL_ARCH_POC_KEYS}

_ARCH_POC_PRINT_MAP = {
    'cgd_arch_poc':       'CGD (Arch-PoC)',
    'cgd_arch_poc_sc1':   'CGD SC1 (Arch-PoC)',
    'cgd_arch_poc_sc2':   'CGD SC2 (Arch-PoC)',
    'cgd_arch_poc_sc3':   'CGD SC3 (Arch-PoC)',
    'cgs_arch_poc':       'CGS Serial (Arch-PoC)',
    'cgs_arch_poc_sc1':   'CGS Serial SC1 (Arch-PoC)',
    'cgs_arch_poc_sc2':   'CGS Serial SC2 (Arch-PoC)',
    'cgs_arch_poc_sc3':   'CGS Serial SC3 (Arch-PoC)',
    'ccgd_arch_poc':      'CCGD Mixed (Arch-PoC)',
    'ccgd_arch_poc_sc1':  'CCGD Mixed SC1 (Arch-PoC)',
    'ccgd_arch_poc_sc2':  'CCGD Mixed SC2 (Arch-PoC)',
    'ccgd_arch_poc_sc3':  'CCGD Mixed SC3 (Arch-PoC)',
}

_ALL_ARCH_POC_RESULT_KEYS = set()
for _, (_pk, _prk, _ck, _rpk, _rtk) in _ARCH_POC_KEY_MAP.items():
    _ALL_ARCH_POC_RESULT_KEYS.update([_pk, _prk, _ck, _rpk, _rtk])


# ======================================================================
# DATA HELPERS  (mirrors 03e)
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
        _arr   = np.asarray(raw, dtype=object)
        _float = np.array(
            [float(v) if v is not None else np.nan for v in _arr], dtype=float)
        y_conc = np.where(np.isnan(_float) | (_float == 0.0), REG_SENTINEL, _float)
    else:
        y_conc = np.full(n_samples, REG_SENTINEL, dtype=float)
        print('  [!] No concentration in training data — sentinel for all samples.')
    n_valid = int((y_conc != REG_SENTINEL).sum())
    print(f'  [03f] Concentration: {n_valid}/{len(y_conc)} valid non-sentinel samples.')
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
    n_cls     = len(unique_cls)
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
        c_tr  = conc_train_scaled[tr_sub]
        c_val = conc_train_scaled[val_sub]
        val_data  = (X_val, {'cls_out': y_val, 'reg_out': c_val})
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

def run_03f_training(
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
                    if rk in _ALL_ARCH_POC_RESULT_KEYS:
                        del re[rk]
                        n_cleared += 1
        if n_cleared:
            print(f'  -> [FORCE RERUN] Cleared {n_cleared} arch_poc cached key(s).')

    f            = None
    filter_name  = 'None (Baseline)'
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

    for m in ALL_ARCH_POC_KEYS:
        preds_key, probs_key, classes_key, reg_preds_key, reg_trues_key = _ARCH_POC_KEY_MAP[m]

        if preds_key in res_entry:
            rstr = _rmse_str(res_entry.get(reg_trues_key, []),
                             res_entry.get(reg_preds_key, []))
            cached_accs = [accuracy_score(yt, yp)
                           for yt, yp in zip(res_entry['y_trues_'], res_entry[preds_key])]
            print(f'     [CACHE] {_ARCH_POC_PRINT_MAP[m]:<30} | '
                  f'acc={np.mean(cached_accs)*100:.2f}%±{np.std(cached_accs)*100:.2f}% | '
                  f'{rstr}')
            continue

        preds_folds, probs_folds, classes_folds = [], [], []
        reg_preds_folds, reg_trues_folds = [], []
        t0 = time.perf_counter()

        for fold_idx, (tr_idx, te_idx) in enumerate(splits):
            X_tr, y_tr, _, fit_y, val_data, callbacks, scaler = _fold_data(
                X_f, y_f, conc_f, tr_idx)

            tf.keras.backend.clear_session()
            model = _FACTORIES[m](T, n_cls)

            # Guard 1: verify multi-output before any training.
            # Keras 3 can unwrap a dict label {'cls_out': y} to a plain tensor when
            # the model has a single output, making y_dict['cls_out'] silently wrong
            # (tensor indexing instead of key lookup → zero / wrong gradients).
            # All 12 arch_poc models must return ≥2 outputs [cls_out, reg_out, ...].
            _probe = model(X_tr[:1], training=False)
            if not isinstance(_probe, (list, tuple)) or len(_probe) < 2:
                raise RuntimeError(
                    f'{m}: expected ≥2 outputs [cls_out, reg_out, ...], '
                    f'got {type(_probe).__name__} len={getattr(_probe, "__len__", lambda: "?")()}. '
                    f'Keras 3 dict-label stripping would produce zero gradients.')

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

            # All 12 models: [cls_out, reg_out, ...] → cls at [0], reg at [1]
            raw_out = model.predict(X_f[te_idx], verbose=0)

            # Guard 2: verify predict returns a list, not a single tensor.
            # If Keras collapsed the model to single output, raw_out[0] would be
            # the first SAMPLE's predictions rather than the first OUTPUT's array.
            if not isinstance(raw_out, (list, tuple)):
                raise RuntimeError(
                    f'{m}: model.predict() returned {type(raw_out).__name__} '
                    f'instead of list. Output 0 would be a sample, not cls_out.')

            cls_probs = raw_out[0]
            pred      = np.argmax(cls_probs, axis=1)

            preds_folds.append(pred)
            probs_folds.append(cls_probs)
            classes_folds.append(encoder_classes)

            reg_scaled = raw_out[1]
            reg_orig   = _inverse_normalize_concentration(reg_scaled[:, 0], scaler)
            reg_preds_folds.append(reg_orig)
            reg_trues_folds.append(conc_f[te_idx])

            gc.collect()

        duration = time.perf_counter() - t0
        accs = [accuracy_score(yt, yp)
                for yt, yp in zip(res_entry['y_trues_'], preds_folds)]
        rstr = _rmse_str(reg_trues_folds, reg_preds_folds)

        res_entry[preds_key]     = preds_folds
        res_entry[probs_key]     = probs_folds
        res_entry[classes_key]   = classes_folds
        res_entry[reg_preds_key] = reg_preds_folds
        res_entry[reg_trues_key] = reg_trues_folds

        results[f] = res_entry
        all_ml_results[clean_title][mode_key] = results
        safe_joblib_dump(all_ml_results, results_file_path, compress=3)

        print(f'     [+] {_ARCH_POC_PRINT_MAP[m]:<30} | '
              f'acc={np.mean(accs)*100:.2f}%±{np.std(accs)*100:.2f}% | '
              f'{rstr} | {_fmt_hms(duration)}')

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
            if not (key.startswith('y_preds_AC_') and 'arch_poc' in key):
                continue
            try:
                accs = [accuracy_score(yt, yp)
                        for yt, yp in zip(res['y_trues_'], val)]
                lbl = _ARCH_POC_PRINT_MAP.get(
                    key.replace('y_preds_AC_', '').rstrip('_'), key)
                rows.append((np.mean(accs) * 100, np.std(accs) * 100, lbl, filter_name))
            except Exception:
                continue
    rows.sort(key=lambda x: x[0], reverse=True)
    W = 95
    print(f'\n  Leaderboard [Arch-PoC] {clean_title} / {mode_key}')
    print('  ' + '-' * W)
    for i, (acc, std, lbl, filt) in enumerate(rows):
        print(f'  {i+1:3d}.  {acc:6.2f}% +/- {std:5.2f}%'
              f'  |  {lbl:<36}  |  Filter: {filt}')
    print('  ' + '-' * W + '\n')


# ======================================================================
# MAIN
# ======================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='03f: Arch-PoC 12-model training')
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
    print(f'[RUNNING] 03f_arch_poc_training.py  [Arch-PoC 12-model]')
    print(f"{'='*70}\n")
    print(f"{'#'*80}\nSTARTING TRAINING FOR: {exp_path.name}\n{'#'*80}")

    arch_poc_dir   = Path(exp_path) / 'arch_poc'
    model_save_dir = arch_poc_dir / 'models'
    model_save_dir.mkdir(parents=True, exist_ok=True)

    joblib_name       = ('arch_poc_results_5fold.joblib' if args.n_splits > 1
                         else 'arch_poc_results.joblib')
    results_file_path = str(arch_poc_dir / joblib_name)

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

        results = run_03f_training(
            X_curves          = X_curves,
            features_df       = features_df,
            y_encoded         = y_full,
            y_concentration   = y_concentration,
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
