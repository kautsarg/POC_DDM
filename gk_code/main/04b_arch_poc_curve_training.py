"""Arch-PoC (04b): 12 ST-LOFO models with LOFO cross-dataset CV (04 mechanism).

Own model classes / Keras packages / result keys (suffix _lofo) — distinct from
03g's _st keys so per-dataset and LOFO results can coexist without collision.

LOFO structure (mirrors 04_cross_dataset_training.py):
  task_id       → config.CROSS_DATASET_GROUPS group index
  curve_type_id → 0=ori_curve · 1=ori_curve_avg · 2=ori_curve_wavelet_sym8 · 3=ori_curve_wavelet_bior35 · 4=ori_curve_sg_p4

All folders in the group are combined + resampled (CurveResampler), then each
folder is left out in turn as the test set.  Single-task classification only —
no concentration, no regression head.

Result file:
  {exp_folder}/cross_dataset_cv/{group_name}/arch_poc_st_lofo_{curve_type}.joblib
  Structure: {fold_label: {None: {y_trues_, y_preds_AC_..._lofo_, ...}}}

Usage:
  python 04b_arch_poc_curve_training.py --task_id 0 --curve_type_id 0
  # SLURM: sbatch --export=TASK_ID=0 lab_arch_poc_curve_training.sh
"""
import os
import sys
import gc
import time
import argparse
import importlib.util
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

_THIS     = Path(__file__).resolve()
_MAIN_DIR = _THIS.parent
sys.path.insert(0, str(_MAIN_DIR))
sys.path.insert(0, str(_MAIN_DIR / "utils"))
sys.path.insert(0, str(_MAIN_DIR / "utils" / "model_training"))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# ── Load 03g via importlib — backbone builders + _fold_data + _fmt_hms ───────
_spec_03g = importlib.util.spec_from_file_location(
    '_03g_arch_poc_st', _MAIN_DIR / '03g_arch_poc_st_training.py')
_03g = importlib.util.module_from_spec(_spec_03g)
_spec_03g.loader.exec_module(_03g)

# ── Load 04 via importlib — LOFO helpers ─────────────────────────────────────
_spec_04 = importlib.util.spec_from_file_location(
    '_04_cross_dataset', _MAIN_DIR / '04_cross_dataset_training.py')
_04 = importlib.util.module_from_spec(_spec_04)
_spec_04.loader.exec_module(_04)

from safe_io import safe_joblib_dump, safe_keras_save
from model_utils import set_global_determinism
from model_utils_supcon import SupConModel, SupConBranch2STModel, SupConBranch3STModel, _proj_head
import joblib
import config

CURVE_TYPES = ['ori_curve', 'ori_curve_avg', 'ori_curve_wavelet_sym8',
               'ori_curve_wavelet_bior35', 'ori_curve_sg_p4']


# ======================================================================
# MODEL CLASSES — 12 unique Keras packages (_lofo suffix)
# ======================================================================

# --- CGD ---
@tf.keras.utils.register_keras_serializable(package='arch_poc_st_lofo_cgd')
class ArchPocSTLofoCGDModel(tf.keras.Model):
    """CGD SC0 ST-LOFO: plain CE via compiled loss."""
    pass

@tf.keras.utils.register_keras_serializable(package='arch_poc_st_lofo_cgd_sc1')
class ArchPocSTLofoCGDSC1Model(SupConModel):
    """CGD SC1 ST-LOFO: CE + SupCon on fused."""
    pass

@tf.keras.utils.register_keras_serializable(package='arch_poc_st_lofo_cgd_sc2')
class ArchPocSTLofoCGDSC2Model(SupConBranch2STModel):
    """CGD SC2 ST-LOFO: CE + SupCon on cnn+seq."""
    pass

@tf.keras.utils.register_keras_serializable(package='arch_poc_st_lofo_cgd_sc3')
class ArchPocSTLofoCGDSC3Model(SupConBranch3STModel):
    """CGD SC3 ST-LOFO: CE + SupCon on cnn+seq+fused."""
    pass

# --- CGS ---
@tf.keras.utils.register_keras_serializable(package='arch_poc_st_lofo_cgs')
class ArchPocSTLofoCGSModel(tf.keras.Model):
    """CGS SC0 ST-LOFO: plain CE via compiled loss."""
    pass

@tf.keras.utils.register_keras_serializable(package='arch_poc_st_lofo_cgs_sc1')
class ArchPocSTLofoCGSSC1Model(SupConModel):
    """CGS SC1 ST-LOFO: CE + SupCon on z."""
    pass

@tf.keras.utils.register_keras_serializable(package='arch_poc_st_lofo_cgs_sc2')
class ArchPocSTLofoCGSSC2Model(SupConBranch2STModel):
    """CGS SC2 ST-LOFO: CE + SupCon on cnn_tap+gru_raw."""
    pass

@tf.keras.utils.register_keras_serializable(package='arch_poc_st_lofo_cgs_sc3')
class ArchPocSTLofoCGSSC3Model(SupConBranch3STModel):
    """CGS SC3 ST-LOFO: CE + SupCon on cnn_tap+gru_raw+z."""
    pass

# --- CCGD ---
@tf.keras.utils.register_keras_serializable(package='arch_poc_st_lofo_ccgd')
class ArchPocSTLofoCCGDModel(tf.keras.Model):
    """CCGD SC0 ST-LOFO: plain CE via compiled loss."""
    pass

@tf.keras.utils.register_keras_serializable(package='arch_poc_st_lofo_ccgd_sc1')
class ArchPocSTLofoCCGDSC1Model(SupConModel):
    """CCGD SC1 ST-LOFO: CE + SupCon on fused."""
    pass

@tf.keras.utils.register_keras_serializable(package='arch_poc_st_lofo_ccgd_sc2')
class ArchPocSTLofoCCGDSC2Model(SupConBranch2STModel):
    """CCGD SC2 ST-LOFO: CE + SupCon on cnn+cnngru."""
    pass

@tf.keras.utils.register_keras_serializable(package='arch_poc_st_lofo_ccgd_sc3')
class ArchPocSTLofoCCGDSC3Model(SupConBranch3STModel):
    """CCGD SC3 ST-LOFO: CE + SupCon on cnn+cnngru+fused."""
    pass


# ======================================================================
# BUILD FUNCTIONS  (reuse backbone builders from 03g)
# ======================================================================

def _heads(z, n_classes):
    return tf.keras.layers.Dense(
        n_classes, activation='softmax', name='cls_out')(
        tf.keras.layers.Dense(16, activation='relu', name='cls_feat')(z))


# CGD
def _build_cgd_lofo(T, n):
    inp = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    return ArchPocSTLofoCGDModel(inputs=inp, outputs=_heads(_03g._build_cnn_gru_dual_branches_mtl(inp), n))

def _build_cgd_lofo_sc1(T, n):
    inp = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z   = _03g._build_cnn_gru_dual_branches_mtl(inp)
    proj = tf.keras.layers.UnitNormalization(axis=1, name='proj')(
           tf.keras.layers.Dense(64, activation='relu', name='proj_hidden')(z))
    return ArchPocSTLofoCGDSC1Model(inputs=inp, outputs=[_heads(z, n), proj])

def _build_cgd_lofo_sc2(T, n):
    inp = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    ce, ge, z = _03g._build_cnn_gru_dual_branches_mtl(inp, return_branches=True)
    return ArchPocSTLofoCGDSC2Model(
        inputs=inp, outputs=[_heads(z, n), _proj_head(ce, 'cnn'), _proj_head(ge, 'seq')])

def _build_cgd_lofo_sc3(T, n):
    inp = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    ce, ge, z = _03g._build_cnn_gru_dual_branches_mtl(inp, return_branches=True)
    return ArchPocSTLofoCGDSC3Model(
        inputs=inp, outputs=[_heads(z, n), _proj_head(ce, 'cnn'), _proj_head(ge, 'seq'), _proj_head(z, 'fused')])

# CGS
def _build_cgs_lofo(T, n):
    inp = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    return ArchPocSTLofoCGSModel(inputs=inp, outputs=_heads(_03g._build_cnn_gru_serial_branches(inp), n))

def _build_cgs_lofo_sc1(T, n):
    inp = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z   = _03g._build_cnn_gru_serial_branches(inp)
    proj = tf.keras.layers.UnitNormalization(axis=1, name='proj')(
           tf.keras.layers.Dense(64, activation='relu', name='proj_hidden')(z))
    return ArchPocSTLofoCGSSC1Model(inputs=inp, outputs=[_heads(z, n), proj])

def _build_cgs_lofo_sc2(T, n):
    inp = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    ct, gr, z = _03g._build_cnn_gru_serial_branches(inp, return_branches=True)
    return ArchPocSTLofoCGSSC2Model(
        inputs=inp, outputs=[_heads(z, n), _proj_head(ct, 'cnn'), _proj_head(gr, 'seq')])

def _build_cgs_lofo_sc3(T, n):
    inp = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    ct, gr, z = _03g._build_cnn_gru_serial_branches(inp, return_branches=True)
    return ArchPocSTLofoCGSSC3Model(
        inputs=inp, outputs=[_heads(z, n), _proj_head(ct, 'cnn'), _proj_head(gr, 'seq'), _proj_head(z, 'fused')])

# CCGD
def _build_ccgd_lofo(T, n):
    inp = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    return ArchPocSTLofoCCGDModel(inputs=inp, outputs=_heads(_03g._build_cnn_cnngru_dual_branches(inp), n))

def _build_ccgd_lofo_sc1(T, n):
    inp = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    z   = _03g._build_cnn_cnngru_dual_branches(inp)
    proj = tf.keras.layers.UnitNormalization(axis=1, name='proj')(
           tf.keras.layers.Dense(64, activation='relu', name='proj_hidden')(z))
    return ArchPocSTLofoCCGDSC1Model(inputs=inp, outputs=[_heads(z, n), proj])

def _build_ccgd_lofo_sc2(T, n):
    inp = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    ce, cge, z = _03g._build_cnn_cnngru_dual_branches(inp, return_branches=True)
    return ArchPocSTLofoCCGDSC2Model(
        inputs=inp, outputs=[_heads(z, n), _proj_head(ce, 'cnn'), _proj_head(cge, 'seq')])

def _build_ccgd_lofo_sc3(T, n):
    inp = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    ce, cge, z = _03g._build_cnn_cnngru_dual_branches(inp, return_branches=True)
    return ArchPocSTLofoCCGDSC3Model(
        inputs=inp, outputs=[_heads(z, n), _proj_head(ce, 'cnn'), _proj_head(cge, 'seq'), _proj_head(z, 'fused')])


# ======================================================================
# KEY LISTS + MAPS
# ======================================================================

CGD_LOFO_KEYS  = ['cgd_arch_poc_st_lofo',  'cgd_arch_poc_st_lofo_sc1',
                   'cgd_arch_poc_st_lofo_sc2', 'cgd_arch_poc_st_lofo_sc3']
CGS_LOFO_KEYS  = ['cgs_arch_poc_st_lofo',  'cgs_arch_poc_st_lofo_sc1',
                   'cgs_arch_poc_st_lofo_sc2', 'cgs_arch_poc_st_lofo_sc3']
CCGD_LOFO_KEYS = ['ccgd_arch_poc_st_lofo', 'ccgd_arch_poc_st_lofo_sc1',
                   'ccgd_arch_poc_st_lofo_sc2', 'ccgd_arch_poc_st_lofo_sc3']
ALL_LOFO_KEYS  = CGD_LOFO_KEYS + CGS_LOFO_KEYS + CCGD_LOFO_KEYS

SC0_LOFO_KEYS  = {CGD_LOFO_KEYS[0], CGS_LOFO_KEYS[0], CCGD_LOFO_KEYS[0]}

_FACTORIES = {
    'cgd_arch_poc_st_lofo':       _build_cgd_lofo,
    'cgd_arch_poc_st_lofo_sc1':   _build_cgd_lofo_sc1,
    'cgd_arch_poc_st_lofo_sc2':   _build_cgd_lofo_sc2,
    'cgd_arch_poc_st_lofo_sc3':   _build_cgd_lofo_sc3,
    'cgs_arch_poc_st_lofo':       _build_cgs_lofo,
    'cgs_arch_poc_st_lofo_sc1':   _build_cgs_lofo_sc1,
    'cgs_arch_poc_st_lofo_sc2':   _build_cgs_lofo_sc2,
    'cgs_arch_poc_st_lofo_sc3':   _build_cgs_lofo_sc3,
    'ccgd_arch_poc_st_lofo':      _build_ccgd_lofo,
    'ccgd_arch_poc_st_lofo_sc1':  _build_ccgd_lofo_sc1,
    'ccgd_arch_poc_st_lofo_sc2':  _build_ccgd_lofo_sc2,
    'ccgd_arch_poc_st_lofo_sc3':  _build_ccgd_lofo_sc3,
}

_KEY_MAP = {m: (f'y_preds_AC_{m}_', f'y_probs_AC_{m}_', f'classes_AC_{m}_')
            for m in ALL_LOFO_KEYS}

_PRINT_MAP = {
    'cgd_arch_poc_st_lofo':       'CGD ST-LOFO (Arch-PoC)',
    'cgd_arch_poc_st_lofo_sc1':   'CGD SC1 ST-LOFO (Arch-PoC)',
    'cgd_arch_poc_st_lofo_sc2':   'CGD SC2 ST-LOFO (Arch-PoC)',
    'cgd_arch_poc_st_lofo_sc3':   'CGD SC3 ST-LOFO (Arch-PoC)',
    'cgs_arch_poc_st_lofo':       'CGS Serial ST-LOFO (Arch-PoC)',
    'cgs_arch_poc_st_lofo_sc1':   'CGS Serial SC1 ST-LOFO (Arch-PoC)',
    'cgs_arch_poc_st_lofo_sc2':   'CGS Serial SC2 ST-LOFO (Arch-PoC)',
    'cgs_arch_poc_st_lofo_sc3':   'CGS Serial SC3 ST-LOFO (Arch-PoC)',
    'ccgd_arch_poc_st_lofo':      'CCGD Mixed ST-LOFO (Arch-PoC)',
    'ccgd_arch_poc_st_lofo_sc1':  'CCGD Mixed SC1 ST-LOFO (Arch-PoC)',
    'ccgd_arch_poc_st_lofo_sc2':  'CCGD Mixed SC2 ST-LOFO (Arch-PoC)',
    'ccgd_arch_poc_st_lofo_sc3':  'CCGD Mixed SC3 ST-LOFO (Arch-PoC)',
}

_ALL_RESULT_KEYS = {k for pk, prk, ck in _KEY_MAP.values() for k in (pk, prk, ck)}


# ======================================================================
# LOFO FOLD TRAINING LOOP
# ======================================================================

def _run_lofo_fold(
    X_curves, y_full, encoder_classes,
    train_idx, test_idx,
    cached_fold, force_rerun,
    lofo_results, fold_label, checkpoint_fn,
    model_save_dir, curve_type,
):
    """Train all 12 ST-LOFO arch-poc models on one LOFO fold."""
    y_tr_all = y_full[train_idx]
    unique_cls, cls_counts = np.unique(y_tr_all, return_counts=True)
    rare = unique_cls[cls_counts < 2]
    if len(rare) > 0:
        train_idx = train_idx[~np.isin(y_tr_all, rare)]
        test_idx  = test_idx[~np.isin(y_full[test_idx], rare)]
        unique_cls, cls_counts = np.unique(y_full[train_idx], return_counts=True)

    n_cls = len(unique_cls)
    if n_cls < 2 or len(train_idx) < 2 * n_cls:
        print("     [Warning] Insufficient classes or train samples. Skipping fold.")
        return

    X_f = np.nan_to_num(X_curves, nan=0.0, posinf=0.0, neginf=0.0)
    T   = X_f.shape[1]

    res_entry = (cached_fold or {}).get(None, {})

    if force_rerun:
        cleared = [res_entry.pop(k) for k in list(res_entry) if k in _ALL_RESULT_KEYS]
        if cleared:
            print(f"     [FORCE RERUN] Cleared {len(cleared)} cached key(s).")

    if 'y_trues_' not in res_entry:
        res_entry['y_trues_'] = [y_full[test_idx]]

    for m in ALL_LOFO_KEYS:
        preds_key, probs_key, classes_key = _KEY_MAP[m]
        is_sc0 = m in SC0_LOFO_KEYS

        if preds_key in res_entry:
            acc = accuracy_score(res_entry['y_trues_'][0], res_entry[preds_key][0])
            print(f"     [CACHE] {_PRINT_MAP[m]:<40} | acc={acc*100:.2f}%")
            continue

        t0 = time.perf_counter()
        X_tr, y_tr, fit_y, val_data, callbacks = _03g._fold_data(
            X_f, y_full, train_idx, is_sc0)

        tf.keras.backend.clear_session()
        model = _FACTORIES[m](T, n_cls)

        _probe = model(X_tr[:1], training=False)
        if is_sc0:
            if not isinstance(_probe, tf.Tensor):
                raise RuntimeError(f'{m}: SC0 expected single Tensor, got {type(_probe).__name__}.')
        else:
            if not isinstance(_probe, (list, tuple)) or len(_probe) < 2:
                raise RuntimeError(f'{m}: expected ≥2 outputs, got {type(_probe).__name__}.')

        if is_sc0:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
                loss='sparse_categorical_crossentropy',
                jit_compile=False)
        else:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
                jit_compile=False)

        _hist = model.fit(X_tr, fit_y,
                  validation_data=val_data, callbacks=callbacks,
                  epochs=500, batch_size=512, shuffle=True, verbose=0)

        safe_keras_save(model, model_save_dir / f'{m}_{curve_type}_model.keras')

        if is_sc0:
            cls_probs = model.predict(X_f[test_idx], verbose=0)
        else:
            raw_out = model.predict(X_f[test_idx], verbose=0)
            if not isinstance(raw_out, (list, tuple)):
                raise RuntimeError(f'{m}: predict() returned {type(raw_out).__name__} instead of list.')
            cls_probs = raw_out[0]

        pred = np.argmax(cls_probs, axis=1)
        acc  = accuracy_score(y_full[test_idx], pred)

        res_entry[preds_key]   = [pred]
        res_entry[probs_key]   = [cls_probs]
        res_entry[classes_key] = [encoder_classes]
        res_entry['train_history'] = [_hist.history]

        lofo_results[fold_label] = {None: res_entry}
        checkpoint_fn(lofo_results[fold_label])

        print(f"     [+] {_PRINT_MAP[m]:<40} | "
              f"acc={acc*100:.2f}% | {_03g._fmt_hms(time.perf_counter() - t0)}")
        gc.collect()

    lofo_results[fold_label] = {None: res_entry}


# ======================================================================
# LEADERBOARD
# ======================================================================

def _print_leaderboard(lofo_results, group_name):
    from collections import defaultdict
    model_accs = defaultdict(list)
    for fold_label, fold_data in lofo_results.items():
        if not isinstance(fold_data, dict):
            continue
        entry = fold_data.get(None, {})
        if 'y_trues_' not in entry:
            continue
        for key, val in entry.items():
            if not key.startswith('y_preds_AC_'):
                continue
            try:
                acc = accuracy_score(entry['y_trues_'][0], val[0])
                lbl = _PRINT_MAP.get(key.replace('y_preds_AC_', '').rstrip('_'), key)
                model_accs[lbl].append(acc * 100)
            except Exception:
                continue
    if not model_accs:
        return
    summary = sorted([(np.mean(v), np.std(v), k) for k, v in model_accs.items()],
                     reverse=True)
    W = 85
    print(f'\n  Leaderboard [Arch-PoC ST-LOFO] {group_name}  (mean ± std across folds)')
    print('  ' + '-' * W)
    for i, (mean_acc, std_acc, lbl) in enumerate(summary):
        print(f'  {i+1:3d}.  {mean_acc:6.2f}% +/- {std_acc:5.2f}%  |  {lbl}')
    print('  ' + '-' * W + '\n')


# ======================================================================
# MAIN
# ======================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='04b: Arch-PoC ST-LOFO cross-dataset training')
    parser.add_argument('--task_id',       type=int, default=0,
                        help='Index into config.CROSS_DATASET_GROUPS')
    parser.add_argument('--exp_folder',    type=str,
                        default='/vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_multi_nc_subtract')
    parser.add_argument('--curve_type_id', type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help='0=ori_curve  1=ori_curve_avg  2=ori_curve_wavelet_sym8  '
                             '3=ori_curve_wavelet_bior35  4=ori_curve_sg_p4')
    parser.add_argument('--force_rerun',   action='store_true')
    args = parser.parse_args()

    curve_type = CURVE_TYPES[args.curve_type_id]

    set_global_determinism(0)

    group_names = list(config.CROSS_DATASET_GROUPS.keys())
    if args.task_id >= len(group_names):
        print(f"task_id {args.task_id} out of bounds ({len(group_names)} groups). Exiting.")
        sys.exit(0)

    group_name   = group_names[args.task_id]
    folder_names = config.CROSS_DATASET_GROUPS[group_name]
    exp_paths    = [Path(args.exp_folder) / name for name in folder_names]

    print(f"\n{'='*70}")
    print(f'[RUNNING] 04b_arch_poc_curve_training.py  [Arch-PoC ST-LOFO]')
    print(f"{'='*70}\n")
    print(f"{'#'*80}\nGROUP: {group_name}  |  curve_type: {curve_type}"
          f"\nFolders: {folder_names}\n{'#'*80}")

    combined = _04.combine_group(exp_paths, group_name, curve_type=curve_type)
    if combined is None:
        print("combine_group returned None. Exiting.")
        sys.exit(0)

    out_dir = Path(args.exp_folder) / "cross_dataset_cv" / group_name
    out_dir.mkdir(parents=True, exist_ok=True)

    resampler_path = out_dir / config.CROSS_DATASET_RESAMPLER_PATH.format(curve_type=curve_type)
    safe_joblib_dump(combined["resampler"], resampler_path, compress=3)
    print(f"  [*] Saved curve resampler -> {resampler_path}")

    results_file_path = out_dir / f'arch_poc_st_lofo_{curve_type}.joblib'

    # (n_samples, T) → (n_samples, T, 1) for CNN input
    X_curves = combined["curves"][:, :, np.newaxis].astype(np.float32)

    encoder = LabelEncoder()
    y_full  = encoder.fit_transform(combined["Y_mapped"])

    lofo_splits = _04.build_lofo_splits(combined["dataset_id"])
    total_folds = len(lofo_splits)

    lofo_results = joblib.load(results_file_path) if results_file_path.exists() else {}

    for fold_idx, (fold_label, (train_idx, test_idx)) in enumerate(
            reversed(list(lofo_splits.items()))):
        progress_pct = (fold_idx + 1) / total_folds * 100
        print(f"\n{'='*75}")
        print(f"[{fold_idx+1}/{total_folds} | {progress_pct:.1f}%] "
              f"FOLD: {fold_label}  train={len(train_idx)}  test={len(test_idx)}")
        print(f"{'='*75}")

        model_save_dir = out_dir / "arch_poc_st_lofo_models" / fold_label
        model_save_dir.mkdir(parents=True, exist_ok=True)

        def checkpoint(fold_data, _fold_label=fold_label):
            lofo_results[_fold_label] = fold_data
            safe_joblib_dump(lofo_results, results_file_path, compress=3)

        _run_lofo_fold(
            X_curves       = X_curves,
            y_full         = y_full,
            encoder_classes= encoder.classes_,
            train_idx      = train_idx,
            test_idx       = test_idx,
            cached_fold    = lofo_results.get(fold_label),
            force_rerun    = args.force_rerun,
            lofo_results   = lofo_results,
            fold_label     = fold_label,
            checkpoint_fn  = checkpoint,
            model_save_dir = model_save_dir,
            curve_type     = curve_type,
        )

        if fold_label in lofo_results:  # guard: fold not skipped
            lofo_results[fold_label]["class_names"] = [str(c) for c in encoder.classes_]
            safe_joblib_dump(lofo_results, results_file_path, compress=3)
        gc.collect()

    print(f"\n  -> Final results saved to {results_file_path}")
    _print_leaderboard(lofo_results, group_name)
