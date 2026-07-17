"""CondReg Oracle — GT-FiLM concept proof (CGD SC0-3).

Architecture identical to RCFD (same CNN+GRU dual backbone, same FiLM, same cls head),
but the early encoder is removed and replaced by a second Keras Input receiving
ground-truth concentration directly. SC0-3 add SupCon auxiliary losses matching
the RCFD SC variants.

SC0: plain CE                          (no SupCon)
SC1: CE + SupCon on z_cond             (post-FiLM fused projection)
SC2: CE + SupCon on CNN + GRU branches (pre-FiLM branch projections)
SC3: CE + SupCon on branches + z_cond  (branches + post-FiLM fused)

Results append as new keys to the existing classification_performances[_10fold].joblib.
Nothing in that file is changed except the addition of the new gt_film_cgd_sc* keys.

Usage:
    python 03c_condreg_oracle_poc.py --task_id 0 --n_splits 5 --mode native
"""
import os
import sys
import gc
import argparse
import time
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score

sys.path.insert(0, 'utils')
sys.path.insert(0, 'utils/model_training')

from safe_io import safe_joblib_dump
from pipeline_utils import get_exp_paths, check_task_id
from model_utils import set_global_determinism
from model_utils_mtl import (
    REG_SENTINEL,
    _normalize_concentration,
    _build_cnn_gru_dual_branches_mtl,
)
from model_utils_rcfd import _apply_film_scalar, _cls_head
from model_utils_supcon import (
    SupConModel,
    SupConBranch2STModel,
    SupConBranch3STModel,
    _proj_head,
)

import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


# ============================================================
# GT-FiLM CGD MODEL BUILDERS
# ============================================================

def _build_gt_film_cgd_sc0(T, n_classes):
    """SC0: plain CE, no SupCon."""
    curve_input = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    conc_input  = tf.keras.layers.Input(shape=(1,),   name='conc_input')
    z_raw  = _build_cnn_gru_dual_branches_mtl(curve_input)
    z_cond = tf.keras.layers.Dropout(0.2, name='film_cgd_drop')(
        _apply_film_scalar(conc_input, z_raw, emb_dim=96, name_prefix='film_cgd'))
    cls_out = _cls_head(z_cond, n_classes)
    model = tf.keras.Model(inputs=[curve_input, conc_input], outputs=cls_out,
                           name='gt_film_cgd_sc0')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model


def _build_gt_film_cgd_sc1(T, n_classes):
    """SC1: CE + SupCon on z_cond (post-FiLM fused)."""
    curve_input = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    conc_input  = tf.keras.layers.Input(shape=(1,),   name='conc_input')
    z_raw     = _build_cnn_gru_dual_branches_mtl(curve_input)
    z_cond    = tf.keras.layers.Dropout(0.2, name='film_cgd_drop')(
        _apply_film_scalar(conc_input, z_raw, emb_dim=96, name_prefix='film_cgd'))
    cls_out   = _cls_head(z_cond, n_classes)
    proj_norm = _proj_head(z_cond, 'fused')
    model = SupConModel(inputs=[curve_input, conc_input], outputs=[cls_out, proj_norm],
                        name='gt_film_cgd_sc1')
    model.compile(optimizer='adam', metrics=['accuracy'])
    return model


def _build_gt_film_cgd_sc2(T, n_classes):
    """SC2: CE + SupCon on CNN + GRU branches (pre-FiLM)."""
    curve_input = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    conc_input  = tf.keras.layers.Input(shape=(1,),   name='conc_input')
    cnn_emb, gru_emb, z_raw = _build_cnn_gru_dual_branches_mtl(
        curve_input, return_branches=True)
    z_cond   = tf.keras.layers.Dropout(0.2, name='film_cgd_drop')(
        _apply_film_scalar(conc_input, z_raw, emb_dim=96, name_prefix='film_cgd'))
    cls_out  = _cls_head(z_cond, n_classes)
    cnn_proj = _proj_head(cnn_emb, 'cnn')
    gru_proj = _proj_head(gru_emb, 'seq')
    model = SupConBranch2STModel(
        inputs=[curve_input, conc_input], outputs=[cls_out, cnn_proj, gru_proj],
        name='gt_film_cgd_sc2')
    model.compile(optimizer='adam', metrics=['accuracy'])
    return model


def _build_gt_film_cgd_sc3(T, n_classes):
    """SC3: CE + SupCon on CNN + GRU branches + z_cond."""
    curve_input = tf.keras.layers.Input(shape=(T, 1), name='curve_input')
    conc_input  = tf.keras.layers.Input(shape=(1,),   name='conc_input')
    cnn_emb, gru_emb, z_raw = _build_cnn_gru_dual_branches_mtl(
        curve_input, return_branches=True)
    z_cond     = tf.keras.layers.Dropout(0.2, name='film_cgd_drop')(
        _apply_film_scalar(conc_input, z_raw, emb_dim=96, name_prefix='film_cgd'))
    cls_out    = _cls_head(z_cond, n_classes)
    cnn_proj   = _proj_head(cnn_emb, 'cnn')
    gru_proj   = _proj_head(gru_emb, 'seq')
    fused_proj = _proj_head(z_cond,  'fused')
    model = SupConBranch3STModel(
        inputs=[curve_input, conc_input],
        outputs=[cls_out, cnn_proj, gru_proj, fused_proj],
        name='gt_film_cgd_sc3')
    model.compile(optimizer='adam', metrics=['accuracy'])
    return model


# ============================================================
# REGISTRY
# ============================================================

_GT_FILM_MODELS = [
    'gt_film_cgd_sc0',
    'gt_film_cgd_sc1',
    'gt_film_cgd_sc2',
    'gt_film_cgd_sc3',
]

_GT_FILM_BUILDERS = {
    'gt_film_cgd_sc0': _build_gt_film_cgd_sc0,
    'gt_film_cgd_sc1': _build_gt_film_cgd_sc1,
    'gt_film_cgd_sc2': _build_gt_film_cgd_sc2,
    'gt_film_cgd_sc3': _build_gt_film_cgd_sc3,
}

_GT_FILM_KEY_MAP = {
    'gt_film_cgd_sc0': ('y_preds_AC_gt_film_cgd_sc0_', 'y_probs_AC_gt_film_cgd_sc0_', 'classes_AC_gt_film_cgd_sc0_'),
    'gt_film_cgd_sc1': ('y_preds_AC_gt_film_cgd_sc1_', 'y_probs_AC_gt_film_cgd_sc1_', 'classes_AC_gt_film_cgd_sc1_'),
    'gt_film_cgd_sc2': ('y_preds_AC_gt_film_cgd_sc2_', 'y_probs_AC_gt_film_cgd_sc2_', 'classes_AC_gt_film_cgd_sc2_'),
    'gt_film_cgd_sc3': ('y_preds_AC_gt_film_cgd_sc3_', 'y_probs_AC_gt_film_cgd_sc3_', 'classes_AC_gt_film_cgd_sc3_'),
}

_GT_FILM_PRINT_MAP = {
    'gt_film_cgd_sc0': 'GT-FiLM CGD SC0',
    'gt_film_cgd_sc1': 'GT-FiLM CGD SC1',
    'gt_film_cgd_sc2': 'GT-FiLM CGD SC2',
    'gt_film_cgd_sc3': 'GT-FiLM CGD SC3',
}

# SC level per key: 0 = plain CE; 1-3 = SupCon variants with dict y target
_GT_FILM_SC_LEVEL = {
    'gt_film_cgd_sc0': 0,
    'gt_film_cgd_sc1': 1,
    'gt_film_cgd_sc2': 2,
    'gt_film_cgd_sc3': 3,
}

_ALL_GT_FILM_RESULT_KEYS = set()
for _pk, _probk, _clsk in _GT_FILM_KEY_MAP.values():
    _ALL_GT_FILM_RESULT_KEYS.update([_pk, _probk, _clsk])


# ============================================================
# CONCENTRATION HELPERS
# ============================================================

def _apply_conc_scaler(conc_array, scaler):
    """Apply a fitted scaler (from _normalize_concentration) to a new fold array."""
    arr = conc_array.astype(float).copy()
    valid = arr != REG_SENTINEL
    if valid.any() and hasattr(scaler, 'mean_'):
        arr[valid] = scaler.transform(
            np.log10(arr[valid]).reshape(-1, 1)
        ).ravel()
    return arr


# ============================================================
# DATA LOADING HELPERS
# ============================================================

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


def _filter_datasets(dataset_name, dataset, kinetic_features):
    out_names, out_data, out_feat = [], [], []
    for name, data, feat in zip(dataset_name, dataset, kinetic_features):
        if not name.startswith('avg_') and not name.startswith('original_fitted_stretched'):
            out_names.append(name)
            out_data.append(data)
            out_feat.append(feat)
    return out_names, out_data, out_feat


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
        print('  [!] No concentration in training data — all samples will use sentinel FiLM.')
    n_valid = int((y_conc != REG_SENTINEL).sum())
    print(f'  [GT-FiLM] Concentration: {n_valid}/{len(y_conc)} valid non-sentinel samples.')
    return y_conc


# ============================================================
# TRAINING LOOP
# ============================================================

def _fmt_hms(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def run_gt_film_training(
    X_curves, features_df, y_encoded, y_concentration,
    outlier_filters, encoder_classes, n_splits,
    cached_results, force_rerun,
    results_file_path, all_ml_results, clean_title, mode_key,
):
    results = cached_results.copy()

    if force_rerun:
        n_cleared = 0
        for res_entry in results.values():
            if isinstance(res_entry, dict):
                for rk in list(res_entry.keys()):
                    if rk in _ALL_GT_FILM_RESULT_KEYS:
                        del res_entry[rk]
                        n_cleared += 1
        if n_cleared:
            print(f'  -> [FORCE RERUN] Cleared {n_cleared} GT-FiLM cached key(s). '
                  f'Existing RCFD/standard results preserved.')

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

        for m in _GT_FILM_MODELS:
            preds_key, probs_key, classes_key = _GT_FILM_KEY_MAP[m]
            is_supcon = _GT_FILM_SC_LEVEL[m] > 0

            if preds_key in res_entry:
                cached_accs = [accuracy_score(yt, yp)
                               for yt, yp in zip(res_entry['y_trues_'], res_entry[preds_key])]
                print(f'     [CACHE] {_GT_FILM_PRINT_MAP[m]} | '
                      f'acc={np.mean(cached_accs)*100:.2f}%±{np.std(cached_accs)*100:.2f}%')
                continue

            preds_folds, probs_folds, classes_folds = [], [], []
            t0 = time.perf_counter()

            for tr_idx, te_idx in splits:
                conc_train_scaled, scaler = _normalize_concentration(conc_f[tr_idx])
                conc_test_scaled          = _apply_conc_scaler(conc_f[te_idx], scaler)

                try:
                    tr_sub, val_sub = train_test_split(
                        np.arange(len(tr_idx)),
                        test_size=0.1, stratify=y_f[tr_idx], random_state=0)
                    X_tr = X_f[tr_idx][tr_sub];  X_val = X_f[tr_idx][val_sub]
                    y_tr = y_f[tr_idx][tr_sub];  y_val = y_f[tr_idx][val_sub]
                    c_tr = conc_train_scaled[tr_sub]; c_val = conc_train_scaled[val_sub]
                    fit_y = {'cls_out': y_tr} if is_supcon else y_tr
                    val_y = {'cls_out': y_val} if is_supcon else y_val
                    val_data  = ([X_val, c_val.reshape(-1, 1)], val_y)
                    es = tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss', patience=10,
                        restore_best_weights=True, verbose=0)
                    callbacks = [es]
                except ValueError:
                    X_tr, y_tr, c_tr = X_f[tr_idx], y_f[tr_idx], conc_train_scaled
                    fit_y     = {'cls_out': y_tr} if is_supcon else y_tr
                    val_data  = None
                    callbacks = []

                model = _GT_FILM_BUILDERS[m](T, n_cls)
                model.fit(
                    [X_tr, c_tr.reshape(-1, 1)],
                    fit_y,
                    validation_data=val_data,
                    callbacks=callbacks,
                    epochs=200,
                    batch_size=64,
                    verbose=0,
                )

                raw   = model.predict(
                    [X_f[te_idx], conc_test_scaled.reshape(-1, 1)], verbose=0)
                probs = raw[0] if is_supcon else raw
                preds = probs.argmax(-1)

                preds_folds.append(preds)
                probs_folds.append(probs)
                classes_folds.append(encoder_classes)

                tf.keras.backend.clear_session()
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

            print(f'     [+] {_GT_FILM_PRINT_MAP[m]:<22} | '
                  f'acc={np.mean(accs)*100:.2f}%±{np.std(accs)*100:.2f}% | '
                  f'{_fmt_hms(duration)}')

        results[f] = res_entry

    return results


# ============================================================
# LEADERBOARD PRINT
# ============================================================

def _print_leaderboard(all_ml_results, clean_title, mode_key, outlier_filters):
    mode_res = all_ml_results.get(clean_title, {}).get(mode_key, {})
    rows = []
    for f in outlier_filters:
        res = mode_res.get(f)
        if not isinstance(res, dict) or 'y_trues_' not in res:
            continue
        filter_name = str(f) if f is not None else 'None (Baseline)'
        comparison_keys = [
            *[(_GT_FILM_KEY_MAP[m][0], _GT_FILM_PRINT_MAP[m]) for m in _GT_FILM_MODELS],
            ('y_preds_AC_cnn_rcfd_cgd_',   'CNN RCFD CGD'),
            ('y_preds_AC_gru_rcfd_cgd_',   'GRU RCFD CGD'),
            ('y_preds_AC_trans_rcfd_cgd_', 'Trans RCFD CGD'),
            ('y_preds_AC_cnn_gru_dual_',   'CNN+GRU Dual'),
        ]
        for key, lbl in comparison_keys:
            if key not in res:
                continue
            accs = [accuracy_score(yt, yp)
                    for yt, yp in zip(res['y_trues_'], res[key])]
            rows.append((np.mean(accs) * 100, np.std(accs) * 100, lbl, filter_name))

    rows.sort(key=lambda x: x[0], reverse=True)
    W = 90
    print(f'\n  \U0001f3c6 Leaderboard [GT-FiLM CGD SC0-3 vs RCFD] {clean_title} / {mode_key}')
    print('  ' + '-' * W)
    for i, (acc, std, lbl, filt) in enumerate(rows):
        print(f'  {i+1:2d}.  {acc:6.2f}% ± {std:5.2f}%'
              f'  |  Model: {lbl:<24}  |  Filter: {filt}')
    print('  ' + '-' * W + '\n')


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CondReg Oracle — GT-FiLM CGD SC0-3 concept proof (03c)')
    parser.add_argument('--task_id',     type=int, default=0)
    parser.add_argument('--exp_folder',  type=str, default=config.DEFAULT_EXP_FOLDER)
    parser.add_argument('--n_splits',    type=int, default=1)
    parser.add_argument('--mode',        type=str, default='native',
                        choices=['native', 'reference'])
    parser.add_argument('--curve_type',  type=str, default='ori_curve')
    parser.add_argument('--force_rerun', action='store_true',
                        help='Clear only GT-FiLM CGD keys and retrain them. '
                             'All other model results in the joblib are untouched.')
    args = parser.parse_args()

    set_global_determinism(0)

    exp_paths = get_exp_paths(args.exp_folder)
    check_task_id(args.task_id, exp_paths)
    exp_path  = exp_paths[args.task_id]

    print(f"\n{'='*70}")
    print(f'[RUNNING] 03c_condreg_oracle_poc.py  [GT-FiLM CGD SC0-3 | mode={args.mode}]')
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

        results = run_gt_film_training(
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
        )

        all_ml_results[clean_title][mode_key] = results
        safe_joblib_dump(all_ml_results, results_file_path, compress=3)
        print(f'\n  -> Final results saved to {results_file_path}')

        _print_leaderboard(all_ml_results, clean_title, mode_key, outlier_filters)

        gc.collect()
