"""Multiplex multi-label training pipeline.

Mirrors main/03 args and structure; adapted for multi-label flat-CSV lab data:
  - y_binary (N, n_targets) + y_combo_int instead of y_encoded
  - StratifiedKFold/ShuffleSplit on y_combo_int (combination strings)
  - Model set determined by --supcon / --condreg (same flag contract as main/03)
  - Single training mode: flat-CSV has no reference/native split. mode_name is
    a cosmetic label describing the model group being run (e.g. "ML SC1",
    "RCFD SC0"). Each run with a different --supcon/--condreg combination
    accumulates into the same result file, identical to how main/03 accumulates
    ST / MTL / RCFD / SupCon runs into one classification_performances*.joblib.
  - Results stored flat: {filter_name -> res_entry} (no Native/Reference nesting)
"""
import os
import sys
import gc
import argparse
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import joblib

_THIS          = Path(__file__).resolve()
_MULTIPLEX_DIR = _THIS.parent
_MAIN_DIR      = _MULTIPLEX_DIR.parent

sys.path.insert(0, str(_MAIN_DIR))
sys.path.insert(0, str(_MAIN_DIR / "utils"))
sys.path.insert(0, str(_MAIN_DIR / "utils" / "model_training"))
sys.path.insert(0, str(_MULTIPLEX_DIR))
sys.path.insert(0, str(_MULTIPLEX_DIR / "utils" / "model_training"))

import config_multiplex as config
from safe_io import safe_joblib_dump
from model_utils import set_global_determinism
from model_utils_mtl import REG_SENTINEL as _REG_SENTINEL
from model_utils_multilabel import (
    ML_MODEL_KEY_MAP,
    ML_MODEL_PRINT_MAP,
    _RCFD_ML_KEYS,
    encode_multilabel_for_training,
    evaluate_outlier_filters_ml,
    print_ml_results_summary,
)


# ======================================================================
# HELPERS
# ======================================================================

def load_training_data(data_path):
    if not os.path.exists(data_path):
        print(f"  -> [SKIP] {data_path} not found. Run 01b + 02 first.")
        sys.exit(0)
    return joblib.load(data_path)


def load_or_init_results(results_path):
    if os.path.exists(results_path):
        try:
            return joblib.load(results_path)
        except Exception as e:
            print(f"  -> [WARNING] Results file corrupt ({e}). Starting fresh.")
    return {}


def _result_keys_for_model(m):
    pk, probk, clsk = ML_MODEL_KEY_MAP[m]
    keys = {pk, probk, clsk}
    if m in _RCFD_ML_KEYS:
        keys.update({f'y_reg_preds_{m}_', f'y_reg_trues_{m}_'})
    return keys


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multiplex Multi-Label Training")
    parser.add_argument("--task_id",    type=int, default=0, help="Array Job ID")
    parser.add_argument("--exp_folder", type=str, default=config.LAB_MULTIPLEX_FOLDER)
    parser.add_argument("--n_splits",   type=int, default=1,
                        help="CV folds (1=StratifiedShuffleSplit, >1=StratifiedKFold)")
    parser.add_argument("--force_rerun", action="store_true",
                        help="Clear cached results for the selected model group and retrain")
    parser.add_argument("--fast_mode", action="store_true",
                        help="Disable strict TF determinism for faster training. "
                             "RNG seeds still set; reruns won't be bit-exact.")
    parser.add_argument("--supcon", type=int, choices=[0, 1, 2, 3], default=0,
                        help="SupCon variant: 0=none (default), 1=SC1 fused SupCon, "
                             "2=SC2 branch heads (CNN+seq), 3=SC3 all heads. "
                             "Use with --condreg for RCFD SupCon variants.")
    parser.add_argument("--condreg", action="store_true",
                        help="Train RCFD models (Regression-Conditioned Feature Dual). "
                             "Scope to SC variant with --supcon.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Sigmoid threshold for binary prediction (default 0.5)")
    parser.add_argument("--rerun_models", type=str, nargs="+", default=None,
                        help="Restrict training (and --force_rerun clearing) to specific "
                             "model keys, e.g. --rerun_models cnn_gru_dual cnn_trans_dual")
    args = parser.parse_args()

    # --rerun_models has no effect without --force_rerun (mirrors main/03)
    if args.rerun_models and not args.force_rerun:
        args.rerun_models = None

    # mode_name: cosmetic label for logging (model group being run).
    # Analogous to "Native"/"Reference" in main/03 — identifies what was run.
    _mode = (
        ("RCFD" if args.condreg else "ML")
        + (f" SC{args.supcon}" if args.supcon else "")
    )
    print(f"\n{'='*70}\n[RUNNING] {os.path.basename(__file__)}  [{_mode}]\n{'='*70}\n")

    set_global_determinism(0, strict=not args.fast_mode)

    exp_folder = args.exp_folder
    subdirs    = sorted([d for d in Path(exp_folder).iterdir() if d.is_dir()])
    task_dirs  = [d for d in subdirs if d.name in config.FILE_MAPPING]

    if not task_dirs:
        print(f"No valid task directories found in {exp_folder}.")
        sys.exit(0)

    exp_path = task_dirs[args.task_id % len(task_dirs)]
    folder   = exp_path.name
    print(f"\n\n{'#'*80}\nSTARTING TRAINING FOR: {folder}\n{'#'*80}")

    if args.n_splits > 1:
        results_path = os.path.join(exp_path, config.TRAINING_10FOLD_RESULT_PATH)
    else:
        results_path = os.path.join(exp_path, config.TRAINING_RESULT_PATH)

    data_path = os.path.join(exp_path, config.TRAINING_DATA_PATH)

    # ── Load data ──────────────────────────────────────────────────────
    data        = load_training_data(data_path)
    curves      = data['curves']['ori_curves']
    label_lists = data['label_lists']
    all_targets = data['all_targets']

    # Concentration: loaded and sentinel-masked whenever RCFD is being run.
    # Mirrors main/03's concentration handling for --mtl / --condreg.
    y_concentration = None
    if args.condreg:
        raw_conc = data.get('concentration')
        if raw_conc is not None:
            _arr       = np.asarray(raw_conc, dtype=object)
            _float_arr = np.array(
                [float(v) if v is not None else np.nan for v in _arr], dtype=float)
            y_concentration = np.where(
                np.isnan(_float_arr) | (_float_arr == 0.0),
                _REG_SENTINEL, _float_arr)
        else:
            y_concentration = np.full(len(curves), _REG_SENTINEL, dtype=float)
        _n_valid = int((y_concentration != _REG_SENTINEL).sum())
        _n_zero  = int((_float_arr == 0.0).sum()) if raw_conc is not None else 0
        print(f"  [RCFD] Concentration: {_n_valid}/{len(y_concentration)} valid"
              + (f" ({_n_zero} zero/negative-control masked)." if _n_zero else "."))

    # Multi-label encoding
    y_binary, y_combo_int, mlb, combo_enc = encode_multilabel_for_training(
        label_lists, all_targets)
    print(f"  -> N={len(curves)}, targets={all_targets}, "
          f"combos={len(np.unique(y_combo_int))}, y_binary.shape={y_binary.shape}")

    # Kinetic features for outlier filter columns
    kf_dict     = data.get('kinetic_features', {})
    features_df = kf_dict.get(folder, pd.DataFrame(index=range(len(curves))))
    if len(features_df) != len(curves):
        features_df = pd.DataFrame(index=range(len(curves)))

    # ── Model group selection (mirrors main/03 --supcon / --condreg) ───
    if args.condreg:
        _rcfd_by_sc = {
            0: ['gru_rcfd_cgd',  'gru_rcfd_ctd',  'trans_rcfd_cgd',  'trans_rcfd_ctd'],
            1: ['gru_rcfd_cgd_supcon_mtl',  'gru_rcfd_ctd_supcon_mtl',
                'trans_rcfd_cgd_supcon_mtl',  'trans_rcfd_ctd_supcon_mtl'],
            2: ['gru_rcfd_cgd_supcon2_mtl', 'gru_rcfd_ctd_supcon2_mtl',
                'trans_rcfd_cgd_supcon2_mtl', 'trans_rcfd_ctd_supcon2_mtl'],
            3: ['gru_rcfd_cgd_supcon3_mtl', 'gru_rcfd_ctd_supcon3_mtl',
                'trans_rcfd_cgd_supcon3_mtl', 'trans_rcfd_ctd_supcon3_mtl'],
        }
        models = _rcfd_by_sc[args.supcon]
    elif args.supcon == 0:
        models = ['cnn', 'gru', 'transformer', 'cnn_gru_dual', 'cnn_trans_dual']
    elif args.supcon == 1:
        models = [
            'cnn_supcon', 'gru_supcon', 'transformer_supcon',
            'cnn_gru_dual_supcon', 'cnn_trans_dual_supcon',
        ]
    elif args.supcon == 2:
        models = ['cnn_gru_dual_supcon2', 'cnn_trans_dual_supcon2']
    elif args.supcon == 3:
        models = ['cnn_gru_dual_supcon3', 'cnn_trans_dual_supcon3']

    if args.rerun_models:
        _allowed = set(args.rerun_models)
        models   = [m for m in models if m in _allowed]

    print(f"  -> Models to train ({_mode}): {models}")
    print(f"[*] Found {len(config.OUTLIER_FILTERS)-1} Dynamic Outlier Filters to test.")

    # ── Cached results ─────────────────────────────────────────────────
    cached_results = load_or_init_results(results_path)

    # ── --force_rerun: clear only the selected group (mirrors main/03) ─
    if args.force_rerun:
        keys_to_clear = set()
        for m in models:
            if m in ML_MODEL_KEY_MAP:
                keys_to_clear.update(_result_keys_for_model(m))

        _mm = set(args.rerun_models) if args.rerun_models else None
        n_cleared = 0
        for filter_res in cached_results.values():
            if not isinstance(filter_res, dict):
                continue
            for rk in list(filter_res.keys()):
                if rk not in keys_to_clear:
                    continue
                if _mm and not any(m in rk for m in _mm):
                    continue
                del filter_res[rk]
                n_cleared += 1
        print(f"  -> [FORCE RERUN] Cleared {n_cleared} cached result key(s) for {_mode}.")

    # ── Checkpoint: update cached_results dict in-place and save ───────
    # Mirrors main/03's make_checkpoint_fn which does:
    #   all_ml_results[clean_title][mode_key] = updated_results
    #   safe_joblib_dump(all_ml_results, ...)
    # Here updated_results is the full results_dict (all filters computed so far).
    def checkpoint_fn(current_results):
        cached_results.update(current_results)
        safe_joblib_dump(cached_results, results_path, compress=3)

    # ── Train ──────────────────────────────────────────────────────────
    results = evaluate_outlier_filters_ml(
        X_curves           = curves,
        features_df        = features_df,
        y_binary           = y_binary,
        y_combo_int        = y_combo_int,
        all_targets        = all_targets,
        # outlier_filters    = config.OUTLIER_FILTERS,
        outlier_filters    = [None],
        dataset_name       = folder,
        mode_name          = _mode,
        ml_model_key_map   = ML_MODEL_KEY_MAP,
        ml_model_print_map = ML_MODEL_PRINT_MAP,
        cached_results     = cached_results,
        models             = models,
        n_splits           = args.n_splits,
        checkpoint_fn      = checkpoint_fn,
        y_concentration    = y_concentration,
        threshold          = args.threshold,
        rerun_models       = args.rerun_models,
    )

    cached_results.update(results)
    safe_joblib_dump(cached_results, results_path, compress=3)
    print(f"\n  -> Final results saved to {results_path}")

    print_ml_results_summary(
        cached_results, config.OUTLIER_FILTERS, folder, _mode,
        ML_MODEL_KEY_MAP, ML_MODEL_PRINT_MAP)

    gc.collect()
