import os
import sys
import gc
import argparse
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
sys.path.insert(0, 'utils')
from safe_io import safe_joblib_dump
from pipeline_utils import get_exp_paths, check_task_id
sys.path.insert(0, 'utils/model_training')
from model_utils import evaluate_outlier_filters, plot_ml_results, set_global_determinism
from model_utils_mtl import REG_SENTINEL as _MTL_REG_SENTINEL
from model_utils_supcon import (SUPCON_MODEL_KEYS, SUPCON_MTL_MODEL_KEYS,
                                BRANCH_SUPCON2_MODEL_KEYS, BRANCH_SUPCON2_MTL_MODEL_KEYS,
                                BRANCH_SUPCON3_MODEL_KEYS, BRANCH_SUPCON3_MTL_MODEL_KEYS,
                                CL_SUPCON_MTL_MODEL_KEYS,
                                CL_BRANCH_SUPCON2_MTL_MODEL_KEYS,
                                CL_BRANCH_SUPCON3_MTL_MODEL_KEYS,
                                ALL_LC_KEYS,
                                STAGED_SUPCON_MODEL_KEYS, STAGED_BRANCH_SUPCON2_MODEL_KEYS,
                                STAGED_BRANCH_SUPCON3_MODEL_KEYS, ALL_STAGED_SUPCON_KEYS)
from model_utils_mtl import CL_MTL_MODEL_KEYS
from model_utils_rcfd import (RCFD_MODEL_KEYS, RCFD_SUPCON_MTL_MODEL_KEYS,
                               RCFD_BRANCH2_MTL_MODEL_KEYS, RCFD_BRANCH3_MTL_MODEL_KEYS)
from sklearn.feature_selection import mutual_info_classif
import numpy as np
import pandas as pd


import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# ============================================================
# HELPERS
# ============================================================
def load_training_data(exp_path):
    data_path = os.path.join(exp_path, config.TRAINING_DATA_PATH)
    if not os.path.exists(data_path):
        print(f"  -> Skipping {exp_path.name}: '{data_path}' not found.")
        sys.exit(0)
    return joblib.load(data_path)

def filter_datasets(dataset_name, dataset, kinetic_features):
    filtered_names, filtered_dataset, filtered_features = [], [], []
    for name, data, features in zip(dataset_name, dataset, kinetic_features):
        if not name.startswith("avg_") and not name.startswith("original_fitted_stretched"):
            filtered_names.append(name)
            filtered_dataset.append(data)
            filtered_features.append(features)
    return filtered_names, filtered_dataset, filtered_features

def load_or_init_results(results_file_path):
    if os.path.exists(results_file_path):
        try:
            return joblib.load(results_file_path)
        except Exception as e:
            print(f"  -> [WARNING] Results file corrupt ({e}), starting fresh: {results_file_path}")
    return {}

def _lc_label(well_label, conc):
    """Format combined label+concentration string for LC target encoding."""
    try:
        c = float(conc)
    except (TypeError, ValueError):
        return f'{well_label}_0'
    if c <= 0 or np.isnan(c):
        return f'{well_label}_0'
    if c >= 1e6:  return f'{well_label}_{c/1e6:.3g}M'
    if c >= 1e3:  return f'{well_label}_{c/1e3:.3g}K'
    return f'{well_label}_{c:.3g}'

def make_checkpoint_fn(all_ml_results, results_file_path, clean_title, mode_key):
    def _checkpoint(updated_results):
        all_ml_results[clean_title][mode_key] = updated_results
        safe_joblib_dump(all_ml_results, results_file_path, compress=3)
    return _checkpoint

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main Training Pipeline")
    parser.add_argument("--task_id", type=int, default=0, help="Array Job ID")
    parser.add_argument("--exp_folder", type=str, default=config.DEFAULT_EXP_FOLDER)
    parser.add_argument("--n_splits", type=int, default=1)
    parser.add_argument("--force_rerun", action="store_true", help="Recompute and overwrite even if presaved results already exist")
    parser.add_argument("--training_mode", type=str, nargs="+", choices=["native", "reference"], default=["native"], help="Which training mode(s) to run: 'native' (train on this dataset's own curves) and/or 'reference' (train on the original ori_curves, using this dataset's outlier filters)")
    parser.add_argument("--curve_type", type=str, nargs="+",
                        default=["ori_curve", "ori_curve_avg", "ori_curve_wavelet_sym8",
                                 "ori_curve_wavelet_bior35", "ori_curve_sg_p4"],
                        help="Which curve variant(s) to train on and save XAI models for (e.g. 'ori_curve' 'ori_curve_avg')")
    parser.add_argument("--fast_mode", action="store_true",
                        help="Disable strict TF determinism (TF_CUDNN_DETERMINISTIC/enable_op_determinism) "
                             "for faster GRU/LSTM/Transformer training. RNG seeds are still set, but reruns "
                             "won't be bit-exact. Only affects this script.")
    parser.add_argument("--k_neighbors", type=int, default=24,
                        help="Neighbours per pixel (within the same well) for "
                             "cnn_gru_dual_cosine_recon/cnn_gru_dual_attn_recon's spatial reconstruction.")
    parser.add_argument("--mtl", action="store_true",
                        help="Train MTL models only (classification + concentration regression heads). "
                             "Results are merged into the same classification_performances*.joblib "
                             "so standard models do not need to be re-run.")
    parser.add_argument("--supcon", type=int, choices=[0, 1, 2, 3], default=0,
                        help="SupCon variant: 0=none (default), 1=original fused SupCon, "
                             "2=branch SupCon v2 (CNN+seq branch heads), "
                             "3=branch SupCon v3 (CNN+seq+fused heads). Use with --mtl for MTL variants.")
    parser.add_argument("--rerun_models", type=str, nargs="+", default=None,
                        help="Restrict training (and --force_rerun clearing) to specific model keys, "
                             "e.g. --rerun_models cnn_gru_dual_cosine_recon cnn_gru_dual_attn_recon")
    parser.add_argument("--mtl_cl", action="store_true",
                        help="Use phase-decoupled curriculum learning for MTL: Phase 1=regression, Phase 2=classification.")
    parser.add_argument("--cl_phase1_epochs", type=int, default=None,
                        help="Fixed number of Phase 1 epochs for CL-MTL. If omitted, auto-detects convergence on val_reg_mse.")
    parser.add_argument("--condreg", action="store_true",
                        help="Train RCFD models (Regression-Conditioned Feature Dual). Implies --mtl.")
    parser.add_argument("--lbl_conc", action="store_true",
                        help="Label Consolidation: combine label+concentration into one classification target. "
                             "Pure ST — mutually exclusive with --mtl.")
    parser.add_argument("--supcon_staged", action="store_true",
                        help="2-stage SupCon: Stage 1 SC-only until plateau, Stage 2 CE-only frozen backbone.")
    args = parser.parse_args()
    if args.mtl_cl:
        args.mtl = True  # --mtl_cl implies --mtl
    if args.condreg:
        args.mtl = True  # --condreg implies --mtl
    if getattr(args, 'lbl_conc', False) and args.mtl:
        sys.exit('[!] --lbl_conc is not compatible with --mtl. Use one or the other.')
    if getattr(args, 'supcon_staged', False) and args.mtl:
        sys.exit('[!] --supcon_staged is ST-only; incompatible with --mtl.')
    if getattr(args, 'supcon_staged', False) and args.supcon == 0:
        sys.exit('[!] --supcon_staged requires --supcon 1, 2, or 3.')
    _mode = ("MTL" if args.mtl else "ST") + (f" SupCon-{args.supcon}" if args.supcon else "") + (" CL" if args.mtl_cl else "") + (f" RCFD SC{args.supcon}" if args.condreg else "") + (f" LC SC{args.supcon}" if getattr(args, 'lbl_conc', False) else "") + (" Staged" if getattr(args, 'supcon_staged', False) else "")
    print(f"\n{'='*70}\n[RUNNING] {os.path.basename(__file__)}  [{_mode}]\n{'='*70}\n")
    if args.rerun_models and not args.force_rerun:
        args.rerun_models = None  # --rerun_models has no effect without --force_rerun

    set_global_determinism(0, strict=not args.fast_mode)
    
    exp_paths = get_exp_paths(args.exp_folder)
    n_splits = args.n_splits
    check_task_id(args.task_id, exp_paths)

    exp_path = exp_paths[args.task_id]
    print(f"\n\n{'#'*80}\nSTARTING TRAINING FOR: {exp_path.name}\n{'#'*80}")

    if(n_splits > 1):
        results_file_path = os.path.join(exp_path, config.TRAINING_10FOLD_RESULT_PATH)
    else:    
        results_file_path = os.path.join(exp_path, config.TRAINING_RESULT_PATH)
    model_plot_path = os.path.join(exp_path, "model_performance")
    os.makedirs(model_plot_path, exist_ok=True)

    training_data = load_training_data(exp_path)

    dataset_name = training_data["dataset_name"]
    dataset = training_data["dataset"]
    kinetic_features = training_data["kinetic_features"]
    Y_well = training_data["Y_well"]

    # Load concentration for MTL regression head or LC label encoding.
    y_concentration = None
    if args.mtl or getattr(args, 'lbl_conc', False):
        raw_conc = training_data.get("concentration", None)
        if raw_conc is not None:
            _arr = np.asarray(raw_conc, dtype=object)
            _float_arr = np.array([float(v) if v is not None else np.nan for v in _arr], dtype=float)
            # Mask None (→ nan) AND zero (negative controls, log10 undefined) as sentinel.
            y_concentration = np.where(np.isnan(_float_arr) | (_float_arr == 0.0),
                                        _MTL_REG_SENTINEL, _float_arr)
        else:
            y_concentration = np.full(len(Y_well), _MTL_REG_SENTINEL, dtype=float)
        _n_valid = int((y_concentration != _MTL_REG_SENTINEL).sum())
        _n_zero  = int((_float_arr == 0.0).sum()) if raw_conc is not None else 0
        print(f"  [MTL] Concentration loaded: {_n_valid} / {len(y_concentration)} samples "
              f"have non-sentinel concentration"
              + (f" ({_n_zero} zero/negative-control masked)." if _n_zero else "."))

    # Spatial metadata for cnn_gru_dual_cosine_recon/cnn_gru_dual_attn_recon (see
    # model_utils.build_neighbor_curve_stack). Soft-optional: unlike 03b_gnn_spatial_training.py
    # (which trains GNN models exclusively and exits if metadata is missing), 03 trains many
    # non-spatial models too -- a dataset lacking metadata just means those two models get
    # skipped (with a warning from evaluate_outlier_filters), everything else still runs.
    # well_ids is derived from the RAW (pre label-mapping) Y_well -- the physical/spatial
    # grouping for neighbour-finding, deliberately independent of how config.LABEL_MAPPINGS
    # later buckets labels for the classification target (mirrors 03b's identical comment).
    coords_full, well_ids_full = None, None
    if "metadata" in training_data:
        metadata_df = pd.DataFrame(training_data["metadata"])
        if {"pixel_row_idx", "pixel_col_idx"}.issubset(metadata_df.columns):
            coords_full = np.stack([
                metadata_df["pixel_row_idx"].values.astype(float),
                metadata_df["pixel_col_idx"].values.astype(float),
            ], axis=1)
            well_ids_full = (metadata_df["well_id"].values if "well_id" in metadata_df.columns
                              else np.array(Y_well).copy())
        else:
            print("  [*] No pixel_row_idx/pixel_col_idx in metadata -- "
                  "cnn_gru_dual_cosine_recon/cnn_gru_dual_attn_recon will be skipped for this dataset.")
    else:
        print("  [*] No 'metadata' in training data -- "
              "cnn_gru_dual_cosine_recon/cnn_gru_dual_attn_recon will be skipped for this dataset.")

    # Filter Clean data only
    dataset_name, dataset, kinetic_features = filter_datasets(dataset_name, dataset, kinetic_features)

    label_mappings = config.get_label_mappings(exp_path)
    if exp_path.name in label_mappings:
        print(f"  [*] Applying custom target label mapping for experiment: {exp_path.name}")
        mapping = label_mappings[exp_path.name]
        
        # Maps matching keys; falls back to the original index value if not found
        Y_well = [mapping.get(w, w) for w in Y_well]
    else:
        print(f"  [*] No custom mapping found for {exp_path.name}. Retaining default well labels.")

    # Convert to encoded labels
    if getattr(args, 'lbl_conc', False):
        if y_concentration is None:
            sys.exit('[!] --lbl_conc requires concentration data in curve_for_training.joblib.')
        # Y_well is already post-label_mapping here; combine with concentration
        combined = [_lc_label(y, c) for y, c in zip(Y_well, y_concentration)]
        encoder = LabelEncoder()
        y_full = encoder.fit_transform(combined)
        print(f'  [LC] {len(set(combined))} combined classes: {sorted(set(combined))}')
    else:
        encoder = LabelEncoder()
        y_full = encoder.fit_transform(Y_well)

    # outlier_filters = config.OUTLIER_FILTERS
    # outlier_filters = [None, 'lstm_ae_glb_ds1_label_elbow', 'spatial_knn_label_elbow', 'spatial_grid_label_elbow']
    outlier_filters = [None]

    print(f"[*] Found {len(outlier_filters)-1} Dynamic Outlier Filters to test.")

    all_ml_results = load_or_init_results(results_file_path)
    if args.force_rerun:
        # Build result-dict key sets per training mode so each flag only clears its own results.
        _mtl_result_keys = set()
        for _k, (_pk, _probk, _clsk) in config.MODEL_KEY_MAP.items():
            if _k in config._MTL_MODEL_KEYS:
                _mtl_result_keys.update([_pk, _probk, _clsk,
                                         f'y_reg_preds_{_k}_', f'y_reg_trues_{_k}_'])
        _supcon_st_result_keys = set()
        _supcon_mtl_result_keys = set()
        for _k, (_pk, _probk, _clsk) in config.MODEL_KEY_MAP.items():
            if _k in config._SUPCON_MODEL_KEYS:
                if 'mtl' in _k:
                    _supcon_mtl_result_keys.update([_pk, _probk, _clsk,
                                                    f'y_reg_preds_{_k}_', f'y_reg_trues_{_k}_'])
                else:
                    _supcon_st_result_keys.update([_pk, _probk, _clsk])
        # Branch SupCon key sets (--supcon 2/3) — kept separate from _SUPCON_MODEL_KEYS
        # so --supcon 1 force_rerun doesn't accidentally clear v2/v3 results and vice versa.
        _bsc_st_keys  = (set(BRANCH_SUPCON2_MODEL_KEYS)     if args.supcon == 2
                         else set(BRANCH_SUPCON3_MODEL_KEYS) if args.supcon == 3 else set())
        _bsc_mtl_keys = (set(BRANCH_SUPCON2_MTL_MODEL_KEYS)     if args.supcon == 2
                         else set(BRANCH_SUPCON3_MTL_MODEL_KEYS) if args.supcon == 3 else set())
        _bsc_st_result_keys  = set()
        _bsc_mtl_result_keys = set()
        _bsc_all_result_keys = set()  # all SC2+SC3 keys — for _is_standard exclusion
        for _k, (_pk, _probk, _clsk) in config.MODEL_KEY_MAP.items():
            if _k in config._BRANCH_SUPCON_MODEL_KEYS:
                _bsc_all_result_keys.update([_pk, _probk, _clsk,
                                              f'y_reg_preds_{_k}_', f'y_reg_trues_{_k}_'])
            if _k in _bsc_st_keys:
                _bsc_st_result_keys.update([_pk, _probk, _clsk])
            elif _k in _bsc_mtl_keys:
                _bsc_mtl_result_keys.update([_pk, _probk, _clsk,
                                              f'y_reg_preds_{_k}_', f'y_reg_trues_{_k}_'])
        # CL key sets — isolated from non-CL MTL so --force_rerun only clears the right variant.
        _cl_base_result_keys  = set()
        _cl_supcon_result_keys = set()
        _cl_bsc2_result_keys  = set()
        _cl_bsc3_result_keys  = set()
        for _k, (_pk, _probk, _clsk) in config.MODEL_KEY_MAP.items():
            if _k in CL_MTL_MODEL_KEYS:
                _cl_base_result_keys.update([_pk, _probk, _clsk, f'y_reg_preds_{_k}_', f'y_reg_trues_{_k}_'])
            elif _k in CL_SUPCON_MTL_MODEL_KEYS:
                _cl_supcon_result_keys.update([_pk, _probk, _clsk, f'y_reg_preds_{_k}_', f'y_reg_trues_{_k}_'])
            elif _k in CL_BRANCH_SUPCON2_MTL_MODEL_KEYS:
                _cl_bsc2_result_keys.update([_pk, _probk, _clsk, f'y_reg_preds_{_k}_', f'y_reg_trues_{_k}_'])
            elif _k in CL_BRANCH_SUPCON3_MTL_MODEL_KEYS:
                _cl_bsc3_result_keys.update([_pk, _probk, _clsk, f'y_reg_preds_{_k}_', f'y_reg_trues_{_k}_'])

        # RCFD key sets — full set (for _is_standard exclusion) + scoped to --supcon variant.
        _rcfd_result_keys = set()
        _rcfd_sc_result_keys = set()
        for _k, (_pk, _probk, _clsk) in config.MODEL_KEY_MAP.items():
            if _k not in config._RCFD_MODEL_KEYS:
                continue
            _rcfd_result_keys.update([_pk, _probk, _clsk,
                                      f'y_reg_preds_{_k}_', f'y_reg_trues_{_k}_'])
            if _k.endswith('_supcon3_mtl'):   _k_rcfd_sc = 3
            elif _k.endswith('_supcon2_mtl'): _k_rcfd_sc = 2
            elif _k.endswith('_supcon_mtl'):  _k_rcfd_sc = 1
            else:                              _k_rcfd_sc = 0
            if _k_rcfd_sc == args.supcon:
                _rcfd_sc_result_keys.update([_pk, _probk, _clsk,
                                              f'y_reg_preds_{_k}_', f'y_reg_trues_{_k}_'])

        # LC key sets — full set (for _is_standard exclusion) + scoped to --supcon variant.
        _lc_result_keys = set()
        _lc_sc_result_keys = set()
        for _k, (_pk, _probk, _clsk) in config.MODEL_KEY_MAP.items():
            if _k not in config._LC_MODEL_KEYS:
                continue
            _lc_result_keys.update([_pk, _probk, _clsk])
            if 'supcon3_lc' in _k:   _k_lc_sc = 3
            elif 'supcon2_lc' in _k: _k_lc_sc = 2
            elif 'supcon_lc' in _k:  _k_lc_sc = 1
            else:                     _k_lc_sc = 0
            if _k_lc_sc == args.supcon:
                _lc_sc_result_keys.update([_pk, _probk, _clsk])

        # Staged SupCon key sets — full set (for _is_standard exclusion) + scoped to --supcon variant.
        _staged_result_keys = set()
        _staged_sc_result_keys = set()
        for _k, (_pk, _probk, _clsk) in config.MODEL_KEY_MAP.items():
            if _k not in config._STAGED_SUPCON_MODEL_KEYS:
                continue
            _staged_result_keys.update([_pk, _probk, _clsk])
            if 'supcon3_staged' in _k:   _k_staged_sc = 3
            elif 'supcon2_staged' in _k: _k_staged_sc = 2
            elif 'supcon_staged' in _k:  _k_staged_sc = 1
            else:                         _k_staged_sc = 0
            if _k_staged_sc == args.supcon:
                _staged_sc_result_keys.update([_pk, _probk, _clsk])

        _is_cl = getattr(args, 'mtl_cl', False)
        if getattr(args, 'supcon_staged', False):
            _which = f'Staged SupCon SC{args.supcon}'
        elif getattr(args, 'lbl_conc', False):
            _which = f'LC SC{args.supcon}'
        elif getattr(args, 'condreg', False):
            _which = f'RCFD SC{args.supcon}'
        elif _is_cl and args.supcon == 0:
            _which = 'CL MTL'
        elif _is_cl and args.supcon == 1:
            _which = 'CL SupCon v1 MTL'
        elif _is_cl and args.supcon in (2, 3):
            _which = f'CL Branch SupCon v{args.supcon} MTL'
        elif args.supcon == 1 and args.mtl:
            _which = 'SupCon v1 MTL'
        elif args.supcon == 1:
            _which = 'SupCon v1 ST'
        elif args.supcon in (2, 3) and args.mtl:
            _which = f'Branch SupCon v{args.supcon} MTL'
        elif args.supcon in (2, 3):
            _which = f'Branch SupCon v{args.supcon} ST'
        elif args.mtl:
            _which = 'MTL'
        else:
            _which = 'standard'
        print(f"  -> [FORCE RERUN] Clearing {_which} cached results; preserving the rest.")
        for _td in all_ml_results.values():
            for _mode_key in ("Native", "Reference"):
                _mode = _td.get(_mode_key)
                if not isinstance(_mode, dict):
                    continue
                for _filter_res in _mode.values():
                    if not isinstance(_filter_res, dict):
                        continue
                    for _rk in list(_filter_res.keys()):
                        if not isinstance(_rk, str):
                            continue
                        _is_mtl         = _rk in _mtl_result_keys
                        _is_supcon_st   = _rk in _supcon_st_result_keys
                        _is_supcon_mtl  = _rk in _supcon_mtl_result_keys
                        _is_model_key   = any(_rk.startswith(p) for p in
                                              ('y_preds_AC_', 'y_probs_AC_', 'classes_AC_',
                                               'y_reg_preds_', 'y_reg_trues_'))
                        _is_bsc_st      = _rk in _bsc_st_result_keys
                        _is_bsc_mtl     = _rk in _bsc_mtl_result_keys
                        _is_bsc_all     = _rk in _bsc_all_result_keys
                        _is_cl_base     = _rk in _cl_base_result_keys
                        _is_cl_supcon   = _rk in _cl_supcon_result_keys
                        _is_cl_bsc2     = _rk in _cl_bsc2_result_keys
                        _is_cl_bsc3     = _rk in _cl_bsc3_result_keys
                        _is_any_cl      = _is_cl_base or _is_cl_supcon or _is_cl_bsc2 or _is_cl_bsc3
                        _is_rcfd        = _rk in _rcfd_result_keys
                        _is_lc          = _rk in _lc_result_keys
                        _is_staged      = _rk in _staged_result_keys
                        _is_standard    = (_is_model_key and not _is_mtl and not _is_supcon_st
                                           and not _is_supcon_mtl and not _is_bsc_all
                                           and not _is_any_cl and not _is_rcfd and not _is_lc
                                           and not _is_staged
                                           and (not args.rerun_models or any(_m in _rk for _m in args.rerun_models)))
                        _mm = not args.rerun_models or any(_m in _rk for _m in args.rerun_models)
                        if _is_cl and args.supcon == 0 and _is_cl_base and _mm:
                            del _filter_res[_rk]
                        elif _is_cl and args.supcon == 1 and _is_cl_supcon and _mm:
                            del _filter_res[_rk]
                        elif _is_cl and args.supcon == 2 and _is_cl_bsc2 and _mm:
                            del _filter_res[_rk]
                        elif _is_cl and args.supcon == 3 and _is_cl_bsc3 and _mm:
                            del _filter_res[_rk]
                        elif args.supcon == 1 and args.mtl and not getattr(args, 'mtl_cl', False) and not getattr(args, 'condreg', False) and _is_supcon_mtl and _mm:
                            del _filter_res[_rk]
                        elif args.supcon == 1 and not args.mtl and not getattr(args, 'lbl_conc', False) and not getattr(args, 'supcon_staged', False) and _is_supcon_st and _mm:
                            del _filter_res[_rk]
                        elif args.supcon in (2, 3) and args.mtl and not getattr(args, 'mtl_cl', False) and not getattr(args, 'condreg', False) and _is_bsc_mtl and _mm:
                            del _filter_res[_rk]
                        elif args.supcon in (2, 3) and not args.mtl and not getattr(args, 'lbl_conc', False) and not getattr(args, 'supcon_staged', False) and _is_bsc_st and _mm:
                            del _filter_res[_rk]
                        elif args.mtl and args.supcon == 0 and not getattr(args, 'mtl_cl', False) and not getattr(args, 'condreg', False) and _is_mtl and _mm:
                            del _filter_res[_rk]
                        elif args.supcon == 0 and not args.mtl and not getattr(args, 'lbl_conc', False) and _is_standard:
                            del _filter_res[_rk]
                        elif getattr(args, 'condreg', False) and _rk in _rcfd_sc_result_keys and _mm:
                            del _filter_res[_rk]
                        elif getattr(args, 'lbl_conc', False) and _rk in _lc_sc_result_keys and _mm:
                            del _filter_res[_rk]
                        elif getattr(args, 'supcon_staged', False) and _rk in _staged_sc_result_keys and _mm:
                            del _filter_res[_rk]

    total_datasets = len(dataset_name)
    total_samples = len(y_full)
    trained_curve = dataset[0].copy()

    # Dataset names that correspond to the requested curve_types.
    _target_names = {config.CURVE_TYPE_ALIASES.get(ct, ct) for ct in args.curve_type}

    # Maps dataset_name back to the CLI curve_type alias used in file names.
    _reverse_alias = {v: k for k, v in config.CURVE_TYPE_ALIASES.items()}

    def _lstm_ae_paths(curve_dataset_name):
        """Path to the pretrained LSTM-AE encoder/scaler for one curve variant, saved
        by 02_outlier_detection_pipeline.py's global LSTM autoencoder (see
        lstm_autoencoder_outlier.py's _save_encoder). Returns (None, None) if either
        file is missing — "lstm_ae_clf" then gets skipped by evaluate_outlier_filters."""
        enc_dir = exp_path / "pretrained_encoders"
        enc_path = enc_dir / f"lstm_ae_encoder_{curve_dataset_name}.keras"
        scaler_path = enc_dir / f"lstm_ae_scaler_{curve_dataset_name}.joblib"
        if enc_path.exists() and scaler_path.exists():
            return str(enc_path), str(scaler_path)
        return None, None

    for idx, (name, features_df, curves_2d) in enumerate(zip(dataset_name, kinetic_features, dataset)):
        if name not in _target_names:
            continue

        clean_title = name.replace("_", " ").title()
        xai_curve_type = _reverse_alias.get(name, name)
        progress_pct = ((idx + 1) / total_datasets) * 100

        print(f"\n{'='*75}")
        print(f"[{idx+1}/{total_datasets} | {progress_pct:.1f}%] Processing Dataset: {clean_title}")
        print(f"{'='*75}")

        if clean_title not in all_ml_results:
            all_ml_results[clean_title] = {}

        # --- FEATURE SELECTION (MUTUAL INFORMATION) on training data only ---
        # MI is computed on the same 90/10 split used for the None-filter baseline,
        # so test-set labels never influence feature selection.
        print(f"\n  [*] Calculating Mutual Information for Top 10 Features (train split only)...")
        X_candidates = features_df[config.LD_FEATURES].values
        X_candidates_clean = np.nan_to_num(X_candidates, nan=0.0, posinf=0.0, neginf=0.0)

        n_classes = len(np.unique(y_full))
        mi_test_size = max(int(len(y_full) * 0.10), n_classes)
        mi_splitter = StratifiedShuffleSplit(n_splits=1, test_size=mi_test_size, random_state=0)
        mi_train_idx, _ = next(mi_splitter.split(X_candidates_clean, y_full))
        mi_scores = mutual_info_classif(X_candidates_clean[mi_train_idx], y_full[mi_train_idx], random_state=0)

        top_10_idx = np.argsort(mi_scores)[-10:][::-1]
        top_10_features = [config.LD_FEATURES[i] for i in top_10_idx]
        print(f"  [*] Selected Top 10 Features: {top_10_features}")

        # if getattr(args, 'condreg', False):
        #     if args.supcon == 0:
        #         models = list(RCFD_MODEL_KEYS)
        #     elif args.supcon == 1:
        #         models = list(RCFD_SUPCON_MTL_MODEL_KEYS)
        #     elif args.supcon == 2:
        #         models = list(RCFD_BRANCH2_MTL_MODEL_KEYS)
        #     elif args.supcon == 3:
        #         models = list(RCFD_BRANCH3_MTL_MODEL_KEYS)
        # elif args.mtl and getattr(args, 'mtl_cl', False):
        #     if args.supcon == 0:
        #         models = list(CL_MTL_MODEL_KEYS)
        #     elif args.supcon == 1:
        #         models = list(CL_SUPCON_MTL_MODEL_KEYS)
        #     elif args.supcon == 2:
        #         models = list(CL_BRANCH_SUPCON2_MTL_MODEL_KEYS)
        #     elif args.supcon == 3:
        #         models = list(CL_BRANCH_SUPCON3_MTL_MODEL_KEYS)
        # elif args.mtl:
        #     if args.supcon == 1:
        #         models = list(SUPCON_MTL_MODEL_KEYS)
        #     elif args.supcon == 2:
        #         models = list(BRANCH_SUPCON2_MTL_MODEL_KEYS)
        #     elif args.supcon == 3:
        #         models = list(BRANCH_SUPCON3_MTL_MODEL_KEYS)
        #     else:
        #         models = [
        #             # "cnn_mtl",
        #             # "gru_mtl", 
        #             "cnn_gru_dual_mtl",
        #             # "transformer_mtl", 
        #             # "cnn_trans_dual_mtl",
        #             "cnn_gru_dual_cosine_recon_mtl", "cnn_gru_dual_attn_recon_mtl",
        #         ]
        # else:
        #     if args.supcon == 1:
        #         models = list(SUPCON_MODEL_KEYS)
        #     elif args.supcon == 2:
        #         models = list(BRANCH_SUPCON2_MODEL_KEYS)
        #     elif args.supcon == 3:
        #         models = list(BRANCH_SUPCON3_MODEL_KEYS)
        #     else:
        #         models = [
        #             # "knn", "cnn",
        #             # "gru", 
        #             "cnn_gru_dual",
        #             # "transformer", 
        #             # "cnn_trans_dual",
        #             "cnn_gru_dual_cosine_recon", "cnn_gru_dual_attn_recon",
        #         ]


        #### CNN+GRU DUAL VARIANTS ONLY
        # STAGED SUPCON (2-STAGE ST) MODELS
        if getattr(args, 'supcon_staged', False):
            if args.supcon == 1:
                models = list(STAGED_SUPCON_MODEL_KEYS)
            elif args.supcon == 2:
                models = list(STAGED_BRANCH_SUPCON2_MODEL_KEYS)
            elif args.supcon == 3:
                models = list(STAGED_BRANCH_SUPCON3_MODEL_KEYS)

        # LABEL CONSOLIDATION (LC) MODELS
        elif getattr(args, 'lbl_conc', False):
            _sc_lc = {
                0: ['cnn_gru_dual_lc', 'cnn_gru_dual_cosine_recon_lc', 'cnn_gru_dual_attn_recon_lc'],
                1: ['cnn_gru_dual_supcon_lc', 'cnn_gru_dual_cosine_recon_supcon_lc', 'cnn_gru_dual_attn_recon_supcon_lc'],
                2: ['cnn_gru_dual_supcon2_lc', 'cnn_gru_dual_cosine_recon_supcon2_lc', 'cnn_gru_dual_attn_recon_supcon2_lc'],
                3: ['cnn_gru_dual_supcon3_lc', 'cnn_gru_dual_cosine_recon_supcon3_lc', 'cnn_gru_dual_attn_recon_supcon3_lc'],
            }
            models = _sc_lc[args.supcon]

        # CONDITIONAL REGRESSION (RCFD) MODELS
        elif getattr(args, 'condreg', False):
            if args.supcon == 0:    
                models = ['cnn_rcfd_cgd', 'gru_rcfd_cgd', 'trans_rcfd_cgd']
            elif args.supcon == 1:
                models = ['cnn_rcfd_cgd_supcon_mtl', 'gru_rcfd_cgd_supcon_mtl', 'trans_rcfd_cgd_supcon_mtl']
            elif args.supcon == 2:
                models = ['cnn_rcfd_cgd_supcon2_mtl', 'gru_rcfd_cgd_supcon2_mtl', 'trans_rcfd_cgd_supcon2_mtl']
            elif args.supcon == 3:
                models = ['cnn_rcfd_cgd_supcon3_mtl', 'gru_rcfd_cgd_supcon3_mtl', 'trans_rcfd_cgd_supcon3_mtl']
        
        # CURRICULUM LEARNING (CL) MTL MODELS
        elif args.mtl and getattr(args, 'mtl_cl', False):
            if args.supcon == 0:
                models = ['cnn_gru_dual_cl_mtl']
            elif args.supcon == 1:
                models = ['cnn_gru_dual_cl_supcon_mtl']
            elif args.supcon == 2:
                models = ['cnn_gru_dual_cl_supcon2_mtl']
            elif args.supcon == 3:
                models = ['cnn_gru_dual_cl_supcon3_mtl']
        
        # MULTI-TASK LEARNING (MTL) MODELS
        elif args.mtl:
            if args.supcon == 1:
                models = ['cnn_gru_dual_supcon_mtl', 'cnn_gru_dual_cosine_recon_supcon_mtl', 'cnn_gru_dual_attn_recon_supcon_mtl']
            elif args.supcon == 2:
                models = ['cnn_gru_dual_supcon2_mtl', 'cnn_gru_dual_cosine_recon_supcon2_mtl', 'cnn_gru_dual_attn_recon_supcon2_mtl']
            elif args.supcon == 3:
                models = ['cnn_gru_dual_supcon3_mtl', 'cnn_gru_dual_cosine_recon_supcon3_mtl', 'cnn_gru_dual_attn_recon_supcon3_mtl']
            else:
                models = ['cnn_gru_dual_mtl', 'cnn_gru_dual_cosine_recon_mtl', 'cnn_gru_dual_attn_recon_mtl']
                
        # SINGLE-TASK (ST) MODELS
        # else:
        #     if args.supcon == 1:
        #         models = ['cnn_gru_dual_supcon', 'cnn_gru_dual_cosine_recon_supcon', 'cnn_gru_dual_attn_recon_supcon', 'ccgd_arch_poc_st_sc1']
        #     elif args.supcon == 2:
        #         models = ['cnn_gru_dual_supcon2', 'cnn_gru_dual_cosine_recon_supcon2', 'cnn_gru_dual_attn_recon_supcon2', 'ccgd_arch_poc_st_sc2']
        #     elif args.supcon == 3:
        #         models = ['cnn_gru_dual_supcon3', 'cnn_gru_dual_cosine_recon_supcon3', 'cnn_gru_dual_attn_recon_supcon3', 'ccgd_arch_poc_st_sc3']
        #     else:
        #         models = ['cnn_gru_dual', 'cnn_gru_dual_cosine_recon', 'cnn_gru_dual_attn_recon', 'ccgd_arch_poc_st']
        else:
            if args.supcon == 1:
                models = ['cnn_gru_dual_supcon', 'cnn_gru_dual_attn_recon_supcon']
            elif args.supcon == 2:
                models = ['cnn_gru_dual_supcon2', 'cnn_gru_dual_attn_recon_supcon2']
            elif args.supcon == 3:
                models = ['cnn_gru_dual_supcon3', 'cnn_gru_dual_attn_recon_supcon3']
            else:
                models = ['cnn_gru_dual', 'cnn_gru_dual_attn_recon']
    

        if args.rerun_models:
            _allowed = set(args.rerun_models)
            models = [m for m in models if m in _allowed]

        model_interp_dir = exp_path / "model_interpretation"

        # --- REFERENCE TRAINING (this dataset's outlier filters, trained on the original curves) ---
        if "reference" in args.training_mode:
            print(f"\n  [MODE] REFERENCE TRAINING")
            cached_ref = all_ml_results[clean_title].get("Reference", {})
            checkpoint_ref = make_checkpoint_fn(all_ml_results, results_file_path, clean_title, "Reference")

            # Save XAI models from the Reference run for ori_curves (Reference curve = ori_curves).
            ref_save_dir = model_interp_dir if name == "ori_curves" else None
            ref_save_ct = xai_curve_type if ref_save_dir else "ori_curve"

            # Reference always trains on dataset[0] (ori_curves), outlier filters are calculated from other curves
            ref_enc_path, ref_scaler_path = _lstm_ae_paths(dataset_name[0])

            res_ref = evaluate_outlier_filters(
                X_curves=trained_curve,
                features_df=features_df,
                y_encoded=y_full,
                outlier_filters=outlier_filters,
                dataset_name=clean_title,
                mode_name="Reference",
                cached_results=cached_ref,
                models=models,
                checkpoint_fn=checkpoint_ref,
                KFS=top_10_features,
                rerun_models=config.RERUN_MODELS,
                n_splits=args.n_splits,
                save_model_dir=ref_save_dir,
                save_model_curve_type=ref_save_ct,
                pretrained_encoder_path=ref_enc_path,
                pretrained_scaler_path=ref_scaler_path,
                coords=coords_full,
                well_ids=well_ids_full,
                k_neighbors=args.k_neighbors,
                multitask=args.mtl,
                y_concentration=(None if getattr(args, 'lbl_conc', False) else y_concentration),
                cl_phase1_epochs=getattr(args, 'cl_phase1_epochs', None),
                lc_classes=(encoder.classes_ if getattr(args, 'lbl_conc', False) else None),
            )

            all_ml_results[clean_title]["Reference"] = res_ref
            safe_joblib_dump(all_ml_results, results_file_path, compress=3)

            prefix_ref = os.path.join(model_plot_path, f"{name}_Reference")
            plot_ml_results(
                results_dict=all_ml_results[clean_title]["Reference"],
                outlier_filters=outlier_filters,
                dataset_name=clean_title,
                mode_name="Reference Training",
                total_count=total_samples,
                save_prefix=prefix_ref,
            )

        # --- NATIVE TRAINING (this dataset's outlier filters AND training curves) ---
        if "native" in args.training_mode:
            print(f"\n  [MODE] NATIVE TRAINING")
            cached_reference = all_ml_results[clean_title].get("Reference")
            if np.array_equal(curves_2d, trained_curve) and cached_reference:
                print(f"  [*] '{clean_title}' curves are identical to the Reference training curves. Reusing saved Reference results, skipping retraining.")
                res_native = cached_reference
            else:
                cached_native = all_ml_results[clean_title].get("Native", {})
                checkpoint_native = make_checkpoint_fn(all_ml_results, results_file_path, clean_title, "Native")

                native_enc_path, native_scaler_path = _lstm_ae_paths(name)

                res_native = evaluate_outlier_filters(
                    X_curves=curves_2d,
                    features_df=features_df,
                    y_encoded=y_full,
                    outlier_filters=outlier_filters,
                    dataset_name=clean_title,
                    mode_name="Native",
                    cached_results=cached_native,
                    models=models,
                    checkpoint_fn=checkpoint_native,
                    KFS=top_10_features,
                    rerun_models=config.RERUN_MODELS,
                    n_splits=args.n_splits,
                    save_model_dir=model_interp_dir,
                    save_model_curve_type=xai_curve_type,
                    pretrained_encoder_path=native_enc_path,
                    pretrained_scaler_path=native_scaler_path,
                    coords=coords_full,
                    well_ids=well_ids_full,
                    k_neighbors=args.k_neighbors,
                    multitask=args.mtl,
                    y_concentration=(None if getattr(args, 'lbl_conc', False) else y_concentration),
                    cl_phase1_epochs=getattr(args, 'cl_phase1_epochs', None),
                    lc_classes=(encoder.classes_ if getattr(args, 'lbl_conc', False) else None),
                )

            all_ml_results[clean_title]["Native"] = res_native
            safe_joblib_dump(all_ml_results, results_file_path, compress=3)

            prefix_native = os.path.join(model_plot_path, f"{name}_Native")
            plot_ml_results(
                results_dict=all_ml_results[clean_title]["Native"],
                outlier_filters=outlier_filters,
                dataset_name=clean_title,
                mode_name="Native Training",
                total_count=total_samples,
                save_prefix=prefix_native,
            )

        if "top_10_features" not in all_ml_results[clean_title]:
            all_ml_results[clean_title]["top_10_features"] = {}
        for f in outlier_filters:
            all_ml_results[clean_title]["top_10_features"][str(f)] = top_10_features
        safe_joblib_dump(all_ml_results, results_file_path, compress=3)
        print(f"  [XAI] Saved feature metadata into {results_file_path}")

        gc.collect()