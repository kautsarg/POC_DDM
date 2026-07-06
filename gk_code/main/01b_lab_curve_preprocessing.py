import os
import sys
sys.path.insert(0, 'utils')
import re
import itertools
import argparse
import numpy as np
import pandas as pd
import joblib
from safe_io import safe_joblib_dump
from pathlib import Path
from joblib import Parallel, delayed

import pywt
import config
import sigmoid_fitting as sp

# ==========================================
# LAB_DDM_paper dataset registry
# ==========================================
# Unlike 01_curve_preprocessing_v6.py (pixel-grid chip data with row/col/well
# coordinates), these are flat per-sample qPCR/qdLAMP curves read straight out
# of a CSV -- one row per sample, no spatial structure at all. Each entry here
# is a subfolder of config.LAB_EXP_FOLDER.
FILE_MAPPING = {
    'ACA_qdPCR':   'dPCR_Dataset_Pure.csv',
    'AMCA_qdPCR':  'dPCR_Amplification_Curves.csv',
    'AMCA_qdLAMP': '5Plex_dLAMP_oldData.csv',
    'Z_area': 'area_strategy.csv',
    'Z_range': 'range_strategy.xlsx',
    'Z_range_filtered': 'range_strategy_filtered.xlsx',
}
FILE_CONC = {
    'ACA_qdPCR': 'Conc',
    'AMCA_qdPCR':  'Conc',
    'AMCA_qdLAMP': 'Conc',
    'Z_area': None,
    'Z_range': None,
    'Z_range_filtered': None
}

FILE_MAPPING_ONE_TO_ONE = {
    '1_area': 'area_strategy.csv',
    '2_range': 'range_strategy.xlsx',
    '3_range_filtered': 'range_strategy_filtered.xlsx',
}
FILE_TARGET = {
    'ACA_qdPCR':   'LoadedPanels',
    'AMCA_qdPCR':  'Target',
    'AMCA_qdLAMP': 'Target',
    'Z_area':   'Target',
    'Z_range':  'Target',
    'Z_range_filtered': 'Target',
}

FILE_TARGET_ONE_TO_ONE = {
    '1_area':   'Target',
    '2_range':  'Target',
    '3_range_filtered': 'Target',
}

def get_numeric_sort_key(col_name):
    nums = re.findall(r'\d+\.?\d*', col_name)
    return float(nums[0]) if nums else 0.0


def _fit_single_curve(y_row, x_time, start_idx=0):
    try:
        y_num = np.asanyarray(y_row, dtype=np.float64)
        x_num = np.asanyarray(x_time, dtype=np.float64)
        y_fit = y_num[start_idx:]
        x_fit = x_num[start_idx:]

        params, _ = sp.fit_5p(x_fit, y_fit, normalize=True)
        fitted_full = sp.sigmoid_5p(x_num, *params)
        fitted_segment = sp.sigmoid_5p(x_fit, *params)

        old_x_norm = np.linspace(0, 1, len(fitted_segment))
        new_x_norm = np.linspace(0, 1, len(x_num))
        fitted_stretched = np.interp(new_x_norm, old_x_norm, fitted_segment)
        rmse = np.sqrt(np.nanmean(np.square(fitted_segment - y_fit)))

        return fitted_full, fitted_stretched, params, rmse

    except Exception:
        nan_array = np.full_like(x_time, np.nan, dtype=np.float64)
        return nan_array, nan_array, np.full(5, np.nan), np.nan


def sigmoid_fitting_5p(curves, ori_timestamps, starting_idxs=None):
    if starting_idxs is None:
        starting_idxs = np.zeros(len(curves), dtype=int)

    results = Parallel(n_jobs=-1, backend="loky", batch_size='auto')(
        delayed(_fit_single_curve)(y, ori_timestamps, t)
        for y, t in zip(curves, starting_idxs)
    )

    curves_out_full, curves_out_stretched, params_out, rmse_out = zip(*results)
    return np.array(curves_out_full), np.array(curves_out_stretched), np.array(params_out), np.array(rmse_out)


def load_lab_curves(exp_path):
    """Read this experiment's CSV/Excel file and return (curves, timestamps, well_labels, concentration)."""
    folder = exp_path.name
    file_path = os.path.join(exp_path, FILE_MAPPING[folder])
    df = _read_table(file_path)

    cycle_cols = [col for col in df.columns
                  if str(col).startswith('Cycle') or re.match(r'^\d+(\.\d+)?$', str(col))]
    sorted_cycle_cols = sorted(cycle_cols, key=get_numeric_sort_key)

    timestamps = np.array([get_numeric_sort_key(col) for col in sorted_cycle_cols])
    curves = df[sorted_cycle_cols].to_numpy()
    well_labels = df[FILE_TARGET[folder]].to_numpy()

    conc_col = FILE_CONC.get(folder)
    concentration = df[conc_col].to_numpy() if conc_col else None

    return curves, timestamps, well_labels, concentration



def _read_table(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(file_path)
    return pd.read_csv(file_path)


def discover_one_to_one_combos(exp_folder):
    combos = []
    for strategy in sorted(FILE_MAPPING_ONE_TO_ONE.keys()):
        file_path = os.path.join(exp_folder, strategy, FILE_MAPPING_ONE_TO_ONE[strategy])
        target_col = FILE_TARGET_ONE_TO_ONE[strategy]
        labels = sorted(_read_table(file_path)[target_col].unique())
        for label1, label2 in itertools.combinations(labels, 2):
            combo_name = f"{strategy}__{label1}_vs_{label2}"
            combos.append((strategy, label1, label2, combo_name))
    return combos


def _strategy_fit_cache_path(exp_folder, strategy):
    return os.path.join(exp_folder, f"_one_to_one_fitted_cache__{strategy}.joblib")


def load_and_fit_strategy_once(exp_folder, strategy):
    cache_path = _strategy_fit_cache_path(exp_folder, strategy)
    if os.path.exists(cache_path):
        try:
            return joblib.load(cache_path)
        except Exception as e:
            print(f"  -> [WARNING] Strategy fit cache at {cache_path} unreadable ({e}). Recomputing.")

    file_path = os.path.join(exp_folder, strategy, FILE_MAPPING_ONE_TO_ONE[strategy])
    target_col = FILE_TARGET_ONE_TO_ONE[strategy]
    df = _read_table(file_path)

    cycle_cols = [col for col in df.columns
                  if str(col).startswith('Cycle') or re.match(r'^\d+(\.\d+)?$', str(col))]
    sorted_cycle_cols = sorted(cycle_cols, key=get_numeric_sort_key)
    timestamps = np.array([get_numeric_sort_key(col) for col in sorted_cycle_cols])

    curves = df[sorted_cycle_cols].to_numpy()
    well_labels = df[target_col].to_numpy()

    print(f"  -> Fitting sigmoid curves for ALL of strategy '{strategy}' once ({len(curves)} curves)...")
    sigmoid_results = sigmoid_fitting_5p(curves, timestamps)

    cached = {
        "curves": curves, "timestamps": timestamps, "well_labels": well_labels,
        "sigmoid_results": sigmoid_results, "concentration": None,
    }
    safe_joblib_dump(cached, cache_path, compress=3)
    return cached


def load_lab_curves_one_to_one(exp_folder, strategy, label1, label2):
    cached = load_and_fit_strategy_once(exp_folder, strategy)
    mask = np.isin(cached["well_labels"], [label1, label2])

    curves = cached["curves"][mask]
    well_labels = cached["well_labels"][mask]
    fitted_full, fitted_stretched, params, rmse = cached["sigmoid_results"]
    sigmoid_results = (fitted_full[mask], fitted_stretched[mask], params[mask], rmse[mask])

    return curves, cached["timestamps"], well_labels, sigmoid_results, cached.get("concentration")


def is_cache_hit(save_path):
    if not os.path.exists(save_path):
        return False
    try:
        existing_data = joblib.load(save_path)
    except Exception as e:
        print(f"  -> [WARNING] Cached file at {save_path} is corrupted ({e}). Recomputing from scratch...")
        return False
    return existing_data is not None and "curves" in existing_data


def patch_missing_variants(save_path, normalize_curves, wavelet_sym8):
    """On a cache hit, add missing curve variants from the already-cached 'ori_curves'
    instead of forcing a full re-run (re-read CSV/Excel + re-fit sigmoids)."""
    if not normalize_curves and not wavelet_sym8:
        return
    existing_data = joblib.load(save_path)
    patched = []
    if normalize_curves and "ori_curves_norm" not in existing_data["curves"]:
        existing_data["curves"]["ori_curves_norm"] = normalize_curves_minmax(existing_data["curves"]["ori_curves"])
        patched.append("ori_curves_norm")
    if wavelet_sym8 and "ori_curves_wavelet_sym8" not in existing_data["curves"]:
        existing_data["curves"]["ori_curves_wavelet_sym8"] = wavelet_denoise_curves(existing_data["curves"]["ori_curves"])
        patched.append("ori_curves_wavelet_sym8")
    if patched:
        safe_joblib_dump(existing_data, save_path, compress=3)
        print(f"  -> Patched missing {', '.join(patched)} into {save_path}")


def normalize_curves_minmax(curves):
    curves = np.asarray(curves, dtype=np.float64)
    row_min = curves.min(axis=1, keepdims=True)
    row_max = curves.max(axis=1, keepdims=True)
    denom = np.where(row_max - row_min == 0, 1, row_max - row_min)
    return (curves - row_min) / denom


def wavelet_denoise_curves(curves, wavelet="sym8", level=5):
    curves = np.asarray(curves, dtype=np.float64)
    out = np.empty_like(curves)
    for i, c in enumerate(curves):
        coeffs = pywt.wavedec(c, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        thr = sigma * np.sqrt(2 * np.log(len(c)))
        new_coeffs = [coeffs[0]] + [pywt.threshold(d, thr, mode="soft") for d in coeffs[1:]]
        out[i] = pywt.waverec(new_coeffs, wavelet)[:len(c)]
    return out


def compute_ttp(curves, threshold_frac=0.1):
    """Per-curve first-cycle index reaching threshold_frac of that curve's own range,
    anchored to the curve's own starting value (cycle 0), not its global min."""
    max_vals = curves.max(axis=1, keepdims=True)
    min_vals = curves[:, [0]]
    threshold_vals = min_vals + (max_vals - min_vals) * threshold_frac
    return np.argmax(curves >= threshold_vals, axis=1)


def _ttp_tradeoff_curve(ttp_idx, well_labels, n_cycles):
    unique_ttps = sorted(set(ttp_idx.tolist()))
    targets = sorted(set(well_labels))
    min_ttp = min(unique_ttps)
    total_counts = {t: int((well_labels == t).sum()) for t in targets}

    cycles_left_list, min_retention_list = [], []
    for max_ttp in unique_ttps:
        keep = ttp_idx <= max_ttp
        cycles_left = n_cycles - (max_ttp - min_ttp)
        retentions = [100 * int(((well_labels == t) & keep).sum()) / total_counts[t] for t in targets]
        cycles_left_list.append(cycles_left)
        min_retention_list.append(min(retentions))
    return np.array(unique_ttps), np.array(cycles_left_list), np.array(min_retention_list)


def _choose_max_ttp_by_knee(unique_ttps, cycles_left, min_retention):
    """max_ttp at the point of max perpendicular distance from the line connecting the
    tradeoff curve's two endpoints (cycles_left vs worst-case label retention)."""
    x = (cycles_left - cycles_left.min()) / (cycles_left.max() - cycles_left.min() + 1e-9)
    y = (min_retention - min_retention.min()) / (min_retention.max() - min_retention.min() + 1e-9)
    p1, p2 = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    line_vec_norm = (p2 - p1) / np.linalg.norm(p2 - p1)
    points = np.stack([x, y], axis=1)
    vecs = points - p1
    proj_points = p1 + np.outer(vecs @ line_vec_norm, line_vec_norm)
    dist = np.linalg.norm(points - proj_points, axis=1)
    return int(unique_ttps[np.argmax(dist)])


def build_ttp_aligned_curves(curves, timestamps, well_labels, preserve_cycles=None, threshold_frac=0.1):
    """TTP-align + baseline-correct curves: compute TTP, pick max_ttp (knee-detected by
    default, or derived from preserve_cycles if given), drop curves whose TTP exceeds it,
    shift+crop survivors to a common length, then subtract each curve's own (post-shift)
    first value so they all start at 0."""
    ttp_idx = compute_ttp(curves, threshold_frac)
    n_cycles = curves.shape[1]
    min_ttp = int(ttp_idx.min())

    if preserve_cycles is not None:
        max_ttp = min_ttp + (n_cycles - preserve_cycles)
    else:
        unique_ttps, cycles_left, min_retention = _ttp_tradeoff_curve(ttp_idx, well_labels, n_cycles)
        max_ttp = _choose_max_ttp_by_knee(unique_ttps, cycles_left, min_retention)

    keep = ttp_idx <= max_ttp
    curves_kept = curves[keep]
    well_labels_kept = well_labels[keep]
    shifts = ttp_idx[keep] - min_ttp
    max_shift = int(shifts.max())
    final_len = n_cycles - max_shift

    aligned = np.empty((curves_kept.shape[0], final_len))
    for i in range(curves_kept.shape[0]):
        s = int(shifts[i])
        aligned[i] = curves_kept[i, s:s + final_len]

    baseline_corrected = aligned - aligned[:, [0]]
    aligned_timestamps = timestamps[:final_len]

    return baseline_corrected, aligned_timestamps, well_labels_kept, max_ttp, keep


def save_experiment_data(save_exp_path, curves, timestamps, well_labels, sigmoid_results, normalize_curves=False, wavelet_sym8=False, concentration=None):
    fitted_full, fitted_stretched, params, rmse = sigmoid_results

    curves_dict = {"ori_curves": curves}
    if normalize_curves:
        curves_dict["ori_curves_norm"] = normalize_curves_minmax(curves)
    if wavelet_sym8:
        curves_dict["ori_curves_wavelet_sym8"] = wavelet_denoise_curves(curves)

    save_data = {
        "curves": curves_dict,
        "sigmoid_curves": {
            "original": {
                "fitted_full": fitted_full,
                "fitted_stretched": fitted_stretched,
                "params": params,
                "rmse": rmse,
            }
        },
        "timestamps": timestamps,
        "well_labels": well_labels,
        "concentration": concentration,
        "metadata": {},
        "baseline_value": 0.0,
    }

    save_path = os.path.join(save_exp_path, config.TRAINING_DATA_PATH)
    existing_state = {}
    if os.path.exists(save_path):
        try:
            existing_state = joblib.load(save_path)
        except Exception as e:
            print(f"  -> [WARNING] Existing shared cache at {save_path} unreadable ({e}). Overwriting.")
            existing_state = {}
    existing_state.update(save_data)   # only overwrites this script's own keys -- 02's keys
    safe_joblib_dump(existing_state, save_path, compress=3)   # (dataset, kinetic_features, ...) are left untouched
    print(f"  -> Saved numerical results and metadata to {save_path}")


if __name__ == "__main__":
    print(f"\n{'='*70}\n[RUNNING] {os.path.basename(__file__)}\n{'='*70}\n")
    parser = argparse.ArgumentParser(description="Lab (LAB_DDM_paper) Curve Preprocessing Pipeline")
    parser.add_argument("--task_id", type=int, default=0, help="Array Job ID")
    parser.add_argument("--exp_folder", type=str, default=config.LAB_EXP_FOLDER, help="Path to LAB_DDM_paper experiment datasets")
    parser.add_argument("--force_rerun", action="store_true", help="Recompute and overwrite even if a presaved file already exists")
    parser.add_argument("--one_to_one", action="store_true",
                        help="Pairwise one-to-one mode (use with --exp_folder=config.LAB_1TO1_EXP_FOLDER): "
                             "build one curve_for_training.joblib per (strategy, label-pair) combination, "
                             "in its own subfolder, instead of one shared joblib per strategy folder.")
    parser.add_argument("--normalize_curves", action="store_true",
                        help="Add an 'ori_curves_norm' variant: each curve independently min-max scaled to "
                             "[0,1]. Selectable downstream via --curve_type ori_curve_norm.")
    parser.add_argument("--wavelet_sym8", action="store_true",
                        help="Add an 'ori_curves_wavelet_sym8' variant: sym8 wavelet denoising with "
                             "Donoho-Johnstone universal threshold. Selectable via --curve_type ori_curve_wavelet_sym8.")
    parser.add_argument("--ori_curve_aligned", action="store_true",
                        help="Also build a separate, self-contained TTP-aligned + baseline-corrected "
                             "dataset in a sibling '<name>_aligned' folder (its own curves/well_labels, "
                             "since TTP filtering drops some curves -- no row-count mismatch with the "
                             "regular dataset). Processed by 02-08 like any other experiment folder.")
    parser.add_argument("--preserve_cycles", type=int, default=None,
                        help="With --ori_curve_aligned: directly specify how many cycles to preserve "
                             "after alignment, instead of auto-picking max_ttp via knee detection.")
    args = parser.parse_args()

    if args.one_to_one:
        combos = discover_one_to_one_combos(args.exp_folder)
        for _, _, _, combo_name in combos:
            os.makedirs(os.path.join(args.exp_folder, combo_name), exist_ok=True)

        if args.task_id >= len(combos):
            print(f"Task ID {args.task_id} is out of bounds for {len(combos)} combinations. Exiting.")
            sys.exit(0)

        strategy, label1, label2, combo_name = combos[args.task_id]
        exp_path = Path(args.exp_folder, combo_name)
        save_path = os.path.join(exp_path, config.TRAINING_DATA_PATH)
        aligned_exp_path = Path(str(exp_path) + "_aligned") if args.ori_curve_aligned else None
        aligned_save_path = os.path.join(aligned_exp_path, config.TRAINING_DATA_PATH) if aligned_exp_path else None

        main_cache_hit = is_cache_hit(save_path) and not args.force_rerun
        aligned_cache_hit = (not args.ori_curve_aligned) or (is_cache_hit(aligned_save_path) and not args.force_rerun)

        if main_cache_hit and aligned_cache_hit:
            patch_missing_variants(save_path, args.normalize_curves, args.wavelet_sym8)
            if args.ori_curve_aligned:
                patch_missing_variants(aligned_save_path, normalize_curves=True, wavelet_sym8=False)
            print(f"Cache hit: {exp_path}")
            print("  ✓ Experiment complete!\n")
            sys.exit(0)

        print(f"Processing One-to-One Combo: {combo_name}  ({strategy}: {label1} vs {label2})")

        curves, timestamps, well_labels, sigmoid_results, concentration = load_lab_curves_one_to_one(
            args.exp_folder, strategy, label1, label2)

        if main_cache_hit:
            patch_missing_variants(save_path, args.normalize_curves, args.wavelet_sym8)
        else:
            save_experiment_data(exp_path, curves, timestamps, well_labels, sigmoid_results,
                                normalize_curves=args.normalize_curves,
                                wavelet_sym8=args.wavelet_sym8,
                                concentration=concentration)

        print(f"  -> X (Curves) shape:       {curves.shape}")
        print(f"  -> well_labels (Targets):  {np.unique(well_labels)}")
        print(f"  -> Mean RMSE Fit Error:    {np.nanmean(sigmoid_results[3]):.4f}")
        print("  ✓ Experiment complete!\n")

        if args.ori_curve_aligned and not aligned_cache_hit:
            aligned_exp_path.mkdir(parents=True, exist_ok=True)
            print(f"Processing TTP-aligned dataset: {aligned_exp_path}")
            aligned_curves, aligned_timestamps, aligned_well_labels, max_ttp, keep = build_ttp_aligned_curves(
                curves, timestamps, well_labels, preserve_cycles=args.preserve_cycles)

            print("  -> Processing Sigmoid Curves (aligned)...")
            aligned_sigmoid_results = sigmoid_fitting_5p(aligned_curves, aligned_timestamps)
            aligned_concentration = concentration[keep] if concentration is not None else None

            save_experiment_data(aligned_exp_path, aligned_curves, aligned_timestamps,
                                aligned_well_labels, aligned_sigmoid_results,
                                normalize_curves=True, concentration=aligned_concentration)

            print(f"  -> max_ttp used:           {max_ttp}")
            print(f"  -> X (Curves) shape:       {aligned_curves.shape} (from {curves.shape})")
            print(f"  -> well_labels (Targets):  {np.unique(aligned_well_labels)}")
            print(f"  -> Mean RMSE Fit Error:    {np.nanmean(aligned_sigmoid_results[3]):.4f}")
            print("  ✓ Aligned experiment complete!\n")
        elif args.ori_curve_aligned:
            print(f"Cache hit: {aligned_exp_path}")
            patch_missing_variants(aligned_save_path, normalize_curves=True, wavelet_sym8=False)
            print("  ✓ Aligned experiment complete!\n")

        sys.exit(0)

    exp_paths = sorted([Path(args.exp_folder, name) for name in os.listdir(args.exp_folder)
                        if os.path.isdir(os.path.join(args.exp_folder, name)) and name in FILE_MAPPING])

    if args.task_id >= len(exp_paths):
        print(f"Task ID {args.task_id} is out of bounds for {len(exp_paths)} folders. Exiting.")
        sys.exit(0)

    exp_path = exp_paths[args.task_id]
    save_path = os.path.join(exp_path, config.TRAINING_DATA_PATH)
    aligned_exp_path = Path(str(exp_path) + "_aligned") if args.ori_curve_aligned else None
    aligned_save_path = os.path.join(aligned_exp_path, config.TRAINING_DATA_PATH) if aligned_exp_path else None

    main_cache_hit = is_cache_hit(save_path) and not args.force_rerun
    aligned_cache_hit = (not args.ori_curve_aligned) or (is_cache_hit(aligned_save_path) and not args.force_rerun)

    if main_cache_hit and aligned_cache_hit:
        patch_missing_variants(save_path, args.normalize_curves, args.wavelet_sym8)
        if args.ori_curve_aligned:
            patch_missing_variants(aligned_save_path, normalize_curves=True, wavelet_sym8=False)
        print(f"Cache hit: {exp_path}")
        print("  ✓ Experiment complete!\n")
        sys.exit(0)

    print(f"Processing Experiment: {exp_path}")

    curves, timestamps, well_labels, concentration = load_lab_curves(exp_path)

    if main_cache_hit:
        patch_missing_variants(save_path, args.normalize_curves, args.wavelet_sym8)
    else:
        print("  -> Processing Sigmoid Curves...")
        sigmoid_results = sigmoid_fitting_5p(curves, timestamps)
        save_experiment_data(exp_path, curves, timestamps, well_labels, sigmoid_results,
                            normalize_curves=args.normalize_curves,
                            concentration=concentration)
        print(f"  -> X (Curves) shape:       {curves.shape}")
        print(f"  -> well_labels (Targets):  {np.unique(well_labels)}")
        print(f"  -> Mean RMSE Fit Error:    {np.nanmean(sigmoid_results[3]):.4f}")

    print("  ✓ Experiment complete!\n")

    if args.ori_curve_aligned and not aligned_cache_hit:
        aligned_exp_path.mkdir(parents=True, exist_ok=True)
        print(f"Processing TTP-aligned dataset: {aligned_exp_path}")
        aligned_curves, aligned_timestamps, aligned_well_labels, max_ttp, keep = build_ttp_aligned_curves(
            curves, timestamps, well_labels, preserve_cycles=args.preserve_cycles)

        print("  -> Processing Sigmoid Curves (aligned)...")
        aligned_sigmoid_results = sigmoid_fitting_5p(aligned_curves, aligned_timestamps)
        aligned_concentration = concentration[keep] if concentration is not None else None

        save_experiment_data(aligned_exp_path, aligned_curves, aligned_timestamps,
                            aligned_well_labels, aligned_sigmoid_results,
                            normalize_curves=True, concentration=aligned_concentration)

        print(f"  -> max_ttp used:           {max_ttp}")
        print(f"  -> X (Curves) shape:       {aligned_curves.shape} (from {curves.shape})")
        print(f"  -> well_labels (Targets):  {np.unique(aligned_well_labels)}")
        print(f"  -> Mean RMSE Fit Error:    {np.nanmean(aligned_sigmoid_results[3]):.4f}")
        print("  ✓ Aligned experiment complete!\n")
    elif args.ori_curve_aligned:
        print(f"Cache hit: {aligned_exp_path}")
        patch_missing_variants(aligned_save_path, normalize_curves=True, wavelet_sym8=False)
        print("  ✓ Aligned experiment complete!\n")
