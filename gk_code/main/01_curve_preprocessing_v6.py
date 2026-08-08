import os
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.ndimage import convolve1d, uniform_filter1d
from joblib import Parallel, delayed

import pywt
from scipy.signal import savgol_filter
import config

# Add custom paths
sys.path.insert(0, 'utils')
from safe_io import safe_joblib_dump
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
sys.path.insert(0, '0_4_AMCA Code on Chip')
sys.path.insert(0, 'utils/01_curve_preprocessing')

from titan_v4.Experiment import Experiment as Experiment_v4
from titan_v4.load_and_preprocessing import titan_load_and_preprocessing as titan_load_and_preprocessing_v4
import sigmoid_fitting as sp

from chip_v6_utils import load_and_preprocess_v6

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================
# 1. MATH & EXTRACTION MODULES
# ==========================================

def moving_average_vec(data, window_size=10):
    mask = np.ones(window_size) / window_size
    return convolve1d(data, mask, axis=-1, mode='nearest')

def first_pos_zero_crossing_vec(data_2d):
    crossings = (data_2d[:, 1:] >= 0) & (data_2d[:, :-1] < 0)
    indices = np.argmax(crossings, axis=1) + 1
    exists = np.any(crossings, axis=1)
    return np.where(exists, indices, 0)

def lowest_integral_to_next_crossing_vec(data_2d):
    def single_lowest_exit(arr):
        sign_changes = np.nonzero(np.diff(np.sign(arr)))[0] + 1
        boundaries = np.concatenate(([0], sign_changes, [len(arr)]))
        lowest_sum = 0
        best_exit_idx = 0 
        c_sum = np.cumsum(arr)
        
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i+1]
            segment_sum = c_sum[end-1] - (c_sum[start-1] if start > 0 else 0)
            if segment_sum < lowest_sum:
                lowest_sum = segment_sum
                best_exit_idx = end
        return best_exit_idx
    
    return np.fromiter((single_lowest_exit(row) for row in data_2d), dtype=int, count=len(data_2d))

def apply_baseline_cleaning(curves, cross_indices, margin):
    cleaned = curves.copy()
    actual_indices = np.clip(cross_indices + margin, 0, curves.shape[1] - 1)
    mask = cross_indices > 0
    for i in range(len(curves)):
        if mask[i]:
            idx = actual_indices[i]
            cleaned[i, :idx] = curves[i, idx]
    return cleaned, np.where(mask, actual_indices, 0)

def get_derivatives(curves_batch, timestamps):
    return Parallel(n_jobs=-1, backend="loky", batch_size='auto')(
        delayed(sp.calculate_first_derivative)(timestamps, y) for y in curves_batch
    )
    
def reconstruct_data(all_exp_data, attr_str):   
    if not all_exp_data:
        return None, np.array([]), None
        
    X_time = all_exp_data[0].wells_list[0].time

    arrays_to_stack = []
    Y_well_list = []

    for exp_data in all_exp_data:
        for well_id, well in enumerate(exp_data.wells_list):
            temp_x = getattr(well, attr_str).copy()
            temp_x = np.swapaxes(temp_x, 0, 1) 
            
            arrays_to_stack.append(temp_x)
            
            num_samples = temp_x.shape[0]
            Y_well_list.extend([well_id] * num_samples)

    vstacked = np.vstack(arrays_to_stack) if arrays_to_stack else None

    return X_time, np.array(Y_well_list), vstacked

# ==========================================
# 2. DATAFRAME GENERATION (PIXELS & TEMPS)
# ==========================================

def extract_pixel_temp_dataframes(all_exp_data):
    df_pix_lin_list, df_pix_nl_list = [], []
    df_temp_lin_list, df_temp_nl_list = [], []

    for vref_idx, exp_data in enumerate(all_exp_data):
        
        for w_idx, well in enumerate(exp_data.wells_list):
            idx_settled = well.idx_settled
            idx_active = well.idx_active
            time_npr = well.time_npr

            well_nrows, well_ncols = well.well_nrows, well.well_ncols
            well_temp_nrows, well_temp_ncols = well.well_temp_nrows, well.well_temp_ncols

            well_temp_lin2d = well.well_temp_lin2d
            well_2d_temp_npr = well.well_2d_temp_npr

            y, x = np.indices((well_nrows, well_ncols))
            temp_group_idx = ((y // 5) * well_temp_ncols + (x // 5)).flatten()

            active_y = np.asarray(y.flatten()[idx_active])
            active_x = np.asarray(x.flatten()[idx_active])
            active_temp_mapping = np.asarray(temp_group_idx[idx_active])

            n_time_lin = well.well_3d_lin.shape[2]
            well_2d_lin = well.well_3d_lin.reshape(-1, n_time_lin, order='C').T
            well_2d_bs = well_2d_lin - well_2d_lin[idx_settled, :]
            well_2d_bs_active = well_2d_bs[:, idx_active]

            n_time_nl = well.well_3d_npr.shape[2]
            well_2d_nl = well.well_3d_npr.reshape(-1, n_time_nl, order='C').T
            well_2d_nl_bs = well_2d_nl - well_2d_nl[idx_settled, :]
            well_2d_nl_bs_active = well_2d_nl_bs[:, idx_active]

            time_cols = [f"Cycle_{t}" for t in time_npr]
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_temp_lin = np.nanmean(well_temp_lin2d, axis=0)
                mean_temp_nl = np.nanmean(well_2d_temp_npr, axis=0)

            # --- Pixel Linear DataFrame ---
            df_pl = pd.DataFrame(well_2d_bs_active.T, columns=time_cols)
            df_pl['well_id'] = w_idx
            df_pl['pixel_row_idx'] = active_y
            df_pl['pixel_col_idx'] = active_x
            df_pl['temp_group_idx'] = active_temp_mapping
            counts = df_pl['temp_group_idx'].value_counts()
            df_pl['num_active_pixels_in_temp_group'] = df_pl['temp_group_idx'].map(counts)
            df_pl['well_temp_lin2d_mean'] = np.asarray(mean_temp_lin[active_temp_mapping])
            df_pl['well_2d_temp_npr_mean'] = np.asarray(mean_temp_nl[active_temp_mapping])
            df_pl['vref_idx'] = vref_idx

            # --- Pixel Non-Linear DataFrame ---
            df_pnl = pd.DataFrame(well_2d_nl_bs_active.T, columns=time_cols)
            df_pnl['well_id'] = w_idx
            df_pnl['pixel_row_idx'] = active_y
            df_pnl['pixel_col_idx'] = active_x
            df_pnl['temp_group_idx'] = active_temp_mapping
            df_pnl['num_active_pixels_in_temp_group'] = df_pnl['temp_group_idx'].map(counts)
            df_pnl['well_temp_lin2d_mean'] = np.asarray(mean_temp_lin[active_temp_mapping])
            df_pnl['well_2d_temp_npr_mean'] = np.asarray(mean_temp_nl[active_temp_mapping])
            df_pnl['vref_idx'] = vref_idx

            # Standardize pixel column order
            meta_cols = ['well_id', 'pixel_row_idx', 'pixel_col_idx', 'temp_group_idx',
                        'num_active_pixels_in_temp_group', 'well_temp_lin2d_mean', 'well_2d_temp_npr_mean', 'vref_idx']

            df_pl = df_pl[meta_cols + time_cols]
            df_pnl = df_pnl[meta_cols + time_cols]

            # --- Temperature Linear DataFrame ---
            df_tl = pd.DataFrame(well_temp_lin2d.T, columns=time_cols)
            df_tl.insert(0, 'well_id', w_idx)
            df_tl.insert(1, 'temp_group_idx', np.arange(well_temp_lin2d.shape[1]))
            df_tl.insert(2, 'vref_idx', vref_idx)

            # --- Temperature Non-Linear DataFrame ---
            df_tnl = pd.DataFrame(well_2d_temp_npr.T, columns=time_cols)
            df_tnl.insert(0, 'well_id', w_idx)
            df_tnl.insert(1, 'temp_group_idx', np.arange(well_2d_temp_npr.shape[1]))
            df_tnl.insert(2, 'vref_idx', vref_idx)


            df_pix_lin_list.append(df_pl)
            df_pix_nl_list.append(df_pnl)
            df_temp_lin_list.append(df_tl)
            df_temp_nl_list.append(df_tnl)

    return {
        "well_2d_bs_active_df": pd.concat(df_pix_lin_list, ignore_index=True),
        "well_2d_nl_bs_active_df": pd.concat(df_pix_nl_list, ignore_index=True),
        "well_temp_lin2d_df": pd.concat(df_temp_lin_list, ignore_index=True),
        "well_2d_temp_npr_df": pd.concat(df_temp_nl_list, ignore_index=True)
    }

# ==========================================
# 3. DATA PROCESSING PIPELINE
# ==========================================

def process_experiment_data(ori_curves, ori_timestamps, window_size_ori, window_size_1stder, margin,
                            compute_sigmoid_fits=False):
    ori_curves_avg = moving_average_vec(ori_curves, window_size_ori)
    # Baseline each curve to start at y=0. Must subtract per-row (axis=1 slice, not a
    # single index) — ori_curves_avg[0] would be pixel 0's first value, subtracted from
    # every pixel; [:, 0:1] keeps the (N_pixels, 1) shape so each row is zeroed by its own start.
    ori_curves_avg = ori_curves_avg - ori_curves_avg[:, 0:1]

    if not compute_sigmoid_fits:
        # Derivative/cleaning chain only feeds sigmoid fitting (run_all_fits) — skip it
        # entirely when sigmoid fits are disabled (default). See joblib_redundancy.md Change 1.
        processed_curves = [ori_curves, None, None, None, None]
        indices_dict = {"cleaned_idx": None, "cleaned_lowest_idx": None}
        return processed_curves, indices_dict, ori_curves_avg

    ori_curve_dydx = np.array(get_derivatives(ori_curves, ori_timestamps))
    ori_dydx_avg = moving_average_vec(ori_curve_dydx, window_size_1stder)

    cleaning_tasks = [
        ("std", ori_dydx_avg, first_pos_zero_crossing_vec),
        ("low", ori_dydx_avg, lowest_integral_to_next_crossing_vec),
    ]

    results = {}
    for name, deriv, func in cleaning_tasks:
        indices = func(deriv)
        cleaned_curves, actual_idxs = apply_baseline_cleaning(ori_curves, indices, margin)
        results[name] = (cleaned_curves, actual_idxs)

    indices_dict = {
        "cleaned_idx": results["std"][1],
        "cleaned_lowest_idx": results["low"][1],
    }

    processed_curves = [
        ori_curves, ori_curve_dydx, ori_dydx_avg,
        results["std"][0], results["low"][0],
    ]

    return processed_curves, indices_dict, ori_curves_avg


# ==========================================
# 4. SIGMOID FITTING MODULE
# ==========================================

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

def run_all_fits(processed_curves, indices_dict, ori_timestamps):
    raw_fits = {
        "original": sigmoid_fitting_5p(processed_curves[0], ori_timestamps, None),
        "cleaned_std": sigmoid_fitting_5p(processed_curves[3], ori_timestamps, indices_dict["cleaned_idx"]),
        "cleaned_lowest": sigmoid_fitting_5p(processed_curves[4], ori_timestamps, indices_dict["cleaned_lowest_idx"]),
    }
    
    structured_fits = {}
    for key, fit_tuple in raw_fits.items():
        structured_fits[key] = {
            "fitted_full": fit_tuple[0],
            "fitted_stretched": fit_tuple[1],
            "params": fit_tuple[2],
            "rmse": fit_tuple[3]
        }
        
    return structured_fits


# ==========================================
# 5. SAVING MODULE
# ==========================================

def normalize_curves_minmax(curves):
    curves = np.asarray(curves, dtype=np.float64)
    row_min = curves.min(axis=1, keepdims=True)
    row_max = curves.max(axis=1, keepdims=True)
    denom = np.where(row_max - row_min == 0, 1, row_max - row_min)
    return (curves - row_min) / denom


def _ensure_odd(w):
    return w + (1 - w % 2)


def _sg_derivative_scores(curves, windows, polyorder, sample_size=400, seed=0):
    """PRS / ROS sweep matching the notebook's _derivative_scores algorithm."""
    rng    = np.random.default_rng(seed)
    sample = curves[rng.choice(len(curves), size=min(sample_size, len(curves)), replace=False)]
    _ref_w = max(polyorder + 2, _ensure_odd(int(sample.shape[1] * 0.03)))
    ref    = savgol_filter(sample, window_length=_ref_w, polyorder=2, axis=1)
    peak_r = np.abs(np.diff(ref, axis=1)).max(axis=1)
    ro_raw = np.std(np.diff(np.diff(sample, axis=1), axis=1), axis=1)
    prs, ros = [], []
    for w in windows:
        w_  = max(_ensure_odd(int(w)), polyorder + 2)
        df  = np.diff(savgol_filter(sample, window_length=w_, polyorder=polyorder, axis=1), axis=1)
        prs.append(np.mean(np.abs(df).max(axis=1) / np.where(peak_r > 0, peak_r, 1)))
        ros.append(np.mean(np.std(np.diff(df, axis=1), axis=1) / np.where(ro_raw > 0, ro_raw, 1)))
    return np.array(prs), np.array(ros)


def _sg_sweet_spot(windows, roughnesses):
    """First window where ROS <= 2x 10th-percentile floor (matches notebook)."""
    floor = np.percentile(roughnesses, 10)
    sweet = np.where(roughnesses <= 2.0 * floor)[0]
    return int(windows[sweet[0]] if len(sweet) else windows[np.argmin(roughnesses)])


def sg_p4_denoise_curves(curves):
    """Auto-sweep optimal window for SG polyorder=4, return (denoised, optimal_w)."""
    T       = curves.shape[1]
    windows = np.unique([_ensure_odd(int(w))
                         for w in np.linspace(5, max(7, int(T * 0.25)), 40)])
    windows = windows[windows >= 6]   # polyorder=4 needs w >= 6
    _, ros  = _sg_derivative_scores(curves, windows.astype(float), polyorder=4)
    opt_w   = _sg_sweet_spot(windows, ros)
    return savgol_filter(curves, window_length=opt_w, polyorder=4, axis=1), opt_w


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


def save_experiment_data_restructured(save_exp_path, fitting_results, processed_curves,
                                     indices_dict, pixel_temp_dfs, baseline_value,
                                     Y_well, X_time, all_exp_data, ori_curves_avg,
                                     window_size_ori, window_size_1stder, margin, max_significant_index,
                                     compute_sigmoid_fits=False, normalize_curves=False,
                                     wavelet_sym8=False, wavelet_bior35=False, sg_p4=False,
                                     moving_avg=False):
    """
    Saves into the SAME file 02_outlier_detection_pipeline.py reads/extends
    (config.TRAINING_DATA_PATH) — 01 and 02 share one joblib per experiment;
    each owns its own keys and patches them in place (see joblib_redundancy.md).
    """
    df_meta = pixel_temp_dfs["well_2d_nl_bs_active_df"]
    well0 = all_exp_data[0].wells_list[0]

    # well_2d_bs_active/well_2d_nl_bs_active/well_temp_lin2d/well_2d_temp_npr/
    # well_temp_mean_then_lin are NOT persisted — confirmed zero consumers anywhere
    # (light_pipeline reconstructs well_2d_bs_active independently, never reads it here).
    curves_dict = {
        "ori_curves": processed_curves[0],
        "ori_curve_dydx": processed_curves[1],
        "ori_dydx_avg": processed_curves[2],
        "cleaned_std": processed_curves[3],
        "cleaned_lowest": processed_curves[4],
    }
    if moving_avg:
        curves_dict["ori_curves_avg"] = ori_curves_avg
    if wavelet_sym8:
        curves_dict["ori_curves_wavelet_sym8"] = wavelet_denoise_curves(processed_curves[0])
    _sg_w = None   # set unconditionally so save_data can always reference it
    if wavelet_bior35:
        curves_dict["ori_curves_wavelet_bior35"] = wavelet_denoise_curves(
            processed_curves[0], wavelet="bior3.5", level=5)
    if sg_p4:
        curves_dict["ori_curves_sg_p4"], _sg_w = sg_p4_denoise_curves(processed_curves[0])
        print(f"  -> SG p=4 optimal window: {_sg_w}")
    if normalize_curves:
        _norm_bases = [k for k in curves_dict
                       if k.startswith("ori_curves") and "dydx" not in k and not k.endswith("_norm")]
        for k in _norm_bases:
            curves_dict[f"{k}_norm"] = normalize_curves_minmax(curves_dict[k])
        print(f"  -> Added norm variants: {[k + '_norm' for k in _norm_bases]}")

    save_data = {
        "curves": curves_dict,
        "sigmoid_curves": fitting_results,
        "idxs": {
            "idx_start": well0.idx_start,
            "idx_settled": well0.idx_settled,
            "idx_end": well0.idx_end,
            "idx_active": well0.idx_active,
            "cleaned_idx": indices_dict.get("cleaned_idx"),
            "cleaned_lowest_idx": indices_dict.get("cleaned_lowest_idx"),
            "max_significant_index": max_significant_index,
        },
        "timestamps": X_time,
        "well_labels": Y_well,
        "metadata": {
            "pixel_row_idx": df_meta['pixel_row_idx'].values,
            "pixel_col_idx": df_meta['pixel_col_idx'].values,
            "temp_group_idx": df_meta['temp_group_idx'].values,
            "num_active_pixels_in_temp_group": df_meta['num_active_pixels_in_temp_group'].values,
            "well_temp_lin2d_mean": df_meta['well_temp_lin2d_mean'].values,
            "well_2d_temp_npr_mean": df_meta['well_2d_temp_npr_mean'].values,
            "vref_idx": df_meta['vref_idx'].values
        },
        "baseline_value": baseline_value,
        "window_size_ori": window_size_ori,
        "window_size_1stder": window_size_1stder if compute_sigmoid_fits else None,
        "margin": margin,
        "concentration": config.get_conc_array(Path(save_exp_path).name, Y_well),
        "sg_p4_optimal_w": _sg_w,
    }

    save_path = os.path.join(save_exp_path, config.TRAINING_DATA_PATH)
    existing_state = {}
    if os.path.exists(save_path):
        try:
            existing_state = joblib.load(save_path)
        except Exception as e:
            print(f"  -> [WARNING] Existing shared cache at {save_path} unreadable ({e}). Overwriting.")
            existing_state = {}
    existing_state.update(save_data)   # only overwrites 01's own keys — 02's keys (dataset,
    safe_joblib_dump(existing_state, save_path, compress=3)   # kinetic_features, ...) are left untouched
    print(f"  -> Saved numerical results and metadata to {save_path}")


# ==========================================
# 9. MAIN EXECUTION LOOP
# ==========================================

if __name__ == "__main__":
    print(f"\n{'='*70}\n[RUNNING] {os.path.basename(__file__)}\n{'='*70}\n")
    parser = argparse.ArgumentParser(description="Curve Preprocessing Pipeline")
    parser.add_argument("--task_id", type=int, default=0, help="Array Job ID")
    parser.add_argument("--exp_folder", type=str, default=config.DEFAULT_EXP_FOLDER, help="Path to experiment datasets")
    parser.add_argument("--n_wells", type=int, default=config.N_WELLS, help="Number of wells")
    parser.add_argument("--n_a_type", type=str, default=config.N_A_TYPE, help="Type of n_a")
    parser.add_argument("--nc_subtract", action="store_true", help="Apply baseline subtraction based on derivatives")
    parser.add_argument("--drop_pc", action="store_true",
                        help="Remove PC-labeled wells from the saved output after using them for truncation.")
    parser.add_argument("--moving_avg", action="store_true",
                        help="Save the moving-average variant (ori_curves_avg) in the output. "
                             "If omitted, only raw curves and denoising variants are stored.")
    parser.add_argument("--force_rerun", action="store_true", help="Recompute and overwrite even if a presaved file already exists")
    parser.add_argument("--compute_sigmoid_fits", action="store_true",
                        help="Compute the derivative/cleaning chain (ori_curve_dydx, ori_dydx_avg, cleaned_std, "
                             "cleaned_lowest) and the 5-parameter sigmoid fits derived from it. Unused by 02-08 "
                             "under default --curve_type args; off by default to save compute and storage.")
    parser.add_argument("--normalize_curves", action="store_true",
                        help="Add an 'ori_curves_norm' variant: each curve independently min-max scaled to "
                             "[0,1]. Selectable downstream via --curve_type ori_curve_norm.")
    parser.add_argument("--wavelet_sym8", action="store_true",
                        help="Add an 'ori_curves_wavelet_sym8' variant: sym8 wavelet denoising with "
                             "Donoho-Johnstone universal threshold. Selectable via --curve_type ori_curve_wavelet_sym8.")
    parser.add_argument("--wavelet_bior35", action="store_true",
                        help="Add an 'ori_curves_wavelet_bior35' variant: bior3.5 wavelet denoising with "
                             "Donoho-Johnstone universal threshold. Selectable via --curve_type ori_curve_wavelet_bior35.")
    parser.add_argument("--sg_p4", action="store_true",
                        help="Add an 'ori_curves_sg_p4' variant: SG polyorder=4 with auto window sweep "
                             "(PRS/ROS scoring). Selectable via --curve_type ori_curve_sg_p4.")
    args = parser.parse_args()

    n_wells = args.n_wells
    n_a_type = args.n_a_type

    margin = (config.WINDOW_SIZE_1STDER - 1) // 2

    exp_paths = sorted([Path(args.exp_folder, name) for name in os.listdir(args.exp_folder)
                        if (os.path.isdir(os.path.join(args.exp_folder, name)) and name not in config.EXCLUDED_FOLDERS)])

    if args.nc_subtract:
        exp_folder_root = Path(args.exp_folder)
        nc_subtract_root = exp_folder_root.parent / f"{exp_folder_root.name}_nc_subtract"
        # Pre-create all sibling folders so downstream pipelines see a consistent,
        # fully-mirrored folder listing regardless of array-task execution order.
        for p in exp_paths:
            os.makedirs(nc_subtract_root / p.name, exist_ok=True)

    curve_labels = ["Original Curve", "1st Derivative", "1st Derivative Moving Avg", "Cleaned Curve", 
                    "Cleaned Curve (Lowest Crossing)"] 

    if args.task_id >= len(exp_paths):
        print(f"Task ID {args.task_id} is out of bounds for {len(exp_paths)} folders. Exiting.")
        sys.exit(0)

    exp_path = exp_paths[args.task_id]
    
    if args.nc_subtract:
        save_exp_path = nc_subtract_root / exp_path.name
    else:
        save_exp_path = exp_path

    # Shared with 02_outlier_detection_pipeline.py — 01 and 02 patch their own keys into
    # the same file rather than 01 writing a separate file 02 copies from. See
    # joblib_redundancy.md "01 + 02: one shared joblib, scripts stay separate".
    save_path = os.path.join(save_exp_path, config.TRAINING_DATA_PATH)
    legacy_save_path = os.path.join(save_exp_path, config.PREPROCESSED_CURVES_PATH)

    if os.path.exists(save_path) and not args.force_rerun:
        try:
            existing_data = joblib.load(save_path)
        except Exception as e:
            print(f"  -> [WARNING] Cached file at {save_path} is corrupted ({e}). Recomputing from scratch...")
            existing_data = None

        if existing_data is not None and "curves" in existing_data:
            needs_avg_patch     = args.moving_avg and "ori_curves_avg" not in existing_data["curves"]
            _cached_norm_bases  = [k for k in existing_data["curves"]
                                   if k.startswith("ori_curves") and "dydx" not in k and not k.endswith("_norm")]
            needs_norm_patch    = args.normalize_curves and any(
                                   f"{k}_norm" not in existing_data["curves"] for k in _cached_norm_bases)
            needs_wavelet_patch = args.wavelet_sym8      and "ori_curves_wavelet_sym8"     not in existing_data["curves"]
            needs_bior35_patch  = args.wavelet_bior35    and "ori_curves_wavelet_bior35"   not in existing_data["curves"]
            needs_sg_p4_patch   = args.sg_p4             and "ori_curves_sg_p4"            not in existing_data["curves"]

            if not any([needs_avg_patch, needs_norm_patch, needs_wavelet_patch,
                        needs_bior35_patch, needs_sg_p4_patch]):
                print(f"Cache hit: {save_exp_path}")
                print("  ✓ Experiment complete!\n")
                sys.exit(0)

            patched_fields = []
            if needs_avg_patch:
                existing_data["curves"]["ori_curves_avg"] = moving_average_vec(
                    existing_data["curves"]["ori_curves"], config.WINDOW_SIZE_ORI
                )
                existing_data["window_size_ori"] = config.WINDOW_SIZE_ORI
                patched_fields.append("ori_curves_avg/window_size_ori")
            if needs_norm_patch:
                _norm_bases = [k for k in existing_data["curves"]
                               if k.startswith("ori_curves") and "dydx" not in k and not k.endswith("_norm")]
                for k in _norm_bases:
                    nk = f"{k}_norm"
                    if nk not in existing_data["curves"]:
                        existing_data["curves"][nk] = normalize_curves_minmax(existing_data["curves"][k])
                        patched_fields.append(nk)
            if needs_wavelet_patch:
                existing_data["curves"]["ori_curves_wavelet_sym8"] = wavelet_denoise_curves(existing_data["curves"]["ori_curves"])
                patched_fields.append("ori_curves_wavelet_sym8")
            if needs_bior35_patch:
                existing_data["curves"]["ori_curves_wavelet_bior35"] = wavelet_denoise_curves(
                    existing_data["curves"]["ori_curves"], wavelet="bior3.5", level=5)
                patched_fields.append("ori_curves_wavelet_bior35")
            if needs_sg_p4_patch:
                existing_data["curves"]["ori_curves_sg_p4"], existing_data["sg_p4_optimal_w"] = \
                    sg_p4_denoise_curves(existing_data["curves"]["ori_curves"])
                patched_fields.append("ori_curves_sg_p4")

            print(f"Cache hit: {save_exp_path} (patching missing {', '.join(patched_fields)})")
            safe_joblib_dump(existing_data, save_path, compress=3)
            print(f"  -> Patched {save_path}")
            print("  ✓ Experiment complete!\n")
            sys.exit(0)
        # else: file exists but 01's keys ("curves") aren't in it yet (e.g. only 02 has
        # written to the shared file so far) — fall through and compute 01's part normally.

    print(f"Processing Experiment: {exp_path}")

    if(n_a_type == "v04"):    
        exp = titan_load_and_preprocessing_v4(exp_path, n_wells=n_wells, start_type="temperature",
                                            end_time_min=60, n_a_type=n_a_type,
                                            print_status=False, plt_gain_calib=False, save_gain_calib=False)
        all_exp_data = [exp]

    elif(n_a_type in ["v05", 'v06']):
        vref_ref_idx = "all"
        all_exp_data = load_and_preprocess_v6(exp_path=exp_path, n_wells=n_wells, n_a_type=n_a_type, vref_ref_idx=vref_ref_idx)

    print("  -> Generating Pixel & Temp DataFrames...")
    pixel_temp_dfs = extract_pixel_temp_dataframes(all_exp_data)

    print("  -> Processing Sigmoid Curves...")
    X_time, Y_well, X_2d_bs_active = reconstruct_data(all_exp_data, attr_str="well_2d_bs_active")
    
    max_significant_index = None

    # -------------------------------------------------------------
    # TRUNCATION (always runs)
    # -------------------------------------------------------------
    exp_folder = os.path.basename(exp_path)
    mapping    = config.LABEL_MAPPINGS.get(exp_folder) if hasattr(config, "LABEL_MAPPINGS") else None
    y_label    = np.array([mapping.get(w, w) for w in Y_well]) if mapping is not None else None

    print("  -> Calculating truncation index...")
    pc_mask = (y_label == 'PC') if y_label is not None else np.zeros(len(Y_well), dtype=bool)

    if np.any(pc_mask):
        # PC argmin: argmin of smoothed mean of PC-labelled wells
        smoothed = uniform_filter1d(np.mean(X_2d_bs_active[pc_mask], axis=0).squeeze(), size=20)
        max_significant_index = int(np.argmin(smoothed))
        print(f"      -> [PC argmin] MSI={max_significant_index} ({int(pc_mask.sum())} PC wells)")
    else:
        # Derivative vote fallback (original method)
        ori_curve_dydx = np.array(get_derivatives(X_2d_bs_active, X_time))
        min_indices    = np.argmin(ori_curve_dydx, axis=1)
        unique_indices, counts = np.unique(min_indices, return_counts=True)
        significant_indices = unique_indices[counts > (len(ori_curve_dydx) / 3)]
        max_significant_index = int(np.max(significant_indices)) if len(significant_indices) > 0 else None
        print(f"      -> [Deriv. vote] MSI={max_significant_index}")

    if max_significant_index is not None:
        if (max_significant_index + 1) < (len(X_time) - 100):
            truncated_curves = X_2d_bs_active[:, max_significant_index + 1:]
            truncated_timestamps = X_time[max_significant_index + 1:]
            X_2d_bs_active = truncated_curves - truncated_curves[:, 0:1]
            X_time = truncated_timestamps
            print(f"      -> Truncated X_time from {len(X_time) + max_significant_index + 1} to {len(X_time)}")
        else:
            print(f"      -> [!] MSI {max_significant_index} too close to end (len={len(X_time)}). Skipping.")
            max_significant_index = None
    else:
        print("      -> No truncation index found. Proceeding with original data.")

    # -------------------------------------------------------------
    # NEGATIVE CONTROL (NC) SUBTRACTION (--nc_subtract only)
    # -------------------------------------------------------------
    if args.nc_subtract:
        print("  -> [nc_subtract=True] Initiating NC Subtraction...")

        if mapping is None:
            print(f"  [!] CRITICAL: Missing mapping rules for '{exp_folder}' in config.LABEL_MAPPINGS.")
            print("  [!] Aborting processing as requested.")
            sys.exit(1)

        # y_label already computed above from mapping
        vref_idx = pixel_temp_dfs["well_2d_bs_active_df"]['vref_idx'].values

        # --- NC Subtraction ---
        print("      -> Performing Negative Control (NC) Subtraction...")

        unique_vrefs = np.unique(vref_idx)
        unique_labels = np.unique(y_label)

        subtracted_curves = X_2d_bs_active.copy()

        if 'NC-ALL' in unique_labels:
            # Universal NC: NC-ALL mean subtracted from every well in each VREF slice
            for vref_val in unique_vrefs:
                vref_mask   = (vref_idx == vref_val)
                nc_all_mask = vref_mask & (y_label == 'NC-ALL')
                if np.any(nc_all_mask):
                    nc_baseline = np.mean(X_2d_bs_active[nc_all_mask], axis=0)
                    subtracted_curves[vref_mask] = X_2d_bs_active[vref_mask] - nc_baseline
                    print(f"         -> Slice VREF {vref_val}: Subtracted 'NC-ALL' mean from all wells ({np.sum(vref_mask)} curves)")
        else:
            # Per-label NC: subtract NC-{b_lbl} mean from matching {b_lbl} wells only
            # Isolate the base classes (e.g., extracts 'C' if 'NC-C' exists)
            base_labels = list(dict.fromkeys([str(lbl).replace('NC-', '') for lbl in unique_labels]))

            for vref_val in unique_vrefs:
                vref_mask = (vref_idx == vref_val)

                for b_lbl in base_labels:
                    nc_target = f"NC-{b_lbl}"

                    # Check if both Sample and corresponding NC exist for this specific VREF
                    if b_lbl in unique_labels and nc_target in unique_labels:
                        nc_mask = vref_mask & (y_label == nc_target)
                        sample_mask = vref_mask & (y_label == b_lbl)

                        if np.any(nc_mask) and np.any(sample_mask):
                            # 3.1 Get Mean of NCs
                            nc_baseline = np.mean(X_2d_bs_active[nc_mask], axis=0)

                            # 3.2 Subtract from corresponding Samples
                            subtracted_curves[sample_mask] = X_2d_bs_active[sample_mask] - nc_baseline

                            # 3.3 Subtract from the NCs themselves (centers the control noise floor at 0)
                            subtracted_curves[nc_mask] = X_2d_bs_active[nc_mask] - nc_baseline

                            print(f"         -> Slice VREF {vref_val}: Subtracted '{nc_target}' mean from '{b_lbl}' ({np.sum(sample_mask)} curves) and normalized NCs ({np.sum(nc_mask)} curves)")

        # Update main tracking array for the rest of the pipeline
        X_2d_bs_active = subtracted_curves
        print("  [✓] NC baseline subtraction complete.")

    # Shift timestamps so they start from 0 (post-truncation, if any)
    X_time = X_time - X_time[0]

    # -------------------------------------------------------------
    # DROP PC (--drop_pc): remove PC wells now that truncation is done
    # -------------------------------------------------------------
    if args.drop_pc:
        if y_label is None:
            print("  [!] --drop_pc: no LABEL_MAPPINGS for this experiment — skipping PC removal.")
        elif not np.any(y_label == 'PC'):
            print("  [!] --drop_pc: no PC-labeled wells found.")
        else:
            keep_mask = (y_label != 'PC')
            n_dropped = int(np.sum(~keep_mask))
            X_2d_bs_active = X_2d_bs_active[keep_mask]
            Y_well         = Y_well[keep_mask]
            y_label        = y_label[keep_mask]
            keep_idx       = np.where(keep_mask)[0]
            for key, df in pixel_temp_dfs.items():
                if len(df) == len(keep_mask):
                    pixel_temp_dfs[key] = df.iloc[keep_idx].reset_index(drop=True)
            print(f"  -> [drop_pc] Removed {n_dropped} PC samples. {len(Y_well)} samples remain.")

    baseline_value = 0

    ##############################################################
    # THIS #######################################################
    # if (np.min(X_2d_bs_active) < 0):
    #     baseline_value = -np.min(X_2d_bs_active) + 1e-9
    #     X_2d_bs_active = X_2d_bs_active + baseline_value
    ##############################################################
    ##############################################################

    processed_curves, indices_dict, ori_curves_avg = process_experiment_data(
        X_2d_bs_active, X_time, config.WINDOW_SIZE_ORI, config.WINDOW_SIZE_1STDER, margin,
        compute_sigmoid_fits=args.compute_sigmoid_fits
    )

    if args.compute_sigmoid_fits:
        fitting_results = run_all_fits(processed_curves, indices_dict, X_time)
    else:
        fitting_results = {}

    save_experiment_data_restructured(save_exp_path, fitting_results, processed_curves,
                                    indices_dict, pixel_temp_dfs, baseline_value,
                                    Y_well, X_time, all_exp_data, ori_curves_avg,
                                    config.WINDOW_SIZE_ORI, config.WINDOW_SIZE_1STDER, margin, max_significant_index,
                                    compute_sigmoid_fits=args.compute_sigmoid_fits,
                                    normalize_curves=args.normalize_curves,
                                    wavelet_sym8=args.wavelet_sym8,
                                    wavelet_bior35=args.wavelet_bior35,
                                    sg_p4=args.sg_p4,
                                    moving_avg=args.moving_avg)
    
    unique_wells = np.unique(Y_well)

    saved_viz = getattr(config, "SAVED_VIZ", [])
    save_plot_flag = bool(saved_viz) and np.any([saved in str(save_exp_path) for saved in saved_viz])
    
    # if(save_plot_flag):
    #     from curve_preprocessing_plots import plot_interactive_sigmoid_grids
    #     plot_interactive_sigmoid_grids(
    #         save_exp_path, unique_wells, Y_well, X_time,
    #         processed_curves, fitting_results, indices_dict, curve_labels,
    #         ds_step=config.PLOT_DOWNSAMPLE_STEP, precision=config.PLOT_DECIMAL_PRECISION
    #     )
    
    print("  ✓ Experiment complete!\n")