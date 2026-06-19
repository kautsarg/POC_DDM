import os
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.ndimage import convolve1d
from joblib import Parallel, delayed

import config

# Add custom paths
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

def save_experiment_data_restructured(save_exp_path, fitting_results, processed_curves,
                                     indices_dict, pixel_temp_dfs, baseline_value,
                                     Y_well, X_time, all_exp_data, ori_curves_avg,
                                     window_size_ori, window_size_1stder, margin, max_significant_index,
                                     compute_sigmoid_fits=False):
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
    save_data = {
        "curves": {
            "ori_curves": processed_curves[0],
            "ori_curves_avg": ori_curves_avg,
            "ori_curve_dydx": processed_curves[1],
            "ori_dydx_avg": processed_curves[2],
            "cleaned_std": processed_curves[3],
            "cleaned_lowest": processed_curves[4],
        },
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
        "margin": margin
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
    joblib.dump(existing_state, save_path, compress=3)   # kinetic_features, ...) are left untouched
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
    parser.add_argument("--force_rerun", action="store_true", help="Recompute and overwrite even if a presaved file already exists")
    parser.add_argument("--compute_sigmoid_fits", action="store_true",
                        help="Compute the derivative/cleaning chain (ori_curve_dydx, ori_dydx_avg, cleaned_std, "
                             "cleaned_lowest) and the 5-parameter sigmoid fits derived from it. Unused by 02-08 "
                             "under default --curve_type args; off by default to save compute and storage.")
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
            if "ori_curves_avg" in existing_data["curves"] and "window_size_ori" in existing_data:
                print(f"Cache hit: {save_exp_path}")
                print("  ✓ Experiment complete!\n")
                sys.exit(0)

            print(f"Cache hit: {save_exp_path} (patching missing 'ori_curves_avg'/'window_size_ori')")
            existing_data["curves"]["ori_curves_avg"] = moving_average_vec(
                existing_data["curves"]["ori_curves"], config.WINDOW_SIZE_ORI
            )
            existing_data["window_size_ori"] = config.WINDOW_SIZE_ORI
            joblib.dump(existing_data, save_path, compress=3)
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
    # TRUNCATION AND NEGATIVE CONTROL (NC) SUBTRACTION LOGIC
    # -------------------------------------------------------------
    if args.nc_subtract:
        print("  -> [nc_subtract=True] Initiating Truncation and NC Subtraction...")
        exp_folder = os.path.basename(exp_path)

        # 1. & 2. Check configuration constraints and ABORT if missing
        if not hasattr(config, "LABEL_MAPPINGS") or exp_folder not in config.LABEL_MAPPINGS:
            print(f"  [!] CRITICAL: Missing mapping rules for '{exp_folder}' in config.LABEL_MAPPINGS.")
            print("  [!] Aborting processing as requested.")
            sys.exit(1)

        # Apply Label Mapping to get y_label
        mapping = config.LABEL_MAPPINGS[exp_folder]
        y_label = np.array([mapping.get(w, w) for w in Y_well])
        vref_idx = pixel_temp_dfs["well_2d_bs_active_df"]['vref_idx'].values

        # --- PART A: Truncation ---
        print("      -> Calculating derivatives for truncation...")
        ori_curve_dydx = np.array(get_derivatives(X_2d_bs_active, X_time))
        min_indices = np.argmin(ori_curve_dydx, axis=1)

        unique_indices, counts = np.unique(min_indices, return_counts=True)
        threshold = len(ori_curve_dydx) / 3
        significant_indices = unique_indices[counts > threshold]

        max_significant_index = np.max(significant_indices) if len(significant_indices) > 0 else None

        if max_significant_index is not None:
            if (max_significant_index + 1) < (len(X_time) - 100):           # Do not truncate if the index is too close to the end to avoid losing critical data or crashing
                truncated_curves = X_2d_bs_active[:, max_significant_index + 1:]
                truncated_timestamps = X_time[max_significant_index + 1:]
                
                # Baseline subtract the truncated curves (zeroing to start)
                X_2d_bs_active = truncated_curves - truncated_curves[:, 0:1]
                X_time = truncated_timestamps
                print(f"      -> Truncated X_time from {len(X_time) + max_significant_index + 1} to {len(X_time)}")
            else:
                print(f"      -> [!] Truncation index {max_significant_index} is too close to the end (Total len: {len(X_time)}). Skipping truncation to prevent crash.")
                max_significant_index = None 
        else:
            print("      -> No significant index found for truncation. Proceeding with original data.")

        # --- PART B: NC Subtraction ---
        print("      -> Performing Negative Control (NC) Subtraction...")
        
        unique_vrefs = np.unique(vref_idx)
        unique_labels = np.unique(y_label)
        
        # Isolate the base classes (e.g., extracts 'C' if 'NC-C' exists)
        base_labels = list(dict.fromkeys([str(lbl).replace('NC-', '') for lbl in unique_labels]))
        
        subtracted_curves = X_2d_bs_active.copy()
        
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
        print("  [✓] Truncation and NC baseline subtraction complete.")

    # Shift timestamps so they start from 0 (post-truncation, if any)
    X_time = X_time - X_time[0]

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
                                    compute_sigmoid_fits=args.compute_sigmoid_fits)
    
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