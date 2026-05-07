import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.ndimage import convolve1d
from joblib import Parallel, delayed

from bokeh.plotting import figure, save, output_file
from bokeh.layouts import gridplot, column, row, Spacer
from bokeh.models import ColumnDataSource, CustomJS, Div, HoverTool, Span, CrosshairTool, TapTool

# Add custom paths
sys.path.insert(0, '..')
sys.path.insert(0, '0_4_AMCA Code on Chip')

from titan.Experiment import Experiment
from titan.load_and_preprocessing import titan_load_and_preprocessing
import sigmoid_fitting as sp

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
        # Identify boundaries based on sign changes
        sign_changes = np.nonzero(np.diff(np.sign(arr)))[0] + 1
        boundaries = np.concatenate(([0], sign_changes, [len(arr)]))
        
        lowest_sum = 0
        best_exit_idx = 0 
        
        # Pre-calculate cumulative sum for O(1) segment sum extraction
        c_sum = np.cumsum(arr)
        
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i+1]
            # Calculate segment sum via cumsum differences
            segment_sum = c_sum[end-1] - (c_sum[start-1] if start > 0 else 0)
            if segment_sum < lowest_sum:
                lowest_sum = segment_sum
                best_exit_idx = end
        return best_exit_idx
    
    # Use fromiter for faster C-level array generation over lists
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
    # Switched to loky backend with auto batching for CPU-bound performance
    return Parallel(n_jobs=-1, backend="loky", batch_size='auto')(
        delayed(sp.calculate_first_derivative)(timestamps, y) for y in curves_batch
    )

def reconstruct_data(exp_data: Experiment, attr_str: str):   
    vstacked = None
    well_ids = []

    for well in exp_data.wells_list:
        temp_x = getattr(well, attr_str).copy()
        temp_x = np.swapaxes(temp_x, 0, 1) 

        if vstacked is None:
            vstacked = temp_x
        else:
            vstacked = np.vstack((vstacked, temp_x))

        well_ids.append(temp_x.shape[0])

    X_time = exp_data.wells_list[0].time
    
    Y_well = []
    for label, count in enumerate(well_ids):
        Y_well.extend([label] * count)

    return X_time, np.array(Y_well), vstacked


# ==========================================
# 2. DATAFRAME GENERATION (PIXELS & TEMPS)
# ==========================================

def extract_pixel_temp_dataframes(exp_data):
    df_pix_lin_list = []
    df_pix_nl_list = []
    df_temp_lin_list = []
    df_temp_nl_list = []

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

        active_y = y.flatten()[idx_active]
        active_x = x.flatten()[idx_active]
        active_temp_mapping = temp_group_idx[idx_active]

        # Calculate Pixel Data
        n_time_lin = well.well_3d_lin.shape[2]
        well_2d_lin = well.well_3d_lin.reshape(-1, n_time_lin, order='C').T
        well_2d_bs = well_2d_lin - well_2d_lin[idx_settled, :]
        well_2d_bs_active = well_2d_bs[:, idx_active]

        n_time_nl = well.well_3d_npr.shape[2]
        well_2d_nl = well.well_3d_npr.reshape(-1, n_time_nl, order='C').T
        well_2d_nl_bs = well_2d_nl - well_2d_nl[idx_settled, :]
        well_2d_nl_bs_active = well_2d_nl_bs[:, idx_active]

        time_cols = [f"Cycle_{t}" for t in time_npr]

        # --- 1. Linearised Pixel DF ---
        df_pl = pd.DataFrame(well_2d_bs_active.T, columns=time_cols)
        df_pl['well_id'] = w_idx
        df_pl['pixel_row_idx'] = active_y
        df_pl['pixel_col_idx'] = active_x
        df_pl['temp_group_idx'] = active_temp_mapping

        counts = df_pl['temp_group_idx'].value_counts()
        df_pl['num_active_pixels_in_temp_group'] = df_pl['temp_group_idx'].map(counts)

        mean_temp_lin = well_temp_lin2d.mean(axis=0)
        df_pl['well_temp_lin2d_mean'] = mean_temp_lin[active_temp_mapping]

        mean_temp_nl = well_2d_temp_npr.mean(axis=0)
        df_pl['well_2d_temp_npr_mean'] = mean_temp_nl[active_temp_mapping]

        # --- 2. Non-Linearised Pixel DF ---
        df_pnl = pd.DataFrame(well_2d_nl_bs_active.T, columns=time_cols)
        df_pnl['well_id'] = w_idx
        df_pnl['pixel_row_idx'] = active_y
        df_pnl['pixel_col_idx'] = active_x
        df_pnl['temp_group_idx'] = active_temp_mapping
        df_pnl['num_active_pixels_in_temp_group'] = df_pnl['temp_group_idx'].map(counts)
        df_pnl['well_temp_lin2d_mean'] = mean_temp_lin[active_temp_mapping]
        df_pnl['well_2d_temp_npr_mean'] = mean_temp_nl[active_temp_mapping]

        # Standardize Metadata Order
        meta_cols = ['well_id', 'pixel_row_idx', 'pixel_col_idx', 'temp_group_idx',
                     'num_active_pixels_in_temp_group', 'well_temp_lin2d_mean', 'well_2d_temp_npr_mean']
        df_pl = df_pl[meta_cols + time_cols]
        df_pnl = df_pnl[meta_cols + time_cols]

        # --- 3. Linearised Temp DF ---
        df_tl = pd.DataFrame(well_temp_lin2d.T, columns=time_cols)
        df_tl.insert(0, 'well_id', w_idx)
        df_tl.insert(1, 'temp_group_idx', np.arange(well_temp_lin2d.shape[1]))

        # --- 4. Non-Linearised Temp DF ---
        df_tnl = pd.DataFrame(well_2d_temp_npr.T, columns=time_cols)
        df_tnl.insert(0, 'well_id', w_idx)
        df_tnl.insert(1, 'temp_group_idx', np.arange(well_2d_temp_npr.shape[1]))

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

def process_experiment_data(ori_curves, ori_timestamps, window_size_ori, window_size_1stder, margin):
    ori_curves_avg = moving_average_vec(ori_curves, window_size_ori)
    ori_curve_dydx = np.array(get_derivatives(ori_curves, ori_timestamps))
    ori_avg_dydx = np.array(get_derivatives(ori_curves_avg, ori_timestamps))
    
    ori_dydx_avg = moving_average_vec(ori_curve_dydx, window_size_1stder)
    ori_avg_dydx_avg = moving_average_vec(ori_avg_dydx, window_size_1stder)

    cleaning_tasks = [
        ("std", ori_dydx_avg, first_pos_zero_crossing_vec),
        ("low", ori_dydx_avg, lowest_integral_to_next_crossing_vec),
        ("avg_std", ori_avg_dydx_avg, first_pos_zero_crossing_vec),
        ("avg_low", ori_avg_dydx_avg, lowest_integral_to_next_crossing_vec)
    ]

    results = {}
    for name, deriv, func in cleaning_tasks:
        indices = func(deriv)
        cleaned_curves, actual_idxs = apply_baseline_cleaning(ori_curves, indices, margin)
        results[name] = (cleaned_curves, actual_idxs)

    indices_dict = {
        "cleaned_idx": results["std"][1],
        "cleaned_lowest_idx": results["low"][1],
        "avg_cleaned_idx": results["avg_std"][1],
        "avg_cleaned_lowest_idx": results["avg_low"][1]
    }

    processed_curves = [
        ori_curves, ori_curve_dydx, ori_dydx_avg, 
        results["std"][0], results["low"][0], 
        ori_curves_avg, ori_avg_dydx, ori_avg_dydx_avg, 
        results["avg_std"][0], results["avg_low"][0]
    ]

    return processed_curves, indices_dict


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
        
    # Loky backend to bypass GIL for CPU bound tasks, with auto batching
    results = Parallel(n_jobs=-1, backend="loky", batch_size='auto')(
        delayed(_fit_single_curve)(y, ori_timestamps, t) 
        for y, t in zip(curves, starting_idxs)
    )
    
    curves_out_full, curves_out_stretched, params_out, rmse_out = zip(*results)
    return np.array(curves_out_full), np.array(curves_out_stretched), np.array(params_out), np.array(rmse_out)

def run_all_fits(processed_curves, indices_dict, ori_timestamps):
    # Calculate all fits
    raw_fits = {
        "original": sigmoid_fitting_5p(processed_curves[0], ori_timestamps, None),
        "cleaned_std": sigmoid_fitting_5p(processed_curves[3], ori_timestamps, indices_dict["cleaned_idx"]),
        "cleaned_lowest": sigmoid_fitting_5p(processed_curves[4], ori_timestamps, indices_dict["cleaned_lowest_idx"]),
        "avg": sigmoid_fitting_5p(processed_curves[5], ori_timestamps, None),
        "avg_cleaned_std": sigmoid_fitting_5p(processed_curves[8], ori_timestamps, indices_dict["avg_cleaned_idx"]),
        "avg_cleaned_lowest": sigmoid_fitting_5p(processed_curves[9], ori_timestamps, indices_dict["avg_cleaned_lowest_idx"])
    }
    
    # Restructure into the requested dictionary format
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
# 5. SAVING MODULE (UPDATED WITH BASELINE)
# ==========================================

def save_experiment_data(exp_path, fitting_results, processed_curves, indices_dict, pixel_temp_dfs, baseline_value):
    save_data = {
        "fitting_results": fitting_results,
        "processed_curves": processed_curves,
        "cleaning_indices": indices_dict,
        "pixel_temp_dfs": pixel_temp_dfs,
        "baseline_value": baseline_value
    }
    save_path = os.path.join(exp_path, "processed_curve_results.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"  -> Saved numerical results and DFs to {save_path}")


# ==========================================
# 6. PLOTTING MODULE 1: SIGMOID GRIDS
# ==========================================

def draw_stats(p, data_array, timestamps):
    y_mean, y_std = np.nanmean(data_array, axis=0), np.nanstd(data_array, axis=0)
    p.line(x=timestamps, y=y_mean, color="red", line_width=0.5)
    p.line(x=timestamps, y=y_mean + y_std, color="red", line_width=0.25, line_dash="dashed", alpha=0.8)
    p.line(x=timestamps, y=y_mean - y_std, color="red", line_width=0.25, line_dash="dashed", alpha=0.8)

def normalize_array(arr):
    c_min = np.nanmin(arr, axis=1, keepdims=True)
    c_max = np.nanmax(arr, axis=1, keepdims=True)
    val_range = c_max - c_min
    val_range[val_range == 0] = 1e-10 
    return (arr - c_min) / val_range

def blank_plot(plot_size):
    p = figure(width=plot_size, height=plot_size, output_backend="webgl")
    p.xaxis.visible = False; p.yaxis.visible = False; p.grid.visible = False; p.outline_line_color = None
    p.scatter(x=[], y=[]) 
    return p

def plot_interactive_sigmoid_grids(exp_path, unique_wells, ori_well, ori_timestamps, processed_curves, fitting_results, indices_dict, curve_labels):
    col_to_idx_map = {
        0: (indices_dict["cleaned_idx"], indices_dict["cleaned_lowest_idx"]), 
        1: (indices_dict["cleaned_idx"], indices_dict["cleaned_lowest_idx"]), 
        2: (indices_dict["cleaned_idx"], indices_dict["cleaned_lowest_idx"]), 
        3: (indices_dict["cleaned_idx"],), 4: (indices_dict["cleaned_lowest_idx"],),
        5: (indices_dict["avg_cleaned_idx"], indices_dict["avg_cleaned_lowest_idx"]), 
        6: (indices_dict["avg_cleaned_idx"], indices_dict["avg_cleaned_lowest_idx"]), 
        7: (indices_dict["avg_cleaned_idx"], indices_dict["avg_cleaned_lowest_idx"]), 
        8: (indices_dict["avg_cleaned_idx"],), 9: (indices_dict["avg_cleaned_lowest_idx"],)
    }

    fit_key_map = {
        0: "original", 
        3: "cleaned_std", 
        4: "cleaned_lowest", 
        5: "avg", 
        8: "avg_cleaned_std", 
        9: "avg_cleaned_lowest"
    }
    stats_cols = {0, 3, 4, 5, 8, 9}
    group_a_cols, group_b_cols = [0, 3, 4, 5, 8, 9], [1, 2, 6, 7]
    
    plot_size = 350 
    scatter_kwargs = dict(size=4, color="grey", alpha=0.25, selection_color="red", selection_alpha=1.0, nonselection_color="grey", nonselection_alpha=0.05)

    for row_idx, well in enumerate(unique_wells):
        filename = f"{exp_path}/well_{well}_sigmoid_curves.html"
        output_file(filename, title=f"Well {well} Sigmoid Curves")
        
        mask = (ori_well == well)
        hex_color = mcolors.to_hex(plt.cm.tab10(row_idx % 10))
        num_curves = np.sum(mask)
        
        data_dict = {'xs': [ori_timestamps for _ in range(num_curves)], 'color': [hex_color] * num_curves}
        
        for col_idx, curves in enumerate(processed_curves):
            # Optimized array to list conversion
            data_dict[f'ys_{col_idx}'] = curves[mask].tolist()
            idx_tuple = col_to_idx_map[col_idx]
            
            idx_1 = idx_tuple[0][mask]
            data_dict[f'mx1_{col_idx}'] = ori_timestamps[idx_1]
            data_dict[f'my1_{col_idx}'] = curves[mask][np.arange(num_curves), idx_1]
            
            if len(idx_tuple) > 1:
                idx_2 = idx_tuple[1][mask]
                data_dict[f'mx2_{col_idx}'] = ori_timestamps[idx_2]
                data_dict[f'my2_{col_idx}'] = curves[mask][np.arange(num_curves), idx_2]
                
            if col_idx in stats_cols:
                fit_key = fit_key_map[col_idx]
                fc_mask = fitting_results[fit_key]["fitted_full"][mask] 
                sc_mask = fitting_results[fit_key]["fitted_stretched"][mask]
                
                norm_fc_mask, norm_sc_mask = normalize_array(fc_mask), normalize_array(sc_mask)
                
                # Optimized array to list conversions
                data_dict[f'fitted_ys_{col_idx}'] = fc_mask.tolist()
                data_dict[f'fitted_my1_{col_idx}'] = fc_mask[np.arange(num_curves), idx_1]
                data_dict[f'stretched_ys_{col_idx}'] = sc_mask.tolist()
                data_dict[f'stretched_my1_{col_idx}'] = sc_mask[np.arange(num_curves), idx_1]
                data_dict[f'norm_fitted_ys_{col_idx}'] = norm_fc_mask.tolist()
                data_dict[f'norm_fitted_my1_{col_idx}'] = norm_fc_mask[np.arange(num_curves), idx_1]
                data_dict[f'norm_stretched_ys_{col_idx}'] = norm_sc_mask.tolist()
                data_dict[f'norm_stretched_my1_{col_idx}'] = norm_sc_mask[np.arange(num_curves), idx_1]
                
                if len(idx_tuple) > 1:
                    data_dict[f'fitted_my2_{col_idx}'] = fc_mask[np.arange(num_curves), idx_2]
                    data_dict[f'stretched_my2_{col_idx}'] = sc_mask[np.arange(num_curves), idx_2]
                    data_dict[f'norm_fitted_my2_{col_idx}'] = norm_fc_mask[np.arange(num_curves), idx_2]
                    data_dict[f'norm_stretched_my2_{col_idx}'] = norm_sc_mask[np.arange(num_curves), idx_2]
                
        source = ColumnDataSource(data=data_dict)
        r1_plots, r2_plots, r3_plots, r4_plots, r5_plots = [], [], [], [], []
        
        for col_idx in range(len(processed_curves)):
            title = curve_labels[col_idx]
            
            p1 = figure(title=title, width=plot_size, height=plot_size, tools="pan,wheel_zoom,box_zoom,reset,tap", output_backend="webgl")
            if col_idx == 0: p1.yaxis.axis_label = f"Well {well}"
            p1.multi_line(xs='xs', ys=f'ys_{col_idx}', color='color', source=source, line_width=1.0, alpha=0.15, selection_color="red", selection_alpha=1.0, nonselection_color=hex_color, nonselection_alpha=0.05)
            p1.scatter(x=f'mx1_{col_idx}', y=f'my1_{col_idx}', source=source, **scatter_kwargs)
            if len(col_to_idx_map[col_idx]) > 1: p1.scatter(x=f'mx2_{col_idx}', y=f'my2_{col_idx}', source=source, **scatter_kwargs)
            if col_idx in stats_cols: draw_stats(p1, processed_curves[col_idx][mask], ori_timestamps)
            r1_plots.append(p1)
            
            if col_idx not in stats_cols:
                r2_plots.append(blank_plot(plot_size)); r3_plots.append(blank_plot(plot_size)); r4_plots.append(blank_plot(plot_size)); r5_plots.append(blank_plot(plot_size))
                continue

            fit_key = fit_key_map[col_idx]
            fc_data = fitting_results[fit_key]["fitted_full"][mask]
            sc_data = fitting_results[fit_key]["fitted_stretched"][mask]
            norm_fc_data, norm_sc_data = normalize_array(fc_data), normalize_array(sc_data)

            p2 = figure(width=plot_size, height=plot_size, tools="pan,wheel_zoom,box_zoom,reset,tap", output_backend="webgl")
            if col_idx == 0: p2.yaxis.axis_label = "Fit"
            p2.multi_line(xs='xs', ys=f'fitted_ys_{col_idx}', color='color', source=source, line_width=1.0, alpha=0.2, selection_color="red", selection_alpha=1.0, nonselection_alpha=0.05)
            p2.scatter(x=f'mx1_{col_idx}', y=f'fitted_my1_{col_idx}', source=source, **scatter_kwargs)
            if len(col_to_idx_map[col_idx]) > 1: p2.scatter(x=f'mx2_{col_idx}', y=f'fitted_my2_{col_idx}', source=source, **scatter_kwargs)
            draw_stats(p2, fc_data, ori_timestamps)
            r2_plots.append(p2)

            p3 = figure(width=plot_size, height=plot_size, tools="pan,wheel_zoom,box_zoom,reset,tap", output_backend="webgl")
            if col_idx == 0: p3.yaxis.axis_label = "Stretched Fit"
            p3.multi_line(xs='xs', ys=f'stretched_ys_{col_idx}', color='color', source=source, line_width=1.0, alpha=0.2, selection_color="red", selection_alpha=1.0, nonselection_alpha=0.05)
            p3.scatter(x=f'mx1_{col_idx}', y=f'stretched_my1_{col_idx}', source=source, **scatter_kwargs)
            if len(col_to_idx_map[col_idx]) > 1: p3.scatter(x=f'mx2_{col_idx}', y=f'stretched_my2_{col_idx}', source=source, **scatter_kwargs)
            draw_stats(p3, sc_data, ori_timestamps)
            r3_plots.append(p3)

            p4 = figure(width=plot_size, height=plot_size, tools="pan,wheel_zoom,box_zoom,reset,tap", output_backend="webgl")
            if col_idx == 0: p4.yaxis.axis_label = "Norm Fit"
            p4.multi_line(xs='xs', ys=f'norm_fitted_ys_{col_idx}', color='color', source=source, line_width=1.0, alpha=0.2, selection_color="red", selection_alpha=1.0, nonselection_alpha=0.05)
            p4.scatter(x=f'mx1_{col_idx}', y=f'norm_fitted_my1_{col_idx}', source=source, **scatter_kwargs)
            if len(col_to_idx_map[col_idx]) > 1: p4.scatter(x=f'mx2_{col_idx}', y=f'norm_fitted_my2_{col_idx}', source=source, **scatter_kwargs)
            draw_stats(p4, norm_fc_data, ori_timestamps)
            r4_plots.append(p4)

            p5 = figure(width=plot_size, height=plot_size, tools="pan,wheel_zoom,box_zoom,reset,tap", output_backend="webgl")
            if col_idx == 0: p5.yaxis.axis_label = "Norm Stretched"
            p5.multi_line(xs='xs', ys=f'norm_stretched_ys_{col_idx}', color='color', source=source, line_width=1.0, alpha=0.2, selection_color="red", selection_alpha=1.0, nonselection_alpha=0.05)
            p5.scatter(x=f'mx1_{col_idx}', y=f'norm_stretched_my1_{col_idx}', source=source, **scatter_kwargs)
            if len(col_to_idx_map[col_idx]) > 1: p5.scatter(x=f'mx2_{col_idx}', y=f'norm_stretched_my2_{col_idx}', source=source, **scatter_kwargs)
            draw_stats(p5, norm_sc_data, ori_timestamps)
            r5_plots.append(p5)
            
        for c in range(len(processed_curves)):
            if c != 0: r1_plots[c].x_range = r1_plots[0].x_range
            r2_plots[c].x_range = r1_plots[0].x_range
            r3_plots[c].x_range = r1_plots[0].x_range
            r4_plots[c].x_range = r1_plots[0].x_range
            r5_plots[c].x_range = r1_plots[0].x_range
                
            if c in group_a_cols and c != group_a_cols[0]: r1_plots[c].y_range = r1_plots[group_a_cols[0]].y_range
            elif c in group_b_cols and c != group_b_cols[0]: r1_plots[c].y_range = r1_plots[group_b_cols[0]].y_range
                
            if c in stats_cols:
                r2_plots[c].y_range = r1_plots[group_a_cols[0]].y_range
                r3_plots[c].y_range = r1_plots[group_a_cols[0]].y_range
                if c != group_a_cols[0]:
                    r4_plots[c].y_range = r4_plots[group_a_cols[0]].y_range
                    r5_plots[c].y_range = r5_plots[group_a_cols[0]].y_range

        well_grid = gridplot([r1_plots, r2_plots, r3_plots, r4_plots, r5_plots])
        save(well_grid)


# ==========================================
# 7. PLOTTING MODULE 2: PIXEL VS TEMP
# ==========================================

def plot_pixel_temp_interactions(exp_data, exp_path):
    output_file(f"{exp_path}/all_wells_interactive.html", title="All Wells Interaction")
    p_width, p_height = 450, 225
    all_well_layouts = []

    for w_idx, well in enumerate(exp_data.wells_list):
        idx_settled = well.idx_settled
        idx_end = well.idx_end
        idx_active = well.idx_active
        time_npr = well.time_npr

        well_nrows, well_ncols = well.well_nrows, well.well_ncols
        well_temp_nrows, well_temp_ncols = well.well_temp_nrows, well.well_temp_ncols
        
        well_temp_2D_NEW = well.well_temp_lin2d 
        well_2d_temp_npr = well.well_2d_temp_npr 
        well_temp_mean_then_lin = well.well_temp_mean_then_lin

        y, x = np.indices((well_nrows, well_ncols))
        temp_group_idx = ((y // 5) * well_temp_ncols + (x // 5)).flatten()

        well_3d_lin = well.well_3d_lin
        n_time = well_3d_lin.shape[2]
        well_2d = well_3d_lin.reshape(-1, n_time, order='C').T
        well_2d_bs = well_2d - well_2d[idx_settled, :]  
        well_2d_bs_active = well_2d_bs[:, idx_active]

        well_3d_npr = well.well_3d_npr
        well_2d_nl = well_3d_npr.reshape(-1, n_time, order='C').T
        well_2d_nl_bs = well_2d_nl - well_2d_nl[idx_settled, :]  
        well_2d_nl_bs_active = well_2d_nl_bs[:, idx_active]
        
        n_active_pixels = well_2d_bs_active.shape[1]
        n_temp_groups = well_temp_2D_NEW.shape[1] 
        active_temp_mapping = temp_group_idx[idx_active]
        active_temp_groups = np.unique(active_temp_mapping)
        inactive_temp_groups = np.setdiff1d(np.arange(n_temp_groups), active_temp_groups)

        source_pixels = ColumnDataSource({
            'xs': [time_npr for _ in range(n_active_pixels)],
            'ys_lin': [well_2d_bs_active[:, i] for i in range(n_active_pixels)],
            'ys_nl': [well_2d_nl_bs_active[:, i] for i in range(n_active_pixels)],
            'group_id': active_temp_mapping, 
            'pixel_id': np.where(idx_active)[0] 
        })

        source_temps_active = ColumnDataSource({
            'xs': [time_npr for _ in active_temp_groups],
            'ys_lin': [well_temp_2D_NEW[:, i] for i in active_temp_groups],
            'ys_nl': [well_2d_temp_npr[:, i] for i in active_temp_groups],
            'group_id': active_temp_groups 
        })
        
        source_temps_inactive = ColumnDataSource({
            'xs': [time_npr for _ in inactive_temp_groups],
            'ys_lin': [well_temp_2D_NEW[:, i] for i in inactive_temp_groups],
            'ys_nl': [well_2d_temp_npr[:, i] for i in inactive_temp_groups],
            'group_id': inactive_temp_groups 
        })

        source_mean = ColumnDataSource({'x': time_npr, 'y': well_temp_mean_then_lin})

        info_div = Div(text=f"<h3 style='color: grey;'>Well {w_idx}: Select a line to see indices here...</h3>", width=1400)

        cb_pixel_to_temp = CustomJS(args=dict(sp=source_pixels, st=source_temps_active, div=info_div), code="""
            const selected_pixels = sp.selected.indices;
            if (selected_pixels.length === 0) {
                st.selected.indices = [];
                div.text = "<h3 style='color: grey;'>Select a line to see indices here...</h3>";
                return;
            }
            const target_group_id = sp.data['group_id'][selected_pixels[0]];
            const temp_groups = st.data['group_id'];
            const temp_to_select = [];
            for (let i = 0; i < temp_groups.length; i++) {
                if (temp_groups[i] === target_group_id) { temp_to_select.push(i); break; }
            }
            st.selected.indices = temp_to_select;
            const pixel_groups = sp.data['group_id'];
            const mapped_pixel_ids = []; 
            for (let i = 0; i < pixel_groups.length; i++) {
                if (pixel_groups[i] === target_group_id) { mapped_pixel_ids.push(sp.data['pixel_id'][i]); }
            }
            div.text = `<h3 style='color: navy;'>Pixel: <b>${sp.data['pixel_id'][selected_pixels[0]]}</b> | Group: <b>${target_group_id}</b> | All Pixels: <b>${mapped_pixel_ids.join(', ')}</b></h3>`;
        """)

        cb_temp_to_pixel = CustomJS(args=dict(sp=source_pixels, st=source_temps_active, div=info_div), code="""
            const selected_temps = st.selected.indices;
            if (selected_temps.length === 0) {
                sp.selected.indices = [];
                div.text = "<h3 style='color: grey;'>Select a line to see indices here...</h3>";
                return;
            }
            const target_group_id = st.data['group_id'][selected_temps[0]];
            const pixel_groups = sp.data['group_id'];
            const pixels_to_select = [];
            const mapped_pixel_ids = []; 
            for (let i = 0; i < pixel_groups.length; i++) {
                if (pixel_groups[i] === target_group_id) {
                    pixels_to_select.push(i);
                    mapped_pixel_ids.push(sp.data['pixel_id'][i]);
                }
            }
            sp.selected.indices = pixels_to_select;
            div.text = `<h3 style='color: darkorange;'>Selected Temp Group: <b>${target_group_id}</b> | Mapped Pixels: <b>${mapped_pixel_ids.join(', ')}</b></h3>`;
        """)

        source_pixels.selected.js_on_change('indices', cb_pixel_to_temp)
        source_temps_active.selected.js_on_change('indices', cb_temp_to_pixel)

        hover_pixel_nl = HoverTool(tooltips=[("Pixel", "@pixel_id"), ("Group", "@group_id"), ("Value", "$y")], line_policy="nearest")
        hover_pixel_lin = HoverTool(tooltips=[("Pixel", "@pixel_id"), ("Group", "@group_id"), ("Value", "$y")], line_policy="nearest")
        linked_crosshair = CrosshairTool(dimensions="height", line_color="black", line_alpha=0.3)

        p0 = figure(title=f"NL Pixels (Well {w_idx})", width=p_width, height=p_height, tools=["pan", "wheel_zoom", "box_zoom", "reset", "tap", hover_pixel_nl, linked_crosshair])
        p0.multi_line(xs='xs', ys='ys_nl', source=source_pixels, color="purple", alpha=0.15, 
                      selection_color="red", selection_alpha=1.0, selection_line_width=2,
                      nonselection_color="purple", nonselection_alpha=0.05)

        p1 = figure(title=f"Lin Pixels (Well {w_idx})", width=p_width, height=p_height, tools=["pan", "wheel_zoom", "box_zoom", "reset", "tap", hover_pixel_lin, linked_crosshair])
        p1.x_range = p0.x_range 
        p1.multi_line(xs='xs', ys='ys_lin', source=source_pixels, color="navy", alpha=0.15, 
                      selection_color="red", selection_alpha=1.0, selection_line_width=2,
                      nonselection_color="navy", nonselection_alpha=0.05)

        p2 = figure(title=f"NL Temps (Well {w_idx})", width=p_width, height=p_height, tools=["pan", "wheel_zoom", "box_zoom", "reset", linked_crosshair])
        p2.x_range = p0.x_range 
        p2.multi_line(xs='xs', ys='ys_nl', source=source_temps_inactive, color="lightgrey", alpha=0.3)
        active_temp_renderer_nl = p2.multi_line(xs='xs', ys='ys_nl', source=source_temps_active, color="darkorange", alpha=0.15, 
                                                selection_color="red", selection_alpha=1.0, selection_line_width=2,
                                                nonselection_color="darkorange", nonselection_alpha=0.1)
        p2.add_tools(HoverTool(tooltips=[("Group", "@group_id"), ("Val", "$y")], renderers=[active_temp_renderer_nl]), TapTool(renderers=[active_temp_renderer_nl]))

        p3 = figure(title=f"Lin Temps (Well {w_idx})", width=p_width, height=p_height, tools=["pan", "wheel_zoom", "box_zoom", "reset", linked_crosshair])
        p3.x_range = p0.x_range 
        p3.multi_line(xs='xs', ys='ys_lin', source=source_temps_inactive, color="lightgrey", alpha=0.3)
        active_temp_renderer_lin = p3.multi_line(xs='xs', ys='ys_lin', source=source_temps_active, color="darkorange", alpha=0.15, 
                                                 selection_color="red", selection_alpha=1.0, selection_line_width=2,
                                                 nonselection_color="darkorange", nonselection_alpha=0.1)
    
        p3.line(x='x', y='y', source=source_mean, color="black", line_width=1.5)
        p3.add_tools(HoverTool(tooltips=[("Group", "@group_id"), ("Val", "$y")], renderers=[active_temp_renderer_lin]), TapTool(renderers=[active_temp_renderer_lin]))

        time_settled_val, time_end_val = time_npr[idx_settled], time_npr[idx_end]
        for p in [p0, p1, p2, p3]:
            p.add_layout(Span(location=time_settled_val, dimension='height', line_color='green', line_dash='dashed'))
            p.add_layout(Span(location=time_end_val, dimension='height', line_color='red', line_dash='dashed'))

        well_layout = column(info_div, row(p0, p1, p2, p3), Spacer(height=50))
        all_well_layouts.append(well_layout)

    master_layout = column(*all_well_layouts)
    save(master_layout)

def save_experiment_data_restructured(exp_path, fitting_results, processed_curves, 
                                     indices_dict, pixel_temp_dfs, baseline_value, 
                                     Y_well, X_time, exp_data, 
                                     window_size_ori, window_size_1stder, margin):
    """
    Saves experiment data into a flat, queryable structure.
    """
    
    # Extract metadata from the Non-Linearised Pixel DF
    df_meta = pixel_temp_dfs["well_2d_nl_bs_active_df"]
    
    # Reference the first well for shared experiment indices
    well0 = exp_data.wells_list[0]
    
    save_data = {
        "curves": {
            "well_2d_bs_active": pixel_temp_dfs["well_2d_bs_active_df"].filter(like="Cycle_").values,
            "well_2d_nl_bs_active": pixel_temp_dfs["well_2d_nl_bs_active_df"].filter(like="Cycle_").values,
            "well_temp_lin2d": pixel_temp_dfs["well_temp_lin2d_df"].filter(like="Cycle_").values,
            "well_2d_temp_npr": pixel_temp_dfs["well_2d_temp_npr_df"].filter(like="Cycle_").values,
            "well_temp_mean_then_lin": well0.well_temp_mean_then_lin,
            "ori_curves": processed_curves[0],
            "ori_curve_dydx": processed_curves[1],
            "ori_dydx_avg": processed_curves[2],
            "cleaned_std": processed_curves[3],
            "cleaned_lowest": processed_curves[4],
            "ori_curves_avg": processed_curves[5],
            "ori_avg_dydx": processed_curves[6],
            "ori_avg_dydx_avg": processed_curves[7],
            "avg_cleaned_std": processed_curves[8],
            "avg_cleaned_lowest": processed_curves[9],
        },

        "sigmoid_curves": fitting_results,

        "idxs": {
            "idx_start": well0.idx_start,
            "idx_settled": well0.idx_settled,
            "idx_end": well0.idx_end,
            "idx_active": well0.idx_active,
            "cleaned_idx": indices_dict["cleaned_idx"],
            "cleaned_lowest_idx": indices_dict["cleaned_lowest_idx"],
            "avg_cleaned_idx": indices_dict["avg_cleaned_idx"],
            "avg_cleaned_lowest_idx": indices_dict["avg_cleaned_lowest_idx"]
        },

        "timestamps": X_time,
        "well_labels": Y_well,

        "metadata": {
            "pixel_row_idx": df_meta['pixel_row_idx'].values,
            "pixel_col_idx": df_meta['pixel_col_idx'].values,
            "temp_group_idx": df_meta['temp_group_idx'].values,
            "num_active_pixels_in_temp_group": df_meta['num_active_pixels_in_temp_group'].values,
            "well_temp_lin2d_mean": df_meta['well_temp_lin2d_mean'].values,
            "well_2d_temp_npr_mean": df_meta['well_2d_temp_npr_mean'].values
        },

        "baseline_value": baseline_value,
        "window_size_ori": window_size_ori,
        "window_size_1stder": window_size_1stder,
        "margin": margin
    }

    save_path = os.path.join(exp_path, "preprocessed_curves_data.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"  -> Saved numerical results and metadata to {save_path}")

# ==========================================
# 8. MAIN EXECUTION LOOP
# ==========================================

if __name__ == "__main__":
    n_wells = 10
    n_a_type = "v04"

    exp_folder = "/Users/kautsarg/Documents/Run Data/trial test data"
    exp_paths = [Path(exp_folder, name) for name in os.listdir(exp_folder) if name != ".DS_Store"]
    # exp_paths = [Path(exp_folder, "D20250808_E00_C00_F4500KHz_U_Sample_7")]
    
    window_size_ori = 50
    window_size_1stder = 200
    margin = (window_size_1stder - 1) // 2

    curve_labels = ["Original Curve", "1st Derivative", "1st Derivative Moving Avg", "Cleaned Curve", 
                    "Cleaned Curve (Lowest Crossing)", "Moving Avg Curve", "1st Derivative", 
                    "1st Derivative Moving Avg", "Cleaned Curve", "Cleaned Curve (Lowest Crossing)"]

    for exp_path in exp_paths:
        print(f"Processing Experiment: {exp_path}")
        
        # 1. Load Data 
        exp = titan_load_and_preprocessing(exp_path, n_wells=n_wells, start_type="temperature",
                                            end_time_min=60, n_a_type=n_a_type,
                                            print_status=False, plt_gain_calib=False, save_gain_calib=False)
        
        # 2. Extract DataFrames (Pixels & Temps with Metadata)
        print("  -> Generating Pixel & Temp DataFrames...")
        pixel_temp_dfs = extract_pixel_temp_dataframes(exp)
        
        # 3. Generate the Pixel vs. Temperature Interaction Grids
        print("  -> Building Pixel vs Temp Interactions...")
        plot_pixel_temp_interactions(exp, exp_path)
        
        # 4. Extract and Stack Arrays Natively for Curve Processing
        print("  -> Processing Sigmoid Curves...")
        X_time, Y_well, X_2d_bs_active = reconstruct_data(exp, attr_str="well_2d_bs_active")
        
        # 5. Apply Baseline Shift (Ensuring all positive for fitting)
        baseline_value = 0
        if (np.min(X_2d_bs_active) < 0):
            baseline_value = -np.min(X_2d_bs_active) + 1e-9
            X_2d_bs_active = X_2d_bs_active + baseline_value
        
        # 6. Process Curves & Extract Features
        processed_curves, indices_dict = process_experiment_data(
            X_2d_bs_active, X_time, window_size_ori, window_size_1stder, margin
        )
        
        # 7. Fit Sigmoid Curves
        fitting_results = run_all_fits(processed_curves, indices_dict, X_time)
        
        # 8. Save Raw Data & DataFrames to Disk (Now includes baseline_value)
        save_experiment_data_restructured(exp_path, fitting_results, processed_curves, 
                                     indices_dict, pixel_temp_dfs, baseline_value, 
                                     Y_well, X_time, exp, 
                                     window_size_ori, window_size_1stder, margin)
        
        # 9. Generate Interactive Sigmoid Visualizations
        print("  -> Building Sigmoid Grids...")
        unique_wells = np.unique(Y_well)
        plot_interactive_sigmoid_grids(
            exp_path, unique_wells, Y_well, X_time, 
            processed_curves, fitting_results, indices_dict, curve_labels
        )
        
        print("  ✓ Experiment complete!\n")