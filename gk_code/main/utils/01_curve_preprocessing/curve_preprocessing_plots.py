import warnings
import numpy as np

from bokeh.plotting import figure, save, output_file
from bokeh.layouts import gridplot, column, row, Spacer
from bokeh.models import (
    ColumnDataSource, CustomJS, Div, HoverTool, Span,
    CrosshairTool, TapTool, CheckboxButtonGroup, CustomJSFilter, CDSView
)

import sys
sys.path.insert(0, '..')
import config

# ==========================================
# PLOTTING MODULE 1: SIGMOID GRIDS
# ==========================================

def draw_stats(p, data_array, timestamps):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        y_mean, y_std = np.nanmean(data_array, axis=0), np.nanstd(data_array, axis=0)
    p.line(x=timestamps, y=y_mean, color="red", line_width=1.0)
    p.line(x=timestamps, y=y_mean + y_std, color="red", line_width=0.5, line_dash="dashed", alpha=0.8)
    p.line(x=timestamps, y=y_mean - y_std, color="red", line_width=0.5, line_dash="dashed", alpha=0.8)

def normalize_array(arr):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
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

def plot_interactive_sigmoid_grids(save_exp_path, unique_wells, ori_well, ori_timestamps, processed_curves, fitting_results, indices_dict, curve_labels, ds_step=5, precision=4):
    col_to_idx_map = {
        0: (indices_dict["cleaned_idx"], indices_dict["cleaned_lowest_idx"]),
        1: (indices_dict["cleaned_idx"], indices_dict["cleaned_lowest_idx"]),
        2: (indices_dict["cleaned_idx"], indices_dict["cleaned_lowest_idx"]),
        3: (indices_dict["cleaned_idx"],), 4: (indices_dict["cleaned_lowest_idx"],),
    }

    fit_key_map = {
        0: "original", 3: "cleaned_std", 4: "cleaned_lowest",
    }
    stats_cols = {0, 3, 4}
    group_a_cols, group_b_cols = [0, 3, 4], [1, 2]

    ts_ds = np.round(ori_timestamps[::ds_step], precision).tolist()

    plot_size = 350
    scatter_kwargs = dict(size=5, color="grey", alpha=0.6, selection_color="red", selection_alpha=1.0, nonselection_color="grey", nonselection_alpha=0.1)

    for row_idx, well in enumerate(unique_wells):
        filename = f"{save_exp_path}/well_{well}_sigmoid_curves.html"
        output_file(filename, title=f"Well {well} Sigmoid Curves")

        mask = (ori_well == well)
        hex_color = config.WELL_COLORS[row_idx % len(config.WELL_COLORS)]
        num_curves = np.sum(mask)

        data_dict = {'xs': [ts_ds for _ in range(num_curves)], 'color': [hex_color] * num_curves}

        for col_idx, curves in enumerate(processed_curves):
            y_visual = np.round(curves[mask][:, ::ds_step], precision)
            data_dict[f'ys_{col_idx}'] = y_visual.tolist()

            idx_tuple = col_to_idx_map[col_idx]
            idx_1 = idx_tuple[0][mask]
            data_dict[f'mx1_{col_idx}'] = np.round(ori_timestamps[idx_1], precision).tolist()
            data_dict[f'my1_{col_idx}'] = np.round(curves[mask][np.arange(num_curves), idx_1], precision).tolist()

            if len(idx_tuple) > 1:
                idx_2 = idx_tuple[1][mask]
                data_dict[f'mx2_{col_idx}'] = np.round(ori_timestamps[idx_2], precision).tolist()
                data_dict[f'my2_{col_idx}'] = np.round(curves[mask][np.arange(num_curves), idx_2], precision).tolist()

            if col_idx in stats_cols:
                fit_key = fit_key_map[col_idx]
                fc_mask = fitting_results[fit_key]["fitted_full"][mask]
                sc_mask = fitting_results[fit_key]["fitted_stretched"][mask]
                norm_fc_mask, norm_sc_mask = normalize_array(fc_mask), normalize_array(sc_mask)

                data_dict[f'fitted_ys_{col_idx}'] = np.round(fc_mask[:, ::ds_step], precision).tolist()
                data_dict[f'stretched_ys_{col_idx}'] = np.round(sc_mask[:, ::ds_step], precision).tolist()
                data_dict[f'norm_fitted_ys_{col_idx}'] = np.round(norm_fc_mask[:, ::ds_step], precision).tolist()
                data_dict[f'norm_stretched_ys_{col_idx}'] = np.round(norm_sc_mask[:, ::ds_step], precision).tolist()

                data_dict[f'fitted_my1_{col_idx}'] = np.round(fc_mask[np.arange(num_curves), idx_1], precision).tolist()
                data_dict[f'stretched_my1_{col_idx}'] = np.round(sc_mask[np.arange(num_curves), idx_1], precision).tolist()
                data_dict[f'norm_fitted_my1_{col_idx}'] = np.round(norm_fc_mask[np.arange(num_curves), idx_1], precision).tolist()
                data_dict[f'norm_stretched_my1_{col_idx}'] = np.round(norm_sc_mask[np.arange(num_curves), idx_1], precision).tolist()

                if len(idx_tuple) > 1:
                    data_dict[f'fitted_my2_{col_idx}'] = np.round(fc_mask[np.arange(num_curves), idx_2], precision).tolist()
                    data_dict[f'stretched_my2_{col_idx}'] = np.round(sc_mask[np.arange(num_curves), idx_2], precision).tolist()
                    data_dict[f'norm_fitted_my2_{col_idx}'] = np.round(norm_fc_mask[np.arange(num_curves), idx_2], precision).tolist()
                    data_dict[f'norm_stretched_my2_{col_idx}'] = np.round(norm_sc_mask[np.arange(num_curves), idx_2], precision).tolist()

        source = ColumnDataSource(data=data_dict)
        r1_plots, r2_plots, r3_plots, r4_plots, r5_plots = [], [], [], [], []

        for col_idx in range(len(processed_curves)):
            title = curve_labels[col_idx]

            p1 = figure(title=title, width=plot_size, height=plot_size, tools="pan,wheel_zoom,box_zoom,reset,tap", output_backend="webgl")
            if col_idx == 0: p1.yaxis.axis_label = f"Well {well}"
            p1.multi_line(xs='xs', ys=f'ys_{col_idx}', color='color', source=source, line_width=1.0, alpha=0.3, selection_color="red", selection_alpha=1.0, nonselection_color=hex_color, nonselection_alpha=0.05)
            p1.scatter(x=f'mx1_{col_idx}', y=f'my1_{col_idx}', source=source, **scatter_kwargs)
            if len(col_to_idx_map[col_idx]) > 1: p1.scatter(x=f'mx2_{col_idx}', y=f'my2_{col_idx}', source=source, **scatter_kwargs)

            if col_idx in stats_cols:
                draw_stats(p1, np.round(processed_curves[col_idx][mask][:, ::ds_step], precision), ts_ds)
            r1_plots.append(p1)

            if col_idx not in stats_cols:
                r2_plots.append(blank_plot(plot_size)); r3_plots.append(blank_plot(plot_size)); r4_plots.append(blank_plot(plot_size)); r5_plots.append(blank_plot(plot_size))
                continue

            fit_key = fit_key_map[col_idx]
            fc_data_ds = np.round(fitting_results[fit_key]["fitted_full"][mask][:, ::ds_step], precision)
            sc_data_ds = np.round(fitting_results[fit_key]["fitted_stretched"][mask][:, ::ds_step], precision)
            norm_fc_data_ds = np.round(normalize_array(fitting_results[fit_key]["fitted_full"][mask])[:, ::ds_step], precision)
            norm_sc_data_ds = np.round(normalize_array(fitting_results[fit_key]["fitted_stretched"][mask])[:, ::ds_step], precision)

            p2 = figure(width=plot_size, height=plot_size, tools="pan,wheel_zoom,box_zoom,reset,tap", output_backend="webgl")
            if col_idx == 0: p2.yaxis.axis_label = "Fit"
            p2.multi_line(xs='xs', ys=f'fitted_ys_{col_idx}', color='color', source=source, line_width=1.0, alpha=0.3, selection_color="red", selection_alpha=1.0, nonselection_alpha=0.05)
            p2.scatter(x=f'mx1_{col_idx}', y=f'fitted_my1_{col_idx}', source=source, **scatter_kwargs)
            if len(col_to_idx_map[col_idx]) > 1: p2.scatter(x=f'mx2_{col_idx}', y=f'fitted_my2_{col_idx}', source=source, **scatter_kwargs)
            draw_stats(p2, fc_data_ds, ts_ds)
            r2_plots.append(p2)

            p3 = figure(width=plot_size, height=plot_size, tools="pan,wheel_zoom,box_zoom,reset,tap", output_backend="webgl")
            if col_idx == 0: p3.yaxis.axis_label = "Stretched Fit"
            p3.multi_line(xs='xs', ys=f'stretched_ys_{col_idx}', color='color', source=source, line_width=1.0, alpha=0.3, selection_color="red", selection_alpha=1.0, nonselection_alpha=0.05)
            p3.scatter(x=f'mx1_{col_idx}', y=f'stretched_my1_{col_idx}', source=source, **scatter_kwargs)
            if len(col_to_idx_map[col_idx]) > 1: p3.scatter(x=f'mx2_{col_idx}', y=f'stretched_my2_{col_idx}', source=source, **scatter_kwargs)
            draw_stats(p3, sc_data_ds, ts_ds)
            r3_plots.append(p3)

            p4 = figure(width=plot_size, height=plot_size, tools="pan,wheel_zoom,box_zoom,reset,tap", output_backend="webgl")
            if col_idx == 0: p4.yaxis.axis_label = "Norm Fit"
            p4.multi_line(xs='xs', ys=f'norm_fitted_ys_{col_idx}', color='color', source=source, line_width=1.0, alpha=0.3, selection_color="red", selection_alpha=1.0, nonselection_alpha=0.05)
            p4.scatter(x=f'mx1_{col_idx}', y=f'norm_fitted_my1_{col_idx}', source=source, **scatter_kwargs)
            if len(col_to_idx_map[col_idx]) > 1: p4.scatter(x=f'mx2_{col_idx}', y=f'norm_fitted_my2_{col_idx}', source=source, **scatter_kwargs)
            draw_stats(p4, norm_fc_data_ds, ts_ds)
            r4_plots.append(p4)

            p5 = figure(width=plot_size, height=plot_size, tools="pan,wheel_zoom,box_zoom,reset,tap", output_backend="webgl")
            if col_idx == 0: p5.yaxis.axis_label = "Norm Stretched"
            p5.multi_line(xs='xs', ys=f'norm_stretched_ys_{col_idx}', color='color', source=source, line_width=1.0, alpha=0.3, selection_color="red", selection_alpha=1.0, nonselection_alpha=0.05)
            p5.scatter(x=f'mx1_{col_idx}', y=f'norm_stretched_my1_{col_idx}', source=source, **scatter_kwargs)
            if len(col_to_idx_map[col_idx]) > 1: p5.scatter(x=f'mx2_{col_idx}', y=f'norm_stretched_my2_{col_idx}', source=source, **scatter_kwargs)
            draw_stats(p5, norm_sc_data_ds, ts_ds)
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
# PLOTTING MODULE 2: PIXEL VS TEMP
# ==========================================

def plot_pixel_temp_interactions(all_exp_data, save_exp_path, ds_step=5, precision=4):
    output_file(f"{save_exp_path}/all_experiments_combined_interactive.html", title="Wells Overlaid Across Experiments")
    p_width, p_height = 450, 225
    all_well_layouts = []

    # ====================================================================
    # COLOR PALETTES
    # ====================================================================
    PALETTE_NL   = ['purple', 'crimson', 'teal', 'darkmagenta']
    PALETTE_LIN  = ['navy', 'darkgreen', 'saddlebrown', 'midnightblue']
    PALETTE_TEMP = ['darkorange', 'dodgerblue', 'deeppink', 'limegreen']

    if not all_exp_data: return
    num_wells = len(all_exp_data[0].wells_list)
    num_exps = len(all_exp_data)

    # ====================================================================
    # 1. GLOBAL MASTER CONTROLS (At the top of the page)
    # ====================================================================
    global_checkbox = CheckboxButtonGroup(
        labels=[f"Toggle Experiment {i}" for i in range(num_exps)],
        active=list(range(num_exps)),
        button_type="success",
        sizing_mode="stretch_width"
    )

    global_js_filter = CustomJSFilter(args=dict(checkbox=global_checkbox), code="""
        const active = checkbox.active;
        const exp_ids = source.data['exp_id'];
        const res = new Array(exp_ids.length);
        for (let i = 0; i < exp_ids.length; i++) {
            res[i] = active.includes(exp_ids[i]);
        }
        return res;
    """)

    all_sources = []
    spans_by_exp = {i: [] for i in range(num_exps)}

    # ====================================================================
    # 2. DATA EXTRACTION & PLOT BUILDING LOOP
    # ====================================================================
    for w_idx in range(num_wells):

        pix_xs, pix_ys_lin, pix_ys_nl = [], [], []
        pix_group_id, pix_pixel_id, pix_exp_id = [], [], []
        pix_color_nl, pix_color_lin = [], []

        temp_act_xs, temp_act_ys_lin, temp_act_ys_nl = [], [], []
        temp_act_group_id, temp_act_exp_id, temp_act_color = [], [], []

        temp_inact_xs, temp_inact_ys_lin, temp_inact_ys_nl = [], [], []
        temp_inact_group_id, temp_inact_exp_id = [], []

        mean_xs, mean_ys, mean_color, mean_exp_id = [], [], [], []

        settled_lines, end_lines = [], []

        for vref_idx, exp_data in enumerate(all_exp_data):
            well = exp_data.wells_list[w_idx]

            c_idx = vref_idx % len(PALETTE_NL)
            color_nl = PALETTE_NL[c_idx]
            color_lin = PALETTE_LIN[c_idx]
            color_temp = PALETTE_TEMP[c_idx]

            idx_settled = well.idx_settled
            idx_end = well.idx_end
            idx_active = well.idx_active
            time_npr = well.time_npr

            time_ds = np.round(time_npr[::ds_step], precision).tolist()

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

            # Accumulate Pixels
            pix_xs.extend([time_ds for _ in range(n_active_pixels)])
            pix_ys_lin.extend([np.round(well_2d_bs_active[::ds_step, i], precision).tolist() for i in range(n_active_pixels)])
            pix_ys_nl.extend([np.round(well_2d_nl_bs_active[::ds_step, i], precision).tolist() for i in range(n_active_pixels)])
            pix_group_id.extend(active_temp_mapping.tolist())
            pix_pixel_id.extend(np.where(idx_active)[0].tolist())
            pix_exp_id.extend([vref_idx] * n_active_pixels)
            pix_color_nl.extend([color_nl] * n_active_pixels)
            pix_color_lin.extend([color_lin] * n_active_pixels)

            # Accumulate Active Temps
            temp_act_xs.extend([time_ds for _ in active_temp_groups])
            temp_act_ys_lin.extend([np.round(well_temp_2D_NEW[::ds_step, i], precision).tolist() for i in active_temp_groups])
            temp_act_ys_nl.extend([np.round(well_2d_temp_npr[::ds_step, i], precision).tolist() for i in active_temp_groups])
            temp_act_group_id.extend(active_temp_groups.tolist())
            temp_act_exp_id.extend([vref_idx] * len(active_temp_groups))
            temp_act_color.extend([color_temp] * len(active_temp_groups))

            # Accumulate Inactive Temps
            temp_inact_xs.extend([time_ds for _ in inactive_temp_groups])
            temp_inact_ys_lin.extend([np.round(well_temp_2D_NEW[::ds_step, i], precision).tolist() for i in inactive_temp_groups])
            temp_inact_ys_nl.extend([np.round(well_2d_temp_npr[::ds_step, i], precision).tolist() for i in inactive_temp_groups])
            temp_inact_group_id.extend(inactive_temp_groups.tolist())
            temp_inact_exp_id.extend([vref_idx] * len(inactive_temp_groups))

            # Accumulate Mean Lines
            mean_xs.append(time_ds)
            mean_ys.append(np.round(well_temp_mean_then_lin[::ds_step], precision).tolist())
            mean_color.append(color_lin)
            mean_exp_id.append(vref_idx)

            settled_lines.append(time_npr[idx_settled])
            end_lines.append(time_npr[idx_end])

        # Create DataSources
        source_pixels = ColumnDataSource({'xs': pix_xs, 'ys_lin': pix_ys_lin, 'ys_nl': pix_ys_nl, 'group_id': pix_group_id, 'pixel_id': pix_pixel_id, 'exp_id': pix_exp_id, 'color_nl': pix_color_nl, 'color_lin': pix_color_lin})
        source_temps_active = ColumnDataSource({'xs': temp_act_xs, 'ys_lin': temp_act_ys_lin, 'ys_nl': temp_act_ys_nl, 'group_id': temp_act_group_id, 'exp_id': temp_act_exp_id, 'color_temp': temp_act_color})
        source_temps_inactive = ColumnDataSource({'xs': temp_inact_xs, 'ys_lin': temp_inact_ys_lin, 'ys_nl': temp_inact_ys_nl, 'group_id': temp_inact_group_id, 'exp_id': temp_inact_exp_id})
        source_mean = ColumnDataSource({'xs': mean_xs, 'ys': mean_ys, 'color_lin': mean_color, 'exp_id': mean_exp_id})

        all_sources.extend([source_pixels, source_temps_active, source_temps_inactive, source_mean])

        view_pixels = CDSView(filter=global_js_filter)
        view_temps_act = CDSView(filter=global_js_filter)
        view_temps_inact = CDSView(filter=global_js_filter)
        view_mean = CDSView(filter=global_js_filter)

        info_div = Div(text=f"<h3 style='color: grey;'>Well {w_idx}: Select a line to see indices here...</h3>", width=1400)

        # Selection Callbacks
        cb_pixel_to_temp = CustomJS(args=dict(sp=source_pixels, st=source_temps_active, div=info_div), code="""
            const selected_pixels = sp.selected.indices;
            if (selected_pixels.length === 0) {
                st.selected.indices = [];
                div.text = "<h3 style='color: grey;'>Select a line to see indices here...</h3>";
                return;
            }
            const target_group_id = sp.data['group_id'][selected_pixels[0]];
            const target_exp_id = sp.data['exp_id'][selected_pixels[0]];
            const ui_color = sp.data['color_lin'][selected_pixels[0]];

            const temp_groups = st.data['group_id'];
            const temp_exps = st.data['exp_id'];
            const temp_to_select = [];
            for (let i = 0; i < temp_groups.length; i++) {
                if (temp_groups[i] === target_group_id && temp_exps[i] === target_exp_id) {
                    temp_to_select.push(i); break;
                }
            }
            st.selected.indices = temp_to_select;

            const pixel_groups = sp.data['group_id'];
            const pixel_exps = sp.data['exp_id'];
            const mapped_pixel_ids = [];
            for (let i = 0; i < pixel_groups.length; i++) {
                if (pixel_groups[i] === target_group_id && pixel_exps[i] === target_exp_id) {
                    mapped_pixel_ids.push(sp.data['pixel_id'][i]);
                }
            }
            div.text = `<h3 style='color: ${ui_color};'>Exp: <b>${target_exp_id}</b> | Pixel: <b>${sp.data['pixel_id'][selected_pixels[0]]}</b> | Group: <b>${target_group_id}</b> | All Pixels: <b>${mapped_pixel_ids.join(', ')}</b></h3>`;
        """)

        cb_temp_to_pixel = CustomJS(args=dict(sp=source_pixels, st=source_temps_active, div=info_div), code="""
            const selected_temps = st.selected.indices;
            if (selected_temps.length === 0) {
                sp.selected.indices = [];
                div.text = "<h3 style='color: grey;'>Select a line to see indices here...</h3>";
                return;
            }
            const target_group_id = st.data['group_id'][selected_temps[0]];
            const target_exp_id = st.data['exp_id'][selected_temps[0]];
            const ui_color = st.data['color_temp'][selected_temps[0]];

            const pixel_groups = sp.data['group_id'];
            const pixel_exps = sp.data['exp_id'];
            const pixels_to_select = [];
            const mapped_pixel_ids = [];
            for (let i = 0; i < pixel_groups.length; i++) {
                if (pixel_groups[i] === target_group_id && pixel_exps[i] === target_exp_id) {
                    pixels_to_select.push(i);
                    mapped_pixel_ids.push(sp.data['pixel_id'][i]);
                }
            }
            sp.selected.indices = pixels_to_select;
            div.text = `<h3 style='color: ${ui_color};'>Exp: <b>${target_exp_id}</b> | Selected Temp Group: <b>${target_group_id}</b> | Mapped Pixels: <b>${mapped_pixel_ids.join(', ')}</b></h3>`;
        """)

        source_pixels.selected.js_on_change('indices', cb_pixel_to_temp)
        source_temps_active.selected.js_on_change('indices', cb_temp_to_pixel)

        hover_pixel_nl = HoverTool(tooltips=[("Exp", "@exp_id"), ("Pixel", "@pixel_id"), ("Group", "@group_id"), ("Value", "$y")], line_policy="nearest")
        hover_pixel_lin = HoverTool(tooltips=[("Exp", "@exp_id"), ("Pixel", "@pixel_id"), ("Group", "@group_id"), ("Value", "$y")], line_policy="nearest")
        linked_crosshair = CrosshairTool(dimensions="height", line_color="black", line_alpha=0.3)

        p0 = figure(title=f"NL Pixels (Well {w_idx} All Exps)", width=p_width, height=p_height, tools=["pan", "wheel_zoom", "box_zoom", "reset", "tap", hover_pixel_nl, linked_crosshair], output_backend="webgl")
        p0.multi_line(xs='xs', ys='ys_nl', source=source_pixels, view=view_pixels, color='color_nl', alpha=0.3, selection_color="red", selection_alpha=1.0, nonselection_color='color_nl', nonselection_alpha=0.05)

        p1 = figure(title=f"Lin Pixels (Well {w_idx} All Exps)", width=p_width, height=p_height, tools=["pan", "wheel_zoom", "box_zoom", "reset", "tap", hover_pixel_lin, linked_crosshair], output_backend="webgl")
        p1.x_range = p0.x_range
        p1.multi_line(xs='xs', ys='ys_lin', source=source_pixels, view=view_pixels, color='color_lin', alpha=0.3, selection_color="red", selection_alpha=1.0, nonselection_color='color_lin', nonselection_alpha=0.05)

        p2 = figure(title=f"NL Temps (Well {w_idx} All Exps)", width=p_width, height=p_height, tools=["pan", "wheel_zoom", "box_zoom", "reset", linked_crosshair], output_backend="webgl")
        p2.x_range = p0.x_range
        p2.multi_line(xs='xs', ys='ys_nl', source=source_temps_inactive, view=view_temps_inact, color="lightgrey", alpha=0.3)
        active_temp_renderer_nl = p2.multi_line(xs='xs', ys='ys_nl', source=source_temps_active, view=view_temps_act, color='color_temp', alpha=0.4, selection_color="red", selection_alpha=1.0, nonselection_color='color_temp', nonselection_alpha=0.1)
        p2.add_tools(HoverTool(tooltips=[("Exp", "@exp_id"), ("Group", "@group_id"), ("Val", "$y")], renderers=[active_temp_renderer_nl]), TapTool(renderers=[active_temp_renderer_nl]))

        p3 = figure(title=f"Lin Temps (Well {w_idx} All Exps)", width=p_width, height=p_height, tools=["pan", "wheel_zoom", "box_zoom", "reset", linked_crosshair], output_backend="webgl")
        p3.x_range = p0.x_range
        p3.multi_line(xs='xs', ys='ys_lin', source=source_temps_inactive, view=view_temps_inact, color="lightgrey", alpha=0.3)
        active_temp_renderer_lin = p3.multi_line(xs='xs', ys='ys_lin', source=source_temps_active, view=view_temps_act, color='color_temp', alpha=0.4, selection_color="red", selection_alpha=1.0, nonselection_color='color_temp', nonselection_alpha=0.1)
        p3.multi_line(xs='xs', ys='ys', source=source_mean, view=view_mean, color='color_lin', line_width=1.5)
        p3.add_tools(HoverTool(tooltips=[("Exp", "@exp_id"), ("Group", "@group_id"), ("Val", "$y")], renderers=[active_temp_renderer_lin]), TapTool(renderers=[active_temp_renderer_lin]))

        for p_fig in [p0, p1, p2, p3]:
            for i_exp, (ts, te) in enumerate(zip(settled_lines, end_lines)):
                span_s = Span(location=ts, dimension='height', line_color='green', line_alpha=0.5, line_dash='dashed')
                span_e = Span(location=te, dimension='height', line_color='red', line_alpha=0.5, line_dash='dashed')
                p_fig.add_layout(span_s)
                p_fig.add_layout(span_e)
                spans_by_exp[i_exp].extend([span_s, span_e])

        well_layout = column(info_div, row(p0, p1, p2, p3), Spacer(height=50))
        all_well_layouts.append(well_layout)

    global_checkbox_cb = CustomJS(args=dict(
        sources=all_sources, spans=spans_by_exp, checkbox=global_checkbox
    ), code="""
        for (let s of sources) {
            s.change.emit();
        }
        const active = checkbox.active;
        for (const [exp_id, span_list] of Object.entries(spans)) {
            const is_visible = active.includes(parseInt(exp_id));
            for (let span of span_list) {
                span.visible = is_visible;
            }
        }
    """)
    global_checkbox.js_on_change('active', global_checkbox_cb)

    header_title = Div(text="<h1>Experiment Visibility Controls</h1>", margin=(10, 10, 5, 10))
    master_layout = column(header_title, global_checkbox, Spacer(height=30), *all_well_layouts)

    save(master_layout)
