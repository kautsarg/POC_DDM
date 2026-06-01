import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re
import math
from scipy.signal import savgol_filter

# Colorblind-friendly palettes and accents reused across the module
DEFAULT_COLORBLIND_PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#F0E442",  # yellow
    "#000000",  # black
]

DEFAULT_SAFE_PALETTE = [
    "#88CCEE",
    "#CC6677",
    "#DDCC77",
    "#117733",
    "#332288",
    "#AA4499",
    "#44AA99",
    "#999933",
    "#882255",
    "#661100",
    "#6699CC",
    "#888888",
]

COLORBLIND_WELL_COLORS = getattr(px.colors.qualitative, "Colorblind", DEFAULT_COLORBLIND_PALETTE)
CSV_COLORBLIND_COLORS = getattr(px.colors.qualitative, "Safe", DEFAULT_SAFE_PALETTE)
TTP_MARKER_COLOR = "#CC79A7"
TTP_MARKER_BORDER_COLOR = "#7B4B94"
SETTLED_LINE_COLOR = "#0072B2"
ALERT_COLOR = "#D55E00"

def _time_minutes_for_well(well, *, zero_at: str = "settled") -> np.ndarray:
    """
    Return a per-sample time vector in minutes aligned to the well's plotted signals.

    Most signals in this module are derived from `well_3d_*[:, :, idx_settled:idx_end]`,
    so the time vector should also be sliced `idx_settled:idx_end`.

    Parameters
    ----------
    well : Well
        Well instance from exp.wells_list
    zero_at : {"settled", "start", "absolute"}
        - "settled": 0 min at idx_settled (matches typical plotted traces)
        - "start": 0 min at idx_start (injection/start-of-experiment)
        - "absolute": use raw `time_npr` converted to minutes (no re-zeroing)
    """
    if hasattr(well, "time_npr") and well.time_npr is not None and hasattr(well, "idx_settled") and hasattr(well, "idx_end"):
        t = np.asarray(well.time_npr)[int(well.idx_settled):int(well.idx_end)]
        if t.size == 0:
            return np.asarray([], dtype=float)
        if zero_at == "settled":
            t0 = t[0]
            return (t - t0) / 60.0
        if zero_at == "start" and hasattr(well, "idx_start"):
            t0 = float(np.asarray(well.time_npr)[int(well.idx_start)])
            return (t - t0) / 60.0
        # "absolute" or fallback
        return t / 60.0

    # Fallback: use well.time_min if available (already sliced idx_settled:idx_end by property),
    # and re-zero if requested.
    if hasattr(well, "time_min") and well.time_min is not None:
        tmin = np.asarray(well.time_min, dtype=float)
        if tmin.size == 0:
            return tmin
        if zero_at == "settled":
            return tmin - tmin[0]
        return tmin

    raise ValueError("Could not determine time vector for well (missing time_npr/time_min).")


def _align_x_y(x: np.ndarray, y: np.ndarray):
    """Trim x/y to common length instead of inventing a new time axis."""
    x = np.asarray(x)
    y = np.asarray(y)
    n = min(len(x), len(y))
    if n == 0:
        return x[:0], y[:0]
    if len(x) != len(y):
        print(
            f"Warning: Time/signal length mismatch (time={len(x)}, signal={len(y)}). "
            f"Trimming both to {n} samples."
        )
    return x[:n], y[:n]


def sanitize_filename(filename):
    """Sanitize filename for safe file saving"""
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Limit length
    if len(sanitized) > 100:
        sanitized = sanitized[:100]
    return sanitized


def _hex_to_rgba(hex_color, alpha=1.0):
    """Convert hex color to rgba string for Plotly"""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f'rgba({r}, {g}, {b}, {alpha})'


def plot_wells_grid_plotly(
    exp,
    save_path=None,
    experiment_name="Experiment",
    figsize=(1400, 1000),
    ttp_results=None,
    show=True,
    baseline_subtract: bool = True,
):
    """
    Create a single plotly plot showing all wells in a 5x2 grid layout, similar to the reference image.

    Parameters
    ----------
    exp : Experiment
        The experiment object containing wells
    save_path : str, optional
        Directory path to save the HTML plot. If None, plot is shown only.
    experiment_name : str
        Name of the experiment for plot title
    figsize : tuple
        Figure size (width, height) in pixels
    ttp_results : dict, optional
        Dictionary containing TTP (Time To Peak) results for each well. If provided, TTP markers will be plotted.
    baseline_subtract : bool, optional
        If True (default), plot ``well_2d_nl_bs_active_mean`` (baseline-subtracted at idx_settled).
        If False, plot ``well_2d_nl_active_mean`` (raw chem active-pixel mean, no subtraction).

    Returns
    -------
    go.Figure
        The figure object containing all wells in a grid layout
    """
    
    n_wells = len(exp.wells_list)
    
    # Create subplots: 5 rows, 2 columns (5x2 grid)
    fig = make_subplots(
        rows=5, cols=2,
        subplot_titles=[f'Well {i+1}' for i in range(n_wells)],
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Colors for different wells
    colors = COLORBLIND_WELL_COLORS
    
    y_title = "Signal (a.u.)"
    if baseline_subtract:
        grid_title = f"{experiment_name} - All Wells (5x2 Grid)"
        html_suffix = "_all_wells_grid"
    else:
        grid_title = f"{experiment_name} - All Wells (5x2 Grid, raw chem, no baseline subtract)"
        html_suffix = "_all_wells_grid_raw_no_bs"

    for well_idx, well in enumerate(exp.wells_list):
        try:
            if baseline_subtract:
                signal_data = well.well_2d_nl_bs_active_mean
            else:
                signal_data = well.well_2d_nl_active_mean

            time_data = _time_minutes_for_well(well, zero_at="settled")
            time_data, signal_data = _align_x_y(time_data, signal_data)
            
            # Debug information for time indexing
            print(f"DEBUG Well {well_idx + 1}: Signal length: {len(signal_data)}, Time length: {len(time_data)}")
            print(f"DEBUG Well {well_idx + 1}: Time range: {time_data[0]:.2f} to {time_data[-1]:.2f} min")
            print(f"DEBUG Well {well_idx + 1}: Signal range: {np.min(signal_data):.4f} to {np.max(signal_data):.4f}")
            
            # Calculate row and column position for this well
            row = (well_idx // 2) + 1
            col = (well_idx % 2) + 1
            
            color = colors[well_idx % len(colors)]
            
            # Plot the main signal for this well
            fig.add_trace(
                go.Scatter(
                    x=time_data,
                    y=signal_data,
                    mode='lines',
                    name=f'Well {well_idx + 1}',
                    line=dict(color=color, width=2),
                    showlegend=False
                ),
                row=row, col=col
            )
        
            
        except Exception as e:
            print(f"Error processing Well {well_idx + 1}: {str(e)}")
            continue
    
    # Update layout
    fig.update_layout(
        title=grid_title,
        height=figsize[1],
        width=figsize[0],
        font=dict(size=12),
        showlegend=False
    )
    
    # Update x-axis labels for bottom row only
    for col in [1, 2]:
        fig.update_xaxes(title_text="Time (minutes)", row=5, col=col)
    
    # Update y-axis labels for left column only
    for row in range(1, 6):
        fig.update_yaxes(title_text=y_title, row=row, col=1)
    
    # Add grid to all subplots
    for row in range(1, 6):
        for col in range(1, 3):
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=row, col=col)
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=row, col=col)
    
    # Show the plot
    if show:
        fig.show()
    
    # Save as HTML if save_path is provided
    if save_path:
        try:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Sanitize the experiment name for use in filename
            safe_experiment_name = sanitize_filename(experiment_name)
            html_filename = f"{safe_experiment_name}{html_suffix}.html"
            html_path = save_path / html_filename
            
            print(f"Saving grid HTML file to: {html_path}")
            
            # Write the HTML file
            fig.write_html(str(html_path))
            print(f"Grid HTML file saved successfully: {html_path}")
            
        except Exception as e:
            print(f"Error saving grid HTML file: {str(e)}")
    
    return fig


def plot_wells_average_combined_plotly(
    exp,
    save_path=None,
    experiment_name="Experiment",
    figsize=(1400, 700),
    wells_to_show=None,
    wells_to_hide=None,
    hide_mode="legendonly",
    show=True,
):
    """
    Plot all wells' average traces on a single figure (no grid).
    Uses the per-well average (unfiltered): well.well_2d_nl_bs_active_mean.

    Parameters
    ----------
    exp : Experiment
        Experiment object containing wells.
    save_path : str, optional
        Directory path to save the HTML plot. If None, plot is shown only.
    experiment_name : str
        Name of the experiment for plot title.
    figsize : tuple
        Figure size (width, height) in pixels.
    wells_to_show : iterable of int, optional
        1-based indices of wells that should be visible when the plot opens.
        If provided, all other wells start hidden but remain available via the legend.
    wells_to_hide : iterable of int, optional
        1-based indices of wells that should start hidden while all others remain visible.
        Ignored if `wells_to_show` is provided.
    hide_mode : {"legendonly", "skip"}, optional
        How hidden wells should be treated. "legendonly" keeps the legend entry so you can
        toggle visibility later; "skip" omits the trace entirely.
    """
    n_wells = len(exp.wells_list)
    fig = go.Figure()

    # Normalize well selection inputs to zero-based indices
    if wells_to_show is not None:
        wells_to_show = {int(idx) - 1 for idx in wells_to_show}
    if wells_to_hide is not None:
        wells_to_hide = {int(idx) - 1 for idx in wells_to_hide}

    if hide_mode not in {"legendonly", "skip"}:
        raise ValueError("hide_mode must be either 'legendonly' or 'skip'")

    colors = COLORBLIND_WELL_COLORS

    for well_idx, well in enumerate(exp.wells_list):
        try:
            y = well.well_2d_nl_bs_active_mean
            x = _time_minutes_for_well(well, zero_at="settled")
            x, y = _align_x_y(x, y)

            # Decide initial visibility for this well
            add_trace = True
            visible_state = True

            if wells_to_show is not None:
                if well_idx not in wells_to_show:
                    if hide_mode == "skip":
                        add_trace = False
                    else:
                        visible_state = "legendonly"
            elif wells_to_hide is not None and well_idx in wells_to_hide:
                if hide_mode == "skip":
                    add_trace = False
                else:
                    visible_state = "legendonly"

            if not add_trace:
                continue

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode='lines',
                    name=f'Well {well_idx + 1}',
                    line=dict(color=colors[well_idx % len(colors)], width=2),
                    showlegend=True,
                    visible=visible_state
                )
            )
        except Exception as e:
            print(f"Error adding Well {well_idx + 1} to average combined plot: {str(e)}")
            continue

    fig.update_layout(
        title=f'{experiment_name} - All Wells Average (Unfiltered) Combined',
        height=figsize[1],
        width=figsize[0],
        font=dict(size=12),
        xaxis_title="Time (minutes)",
        yaxis_title="NL BS Active Mean (a.u.)",
        showlegend=True
    )

    if show:
        fig.show()

    if save_path:
        try:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            safe_experiment_name = sanitize_filename(experiment_name)
            html_filename = f"{safe_experiment_name}_avg_combined.html"
            html_path = save_path / html_filename
            print(f"Saving average combined HTML file to: {html_path}")
            fig.write_html(str(html_path))
            print(f"Average combined HTML file saved successfully: {html_path}")
        except Exception as e:
            print(f"Error saving average combined HTML file: {str(e)}")

    return fig


def plot_all_wells_combined_plotly(exp, save_path=None, experiment_name="Experiment", figsize=(1400, 900), ttp_results=None, show=True):
    """
    Create a combined plotly plot showing all wells together for comparison.
    
    Parameters
    ----------
    exp : Experiment
        The experiment object containing wells
    save_path : str, optional
        Directory path to save the HTML plot. If None, plot is shown only.
    experiment_name : str
        Name of the experiment for plot title
    figsize : tuple
        Figure size (width, height) in pixels
    ttp_results : dict, optional
        Dictionary containing TTP (Time To Peak) results for each well. If provided, TTP markers will be plotted on the signal plot.
        
    Returns
    -------
    go.Figure
        The combined figure object
    """
    
    n_wells = len(exp.wells_list)
    
    # Create subplots: 3 rows (signal, first deriv, second deriv), 1 column
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('All Wells - NL BS Active Mean (Filtered)', 'All Wells - First Derivative', 'All Wells - Second Derivative'),
        vertical_spacing=0.1,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Colors for different wells
    colors = COLORBLIND_WELL_COLORS
    
    for well_idx, well in enumerate(exp.wells_list):
        try:
            # Get the data
            signal_data = well.well_2d_nl_bs_active_mean_filt
            
            time_data = _time_minutes_for_well(well, zero_at="settled")
            time_data, signal_data = _align_x_y(time_data, signal_data)
            
            # Calculate derivatives
            first_derivative = np.gradient(signal_data)
            second_derivative = np.gradient(first_derivative)
            
            # Debug information for time indexing
            print(f"DEBUG Combined Plot Well {well_idx + 1}: Signal length: {len(signal_data)}, Time length: {len(time_data)}")
            print(f"DEBUG Combined Plot Well {well_idx + 1}: Time range: {time_data[0]:.2f} to {time_data[-1]:.2f} min")
            
            color = colors[well_idx % len(colors)]
            well_name = f'Well {well_idx + 1}'
            
            # Plot 1: NL Active Mean
            fig.add_trace(
                go.Scatter(
                    x=time_data,
                    y=signal_data,
                    mode='lines',
                    name=well_name,
                    line=dict(color=color, width=2),
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Plot 2: First Derivative
            fig.add_trace(
                go.Scatter(
                    x=time_data,
                    y=first_derivative,
                    mode='lines',
                    name=well_name,
                    line=dict(color=color, width=2),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Plot 3: Second Derivative
            fig.add_trace(
                go.Scatter(
                    x=time_data,
                    y=second_derivative,
                    mode='lines',
                    name=well_name,
                    line=dict(color=color, width=2),
                    showlegend=False
                ),
                row=3, col=1
            )
            
            # TTP markers hidden for raw signal plot as requested
            # if ttp_results is not None and well_idx in ttp_results and ttp_results[well_idx]['found_ttp']:
            #     ttp_time = ttp_results[well_idx]['ttp_time']
            #     # Find the closest time index for accurate plotting
            #     closest_idx = np.argmin(np.abs(time_data - ttp_time))
            #     ttp_value_at_time = signal_data[closest_idx]
            #     
            #     # Plot the TTP as a purple diamond on the signal plot
            #     fig.add_trace(
            #         go.Scatter(
            #             x=[ttp_time],
            #             y=[ttp_value_at_time],
            #             mode='markers',
            #             name=f'Well {well_idx + 1} TTP',
            #             marker=dict(color='purple', size=12, symbol='diamond', line=dict(color='darkviolet', width=2)),
            #             showlegend=False
            #         ),
            #         row=1, col=1
            #     )
            #     
            #     # Add TTP annotation
            #     fig.add_annotation(
            #         x=ttp_time,
            #         y=ttp_value_at_time,
            #         text=f'Well {well_idx + 1}<br>TTP: {ttp_time:.1f} min',
            #         showarrow=True,
            #         arrowhead=2,
            #         arrowsize=1,
            #         arrowwidth=2,
            #         arrowcolor='purple',
            #         ax=20,
            #         ay=-30,
            #         bgcolor='purple',
            #         bordercolor='darkviolet',
            #         borderwidth=2,
            #         font=dict(color='white', size=10),
            #         row=1, col=1
            #     )
            
        except Exception as e:
            print(f"Error processing Well {well_idx + 1} for combined plot: {str(e)}")
            continue
    
    # Update layout
    fig.update_layout(
        title=f'{experiment_name} - All Wells Combined',
        height=figsize[1],
        width=figsize[0],
        font=dict(size=12),
        showlegend=True
    )
    
    # Update x-axis labels
    fig.update_xaxes(title_text="Time (minutes)", row=3, col=1)
    
    # Update y-axis labels
    fig.update_yaxes(title_text="NL BS Active Mean (Filtered) (a.u.)", row=1, col=1)
    fig.update_yaxes(title_text="First Derivative (a.u./min)", row=2, col=1)
    fig.update_yaxes(title_text="Second Derivative (a.u./min²)", row=3, col=1)
    
    # Add zero lines for derivatives
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
    
    # Show the plot
    if show:
        fig.show()
    
    # Save as HTML if save_path is provided
    if save_path:
        try:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Sanitize the experiment name for use in filename
            safe_experiment_name = sanitize_filename(experiment_name)
            html_filename = f"{safe_experiment_name}_all_wells_combined.html"
            html_path = save_path / html_filename
            
            print(f"Saving combined HTML file to: {html_path}")
            
            # Write the HTML file
            fig.write_html(str(html_path))
            print(f"Combined HTML file saved successfully: {html_path}")
            
        except Exception as e:
            print(f"Error saving combined HTML file: {str(e)}")
    
    return fig


def plot_all__wells_Combined_plotly(
    exp,
    save_path=None,
    experiment_name="Experiment",
    figsize=(1400, 600),
    ttp_results=None,
    wells_to_show=None,
    wells_to_hide=None,
    hide_mode="legendonly",
):
    """
    Create a combined plotly plot showing the linearized signal for all wells on a single set of axes.

    Parameters
    ----------
    exp : Experiment
        The experiment object containing wells.
    save_path : str, optional
        Directory path to save the HTML plot. If None, plot is shown only.
    experiment_name : str
        Name of the experiment for plot title.
    figsize : tuple
        Figure size (width, height) in pixels.
    ttp_results : dict, optional
        Dictionary containing TTP (Time To Peak) results for each well. If provided, TTP markers will be plotted.

    Returns
    -------
    go.Figure
        The figure object containing all wells' linearized signals combined.
    """

    n_wells = len(exp.wells_list)

    fig = go.Figure()
    colors = COLORBLIND_WELL_COLORS
    
    # Normalize well selection inputs to zero-based indices
    if wells_to_show is not None:
        wells_to_show = {int(idx) - 1 for idx in wells_to_show}
    if wells_to_hide is not None:
        wells_to_hide = {int(idx) - 1 for idx in wells_to_hide}
    if hide_mode not in {"legendonly", "skip"}:
        raise ValueError("hide_mode must be either 'legendonly' or 'skip'")
    
    for well_idx, well in enumerate(exp.wells_list):
        try:
            if not hasattr(well, "well_2d_bs_active_mean") or well.well_2d_bs_active_mean is None:
                print(f"Warning: No linearized signal data found for Well {well_idx + 1}")
                continue

            signal_data = well.well_2d_bs_active_mean
            x = _time_minutes_for_well(well, zero_at="settled")
            x, signal_data = _align_x_y(x, signal_data)

            add_trace = True
            visible_state = True
            if wells_to_show is not None:
                if well_idx not in wells_to_show:
                    if hide_mode == "skip":
                        add_trace = False
                    else:
                        visible_state = "legendonly"
            elif wells_to_hide is not None and well_idx in wells_to_hide:
                if hide_mode == "skip":
                    add_trace = False
                else:
                    visible_state = "legendonly"
    
            if not add_trace:
                continue
    
            color = colors[well_idx % len(colors)]
            well_name = f"Well {well_idx + 1}"
    
            x = np.asarray(x)
    
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=signal_data,
                    mode="lines",
                    name=well_name,
                    line=dict(color=color, width=2),
                    showlegend=True,
                    visible=visible_state,
                )
            )

            if ttp_results is not None and well_idx in ttp_results and ttp_results[well_idx]["found_ttp"]:
                ttp_time = ttp_results[well_idx]["ttp_time"]
                closest_idx = np.argmin(np.abs(x - ttp_time))
                ttp_value = signal_data[closest_idx]

                fig.add_trace(
                    go.Scatter(
                        x=[x[closest_idx]],
                        y=[ttp_value],
                        mode="markers",
                        name=f"{well_name} TTP",
                        marker=dict(
                            color=TTP_MARKER_COLOR,
                            size=12,
                            symbol="diamond",
                            line=dict(color=TTP_MARKER_BORDER_COLOR, width=2)
                        ),
                        showlegend=False,
                    )
                )
        except Exception as exc:
            print(f"Error processing Well {well_idx + 1} for linearized combined plot: {exc}")
            continue

    fig.update_layout(
        title=f"{experiment_name} - Linearized Signal (All Wells)",
        xaxis_title="Time (minutes)",
        yaxis_title="Linearized Signal (a.u.)",
        height=figsize[1],
        width=figsize[0],
        font=dict(size=12),
        showlegend=True,
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

    fig.show()

    if save_path:
        try:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            safe_experiment_name = sanitize_filename(experiment_name)
            html_filename = f"{safe_experiment_name}_all_wells_linearized_single_plot.html"
            html_path = save_path / html_filename
            print(f"Saving linearized single-plot HTML file to: {html_path}")
            fig.write_html(str(html_path))
            print(f"Linearized single-plot HTML file saved successfully: {html_path}")
        except Exception as exc:
            print(f"Error saving linearized single-plot HTML file: {exc}")

    return fig


def plot_wells_active_pixels_plotly(exp, save_path=None, experiment_name="Experiment", figsize=(1600, 600), ttp_results=None, show=True):
    """
    Create a single plotly plot showing active pixels for the entire chip as one heatmap.
    Reconstructs the full chip layout by combining all well active pixel masks.
    
    Parameters
    ----------
    exp : Experiment
        The experiment object containing wells
    save_path : str, optional
        Directory path to save the HTML plot. If None, plot is shown only.
    experiment_name : str
        Name of the experiment for plot title
    figsize : tuple
        Figure size (width, height) in pixels. Default (1600, 600) for rectangular aspect ratio.
    ttp_results : dict, optional
        Dictionary containing TTP (Time To Peak) results for each well. If provided, TTP information will be added to the plot title.
        
    Returns
    -------
    go.Figure
        The figure object containing the full chip active pixel map
    """
    
    n_wells = len(exp.wells_list)
    
    # Original chip dimensions
    nrows = 290
    ncols = 204
    
    # Initialize the full chip active pixel mask
    full_chip_mask = np.zeros((nrows, ncols), dtype=bool)
    
    # Reconstruct the full chip by placing each well's active pixels back to their original positions
    try:
        if n_wells == 1:
            # Single well - use the well's active mask directly
            well = exp.wells_list[0]
            active_mask_2d = well.idx_active.reshape(nrows, ncols, order='C')
            full_chip_mask = active_mask_2d
            
        elif n_wells == 2:
            # Two wells - top and bottom halves
            for well_idx, well in enumerate(exp.wells_list):
                active_mask_1d = well.idx_active
                well_nrows = well.well_nrows
                well_ncols = well.well_ncols
                active_mask_2d = active_mask_1d.reshape(well_nrows, well_ncols, order='C')
                
                if well_idx == 0:  # Top well
                    full_chip_mask[:well_nrows, :] = active_mask_2d
                else:  # Bottom well
                    full_chip_mask[well_nrows:, :] = active_mask_2d
                    
        elif n_wells == 4 or n_wells == 6 or n_wells == 10:
            # Multiple wells arranged in a grid
            wells_per_row = 2
            rows_per_well = nrows // (n_wells // wells_per_row)
            cols_per_well = ncols // wells_per_row
            
            for well_idx, well in enumerate(exp.wells_list):
                active_mask_1d = well.idx_active
                well_nrows = well.well_nrows
                well_ncols = well.well_ncols
                active_mask_2d = active_mask_1d.reshape(well_nrows, well_ncols, order='C')
                
                # Calculate position in the full chip
                row_start = (well_idx // wells_per_row) * rows_per_well
                col_start = (well_idx % wells_per_row) * cols_per_well
                row_end = row_start + well_nrows
                col_end = col_start + well_ncols
                
                # Place the well's active mask in the correct position
                full_chip_mask[row_start:row_end, col_start:col_end] = active_mask_2d
                
        else:
            raise ValueError(f"Unsupported number of wells: {n_wells}")
            
    except Exception as e:
        print(f"Error reconstructing full chip active pixels: {str(e)}")
        # Fallback: create a simple mask if reconstruction fails
        full_chip_mask = np.zeros((nrows, ncols), dtype=bool)
        for well_idx, well in enumerate(exp.wells_list):
            try:
                active_mask_1d = well.idx_active
                well_nrows = well.well_nrows
                well_ncols = well.well_ncols
                active_mask_2d = active_mask_1d.reshape(well_nrows, well_ncols, order='C')
                
                # Simple placement - just stack wells vertically
                start_row = well_idx * well_nrows
                end_row = min(start_row + well_nrows, nrows)
                actual_rows = end_row - start_row
                full_chip_mask[start_row:end_row, :well_ncols] = active_mask_2d[:actual_rows, :]
            except Exception as well_error:
                print(f"Error processing well {well_idx}: {str(well_error)}")
                continue
    
    # Create the plot title with TTP information if available
    title = f'{experiment_name} - Full Chip Active Pixels'
    if ttp_results is not None:
        ttp_wells = []
        for well_idx in range(n_wells):
            if well_idx in ttp_results and ttp_results[well_idx]['found_ttp']:
                ttp_time = ttp_results[well_idx]['ttp_time']
                ttp_wells.append(f'Well {well_idx + 1}: {ttp_time:.1f} min')
        if ttp_wells:
            title += f'<br>TTP Times: {", ".join(ttp_wells)}'
    
    # Create a single heatmap plot
    fig = go.Figure()
    
    # Add the full chip heatmap
    fig.add_trace(
        go.Heatmap(
            z=full_chip_mask.astype(int),
            colorscale='cividis',
            showscale=True,
            name='Active Pixels',
            colorbar=dict(
                title="Active Pixels",
                tickmode="array",
                tickvals=[0, 1],
                ticktext=["Inactive", "Active"]
            )
        )
    )
    
    # Update layout with rectangular aspect ratio
    fig.update_layout(
        title=title,
        xaxis_title="Column",
        yaxis_title="Row",
        height=figsize[1],
        width=figsize[0],
        font=dict(size=12),
        showlegend=False,
        # Ensure the plot fills the available space properly
        margin=dict(l=60, r=60, t=80, b=60),
        # Set aspect ratio to be more rectangular
        autosize=False
    )
    
    # Update axes with proper scaling for rectangular aspect ratio
    fig.update_xaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='lightgray',
        scaleanchor="y",  # Link x and y axis scaling
        scaleratio=1.0,   # Maintain 1:1 aspect ratio for the data
        constrain="domain"  # Constrain the axis to the plot domain
    )
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='lightgray',
        constrain="domain",  # Constrain the axis to the plot domain
        autorange="reversed"  # Ensure origin is top-left (row 0 at top)
    )
    
    # Show the plot
    if show:
        fig.show()
    
    # Save as HTML if save_path is provided
    if save_path:
        try:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Sanitize the experiment name for use in filename
            safe_experiment_name = sanitize_filename(experiment_name)
            html_filename = f"{safe_experiment_name}_active_pixels_full_chip.html"
            html_path = save_path / html_filename
            
            print(f"Saving full chip active pixels HTML file to: {html_path}")
            
            # Write the HTML file
            fig.write_html(str(html_path))
            print(f"Full chip active pixels HTML file saved successfully: {html_path}")
            
        except Exception as e:
            print(f"Error saving full chip active pixels HTML file: {str(e)}")
    
    return fig


def plot_raw_chip_video_plotly(
    exp,
    save_path=None,
    experiment_name="Experiment",
    figsize=(1000, 800),
    max_frames=200,
    from_settled=True,
    show=True,
    auto_open_html=False,
    zmin=0,
    zmax=1000,
):
    """
    Animated Plotly heatmap of raw chip output over time (play button + slider).
    Same chip tiling as ``plot_wells_active_pixels_plotly``; uses ``well_3d_npr`` per well.
    """
    from export_raw_chip_video import _reconstruct_full_chip_raw_3d

    full_chip_3d, time_vect = _reconstruct_full_chip_raw_3d(exp)
    if full_chip_3d is None:
        print("Warning: Could not get raw chip data for video; skipping.")
        return None

    _, _, n_time = full_chip_3d.shape
    if from_settled and hasattr(exp.wells_list[0], "idx_settled"):
        idx_settled = exp.wells_list[0].idx_settled
        if idx_settled < n_time:
            full_chip_3d = full_chip_3d[:, :, idx_settled:]
            time_vect = time_vect[idx_settled:]
            n_time = full_chip_3d.shape[2]

    if n_time <= 0:
        print("Warning: No time samples left for raw chip video after idx_settled; skipping.")
        return None

    if n_time > max_frames:
        step = max(1, n_time // max_frames)
        indices = np.arange(0, n_time, step)
        if indices[-1] != n_time - 1:
            indices = np.r_[indices, n_time - 1]
        full_chip_3d = full_chip_3d[:, :, indices]
        time_vect = time_vect[indices]
        n_time = len(indices)
    time_min = time_vect / 60.0

    frames = []
    for t in range(n_time):
        z = full_chip_3d[:, :, t]
        frames.append(
            go.Frame(
                data=[
                    go.Heatmap(
                        z=z,
                        colorscale="Turbo",
                        showscale=True,
                        colorbar=dict(title="Raw"),
                        zmin=zmin,
                        zmax=zmax,
                    )
                ],
                layout=go.Layout(title_text=f"{experiment_name} — t = {time_min[t]:.2f} min"),
                name=str(t),
            )
        )

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title=dict(text=f"{experiment_name} — Raw chip video (click Play)"),
            xaxis_title="Column",
            yaxis_title="Row",
            height=figsize[1],
            width=figsize[0],
            font=dict(size=12),
            margin=dict(l=60, r=80, t=80, b=60),
            yaxis=dict(autorange="reversed", scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain="domain"),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.1,
                    y=0,
                    xanchor="right",
                    yanchor="top",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    active=0,
                    x=0.1,
                    len=0.9,
                    y=0,
                    yanchor="top",
                    pad=dict(t=40, b=10),
                    currentvalue=dict(visible=True, prefix="Time: ", suffix=" min", xanchor="center"),
                    steps=[
                        dict(
                            args=[
                                [f.name],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            label=f"{time_min[i]:.2f}",
                            method="animate",
                        )
                        for i, f in enumerate(frames)
                    ],
                )
            ],
        ),
    )
    fig.layout.title.text = f"{experiment_name} — Raw chip video (t = {time_min[0]:.2f} min)"

    if show:
        fig.show()

    if save_path:
        try:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            safe_name = sanitize_filename(experiment_name)
            html_path = save_path / f"{safe_name}_raw_chip_video.html"
            print(f"Saving raw chip video HTML to: {html_path}")
            fig.write_html(str(html_path))
            print(f"Raw chip video saved: {html_path}")
            if auto_open_html:
                try:
                    import webbrowser

                    webbrowser.open(html_path.as_uri())
                    print(f"Opened raw chip video HTML in browser: {html_path}")
                except Exception as e:
                    print(f"Warning: Could not auto-open raw chip video HTML: {e}")
        except Exception as e:
            print(f"Error saving raw chip video HTML: {str(e)}")

    return fig


def plot_wells_first_derivative_grid_plotly(exp, well_data, save_path=None, experiment_name="Experiment", figsize=(1400, 1000), ttp_results=None, show=True):
    """
    Create a single plotly plot showing first derivatives for all wells in a 5x2 grid layout.
    Each well shows the first derivative with TTP markers if available.
    
    Parameters
    ----------
    exp : Experiment
        The experiment object containing wells
    well_data : dict
        Dictionary containing first derivative data for each well
    save_path : str, optional
        Directory path to save the HTML plot. If None, plot is shown only.
    experiment_name : str
        Name of the experiment for plot title
    figsize : tuple
        Figure size (width, height) in pixels
    ttp_results : dict, optional
        Dictionary containing TTP (Time To Peak) results for each well. If provided, TTP markers will be plotted.
        
    Returns
    -------
    go.Figure
        The figure object containing all wells' first derivatives in a grid layout
    """
    
    n_wells = len(exp.wells_list)
    
    # Create subplots: 5 rows, 2 columns (5x2 grid)
    fig = make_subplots(
        rows=5, cols=2,
        subplot_titles=[f'Well {i+1} - First Derivative' for i in range(n_wells)],
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Colors for different wells
    colors = COLORBLIND_WELL_COLORS
    
    for well_idx, well in enumerate(exp.wells_list):
        try:
            # Get the first derivative data for this well
            if well_idx not in well_data:
                print(f"Warning: No derivative data found for Well {well_idx + 1}")
                continue
                
            derivative_data = well_data[well_idx]['derivative']
            time_data = well_data[well_idx]['time']
            
            # Debug information
            print(f"DEBUG First Derivative Well {well_idx + 1}: Derivative length: {len(derivative_data)}, Time length: {len(time_data)}")
            print(f"DEBUG First Derivative Well {well_idx + 1}: Time range: {time_data[0]:.2f} to {time_data[-1]:.2f} min")
            print(f"DEBUG First Derivative Well {well_idx + 1}: Derivative range: {np.min(derivative_data):.6f} to {np.max(derivative_data):.6f}")
            
            # Calculate row and column position for this well
            row = (well_idx // 2) + 1
            col = (well_idx % 2) + 1
            
            color = colors[well_idx % len(colors)]
            
            # Plot the first derivative for this well
            fig.add_trace(
                go.Scatter(
                    x=time_data,
                    y=derivative_data,
                    mode='lines',
                    name=f'Well {well_idx + 1}',
                    line=dict(color=color, width=2),
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=row, col=col)
            
            # Add TTP marker if available
            if ttp_results is not None and well_idx in ttp_results and ttp_results[well_idx]['found_ttp']:
                ttp_time = ttp_results[well_idx]['ttp_time']
                # Find the closest time index for accurate plotting
                closest_idx = np.argmin(np.abs(time_data - ttp_time))
                ttp_value_at_time = derivative_data[closest_idx]
                
                # Plot the TTP marker
                fig.add_trace(
                    go.Scatter(
                        x=[ttp_time],
                        y=[ttp_value_at_time],
                        mode='markers',
                        name=f'Well {well_idx + 1} TTP',
                        marker=dict(
                            color=TTP_MARKER_COLOR,
                            size=12,
                            symbol='diamond',
                            line=dict(color=TTP_MARKER_BORDER_COLOR, width=2)
                        ),
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
                # Add TTP annotation
                fig.add_annotation(
                    x=ttp_time,
                    y=ttp_value_at_time,
                    text=f'TTP<br>{ttp_time:.1f} min',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=TTP_MARKER_COLOR,
                    ax=20,
                    ay=-30,
                    bgcolor=TTP_MARKER_COLOR,
                    bordercolor=TTP_MARKER_BORDER_COLOR,
                    borderwidth=2,
                    font=dict(color='white', size=10),
                    row=row, col=col
                )
            
        except Exception as e:
            print(f"Error processing Well {well_idx + 1} first derivative: {str(e)}")
            continue
    
    # Update layout
    fig.update_layout(
        title=f'{experiment_name} - First Derivatives (5x2 Grid)',
        height=figsize[1],
        width=figsize[0],
        font=dict(size=12),
        showlegend=False
    )
    
    # Update x-axis labels for bottom row only
    for col in [1, 2]:
        fig.update_xaxes(title_text="Time (minutes)", row=5, col=col)
    
    # Update y-axis labels for left column only
    for row in range(1, 6):
        fig.update_yaxes(title_text="First Derivative (a.u./min)", row=row, col=1)
    
    # Add grid to all subplots
    for row in range(1, 6):
        for col in range(1, 3):
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=row, col=col)
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=row, col=col)
    
    # Show the plot
    if show:
        fig.show()
    
    # Save as HTML if save_path is provided
    if save_path:
        try:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Sanitize the experiment name for use in filename
            safe_experiment_name = sanitize_filename(experiment_name)
            html_filename = f"{safe_experiment_name}_first_derivatives_grid.html"
            html_path = save_path / html_filename
            
            print(f"Saving first derivatives grid HTML file to: {html_path}")
            
            # Write the HTML file
            fig.write_html(str(html_path))
            print(f"First derivatives grid HTML file saved successfully: {html_path}")
            
        except Exception as e:
            print(f"Error saving first derivatives grid HTML file: {str(e)}")
    
    return fig


def plot_wells_second_derivative_grid_plotly(exp, second_deriv_data, save_path=None, experiment_name="Experiment", figsize=(1400, 1000), ttp_results=None, show=True):
    """
    Create a single plotly plot showing second derivatives for all wells in a 5x2 grid layout.
    Each well shows the second derivative with TTP markers if available.
    
    Parameters
    ----------
    exp : Experiment
        The experiment object containing wells
    second_deriv_data : dict
        Dictionary containing second derivative data for each well
    save_path : str, optional
        Directory path to save the HTML plot. If None, plot is shown only.
    experiment_name : str
        Name of the experiment for plot title
    figsize : tuple
        Figure size (width, height) in pixels
    ttp_results : dict, optional
        Dictionary containing TTP (Time To Peak) results for each well. If provided, TTP markers will be plotted.
        
    Returns
    -------
    go.Figure
        The figure object containing all wells' second derivatives in a grid layout
    """
    
    n_wells = len(exp.wells_list)
    
    # Create subplots: 5 rows, 2 columns (5x2 grid)
    fig = make_subplots(
        rows=5, cols=2,
        subplot_titles=[f'Well {i+1} - Second Derivative' for i in range(n_wells)],
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Colors for different wells
    colors = COLORBLIND_WELL_COLORS
    
    for well_idx, well in enumerate(exp.wells_list):
        try:
            # Get the second derivative data for this well
            if well_idx not in second_deriv_data:
                print(f"Warning: No second derivative data found for Well {well_idx + 1}")
                continue
                
            second_derivative_data = second_deriv_data[well_idx]['second_derivative']
            time_data = second_deriv_data[well_idx]['time']
            
            # Debug information
            print(f"DEBUG Second Derivative Well {well_idx + 1}: Second derivative length: {len(second_derivative_data)}, Time length: {len(time_data)}")
            print(f"DEBUG Second Derivative Well {well_idx + 1}: Time range: {time_data[0]:.2f} to {time_data[-1]:.2f} min")
            print(f"DEBUG Second Derivative Well {well_idx + 1}: Second derivative range: {np.min(second_derivative_data):.6f} to {np.max(second_derivative_data):.6f}")
            
            # Calculate row and column position for this well
            row = (well_idx // 2) + 1
            col = (well_idx % 2) + 1
            
            color = colors[well_idx % len(colors)]
            
            # Plot the second derivative for this well
            fig.add_trace(
                go.Scatter(
                    x=time_data,
                    y=second_derivative_data,
                    mode='lines',
                    name=f'Well {well_idx + 1}',
                    line=dict(color=color, width=2),
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=row, col=col)
            
            # Add TTP marker if available
            if ttp_results is not None and well_idx in ttp_results and ttp_results[well_idx]['found_ttp']:
                ttp_time = ttp_results[well_idx]['ttp_time']
                # Find the closest time index for accurate plotting
                closest_idx = np.argmin(np.abs(time_data - ttp_time))
                ttp_value_at_time = second_derivative_data[closest_idx]
                
                # Plot the TTP marker
                fig.add_trace(
                    go.Scatter(
                        x=[ttp_time],
                        y=[ttp_value_at_time],
                        mode='markers',
                        name=f'Well {well_idx + 1} TTP',
                        marker=dict(
                            color=TTP_MARKER_COLOR,
                            size=12,
                            symbol='diamond',
                            line=dict(color=TTP_MARKER_BORDER_COLOR, width=2)
                        ),
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
                # Add TTP annotation
                fig.add_annotation(
                    x=ttp_time,
                    y=ttp_value_at_time,
                    text=f'TTP<br>{ttp_time:.1f} min',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=TTP_MARKER_COLOR,
                    ax=20,
                    ay=-30,
                    bgcolor=TTP_MARKER_COLOR,
                    bordercolor=TTP_MARKER_BORDER_COLOR,
                    borderwidth=2,
                    font=dict(color='white', size=10),
                    row=row, col=col
                )
            
        except Exception as e:
            print(f"Error processing Well {well_idx + 1} second derivative: {str(e)}")
            continue
    
    # Update layout
    fig.update_layout(
        title=f'{experiment_name} - Second Derivatives (5x2 Grid)',
        height=figsize[1],
        width=figsize[0],
        font=dict(size=12),
        showlegend=False
    )
    
    # Update x-axis labels for bottom row only
    for col in [1, 2]:
        fig.update_xaxes(title_text="Time (minutes)", row=5, col=col)
    
    # Update y-axis labels for left column only
    for row in range(1, 6):
        fig.update_yaxes(title_text="Second Derivative (a.u./min²)", row=row, col=1)
    
    # Add grid to all subplots
    for row in range(1, 6):
        for col in range(1, 3):
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=row, col=col)
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=row, col=col)
    
    # Show the plot
    if show:
        fig.show()
    
    # Save as HTML if save_path is provided
    if save_path:
        try:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Sanitize the experiment name for use in filename
            safe_experiment_name = sanitize_filename(experiment_name)
            html_filename = f"{safe_experiment_name}_second_derivatives_grid.html"
            html_path = save_path / html_filename
            
            print(f"Saving second derivatives grid HTML file to: {html_path}")
            
            # Write the HTML file
            fig.write_html(str(html_path))
            print(f"Second derivatives grid HTML file saved successfully: {html_path}")
            
        except Exception as e:
            print(f"Error saving second derivatives grid HTML file: {str(e)}")
    
    return fig


def plot_wells_linearized_signal_combined_plotly(
    exp,
    save_path=None,
    experiment_name="Experiment",
    figsize=(1400, 600),
    ttp_results=None,
    wells_to_show=None,
    wells_to_hide=None,
    hide_mode="legendonly",
    scale_factor=1.0,
    y_offset=0.0,
    show=True,
):
    """
    Create a single plotly plot showing linearized signal (well_2d_bs_active_mean) for all wells together on one plot.
    All wells are overlaid on the same axes for easy comparison.
    
    Parameters
    ----------
    exp : Experiment
        The experiment object containing wells
    save_path : str, optional
        Directory path to save the HTML plot. If None, plot is shown only.
    experiment_name : str
        Name of the experiment for plot title
    figsize : tuple
        Figure size (width, height) in pixels
    ttp_results : dict, optional
        Dictionary containing TTP (Time To Peak) results for each well. If provided, TTP markers will be plotted.
    scale_factor : float, optional
        Multiplier applied to the linearized signal prior to plotting (default 1.0).
    y_offset : float, optional
        Constant value added to the (optionally scaled) linearized signal prior to plotting.
        
    Returns
    -------
    go.Figure
        The figure object containing all wells' linearized signals combined
    """
    
    n_wells = len(exp.wells_list)
    
    # Normalize well selection inputs to zero-based indices
    if wells_to_show is not None:
        wells_to_show = {int(idx) - 1 for idx in wells_to_show}
    if wells_to_hide is not None:
        wells_to_hide = {int(idx) - 1 for idx in wells_to_hide}
    if hide_mode not in {"legendonly", "skip"}:
        raise ValueError("hide_mode must be either 'legendonly' or 'skip'")
    
    # Create a single plot
    fig = go.Figure()
    
    # Colors for different wells
    colors = COLORBLIND_WELL_COLORS
    
    for well_idx, well in enumerate(exp.wells_list):
        try:
            # Lacewing-style baseline: start from pre-heat idx_start.
            if hasattr(well, 'well_2d_start_bs_active_mean') and well.well_2d_start_bs_active_mean is not None:
                signal_data = well.well_2d_start_bs_active_mean
            elif hasattr(well, 'well_2d_bs_active_mean') and well.well_2d_bs_active_mean is not None:
                signal_data = well.well_2d_bs_active_mean
            else:
                print(f"Warning: No linearized signal data found for Well {well_idx + 1}")
                continue
            
            time_data = _time_minutes_for_well(well, zero_at="start")
            time_data, signal_data = _align_x_y(time_data, signal_data)
            
            # Debug information for time indexing
            print(f"DEBUG Linearized Combined Well {well_idx + 1}: Signal length: {len(signal_data)}, Time length: {len(time_data)}")
            print(f"DEBUG Linearized Combined Well {well_idx + 1}: Time range: {time_data[0]:.2f} to {time_data[-1]:.2f} min")
            print(f"DEBUG Linearized Combined Well {well_idx + 1}: Linearized signal range: {np.min(signal_data):.4f} to {np.max(signal_data):.4f}")
            
            add_trace = True
            visible_state = True
            if wells_to_show is not None:
                if well_idx not in wells_to_show:
                    if hide_mode == "skip":
                        add_trace = False
                    else:
                        visible_state = "legendonly"
            elif wells_to_hide is not None and well_idx in wells_to_hide:
                if hide_mode == "skip":
                    add_trace = False
                else:
                    visible_state = "legendonly"
            
            if not add_trace:
                continue
            
            color = colors[well_idx % len(colors)]
            well_name = f'Well {well_idx + 1}'
            
            # Decide whether to rescale the signal for plotting
            y_data = signal_data * scale_factor if scale_factor is not None else signal_data
            if y_offset:
                y_data = y_data + y_offset

            # Plot the linearized signal for this well
            fig.add_trace(
                go.Scatter(
                    x=time_data,
                    y=y_data,
                    mode='lines',
                    name=well_name,
                    line=dict(color=color, width=2),
                    showlegend=True,
                    visible=visible_state,
                )
            )
            
            # Add TTP marker if available
            if ttp_results is not None and well_idx in ttp_results and ttp_results[well_idx]['found_ttp']:
                ttp_time = ttp_results[well_idx]['ttp_time']
                # Find the closest time index for accurate plotting
                closest_idx = np.argmin(np.abs(time_data - ttp_time))
                ttp_value_at_time = y_data[closest_idx]
                
                # Plot the TTP marker
                fig.add_trace(
                    go.Scatter(
                        x=[ttp_time],
                        y=[ttp_value_at_time],
                        mode='markers',
                        name=f'Well {well_idx + 1} TTP',
                        marker=dict(
                            color=TTP_MARKER_COLOR,
                            size=12,
                            symbol='diamond',
                            line=dict(color=TTP_MARKER_BORDER_COLOR, width=2)
                        ),
                        showlegend=False,
                    )
                )
            
        except Exception as e:
            print(f"Error processing Well {well_idx + 1} linearized signal for combined plot: {str(e)}")
            continue
    
    # Update layout
    y_axis_title = 'Linearized Signal (a.u.)'
    if scale_factor not in (None, 1.0):
        y_axis_title += f' * {scale_factor:g}'
    if y_offset:
        y_axis_title += f' + {y_offset:g}'

    fig.update_layout(
        title=f'{experiment_name} - All Wells Linearized Signal Combined',
        xaxis_title='Time (minutes)',
        yaxis_title=y_axis_title,
        height=figsize[1],
        width=figsize[0],
        font=dict(size=12),
        showlegend=True
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    # Show the plot
    if show:
        fig.show()
    
    # Save as HTML if save_path is provided
    if save_path:
        try:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Sanitize the experiment name for use in filename
            safe_experiment_name = sanitize_filename(experiment_name)
            html_filename = f"{safe_experiment_name}_linearized_signal_combined.html"
            html_path = save_path / html_filename
            
            print(f"Saving linearized signal combined HTML file to: {html_path}")
            
            # Write the HTML file
            fig.write_html(str(html_path))
            print(f"Linearized signal combined HTML file saved successfully: {html_path}")
            
        except Exception as e:
            print(f"Error saving linearized signal combined HTML file: {str(e)}")
    
    return fig


def plot_wells_linearized_signal_combined_savgol_plotly(
    exp,
    save_path=None,
    experiment_name="Experiment",
    figsize=(1400, 600),
    ttp_results=None,
    wells_to_show=None,
    wells_to_hide=None,
    hide_mode="legendonly",
    scale_factor=1.0,
    y_offset=0.0,
    show=True,
    savgol_window_length=31,
    savgol_polyorder=3,
):
    """
    Combined linearized-signal plot with Savitzky-Golay smoothing per well.
    """
    n_wells = len(exp.wells_list)

    if wells_to_show is not None:
        wells_to_show = {int(idx) - 1 for idx in wells_to_show}
    if wells_to_hide is not None:
        wells_to_hide = {int(idx) - 1 for idx in wells_to_hide}
    if hide_mode not in {"legendonly", "skip"}:
        raise ValueError("hide_mode must be either 'legendonly' or 'skip'")

    fig = go.Figure()
    colors = COLORBLIND_WELL_COLORS

    for well_idx, well in enumerate(exp.wells_list):
        try:
            if hasattr(well, 'well_2d_start_bs_active_mean') and well.well_2d_start_bs_active_mean is not None:
                signal_data = np.asarray(well.well_2d_start_bs_active_mean)
            elif hasattr(well, 'well_2d_bs_active_mean') and well.well_2d_bs_active_mean is not None:
                signal_data = np.asarray(well.well_2d_bs_active_mean)
            else:
                print(f"Warning: No linearized signal data found for Well {well_idx + 1}")
                continue

            time_data = _time_minutes_for_well(well, zero_at="start")
            time_data, signal_data = _align_x_y(time_data, signal_data)

            add_trace = True
            visible_state = True
            if wells_to_show is not None:
                if well_idx not in wells_to_show:
                    if hide_mode == "skip":
                        add_trace = False
                    else:
                        visible_state = "legendonly"
            elif wells_to_hide is not None and well_idx in wells_to_hide:
                if hide_mode == "skip":
                    add_trace = False
                else:
                    visible_state = "legendonly"
            if not add_trace:
                continue

            y_smooth = signal_data
            n_pts = len(signal_data)
            if n_pts >= 5:
                win = int(savgol_window_length)
                if win % 2 == 0:
                    win += 1
                win = min(win, n_pts if n_pts % 2 == 1 else n_pts - 1)
                if win >= 5 and win > int(savgol_polyorder):
                    y_smooth = savgol_filter(signal_data, window_length=win, polyorder=int(savgol_polyorder), mode="interp")

            y_data = y_smooth * scale_factor if scale_factor is not None else y_smooth
            if y_offset:
                y_data = y_data + y_offset

            color = colors[well_idx % len(colors)]
            well_name = f'Well {well_idx + 1}'
            fig.add_trace(
                go.Scatter(
                    x=time_data,
                    y=y_data,
                    mode='lines',
                    name=well_name,
                    line=dict(color=color, width=2),
                    showlegend=True,
                    visible=visible_state,
                )
            )

            if ttp_results is not None and well_idx in ttp_results and ttp_results[well_idx]['found_ttp']:
                ttp_time = ttp_results[well_idx]['ttp_time']
                closest_idx = np.argmin(np.abs(time_data - ttp_time))
                ttp_value_at_time = y_data[closest_idx]
                fig.add_trace(
                    go.Scatter(
                        x=[ttp_time],
                        y=[ttp_value_at_time],
                        mode='markers',
                        name=f'Well {well_idx + 1} TTP',
                        marker=dict(
                            color=TTP_MARKER_COLOR,
                            size=12,
                            symbol='diamond',
                            line=dict(color=TTP_MARKER_BORDER_COLOR, width=2)
                        ),
                        showlegend=False,
                    )
                )
        except Exception as e:
            print(f"Error processing Well {well_idx + 1} smoothed linearized signal: {str(e)}")
            continue

    y_axis_title = 'Linearized Signal (Savitzky-Golay) (a.u.)'
    if scale_factor not in (None, 1.0):
        y_axis_title += f' * {scale_factor:g}'
    if y_offset:
        y_axis_title += f' + {y_offset:g}'

    fig.update_layout(
        title=f'{experiment_name} - All Wells Linearized Signal Combined (Savitzky-Golay)',
        xaxis_title='Time (minutes)',
        yaxis_title=y_axis_title,
        height=figsize[1],
        width=figsize[0],
        font=dict(size=12),
        showlegend=True
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    if show:
        fig.show()

    if save_path:
        try:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            safe_experiment_name = sanitize_filename(experiment_name)
            html_filename = f"{safe_experiment_name}_linearized_signal_combined_savgol.html"
            html_path = save_path / html_filename
            print(f"Saving smoothed linearized signal combined HTML file to: {html_path}")
            fig.write_html(str(html_path))
            print(f"Smoothed linearized signal combined HTML file saved successfully: {html_path}")
        except Exception as e:
            print(f"Error saving smoothed linearized signal combined HTML file: {str(e)}")

    return fig


def plot_wells_temperature_plotly(exp, save_path=None, experiment_name="Experiment", figsize=(1600, 800), ttp_results=None, show=True):
    """
    Create a single plotly plot showing well_temp_mean_NEW data for all wells on one plot.
    Mimics the plotting style from plt_Experiment_summary.py with vertical lines for idx_start, idx_settled, and idx_end.
    
    Parameters
    ----------
    exp : Experiment
        The experiment object containing wells
    save_path : str, optional
        Directory path to save the HTML plot. If None, plot is shown only.
    experiment_name : str
        Name of the experiment for plot title
    figsize : tuple
        Figure size (width, height) in pixels
    ttp_results : dict, optional
        Dictionary containing TTP (Time To Peak) results for each well. If provided, TTP markers will be plotted.
        
    Returns
    -------
    go.Figure
        The figure object containing temperature data for all wells on a single plot
    """
    
    n_wells = len(exp.wells_list)
    print(f"DEBUG: Starting temperature plot for {n_wells} wells")
    
    # Create a single plot with secondary y-axis for dual units
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Colors for different wells
    colors = COLORBLIND_WELL_COLORS
    
    for well_idx, well in enumerate(exp.wells_list):
        try:
            # Get the temperature data (well_temp_mean_NEW) - matching plt_Experiment_summary.py
            temp_data = well.well_temp_mean_NEW
            
            # If well_temp_mean_NEW is empty (no active temp pixels > 10), try well_temp_lin2d as fallback
            if temp_data is None or len(temp_data) == 0:
                print(f"  - well_temp_mean_NEW is empty, trying well_temp_lin2d as fallback")
                temp_data = well.well_temp_lin2d
                if temp_data is not None and len(temp_data) > 0:
                    print(f"  - Using well_temp_lin2d as fallback")
                else:
                    print(f"  - well_temp_lin2d also empty, trying well_temp_mean_then_lin")
                    temp_data = well.well_temp_mean_then_lin
            
            # Get the full time data from time_npr (matching plt_Experiment_summary.py)
            time_data = well.time_npr
            
            # Debug information
            print(f"DEBUG Temperature Well {well_idx + 1}:")
            print(f"  - temp_data length: {len(temp_data) if temp_data is not None else 'None'}")
            print(f"  - time_data length: {len(time_data) if time_data is not None else 'None'}")
            print(f"  - temp_data type: {type(temp_data)}")
            print(f"  - time_data type: {type(time_data)}")
            
            if temp_data is None or time_data is None:
                print(f"  - ERROR: Missing data for Well {well_idx + 1}")
                continue
                
            if len(temp_data) == 0 or len(time_data) == 0:
                print(f"  - ERROR: Empty data for Well {well_idx + 1}")
                continue
            
            # Ensure time_data matches temp_data length
            if len(time_data) != len(temp_data):
                print(f"  - WARNING: Length mismatch for Well {well_idx + 1}. Truncating to shorter length.")
                min_len = min(len(time_data), len(temp_data))
                time_data = time_data[:min_len]
                temp_data = temp_data[:min_len]
            
            print(f"  - Final lengths: time={len(time_data)}, temp={len(temp_data)}")
            print(f"  - Time range: {time_data[0]:.2f} to {time_data[-1]:.2f}")
            print(f"  - Temp range: {np.min(temp_data):.4f} to {np.max(temp_data):.4f}")
            
            color = colors[well_idx % len(colors)]
            
            # Plot the temperature signal for this well (primary y-axis)
            fig.add_trace(
                go.Scatter(
                    x=time_data,
                    y=temp_data,
                    mode='lines',
                    name=f'Well {well_idx + 1}',
                    line=dict(color=color, width=2),
                    showlegend=True
                ),
                secondary_y=False
            )
            
            # Add vertical lines for idx_start, idx_settled, and idx_end (matching plt_Experiment_summary.py)
            # idx_start (black dashed)
            fig.add_vline(
                x=time_data[well.idx_start],
                line=dict(color='black', width=2, dash='dash'),
                annotation_text=f'Start',
                annotation_position='top'
            )
            
            # idx_settled (distinct dashed line)
            fig.add_vline(
                x=time_data[well.idx_settled],
                line=dict(color=SETTLED_LINE_COLOR, width=2, dash='dash'),
                annotation_text=f'Settled',
                annotation_position='top'
            )
            
            # idx_end (black dashed)
            fig.add_vline(
                x=time_data[well.idx_end],
                line=dict(color='black', width=2, dash='dash'),
                annotation_text=f'End',
                annotation_position='top'
            )
            
            # Add TTP marker if available
            if ttp_results is not None and well_idx in ttp_results and ttp_results[well_idx]['found_ttp']:
                ttp_time = ttp_results[well_idx]['ttp_time']
                # Convert TTP time to the same time scale as time_data
                ttp_time_abs = time_data[0] + ttp_time * 60  # Convert minutes to seconds
                
                # Find the closest time index for accurate plotting
                closest_idx = np.argmin(np.abs(time_data - ttp_time_abs))
                ttp_value_at_time = temp_data[closest_idx]
                
                # Plot the TTP marker (primary y-axis)
                fig.add_trace(
                    go.Scatter(
                        x=[ttp_time_abs],
                        y=[ttp_value_at_time],
                        mode='markers',
                        name=f'Well {well_idx + 1} TTP',
                        marker=dict(
                            color=TTP_MARKER_COLOR,
                            size=12,
                            symbol='diamond',
                            line=dict(color=TTP_MARKER_BORDER_COLOR, width=2)
                        ),
                        showlegend=True
                    ),
                    secondary_y=False
                )
            
        except Exception as e:
            print(f"Error processing Well {well_idx + 1} temperature: {str(e)}")
            continue
    
    # Load and plot all CSV temperature files from the specified folder
    try:
        import os
        import glob
        import pandas as pd
        from datetime import datetime, timedelta
        
        csv_folder = r"C:\Users\Lacewing-01\OneDrive - ProtonDx\Data\Temperature Experiment"
        print(f"DEBUG: Loading all CSV temperature files from {csv_folder}")
        
        # Find all CSV files in the folder
        csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))
        print(f"DEBUG: Found {len(csv_files)} CSV files in {csv_folder}")
        
        # Colors for different CSV files (colorblind-safe palette)
        csv_colors = CSV_COLORBLIND_COLORS if CSV_COLORBLIND_COLORS else COLORBLIND_WELL_COLORS
        
        csv_files_loaded = 0
        
        for csv_idx, csv_file_path in enumerate(csv_files):
            try:
                print(f"DEBUG: Loading CSV file {csv_idx + 1}: {os.path.basename(csv_file_path)}")
                
                # Load the CSV file
                df = pd.read_csv(csv_file_path)
                
                # Get the time and temperature columns
                time_col = df.columns[0]  # First column is time
                temp_col = df.columns[1]  # Second column is temperature (Channel 1 Last)
                
                # Convert time strings to datetime objects
                base_date = datetime(2024, 1, 1)
                times = []
                
                for time_str in df[time_col]:
                    time_str = str(time_str).strip('"')
                    try:
                        time_parts = time_str.split(':')
                        hours = int(time_parts[0])
                        minutes = int(time_parts[1])
                        seconds = int(time_parts[2])
                        
                        dt = base_date + timedelta(hours=hours, minutes=minutes, seconds=seconds)
                        times.append(dt)
                    except:
                        times.append(base_date + timedelta(seconds=len(times)))
                
                # Convert to seconds from start
                times_seconds = [(t - times[0]).total_seconds() for t in times]
                
                # Get temperature data
                temp_data_csv = df[temp_col].astype(float)
                
                print(f"DEBUG: CSV {csv_idx + 1} loaded - {len(times_seconds)} time points, {len(temp_data_csv)} temp points")
                print(f"DEBUG: CSV {csv_idx + 1} time range: {times_seconds[0]:.2f} to {times_seconds[-1]:.2f} seconds")
                print(f"DEBUG: CSV {csv_idx + 1} temp range: {temp_data_csv.min():.2f} to {temp_data_csv.max():.2f} °C")
                
                # Choose color for this CSV file
                color = csv_colors[csv_idx % len(csv_colors)]
                
                # Plot CSV data on secondary y-axis (Celsius)
                fig.add_trace(
                    go.Scatter(
                        x=times_seconds,
                        y=temp_data_csv,
                        mode='lines',
                        name=f'{os.path.basename(csv_file_path)} - {temp_col}',
                        line=dict(color=color, width=2, dash='dot'),
                        showlegend=True
                    ),
                    secondary_y=True
                )
                
                csv_files_loaded += 1
                
            except Exception as e:
                print(f"Error loading CSV file {csv_file_path}: {str(e)}")
                continue
        
        if csv_files_loaded == 0:
            print("WARNING: No CSV files could be loaded successfully")
            # Add a dummy trace if no CSV files loaded
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode='lines',
                    name='CSV Data Not Available',
                    line=dict(color=ALERT_COLOR, width=2, dash='dash'),
                    showlegend=True
                ),
                secondary_y=True
            )
        else:
            print(f"Successfully loaded {csv_files_loaded} CSV temperature files")
        
    except Exception as e:
        print(f"Error accessing CSV folder: {str(e)}")
        # Add a dummy trace if folder access fails
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='CSV Folder Not Accessible',
                line=dict(color=ALERT_COLOR, width=2, dash='dash'),
                showlegend=True
            ),
            secondary_y=True
        )
    
    # Check if any traces were added
    if len(fig.data) == 0:
        print("WARNING: No temperature data was plotted. Check if well_temp_mean_NEW data is available.")
        # Add a dummy trace to show something
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='No Data Available',
                line=dict(color=ALERT_COLOR, width=2, dash='dash'),
                showlegend=True
            )
        )
    
    print(f"Total traces added to temperature plot: {len(fig.data)}")
    
    # Update layout
    fig.update_layout(
        title=f'{experiment_name} - Temperature Data Overlay (Wells + All CSV Files)',
        xaxis_title='Time (seconds)',
        height=figsize[1],
        width=figsize[0],
        font=dict(size=12),
        showlegend=True,
        hovermode='x unified'
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text="Well Temperature (a.u.)", secondary_y=False)
    fig.update_yaxes(title_text="CSV Temperature (°C)", secondary_y=True)
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', secondary_y=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', secondary_y=True)
    
    # Show the plot
    if show:
        fig.show()
    
    # Save as HTML if save_path is provided
    if save_path:
        try:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Sanitize the experiment name for use in filename
            safe_experiment_name = sanitize_filename(experiment_name)
            html_filename = f"{safe_experiment_name}_temperature_overlay.html"
            html_path = save_path / html_filename
            
            print(f"Saving temperature overlay HTML file to: {html_path}")
            
            # Write the HTML file
            fig.write_html(str(html_path))
            print(f"Temperature overlay HTML file saved successfully: {html_path}")
            
        except Exception as e:
            print(f"Error saving temperature grid HTML file: {str(e)}")
    
    return fig


def plot_wells_temperature_mean_then_lin_plotly(
    exp,
    save_path=None,
    experiment_name="Experiment",
    figsize=(1600, 800),
    show=True,
    auto_open_html=False,
):
    """
    Plot `Well.well_temp_mean_then_lin` for all wells on a single Plotly figure.

    This matches `Well.well_temp_mean_then_lin` (avg temp pixels first, then log-linearize, baseline-subtract).
    """
    n_wells = len(exp.wells_list)
    if n_wells == 0:
        raise ValueError("Experiment has no wells")

    fig = go.Figure()
    colors = COLORBLIND_WELL_COLORS

    for well_idx, well in enumerate(exp.wells_list):
        try:
            y = well.well_temp_mean_then_lin
            t = np.asarray(getattr(well, "time_npr", None))
            if t is None or len(t) == 0 or y is None or len(y) == 0:
                continue
            n = min(len(t), len(y))
            t = t[:n]
            y = np.asarray(y)[:n]
            # Plot time in minutes from 0 for readability
            x = (t - t[0]) / 60.0

            color = colors[well_idx % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=f"Well {well_idx+1}",
                    line=dict(color=color, width=2),
                )
            )
        except Exception as e:
            print(f"Error plotting well_temp_mean_then_lin for Well {well_idx+1}: {e}")
            continue

    fig.update_layout(
        title=f"{experiment_name} - well_temp_mean_then_lin",
        height=figsize[1],
        width=figsize[0],
        font=dict(size=12),
        xaxis_title="Time (minutes)",
        yaxis_title="Temp (mean-then-lin, baseline-subtracted)",
        showlegend=True,
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

    if show:
        fig.show()

    if save_path:
        try:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            safe_experiment_name = sanitize_filename(experiment_name)
            html_filename = f"{safe_experiment_name}_well_temp_mean_then_lin.html"
            html_path = save_path / html_filename
            print(f"Saving mean-then-lin temperature HTML file to: {html_path}")
            fig.write_html(str(html_path))
            print(f"Mean-then-lin temperature HTML file saved successfully: {html_path}")
            if auto_open_html:
                try:
                    import webbrowser
                    webbrowser.open(html_path.as_uri())
                except Exception as e:
                    print(f"Warning: Could not auto-open mean-then-lin temperature HTML: {e}")
        except Exception as e:
            print(f"Error saving mean-then-lin temperature HTML file: {e}")

    return fig


def plot_wells_temperature_mean_then_lin_global_plotly(
    exp,
    save_path=None,
    experiment_name="Experiment",
    figsize=(1600, 800),
    show=True,
    auto_open_html=False,
    save_excel=True,
    scale_factor=1.0,
    y_offset=0.0,
):
    """
    Plot a **single** trace: the average across all wells of `Well.well_temp_mean_then_lin`,
    aligned on a shared time axis, and export to Excel with time.

    - X axis: time in minutes from 0
    - Y axis: mean across wells at each timepoint (NaN-safe)
    - Optional display scaling via `scale_factor` and `y_offset`
    """
    import time as _time_mod
    import pandas as pd

    n_wells = len(exp.wells_list)
    if n_wells == 0:
        raise ValueError("Experiment has no wells")

    # ------------------------------------------------------------------
    # v06 path: prefer the merged "heating ramp + readout phase" trace built by
    # `titan_load_and_preprocessing` (using temp_log.bin + readout meta tail).
    # Falls back to the legacy per-well property if the merged trace isn't present.
    # ------------------------------------------------------------------
    merged_t_s = getattr(exp, "temperature_time_s_v06_with_ramp", None)
    merged_y = getattr(exp, "temperature_1d_v06_with_ramp", None)
    heat_n = int(getattr(exp, "temperature_v06_heat_n", 0) or 0)

    if merged_t_s is not None and merged_y is not None and len(merged_t_s) == len(merged_y) and len(merged_y) > 0:
        t0_s = np.asarray(merged_t_s, dtype=float)
        y_full = np.asarray(merged_y, dtype=float)

        # Baseline-subtract relative to the first finite sample so the y axis matches
        # the legacy `well_temp_mean_then_lin` convention.
        finite = y_full[np.isfinite(y_full)]
        baseline = float(finite[0]) if finite.size else 0.0
        y_mean = y_full - baseline

        x_min = t0_s / 60.0
        y_plot = y_mean * float(scale_factor)
        if y_offset:
            y_plot = y_plot + float(y_offset)

        # `temp_log.bin` already contains the full experiment temperature trace
        # (heating phase + post-settling readout phase) on a single time base, so we
        # render it as one continuous line.
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x_min,
                y=y_plot,
                mode="lines",
                name=f"Temperature (n={len(x_min)})",
                line=dict(color="firebrick", width=3),
            )
        )

        # Provide enough downstream variables for the existing save/Excel block.
        n_use = int(len(x_min))
        t0 = t0_s
    else:
        # Build aligned (truncate-to-shortest) time axis and stack series
        series = []
        time_secs = []
        for well in exp.wells_list:
            t = np.asarray(getattr(well, "time_npr", None))
            y = getattr(well, "well_temp_mean_then_lin", None)
            if t is None or y is None:
                continue
            t = np.asarray(t)
            y = np.asarray(y)
            if t.size == 0 or y.size == 0:
                continue
            n = int(min(len(t), len(y)))
            time_secs.append(t[:n])
            series.append(y[:n].astype(float, copy=False))

        if not series:
            raise ValueError("No wells had well_temp_mean_then_lin data")

        # Truncate all to shortest length and use the first well's time vector
        n_use = int(min(len(s) for s in series))
        t0 = time_secs[0][:n_use]
        x_min = (t0 - t0[0]) / 60.0
        y_stack = np.vstack([s[:n_use] for s in series])  # (n_wells_used, n_time)
        y_mean = np.nanmean(y_stack, axis=0)
        y_plot = y_mean * float(scale_factor)
        if y_offset:
            y_plot = y_plot + float(y_offset)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x_min,
                y=y_plot,
                mode="lines",
                name="Mean across wells",
                line=dict(color="black", width=3),
            )
        )

    y_axis_title = "Temp (mean-then-lin, baseline-subtracted)"
    if scale_factor not in (None, 1.0):
        y_axis_title += f" * {float(scale_factor):g}"
    if y_offset:
        y_axis_title += f" + {float(y_offset):g}"

    title_suffix = " (full experiment)" if merged_t_s is not None else ""
    fig.update_layout(
        title=f"{experiment_name} - mean(well_temp_mean_then_lin) across wells{title_suffix}",
        height=figsize[1],
        width=figsize[0],
        font=dict(size=12),
        xaxis_title="Time (minutes)",
        yaxis_title=y_axis_title,
        showlegend=True,
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

    if show:
        fig.show()

    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        safe_experiment_name = sanitize_filename(experiment_name)

        # Save HTML
        html_path = save_path / f"{safe_experiment_name}_well_temp_mean_then_lin_global_mean.html"
        try:
            print(f"Saving global mean-then-lin temperature HTML file to: {html_path}")
            fig.write_html(str(html_path))
            print(f"Global mean-then-lin temperature HTML file saved successfully: {html_path}")
            if auto_open_html:
                try:
                    import webbrowser
                    webbrowser.open(html_path.as_uri())
                except Exception as e:
                    print(f"Warning: Could not auto-open global mean-then-lin temperature HTML: {e}")
        except Exception as e:
            print(f"Error saving global mean-then-lin temperature HTML file: {e}")

        # Save Excel
        if save_excel:
            xlsx_base = save_path / f"{safe_experiment_name}_well_temp_mean_then_lin_global_mean.xlsx"
            df_data = {
                "time_s": (t0[:n_use] - t0[0]),
                "time_min": x_min,
                "temp_mean_then_lin_global_mean": y_plot,
            }
            df = pd.DataFrame(df_data)

            def _candidate_paths(base: Path):
                yield base
                ts = int(_time_mod.time())
                yield base.with_name(base.stem + f"_{ts}" + base.suffix)
                for i in range(1, 10):
                    yield base.with_name(base.stem + f"_{i}" + base.suffix)

            last_exc = None
            for p_try in _candidate_paths(xlsx_base):
                try:
                    with pd.ExcelWriter(str(p_try), engine="openpyxl") as writer:
                        df.to_excel(writer, sheet_name="global_mean", index=False)
                    print(f"Saved global mean-then-lin temperature Excel file to: {p_try}")
                    break
                except Exception as e:
                    last_exc = e
                    continue
            else:
                print(f"Error saving global mean-then-lin temperature Excel file: {last_exc}")

    return fig


def plot_wells_raw_plotly(
    exp,
    save_path=None,
    experiment_name="Experiment",
    figsize=(1600, 1200),
    show=True,
    auto_open_html=False,
    save_excel=True,
):
    """
    Plot raw chem data per well (time_npr vs well_2d_npr) in a grid using Plotly.
    Mirrors the matplotlib view in plt_Experiment_summary.py (lines 29-34):
      - all pixel traces per well (light)
      - mean trace (bold)
      - vertical lines at idx_start, idx_settled, idx_end
    """
    #region agent log
    import json as _agent_json
    import time as _agent_time
    import traceback as _agent_traceback
    from pathlib import Path as _agent_Path
    def _agent_log(hypothesisId, location, message, data=None, runId="pre-fix"):
        try:
            payload = {
                "sessionId": "debug-session",
                "runId": runId,
                "hypothesisId": hypothesisId,
                "location": location,
                "message": message,
                "data": data or {},
                "timestamp": int(_agent_time.time() * 1000),
            }
            with open(r"c:\Users\Matthew L\Cursor Projects\titan-processing-costanza\.cursor\debug.log", "a", encoding="utf-8") as f:
                f.write(_agent_json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass
    def _agent_log_visible(payload):
        try:
            candidates = []
            try:
                repo_root = _agent_Path(__file__).resolve().parents[1]
                candidates.append(repo_root / "debug_visible.ndjson")
                candidates.append(repo_root / "titan" / "debug_visible.ndjson")
            except Exception:
                candidates.append(_agent_Path("debug_visible.ndjson"))
                candidates.append(_agent_Path("titan") / "debug_visible.ndjson")

            last_exc = None
            for p in candidates:
                try:
                    with open(str(p), "a", encoding="utf-8") as f:
                        f.write(_agent_json.dumps(payload, ensure_ascii=False) + "\n")
                    return
                except Exception as e:
                    last_exc = e
                    continue
            try:
                print(f"[agent-debug] Could not write debug_visible log: {last_exc}", file=__import__("sys").stderr)
            except Exception:
                pass
        except Exception:
            pass
    _agent_log(
        "C",
        "titan/plot_derivatives_plotly.py:1959",
        "Entered plot_wells_raw_plotly",
        {
            "exp_type": str(type(exp)),
            "has_wells_list": hasattr(exp, "wells_list"),
            "n_wells_list": (len(exp.wells_list) if hasattr(exp, "wells_list") and exp.wells_list is not None else None),
            "save_path": (str(save_path) if save_path is not None else None),
            "experiment_name": str(experiment_name),
        },
        runId="pre-fix",
    )
    _agent_log_visible({
        "sessionId": "debug-session",
        "runId": "pre-fix",
        "hypothesisId": "C",
        "location": "titan/plot_derivatives_plotly.py:1959",
        "message": "Entered plot_wells_raw_plotly",
        "data": {
            "exp_type": str(type(exp)),
            "has_wells_list": hasattr(exp, "wells_list"),
            "n_wells_list": (len(exp.wells_list) if hasattr(exp, "wells_list") and exp.wells_list is not None else None),
            "save_path": (str(save_path) if save_path is not None else None),
            "experiment_name": str(experiment_name),
        },
        "timestamp": int(_agent_time.time() * 1000),
    })
    #endregion agent log
    n_wells = len(exp.wells_list)
    if n_wells == 0:
        raise ValueError("Experiment has no wells")

    rows = int(math.ceil(n_wells / 2))
    cols = 2 if n_wells > 1 else 1

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"Well {i+1}" for i in range(n_wells)],
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )

    colors = COLORBLIND_WELL_COLORS

    # Simple counters to verify whether we actually plotted data vs skipped everything
    _agent_plotted_wells = 0
    _agent_skipped_missing = 0
    _agent_skipped_mismatch = 0
    _agent_exc_wells = 0

    # Store the plotted raw-chem traces so we can export them to Excel.
    raw_chem_export = {}

    for well_idx, well in enumerate(exp.wells_list):
        try:
            time = getattr(well, "time_npr", None)
            data_2d = getattr(well, "well_2d_npr", None)
            if time is None or data_2d is None:
                print(f"Warning: Well {well_idx + 1} missing time or raw data")
                _agent_skipped_missing += 1
                continue

            time = np.asarray(time)
            data_2d = np.asarray(data_2d)

            # data_2d expected shape: (n_time, n_pixels); if not, try transpose
            if data_2d.shape[0] != len(time) and data_2d.shape[1] == len(time):
                data_2d = data_2d.T

            if data_2d.shape[0] != len(time):
                print(
                    f"Warning: Well {well_idx + 1} time/data length mismatch "
                    f"({len(time)} vs {data_2d.shape[0]}); skipping"
                )
                _agent_skipped_mismatch += 1
                continue

            # Slice from idx_settled onwards
            idx_settled = getattr(well, 'idx_settled', 0)
            if idx_settled >= len(time):
                print(f"Warning: Well {well_idx + 1} idx_settled ({idx_settled}) >= time length ({len(time)}); using full range")
                idx_settled = 0
            
            time = time[idx_settled:]
            data_2d = data_2d[idx_settled:, :]

            # Downsample to keep plots responsive (aim ~1200 points)
            max_points = 1200
            if len(time) > max_points:
                step = max(1, int(math.ceil(len(time) / max_points)))
                time_ds = time[::step]
                data_2d_ds = data_2d[::step, :]
            else:
                time_ds = time
                data_2d_ds = data_2d

            row = (well_idx // 2) + 1
            col = (well_idx % 2) + 1

            # Plot individual pixel traces
            # Use Scattergl for better performance with many traces
            well_color = colors[well_idx % len(colors)]
            pixel_color = _hex_to_rgba(well_color, 0.1)  # Very transparent for individual pixels
            
            # Plot each pixel trace (use Scattergl for performance with many traces)
            n_pixels = data_2d_ds.shape[1]
            # Limit to reasonable number of pixels to avoid browser crash
            max_pixels_to_plot = 2000
            if n_pixels > max_pixels_to_plot:
                # Sample pixels evenly
                pixel_indices = np.linspace(0, n_pixels - 1, max_pixels_to_plot, dtype=int)
                print(f"Warning: Well {well_idx + 1} has {n_pixels} pixels, plotting {max_pixels_to_plot} sampled pixels")
            else:
                pixel_indices = np.arange(n_pixels)
            
            for pix_idx in pixel_indices:
                show_in_legend = bool(pix_idx < 5)  # Convert to Python bool
                fig.add_trace(
                    go.Scattergl(  # Use Scattergl for better performance
                        x=time_ds,
                        y=data_2d_ds[:, pix_idx],
                        mode="lines",
                        line=dict(color=pixel_color, width=1),
                        name=f"Well {well_idx + 1} pixel {pix_idx}" if show_in_legend else None,  # Only show first few in legend
                        showlegend=show_in_legend,  # Only show first few in legend to avoid clutter
                        legendgroup=f"well_{well_idx + 1}",
                    ),
                    row=row,
                    col=col,
                )

            # Vertical markers (adjusted for idx_settled offset)
            # Get original time_npr for marker positions
            time_full = np.asarray(well.time_npr)
            idx_start = getattr(well, 'idx_start', 0)
            idx_settled = getattr(well, 'idx_settled', 0)
            idx_end = getattr(well, 'idx_end', len(time_full) - 1)

            # Keep exactly what is displayed in the plot (downsampled + sampled pixels).
            export_cols = {"time_s": time_ds}
            for pix_idx in pixel_indices:
                export_cols[f"pixel_{int(pix_idx)}"] = data_2d_ds[:, int(pix_idx)]
            export_cols["mean_signal"] = np.nanmean(data_2d_ds, axis=1)
            raw_chem_export[f"well_{well_idx + 1}"] = {
                "df": export_cols,
                "idx_start": idx_start,
                "idx_settled": idx_settled,
                "idx_end": idx_end,
            }
            
            # Start marker (only if idx_start < idx_settled, otherwise it's before our data)
            if idx_start < idx_settled:
                fig.add_vline(
                    x=time_full[idx_start],
                    line=dict(color="black", width=2, dash="dash"),
                    annotation_text="Start",
                    annotation_position="top",
                    row=row,
                    col=col,
                )
            
            # Settled marker is at the start of our data (time[0])
            fig.add_vline(
                x=time_ds[0],
                line=dict(color=SETTLED_LINE_COLOR, width=2, dash="dash"),
                annotation_text="Settled",
                annotation_position="top",
                row=row,
                col=col,
            )
            
            # End marker (adjusted for idx_settled offset)
            if idx_end >= idx_settled:
                end_idx_in_sliced = idx_end - idx_settled
                if end_idx_in_sliced < len(time_ds):
                    fig.add_vline(
                        x=time_ds[end_idx_in_sliced],
                        line=dict(color="black", width=2, dash="dash"),
                        annotation_text="End",
                        annotation_position="top",
                        row=row,
                        col=col,
                    )

            print(f"DEBUG Raw Plot Well {well_idx + 1}: time len={len(time)}, data shape={data_2d.shape}")
            _agent_plotted_wells += 1

        except Exception as exc:
            print(f"Error plotting raw data for Well {well_idx + 1}: {exc}")
            _agent_exc_wells += 1
            continue

    fig.update_layout(
        title=f"{experiment_name} - Raw Chem Data (time_npr vs well_2d_npr)",
        height=max(figsize[1], rows * 240),
        width=figsize[0],
        font=dict(size=12),
        showlegend=True,
    )

    # Axes labels
    for r in range(1, rows + 1):
        fig.update_yaxes(title_text="Signal (a.u.)", row=r, col=1)
    for c in range(1, cols + 1):
        fig.update_xaxes(title_text="Time (s)", row=rows, col=c)

    # Grid
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray", row=r, col=c)
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray", row=r, col=c)

    #region agent log
    _agent_log("D", "titan/plot_derivatives_plotly.py:2119", "About to call fig.show()", {}, runId="pre-fix")
    _agent_log_visible({
        "sessionId": "debug-session",
        "runId": "pre-fix",
        "hypothesisId": "D",
        "location": "titan/plot_derivatives_plotly.py:2119",
        "message": "About to call fig.show()",
        "data": {},
        "timestamp": int(_agent_time.time() * 1000),
    })
    #endregion agent log
    if show:
        try:
            fig.show()
        except Exception as e:
            # Don't fail the pipeline if the interactive renderer can't open (common in script/headless runs)
            print(f"Warning: fig.show() failed for raw chem plot: {e}")
    #region agent log
    _agent_log("D", "titan/plot_derivatives_plotly.py:2119", "fig.show() returned", {}, runId="pre-fix")
    _agent_log_visible({
        "sessionId": "debug-session",
        "runId": "pre-fix",
        "hypothesisId": "D",
        "location": "titan/plot_derivatives_plotly.py:2119",
        "message": "fig.show() returned",
        "data": {},
        "timestamp": int(_agent_time.time() * 1000),
    })
    #endregion agent log

    if save_path:
        try:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            safe_experiment_name = sanitize_filename(experiment_name)
            html_filename = f"{safe_experiment_name}_raw_chem_plotly.html"
            html_path = save_path / html_filename
            print(f"Saving raw chem HTML file to: {html_path}")
            #region agent log
            _agent_log(
                "E",
                "titan/plot_derivatives_plotly.py:2121-2130",
                "About to write_html for raw chem plot",
                {"html_path": str(html_path), "safe_experiment_name": str(safe_experiment_name)},
                runId="pre-fix",
            )
            _agent_log_visible({
                "sessionId": "debug-session",
                "runId": "pre-fix",
                "hypothesisId": "E",
                "location": "titan/plot_derivatives_plotly.py:2121-2130",
                "message": "About to write_html for raw chem plot",
                "data": {"html_path": str(html_path), "safe_experiment_name": str(safe_experiment_name)},
                "timestamp": int(_agent_time.time() * 1000),
            })
            #endregion agent log
            fig.write_html(str(html_path))
            print(f"Raw chem HTML file saved successfully: {html_path}")
            #region agent log
            _agent_log("E", "titan/plot_derivatives_plotly.py:2121-2130", "write_html succeeded for raw chem plot", {"html_path": str(html_path)}, runId="pre-fix")
            _agent_log_visible({
                "sessionId": "debug-session",
                "runId": "pre-fix",
                "hypothesisId": "E",
                "location": "titan/plot_derivatives_plotly.py:2121-2130",
                "message": "write_html succeeded for raw chem plot",
                "data": {"html_path": str(html_path)},
                "timestamp": int(_agent_time.time() * 1000),
            })
            #endregion agent log
            if auto_open_html:
                try:
                    import webbrowser
                    #region agent log
                    _agent_log("F", "titan/plot_derivatives_plotly.py:auto_open_html", "Attempting to open raw chem HTML in browser", {"html_path": str(html_path)}, runId="pre-fix")
                    _agent_log_visible({
                        "sessionId": "debug-session",
                        "runId": "pre-fix",
                        "hypothesisId": "F",
                        "location": "titan/plot_derivatives_plotly.py:auto_open_html",
                        "message": "Attempting to open raw chem HTML in browser",
                        "data": {"html_path": str(html_path)},
                        "timestamp": int(_agent_time.time() * 1000),
                    })
                    #endregion agent log
                    webbrowser.open(html_path.as_uri())
                    print(f"Opened raw chem HTML in browser: {html_path}")
                except Exception as e:
                    print(f"Warning: Could not auto-open raw chem HTML: {e}")
                    #region agent log
                    _agent_log("F", "titan/plot_derivatives_plotly.py:auto_open_html", "Failed to open raw chem HTML in browser", {"exc_type": str(type(e)), "exc_str": str(e), "traceback": _agent_traceback.format_exc()}, runId="pre-fix")
                    _agent_log_visible({
                        "sessionId": "debug-session",
                        "runId": "pre-fix",
                        "hypothesisId": "F",
                        "location": "titan/plot_derivatives_plotly.py:auto_open_html",
                        "message": "Failed to open raw chem HTML in browser",
                        "data": {"exc_type": str(type(e)), "exc_str": str(e), "traceback": _agent_traceback.format_exc()},
                        "timestamp": int(_agent_time.time() * 1000),
                    })
                    #endregion agent log
            if save_excel:
                try:
                    import pandas as pd

                    xlsx_filename = f"{safe_experiment_name}_raw_chem_plotly.xlsx"
                    xlsx_path = save_path / xlsx_filename
                    with pd.ExcelWriter(str(xlsx_path), engine="openpyxl") as writer:
                        # Metadata sheet for quick context.
                        meta_rows = []
                        for sheet_name, payload in raw_chem_export.items():
                            meta_rows.append(
                                {
                                    "sheet": sheet_name,
                                    "idx_start": payload.get("idx_start"),
                                    "idx_settled": payload.get("idx_settled"),
                                    "idx_end": payload.get("idx_end"),
                                }
                            )
                        if meta_rows:
                            pd.DataFrame(meta_rows).to_excel(writer, sheet_name="metadata", index=False)

                        for sheet_name, payload in raw_chem_export.items():
                            df_sheet = pd.DataFrame(payload["df"])
                            # Excel sheet names have a 31-char limit.
                            safe_sheet_name = sheet_name[:31]
                            df_sheet.to_excel(writer, sheet_name=safe_sheet_name, index=False)
                    print(f"Raw chem Excel file saved successfully: {xlsx_path}")
                except Exception as e:
                    print(f"Warning: Could not save raw chem Excel file: {e}")
        except Exception as e:
            print(f"Error saving raw chem HTML file: {str(e)}")
            #region agent log
            _agent_log(
                "E",
                "titan/plot_derivatives_plotly.py:2131-2132",
                "Exception while saving raw chem HTML",
                {"exc_type": str(type(e)), "exc_str": str(e), "traceback": _agent_traceback.format_exc()},
                runId="pre-fix",
            )
            _agent_log_visible({
                "sessionId": "debug-session",
                "runId": "pre-fix",
                "hypothesisId": "E",
                "location": "titan/plot_derivatives_plotly.py:2131-2132",
                "message": "Exception while saving raw chem HTML",
                "data": {"exc_type": str(type(e)), "exc_str": str(e), "traceback": _agent_traceback.format_exc()},
                "timestamp": int(_agent_time.time() * 1000),
            })
            #endregion agent log

    #region agent log
    _agent_log(
        "G",
        "titan/plot_derivatives_plotly.py:summary",
        "Raw chem plot summary counters",
        {
            "n_wells": int(n_wells),
            "plotted_wells": int(_agent_plotted_wells),
            "skipped_missing_time_or_data": int(_agent_skipped_missing),
            "skipped_time_data_mismatch": int(_agent_skipped_mismatch),
            "exception_wells": int(_agent_exc_wells),
        },
        runId="pre-fix",
    )
    _agent_log_visible({
        "sessionId": "debug-session",
        "runId": "pre-fix",
        "hypothesisId": "G",
        "location": "titan/plot_derivatives_plotly.py:summary",
        "message": "Raw chem plot summary counters",
        "data": {
            "n_wells": int(n_wells),
            "plotted_wells": int(_agent_plotted_wells),
            "skipped_missing_time_or_data": int(_agent_skipped_missing),
            "skipped_time_data_mismatch": int(_agent_skipped_mismatch),
            "exception_wells": int(_agent_exc_wells),
        },
        "timestamp": int(_agent_time.time() * 1000),
    })
    #endregion agent log

    return fig

def plot_gain_3d_plotly(arr_gain_3d, save_path=None, experiment_name="Experiment", figsize=(1400, 700), vref=None):
    """
    Plot the full chip gain data (arr_gain_3d) using Plotly.
    Shows all gain frames on a single plot with all pixel values.
    
    Parameters
    ----------
    arr_gain_3d : np.ndarray
        3D array of shape (nrows, ncols, n_frames) containing gain data
    save_path : str, optional
        Directory path to save the HTML plot. If None, plot is shown only.
    experiment_name : str
        Name of the experiment for plot title
    figsize : tuple
        Figure size (width, height) in pixels
    vref : float, optional
        Reference voltage value to include in title
        
    Returns
    -------
    go.Figure
        The figure object containing gain data visualizations
    """
    from pathlib import Path
    
    if arr_gain_3d is None:
        raise ValueError("arr_gain_3d is None")
    
    if len(arr_gain_3d.shape) != 3:
        raise ValueError(f"arr_gain_3d must be 3D, got shape: {arr_gain_3d.shape}")
    
    n_frames = arr_gain_3d.shape[2]
    nrows = arr_gain_3d.shape[0]
    ncols = arr_gain_3d.shape[1]
    
    print(f"Plotting gain data: shape={arr_gain_3d.shape}, n_frames={n_frames}")
    
    # Reshape gain data: (n_pixels, n_frames)
    gain_2d = arr_gain_3d.reshape(-1, n_frames)
    n_pixels = gain_2d.shape[0]
    
    # Create a single plot
    fig = go.Figure()
    
    # Frame indices for x-axis
    frame_indices = np.arange(n_frames)
    
    # Plot all pixel traces for all frames
    # Use Scattergl for better performance with many traces
    # Limit number of pixels to plot for performance (sample evenly)
    max_pixels_to_plot = min(2000, n_pixels)
    if n_pixels > max_pixels_to_plot:
        pixel_indices = np.linspace(0, n_pixels - 1, max_pixels_to_plot, dtype=int)
        print(f"Plotting {max_pixels_to_plot} sampled pixels out of {n_pixels} total")
    else:
        pixel_indices = np.arange(n_pixels)
    
    # Use colorblind-friendly colors for different frames
    colors = COLORBLIND_WELL_COLORS
    
    for frame_idx in range(n_frames):
        frame_color = colors[frame_idx % len(colors)]
        pixel_color = _hex_to_rgba(frame_color, 0.1)  # Very transparent for individual pixels
        
        # Plot each pixel trace for this frame
        for pix_idx in pixel_indices:
            fig.add_trace(
                go.Scattergl(
                    x=frame_indices,
                    y=gain_2d[pix_idx, :],
                    mode='lines',
                    line=dict(color=pixel_color, width=1),
                    name=f'Frame {frame_idx} pixel {pix_idx}' if pix_idx < 5 and frame_idx == 0 else None,
                    showlegend=bool(pix_idx < 5 and frame_idx == 0),  # Only show first few in legend
                    legendgroup=f"frame_{frame_idx}",
                )
            )
        
        # Add mean trace for this frame (more visible)
        mean_trace = np.nanmean(gain_2d[:, :], axis=0)  # Mean across all pixels for each frame
        fig.add_trace(
            go.Scatter(
                x=frame_indices,
                y=mean_trace,
                mode='lines+markers',
                name=f'Frame {frame_idx} mean',
                line=dict(color=frame_color, width=3),
                marker=dict(size=8, color=frame_color),
                showlegend=True,
            )
        )
    
    # Update layout
    title = f'{experiment_name} - Full Chip Gain Data'
    if vref is not None:
        title += f' (vref: {vref:.2f})'
    
    fig.update_layout(
        title=title,
        height=figsize[1],
        width=figsize[0],
        font=dict(size=12),
        xaxis_title="Gain Frame",
        yaxis_title="Gain Value (a.u.)",
        showlegend=True,
        hovermode='x unified'
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', range=[0, 2000])
    
    # Show the plot
    fig.show()
    
    # Save as HTML if save_path is provided
    if save_path:
        try:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            safe_experiment_name = sanitize_filename(experiment_name)
            html_filename = f"{safe_experiment_name}_gain_3d_full_chip.html"
            html_path = save_path / html_filename
            
            print(f"Saving gain 3D HTML file to: {html_path}")
            fig.write_html(str(html_path))
            print(f"Gain 3D HTML file saved successfully: {html_path}")
            
        except Exception as e:
            print(f"Error saving gain 3D HTML file: {str(e)}")
    
    return fig


# ----- dVref multi-payload: selection model (TTN_SweepSearch_dVref_Multi, L_ttn.c) -----

TTN_MAX_VREF_FRAMES = 3
MIN_VREF_GAP_DAC = 50  # L_ttn.c min_vref_gap


def reconstruct_dvref_selection(
    tele: Dict[str, Any],
    max_vref_frames: int = TTN_MAX_VREF_FRAMES,
    min_vref_gap: int = MIN_VREF_GAP_DAC,
) -> Dict[str, Any]:
    """
    Replicate coarse-stage Vref picks from ``TTN_SweepSearch_dVref_Multi`` (L_ttn.c ~802–1003).

    After this, firmware runs a fine sweep (±200 DAC, step 10) and may adjust each entry
    (``vref_array[p] += 100`` vs readout path); final DACs are in ``tele['vrefs']``.
    """
    coarse = tele.get("coarse") or {}
    model = tele.get("model") or {}
    cv = list(coarse.get("vref") or [])
    cd = list(coarse.get("d_cnt_fast") or [])

    thr_x16 = int(model.get("coarse_dcnt_x16_threshold", (14100 * 9) // 10))
    if thr_x16 <= 0:
        thr_x16 = (14100 * 9) // 10

    candidates: List[Tuple[int, int]] = []
    for n in range(1, len(cv)):
        d = int(cd[n]) if n < len(cd) else 0
        if d > 0:
            candidates.append((int(cv[n]), d))

    candidates.sort(key=lambda t: -t[1])

    out: Dict[str, Any] = {
        "threshold_x16": thr_x16,
        "threshold_delta": thr_x16 / 16.0,
        "candidates": [{"vref_base": v, "delta": d} for v, d in candidates],
        "path": "none",
        "picks_coarse_plus100": [],
        "model_total_x16_reconstructed": 0,
        "final_payload_vrefs": list(tele.get("vrefs") or []),
        "model_payload": dict(model) if model else {},
    }

    if not candidates:
        return out

    best_d = candidates[0][1]
    best_v_base = candidates[0][0]

    if best_d * 16 >= thr_x16:
        v_sel = min(best_v_base + 100, 4095)
        out["path"] = "single_peak"
        out["picks_coarse_plus100"] = [v_sel]
        out["model_total_x16_reconstructed"] = best_d * 16
        return out

    picks: List[int] = []
    total_x16 = 0
    for v_base, d in candidates:
        if len(picks) >= max_vref_frames:
            break
        v_sel = int(min(v_base + 100, 4095))
        ok = True
        for p in picks:
            if abs(v_sel - int(p)) <= min_vref_gap:
                ok = False
                break
        if not ok:
            continue
        picks.append(v_sel)
        total_x16 += d * 16
        if total_x16 >= thr_x16:
            break

    out["path"] = "greedy_multi"
    out["picks_coarse_plus100"] = picks
    out["model_total_x16_reconstructed"] = total_x16
    return out


def plot_dvref_selection_model_plotly(
    dvref_tele: Dict[str, Any],
    *,
    save_path=None,
    experiment_name: str = "Experiment",
    show: bool = False,
    auto_open_html: bool = False,
    height: int = 820,
) -> Optional[go.Figure]:
    """
    Interactive Plotly report: coarse sweep, Δ(count), reconstructed picks, payload finals.

    Data source: ``*_dvref_multi_payload.bin`` decoded like ``parse_ttn_dvref_multi_payload``
    in ``Lacewing_Thread.py`` / ``load_functions.py``.
    """
    coarse = dvref_tele.get("coarse")
    model = dvref_tele.get("model")
    if not coarse or not model:
        print("Skipping dVref selection plot — no coarse + model telemetry in payload.")
        return None

    v = np.asarray(coarse.get("vref"), dtype=float)
    cnt = np.asarray(coarse.get("cnt_fast"), dtype=float)
    dcnt = np.asarray(coarse.get("d_cnt_fast"), dtype=float)

    recon = reconstruct_dvref_selection(dvref_tele)

    rows = [
        {
            "Role": "Coarse pick (pre-fine, +100)",
            "Vref_DAC": p,
            "Note": f"Algorithm: {recon['path']}",
        }
        for p in recon["picks_coarse_plus100"]
    ]
    for i, fv in enumerate(recon["final_payload_vrefs"]):
        rows.append(
            {
                "Role": "Final (payload, post-fine)",
                "Vref_DAC": int(fv),
                "Note": f"Slice index {i}",
            }
        )
    summary_df = pd.DataFrame(rows)
    print("\n" + "=" * 72)
    print(f"dVref VREF selection - {experiment_name} (L_ttn.c TTN_SweepSearch_dVref_Multi)")
    print("=" * 72)
    print(
        f"  Model pass_flag (payload): {recon['model_payload'].get('pass_fail')}  "
        f"total_dcnt_x16 (payload): {recon['model_payload'].get('total_dcnt_x16')}  "
        f"vrefs_to_pass (payload): {recon['model_payload'].get('vrefs_to_pass')}"
    )
    print(f"  Threshold d_cnt*16: {recon['threshold_x16']}  (delta >= {recon['threshold_delta']:.2f})")
    print(f"  Reconstructed path: {recon['path']}  reconstructed total (d_cnt*16): {recon['model_total_x16_reconstructed']}")
    print(summary_df.to_string(index=False))
    print("=" * 72 + "\n")

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=(
            "Coarse sweep: TTN_Count_Fast_Pixels vs Vref (DAC)",
            "First difference Δ(count) vs Vref — positives are model candidates",
            "Second difference ΔΔ (coarse curvature)",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=v,
            y=cnt,
            mode="lines+markers",
            name="cnt_fast",
            line=dict(color=DEFAULT_COLORBLIND_PALETTE[0], width=2),
            marker=dict(size=6),
        ),
        row=1,
        col=1,
    )

    colors = np.where(dcnt > 0, "rgba(230, 159, 0, 0.85)", "rgba(100, 100, 100, 0.45)")
    fig.add_trace(
        go.Bar(
            x=v,
            y=dcnt,
            name="d_cnt_fast",
            marker_color=colors,
        ),
        row=2,
        col=1,
    )

    thr_delta = recon["threshold_delta"]
    thr_x16_i = recon["threshold_x16"]
    fig.add_hline(
        y=thr_delta,
        line_dash="dash",
        line_color="#D55E00",
        annotation_text=f"Threshold Δ = {thr_x16_i}/16",
        annotation_position="right",
        row=2,
        col=1,
    )

    dd = np.asarray(coarse.get("dd_cnt_fast"), dtype=float)
    if dd.size == v.size:
        fig.add_trace(
            go.Scatter(
                x=v,
                y=dd,
                mode="lines+markers",
                name="dd_cnt_fast",
                line=dict(color="#009E73", width=1.5),
                marker=dict(size=5),
            ),
            row=3,
            col=1,
        )

    # Vertical markers: reconstructed coarse picks (pre-fine)
    for p in recon["picks_coarse_plus100"]:
        for r in (1, 2, 3):
            fig.add_vline(
                x=float(p),
                line_width=2,
                line_dash="dot",
                line_color="#F0E442",
                row=r,
                col=1,
            )

    finals = recon["final_payload_vrefs"]
    for fv in finals:
        for r in (1, 2, 3):
            fig.add_vline(
                x=float(fv),
                line_width=2,
                line_dash="dash",
                line_color="#009E73",
                row=r,
                col=1,
            )

    thr_x16 = recon["threshold_x16"]
    anno = (
        "<b>dVref multi selection</b> (see L_ttn.c TTN_SweepSearch_dVref_Multi)<br>"
        "① Coarse: Vref 250..4050 step 100, count fast pixels<br>"
        "② Candidates: Δ&gt;0 at coarse step; sort by Δ descending<br>"
        f"③ If max(Δ)×16≥{thr_x16}: pick one (Vref<sub>coarse</sub>+100); else greedy add spaced ≥{MIN_VREF_GAP_DAC} DAC until sum Δ×16≥threshold "
        f"(cap {TTN_MAX_VREF_FRAMES} refs)<br>"
        "④ Fine sweep ±200 step 10 per pick (payload shows finals)<br>"
        "<span style='color:#F0E442'>■</span> yellow dot-dash: coarse-stage pick (+100)<br>"
        "<span style='color:#009E73'>■</span> green dash: final payload Vref<br>"
        f"Payload: pass={recon['model_payload'].get('pass_fail')}, "
        f"total_dcnt×16={recon['model_payload'].get('total_dcnt_x16')}, "
        f"vrefs_to_pass={recon['model_payload'].get('vrefs_to_pass')}"
    )

    safe_name = sanitize_filename(experiment_name)
    fig.update_layout(
        title=f"{experiment_name} — dVref selection model (host payload + L_ttn.c logic)",
        height=height,
        width=980,
        font=dict(size=11),
        margin=dict(t=140, b=60),
        annotations=[
            dict(
                text=anno,
                xref="paper",
                yref="paper",
                x=0,
                y=1.12,
                xanchor="left",
                yanchor="bottom",
                showarrow=False,
                align="left",
                bordercolor="#cccccc",
                borderwidth=1,
                bgcolor="rgba(255,255,255,0.92)",
                font=dict(size=10),
            )
        ],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Vref (DAC)", row=3, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Δ count", row=2, col=1)
    fig.update_yaxes(title_text="ΔΔ", row=3, col=1)

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="#F0E442", width=2, dash="dot"),
            name="Coarse pick (+100, pre-fine)",
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="#009E73", width=2, dash="dash"),
            name="Final payload Vref",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    if save_path:
        try:
            sp = Path(save_path)
            sp.mkdir(parents=True, exist_ok=True)
            html_path = sp / f"{safe_name}_dvref_selection_model.html"
            fig.write_html(str(html_path))
            print(f"dVref selection model plot saved to: {html_path}")
        except Exception as e:
            print(f"Warning: Could not save dVref selection plot: {e}")

    if show:
        fig.show()

    if auto_open_html and save_path:
        try:
            import os
            import sys

            html_path = Path(save_path) / f"{safe_name}_dvref_selection_model.html"
            if sys.platform == "win32" and html_path.is_file():
                os.startfile(str(html_path))
        except Exception:
            pass

    return fig


if __name__ == "__main__":
    # Example usage
    print("This module provides plotting functions for wells using plotly.")
    print("Import and use the functions:")
    print("  - plot_wells_grid_plotly(exp, save_path, experiment_name)")
    print("  - plot_all_wells_combined_plotly(exp, save_path, experiment_name)")
    print("  - plot_wells_active_pixels_plotly(exp, save_path, experiment_name)")
    print("  - plot_wells_temperature_plotly(exp, save_path, experiment_name)")
