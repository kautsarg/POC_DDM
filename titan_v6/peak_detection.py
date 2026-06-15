import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter issues
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def calculate_well_derivatives(exp, total_wells, filter_order=1):
    """
    Calculate the first derivative for all wells in the experiment.
    
    Parameters
    ----------
    exp : Experiment
        The experiment object containing wells_list
    total_wells : int
        Total number of wells to process
        
    Returns
    -------
    dict
        Dictionary containing well data, derivatives, and time arrays for each well
    """
    if not hasattr(exp, 'wells_list') or len(exp.wells_list) == 0:
        raise ValueError("No wells available in experiment")
    
    # Limit to the specified total number of wells
    wells_to_process = exp.wells_list[:total_wells]
    n_wells = len(wells_to_process)
    
    # Dictionary to store results for each well
    well_data = {}
        
        # Process each well
    for well_idx, well in enumerate(wells_to_process):
        # Get the well data (preferring filtered data)
        try:
            if hasattr(well, 'well_2d_bs_active_filt'):
                signal_data = well.well_2d_bs_active_filt()  # Call the method
            elif hasattr(well, 'well_2d_bs_active_mean'):
                signal_data = well.well_2d_bs_active_mean  # This is a property
            elif hasattr(well, 'well_2d_nl_bs_active_mean'):
                signal_data = well.well_2d_nl_bs_active_mean  # This is a property
            else:
                # Fallback to raw data
                signal_data = np.mean(well.well_2d_npr, axis=1)
        except Exception as e:
            # If method call fails, try property
            try:
                signal_data = well.well_2d_bs_active_mean
            except:
                # Final fallback to raw data
                signal_data = np.mean(well.well_2d_npr, axis=1)
            
        # Ensure signal_data is 1D by taking mean across pixels if it's multi-dimensional
        if len(signal_data.shape) > 1:
            signal_data = np.mean(signal_data, axis=1)
        
        # Get time data in minutes aligned to the plotted/processed signal window.
        # Most well signals used here come from idx_settled:idx_end, so we slice the same range.
        try:
            if hasattr(well, "time_npr") and well.time_npr is not None and hasattr(well, "idx_settled") and hasattr(well, "idx_end"):
                t = np.asarray(well.time_npr)[int(well.idx_settled):int(well.idx_end)]
                if t.size == 0:
                    time_data = np.asarray([], dtype=float)
                else:
                    time_data = (t - t[0]) / 60.0  # minutes, zero at settled
            elif hasattr(well, "time_min") and well.time_min is not None:
                tmin = np.asarray(well.time_min, dtype=float)
                time_data = (tmin - tmin[0]) if tmin.size else tmin
            elif hasattr(well, "time") and well.time is not None:
                tsec = np.asarray(well.time, dtype=float)
                tmin = tsec / 60.0
                time_data = (tmin - tmin[0]) if tmin.size else tmin
            else:
                raise ValueError("No time information on well")
        except Exception as e:
            raise ValueError(f"Could not determine time axis for derivatives for well {well_idx + 1}: {e}")

        # Align time/signal lengths by trimming rather than inventing a synthetic time axis.
        n = min(len(time_data), len(signal_data))
        if n == 0:
            raise ValueError(f"Empty time/signal for well {well_idx + 1} (time={len(time_data)}, signal={len(signal_data)})")
        if len(time_data) != len(signal_data):
            time_data = time_data[:n]
            signal_data = signal_data[:n]
    
    # Calculate first derivative
        derivative_data = np.gradient(signal_data)
        
        # Apply simple filtering if filter_order > 1
        if filter_order > 1:
            # Simple moving average filter
            kernel = np.ones(filter_order) / filter_order
            derivative_data = np.convolve(derivative_data, kernel, mode='same')
        
        # Store data for this well
        well_data[well_idx] = {
            'signal': signal_data,
            'derivative': derivative_data,
            'time': time_data
        }
    
    return well_data


def detect_derivative_peaks(well_data, min_peak_height, min_peak_width_min=1.0, 
                          deriv_start_offset_min=1.0, deriv_end_offset_min=5.0):
    """
    Detect peaks in the first derivative data using fixed threshold approach.
    Mimics the MATLAB peak detection logic.
    
    Parameters
    ----------
    well_data : dict
        Dictionary containing well data from calculate_well_derivatives()
    min_peak_height : float
        Fixed threshold on derivative amplitude (default: 0.02)
    min_peak_width_min : float
        Minimum peak width in minutes (default: 1.0)
    deriv_start_offset_min : float
        Start detection window offset from beginning in minutes (default: 1.0)
    deriv_end_offset_min : float
        End detection window offset from end in minutes (default: 5.0)
        
    Returns
    -------
    dict
        Dictionary containing peak detection results for each well
    """
    if not well_data:
        raise ValueError("No well data provided")
    
        # Calculate time step (assuming uniform sampling)
    first_well = list(well_data.values())[0]
    time_data = first_well['time']
    dt = np.median(np.diff(time_data))  # time step per sample in minutes
    
    # Convert minimum peak width from minutes to samples
    min_peak_width_smp = max(1, round(min_peak_width_min / dt))
    
    # Results storage
    peak_results = {}
    
    for well_idx, data in well_data.items():
        derivative_data = data['derivative']
        time_data = data['time']
        
        # Define detection window
        deriv_start_time = time_data[0] + deriv_start_offset_min
        deriv_end_time = time_data[-1] - deriv_end_offset_min
        
        # Create mask for detection window
        mask_detection_window = (time_data >= deriv_start_time) & (time_data <= deriv_end_time)
        
        # Initialize results for this well
        peak_results[well_idx] = {
            'peak_time': 0.0,
            'peak_value': 0.0,
            'peak_detected': False
        }
        
        # Skip if detection window is too small
        if np.sum(mask_detection_window) < 3:
            continue
            
        # Extract data within detection window
        deriv_window = derivative_data[mask_detection_window]
        time_window = time_data[mask_detection_window]
        
        # Fixed-threshold peak search using scipy.signal.find_peaks
        peaks, properties = find_peaks(
            deriv_window,
            height=min_peak_height,
            width=min_peak_width_smp
        )
        
        if len(peaks) == 0:
            # No peaks found - leave as zeros
            continue
            
        # Choose the tallest peak (maximum height)
        peak_heights = deriv_window[peaks]
        max_peak_idx = int(np.argmax(peak_heights))
        
        # Store results
        peak_time = time_window[peaks[max_peak_idx]]
        peak_value = peak_heights[max_peak_idx]
        
        peak_results[well_idx] = {
            'peak_time': peak_time,
            'peak_value': peak_value,
            'peak_detected': True,
            'all_peaks_times': time_window[peaks],
            'all_peaks_values': peak_heights
        }
    
    return peak_results


def detect_second_derivative_peaks_from_first_derivative(well_data, second_deriv_data, peak_results, 
                                                       deriv2_start_min=1.0, deriv_end_min=5.0):
    """
    Detect peaks in second derivative using first derivative peaks as anchors.
    Implements the MATLAB logic for finding zero crossings and tracking back to peaks.
    
    Parameters
    ----------
    well_data : dict
        Dictionary containing first derivative data
    second_deriv_data : dict
        Dictionary containing second derivative data
    peak_results : dict
        Dictionary containing first derivative peak detection results
    deriv2_start_min : float
        Start time for second derivative analysis in minutes
    deriv_end_min : float
        End time for second derivative analysis in minutes
        
    Returns
    -------
    dict
        Dictionary containing TTP (Time To Peak) and ZC (Zero Crossing) times for each well
    """
    if not well_data or not second_deriv_data:
        raise ValueError("No well data provided")
    
    # Parameters (from MATLAB code)
    EXPAND_WINS = [0.5, 1.0, 2.0, 5.0]  # min; expanding search windows around anchor
    LOOKBACK_MAX = 10.0                   # min; only search D² peak this far BEFORE the ZC/anchor
    MIN_PEAK_SEP = 0.1                    # min; minimum separation between D² micro-peaks
    EPS_FACTOR = 2                         # near-zero tolerance = EPS_FACTOR * MAD in window
    MAX_GAP = 8.0                          # min; max distance from anchor for global ZC fallback
    
    # Get time base from first well for dt2 calculation (should be similar across wells)
    first_well = list(well_data.values())[0]
    reference_time = first_well['time']
    dt2 = np.median(np.diff(reference_time))  # minutes/sample for D²
    
    # Results storage
    results = {}
    
    print(f"\nTTP Calculation for {len(well_data)} wells:")
    print(f"Time window: {deriv2_start_min:.1f} to {deriv_end_min:.1f} minutes")
    print(f"dt2: {dt2:.4f} minutes/sample")
    print("=" * 60)
    
    for well_idx in well_data.keys():
        # Initialize per-well state
        zc = np.nan
        found_zc = False
        anchor = np.nan
        
        # Get data for this well
        first_deriv_data = well_data[well_idx]['derivative']
        second_deriv_data_well = second_deriv_data[well_idx]['second_derivative']
        well_time_data = well_data[well_idx]['time']  # Use this well's time data
        
        # 0) Anchor from Diff¹ (guard: explicit 0 means "skip this well")
        if well_idx in peak_results and peak_results[well_idx]['peak_detected']:
            anchor = peak_results[well_idx]['peak_time']
            print(f"  Well {well_idx + 1}: Anchor (1st deriv peak) at {anchor:.2f} min")
        else:
            # Skip this well if no first derivative peak
            print(f"  Well {well_idx + 1}: No first derivative peak found, skipping")
            continue
        
        # Clamp anchor to time range
        anchor = np.clip(anchor, well_time_data[0], well_time_data[-1])
        print(f"  Well {well_idx + 1}: Anchor clamped to {anchor:.2f} min (time range: {well_time_data[0]:.2f} to {well_time_data[-1]:.2f} min)")
        
        # Create mask for valid time window
        mask = (well_time_data >= deriv2_start_min) & (well_time_data <= deriv_end_min)
        if not np.any(mask):
            print(f"  Well {well_idx + 1}: No valid time window, skipping")
            continue
            
        t2 = well_time_data[mask]
        x2 = second_deriv_data_well[mask]
        print(f"  Well {well_idx + 1}: Second derivative data: {len(t2)} points from {t2[0]:.2f} to {t2[-1]:.2f} min")
        
        # 1) FIRST descending ZC (+ → −) *after* the anchor (expanding windows)
        print(f"    Well {well_idx + 1}: Searching for ZC after anchor {anchor:.2f} min")
        for hw in EXPAND_WINS:
            # Only AFTER the anchor (no symmetric window)
            win = (t2 >= anchor) & (t2 <= anchor + hw)
            if np.sum(win) < 2:
                continue
                
            tw = t2[win]
            xw = x2[win]
            
            # Descending zero-crossing: + -> -
            s1 = xw[:-1]
            s2 = xw[1:]
            desc = (s1 > 0) & (s2 <= 0) & (s2 != s1)
            
            if np.any(desc):
                j = np.where(desc)[0][0]  # earliest AFTER anchor
                # Linear interpolation for precise ZC time
                zc = tw[j] - s1[j] * (tw[j+1] - tw[j]) / (s2[j] - s1[j])
                found_zc = True
                print(f"    Well {well_idx + 1}: Found ZC at {zc:.2f} min (window: {hw:.1f} min)")
                break
        
        # Fallback #1: look for descending ZC anywhere on t2
        if not found_zc:
            print(f"    Well {well_idx + 1}: No ZC found in expanding windows, trying Fallback #1")
            s1 = x2[:-1]
            s2 = x2[1:]
            desc = (s1 > 0) & (s2 <= 0) & (s2 != s1)
            if np.any(desc):
                idxs = np.where(desc)[0]
                tcand = t2[idxs] - s1[idxs] * (t2[idxs+1] - t2[idxs]) / (s2[idxs] - s1[idxs])
                
                # nearest to anchor
                gap_nearest = np.min(np.abs(tcand - anchor))
                j_near = np.argmin(np.abs(tcand - anchor))
                zc_nearest = tcand[j_near]
                
                # earliest descending ZC
                zc_earliest = np.min(tcand)
                
                if gap_nearest <= MAX_GAP:
                    zc = zc_nearest
                    found_zc = True
                    print(f"    Well {well_idx + 1}: Found ZC (Fallback #1) at {zc:.2f} min (gap: {gap_nearest:.2f} min)")
                else:
                    # EARLY-CASE: accept earliest descending ZC if it's before anchor
                    if zc_earliest < anchor:
                        zc = zc_earliest
                        found_zc = True
                        print(f"    Well {well_idx + 1}: Found ZC (Fallback #1 early) at {zc:.2f} min")
                    else:
                        print(f"    Well {well_idx + 1}: ZC found but too far from anchor (gap: {gap_nearest:.2f} min > {MAX_GAP:.1f} min)")
            else:
                print(f"    Well {well_idx + 1}: No descending ZC found in Fallback #1")
        
        # Fallback #2: center D² then search descending ZC
        if not found_zc:
            print(f"    Well {well_idx + 1}: No ZC found in Fallback #1, trying Fallback #2")
            win_smp = max(3, round(2/dt2))  # ~2 min window
            x2c = x2 - np.median(x2)  # local-mean centered (simplified)
            s1 = x2c[:-1]
            s2 = x2c[1:]
            desc = (s1 > 0) & (s2 <= 0) & (s2 != s1)
            if np.any(desc):
                idxs = np.where(desc)[0]
                tcand = t2[idxs] - s1[idxs] * (t2[idxs+1] - t2[idxs]) / (s2[idxs] - s1[idxs])
                gap = np.min(np.abs(tcand - anchor))
                j = np.argmin(np.abs(tcand - anchor))
                if np.isfinite(gap) and gap <= MAX_GAP:
                    zc = tcand[j]
                    found_zc = True
                    print(f"    Well {well_idx + 1}: Found ZC (Fallback #2) at {zc:.2f} min (gap: {gap:.2f} min)")
            else:
                print(f"    Well {well_idx + 1}: No descending ZC found in Fallback #2")
        
        # Fallback #3: no ZC → set TTP to largest |D²| BEFORE anchor and continue
        if not found_zc:
            print(f"    Well {well_idx + 1}: No ZC found in Fallback #2, using Fallback #3")
            pre_start_time = max(deriv2_start_min, anchor - LOOKBACK_MAX)
            pre_mask = (t2 >= pre_start_time) & (t2 < (anchor - max(2*dt2, 0.02)))
            print(f"    Well {well_idx + 1}: Fallback #3 window: {pre_start_time:.2f} to {anchor - max(2*dt2, 0.02):.2f} min")
            if np.any(pre_mask):
                tp = t2[pre_mask]
                xp = x2[pre_mask]
                loc = np.argmax(np.abs(xp))
                ttp_time = tp[loc]
                results[well_idx] = {
                    'ttp_time': ttp_time,
                    'zc_time': np.nan,
                    'found_zc': False,
                    'found_ttp': True
                }
                
                # Debug output for Fallback #3 case
                print(f"  Well {well_idx + 1}: TTP (Fallback #3) at {ttp_time:.2f} min (anchor: {anchor:.2f} min)")
            else:
                print(f"    Well {well_idx + 1}: Fallback #3 window is empty, skipping well")
            continue  # skip ZC_time assignment
        
        # Guard & reduction to a single scalar ZC
        if not np.isscalar(zc) or not np.isfinite(zc):
            print(f"    Well {well_idx + 1}: ZC is not scalar or finite ({zc}), skipping")
            continue
            
        print(f"    Well {well_idx + 1}: ZC validation passed, proceeding with TTP calculation")
            
        # 3) Largest prior D² peak BEFORE this ZC
        pre_start_time = max(deriv2_start_min, zc - LOOKBACK_MAX)
        pre_mask = (t2 >= pre_start_time) & (t2 < (zc - max(2*dt2, 0.02)))
        print(f"    Well {well_idx + 1}: Looking for TTP in window {pre_start_time:.2f} to {zc - max(2*dt2, 0.02):.2f} min")
        if not np.any(pre_mask):
            print(f"    Well {well_idx + 1}: No valid pre-ZC window found, trying anchor fallback")
            # extra guard: if ZC is too early, try window before anchor instead
            pre_mask = (t2 >= max(deriv2_start_min, anchor - LOOKBACK_MAX)) & \
                      (t2 < (anchor - max(2*dt2, 0.02)))
            if not np.any(pre_mask):
                print(f"    Well {well_idx + 1}: No valid pre-ZC window found even with anchor fallback")
                continue
                
        tp = t2[pre_mask]
        xp = x2[pre_mask]
        if pre_start_time == max(deriv2_start_min, anchor - LOOKBACK_MAX):
            print(f"    Well {well_idx + 1}: Using anchor fallback window {max(deriv2_start_min, anchor - LOOKBACK_MAX):.2f} to {anchor - max(2*dt2, 0.02):.2f} min")
        else:
            print(f"    Well {well_idx + 1}: Using ZC-based window {pre_start_time:.2f} to {zc - max(2*dt2, 0.02):.2f} min")
        
        # Determine recent sign before ZC to pick the proper extremum
        recent_mask = (tp >= (zc - min(1.0, LOOKBACK_MAX))) & (tp < zc)
        if not np.any(recent_mask):
            recent_mask = np.ones(len(tp), dtype=bool)
            print(f"    Well {well_idx + 1}: No recent data before ZC, using all data")
        else:
            print(f"    Well {well_idx + 1}: Using recent data from {tp[recent_mask][0]:.2f} to {tp[recent_mask][-1]:.2f} min")
            
        sgn = np.sign(np.median(xp[recent_mask]))
        if sgn == 0:
            # Choose orientation by the strongest magnitude if median is ~0
            kmax = int(np.argmax(np.abs(xp[recent_mask])))
            r_idx = np.where(recent_mask)[0]
            sgn = np.sign(xp[r_idx[0] + kmax])
            if sgn == 0:
                sgn = 1
                
        print(f"    Well {well_idx + 1}: Sign determined as {sgn}")
        y = sgn * xp  # flip so desired extremum is positive
        
        # Peak pick
        min_sep_idx = max(1, round(MIN_PEAK_SEP / dt2))
        min_prom = max(2 * np.median(np.abs(xp - np.median(xp))), 1e-4)
        print(f"    Well {well_idx + 1}: Peak detection params - min_sep: {min_sep_idx} samples, min_prom: {min_prom:.6f}")
        
        try:
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(y, distance=min_sep_idx, prominence=min_prom)
            
            if len(peaks) == 0:
                # fallback: biggest |D²| prior to ZC
                loc = int(np.argmax(np.abs(xp)))
                ttp_time = tp[loc]
                print(f"    Well {well_idx + 1}: No peaks found, using max |D²| at {ttp_time:.2f} min")
            else:
                imax = int(np.argmax(peaks))
                ttp_time = tp[peaks[imax]]
                print(f"    Well {well_idx + 1}: Found {len(peaks)} peaks, using peak {imax} at {ttp_time:.2f} min")
        except ImportError:
            # fallback if scipy not available
            loc = int(np.argmax(np.abs(xp)))
            ttp_time = tp[loc]
            print(f"    Well {well_idx + 1}: Scipy not available, using max |D²| at {ttp_time:.2f} min")
        
        # Store results
        results[well_idx] = {
            'ttp_time': ttp_time,
            'zc_time': zc,
            'found_zc': True,
            'found_ttp': True
        }
        
        # Debug output to verify TTP calculation
        print(f"  Well {well_idx + 1}: TTP calculated at {ttp_time:.2f} min (anchor: {anchor:.2f} min, ZC: {zc:.2f} min)")
        print(f"  Well {well_idx + 1}: TTP calculation complete")
    
    return results


def calculate_well_second_derivatives(well_data, filter_order):
    """
    Calculate second derivatives for all wells with optional filtering.
    
    Args:
        well_data: Dictionary containing first derivative data from calculate_well_derivatives
        filter_order: Filter order for smoothing (1 = no filter, higher = more smoothing)
    
    Returns:
        Dictionary with well data containing 'signal', 'derivative', 'second_derivative', and 'time' for each well
    """
    second_deriv_data = {}
    
    for well_idx, data in well_data.items():
        # Get the first derivative data
        first_derivative = data['derivative']
        time_data = data['time']
        
        # Calculate second derivative using gradient
        second_derivative = np.gradient(first_derivative)
        
        # Apply filtering if filter_order > 1
        if filter_order > 1:
            kernel = np.ones(filter_order) / filter_order
            second_derivative = np.convolve(second_derivative, kernel, mode='same')
        
        # Store the data
        second_deriv_data[well_idx] = {
            'signal': data['signal'],
            'derivative': data['derivative'],
            'second_derivative': second_derivative,
            'second_derivative_filtered': second_derivative,  # Store both filtered and unfiltered
            'time': time_data
        }
    
    return second_deriv_data


def plot_well_derivatives(well_data, peak_results=None, second_peak_results=None, 
                          save_path=None, experiment_name=None, figsize=(28, 18)):
    """
    Plot the raw signal and first derivative for all wells.
    
    Parameters
    ----------
    well_data : dict
        Dictionary containing well data from calculate_well_derivatives()
    peak_results : dict, optional
        Dictionary containing first derivative peak detection results
    second_peak_results : dict, optional
        Dictionary containing second derivative peak detection results (TTP and ZC)
    save_path : str, optional
        Directory path to save the figure. If None, figure is shown instead.
    experiment_name : str, optional
        Name of the experiment to use in filename
    figsize : tuple
        Figure size (width, height)
        
    Returns
    -------
    tuple
        Figure and axes objects
    """
    if not well_data:
        raise ValueError("No well data provided")
    
    n_wells = len(well_data)
    
    # Calculate subplot layout
    n_cols = min(3, n_wells)  # Max 3 columns for readability
    n_rows = (n_wells + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=figsize)
    
    # Handle single well case
    if n_wells == 1:
        axes = axes.reshape(2, 1)
    elif n_rows == 1:
        axes = axes.reshape(2, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each well
    for well_idx, data in well_data.items():
        signal_data = data['signal']
        derivative_data = data['derivative']
        time_data = data['time']
        
        # Calculate row and column indices
        row_raw = (well_idx // n_cols) * 2
        row_deriv = (well_idx // n_cols) * 2 + 1
        col = well_idx % n_cols
        
        # Get axes for this well
        ax_raw = axes[row_raw, col]
        ax_deriv = axes[row_deriv, col]
        
        # Plot raw signal
        ax_raw.plot(time_data, signal_data, 'b-', linewidth=1.5, label='Raw Signal')
        
        # Overlay TTP from second derivative if available
        if second_peak_results is not None and well_idx in second_peak_results:
            ttp_info = second_peak_results[well_idx]
            if ttp_info['found_ttp']:
                ttp_time = ttp_info['ttp_time']
                closest_idx = np.argmin(np.abs(time_data - ttp_time))
                ttp_value_at_time = signal_data[closest_idx]
                
                # Plot the TTP as a purple diamond
                ax_raw.plot(ttp_time, ttp_value_at_time, 'D', color='purple', markersize=12, 
                           label=f'TTP: {ttp_time:.2f} min', markeredgecolor='darkviolet', 
                           markeredgewidth=2, alpha=0.8)
                
                # Add annotation with TTP information
                ax_raw.annotate(f'TTP\n{ttp_time:.2f} min',
                               xy=(ttp_time, ttp_value_at_time),
                               xytext=(10, 10), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='purple', alpha=0.7),
                               fontsize=8, ha='left', color='white')
        
        ax_raw.set_title(f'Well {well_idx + 1} - Raw Signal', fontsize=11, fontweight='bold')
        ax_raw.set_ylabel('Signal Amplitude', fontsize=10)
        ax_raw.legend(fontsize=9)
        ax_raw.grid(True, alpha=0.3)
        ax_raw.tick_params(labelsize=9)
        
        # Plot first derivative
        ax_deriv.plot(time_data, derivative_data, 'g-', linewidth=1.5, label='First Derivative')
        
        # Overlay detected peaks as circles if peak_results are provided
        if peak_results is not None and well_idx in peak_results:
            peak_info = peak_results[well_idx]
            if peak_info['peak_detected']:
                # Find the closest time index for accurate plotting
                peak_time = peak_info['peak_time']
                closest_idx = np.argmin(np.abs(time_data - peak_time))
                peak_value_at_time = derivative_data[closest_idx]
                
                # Plot the peak as a red circle
                ax_deriv.plot(peak_time, peak_value_at_time, 'ro', markersize=10, 
                            label=f'Peak: {peak_time:.2f} min', markeredgecolor='darkred', 
                            markeredgewidth=2, alpha=0.8)
                
                # Add annotation with peak information
                ax_deriv.annotate(f'{peak_time:.2f} min\n{peak_info["peak_value"]:.4f}',
                                xy=(peak_time, peak_value_at_time),
                                xytext=(10, 10), textcoords='offset points',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                                fontsize=8, ha='left')
        
        ax_deriv.set_title(f'Well {well_idx + 1} - First Derivative', fontsize=11, fontweight='bold')
        ax_deriv.set_xlabel('Time', fontsize=10)
        ax_deriv.set_ylabel('First Derivative', fontsize=10)
        ax_deriv.legend(fontsize=9)
        ax_deriv.grid(True, alpha=0.3)
        ax_deriv.tick_params(labelsize=9)
        
        # Add well number annotation
        ax_raw.text(0.02, 0.95, f'Well {well_idx + 1}', transform=ax_raw.transAxes, 
                   fontsize=10, verticalalignment='top', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    # Hide unused subplots
    for row in range(n_rows * 2):
        for col in range(n_cols):
            # Calculate which well this subplot would belong to
            well_position = (row // 2) * n_cols + col
            # Hide if this position is beyond our actual number of wells
            if well_position >= n_wells:
                axes[row, col].set_visible(False)
    
    # Add overall title
    plt.suptitle(f'All Wells - Raw Signal and First Derivative (Total Wells: {n_wells})', 
                 fontsize=14, y=0.98)
    
    # Adjust layout to minimize blank space
    plt.subplots_adjust(
        top=0.94,      # Keep title close
        bottom=0.06,   # Minimal bottom margin
        left=0.06,     # Minimal left margin
        right=0.94,    # Minimal right margin
        hspace=0.9,   # Slightly more vertical spacing to prevent title overlap
        wspace=0.15    # Tighter horizontal spacing between subplots
    )
    
    # Save or show the figure
    if save_path is not None and experiment_name is not None:
        import os
        os.makedirs(save_path, exist_ok=True)
        filename = f"{experiment_name}_raw_signal_and_first_derivative.png"
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved first derivative plot: {filepath}")
        plt.close()
    else:
        plt.close()  # Close the figure to free memory
    
    return fig, axes


def plot_well_second_derivatives(well_data, second_peak_results=None, 
                                save_path=None, experiment_name=None, figsize=(28, 18)):
    """
    Plot second derivatives for all wells.
    
    Args:
        well_data: Dictionary containing second derivative data from calculate_well_second_derivatives
        second_peak_results: Dictionary containing second derivative peak detection results (TTP and ZC)
        save_path: Directory path to save the figure. If None, figure is shown instead.
        experiment_name: Name of the experiment to use in filename
        figsize: Figure size tuple (width, height)
    
    Returns:
        Figure and axes objects
    """
    n_wells = len(well_data)
    
    # Calculate subplot layout
    n_cols = min(5, n_wells)  # Maximum 5 columns
    n_rows = (n_wells + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Handle single subplot case
    if n_wells == 1:
        axes = np.array([axes])
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each well
    for well_idx, data in well_data.items():
        row = well_idx // n_cols
        col = well_idx % n_cols
        
        # Get axes for this well
        ax = axes[row, col]
        
        # Plot second derivative
        time_data = data['time']
        second_derivative = data['second_derivative']
        
        ax.plot(time_data, second_derivative, 'r-', linewidth=1.5, label='Second Derivative')
        
        # Overlay ZC from second derivative analysis if available
        if second_peak_results is not None and well_idx in second_peak_results:
            zc_info = second_peak_results[well_idx]
            if zc_info['found_zc']:
                zc_time = zc_info['zc_time']
                closest_idx = np.argmin(np.abs(time_data - zc_time))
                zc_value_at_time = second_derivative[closest_idx]
                
                # Plot the ZC as a green triangle
                ax.plot(zc_time, zc_value_at_time, '^', color='green', markersize=12, 
                       label=f'ZC: {zc_time:.2f} min', markeredgecolor='darkgreen', 
                       markeredgewidth=2, alpha=0.8)
                
                # Add annotation with ZC information
                ax.annotate(f'ZC\n{zc_time:.2f} min',
                           xy=(zc_time, zc_value_at_time),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7),
                           fontsize=8, ha='left', color='white')
        
        ax.set_title(f'Well {well_idx + 1} - Second Derivative', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (min)', fontsize=10)
        ax.set_ylabel('Second Derivative', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)
        
        # Add well number annotation
        ax.text(0.02, 0.95, f'Well {well_idx + 1}', transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
    
    # Hide unused subplots
    for row in range(n_rows):
        for col in range(n_cols):
            well_position = row * n_cols + col
            if well_position >= n_wells:
                axes[row, col].set_visible(False)
    
    # Add overall title
    plt.suptitle(f'All Wells - Second Derivative (Total Wells: {n_wells})', 
                 fontsize=14, y=0.98)
    
    # Adjust layout
    plt.subplots_adjust(
        top=0.94,
        bottom=0.06,
        left=0.06,
        right=0.94,
        hspace=0.9,
        wspace=0.15
    )
    
    # Save or show the figure
    if save_path is not None and experiment_name is not None:
        import os
        os.makedirs(save_path, exist_ok=True)
        filename = f"{experiment_name}_second_derivative.png"
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved second derivative plot: {filepath}")
        plt.close()
    else:
        plt.close()  # Close the figure to free memory
    
    return fig, axes


def plot_all_wells_raw_and_derivative(exp, total_wells, filter_order=1, min_peak_height=0.0001):
    """
    Convenience function that combines calculation, peak detection, and plotting.
    
    Parameters
    ----------
    exp : Experiment
        The experiment object containing wells_list
    total_wells : int
        Total number of wells to process and plot
    filter_order : int
        Filter order for smoothing derivatives
    min_peak_height : float
        Minimum peak height threshold for detection
    """
    # Calculate derivatives for all wells
    well_data = calculate_well_derivatives(exp, total_wells, filter_order)
    
    # Detect peaks in derivatives
    peak_results = detect_derivative_peaks(well_data, min_peak_height)
    
    # Plot the results with peak overlay
    return plot_well_derivatives(well_data, peak_results)
