import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

import functions
import load_functions as load
import preprocessing_functions as preprocess
import linearise as linearise
from Well import Well
from Experiment import Experiment
from plt_Experiment_summary import titan_plt_summary

def titan_load_and_preprocessing(exp_path: Path,
                                 n_wells=1, start_type=None, n_a_type="v04", end_time_min=30,
                                 print_status=False, plt_gain_calib=False, save_gain_calib=False,
                                 nrows=290, ncols=204, plot_gain_3d=False,
                                 ref_idx: int = -1):
    """
    Load and pre-process the data from the .bin input files exp_path_calib and exp_path_readout

    Parameters
    ----------
    exp_path: Path
        Path of the experiment folder
    n_wells: int
        Number of wells in the experiment. Optional, Default is 1; can be 1, 2, 4, 6, 10.
    start_type: str
        string specifying the type of injection time detection. Can be 'manual', 'auto', None. Default None.
        'manual' means finding the injection time in a csv file in the mother folder;
        'auto' means finding the highest derivative in the first 10 minutes of the experiment;
        None means that the experiment is already injected at the beginning of the recording.
    print_status: bool
        if True, print updates on the state of loading and preprocessing. Optional, Default False.

    Returns
    -------
    Experiment
        object of type Experiment
    """
    
    #  1) LOAD .BIN DATA
    # 1A) First load the new Vref file to acertain the number of vrefs within an experiment
    exp_path_vref = None
    n_vrefs, vref_vect = 0, None
    try:
        exp_path_vref = load.find_most_recent_vref(exp_path)
        if print_status:
            print(f'EXP PATH {exp_path} \nVREF PATH {exp_path_vref} \nLoad vref... ')

        n_vrefs, vref_vect = load.load_vref_sweep(exp_path)
        if print_status:
            print(f"Loaded vref list: n_vrefs={n_vrefs}, first={vref_vect[0] if n_vrefs else None}, last={vref_vect[-1] if n_vrefs else None}")
    except Exception:
        # Older experiments may not have a vref sweep file; keep going with legacy readout parsing.
        exp_path_vref = None


    #  1A) READOUT: Find most recent readout file
    exp_path_readout = load.find_most_recent_readout(exp_path)
    if print_status: print(f'EXP PATH {exp_path} \nREADOUT PATH {exp_path_readout} \nLoad data... ')

    # Load data from bin files: TIME to time_vect and CHEM DATA to frame_3d (readout)
    data_readout_list = load.binary_file_read(exp_path_readout)

    frame_4d = None
    # When v05/v06 is detected, readout/gain/linearization are 4D with a ref/vref axis.
    # Downstream code in this repo still expects a single 3D slice, so we select one ref index.
    ref_idx_norm = None
    # `readout_meta` carries the per-frame parameter tail decoded from the binary file
    # (timestamp, vref_array, chem_avg, temp_avg, temp_avg_Vs_shifted, adc, tam, tch, n_wells, version).
    # For v06 we use this in place of the (now zero-valued) per-pixel temperature stream.
    readout_meta = None
    n_a_lower = str(n_a_type).lower()
    is_v05_like = n_a_lower in ("v05", "v06")
    # Pick the format string we use for parsing the readout/gain. v05 and v06 share the on-disk format.
    parse_n_a_type = "v05" if is_v05_like else n_a_type
    # If a vref sweep file is present, we *try* the new v05/v06 readout format (pixels for all refs per timestamp).
    # Some datasets may have a vref sweep file but still use legacy readout formatting, so fall back safely.
    try:
        if exp_path_vref is not None:
            time_vect, frame_4d, readout_meta = load.list_to_numpy_readout(
                data_readout_list, nrows=nrows, ncols=ncols, n_a_type=parse_n_a_type, return_meta=True
            )
            # v05/v06 stores multiple "ref"/vref pixel frames per timestamp, concatenated in the binary file.
            # `frame_4d` is the unpacked pixel tensor with shape:
            #   (nrows, ncols, n_refs, n_time)
            # where:
            # - axis 0-1: spatial pixel coordinates (row, col)
            # - axis 2: ref/vref index (frame0, frame1, frame2, ...)
            # - axis 3: time index (timestamp/frame over time)
            #
            # Downstream code in this repo expects a 3D readout array (nrows, ncols, n_time),
            # so we pick one ref slice.
            n_refs = int(frame_4d.shape[2])
            ref_idx_norm = int(ref_idx)
            if ref_idx_norm < 0:
                ref_idx_norm = n_refs + ref_idx_norm
            ref_idx_norm = max(0, min(ref_idx_norm, n_refs - 1))
            frame_3d = frame_4d[:, :, ref_idx_norm, :]
        else:
            time_vect, frame_3d = load.list_to_numpy_readout(data_readout_list, nrows=nrows, ncols=ncols, n_a_type=n_a_type)
    except Exception as e:
        fallback_type = "v04" if is_v05_like else n_a_type
        if print_status:
            print(f"Warning: {parse_n_a_type} parsing failed ({str(e)}). Falling back to n_a_type={fallback_type}.")
        time_vect, frame_3d = load.list_to_numpy_readout(
            data_readout_list, nrows=nrows, ncols=ncols, n_a_type=fallback_type
        )
        frame_4d = None
        ref_idx_norm = None
        readout_meta = None
    if frame_4d is not None:
        print(np.shape(frame_4d))
    # Time: check if the time resets during the experiment. If so, fix time axis.
    time_vect = load.unwrap_time_vect(time_vect)
    if print_status:
        try:
            dt = np.diff(time_vect.astype(float))
            dt_pos = dt[dt > 0]
            dt_med = float(np.median(dt_pos)) if dt_pos.size else float("nan")
            dur_min = float((time_vect[-1] - time_vect[0]) / 60.0) if len(time_vect) else float("nan")
            print(f"Time vector: n={len(time_vect)}, median_dt={dt_med:.4g}s, duration≈{dur_min:.2f}min")
        except Exception:
            print(f"Time vector: n={len(time_vect)}")
    else:
        print(len(time_vect))

    # 1B) ACTIVE PIXELS: Load data from bin files
    exp_path_idx_active = load.find_corresp_file(exp_path, exp_path_readout, 'idx_active')
    idx_active_list = load.binary_file_read(exp_path_idx_active)
    idx_active_input = np.array(idx_active_list[:nrows*ncols]).reshape(nrows, ncols, order='F').reshape(nrows*ncols, order='C')
    print(np.shape(idx_active_input))
    # 1C) GAIN FILE + LOG FILE & LINEARISATION



    exp_path_idx_active = load.find_corresp_file(exp_path, exp_path_readout, 'gain')
    gain_list = load.binary_file_read(exp_path_idx_active)
    if n_a_type == "v01" or n_a_type == "v02" or n_a_type == "v03":
        arr_gain_3d, vref = load.list_to_numpy_gain(gain_list, n_a_type=n_a_type, exp_path=exp_path, exp_path_readout=exp_path_readout, plt_show=False)
    elif n_a_type == "v04":
        arr_gain_3d, vref = load.list_to_numpy_gain(gain_list, n_a_type=n_a_type, plt_show=False)
    elif is_v05_like:
        # v05/v06 gain files store pixels for all refs per record; use the known vref sweep count as n_refs.
        if vref_vect is None or n_vrefs == 0:
            raise Exception(f"RAISED titan_load_and_preprocessing: {n_a_type} requires vref sweep file to infer n_refs for gain parsing")
        arr_gain_4d, _vref_unused = load.list_to_numpy_gain(
            gain_list, n_a_type=parse_n_a_type, n_refs=int(n_vrefs), return_4d=True, plt_show=False
        )
        # Keep the existing 3D variable for now so downstream code still works:
        # pick the same ref slice we selected for readout (or default to last).
        n_refs_gain = int(arr_gain_4d.shape[2])
        if ref_idx_norm is None:
            ref_idx_norm = n_refs_gain - 1
        if ref_idx_norm < 0:
            ref_idx_norm = n_refs_gain + int(ref_idx_norm)
        ref_idx_norm = max(0, min(int(ref_idx_norm), n_refs_gain - 1))

        arr_gain_3d = arr_gain_4d[:, :, ref_idx_norm, :]

        # Provide a scalar vref matching the selected ref slice (important for linearization).
        vref = float(vref_vect[ref_idx_norm]) if vref_vect is not None and ref_idx_norm < len(vref_vect) else float(vref_vect[0])
    else:
        raise f"RAISED titan_load_and_preprocessing: Impossible to call list_to_numpy_gain with n_a_type {n_a_type}"
    



    # Gain-based active pixel filtering.
    # Legacy gain is 3D: (nrows, ncols, 8).
    # v05/v06 gain is 4D: (nrows, ncols, n_refs, 8) and we want pixels that pass the gain filter
    # for each ref/vref slice.
    if is_v05_like:
        # Run the existing 3D filter once per ref slice.
        # NOTE: downstream currently uses a single ref slice, so combining masks across refs (AND)
        # can zero out the active-pixel mask and cause NaN-only downstream plots.
        idx_active_gain_list = []
        for ref_i in range(arr_gain_4d.shape[2]):
            idx_ref = preprocess.filter_by_gain(
                arr_gain_4d[:, :, ref_i, :],
                plt_show=False,
                plt_save=False,
                exp_path=exp_path,
            )
            idx_active_gain_list.append(idx_ref)

        # Mask used by the rest of the pipeline: pick the selected ref slice.
        idx_active_gain = idx_active_gain_list[ref_idx_norm]

        # Optional plotting/saving: show the last-ref slice (keeps behavior similar to legacy).
        if plt_gain_calib or save_gain_calib:
            _ = preprocess.filter_by_gain(
                arr_gain_3d,
                plt_show=plt_gain_calib,
                plt_save=save_gain_calib,
                exp_path=exp_path,
            )
    else:
        idx_active_gain = preprocess.filter_by_gain(
            arr_gain_3d, plt_show=plt_gain_calib, plt_save=save_gain_calib, exp_path=exp_path
        )
    print("new gain shape ", np.shape(idx_active_gain))
    # Plot filtered gain data using Plotly (only active pixels)
    if plot_gain_3d:
        try:
            from plot_derivatives_plotly import plot_gain_3d_plotly
            experiment_name = exp_path.name if hasattr(exp_path, 'name') else str(exp_path)
            # Apply filter_by_vrange to gain data as well (for this plot only)
            if is_v05_like:
                # Reshape to (n_frames, n_pixels, array_size=n_refs)
                n_frames = arr_gain_4d.shape[3]
                array_size = arr_gain_4d.shape[2]
                gain_3d = arr_gain_4d.reshape(-1, array_size, n_frames).transpose(2, 0, 1)

                # Run vrange filter per ref slice and combine (AND)
                idx_active_vrange_list = [
                    preprocess.filter_by_vrange(gain_3d[:, :, ref_i]) for ref_i in range(array_size)
                ]
                # Keep per-ref masks (no reduction) so callers can inspect how each ref behaves.
                idx_active_vrange = idx_active_vrange_list
            else:
                # Reshape arr_gain_3d to 2D (n_frames x n_pixels) for filter_by_vrange
                gain_2d = arr_gain_3d.reshape(-1, arr_gain_3d.shape[2]).T  # Shape: (n_frames, n_pixels)
                idx_active_vrange = preprocess.filter_by_vrange(gain_2d)
            # Combine both filters
            if is_v05_like:
                # Pairwise per-ref comparison: gain(ref_i) vs vrange(ref_i)
                # - idx_active_gain_list[ref_i] is the gain-based mask for that ref
                # - idx_active_vrange_list[ref_i] is the vrange-based mask for that ref
                # Keep the per-ref combined masks for inspection/debugging.
                idx_active_gain_combined_list = [
                    idx_active_gain_list[ref_i] & idx_active_vrange_list[ref_i]
                    for ref_i in range(array_size)
                ]
                # Plot once per ref/vref slice.
                for ref_i in range(array_size):
                    # Diagnostics: how many pixels survive each filter for this ref?
                    n_gain_ok = int(np.sum(idx_active_gain_list[ref_i]))
                    n_vrange_ok = int(np.sum(idx_active_vrange_list[ref_i]))
                    n_both_ok = int(np.sum(idx_active_gain_combined_list[ref_i]))
                    if print_status:
                        print(
                            f"{n_a_type} gain filters (ref={ref_i}, vref={vref_vect[ref_i] if vref_vect is not None and ref_i < len(vref_vect) else None}): "
                            f"gain_ok={n_gain_ok}, vrange_ok={n_vrange_ok}, both_ok={n_both_ok}"
                        )

                    # For plotting, use the gain-only mask so you can visually compare with vrange/both counts above.
                    # If you want the strictest view, switch this to idx_active_gain_combined_list[ref_i].
                    idx_active_gain_2d = idx_active_gain_list[ref_i].reshape(nrows, ncols)
                    arr_gain_3d_filtered = arr_gain_4d[:, :, ref_i, :].copy()
                    # Set inactive pixels to NaN so they don't appear in the plot
                    arr_gain_3d_filtered[~idx_active_gain_2d, :] = np.nan
                    plot_gain_3d_plotly(
                        arr_gain_3d_filtered,
                        save_path=None,
                        experiment_name=f"{experiment_name} (vref_idx={ref_i})",
                        vref=vref_vect[ref_i],
                    )
            else:
                idx_active_gain_combined = idx_active_gain & idx_active_vrange
                # Apply the combined filter mask to arr_gain_3d: reshape to 2D and mask inactive pixels
                idx_active_gain_2d = idx_active_gain_combined.reshape(nrows, ncols)
                arr_gain_3d_filtered = arr_gain_3d.copy()
                # Set inactive pixels to NaN so they don't appear in the plot
                arr_gain_3d_filtered[~idx_active_gain_2d, :] = np.nan
                plot_gain_3d_plotly(arr_gain_3d_filtered, save_path=None, experiment_name=experiment_name, vref=vref)
        except Exception as e:
            if print_status:
                print(f"Warning: Could not plot filtered gain data with Plotly: {str(e)}")

    # LINEARISATION
    if is_v05_like:
        # Linearise per ref/vref slice:
        # - readout pixels: frame_4d[:, :, ref_i, :]
        # - gain calibration: arr_gain_4d[:, :, ref_i, :]   (typically 8 frames)
        # - scalar vref: vref_vect[ref_i]
        frame_4d_lin = np.zeros_like(frame_4d)
        idx_active_lin_list = []
        array_size = frame_4d.shape[2]
        for ref_i in range(array_size):
            vref_i = float(vref_vect[ref_i]) if vref_vect is not None and ref_i < len(vref_vect) else float(vref)
            params_i = linearise.generate_params(frame_4d[:, :, ref_i, :], arr_gain_4d[:, :, ref_i, :], vref_i)
            frame_lin_i, idx_lin_i = linearise.linearise(
                frame_4d[:, :, ref_i, :],
                params_i,
                idx_active_gain_list[ref_i],
            )
            frame_4d_lin[:, :, ref_i, :] = frame_lin_i
            idx_active_lin_list.append(idx_lin_i)

        # Active pixels used downstream: pick the selected ref slice (consistent with `frame_3d`).
        idx_active_lin = idx_active_lin_list[ref_idx_norm]
        frame_3d_lin = frame_4d_lin[:, :, ref_idx_norm, :]
    else:
        params = linearise.generate_params(frame_3d, arr_gain_3d, vref)
        frame_3d_lin, idx_active_lin = linearise.linearise(frame_3d, params, idx_active_gain)
    if is_v05_like:
        print(np.shape(frame_4d_lin))
    #  2) PREPROCESSING
    if print_status: print(f'Data loaded. \nPreprocessing start...')

    #  Split temperature and chemical data. Temperature pixels are inactive.
    arr_temp_3d, arr_chem_3d, idx_active_temp = preprocess.split_chem_and_temp(frame_3d, chip='titan')

    # The active pixels are those found active by lacewing (idx_active_input)
    # & are not temperature pixels (idx_active_temp)
    idx_active = (idx_active_input == 400) & idx_active_temp & idx_active_gain & idx_active_lin
    if print_status:
        try:
            print(
                f"Active pixel counts (ref_idx={ref_idx_norm}, vref={vref}): "
                f"lacewing={(idx_active_input == 400).sum()}, "
                f"non_temp={idx_active_temp.sum()}, "
                f"gain={idx_active_gain.sum()}, "
                f"lin={idx_active_lin.sum()}, "
                f"combined={idx_active.sum()}"
            )
        except Exception:
            pass

    # Split wells and create 2d arrays for each
    if n_wells == 1:
        all_well_3d, all_well_temp_3d = [arr_chem_3d], [arr_temp_3d]
        all_well_gain_3d = [arr_gain_3d]
        all_idx_active = [idx_active]
        all_well_3d_lin = [frame_3d_lin]
    elif n_wells == 2 or n_wells == 4 or n_wells == 6 or n_wells == 10:
        all_well_3d = preprocess.split_wells(arr_chem_3d, n_wells)  # 1.CHEM DATA
        all_well_temp_3d = preprocess.split_wells(arr_temp_3d, n_wells)  # 2.TEMP DATA
        all_well_gain_3d = preprocess.split_wells(arr_gain_3d, n_wells)  # 3.GAIN DATA
        all_idx_active = preprocess.split_wells_idxactive(idx_active, n_wells)  # 4.ACTIVE PIXELS
        all_well_3d_lin = preprocess.split_wells(frame_3d_lin, n_wells)  # 5.LINEARISED 3D CHEM DATA

    else:
        raise f"RAISED: n_wells = {n_wells} is not a valid input :( Choose 1, 2, 4, 6 or 10"

    # Find beginning and end of the experiment.
    #
    # For pre-v06 firmware, every chip pixel was streamed every frame, so we could
    # extract the temperature trace by spatially slicing the readout (positions where
    # row%5==2 && col%5==2) and linearising those raw ADC values with the (a,b,c,d)
    # coefficients below.
    #
    # For v06 firmware (Lacewing_Integrated_new_temp_read), TTN_Readout_Time() skips
    # temperature pixels entirely (Lacewing_Thread.py:3103). Their positions in the
    # streamed pixel block are hard-zero, so the spatial extraction gives an
    # all-NaN trace and crashes find_start_end_temp. Instead, the per-frame
    # parameter tail carries `temp_avg_Vs_shifted = tem_lin + 100`, which the
    # readout parser already exposes via `meta` (titan/load_functions.py).
    a = 1913.44
    b = 26.68
    c = np.min(arr_temp_3d) - 0.1
    d = 0.76

    use_meta_temp = bool(
        n_a_lower == "v06"
        and isinstance(readout_meta, dict)
        and "temp_avg_Vs_shifted" in readout_meta
    )

    if use_meta_temp:
        # `temp_avg_Vs_shifted` is the firmware-linearised temperature with a fixed +100 offset
        # (Lacewing_Thread.py:2876). Reverse the offset to recover degrees-C-equivalent units
        # consistent with the legacy spatial pipeline.
        temp_1d = np.asarray(readout_meta["temp_avg_Vs_shifted"], dtype=float) - 100.0
        if print_status:
            n_finite = int(np.sum(np.isfinite(temp_1d)))
            print(
                f"v06 temperature trace from readout meta: n={temp_1d.size}, "
                f"finite={n_finite}, "
                f"min={np.nanmin(temp_1d) if n_finite else float('nan'):.3g}, "
                f"max={np.nanmax(temp_1d) if n_finite else float('nan'):.3g}"
            )
    else:
        temp_2dNL = arr_temp_3d.reshape(-1, arr_temp_3d.shape[2]).T
        new = -(1 / b) * np.log((arr_temp_3d - c) / a) + d
        temp_2d = new.reshape(-1, new.shape[2]).T
        idx_active_temp_local = (np.mean(temp_2dNL, axis=0) > 10)
        if not np.any(idx_active_temp_local):
            # Defensive fallback: no temperature pixels passed the >10 sanity gate.
            # Either firmware no longer streams them (use n_a_type="v06") or the
            # capture has no usable thermal data. Emit an empty trace so downstream
            # detection raises a clear error rather than an obscure IndexError.
            if print_status:
                print(
                    "Warning: spatial temperature extraction found 0 active temp pixels "
                    "(firmware may be skipping them; consider n_a_type='v06')."
                )
            temp_1d = np.full(arr_temp_3d.shape[2], np.nan, dtype=float)
        else:
            temp_1d = np.mean(temp_2d[:, idx_active_temp_local], axis=1)

    idx_start, idx_settled, idx_end = preprocess.find_start_end_idx(time_vect,
                                                                    np.mean(arr_chem_3d.reshape(-1, arr_chem_3d.shape[2], order='C').T, axis=1),
                                                                    temp_1d, end_time_min, start_type)

    # # DEBUG PLOT FOR LINEARISATION
    # fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    # frame_2d_bs = frame_3d_lin.reshape(-1, frame_3d_lin.shape[2], order='C').T - frame_3d_lin.reshape(-1, frame_3d_lin.shape[2], order='C').T[0, :]
    # ax[0].plot(np.mean(frame_2d_bs, axis=1)[idx_settled:])
    # ax[0].set(title="New linearisarion", xlabel="sample", ylabel="amplitude")
    #
    # arr_tmp = arr_chem_3d[:, :, idx_settled:]
    # a = 24328.12
    # b = -0.010848
    # c = np.min(arr_tmp)-0.1
    # new = 1 / b * np.log((arr_tmp - c) / a)
    # temp_2d = new.reshape(-1, new.shape[2]).T
    # temp_1d = np.mean(temp_2d, axis=1)
    # ax[1].plot(temp_1d)
    # ax[1].set(title="Old linearisarion", xlabel="sample", ylabel="amplitude")
    # plt.tight_layout()
    # plt.show()
    # # DEBUG END

    # For v06 we have a single per-frame temperature trace (from readout meta tail) rather than
    # per-pixel temperature data. Share that trace with every Well so `well_temp_mean_then_lin`
    # can return a real (non-NaN) baseline-subtracted temperature plot.
    well_temp_1d_v06 = None
    if use_meta_temp:
        well_temp_1d_v06 = np.asarray(temp_1d, dtype=float)

    # Save relevant data for each well
    all_well_summary = []
    for i_well in range(n_wells):
        # This is required to find active pixels in the correct section of the experiment
        n_times = all_well_3d[i_well][:, :, idx_settled:idx_end].shape[2]
        well_2d_exp = all_well_3d[i_well][:, :, idx_settled:idx_end].reshape(-1, n_times, order='C').T
        # Filter active pixels by Vrange: Each pixel needs to be in range for the entire duration of the experiment
        #idx_active_range = preprocess.filter_by_vrange(well_2d_exp) & preprocess.filter_by_derivative(well_2d_exp)
        idx_active_range = preprocess.filter_by_vrange(well_2d_exp)
        all_idx_active[i_well] = all_idx_active[i_well] & idx_active_range

        # Initialise well instance
        this_well = Well(time_vect, all_well_3d[i_well], all_well_temp_3d[i_well], all_well_gain_3d[i_well], all_well_3d_lin[i_well], idx_start, idx_settled, idx_end, all_idx_active[i_well])
        if well_temp_1d_v06 is not None:
            this_well.well_temp_1d_v06 = well_temp_1d_v06
        all_well_summary.append(this_well)

    # Initialise Experiment instance
    experiment = Experiment(all_well_summary, exp_path_str=str(exp_path_readout))
    experiment.temperature_3d = arr_temp_3d  # OPTIONAL: saving temperature 3d array
    # For v06, also expose the global per-frame temperature trace (already baseline-shift-friendly)
    # so any caller that wants a chip-level view (rather than a per-well property) has it.
    if well_temp_1d_v06 is not None:
        experiment.temperature_1d_v06 = well_temp_1d_v06

        # Use ``*_temp_log.bin`` as the single source of truth for the temperature trace.
        # The firmware keeps appending to ``heat_time_stamp`` / ``ttn_tem_data_lin`` for
        # BOTH the pre-settling heating phase AND the post-settling readout phase
        # (Lacewing_UI_v05/Lacewing_Thread.py: every iteration of the readout loop also
        # runs the heating block and pushes a new sample). So temp_log.bin already
        # carries the full cold-start -> ramp -> settled-readout curve in a single time
        # base anchored at heating t=0.
        #
        # The per-frame ``temp_avg_Vs_shifted`` we already pulled from the readout meta
        # tail is the same temperature signal sampled only during the readout segment,
        # so concatenating it onto temp_log.bin would double-plot the post-settling
        # portion. Prefer temp_log.bin; fall back to the readout-only trace if the
        # log file is missing/malformed.
        try:
            full_t_s, full_temp_lin = load.load_temp_log(exp_path)
        except Exception as e:
            full_t_s, full_temp_lin = None, None
            if print_status:
                print(f"Warning: load_temp_log failed: {e}")

        if (
            full_t_s is not None
            and full_temp_lin is not None
            and full_t_s.size > 0
            and full_temp_lin.size == full_t_s.size
        ):
            merged_time_s = full_t_s.astype(float)
            merged_temp = full_temp_lin.astype(float)
            full_n = int(full_t_s.size)
            if print_status:
                print(
                    f"v06 temperature: using temp_log.bin ({full_n} samples, "
                    f"0..{merged_time_s[-1]:.1f}s = {merged_time_s[-1] / 60.0:.2f} min)"
                )
        else:
            readout_t_s = np.asarray(time_vect, dtype=float)
            if readout_t_s.size > 0:
                merged_time_s = readout_t_s - float(readout_t_s[0])
            else:
                merged_time_s = readout_t_s
            merged_temp = well_temp_1d_v06.astype(float)
            if print_status:
                print(
                    "v06 temperature: no usable temp_log.bin found, plotting readout phase only "
                    f"({int(merged_temp.size)} samples)."
                )

        experiment.temperature_time_s_v06_with_ramp = merged_time_s
        experiment.temperature_1d_v06_with_ramp = merged_temp
        # heat_n is preserved as a tag for the plotting code, but with temp_log.bin
        # carrying the full timeline there is no separate "readout-only" segment to
        # colour differently any more.
        experiment.temperature_v06_heat_n = 0
        experiment.temperature_v06_readout_n = int(merged_temp.size)

    if print_status: print('Preprocessing end.')
    return experiment


