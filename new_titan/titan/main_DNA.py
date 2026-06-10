import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter issues
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
import pandas as pd
from scipy.signal import lfilter
import os
import importlib.util


from plt_Experiment_summary import titan_plt_summary, titan_plt_summary_means, titan_plt_summary_temp
from load_and_preprocessing import titan_load_and_preprocessing
import load_functions as load
from peak_detection import calculate_well_derivatives, plot_well_derivatives, detect_derivative_peaks, calculate_well_second_derivatives, plot_well_second_derivatives
from plot_derivatives_plotly import plot_wells_grid_plotly, plot_all_wells_combined_plotly, plot_wells_active_pixels_plotly, plot_wells_temperature_plotly, plot_wells_temperature_mean_then_lin_plotly, plot_wells_temperature_mean_then_lin_global_plotly, plot_wells_first_derivative_grid_plotly, plot_wells_second_derivative_grid_plotly, plot_wells_linearized_signal_combined_plotly, plot_wells_linearized_signal_combined_savgol_plotly, plot_wells_average_combined_plotly, plot_all__wells_Combined_plotly, plot_wells_raw_plotly, plot_raw_chip_video_plotly, plot_dvref_selection_model_plotly, sanitize_filename
from export_raw_chip_video import export_raw_chip_video
from plot_find_active_pixels import MANIFOLD_WELL_LAYOUT, show_find_active_for_experiment
from well_outlines import get_well_outlines, get_inactive_well_outlines

if __name__ == '__main__':
    #region agent log
    # Minimal "run marker" to prove this exact file is being executed and where it lives.
    # (We can't reliably read hidden `.cursor/` logs from tools, so we also write a visible marker file.)
    try:
        import json as _agent_json0
        import time as _agent_time0
        from pathlib import Path as _agent_Path0
        _file0 = _agent_Path0(__file__).resolve()
        _repo_root0 = _file0.parents[1]
        _cwd0 = _agent_Path0.cwd()
        _payload0 = {
            "sessionId": "debug-session",
            "runId": "post-fix",
            "hypothesisId": "Z",
            "location": "titan/main_DNA.py:__main__",
            "message": "main_DNA.py started",
            "data": {"__file__": str(_file0), "repo_root": str(_repo_root0), "cwd": str(_cwd0)},
            "timestamp": int(_agent_time0.time() * 1000),
        }

        _candidates0 = [
            _repo_root0 / "agent_run_marker.ndjson",
            _repo_root0 / "titan" / "agent_run_marker.ndjson",
            _cwd0 / "agent_run_marker.ndjson",
        ]

        _written0 = False
        _last_exc0 = None
        for _p0 in _candidates0:
            try:
                with open(str(_p0), "a", encoding="utf-8") as _f0:
                    _f0.write(_agent_json0.dumps(_payload0, ensure_ascii=False) + "\n")
                print(f"[agent-debug] Wrote run marker: {_p0}")
                _written0 = True
                break
            except Exception as _e0:
                _last_exc0 = _e0
                continue
        if not _written0:
            print(f"[agent-debug] Could not write agent_run_marker.ndjson. Tried: {[str(p) for p in _candidates0]}. Last error: {_last_exc0}", file=__import__('sys').stderr)
    except Exception as _e0:
        try:
            print(f"[agent-debug] Could not write agent_run_marker.ndjson: {_e0}", file=__import__("sys").stderr)
        except Exception:
            pass
    #endregion agent log

    # ===== PEAK DETECTION THRESHOLD CONFIGURATION =====
    # Set experiment-specific peak detection thresholds
    # 
    # HOW TO USE:
    # 1. The experiment name is extracted from the folder path after "KHz_U_"
    #    Example: "D20251013_E00_C00_F4500KHz_U_Sample_19_repeat" -> "Sample_19_repeat"
    # 2. Add an entry to PEAK_THRESHOLD_CONFIG with the experiment name as the key
    # 3. Set the threshold value (typical range: 0.00001 to 0.001)
    #    - LOWER values = more sensitive (detect smaller/weaker peaks)
    #    - HIGHER values = less sensitive (only detect larger/stronger peaks)
    # 4. If an experiment is not in the dictionary, DEFAULT_PEAK_THRESHOLD is used
    # 
    # The threshold is printed during processing to help you verify which value is being used
    
    PEAK_THRESHOLD_CONFIG = {
        'Sample_19_repeat': 0.0001,     # Example: specific threshold for this experiment
        # Add more experiments here as needed:
        # 'demo_pipette': 0.0002,
        # 'another_experiment': 0.00015,
    }
    
    # Default threshold for experiments not specified above
    DEFAULT_PEAK_THRESHOLD = 0.0001
    
    # ===== PLOT CONFIGURATION =====
    # Set to True/False to enable/disable each type of plot
    # 
    # To use a different configuration, uncomment one of the lines below and comment out PLOT_CONFIG:
    # PLOT_CONFIG = {'raw_signal_combined': True, 'raw_signal_grid': True, 'linearized_signal_combined': True, 'first_derivative_grid': True, 'second_derivative_grid': True, 'active_pixels': True, 'temperature': True}  # All plots
    # PLOT_CONFIG = {'raw_signal_combined': True, 'raw_signal_grid': True, 'linearized_signal_combined': True, 'first_derivative_grid': True, 'second_derivative_grid': True, 'active_pixels': False, 'temperature': False}  # Signals and derivatives only
    # PLOT_CONFIG = {'raw_signal_combined': False, 'raw_signal_grid': False, 'linearized_signal_combined': False, 'first_derivative_grid': False, 'second_derivative_grid': False, 'active_pixels': True, 'temperature': False}  # Active pixels only
    # PLOT_CONFIG = {'raw_signal_combined': False, 'raw_signal_grid': False, 'linearized_signal_combined': False, 'first_derivative_grid': False, 'second_derivative_grid': False, 'active_pixels': False, 'temperature': True}  # Temperature only
    # PLOT_CONFIG = {'raw_signal_combined': False, 'raw_signal_grid': True, 'linearized_signal_combined': True, 'first_derivative_grid': True, 'second_derivative_grid': True, 'active_pixels': False, 'temperature': False}  # Grid and combined plots only
    
    PLOT_CONFIG = {
        'raw_signal_combined': True,      # Combined plot showing all wells together
        'raw_signal_grid': True,          # Grid: baseline-subtracted active-pixel mean (idx_settled:idx_end)
        'raw_chem_grid': True,            # Same 5x2 layout, raw chem mean (no baseline subtraction)
        'linearized_signal_combined': True,   # Combined plot showing linearized signals for all wells together
        'first_derivative_grid': True,    # Grid plot showing first derivatives for each well (5x2)
        'second_derivative_grid': True,   # Grid plot showing second derivatives for each well (5x2)
        'active_pixels': True,            # Active pixels heatmap (full chip view)
        'temperature': True,             # Temperature overlay plot
        'gain_3d_filtered': True,          # Filtered gain 3D plot (from load_and_preprocessing)
        'raw_chem_plotly': True,           # Raw chem data plot (plotly) - individual pixel traces
        'linearized_chem_plotly': True,    # Same grid as raw_chem_plotly: per-pixel well_2d_bs (active only)
        'raw_chem_export_all_pixels': True,  # Export raw chem (all pixels) to CSV per well across all vrefs
        'average_combined': False,           # Average (unfiltered) combined plot for all wells
        'raw_chip_video': True,             # Plotly animation: full-chip raw heatmap over time (HTML)
        'raw_chip_video_mp4': True,        # Also export MP4 (needs imageio / ffmpeg); uses same downsampling as HTML
        # dVref multi: coarse sweep + Δ/ΔΔ + L_ttn.c selection model (needs *_dvref_multi_payload.bin telemetry)
        'dvref_selection_model': True,
    }

    # Optional visibility controls for average combined plot (1-based well indices)
    AVERAGE_COMBINED_VISIBILITY = {
        'wells_to_show': None,      # e.g. [1, 2, 5] to show only these wells initially
        'wells_to_hide': None,      # e.g. [3, 4] to start these wells hidden
        'hide_mode': 'legendonly'   # 'legendonly' keeps legend toggle, 'skip' omits traces
    }
    
    LINEARIZED_COMBINED_VISIBILITY = {
        'wells_to_show': None,
        'wells_to_hide': None,
        'hide_mode': 'legendonly'
    }
    
    # Print current configuration
    print("="*60)
    print("PLOT CONFIGURATION")
    print("="*60)
    for plot_type, enabled in PLOT_CONFIG.items():
        status = "ENABLED" if enabled else "DISABLED"
        print(f"{plot_type.replace('_', ' ').title()}: {status}")
    print("="*60)
    print()

    # Load change-point detector from the requested standalone file:
    # `titan/ttp calculation .py`
    change_point_detection_fn = None
    try:
        _cp_path = Path(__file__).resolve().parent / "ttp calculation .py"
        _cp_spec = importlib.util.spec_from_file_location("ttp_calculation_with_space", str(_cp_path))
        if _cp_spec is not None and _cp_spec.loader is not None:
            _cp_module = importlib.util.module_from_spec(_cp_spec)
            _cp_spec.loader.exec_module(_cp_module)
            change_point_detection_fn = getattr(_cp_module, "change_point_detection", None)
        if change_point_detection_fn is None:
            print(f"Warning: change_point_detection not found in {_cp_path}")
    except Exception as e:
        print(f"Warning: Could not load change-point module: {e}")

    # ##########  A. IF THE FILES ARE IN THE EXCEL ##########
    # onedrive_path = Path("..", "..", "..", "..", "..", "Costanza", "OneDrive - Imperial College London")  # CHANGE THE NAME OF THE PATH HERE
    # exp_folder = Path(onedrive_path, "Master Data Folder", "Matthew Run Data")  # CHANGE THE NAME OF THE DATA FOLDER HERE
    #
    # excel_path = Path(onedrive_path, "Master Data Folder", "Run Tracker.xlsx")  # CHANGE PATH OF THE EXCEL SUMMARY HERE
    # excel_df = pd.read_excel(excel_path, sheet_name="Master Run")  # CHANGE THE NAME OF THE EXCEL SHEET NAME HERE
    #
    # # exp_paths = [f for f in exp_folder.glob('*') if f.is_dir()]  # USE THIS TO RUN ALL EXPERIMENTS IN A FOLDER
    # exp_paths = [Path(exp_folder, "D20250409_E00_C00_F4500KHz_U_demo_pipette")]  # OR THIS TO RUN ONE EXPERIMENT
    #
    # print(f"DEBUG: EXP_PATHS")
    # for i_path in range(len(exp_paths)):
    #     print(f"i_path {i_path}, path {exp_paths[i_path]}")
    #
    # for i_path, exp_path in enumerate(exp_paths):
    #     path_readout_str = str(exp_path)
    #     exp_id = path_readout_str[path_readout_str.rfind('D'):]
    #     print(f"\n-------\nDEBUG: RUN N {i_path} -- EXP_ID {exp_id}")
    #
    #     exp_row = excel_df[excel_df["File Name"] == exp_id]
    #     if exp_row["No. Of Wells"].size == 0:
    #         raise "Experiment not in excel"
    #     if exp_row["No. Of Wells"].size > 1:
    #         raise "More than one line in Excel corresponding to this experiment"
    #     n_wells = int(exp_row["No. Of Wells"])
    #     n_a_type = np.array(exp_row["Version No."])[0]
    #     print(f"DEBUG: RUN N {i_path} -- NWELLS {n_wells} -- N_A_TYPE {n_a_type}")
    #
    #     exp = titan_load_and_preprocessing(exp_path, n_wells=n_wells, start_type="temperature",
    #                                        end_time_min=60, n_a_type=n_a_type,
    #                                        print_status=True, plt_gain_calib=True, save_gain_calib=True)
    #
    #     times = exp.wells_list[0].time_min
    #     max_time = times[-1]
    #     idx30 = functions.time_to_index([30], times)[0]
    #     print(f"------------------------------->>>> NTIMES {times.shape} --- MAX TIME {max_time} --- IDX30MIN {idx30}")
    #
    #     titan_plt_summary(exp, exp_path, plt_save=True)
    #     titan_plt_summary_means(exp, exp_path, plt_save=True)
    #     titan_plt_infl(exp, exp_path, plt_show=True, plt_save=True)


    ##########  B. IF THE FILES ARE NOT IN THE EXCEL  ##########
    n_wells = 10
    n_a_type = "v06"  # SPECIFY NUMBER OF WELLS AND VERSION HERE
    # v06 reads the same on-disk binary as v05 (Multi_Vref / Lacewing_Integrated_new_temp_read firmware
    # version=5), but pulls per-frame temperature from the readout payload tail
    # (`temp_avg_Vs_shifted`) instead of from chip pixel positions, which the firmware no longer
    # streams. Use "v05" for legacy datasets that still expose temperature pixels in the stream.
    # For v05/v06 (Multi_Vref) datasets, the loader contains multiple Vref "ref" slices per timestamp.
    # Downstream analysis/plots currently run on a single selected slice.
    # - Use -1 for the last Vref slice (legacy behavior)
    # - Use 0..(n_vrefs-1) to pick a specific slice
    # - Use "all" to loop over *all* Vref slices and save one set of plots per slice
    VREF_REF_IDX = "all"
    # When looping over many Vref slices, avoid spamming interactive windows.
    # Plots will still be saved to HTML in `save_path`.
    MULTI_VREF_SHOW_PLOTS = True    
    MULTI_VREF_AUTO_OPEN_HTML = True
    # Prefer an absolute OneDrive path so the script works regardless of current working directory.
    # (The previous relative path depended on running from a specific folder.)
    # onedrive_path = Path.home() / "OneDrive - ProtonDx"
    # exp_folder = Path(onedrive_path,"Data","Multi_Vref")  # CHANGE THE NAME OF THE DATA FOLDER HERE
    exp_folder = Path("/vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_multi")  # OR THIS TO RUN ON TEST DATA (IGNORES ONE DRIVE PATH)
    # exp_folder = Path(onedrive_path, "Master Data Folder", "Lacewing - 25_Zambia_Malaria")

    #exp_paths = [f for f in exp_folder.glob('*') if f.is_dir()]  # USE THIS TO RUN ALL EXPERIMENTS IN A FOLDER
    # exp_paths = [Path(exp_folder, "D20260320_E00_C00_F4500KHz_U_Elena_steap_cv")]  # OR THIS TO RUN ONE EXPERIMENT
    # exp_paths = [Path(exp_folder, "D20260522_E00_C00_F4500KHz_U_manifold_test_05")]  # OR THIS TO RUN ONE EXPERIMENT
    # exp_paths = [Path(exp_folder, "D20260608_E00_C00_F4500KHz_U_norm_temp_04")]  # OR THIS TO RUN ONE EXPERIMENT
    # exp_paths = [Path(exp_folder, "D20260609_E00_C00_F4500KHz_U_norm_temp_read_06")]  # OR THIS TO RUN ONE EXPERIMENT
    exp_paths = [Path(exp_folder, "D20260609_E00_C00_F4500KHz_U_norm_temp_ready_08")]  # OR THIS TO RUN ONE EXPERIMENT
    
    # Optional: overlay several runs on one plot (t=0 at vref jump / idx_settled).
    # Set enabled=True and list folders in overlay_paths (or None to reuse exp_paths).
    OVERLAY_CONFIG = {
        "enabled": False,
        "overlay_paths": None,
        "signal": "linearized_mean",
        "wells": [1],
        "ref_idx": 0,
        "output_name": "manifold_overlay_vref_jump",
    }

    print(f"DEBUG: EXP_PATHS")
    for i_path in range(len(exp_paths)):
        print(f"i_path {i_path}, path {exp_paths[i_path]}")

    for i_path, exp_path in enumerate(exp_paths):
        dvref_tele = None  # set when v05/v06 load_vref_sweep returns telemetry
        path_readout_str = str(exp_path)
        print(f"\n-------\nDEBUG: RUN N {i_path} -- NWELLS {n_wells} -- N_A_TYPE {n_a_type}")
        
        # Extract experiment name from path (part after "KHz_U_")
        path_str = str(exp_path)
        if "KHz_U_" in path_str:
            experiment_name = path_str.split("KHz_U_")[1]
            # Remove any trailing path separators
            experiment_name = experiment_name.rstrip("\\").rstrip("/")
        else:
            print(f"Skipping {exp_path.name} - not a valid experiment directory (no 'KHz_U_' found)")
            continue
        
        print(f"Experiment name: {experiment_name}")
        
        # Get experiment-specific peak threshold or use default
        peak_threshold = PEAK_THRESHOLD_CONFIG.get(experiment_name, DEFAULT_PEAK_THRESHOLD)
        print(f"Peak detection threshold for this experiment: {peak_threshold}")
        
        # Check if this directory contains readout files before processing
        readout_files = list(exp_path.glob("*readout*.bin"))
        #region agent log
        # Visible evidence for why we might skip before reaching the plotly raw-chem block
        try:
            import json as _agent_json_ro
            import time as _agent_time_ro
            from pathlib import Path as _agent_Path_ro
            _repo_root_ro = _agent_Path_ro(__file__).resolve().parents[1]
            _dbg_path_ro = _repo_root_ro / "debug_visible.ndjson"
            _bin_files_ro = sorted([p.name for p in exp_path.glob("*.bin")])
            _ro_files_ro = sorted([p.name for p in readout_files])
            with open(str(_dbg_path_ro), "a", encoding="utf-8") as _f_ro:
                _f_ro.write(_agent_json_ro.dumps({
                    "sessionId": "debug-session",
                    "runId": "pre-fix",
                    "hypothesisId": "H9",
                    "location": "titan/main_DNA.py:readout_check",
                    "message": "Readout file discovery",
                    "data": {
                        "exp_path": str(exp_path),
                        "readout_glob": "*readout*.bin",
                        "readout_count": len(readout_files),
                        "readout_names_head": _ro_files_ro[:10],
                        "bin_count": len(_bin_files_ro),
                        "bin_names_head": _bin_files_ro[:10],
                    },
                    "timestamp": int(_agent_time_ro.time() * 1000),
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass
        #endregion agent log
        if not readout_files:
            find_active_files = list(exp_path.glob("*find_active*.bin"))
            if find_active_files and PLOT_CONFIG.get("active_pixels", False):
                print(
                    f"No readout in {exp_path.name}; plotting active pixels from "
                    f"{len(find_active_files)} find_active file(s)..."
                )
                save_path_find_active = '/vol/bitbucket/gk225/multi_viz'
                show_find_active_for_experiment(
                    exp_path,
                    save_path=save_path_find_active,
                    experiment_name=experiment_name,
                    show=MULTI_VREF_SHOW_PLOTS,
                    well_layout=MANIFOLD_WELL_LAYOUT,
                )
            else:
                if not find_active_files:
                    print(f"Skipping {exp_path.name} - no readout or find_active files found")
                else:
                    print(
                        f"Skipping {exp_path.name} - no readout files "
                        f"(enable PLOT_CONFIG['active_pixels'] to plot find_active.bin only)"
                    )
            continue

        print(f"Found {len(readout_files)} readout files, processing...")
        
        # Define save path
        save_path = r"/vol/bitbucket/gk225/multi_viz"

        # Export underlying data for the Plotly `raw_signal_grid` trace across all vref slices.
        raw_signal_grid_csv_path = Path(save_path) / f"{experiment_name}_raw_signal_grid_all_vrefs.csv"
        # Overwrite to avoid duplicating rows when re-running the script.
        try:
            if raw_signal_grid_csv_path.exists():
                raw_signal_grid_csv_path.unlink()
        except Exception:
            pass
        raw_signal_grid_csv_written_header = False

        # Export underlying data for `linearized_signal_combined` to Excel.
        linearized_combined_export_frames = []

        # Export full raw-chem data (all pixels) to CSV, one file per well, across all vrefs.
        raw_chem_export_paths = {}
        raw_chem_export_written_header = {}
        if PLOT_CONFIG.get('raw_chem_export_all_pixels', False):
            save_path_obj = Path(save_path)
            save_path_obj.mkdir(parents=True, exist_ok=True)
            for well_number in range(1, n_wells + 1):
                p = save_path_obj / f"{experiment_name}_raw_chem_all_vrefs_well_{well_number}.csv"
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass
                raw_chem_export_paths[well_number] = p
                raw_chem_export_written_header[well_number] = False

        # Resolve which ref/vref slices to run.
        vref_vect = None
        ref_indices = []
        is_multi_vref = False
        if str(n_a_type).lower() in ("v05", "v06"):
            want_all = (VREF_REF_IDX is None) or (isinstance(VREF_REF_IDX, str) and VREF_REF_IDX.lower() == "all")
            try:
                exp_path_vref = load.find_most_recent_vref(exp_path)
                n_vrefs, vref_vect, dvref_tele = load.load_vref_sweep(exp_path, return_telemetry=True)
                if dvref_tele is not None and dvref_tele.get("coarse") is not None:
                    print(
                        f"Loaded dvref multi payload from {exp_path_vref.name}: "
                        f"n_vrefs={n_vrefs}, vrefs={dvref_tele['vrefs']}"
                    )
                    if dvref_tele.get("model") is not None:
                        m = dvref_tele["model"]
                        print(
                            "  TTN model: pass_fail=%s total_dcnt_x16=%s threshold=%s"
                            % (m["pass_fail"], m["total_dcnt_x16"], m["coarse_dcnt_x16_threshold"])
                        )
                else:
                    print(f"Loaded vref sweep from {exp_path_vref.name}: n_vrefs={n_vrefs}")
                if want_all and n_vrefs:
                    ref_indices = list(range(int(n_vrefs)))
                else:
                    # Support scalar or list/tuple of indices
                    if isinstance(VREF_REF_IDX, (list, tuple, np.ndarray)):
                        ref_indices = [int(x) for x in VREF_REF_IDX]
                    else:
                        ref_indices = [int(VREF_REF_IDX)]
            except Exception as e:
                # If we can't read the vref sweep, fall back to legacy single-slice behavior.
                print(f"Warning: could not load vref sweep for {exp_path.name}: {e}")
                ref_indices = [-1 if want_all else int(VREF_REF_IDX)]
                vref_vect = None
                dvref_tele = None

            is_multi_vref = len(ref_indices) > 1
        else:
            # Non-v05 datasets do not have multiple Vref slices.
            ref_indices = [VREF_REF_IDX]

        # Collect full-chip active-pixel masks so we can build one overlay across all VREFs.
        active_pixels_overlay_data = []

        for _ref_idx in ref_indices:
            # Create a stable label for filenames/titles.
            slice_label = None
            vref_value = None
            ref_idx_norm = None
            try:
                ref_idx_norm = int(_ref_idx)
            except Exception:
                ref_idx_norm = -1

            if vref_vect is not None and len(vref_vect) > 0:
                # Normalize negative indices like Python does.
                if ref_idx_norm < 0:
                    ref_idx_norm = len(vref_vect) + ref_idx_norm
                if 0 <= ref_idx_norm < len(vref_vect):
                    try:
                        vref_value = float(vref_vect[ref_idx_norm])
                    except Exception:
                        vref_value = None

            if str(n_a_type).lower() in ("v05", "v06"):
                if vref_value is not None:
                    slice_label = f"vref_idx={ref_idx_norm}_vref={vref_value:g}"
                else:
                    slice_label = f"vref_idx={ref_idx_norm}"
            experiment_name_slice = experiment_name
            if slice_label is not None and (is_multi_vref or (VREF_REF_IDX not in (-1, "-1"))):
                experiment_name_slice = f"{experiment_name}__{slice_label}"

            print(f"\n--- Processing slice: {experiment_name_slice} ---")

            try:
                exp = titan_load_and_preprocessing(
                    exp_path,
                    n_wells=n_wells,
                    start_type="temperature",
                    end_time_min=60,
                    n_a_type=n_a_type,
                    print_status=True,
                    plt_gain_calib=True,
                    save_gain_calib=True,
                    plot_gain_3d=PLOT_CONFIG.get('gain_3d_filtered', False),
                    ref_idx=int(_ref_idx) if str(n_a_type).lower() in ("v05", "v06") else _ref_idx,
                )
            except Exception as e:
                print(f"Error processing {exp_path.name} (ref_idx={_ref_idx}): {str(e)}")
                print("Skipping this ref slice and continuing...")
                continue

            # Export raw chem data (all pixels) for this ref slice.
            if PLOT_CONFIG.get('raw_chem_export_all_pixels', False):
                print("  - Exporting raw chem all-pixels CSVs (per well)...")
                try:
                    vref_idx_to_write = ref_idx_norm if ref_idx_norm is not None else np.nan
                    vref_value_to_write = vref_value if vref_value is not None else np.nan

                    for well_idx, well in enumerate(exp.wells_list):
                        well_number = well_idx + 1
                        time_npr = getattr(well, "time_npr", None)
                        data_2d = getattr(well, "well_2d_npr", None)
                        idx_start = int(getattr(well, "idx_start", 0))
                        idx_settled = int(getattr(well, "idx_settled", 0))
                        idx_end = int(getattr(well, "idx_end", 0))

                        if time_npr is None or data_2d is None:
                            continue
                        time_npr = np.asarray(time_npr)
                        data_2d = np.asarray(data_2d)
                        if time_npr.size == 0 or data_2d.size == 0:
                            continue

                        # Ensure shape is (n_time, n_pixels)
                        if data_2d.ndim != 2:
                            continue
                        if data_2d.shape[0] != len(time_npr) and data_2d.shape[1] == len(time_npr):
                            data_2d = data_2d.T
                        if data_2d.shape[0] != len(time_npr):
                            continue

                        # Match raw-chem plotting convention: from idx_settled onward.
                        if idx_settled < 0 or idx_settled >= len(time_npr):
                            idx_settled = 0
                        time_slice = time_npr[idx_settled:]
                        data_slice = data_2d[idx_settled:, :]
                        if time_slice.size == 0 or data_slice.size == 0:
                            continue

                        n_time, n_pixels = data_slice.shape
                        export_dict = {
                            "vref_idx": np.full(n_time, vref_idx_to_write),
                            "vref_value": np.full(n_time, vref_value_to_write),
                            "well": np.full(n_time, well_number),
                            "idx_start": np.full(n_time, idx_start),
                            "idx_settled": np.full(n_time, idx_settled),
                            "idx_end": np.full(n_time, idx_end),
                            "time_s": time_slice,
                            "time_min": (time_slice - time_slice[0]) / 60.0,
                        }
                        for pix in range(n_pixels):
                            export_dict[f"pixel_{pix}"] = data_slice[:, pix]

                        df_well = pd.DataFrame(export_dict)
                        csv_path = raw_chem_export_paths.get(well_number)
                        if csv_path is None:
                            continue
                        df_well.to_csv(
                            csv_path,
                            mode="a",
                            header=not raw_chem_export_written_header[well_number],
                            index=False,
                        )
                        raw_chem_export_written_header[well_number] = True
                except Exception as e:
                    print(f"    Warning: Could not export raw chem all-pixels CSVs: {e}")

            #titan_plt_summary(exp, exp_path, plt_save=True)
            #titan_plt_summary_temp(exp, exp_path, plt_save=False)
            #titan_plt_summary_means(exp, exp_path, plt_save=True)
            #titan_plt_infl(exp, exp_path, plt_show=True, plt_save=True)
            
            try:
                # Compute well outlines (chip coordinates)
                try:
                    outlines = get_well_outlines(exp, n_wells=None)
                    print("Well outlines (col,row polygons):")
                    for o in outlines:
                        print(f"  Well {o.well_index + 1}: {o.polygon}")
                except Exception as e:
                    print(f"Warning: could not compute well outlines: {e}")
                # Compute inactive-pixel bounding boxes
                try:
                    inactive_outlines = get_inactive_well_outlines(exp, n_wells=None)
                    print("Inactive-pixel outlines (col,row polygons):")
                    for o in inactive_outlines:
                        print(f"  Well {o.well_index + 1}: {o.polygon}")
                except Exception as e:
                    print(f"Warning: could not compute inactive outlines: {e}")

                # Plot raw chem data (plotly) matching plt_Experiment_summary raw view
                if PLOT_CONFIG['raw_chem_plotly']:
                    print("  - Creating raw chem plot (plotly)...")
                    try:
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
                                with open(r"/vol/bitbucket/gk225/multi_viz/debug.log", "a", encoding="utf-8") as f:
                                    f.write(_agent_json.dumps(payload, ensure_ascii=False) + "\n")
                            except Exception:
                                pass
                        def _agent_log_visible(payload):
                            try:
                                # Try a couple of workspace-visible paths regardless of current working directory
                                candidates = []
                                try:
                                    repo_root = _agent_Path(__file__).resolve().parents[1]
                                    candidates.append(repo_root / "debug_visible.ndjson")
                                    candidates.append(repo_root / "titan" / "debug_visible.ndjson")
                                except Exception:
                                    # Fallback to relative paths
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
                                # One small hint if logging is broken (so we can fix it); no secrets
                                try:
                                    print(f"[agent-debug] Could not write debug_visible log: {last_exc}", file=__import__("sys").stderr)
                                except Exception:
                                    pass
                            except Exception:
                                pass
                        _agent_log(
                            "A",
                            "titan/main_DNA.py:206-212",
                            "About to call plot_wells_raw_plotly",
                            {
                                "save_path": str(save_path),
                                "experiment_name": str(experiment_name_slice),
                                "exp_type": str(type(exp)),
                                "has_wells_list": hasattr(exp, "wells_list"),
                                "n_wells_list": (len(exp.wells_list) if hasattr(exp, "wells_list") and exp.wells_list is not None else None),
                            },
                        )
                        _agent_log_visible({
                            "sessionId": "debug-session",
                            "runId": "pre-fix",
                            "hypothesisId": "A",
                            "location": "titan/main_DNA.py:206-212",
                            "message": "About to call plot_wells_raw_plotly",
                            "data": {
                                "save_path": str(save_path),
                                "experiment_name": str(experiment_name_slice),
                                "exp_type": str(type(exp)),
                                "has_wells_list": hasattr(exp, "wells_list"),
                                "n_wells_list": (len(exp.wells_list) if hasattr(exp, "wells_list") and exp.wells_list is not None else None),
                            },
                            "timestamp": int(_agent_time.time() * 1000),
                        })
                        #endregion agent log
                        _show = True
                        _auto_open = True
                        if is_multi_vref:
                            _show = bool(MULTI_VREF_SHOW_PLOTS)
                            _auto_open = bool(MULTI_VREF_AUTO_OPEN_HTML)
                        plot_wells_raw_plotly(
                            exp,
                            save_path=save_path,
                            experiment_name=experiment_name_slice,
                            show=_show,
                            auto_open_html=_auto_open,
                        )
                        #region agent log
                        _agent_log(
                            "B",
                            "titan/main_DNA.py:raw_chem_plotly",
                            "plot_wells_raw_plotly completed",
                            {},
                            runId="pre-fix",
                        )
                        _agent_log_visible({
                            "sessionId": "debug-session",
                            "runId": "pre-fix",
                            "hypothesisId": "B",
                            "location": "titan/main_DNA.py:raw_chem_plotly",
                            "message": "plot_wells_raw_plotly completed",
                            "data": {},
                            "timestamp": int(_agent_time.time() * 1000),
                        })
                        #endregion agent log
                    except Exception as e:
                        print(f"    Warning: Could not create raw chem plot: {str(e)}")
                        #region agent log
                        _agent_log(
                            "B",
                            "titan/main_DNA.py:raw_chem_plotly",
                            "plot_wells_raw_plotly raised exception",
                            {"exc_type": str(type(e)), "exc_str": str(e), "traceback": _agent_traceback.format_exc()},
                            runId="pre-fix",
                        )
                        _agent_log_visible({
                            "sessionId": "debug-session",
                            "runId": "pre-fix",
                            "hypothesisId": "B",
                            "location": "titan/main_DNA.py:raw_chem_plotly",
                            "message": "plot_wells_raw_plotly raised exception",
                            "data": {"exc_type": str(type(e)), "exc_str": str(e), "traceback": _agent_traceback.format_exc()},
                            "timestamp": int(_agent_time.time() * 1000),
                        })
                        #endregion agent log

                # ===== STEP 1: CALCULATE FIRST AND SECOND DERIVATIVES =====
                print("\n" + "="*60)
                print("STEP 1: CALCULATING FIRST AND SECOND DERIVATIVES")
                print("="*60)
                
                # Calculate first derivatives for all wells with filtering
                print("Calculating first derivatives for all wells...")
                filter_order = 20  # You can adjust this value: 1=no filter, higher values = more smoothing
                well_data = calculate_well_derivatives(exp, n_wells, filter_order)
                print(f"First derivatives calculated for {len(well_data)} wells with filter order {filter_order}")
                
                # Calculate second derivatives for all wells
                print("Calculating second derivatives for all wells...")
                second_deriv_data = calculate_well_second_derivatives(well_data, filter_order)
                print(f"Second derivatives calculated for {len(well_data)} wells with filter order {filter_order}")
                
                # ===== STEP 2: FIRST DERIVATIVE PEAK DETECTION =====
                print("\n" + "="*60)
                print("STEP 2: FIRST DERIVATIVE PEAK DETECTION")
                print("="*60)
                
                # Detect peaks in the first derivatives
                print("Detecting peaks in first derivatives...")
                
                # First, check the derivative ranges to help set appropriate threshold
                print("First derivative ranges for each well:")
                for well_idx, data in well_data.items():
                    deriv_min = np.min(data['derivative'])
                    deriv_max = np.max(data['derivative'])
                    print(f"  Well {well_idx + 1}: min={deriv_min:.6f}, max={deriv_max:.6f}")
                
                # Use the experiment-specific peak threshold (already set above)
                peak_results = detect_derivative_peaks(well_data, min_peak_height=peak_threshold)
                
                # Print peak detection results
                print(f"\nFirst Derivative Peak Detection Results (threshold: {peak_threshold}):")
                for well_idx, result in peak_results.items():
                    if result['peak_detected']:
                        # Time is now already in minutes
                        print(f"  Well {well_idx + 1}: Peak at time {result['peak_time']:.2f} min, value {result['peak_value']:.4f}")
                    else:
                        print(f"  Well {well_idx + 1}: No peak detected")
                
                # TTP calculation removed by request.
                second_peak_results = None

                # ===== STEP 3: PLOTTING ALL DATA USING PLOTLY =====
                print("\n" + "="*60)
                print("STEP 3: PLOTTING ALL DATA USING PLOTLY")
                print("="*60)
                
                # Plot first derivative results (TTP overlays disabled).
                print("Saving first derivative plots with detected peaks...")
                plot_well_derivatives(well_data, peak_results, second_peak_results, 
                                     save_path=save_path, experiment_name=experiment_name_slice)
                
                # Plot second derivatives (no TTP/ZC overlays).
                print("Saving second derivative plots...")
                plot_well_second_derivatives(second_deriv_data, second_peak_results,
                                           save_path=save_path, experiment_name=experiment_name_slice)
                
                # Create interactive plotly plots (TTP markers disabled).
                print("\nCreating interactive plotly plots for well derivatives...")
                try:
                    plots_created = []
                    
                    # Plot all wells in a 5x2 grid layout.
                    if PLOT_CONFIG['raw_signal_grid']:
                        print("  - Creating raw signal grid plot...")
                        plot_wells_grid_plotly(
                            exp,
                            save_path=save_path,
                            experiment_name=experiment_name_slice,
                            ttp_results=None,
                            show=(True if not is_multi_vref else bool(MULTI_VREF_SHOW_PLOTS)),
                        )
                        plots_created.append("Raw Signal Grid")

                        # Export the same underlying traces to a CSV (long format).
                        # This lets you compare wells across vref slices without re-parsing HTML.
                        try:
                            vref_idx_to_write = ref_idx_norm if ref_idx_norm is not None else np.nan
                            vref_value_to_write = vref_value if vref_value is not None else np.nan

                            n_wells = len(exp.wells_list)

                            # Build wide-format export:
                            # - rows: timepoints
                            # - columns: wells (well_1..well_N)
                            # - extra columns: vref_idx/vref_value repeated for each row
                            well_series = {}
                            time_s_ref = None
                            min_len = None

                            for well_idx, well in enumerate(exp.wells_list):
                                time_npr = getattr(well, "time_npr", None)
                                idx_settled = getattr(well, "idx_settled", 0)
                                idx_end = getattr(well, "idx_end", None)
                                # Match plot_wells_grid_plotly exactly: active-mean, no filtering.
                                y = getattr(well, "well_2d_nl_bs_active_mean", None)

                                if time_npr is None or y is None:
                                    continue
                                time_npr = np.asarray(time_npr)
                                y = np.asarray(y)
                                if time_npr.size == 0 or y.size == 0:
                                    continue

                                if idx_end is None:
                                    idx_end = len(time_npr)
                                time_slice = time_npr[int(idx_settled) : int(idx_end)]
                                if time_slice.size == 0:
                                    continue

                                # Match `plot_wells_grid_plotly`: zero at settled time.
                                time_s = time_slice - time_slice[0]
                                time_min = time_s / 60.0

                                n = int(min(len(time_min), len(y)))
                                if n <= 0:
                                    continue

                                if time_s_ref is None:
                                    time_s_ref = time_s[:n]
                                else:
                                    # Keep the same reference; we'll truncate later.
                                    pass

                                min_len = n if min_len is None else min(min_len, n)
                                well_series[well_idx + 1] = y[:n]

                            if time_s_ref is not None and well_series and min_len is not None and min_len > 0:
                                time_s_common = time_s_ref[:min_len]
                                time_min_common = time_s_common / 60.0

                                df_export = pd.DataFrame(
                                    {
                                        "vref_idx": vref_idx_to_write,
                                        "vref_value": vref_value_to_write,
                                        "time_s": time_s_common,
                                        "time_min": time_min_common,
                                    }
                                )

                                # Fill each well column; missing wells get NaN.
                                for well_number in range(1, n_wells + 1):
                                    col_name = f"well_{well_number}"
                                    if well_number in well_series:
                                        df_export[col_name] = well_series[well_number][:min_len]
                                    else:
                                        df_export[col_name] = np.full(min_len, np.nan, dtype=float)

                                df_export.to_csv(
                                    raw_signal_grid_csv_path,
                                    mode="a",
                                    header=not raw_signal_grid_csv_written_header,
                                    index=False,
                                )
                                raw_signal_grid_csv_written_header = True
                        except Exception as e:
                            print(f"    Warning: Failed to export raw_signal_grid CSV: {str(e)}")

                    if PLOT_CONFIG.get('raw_chem_grid', False):
                        print("  - Creating raw chem grid plot (no baseline subtraction)...")
                        plot_wells_grid_plotly(
                            exp,
                            save_path=save_path,
                            experiment_name=experiment_name_slice,
                            ttp_results=None,
                            show=(True if not is_multi_vref else bool(MULTI_VREF_SHOW_PLOTS)),
                            baseline_subtract=False,
                        )
                        plots_created.append("Raw Chem Grid (no BS)")
                    
                    # Plot linearized signals combined for all wells together
                    if PLOT_CONFIG['linearized_signal_combined']:
                        print("  - Creating linearized signal combined plot...")
                        plot_wells_linearized_signal_combined_plotly(
                            exp,
                            save_path=save_path,
                            experiment_name=experiment_name_slice,
                            ttp_results=None,
                            wells_to_show=LINEARIZED_COMBINED_VISIBILITY.get('wells_to_show'),
                            wells_to_hide=LINEARIZED_COMBINED_VISIBILITY.get('wells_to_hide'),
                            hide_mode=LINEARIZED_COMBINED_VISIBILITY.get('hide_mode', 'legendonly'),
                            scale_factor=1e3,
                            y_offset=500,
                            show=(True if not is_multi_vref else bool(MULTI_VREF_SHOW_PLOTS)),
                        )
                        # Export the exact traces used in the linearized combined plot to Excel:
                        # signal = well_2d_start_bs_active_mean, time = start-zeroed minutes,
                        # y_plot = signal * 1e3 + 500.
                        try:
                            vref_idx_to_write = ref_idx_norm if ref_idx_norm is not None else np.nan
                            vref_value_to_write = vref_value if vref_value is not None else np.nan
                            scale_factor = 1e3
                            y_offset = 500

                            for well_idx, well in enumerate(exp.wells_list):
                                y_raw = getattr(well, "well_2d_start_bs_active_mean", None)
                                time_npr = getattr(well, "time_npr", None)
                                idx_start = int(getattr(well, "idx_start", 0))
                                idx_end = getattr(well, "idx_end", None)

                                if y_raw is None or time_npr is None:
                                    continue
                                y_raw = np.asarray(y_raw, dtype=float)
                                time_npr = np.asarray(time_npr, dtype=float)
                                if y_raw.size == 0 or time_npr.size == 0:
                                    continue

                                if idx_end is None:
                                    idx_end = len(time_npr)
                                time_slice = time_npr[int(idx_start):int(idx_end)]
                                if time_slice.size == 0:
                                    continue

                                time_min = (time_slice - time_slice[0]) / 60.0
                                n = int(min(len(time_min), len(y_raw)))
                                if n <= 0:
                                    continue

                                y_raw_n = y_raw[:n]
                                y_plot_n = (y_raw_n * scale_factor) + y_offset

                                # Call user-requested change-point function on linearized data.
                                if change_point_detection_fn is not None:
                                    try:
                                        cp_result = change_point_detection_fn(time_min[:n], y_raw_n)
                                        if cp_result and cp_result.get("found_change_point", False):
                                            print(
                                                f"    Change point - Well {well_idx + 1}: "
                                                f"{cp_result.get('change_time_min', np.nan):.2f} min "
                                                f"(score={cp_result.get('change_score', np.nan):.4g})"
                                            )
                                    except Exception as e:
                                        print(f"    Warning: change_point_detection failed for Well {well_idx + 1}: {e}")

                                df_lin = pd.DataFrame(
                                    {
                                        "experiment": experiment_name,
                                        "vref_idx": vref_idx_to_write,
                                        "vref_value": vref_value_to_write,
                                        "well": int(well_idx + 1),
                                        "time_min": time_min[:n],
                                        "linearized_signal_raw": y_raw_n,
                                        "linearized_signal_plot": y_plot_n,
                                        "scale_factor": scale_factor,
                                        "y_offset": y_offset,
                                    }
                                )
                                linearized_combined_export_frames.append(df_lin)
                        except Exception as e:
                            print(f"    Warning: Failed to prepare linearized combined export: {str(e)}")
                        print("  - Creating linearized signal single-axis combined plot...")
                        plots_created.append("Linearized Signal Combined")
                        print("  - Creating linearized signal combined plot (Savitzky-Golay smoothed)...")
                        plot_wells_linearized_signal_combined_savgol_plotly(
                            exp,
                            save_path=save_path,
                            experiment_name=experiment_name_slice,
                            ttp_results=None,
                            wells_to_show=LINEARIZED_COMBINED_VISIBILITY.get('wells_to_show'),
                            wells_to_hide=LINEARIZED_COMBINED_VISIBILITY.get('wells_to_hide'),
                            hide_mode=LINEARIZED_COMBINED_VISIBILITY.get('hide_mode', 'legendonly'),
                            scale_factor=1e3,
                            y_offset=500,
                            show=(True if not is_multi_vref else bool(MULTI_VREF_SHOW_PLOTS)),
                            savgol_window_length=150,
                            savgol_polyorder=3,
                        )
                        plots_created.append("Linearized Signal Combined (Savgol)")
                    
                    # Plot first derivatives in a 5x2 grid layout.
                    if PLOT_CONFIG['first_derivative_grid']:
                        print("  - Creating first derivative grid plot...")
                        plot_wells_first_derivative_grid_plotly(
                            exp,
                            well_data,
                            save_path=save_path,
                            experiment_name=experiment_name_slice,
                            ttp_results=None,
                            show=(True if not is_multi_vref else bool(MULTI_VREF_SHOW_PLOTS)),
                        )
                        plots_created.append("First Derivative Grid")
                    
                    # Plot second derivatives in a 5x2 grid layout.
                    if PLOT_CONFIG['second_derivative_grid']:
                        print("  - Creating second derivative grid plot...")
                        plot_wells_second_derivative_grid_plotly(
                            exp,
                            second_deriv_data,
                            save_path=save_path,
                            experiment_name=experiment_name_slice,
                            ttp_results=None,
                            show=(True if not is_multi_vref else bool(MULTI_VREF_SHOW_PLOTS)),
                        )
                        plots_created.append("Second Derivative Grid")
                    
                    # Plot all wells combined for comparison.
                    if PLOT_CONFIG['raw_signal_combined']:
                        print("  - Creating raw signal combined plot...")
                        plot_all_wells_combined_plotly(
                            exp,
                            save_path=save_path,
                            experiment_name=experiment_name_slice,
                            ttp_results=None,
                            show=(True if not is_multi_vref else bool(MULTI_VREF_SHOW_PLOTS)),
                        )
                        plots_created.append("Raw Signal Combined")
                    
                    # Plot average (unfiltered) combined plot
                    if PLOT_CONFIG['average_combined']:
                        print("  - Creating average (unfiltered) combined plot...")
                        try:
                            plot_wells_average_combined_plotly(
                                exp,
                                save_path=save_path,
                                experiment_name=experiment_name_slice,
                                wells_to_show=AVERAGE_COMBINED_VISIBILITY.get('wells_to_show'),
                                wells_to_hide=AVERAGE_COMBINED_VISIBILITY.get('wells_to_hide'),
                                hide_mode=AVERAGE_COMBINED_VISIBILITY.get('hide_mode', 'legendonly'),
                                show=(True if not is_multi_vref else bool(MULTI_VREF_SHOW_PLOTS)),
                            )
                            plots_created.append("Average Combined")
                        except Exception as e:
                            print(f"    Warning: Could not create average combined plot: {str(e)}")
                    
                    # Plot active pixels for all wells.
                    if PLOT_CONFIG['active_pixels']:
                        print("  - Creating active pixels plot...")
                        active_fig = plot_wells_active_pixels_plotly(
                            exp,
                            save_path=(save_path if not is_multi_vref else None),
                            experiment_name=experiment_name_slice,
                            ttp_results=None,
                            figsize=(1600, 600),
                            show=(True if not is_multi_vref else False),
                        )
                        # For multi-VREF runs, collect masks for one combined overlay plot.
                        if is_multi_vref:
                            try:
                                if active_fig is not None and getattr(active_fig, "data", None):
                                    z = np.asarray(active_fig.data[0].z, dtype=float)
                                    if z.ndim == 2:
                                        active_pixels_overlay_data.append(
                                            {
                                                "vref_idx": ref_idx_norm,
                                                "vref_value": vref_value,
                                                "mask": (z > 0.5),
                                            }
                                        )
                            except Exception as e:
                                print(f"    Warning: Could not collect active-pixel mask for overlay plot: {e}")
                        plots_created.append("Active Pixels")
                        if not is_multi_vref:
                            try:
                                if active_fig is not None and getattr(active_fig, "data", None):
                                    z = np.asarray(active_fig.data[0].z, dtype=float)
                                    if z.ndim == 2:
                                        n_act = int(np.sum(z > 0.5))
                                        print("\n" + "=" * 60)
                                        print(f"ACTIVE PIXELS PER VREF — {experiment_name_slice}")
                                        print("=" * 60)
                                        print(
                                            pd.DataFrame(
                                                [
                                                    {
                                                        "Vref_Idx": ref_idx_norm,
                                                        "Vref_Value": (
                                                            vref_value
                                                            if vref_value is not None
                                                            and not (
                                                                isinstance(vref_value, float)
                                                                and np.isnan(vref_value)
                                                            )
                                                            else ""
                                                        ),
                                                        "Active_Pixels": n_act,
                                                    }
                                                ]
                                            ).to_string(index=False)
                                        )
                                        print("=" * 60 + "\n")
                            except Exception as e:
                                print(f"    Warning: Could not print active-pixel count table: {e}")

                    if PLOT_CONFIG.get('raw_chip_video', False):
                        print("  - Creating raw chip video (Plotly)...")
                        try:
                            _show_rc = True if not is_multi_vref else bool(MULTI_VREF_SHOW_PLOTS)
                            _auto_open_rc = True if not is_multi_vref else bool(MULTI_VREF_AUTO_OPEN_HTML)
                            plot_raw_chip_video_plotly(
                                exp,
                                save_path=save_path,
                                experiment_name=experiment_name_slice,
                                show=_show_rc,
                                auto_open_html=_auto_open_rc,
                            )
                            plots_created.append("Raw Chip Video")
                            if PLOT_CONFIG.get('raw_chip_video_mp4', False) and save_path:
                                safe_name = sanitize_filename(experiment_name_slice)
                                mp4_path = Path(save_path) / f"{safe_name}_raw_chip.mp4"
                                export_raw_chip_video(
                                    exp,
                                    mp4_path,
                                    experiment_name=experiment_name_slice,
                                    fps=10,
                                    max_frames=200,
                                )
                        except Exception as e:
                            print(f"    Warning: Could not create raw chip video: {str(e)}")
                    
                    if PLOT_CONFIG.get('linearized_chem_plotly', False):
                        print("  - Creating linearized chem plot (plotly, per-pixel)...")
                        try:
                            plot_wells_raw_plotly(
                                exp,
                                save_path=save_path,
                                experiment_name=experiment_name_slice,
                                show=(True if not is_multi_vref else bool(MULTI_VREF_SHOW_PLOTS)),
                                auto_open_html=(True if not is_multi_vref else bool(MULTI_VREF_AUTO_OPEN_HTML)),
                                signal_mode="linearized",
                            )
                            plots_created.append("Linearized Chem Plotly (per-pixel)")
                        except Exception as e:
                            print(f"    Warning: Could not create linearized chem plot: {str(e)}")

                    # Plot temperature data for all wells.
                    if PLOT_CONFIG['temperature']:
                        print("  - Creating temperature plot...")
                        plot_wells_temperature_mean_then_lin_global_plotly(
                            exp,
                            save_path=save_path,
                            experiment_name=experiment_name_slice,
                            show=(True if not is_multi_vref else bool(MULTI_VREF_SHOW_PLOTS)),
                            auto_open_html=(True if not is_multi_vref else bool(MULTI_VREF_AUTO_OPEN_HTML)),
                        )
                        plots_created.append("Temperature")
                    
                    if plots_created:
                        print(f"Interactive plotly plots created successfully: {', '.join(plots_created)}")
                    else:
                        print("No plots were created (all plot types disabled in configuration)")
                        
                except Exception as e:
                    print(f"Warning: Error creating plotly plots: {str(e)}")
                    print("Continuing with processing...")
                
                # Store results for summary table
                if 'all_results' not in locals():
                    all_results = []
                
                # Collect results for this experiment
                for well_idx in range(n_wells):
                    result_row = {
                        'Experiment': experiment_name_slice,
                        'Vref_Idx': (ref_idx_norm if str(n_a_type).lower() in ("v05", "v06") else np.nan),
                        'Vref_Value': (vref_value if str(n_a_type).lower() in ("v05", "v06") else np.nan),
                        'Well': well_idx + 1,
                        'First_Deriv_Peak_Time': np.nan,
                        'First_Deriv_Peak_Value': np.nan,
                    }
                    
                    # Add first derivative peak results
                    if well_idx in peak_results and peak_results[well_idx]['peak_detected']:
                        result_row['First_Deriv_Peak_Time'] = peak_results[well_idx]['peak_time']
                        result_row['First_Deriv_Peak_Value'] = peak_results[well_idx]['peak_value']
                    
                    all_results.append(result_row)
                    
                print(f"Successfully processed {experiment_name_slice}")
                
            except Exception as e:
                print(f"Error during processing of {experiment_name_slice}: {str(e)}")
                print("Skipping this ref slice and continuing...")
                continue

        # Create one combined active-pixels overlay plot across all VREF slices.
        if PLOT_CONFIG.get('active_pixels', False) and is_multi_vref and len(active_pixels_overlay_data) > 0:
            try:
                print("Creating combined full-chip active-pixels overlay across all VREFs...")
                fig_overlay = go.Figure()
                # Requested VREF colors: yellow, green, red; keep blue background.
                overlay_palette = [
                    "#F0E442",  # yellow
                    "#009E73",  # green
                    "#D55E00",  # red
                    "#CC79A7",
                    "#56B4E9",
                    "#E69F00",
                    "#000000",
                ]

                # Keep VREF ordering stable in legend.
                active_pixels_overlay_data.sort(
                    key=lambda d: (d["vref_idx"] if d["vref_idx"] is not None else 10**9)
                )

                vref_pixel_rows = []
                for overlay_item in active_pixels_overlay_data:
                    mask_i = np.asarray(overlay_item["mask"], dtype=bool)
                    vref_val = overlay_item.get("vref_value")
                    vref_pixel_rows.append(
                        {
                            "Vref_Idx": overlay_item.get("vref_idx"),
                            "Vref_Value": (
                                vref_val
                                if vref_val is not None
                                and not (isinstance(vref_val, float) and np.isnan(vref_val))
                                else ""
                            ),
                            "Active_Pixels": int(np.sum(mask_i)),
                        }
                    )
                if vref_pixel_rows:
                    vref_pixels_df = pd.DataFrame(vref_pixel_rows)
                    print("\n" + "=" * 60)
                    print(f"ACTIVE PIXELS PER VREF — {experiment_name}")
                    print("=" * 60)
                    print(vref_pixels_df.to_string(index=False))
                    print("=" * 60)

                # Build a single chip-sized label map:
                # -1: inactive, 0..N-1: active and assigned to that VREF.
                # In overlap regions (active in multiple VREFs), later VREFs overwrite earlier ones.
                mask_shape = active_pixels_overlay_data[0]["mask"].shape
                label_map = np.full(mask_shape, -1, dtype=int)
                union_mask = np.zeros(mask_shape, dtype=bool)
                vref_labels = []

                for i_overlay, overlay_item in enumerate(active_pixels_overlay_data):
                    mask = np.asarray(overlay_item["mask"], dtype=bool)
                    if mask.shape != mask_shape:
                        continue
                    union_mask |= mask
                    label_map[mask] = i_overlay

                    vref_idx_disp = overlay_item["vref_idx"]
                    vref_val_disp = overlay_item["vref_value"]
                    if vref_val_disp is None or (isinstance(vref_val_disp, float) and np.isnan(vref_val_disp)):
                        vref_labels.append(f"VREF idx {vref_idx_disp}")
                    else:
                        vref_labels.append(f"VREF idx {vref_idx_disp} ({vref_val_disp:g})")

                if vref_pixel_rows:
                    print(f"Unique active pixels (any VREF): {int(np.sum(union_mask))}\n")

                # Base layer: old-style inactive background / active foreground mask.
                fig_overlay.add_trace(
                    go.Heatmap(
                        z=union_mask.astype(int),
                        colorscale=[
                            [0.0, "#0b1f5f"],   # inactive (dark blue)
                            [0.499, "#0b1f5f"],
                            [0.5, "#0b1f5f"],   # active also blue; VREF layer colors active pixels
                            [1.0, "#0b1f5f"],
                        ],
                        showscale=False,
                        hoverinfo="skip",
                    )
                )

                # Overlay layer: VREF category colors on active pixels only.
                z_vref = np.where(union_mask, label_map.astype(float), np.nan)
                n_vrefs = max(len(vref_labels), 1)
                discrete_scale = []
                for i_color in range(n_vrefs):
                    color = overlay_palette[i_color % len(overlay_palette)]
                    left = i_color / n_vrefs
                    right = (i_color + 1) / n_vrefs
                    discrete_scale.append([left, color])
                    discrete_scale.append([right, color])

                fig_overlay.add_trace(
                    go.Heatmap(
                        z=z_vref,
                        colorscale=discrete_scale,
                        zmin=-0.5,
                        zmax=n_vrefs - 0.5,
                        opacity=1.0,
                        colorbar=dict(
                            title="VREF",
                            tickmode="array",
                            tickvals=list(range(len(vref_labels))),
                            ticktext=vref_labels,
                        ),
                        hovertemplate="Row %{y}<br>Column %{x}<br>%{z}<extra></extra>",
                    )
                )

                # Keep chip pixels square and size figure from chip geometry
                # to avoid stretched/squashed appearance.
                nrows_chip, ncols_chip = mask_shape
                target_height = 900
                target_width = int((target_height * ncols_chip / nrows_chip) + 260)  # room for colorbar

                fig_overlay.update_layout(
                    title=f"{experiment_name} - Full Chip Active Pixels Across VREFs",
                    xaxis_title="Column",
                    yaxis_title="Row",
                    height=target_height,
                    width=target_width,
                    autosize=False,
                    margin=dict(l=60, r=90, t=80, b=60),
                )
                fig_overlay.update_xaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="lightgray",
                    scaleanchor="y",
                    scaleratio=1.0,
                    constrain="domain",
                )
                fig_overlay.update_yaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="lightgray",
                    constrain="domain",
                    autorange="reversed",
                )

                if bool(MULTI_VREF_SHOW_PLOTS):
                    fig_overlay.show()

                save_path_obj = Path(save_path)
                save_path_obj.mkdir(parents=True, exist_ok=True)
                safe_exp_name = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in experiment_name)
                overlay_html = save_path_obj / f"{safe_exp_name}_active_pixels_full_chip_all_vrefs_overlay.html"
                fig_overlay.write_html(str(overlay_html), auto_open=bool(MULTI_VREF_AUTO_OPEN_HTML))
                print(f"Combined active-pixels overlay saved to: {overlay_html}")
            except Exception as e:
                print(f"Warning: Could not create combined active-pixels overlay: {e}")

        # dVref coarse model + selection (L_ttn.c TTN_SweepSearch_dVref_Multi vs Lacewing_Thread payload)
        if PLOT_CONFIG.get("dvref_selection_model", False):
            if dvref_tele is not None and dvref_tele.get("coarse") and dvref_tele.get("model"):
                try:
                    plot_dvref_selection_model_plotly(
                        dvref_tele,
                        save_path=save_path,
                        experiment_name=experiment_name,
                        show=bool(MULTI_VREF_SHOW_PLOTS),
                        auto_open_html=bool(MULTI_VREF_AUTO_OPEN_HTML),
                    )
                except Exception as e:
                    print(f"Warning: Could not create dVref selection model plot: {e}")

        # Save linearized combined export to Excel (one file per experiment).
        if PLOT_CONFIG.get('linearized_signal_combined', False) and len(linearized_combined_export_frames) > 0:
            try:
                linearized_df = pd.concat(linearized_combined_export_frames, ignore_index=True)
                save_path_obj = Path(save_path)
                save_path_obj.mkdir(parents=True, exist_ok=True)
                safe_exp_name = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in experiment_name)
                linearized_xlsx_path = save_path_obj / f"{safe_exp_name}_linearized_signal_combined_all_vrefs.xlsx"
                linearized_df.to_excel(linearized_xlsx_path, index=False, sheet_name="linearized_combined")
                print(f"Linearized combined Excel export saved to: {linearized_xlsx_path}")
            except Exception as e:
                print(f"Warning: Could not save linearized combined Excel export: {e}")

        # PID gain recommendations (readout version=5 tail + temp_log.bin).
        if "pid" in str(exp_path).lower():
            try:
                from optimize_pid import load_pid_ingest, run_models, recommendations_to_dict

                print("\n" + "=" * 60)
                print("PID TUNING (optimize_pid.py)")
                print("=" * 60)
                pid_ingest = load_pid_ingest(exp_path)
                pid_recs = run_models(pid_ingest)
                for r in pid_recs:
                    g = r.gains
                    print(f"--- {r.name} [{r.model_id}] ---")
                    if r.score is not None:
                        print(f"  score (ITAE): {r.score:.6g}")
                    print(f"  TEMP_P = {g.kp:.6g}  TEMP_I = {g.ki:.6g}  TEMP_D = {g.kd:.6g}")
                    if g.notes:
                        print(f"  notes: {g.notes}")
                save_path_obj = Path(save_path)
                save_path_obj.mkdir(parents=True, exist_ok=True)
                safe_exp_name = "".join(
                    c if (c.isalnum() or c in ("-", "_")) else "_" for c in experiment_name
                )
                pid_json_path = save_path_obj / f"{safe_exp_name}_pid_recommendations.json"
                import json as _pid_json

                with open(pid_json_path, "w", encoding="utf-8") as _pid_f:
                    _pid_json.dump(
                        {
                            "exp_path": str(exp_path),
                            "readout_path": str(pid_ingest.readout_path),
                            "n_frames": pid_ingest.n_frames,
                            "has_pid_tail": pid_ingest.has_pid_tail,
                            "recommendations": recommendations_to_dict(pid_recs),
                        },
                        _pid_f,
                        indent=2,
                    )
                print(f"PID recommendations saved to: {pid_json_path}")
            except Exception as e:
                print(f"Warning: PID tuning skipped for {exp_path.name}: {e}")
    
    # Display and save summary table
    if 'all_results' in locals() and len(all_results) > 0:
        print("\n" + "="*80)
        print("SUMMARY TABLE - ALL EXPERIMENTS")
        print("="*80)
        
        # Create DataFrame for nice display
        results_df = pd.DataFrame(all_results)
        
        # Display the table
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        print(results_df.to_string(index=False))

        # Display dedicated table for first derivative peak times
        first_deriv_table = results_df[['Experiment', 'Well', 'First_Deriv_Peak_Time']].copy()
        print("\nFirst Derivative Peak Times (minutes):")
        print(first_deriv_table.to_string(index=False))
        
        # Save to Excel
        save_path = r"/vol/bitbucket/gk225/multi_viz"
        os.makedirs(save_path, exist_ok=True)
        excel_filename = "peak_detection_summary.xlsx"
        excel_filepath = os.path.join(save_path, excel_filename)
        
        with pd.ExcelWriter(excel_filepath, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='Peak_Detection_Results', index=False)
            first_deriv_table.to_excel(writer, sheet_name='First_Deriv_Peak_Times', index=False)
            
            # Auto-adjust column widths
            worksheet = writer.sheets['Peak_Detection_Results']
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width

            worksheet_fd = writer.sheets['First_Deriv_Peak_Times']
            for column in worksheet_fd.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet_fd.column_dimensions[column_letter].width = adjusted_width
        
        print(f"\nSummary table saved to: {excel_filepath}")
        
        # Also save as CSV for easy viewing
        csv_filename = "peak_detection_summary.csv"
        csv_filepath = os.path.join(save_path, csv_filename)
        results_df.to_csv(csv_filepath, index=False)
        print(f"Summary table also saved as CSV: {csv_filepath}")

        first_deriv_csv_filename = "first_derivative_peak_times.csv"
        first_deriv_csv_filepath = os.path.join(save_path, first_deriv_csv_filename)
        first_deriv_table.to_csv(first_deriv_csv_filepath, index=False)
        print(f"First derivative peak times saved as CSV: {first_deriv_csv_filepath}")
        
    else:
        print("\nNo results to display in summary table.")

    # Overlay multiple experiment folders on one graph (aligned at vref jump).
    if OVERLAY_CONFIG.get("enabled"):
        try:
            from overlay_experiments_plotly import plot_experiments_overlay

            overlay_paths = OVERLAY_CONFIG.get("overlay_paths") or exp_paths
            overlay_save = OVERLAY_CONFIG.get(
                "save_path",
                r"/vol/bitbucket/gk225/multi_viz",
            )
            plot_experiments_overlay(
                overlay_paths,
                n_wells=n_wells,
                n_a_type=n_a_type,
                ref_idx=int(OVERLAY_CONFIG.get("ref_idx", 0)),
                signal=str(OVERLAY_CONFIG.get("signal", "linearized_mean")),
                wells=OVERLAY_CONFIG.get("wells"),
                save_path=overlay_save,
                output_name=str(OVERLAY_CONFIG.get("output_name", "experiments_overlay_vref_jump")),
                show=bool(OVERLAY_CONFIG.get("show", True)),
                print_status=True,
            )
        except Exception as e:
            print(f"Warning: experiment overlay plot failed: {e}")