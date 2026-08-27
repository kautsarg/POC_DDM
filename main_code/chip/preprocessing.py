import os
import sys
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter
from joblib import Parallel, delayed

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "utils"))

import config
from safe_io import safe_joblib_dump
import sigmoid_fitting as sp
from chip_v6_utils import load_and_preprocess_v6
from kinetic_features import build_kinetic_features

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================
# CURVE RECONSTRUCTION (from titan_v6 Experiment/Well objects)
# ==========================================

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

            df_pnl = pd.DataFrame(well_2d_nl_bs_active.T, columns=time_cols)
            df_pnl['well_id'] = w_idx
            df_pnl['pixel_row_idx'] = active_y
            df_pnl['pixel_col_idx'] = active_x
            df_pnl['temp_group_idx'] = active_temp_mapping
            df_pnl['num_active_pixels_in_temp_group'] = df_pnl['temp_group_idx'].map(counts)
            df_pnl['well_temp_lin2d_mean'] = np.asarray(mean_temp_lin[active_temp_mapping])
            df_pnl['well_2d_temp_npr_mean'] = np.asarray(mean_temp_nl[active_temp_mapping])
            df_pnl['vref_idx'] = vref_idx

            meta_cols = ['well_id', 'pixel_row_idx', 'pixel_col_idx', 'temp_group_idx',
                        'num_active_pixels_in_temp_group', 'well_temp_lin2d_mean', 'well_2d_temp_npr_mean', 'vref_idx']

            df_pl = df_pl[meta_cols + time_cols]
            df_pnl = df_pnl[meta_cols + time_cols]

            df_tl = pd.DataFrame(well_temp_lin2d.T, columns=time_cols)
            df_tl.insert(0, 'well_id', w_idx)
            df_tl.insert(1, 'temp_group_idx', np.arange(well_temp_lin2d.shape[1]))
            df_tl.insert(2, 'vref_idx', vref_idx)

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
        "well_2d_temp_npr_df": pd.concat(df_temp_nl_list, ignore_index=True),
    }


# ==========================================
# CURVE VARIANTS: normalization + SG-p4 denoising
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


# ==========================================
# LSTM-AE GLOBAL OUTLIER DETECTION (optional, --run_outlier_detection)
# ==========================================

def run_outlier_detection_step(exp_path, dataset_name, dataset, Y_well, kinetic_features,
                                save_plot_flag, batch_size=128):
    """Global LSTM-AE outlier detection; merges label columns into kinetic_features."""
    sys.path.insert(0, str(_ROOT / "utils" / "model_training"))
    sys.path.insert(0, str(_ROOT / "utils" / "outlier_detection"))
    from model_utils import set_global_determinism
    from lstm_autoencoder_outlier import run_lstm_autoencoder_pipeline

    set_global_determinism(0, strict=True)

    print("\n=== RUNNING LSTM-AE GLOBAL OUTLIER DETECTION ===")
    extracted_dfs = run_lstm_autoencoder_pipeline(
        dataset_name, dataset, Y_well, ref_curves=dataset[0],
        ae_plot_path=str(Path(exp_path) / "lstm_ae_outlier"),
        threshold_percentiles=["elbow", 90, 95],
        save_plot=save_plot_flag,
        downsample_factor=config.AE_DOWNSAMPLE_FACTOR,
        save_encoder_dir=str(Path(exp_path) / "pretrained_encoders"),
        batch_size=batch_size,
    )
    return [pd.concat([kf, extracted], axis=1) for kf, extracted in zip(kinetic_features, extracted_dfs)]


# ==========================================
# SAVE
# ==========================================

def save_experiment_data(save_exp_path, curves_dict, dataset_name, dataset, kinetic_features,
                         pixel_temp_dfs, Y_well, X_time, max_significant_index, sg_p4_optimal_w,
                         pc_wells_data=None):
    df_meta = pixel_temp_dfs["well_2d_nl_bs_active_df"]

    save_data = {
        "curves": curves_dict,
        "dataset_name": dataset_name,
        "dataset": dataset,
        "kinetic_features": kinetic_features,
        "Y_well": Y_well,
        "idxs": {"max_significant_index": max_significant_index},
        "timestamps": X_time,
        "well_labels": Y_well,
        "metadata": {
            "pixel_row_idx": df_meta['pixel_row_idx'].values,
            "pixel_col_idx": df_meta['pixel_col_idx'].values,
            "temp_group_idx": df_meta['temp_group_idx'].values,
            "num_active_pixels_in_temp_group": df_meta['num_active_pixels_in_temp_group'].values,
            "well_temp_lin2d_mean": df_meta['well_temp_lin2d_mean'].values,
            "well_2d_temp_npr_mean": df_meta['well_2d_temp_npr_mean'].values,
            "vref_idx": df_meta['vref_idx'].values,
        },
        "concentration": config.get_conc_array(Path(save_exp_path).name, Y_well),
        "sg_p4_optimal_w": sg_p4_optimal_w,
    }
    if pc_wells_data is not None:
        save_data["pc_wells"] = pc_wells_data

    save_path = os.path.join(save_exp_path, config.TRAINING_DATA_PATH)
    safe_joblib_dump(save_data, save_path, compress=3)
    print(f"  -> Saved {save_path}")


def main(argv=None):
    print(f"\n{'='*70}\n[RUNNING] chip/preprocessing.py\n{'='*70}\n")
    parser = argparse.ArgumentParser(description="CHIP curve preprocessing + kinetic features + optional LSTM-AE outlier detection")
    parser.add_argument("--task_id", type=int, default=0, help="Array Job ID")
    parser.add_argument("--exp_folder", type=str, default=config.DEFAULT_EXP_FOLDER, help="Path to experiment datasets")
    parser.add_argument("--force_rerun", action="store_true", help="Recompute and overwrite even if a presaved file already exists")
    parser.add_argument("--run_outlier_detection", action="store_true",
                        help="Also run the global LSTM-AE outlier detector (TensorFlow/GPU) and save a "
                             "pretrained encoder. Off by default -- curve preprocessing + kinetic feature "
                             "extraction alone don't need TensorFlow.")
    args = parser.parse_args(argv)

    n_wells = config.N_WELLS
    n_a_type = config.N_A_TYPE

    exp_paths = sorted([Path(args.exp_folder, name) for name in os.listdir(args.exp_folder)
                        if (os.path.isdir(os.path.join(args.exp_folder, name)) and name not in config.EXCLUDED_FOLDERS)])

    if args.task_id >= len(exp_paths):
        print(f"Task ID {args.task_id} is out of bounds for {len(exp_paths)} folders. Exiting.")
        sys.exit(0)

    exp_path = exp_paths[args.task_id]
    save_exp_path = exp_path
    save_path = os.path.join(save_exp_path, config.TRAINING_DATA_PATH)

    saved_viz = getattr(config, "SAVED_VIZ", [])
    save_plot_flag = bool(saved_viz) and any(s in str(save_exp_path) for s in saved_viz)

    existing_data = {}
    if os.path.exists(save_path):
        try:
            existing_data = joblib.load(save_path)
        except Exception as e:
            print(f"  -> [WARNING] Cached file at {save_path} is corrupted ({e}). Recomputing from scratch...")
            existing_data = {}

    have_base = (not args.force_rerun) and existing_data.get("curves") and existing_data.get("kinetic_features")

    if have_base:
        curves_dict = existing_data["curves"]
        dataset_name = existing_data["dataset_name"]
        dataset = existing_data["dataset"]
        kinetic_features = existing_data["kinetic_features"]
        Y_well = existing_data["Y_well"]
        X_time = existing_data["timestamps"]
        pixel_temp_dfs = None  # not needed again -- metadata already saved
        max_significant_index = existing_data["idxs"]["max_significant_index"]
        sg_p4_optimal_w = existing_data.get("sg_p4_optimal_w")
        pc_wells_data = existing_data.get("pc_wells")

        already_has_outlier_cols = any(
            c.startswith(f"lstm_ae_glb_ds{config.AE_DOWNSAMPLE_FACTOR}_label_")
            for c in kinetic_features[0].columns
        )
        if not args.run_outlier_detection or already_has_outlier_cols:
            print(f"Cache hit: {save_exp_path}")
            print("  ✓ Experiment complete!\n")
            sys.exit(0)

        print(f"Cache hit for base preprocessing at {save_exp_path}; running requested LSTM-AE outlier detection...")
        kinetic_features = run_outlier_detection_step(
            save_exp_path, dataset_name, dataset, Y_well, kinetic_features,
            save_plot_flag=save_plot_flag,
        )
        save_experiment_data(save_exp_path, curves_dict, dataset_name, dataset, kinetic_features,
                            {"well_2d_nl_bs_active_df": pd.DataFrame(existing_data["metadata"])},
                            Y_well, X_time, max_significant_index, sg_p4_optimal_w, pc_wells_data)
        print("  ✓ Experiment complete!\n")
        sys.exit(0)

    print(f"Processing Experiment: {exp_path}")

    all_exp_data = load_and_preprocess_v6(exp_path=exp_path, n_wells=n_wells, n_a_type=n_a_type, vref_ref_idx="all")

    print("  -> Generating Pixel & Temp DataFrames...")
    pixel_temp_dfs = extract_pixel_temp_dataframes(all_exp_data)

    print("  -> Reconstructing curves...")
    X_time, Y_well, X_2d_bs_active = reconstruct_data(all_exp_data, attr_str="well_2d_bs_active")

    # -------------------------------------------------------------
    # TRUNCATION (always runs)
    # -------------------------------------------------------------
    exp_folder = os.path.basename(exp_path)
    mapping = config.LABEL_MAPPINGS.get(exp_folder)
    y_label = np.array([mapping.get(w, w) for w in Y_well]) if mapping is not None else None

    print("  -> Calculating truncation index...")
    pc_mask = (y_label == 'PC') if y_label is not None else np.zeros(len(Y_well), dtype=bool)

    if np.any(pc_mask):
        smoothed = uniform_filter1d(np.mean(X_2d_bs_active[pc_mask], axis=0).squeeze(), size=20)
        max_significant_index = int(np.argmin(smoothed))
        print(f"      -> [PC argmin] MSI={max_significant_index} ({int(pc_mask.sum())} PC wells)")
    else:
        ori_curve_dydx = np.array(get_derivatives(X_2d_bs_active, X_time))
        min_indices = np.argmin(ori_curve_dydx, axis=1)
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

    # Shift timestamps so they start from 0 (post-truncation, if any)
    X_time = X_time - X_time[0]

    # -------------------------------------------------------------
    # DROP PC (always on): remove PC wells now that truncation is done,
    # but keep a snapshot of their curves for cross-dataset PC-recentering.
    # -------------------------------------------------------------
    pc_wells_data = None
    if y_label is None:
        print("  [!] drop_pc: no LABEL_MAPPINGS for this experiment -- skipping PC removal.")
    elif not np.any(y_label == 'PC'):
        print("  [!] drop_pc: no PC-labeled wells found.")
    else:
        pc_mask2 = (y_label == 'PC')
        pc_curves_raw = X_2d_bs_active[pc_mask2].copy()
        pc_Y_well = Y_well[pc_mask2].copy()

        keep_mask = ~pc_mask2
        n_dropped = int(np.sum(pc_mask2))
        keep_idx = np.where(keep_mask)[0]
        X_2d_bs_active = X_2d_bs_active[keep_mask]
        Y_well = Y_well[keep_mask]
        y_label = y_label[keep_mask]
        for key, df in pixel_temp_dfs.items():
            if len(df) == len(keep_mask):
                pixel_temp_dfs[key] = df.iloc[keep_idx].reset_index(drop=True)
        print(f"  -> [drop_pc] Removed {n_dropped} PC samples. {len(Y_well)} samples remain.")

        pc_curves = {"ori_curves": pc_curves_raw}
        pc_curves["ori_curves_sg_p4"], _ = sg_p4_denoise_curves(pc_curves_raw)
        pc_curves["ori_curves_norm"] = normalize_curves_minmax(pc_curves["ori_curves"])
        pc_curves["ori_curves_sg_p4_norm"] = normalize_curves_minmax(pc_curves["ori_curves_sg_p4"])
        pc_wells_data = {"curves": pc_curves, "Y_well": pc_Y_well}
        print(f"  -> [drop_pc] Saved {len(pc_Y_well)} PC curve variants: {list(pc_curves.keys())}")

    # -------------------------------------------------------------
    # CURVE VARIANTS: raw, SG-p4 denoised, and min-max normalized siblings
    # -------------------------------------------------------------
    curves_dict = {"ori_curves": X_2d_bs_active}
    curves_dict["ori_curves_sg_p4"], sg_p4_optimal_w = sg_p4_denoise_curves(X_2d_bs_active)
    print(f"  -> SG p=4 optimal window: {sg_p4_optimal_w}")
    curves_dict["ori_curves_norm"] = normalize_curves_minmax(curves_dict["ori_curves"])
    curves_dict["ori_curves_sg_p4_norm"] = normalize_curves_minmax(curves_dict["ori_curves_sg_p4"])

    dataset_name = list(curves_dict.keys())
    dataset = list(curves_dict.values())

    print("  -> Extracting kinetic features for all curve variants (CPU bound)...")
    metadata_df = pd.DataFrame({
        "pixel_row_idx": pixel_temp_dfs["well_2d_nl_bs_active_df"]['pixel_row_idx'].values,
        "pixel_col_idx": pixel_temp_dfs["well_2d_nl_bs_active_df"]['pixel_col_idx'].values,
        "temp_group_idx": pixel_temp_dfs["well_2d_nl_bs_active_df"]['temp_group_idx'].values,
        "num_active_pixels_in_temp_group": pixel_temp_dfs["well_2d_nl_bs_active_df"]['num_active_pixels_in_temp_group'].values,
        "well_temp_lin2d_mean": pixel_temp_dfs["well_2d_nl_bs_active_df"]['well_temp_lin2d_mean'].values,
        "well_2d_temp_npr_mean": pixel_temp_dfs["well_2d_nl_bs_active_df"]['well_2d_temp_npr_mean'].values,
        "vref_idx": pixel_temp_dfs["well_2d_nl_bs_active_df"]['vref_idx'].values,
    })
    kinetic_features = [build_kinetic_features(c, X_time, metadata_df) for c in dataset]

    if args.run_outlier_detection:
        kinetic_features = run_outlier_detection_step(
            save_exp_path, dataset_name, dataset, Y_well, kinetic_features,
            save_plot_flag=save_plot_flag,
        )

    save_experiment_data(save_exp_path, curves_dict, dataset_name, dataset, kinetic_features,
                        pixel_temp_dfs, Y_well, X_time, max_significant_index, sg_p4_optimal_w,
                        pc_wells_data)

    print("  ✓ Experiment complete!\n")


if __name__ == "__main__":
    main()
