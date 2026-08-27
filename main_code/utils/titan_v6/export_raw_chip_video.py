"""
Export the raw chip animation as a video file (MP4 or GIF).
Uses the same full-chip reconstruction as the Plotly raw chip video.
Requires: imageio, and for MP4: imageio-ffmpeg (or system ffmpeg).
  pip install imageio imageio-ffmpeg
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def _reconstruct_full_chip_raw_3d(exp, nrows=290, ncols=204):
    """
    Reconstruct full-chip raw chem data (nrows, ncols, n_time) from experiment wells.
    Same layout as plot_wells_active_pixels_plotly (split_wells order).
    """
    n_wells = len(exp.wells_list)
    well0 = exp.wells_list[0]
    raw_3d = getattr(well0, "well_3d_npr", None)
    if raw_3d is None:
        return None, None
    n_time = raw_3d.shape[2]
    time_vect = np.asarray(well0.time_npr)
    if len(time_vect) != n_time:
        time_vect = time_vect[:n_time] if len(time_vect) >= n_time else np.arange(n_time, dtype=float)

    full_chip_3d = np.zeros((nrows, ncols, n_time), dtype=raw_3d.dtype)
    full_chip_3d[:] = np.nan

    if n_wells == 1:
        full_chip_3d[:, :, :] = raw_3d
        return full_chip_3d, time_vect

    if n_wells == 2:
        well_nrows = raw_3d.shape[0]
        full_chip_3d[:well_nrows, :, :] = exp.wells_list[0].well_3d_npr
        full_chip_3d[well_nrows:, :, :] = exp.wells_list[1].well_3d_npr
        return full_chip_3d, time_vect

    wells_per_row = 2
    rows_per_well = nrows // (n_wells // wells_per_row)
    cols_per_well = ncols // wells_per_row
    for well_idx, well in enumerate(exp.wells_list):
        w3d = well.well_3d_npr
        well_nrows, well_ncols = w3d.shape[0], w3d.shape[1]
        row_start = (well_idx // wells_per_row) * rows_per_well
        col_start = (well_idx % wells_per_row) * cols_per_well
        row_end = row_start + well_nrows
        col_end = col_start + well_ncols
        full_chip_3d[row_start:row_end, col_start:col_end, :] = w3d
    return full_chip_3d, time_vect


def export_raw_chip_video(
    exp,
    output_path,
    experiment_name="Experiment",
    fps=10,
    max_frames=200,
    from_settled=True,
    figsize_px=(1000, 800),
    dpi=100,
    vmin=0,
    vmax=1000,
    cmap="turbo",
):
    """
    Render the raw chip animation to a video file (MP4 preferred; falls back to GIF).

    Parameters
    ----------
    exp : Experiment
        Loaded experiment (e.g. from titan_load_and_preprocessing).
    output_path : str or Path
        Output file path. Use .mp4 for MP4 (needs ffmpeg) or .gif for GIF.
    experiment_name : str
        Used in the frame title.
    fps : int
        Frames per second in the output video.
    max_frames : int
        Maximum number of frames (time is downsampled if needed).
    from_settled : bool
        If True, start from idx_settled.
    figsize_px : tuple
        (width, height) of each frame in pixels.
    dpi : int
        DPI for matplotlib figure (figsize_inch = figsize_px / dpi).
    vmin, vmax : float
        Color scale range for the heatmap.
    cmap : str
        Matplotlib colormap name (e.g. 'turbo', 'viridis').

    Returns
    -------
    Path
        Path to the written file, or None on failure.
    """
    try:
        import imageio
    except ImportError:
        print("export_raw_chip_video requires imageio. Install with: pip install imageio imageio-ffmpeg")
        return None

    full_chip_3d, time_vect = _reconstruct_full_chip_raw_3d(exp)
    if full_chip_3d is None:
        print("Could not reconstruct full-chip raw data.")
        return None

    nrows, ncols, n_time = full_chip_3d.shape
    if from_settled and hasattr(exp.wells_list[0], "idx_settled"):
        idx_settled = exp.wells_list[0].idx_settled
        if idx_settled < n_time:
            full_chip_3d = full_chip_3d[:, :, idx_settled:]
            time_vect = time_vect[idx_settled:]
            n_time = full_chip_3d.shape[2]

    if n_time > max_frames:
        step = max(1, n_time // max_frames)
        indices = np.arange(0, n_time, step)
        if indices[-1] != n_time - 1:
            indices = np.r_[indices, n_time - 1]
        full_chip_3d = full_chip_3d[:, :, indices]
        time_vect = time_vect[indices]
        n_time = len(indices)
    time_min = time_vect / 60.0

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    w_in = figsize_px[0] / dpi
    h_in = figsize_px[1] / dpi
    fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=dpi)
    ax.set_axis_off()
    im = ax.imshow(
        full_chip_3d[:, :, 0],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        origin="upper",
    )
    title = ax.set_title(f"{experiment_name} — t = {time_min[0]:.2f} min", fontsize=12)
    plt.tight_layout(pad=0.5)

    frames_rgb = []
    for t in range(n_time):
        im.set_data(full_chip_3d[:, :, t])
        title.set_text(f"{experiment_name} — t = {time_min[t]:.2f} min")
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames_rgb.append(rgba[:, :, :3].copy())
    plt.close(fig)

    try:
        if output_path.suffix.lower() in (".mp4", ".mpeg", ".mov"):
            try:
                imageio.mimwrite(str(output_path), frames_rgb, fps=fps, codec="libx264", quality=8)
            except Exception:
                imageio.mimwrite(str(output_path), frames_rgb, fps=fps)
        else:
            imageio.mimwrite(str(output_path), frames_rgb, fps=fps)
    except Exception as e:
        print(f"Could not write video: {e}")
        return None

    print(f"Saved raw chip video: {output_path} ({n_time} frames, {fps} fps)")
    return output_path


if __name__ == "__main__":
    import sys

    _dir = Path(__file__).resolve().parent
    if str(_dir.parent) not in sys.path:
        sys.path.insert(0, str(_dir.parent))

    from titan_v6.load_and_preprocessing import titan_load_and_preprocessing

    n_wells = 6
    n_a_type = "v04"
    onedrive_path = Path("..", "..", "OneDrive - ProtonDx")
    exp_folder = Path(onedrive_path, "Data", "Air drying Trials")
    exp_path = Path(exp_folder, "D20251118_E00_C00_F4500KHz_U_AD_Human_6w_Cheeky_02")

    if not exp_path.exists():
        print(f"Experiment path not found: {exp_path}")
        print("Edit the path in export_raw_chip_video.py __main__ or pass your own.")
        sys.exit(1)

    exp = titan_load_and_preprocessing(
        exp_path,
        n_wells=n_wells,
        start_type="temperature",
        end_time_min=60,
        n_a_type=n_a_type,
        print_status=True,
    )
    out = Path(exp_folder, "Saved Videos", "AD_Human_6w_Cheeky_02_raw_chip.mp4")
    export_raw_chip_video(exp, out, experiment_name="AD_Human_6w_Cheeky_02", fps=10, max_frames=200)
