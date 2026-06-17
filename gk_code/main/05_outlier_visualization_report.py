import os
import sys
import argparse
import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config


def available_filter_columns(df):
    """Outlier filter columns from config.OUTLIER_FILTERS that exist in this dataframe."""
    return [c for c in config.OUTLIER_FILTERS if c is not None and c in df.columns]


def render_filter_plot(curves, labels, Y_well, label_map, filter_col):
    """n_wells x 2 grid (inliers | outliers) for a single outlier filter."""
    unique_wells = np.unique(Y_well)
    n_wells = len(unique_wells)

    fig, axes = plt.subplots(n_wells, 2, figsize=(8, 2.5 * n_wells), squeeze=False, sharey="row")

    for i, well in enumerate(unique_wells):
        well_mask = (Y_well == well)
        well_curves = curves[well_mask]
        well_labels = labels[well_mask]

        inlier_mask = well_labels == 1
        outlier_mask = well_labels == -1

        well_title = f"Well {well}"
        try:
            well_key = int(well)
        except (TypeError, ValueError):
            well_key = well
        if well_key in label_map:
            well_title += f" ({label_map[well_key]})"

        ax_in, ax_out = axes[i, 0], axes[i, 1]

        if inlier_mask.sum() > 0:
            ax_in.plot(well_curves[inlier_mask].T, color="#2980b9", alpha=0.2, linewidth=0.8)
        ax_in.set_title(f"{well_title} - Inliers (n={int(inlier_mask.sum())})", fontsize=9, fontweight="bold")

        if outlier_mask.sum() > 0:
            ax_out.plot(well_curves[outlier_mask].T, color="#e74c3c", alpha=0.4, linewidth=0.8)
        ax_out.set_title(f"{well_title} - Outliers (n={int(outlier_mask.sum())})", fontsize=9, fontweight="bold")

        for ax in (ax_in, ax_out):
            ax.set_xlabel("Time index", fontsize=8)
            ax.set_ylabel("Signal", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)

    fig.suptitle(filter_col, fontweight="bold", fontsize=12, y=1.0)
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def build_html(title, filter_names, img_buffers, save_path):
    html = f"""<html>
<head><title>{title}</title></head>
<body style="font-family: Arial, sans-serif; background-color: #f4f4f9; margin: 0; padding: 20px;">
    <h1 style="color: #333;">{title}</h1>
    <div style="display: flex; flex-wrap: nowrap; overflow-x: auto; gap: 20px; padding-bottom: 20px; align-items: flex-start;">
"""
    for name, buf in zip(filter_names, img_buffers):
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        html += f"""
        <div style="flex: 0 0 auto; background: white; padding: 10px; box-shadow: 0px 4px 10px rgba(0,0,0,0.1); border-radius: 8px;">
            <h3 style="text-align:center; margin: 0 0 10px 0;">{name}</h3>
            <img src="data:image/png;base64,{img_b64}" style="height: auto; max-width: 600px;">
        </div>
        """
    html += "    </div>\n</body></html>"

    with open(save_path, "w") as f:
        f.write(html)


def process_experiment(exp_path, force_rerun, curve_type="ori_curve"):
    out_dir = config.get_viz_dir(exp_path.parent, "outlier_visualisation")
    out_path = out_dir / f"{exp_path.name}_{curve_type}_outlier.html"

    if out_path.exists() and not force_rerun:
        print(f"  -> [SKIP] Report already exists: {out_path}")
        return

    state_path = os.path.join(exp_path, config.TRAINING_DATA_PATH)
    if not os.path.exists(state_path):
        print(f"  -> Skipping {exp_path.name}: '{config.TRAINING_DATA_PATH}' not found.")
        return

    print(f"\n{'#'*80}\nGENERATING OUTLIER VISUALIZATION REPORT FOR: {exp_path.name} (curve_type: {curve_type})\n{'#'*80}")
    try:
        state = joblib.load(state_path)
    except Exception as e:
        print(f"  -> Skipping {exp_path.name}: failed to load '{config.TRAINING_DATA_PATH}' ({e}).")
        return

    dataset_name = list(state["dataset_name"])
    dataset = state["dataset"]
    Y_well = np.asarray(state["Y_well"])
    kinetic_features = state["kinetic_features"]

    try:
        curve_idx, _ = config.resolve_curve_dataset_idx(curve_type, dataset_name)
    except ValueError as e:
        print(f"  -> [SKIP] {e}")
        return

    filter_cols = available_filter_columns(kinetic_features[curve_idx])
    if not filter_cols:
        print(f"  -> No outlier filter columns found for {exp_path.name}. Skipping.")
        return

    label_map = config.LABEL_MAPPINGS.get(exp_path.name, {})
    curves = dataset[curve_idx]

    img_buffers = []
    panel_titles = []
    for filter_col in filter_cols:
        print(f"  -> Rendering: {filter_col}")
        labels = kinetic_features[curve_idx][filter_col].values
        img_buffers.append(render_filter_plot(curves, labels, Y_well, label_map, filter_col))

        n_inliers = int((labels == 1).sum())
        n_outliers = int((labels == -1).sum())
        total = n_inliers + n_outliers
        pct_outlier = (n_outliers / total * 100) if total > 0 else 0.0
        panel_titles.append(
            f"{filter_col}<br>Inliers: {n_inliers}  |  Outliers: {n_outliers}  |  Outlier %: {pct_outlier:.2f}%"
        )

    os.makedirs(out_dir, exist_ok=True)
    build_html(f"Outlier Filter Visualization: {exp_path.name} ({curve_type})", panel_titles, img_buffers, out_path)
    print(f"  -> [SAVED] {out_path}")


if __name__ == "__main__":
    print(f"\n{'='*70}\n[RUNNING] {os.path.basename(__file__)}\n{'='*70}\n")
    parser = argparse.ArgumentParser(description="Static Outlier Filter Visualization Report")
    parser.add_argument("--task_id", type=int, default=0, help="Array Job ID")
    parser.add_argument("--exp_folder", type=str, default=config.DEFAULT_EXP_FOLDER)
    parser.add_argument("--force_rerun", action="store_true", help="Regenerate the HTML reports even if they already exist")
    parser.add_argument("--curve_type", type=str, nargs="+", default=["ori_curve", "ori_curve_avg"], help="Which curve dataset(s) to report on (e.g. 'ori_curve', 'ori_curve_avg', or a raw dataset_name entry)")
    args = parser.parse_args()

    exp_paths = sorted([
        Path(args.exp_folder, name) for name in os.listdir(args.exp_folder)
        if os.path.isdir(os.path.join(args.exp_folder, name)) and name not in config.EXCLUDED_FOLDERS
    ])
    if args.task_id >= len(exp_paths):
        print(f"Task ID {args.task_id} is out of bounds for {len(exp_paths)} folders. Exiting.")
        sys.exit(0)
    exp_path = exp_paths[args.task_id]

    for curve_type in args.curve_type:
        process_experiment(exp_path, args.force_rerun, curve_type=curve_type)
