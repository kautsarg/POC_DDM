import os
import sys
import argparse
from pathlib import Path

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_utils import CurveResampler, set_global_determinism
import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
set_global_determinism(0)


# ============================================================
# HELPERS
# ============================================================
def load_ori_curves(exp_path):
    """Load the 'ori_curves' dataset, label-mapped Y_well, and timestamps for one experiment folder."""
    data_path = os.path.join(exp_path, config.TRAINING_DATA_PATH)
    if not os.path.exists(data_path):
        print(f"  -> Skipping {exp_path.name}: '{data_path}' not found.")
        return None

    data = joblib.load(data_path)
    dataset_name = list(data["dataset_name"])
    if "ori_curves" not in dataset_name:
        print(f"  -> Skipping {exp_path.name}: 'ori_curves' not found.")
        return None
    idx = dataset_name.index("ori_curves")

    if exp_path.name not in config.LABEL_MAPPINGS:
        print(f"  -> Skipping {exp_path.name}: no LABEL_MAPPINGS entry found.")
        return None

    return {
        "curves": data["dataset"][idx],
        "Y_well_raw": np.asarray(data["Y_well"]),
        "well_to_label": config.LABEL_MAPPINGS[exp_path.name],
        "timestamps": np.asarray(data["timestamps"], dtype=float),
    }


def plot_resampling_check(parts, resampler, well_to_label, save_prefix, max_lines=30):
    """
    Diagnostic plots comparing curves before vs. after CurveResampler resampling.

    `parts` is a list of dicts, one per dataset, each with keys:
      "dataset_id", "t_zeroed" (1D, native zeroed timestamps), "curves_orig"
      (N, T_native), "curves" (N, len(resampler.t_grid)), "Y_well_raw" (N,).

    Saves two figures (rows = datasets):
      - f"{save_prefix}_overall.png": mean +/- std curve, before vs. after,
        with a vertical line marking the common grid's duration.
      - f"{save_prefix}_by_well.png": per-well mean curve, colored by the
        well's mapped target label, before vs. after.
    """
    n = len(parts)
    t_grid = resampler.t_grid
    duration = t_grid[-1]
    rng = np.random.RandomState(0)

    # --- Overall: mean +/- std, before vs. after ---
    fig, axes = plt.subplots(n, 2, figsize=(10, 3 * n), squeeze=False)
    for i, p in enumerate(parts):
        t_orig, c_orig, c_new = p["t_zeroed"], p["curves_orig"], p["curves"]
        ax_before, ax_after = axes[i, 0], axes[i, 1]

        sample_idx = rng.choice(len(c_orig), size=min(max_lines, len(c_orig)), replace=False)
        for idx in sample_idx:
            ax_before.plot(t_orig, c_orig[idx], color="grey", alpha=0.15, linewidth=0.6)
            ax_after.plot(t_grid, c_new[idx], color="grey", alpha=0.15, linewidth=0.6)

        mean_orig, std_orig = c_orig.mean(axis=0), c_orig.std(axis=0)
        mean_new, std_new = c_new.mean(axis=0), c_new.std(axis=0)

        ax_before.plot(t_orig, mean_orig, color="#2980b9", linewidth=1.5)
        ax_before.fill_between(t_orig, mean_orig - std_orig, mean_orig + std_orig, color="#2980b9", alpha=0.2)
        ax_before.axvline(duration, color="red", linestyle="--", linewidth=1, label=f"grid duration={duration:.3g}")
        ax_before.legend(fontsize=7, loc="upper right")

        ax_after.plot(t_grid, mean_new, color="#2980b9", linewidth=1.5)
        ax_after.fill_between(t_grid, mean_new - std_new, mean_new + std_new, color="#2980b9", alpha=0.2)

        xmax = max(t_orig[-1], duration)
        ax_before.set_xlim(0, xmax)
        ax_after.set_xlim(0, xmax)

        ax_before.set_title(f"{p['dataset_id']} - Before (n={len(c_orig)}, {len(t_orig)} pts)", fontsize=9, fontweight="bold")
        ax_after.set_title(f"{p['dataset_id']} - After ({len(t_grid)} pts)", fontsize=9, fontweight="bold")

        for ax in (ax_before, ax_after):
            ax.set_xlabel("Time (zeroed)", fontsize=8)
            ax.set_ylabel("Signal", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)

    fig.suptitle("Curve Resampling Check: Overall (mean +/- std)", fontweight="bold", fontsize=12, y=1.0)
    plt.tight_layout()
    fig.savefig(f"{save_prefix}_overall.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # --- Per-well mean, colored by mapped target label ---
    labels = sorted(set(well_to_label.values()))
    label_colors = config.get_palette(labels)

    fig, axes = plt.subplots(n, 2, figsize=(10, 3 * n), squeeze=False)
    for i, p in enumerate(parts):
        t_orig, c_orig, c_new, well_idx = p["t_zeroed"], p["curves_orig"], p["curves"], p["Y_well_raw"]
        ax_before, ax_after = axes[i, 0], axes[i, 1]

        seen_labels = set()
        for w in np.unique(well_idx):
            mask = well_idx == w
            label = well_to_label.get(w, str(w))
            color = label_colors.get(label, "grey")
            show_label = f"{label} (well {w})" if label not in seen_labels else None
            seen_labels.add(label)

            ax_before.plot(t_orig, c_orig[mask].mean(axis=0), color=color, linewidth=1.2, label=show_label)
            ax_after.plot(t_grid, c_new[mask].mean(axis=0), color=color, linewidth=1.2)

        ax_before.axvline(duration, color="red", linestyle="--", linewidth=1)

        xmax = max(t_orig[-1], duration)
        ax_before.set_xlim(0, xmax)
        ax_after.set_xlim(0, xmax)

        ax_before.set_title(f"{p['dataset_id']} - Before (per-well mean)", fontsize=9, fontweight="bold")
        ax_after.set_title(f"{p['dataset_id']} - After (per-well mean)", fontsize=9, fontweight="bold")
        ax_before.legend(fontsize=6, loc="upper right")

        for ax in (ax_before, ax_after):
            ax.set_xlabel("Time (zeroed)", fontsize=8)
            ax.set_ylabel("Signal", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)

    fig.suptitle("Curve Resampling Check: Per-Well Mean (colored by target label)", fontweight="bold", fontsize=12, y=1.0)
    plt.tight_layout()
    fig.savefig(f"{save_prefix}_by_well.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize curves before vs. after CurveResampler resampling")
    parser.add_argument("--task_id", type=int, default=0, help="Array Job ID -> index into CROSS_DATASET_GROUPS")
    parser.add_argument("--exp_folder", type=str, default=config.DEFAULT_EXP_FOLDER)
    args = parser.parse_args()

    group_names = list(config.CROSS_DATASET_GROUPS.keys())
    if not group_names:
        print("CROSS_DATASET_GROUPS is empty in config.py. Define at least one group to run this script.")
        sys.exit(0)
    if args.task_id >= len(group_names):
        print(f"Task ID {args.task_id} is out of bounds for {len(group_names)} groups. Exiting.")
        sys.exit(0)

    group_name = group_names[args.task_id]
    folder_names = config.CROSS_DATASET_GROUPS[group_name]
    exp_paths = [Path(args.exp_folder, name) for name in folder_names]

    out_dir = Path(args.exp_folder) / "cross_dataset_cv" / group_name
    resampler_path = out_dir / config.CROSS_DATASET_RESAMPLER_PATH
    if not resampler_path.exists():
        print(f"  -> Resampler not found at {resampler_path}. Run 03b_par-cross_dataset_training.py first.")
        sys.exit(0)
    resampler = joblib.load(resampler_path)

    print(f"\n{'#'*80}\nRESAMPLING CHECK FOR GROUP: {group_name}\nFolders: {folder_names}\n{'#'*80}")

    parts = []
    ref_mapping = None
    for exp_path in exp_paths:
        d = load_ori_curves(exp_path)
        if d is None:
            continue
        if ref_mapping is None:
            ref_mapping = d["well_to_label"]
        d["dataset_id"] = exp_path.name
        d["t_zeroed"] = d["timestamps"] - d["timestamps"][0]
        d["curves_orig"] = d["curves"]
        d["curves"] = resampler.transform(d["timestamps"], d["curves"])
        parts.append(d)

    if not parts:
        print(f"  -> No usable datasets found for group '{group_name}'.")
        sys.exit(0)

    save_prefix = out_dir / "resampling_check"
    plot_resampling_check(parts, resampler, ref_mapping, str(save_prefix))
    print(f"  [*] Saved resampling check plots -> {save_prefix}_overall.png / _by_well.png")
