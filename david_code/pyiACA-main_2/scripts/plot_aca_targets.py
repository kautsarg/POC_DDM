#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / ".cache"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


META_COLUMNS = {
    "Channel",
    "PrimerMix",
    "Target",
    "Assay",
    "Conc",
    "Exp_id",
    "MeltPeaks",
}

TARGET_COLORS = {
    "Hadv": "#2563eb",
    "IAV": "#dc2626",
    "IBV": "#059669",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot train_aca-compatible amplification curves grouped by Target, "
            "with one subplot per target."
        )
    )
    parser.add_argument(
        "dataset_csv",
        nargs="?",
        type=Path,
        default=None,
        help="ACA-ready CSV path. If omitted, the script will prompt for it.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path. Defaults to .cache/plots/<dataset>_targets.png.",
    )
    parser.add_argument(
        "--max-curves-per-target",
        type=int,
        default=0,
        help="Optional cap on curves plotted per target. 0 means plot all curves.",
    )
    return parser.parse_args()


def prompt_for_dataset() -> Path:
    raw = input("Please enter the ACA dataset CSV path: ").strip().strip('"').strip("'")
    if not raw:
        raise ValueError("No ACA dataset CSV path provided.")
    return Path(raw).expanduser().resolve()


def default_output_path(dataset_csv: Path) -> Path:
    out_dir = CACHE_DIR / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{dataset_csv.stem}_targets.png"


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    dataset_csv = args.dataset_csv.expanduser().resolve() if args.dataset_csv else prompt_for_dataset()
    output_path = args.output.expanduser().resolve() if args.output else default_output_path(dataset_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return dataset_csv, output_path


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols: list[tuple[float, str]] = []
    for col in df.columns:
        if col in META_COLUMNS:
            continue
        try:
            cols.append((float(col), col))
        except ValueError:
            continue
    if not cols:
        raise ValueError("No numeric amplification feature columns were found in the dataset.")
    cols.sort(key=lambda item: item[0])
    return [name for _, name in cols]


def target_base_name(target: str) -> str:
    return target.split("_", 1)[0]


def target_color(target: str) -> str:
    return TARGET_COLORS.get(target_base_name(target), "#7c3aed")


def prepare_axes(n_targets: int):
    ncols = min(3, n_targets)
    nrows = math.ceil(n_targets / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6.0 * ncols, 4.6 * nrows),
        dpi=220,
        sharex=True,
        sharey=True,
    )
    axes_array = np.atleast_1d(axes).ravel()
    return fig, axes_array


def plot_dataset(
    dataset_csv: Path,
    output_path: Path,
    max_curves_per_target: int = 0,
) -> tuple[Path, dict[str, int]]:
    df = pd.read_csv(dataset_csv, index_col=0, low_memory=False)
    if "Target" not in df.columns:
        raise ValueError("Dataset is missing the required Target column.")

    amp_cols = feature_columns(df)
    x = np.array([float(col) for col in amp_cols], dtype=float)
    targets = sorted(df["Target"].dropna().unique().tolist())
    if not targets:
        raise ValueError("Dataset does not contain any target values.")

    fig, axes = prepare_axes(len(targets))
    counts: dict[str, int] = {}

    global_y_min = float(df[amp_cols].min().min())
    global_y_max = float(df[amp_cols].max().max())
    y_pad = max((global_y_max - global_y_min) * 0.06, 0.02)

    for ax, target in zip(axes, targets):
        subset = df[df["Target"] == target]
        curves = subset.loc[:, amp_cols]
        counts[target] = len(curves)

        if max_curves_per_target and len(curves) > max_curves_per_target:
            curves_to_plot = curves.sample(max_curves_per_target, random_state=0)
        else:
            curves_to_plot = curves

        y = curves_to_plot.to_numpy(dtype=float).T
        base_color = target_color(target)

        ax.plot(
            x,
            y,
            color=base_color,
            alpha=0.015,
            linewidth=0.35,
            rasterized=True,
        )

        q25 = curves.quantile(0.25).to_numpy(dtype=float)
        q75 = curves.quantile(0.75).to_numpy(dtype=float)
        median = curves.median().to_numpy(dtype=float)
        mean = curves.mean().to_numpy(dtype=float)

        ax.fill_between(x, q25, q75, color=base_color, alpha=0.18, linewidth=0)
        ax.plot(x, median, color="#111827", linewidth=2.3, label="Median")
        ax.plot(x, mean, color=base_color, linewidth=1.8, linestyle="--", label="Mean")

        ax.set_title(f"{target}  (n={len(curves)})", fontsize=12, weight="bold")
        ax.grid(True, alpha=0.18)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(global_y_min - y_pad, global_y_max + y_pad)
        ax.legend(frameon=False, fontsize=9, loc="lower right")

    for ax in axes[len(targets):]:
        ax.remove()

    fig.suptitle(
        f"Amplification Curves by Target\n{dataset_csv.name}",
        fontsize=15,
        weight="bold",
        y=0.99,
    )
    fig.supxlabel("ACA Feature Index", fontsize=12)
    fig.supylabel("Normalized Fluorescence", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return output_path, counts


def main() -> None:
    args = parse_args()
    dataset_csv, output_path = resolve_paths(args)
    output_path, counts = plot_dataset(
        dataset_csv=dataset_csv,
        output_path=output_path,
        max_curves_per_target=args.max_curves_per_target,
    )

    print(f"Dataset: {dataset_csv}")
    print(f"Saved plot to: {output_path}")
    print("Target counts:")
    for target, count in counts.items():
        print(f"- {target}: {count}")


if __name__ == "__main__":
    main()
