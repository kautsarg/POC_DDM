"""
Overlay traces from multiple experiment folders on one Plotly figure.

Time axis is aligned at the vref jump using each run's ``Well.idx_settled`` (same
logic as ``find_start_end_temp`` / ``_time_minutes_for_well(..., zero_at="settled")``).

Typical use (standalone):

    python overlay_experiments_plotly.py \\
        --exp-folder "C:\\...\\Manifold test" \\
        --names manifold_test_03 manifold_test_05

Or enable ``OVERLAY_CONFIG`` at the bottom of ``main_DNA.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import plotly.graph_objects as go

from load_and_preprocessing import titan_load_and_preprocessing
from plot_derivatives_plotly import (
    COLORBLIND_WELL_COLORS,
    _align_x_y,
    _time_minutes_for_well,
    sanitize_filename,
)

PathLike = Union[str, Path]

# linearized_mean: post-settled active-pixel mean (idx_settled:idx_end)
# linearized_preheat: same window but includes pre-settled samples, t=0 at vref jump
# temperature: well_temp_mean_then_lin (v06 uses firmware meta override when set)
SIGNAL_CHOICES = ("linearized_mean", "linearized_preheat", "temperature")


def experiment_label(exp_path: PathLike) -> str:
    path_str = str(exp_path)
    if "KHz_U_" in path_str:
        return path_str.split("KHz_U_")[1].rstrip("\\/")
    return Path(exp_path).name


def vref_jump_index(well) -> int:
    """Frame index where preprocessing anchored settled time (vref jump)."""
    return int(getattr(well, "idx_settled", 0))


def extract_aligned_trace(
    exp,
    well_idx: int = 0,
    signal: str = "linearized_mean",
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Return (time_min, y, idx_settled) with time zeroed at the vref jump.
    """
    if signal not in SIGNAL_CHOICES:
        raise ValueError(f"signal must be one of {SIGNAL_CHOICES}, got {signal!r}")

    wells = exp.wells_list
    if well_idx < 0 or well_idx >= len(wells):
        raise IndexError(f"well_idx {well_idx} out of range (n_wells={len(wells)})")

    well = wells[well_idx]
    idx_jump = vref_jump_index(well)
    x = _time_minutes_for_well(well, zero_at="settled")

    if signal == "linearized_mean":
        y = np.asarray(well.well_2d_bs_active_mean, dtype=float)
    elif signal == "linearized_preheat":
        y_full = np.asarray(well.well_2d_start_bs_active_mean, dtype=float)
        offset = max(0, idx_jump - int(getattr(well, "idx_start", 0)))
        y = y_full[offset:]
    else:
        y = np.asarray(well.well_temp_mean_then_lin, dtype=float)

    x, y = _align_x_y(x, y)
    return x, y, idx_jump


def plot_experiments_overlay(
    exp_paths: Sequence[PathLike],
    *,
    n_wells: int = 10,
    n_a_type: str = "v06",
    ref_idx: int = 0,
    start_type: str = "temperature",
    end_time_min: int = 60,
    signal: str = "linearized_mean",
    wells: Optional[Sequence[int]] = None,
    save_path: Optional[PathLike] = None,
    output_name: str = "experiments_overlay_vref_jump",
    show: bool = True,
    print_status: bool = True,
) -> go.Figure:
    """
    Load each experiment, extract aligned traces, and overlay on one figure.
    """
    exp_paths = [Path(p) for p in exp_paths]
    if not exp_paths:
        raise ValueError("exp_paths is empty")

    if wells is None:
        well_indices = list(range(n_wells))
    else:
        well_indices = [int(w) - 1 for w in wells]

    fig = go.Figure()
    palette = COLORBLIND_WELL_COLORS

    for exp_i, exp_path in enumerate(exp_paths):
        label = experiment_label(exp_path)
        if print_status:
            print(f"\n[overlay] Loading {label} ({exp_path})")

        try:
            exp = titan_load_and_preprocessing(
                exp_path,
                n_wells=n_wells,
                start_type=start_type,
                end_time_min=end_time_min,
                n_a_type=n_a_type,
                print_status=print_status,
                plt_gain_calib=False,
                save_gain_calib=False,
                ref_idx=ref_idx,
            )
        except Exception as exc:
            print(f"[overlay] Skipping {label}: {exc}")
            continue

        for well_idx in well_indices:
            if well_idx >= len(exp.wells_list):
                continue
            try:
                x, y, idx_jump = extract_aligned_trace(exp, well_idx, signal=signal)
            except Exception as exc:
                print(f"[overlay] {label} well {well_idx + 1}: {exc}")
                continue
            if x.size == 0 or y.size == 0:
                print(f"[overlay] {label} well {well_idx + 1}: empty trace")
                continue

            trace_name = f"{label} — W{well_idx + 1}"
            color = palette[(exp_i * max(1, len(well_indices)) + well_idx) % len(palette)]
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=trace_name,
                    line=dict(color=color, width=2),
                    legendgroup=label,
                    meta=dict(
                        experiment=label,
                        well=well_idx + 1,
                        idx_settled=idx_jump,
                        exp_path=str(exp_path),
                    ),
                )
            )
            if print_status:
                print(
                    f"  + {trace_name}: idx_settled={idx_jump}, "
                    f"n={len(x)}, t=0..{x[-1]:.2f} min"
                )

    y_titles = {
        "linearized_mean": "Linearized signal (active mean, baseline at settled)",
        "linearized_preheat": "Linearized signal (preheat window, t=0 at vref jump)",
        "temperature": "Temperature (mean-then-lin, baseline at vref jump)",
    }
    fig.update_layout(
        title=f"{output_name} — aligned at vref jump (idx_settled)",
        xaxis_title="Time after vref jump (minutes)",
        yaxis_title=y_titles.get(signal, signal),
        hovermode="x unified",
        width=1400,
        height=700,
        font=dict(size=12),
    )
    fig.update_xaxes(showgrid=True, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridcolor="lightgray")

    if save_path:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        html_path = save_dir / f"{sanitize_filename(output_name)}.html"
        fig.write_html(str(html_path))
        print(f"[overlay] Saved: {html_path}")

    if show:
        fig.show()

    return fig


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--exp-paths",
        nargs="+",
        type=Path,
        help="Full paths to experiment directories",
    )
    p.add_argument(
        "--exp-folder",
        type=Path,
        help="Parent folder; use with --names to build paths",
    )
    p.add_argument(
        "--names",
        nargs="+",
        help="Folder name suffixes after KHz_U_ (used with --exp-folder)",
    )
    p.add_argument("--n-wells", type=int, default=10)
    p.add_argument("--n-a-type", default="v06")
    p.add_argument("--ref-idx", type=int, default=0)
    p.add_argument("--signal", choices=SIGNAL_CHOICES, default="linearized_mean")
    p.add_argument("--wells", nargs="+", type=int, help="1-based well numbers (default: all)")
    p.add_argument("--save-path", type=Path, default=None)
    p.add_argument("--output-name", default="experiments_overlay_vref_jump")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def resolve_exp_paths(args: argparse.Namespace) -> List[Path]:
    if args.exp_paths:
        return list(args.exp_paths)
    if args.exp_folder and args.names:
        folder = Path(args.exp_folder)
        paths = []
        for name in args.names:
            hits = list(folder.glob(f"*KHz_U_{name}"))
            if not hits:
                hits = list(folder.glob(f"*{name}*"))
            if not hits:
                raise FileNotFoundError(f"No experiment folder matching {name!r} under {folder}")
            paths.append(hits[0])
        return paths
    raise ValueError("Provide --exp-paths or both --exp-folder and --names")


if __name__ == "__main__":
    args = _parse_args()
    paths = resolve_exp_paths(args)
    plot_experiments_overlay(
        paths,
        n_wells=args.n_wells,
        n_a_type=args.n_a_type,
        ref_idx=args.ref_idx,
        signal=args.signal,
        wells=args.wells,
        save_path=args.save_path,
        output_name=args.output_name,
        show=not args.no_show,
    )
