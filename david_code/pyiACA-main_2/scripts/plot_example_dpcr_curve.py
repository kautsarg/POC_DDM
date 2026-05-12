from __future__ import annotations

import csv
import html
from pathlib import Path
from statistics import mean, median


def load_partition_curves(path: Path) -> tuple[list[float], list[list[float]]]:
    rows: list[tuple[list[float], list[float]]] = []
    with path.open(newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue
            vals = [float(x) for x in row if x != ""]
            x = vals[0::2]
            y = vals[1::2]
            rows.append((x, y))

    if not rows:
        raise ValueError(f"No data found in {path}")

    cycles = [r[0][0] for r in rows]
    partition_count = len(rows[0][1])
    curves = [[row[1][idx] for row in rows] for idx in range(partition_count)]
    return cycles, curves


def choose_representative_partition(curves: list[list[float]]) -> int:
    final_vals = [curve[-1] for curve in curves]
    positive_idx = [idx for idx, value in enumerate(final_vals) if value > 0.8]
    if not positive_idx:
        return max(range(len(final_vals)), key=final_vals.__getitem__)

    target = median(final_vals[idx] for idx in positive_idx)
    return min(positive_idx, key=lambda idx: abs(final_vals[idx] - target))


def mean_curve(curves: list[list[float]]) -> list[float]:
    return [mean(values) for values in zip(*curves)]


def scale_points(
    xs: list[float],
    ys: list[float],
    left: float,
    top: float,
    width: float,
    height: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> str:
    coords: list[str] = []
    for x, y in zip(xs, ys):
        px = left + (x - x_min) / (x_max - x_min) * width
        py = top + height - (y - y_min) / (y_max - y_min) * height
        coords.append(f"{px:.2f},{py:.2f}")
    return " ".join(coords)


def build_svg(cycles: list[float], curves: list[list[float]], rep_idx: int) -> str:
    width = 1000
    height = 620
    left = 90
    right = 40
    top = 90
    bottom = 80
    plot_w = width - left - right
    plot_h = height - top - bottom

    sample_count = min(20, len(curves))
    sample_idx = [
        round(i * (len(curves) - 1) / (sample_count - 1)) if sample_count > 1 else 0
        for i in range(sample_count)
    ]
    sample_curves = [curves[idx] for idx in sample_idx]
    rep_curve = curves[rep_idx]
    avg_curve = mean_curve(curves)

    all_y = [y for curve in sample_curves + [rep_curve, avg_curve] for y in curve]
    y_min = min(all_y)
    y_max = max(all_y)
    pad = max((y_max - y_min) * 0.08, 0.05)
    y_min -= pad
    y_max += pad

    x_min = min(cycles)
    x_max = max(cycles)

    lines: list[str] = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    lines.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    lines.append(
        f'<text x="{width/2:.0f}" y="36" text-anchor="middle" font-size="24" font-family="Arial, sans-serif" font-weight="700" fill="#111827">'
        "Example dPCR Amplification Curve</text>"
    )
    lines.append(
        f'<text x="{width/2:.0f}" y="64" text-anchor="middle" font-size="16" font-family="Arial, sans-serif" fill="#4B5563">'
        "Well 10 with 770 partition curves</text>"
    )

    for tick in range(int(x_min), int(x_max) + 1, 5):
        x = left + (tick - x_min) / (x_max - x_min) * plot_w
        lines.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#E5E7EB" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{x:.2f}" y="{top + plot_h + 28}" text-anchor="middle" font-size="12" font-family="Arial, sans-serif" fill="#6B7280">{tick}</text>'
        )

    y_ticks = 6
    for idx in range(y_ticks + 1):
        tick_value = y_min + (y_max - y_min) * idx / y_ticks
        y = top + plot_h - (tick_value - y_min) / (y_max - y_min) * plot_h
        lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#E5E7EB" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="12" font-family="Arial, sans-serif" fill="#6B7280">{tick_value:.2f}</text>'
        )

    lines.append(
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#9CA3AF" stroke-width="1.2"/>'
    )

    for curve in sample_curves:
        points = scale_points(cycles, curve, left, top, plot_w, plot_h, x_min, x_max, y_min, y_max)
        lines.append(
            f'<polyline fill="none" stroke="#C9CED6" stroke-width="1.1" stroke-linejoin="round" stroke-linecap="round" opacity="0.75" points="{points}"/>'
        )

    rep_points = scale_points(cycles, rep_curve, left, top, plot_w, plot_h, x_min, x_max, y_min, y_max)
    avg_points = scale_points(cycles, avg_curve, left, top, plot_w, plot_h, x_min, x_max, y_min, y_max)
    lines.append(
        f'<polyline fill="none" stroke="#2563EB" stroke-width="3.2" stroke-linejoin="round" stroke-linecap="round" points="{rep_points}"/>'
    )
    lines.append(
        f'<polyline fill="none" stroke="#111827" stroke-width="2.4" stroke-dasharray="8 6" stroke-linejoin="round" stroke-linecap="round" points="{avg_points}"/>'
    )

    legend_x = left + plot_w - 260
    legend_y = top + 20
    lines.append(f'<rect x="{legend_x}" y="{legend_y}" width="235" height="74" rx="10" fill="#FFFFFF" stroke="#D1D5DB"/>')
    lines.append(
        f'<line x1="{legend_x + 18}" y1="{legend_y + 22}" x2="{legend_x + 56}" y2="{legend_y + 22}" stroke="#C9CED6" stroke-width="2"/>'
    )
    lines.append(
        f'<text x="{legend_x + 66}" y="{legend_y + 27}" font-size="13" font-family="Arial, sans-serif" fill="#4B5563">Sample partition curves</text>'
    )
    lines.append(
        f'<line x1="{legend_x + 18}" y1="{legend_y + 43}" x2="{legend_x + 56}" y2="{legend_y + 43}" stroke="#2563EB" stroke-width="3"/>'
    )
    lines.append(
        f'<text x="{legend_x + 66}" y="{legend_y + 48}" font-size="13" font-family="Arial, sans-serif" fill="#111827">Representative partition #{rep_idx + 1}</text>'
    )
    lines.append(
        f'<line x1="{legend_x + 18}" y1="{legend_y + 64}" x2="{legend_x + 56}" y2="{legend_y + 64}" stroke="#111827" stroke-width="2.4" stroke-dasharray="8 6"/>'
    )
    lines.append(
        f'<text x="{legend_x + 66}" y="{legend_y + 69}" font-size="13" font-family="Arial, sans-serif" fill="#111827">Well mean curve</text>'
    )

    lines.append(
        f'<text x="{left + plot_w/2:.2f}" y="{height - 24}" text-anchor="middle" font-size="15" font-family="Arial, sans-serif" fill="#111827">PCR Cycle</text>'
    )
    lines.append(
        f'<text x="26" y="{top + plot_h/2:.2f}" transform="rotate(-90 26 {top + plot_h/2:.2f})" text-anchor="middle" font-size="15" font-family="Arial, sans-serif" fill="#111827">Fluorescence Value</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines)


def main() -> None:
    src = Path("/Users/david/Desktop/Raw Data/10.txt")
    out_dir = Path("/Users/david/Downloads/pyiACA-main 2/.cache/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "example_dpcr_amplification_curve_well10.svg"

    cycles, curves = load_partition_curves(src)
    rep_idx = choose_representative_partition(curves)
    svg = build_svg(cycles, curves, rep_idx)
    out_path.write_text(svg)

    print(out_path)
    print(f"representative_partition={rep_idx + 1}")
    print(f"cycle_count={len(cycles)} partition_count={len(curves)}")


if __name__ == "__main__":
    main()
