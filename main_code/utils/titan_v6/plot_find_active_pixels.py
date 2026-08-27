"""
Load and plot active pixels from ``*find_active*.bin`` when no readout is available.

Lacewing writes one uint16 per pixel (Fortran-order on disk, C-order for display).
Active chemical pixels are marked with value 400 (same convention as titan_load_and_preprocessing).
Inactive / background pixels are typically 1023; 511 marks the temperature-pixel grid.

For the manifold (10×3) chip, well size and gutter spacing are **not** the same as the
6-well ``dy_wells=25, dx_wells=25`` defaults — auto-detection from the file is preferred.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

import load_functions as load
from plot_derivatives_plotly import sanitize_filename

NROWS = 290
NCOLS = 204
LACEWING_ACTIVE_VALUE = 400
INACTIVE_VALUE = 1023
TEMP_PIXEL_VALUE = 511

# Measured manifold defaults (auto-detect refines these per file).
MANIFOLD_WELL_LAYOUT = {
    "n_wells": 30,
    "Wr": 10,
    "Wc": 3,
    "col_origin": 9,
    "row_origin": 3,
    "x_wells": 43,
    "y_wells": 11,
    "dx_wells": 28,
    "dy_wells": 17,
    "temp_ref": 410,
}

SIX_WELL_LAYOUT = {
    "n_wells": 6,
    "Wr": 3,
    "Wc": 2,
    "y_offset": 0,
    "col_origin": 0,
    "row_origin": 0,
    "dy_wells": 25,
    "dx_wells": 25,
    "temp_ref": 410,
    "assignment": "firmware",  # n_wells==6 || n_wells==11 gutter logic in C
}

# 11-channel single-column layout (same C branch as n_wells==6).
ELEVEN_WELL_LAYOUT = {
    "n_wells": 11,
    "Wr": 11,
    "Wc": 1,
    "y_offset": 0,
    "col_origin": 0,
    "row_origin": 0,
    "dy_wells": 17,
    "dx_wells": 25,
    "temp_ref": 410,
    "assignment": "firmware",
}


def _uses_firmware_assignment(layout: dict) -> bool:
    if layout.get("assignment") == "firmware":
        return True
    n = int(layout.get("n_wells", 0))
    return n in (6, 11)


def firmware_assign_well(
    row: int,
    col: int,
    geom: "CWellGeometry",
    y_offset: Optional[int] = None,
) -> int:
    """
    Mirror Lacewing C (``n_wells==11 || n_wells==6`` branch):

        current_well_row = floor((row+y_offset)/(y_wells+dy_wells)) + 1;
        gutter row if (row+y_offset) % stride is outside [dy/2, dy/2+y_wells]
        current_well_col = floor(col/(x_wells+dx_wells)) + 1;
        gutter col if col % stride is outside [dx/2, dx/2+x_wells]
        current_well = (row-1)*Wc + col  (1-based), else 0
    """
    yo = int(geom.y_offset if y_offset is None else y_offset)
    stride_y = geom.y_wells + geom.dy_wells
    stride_x = geom.x_wells + geom.dx_wells
    dy_half = geom.dy_wells // 2
    dx_half = geom.dx_wells // 2

    r_adj = row + yo
    rem_y = r_adj % stride_y
    well_row = (r_adj // stride_y) + 1
    if rem_y < dy_half or rem_y > dy_half + geom.y_wells:
        well_row = 0

    rem_x = col % stride_x
    well_col = (col // stride_x) + 1
    if rem_x < dx_half or rem_x > dx_half + geom.x_wells:
        well_col = 0

    if well_row > 0 and well_col > 0 and well_row <= geom.Wr and well_col <= geom.Wc:
        return (well_row - 1) * geom.Wc + well_col
    return 0


def firmware_well_index_map(
    geom: "CWellGeometry",
    nrows: int = NROWS,
    ncols: int = NCOLS,
    y_offset: Optional[int] = None,
) -> np.ndarray:
    """Per-pixel well index (0 = gutter/margin, 1..n_wells = well)."""
    yo = int(geom.y_offset if y_offset is None else y_offset)
    rows = np.arange(nrows, dtype=np.int32)[:, None]
    cols = np.arange(ncols, dtype=np.int32)[None, :]

    stride_y = geom.y_wells + geom.dy_wells
    stride_x = geom.x_wells + geom.dx_wells
    dy_half = geom.dy_wells // 2
    dx_half = geom.dx_wells // 2

    r_adj = rows + yo
    rem_y = r_adj % stride_y
    well_row = (r_adj // stride_y) + 1
    in_row_gutter = (rem_y < dy_half) | (rem_y > dy_half + geom.y_wells)
    well_row = np.where(in_row_gutter, 0, well_row)

    rem_x = cols % stride_x
    well_col = (cols // stride_x) + 1
    in_col_gutter = (rem_x < dx_half) | (rem_x > dx_half + geom.x_wells)
    well_col = np.where(in_col_gutter, 0, well_col)

    valid = (
        (well_row > 0)
        & (well_col > 0)
        & (well_row <= geom.Wr)
        & (well_col <= geom.Wc)
    )
    well_idx = np.where(valid, (well_row - 1) * geom.Wc + well_col, 0)
    return well_idx.astype(np.int32)


def spans_from_well_index_map(well_map: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Bounding box per well index from firmware assignment map."""
    spans: List[Tuple[int, int, int, int]] = []
    for w in range(1, int(well_map.max()) + 1):
        mask = well_map == w
        if not np.any(mask):
            continue
        rs, cs = np.where(mask)
        spans.append((int(rs.min()), int(rs.max()) + 1, int(cs.min()), int(cs.max()) + 1))
    return spans


def format_firmware_assignment_c() -> str:
    """Copy-paste C well-index assignment (n_wells==11 || n_wells==6)."""
    return """    else if (n_wells==11 || n_wells==6)
    {
        // Calculate well row and col
        current_well_row = floor((row+y_offset)/(y_wells+dy_wells)) + 1;
        if (((row+y_offset) % (y_wells+dy_wells) < dy_wells/2) || ((row+y_offset) % (y_wells+dy_wells) > dy_wells/2 + y_wells))
        {
            current_well_row = 0;
        }

        current_well_col = floor(col/(x_wells+dx_wells)) + 1;
        if ((col % (x_wells+dx_wells) < dx_wells/2) || (col % (x_wells+dx_wells) > dx_wells/2 + x_wells))
        {
            current_well_col = 0;
        }

        // Convert (well_row,well_col) to well index
        if ((current_well_row > 0) && (current_well_col > 0))
        {
            current_well = (current_well_row-1)*Wc+current_well_col;
        }
        else
        {
            current_well = 0;
        }
    }"""


def _contiguous_segments(indices: np.ndarray, min_width: int = 1) -> List[Tuple[int, int]]:
    """Convert sorted index array to [start, end) segments."""
    if indices.size == 0:
        return []
    segs: List[Tuple[int, int]] = []
    start = int(indices[0])
    prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx != prev + 1:
            if prev + 1 - start >= min_width:
                segs.append((start, prev + 1))
            start = idx
        prev = idx
    if prev + 1 - start >= min_width:
        segs.append((start, prev + 1))
    return segs


def detect_manifold_grid(values_2d: np.ndarray) -> dict:
    """
    Infer 10×3 manifold well grid from find_active uint16 map.

    Well columns: wide bands where 1023 appears inside the column stack.
    Well row bands: inside those columns, rows dominated by 400 (firmware mark).
    Vertical gutters between columns are mostly 400 — these sit *between* col bands.
    """
    values_2d = np.asarray(values_2d)
    nrows, ncols = values_2d.shape

    inactive_col = (values_2d == INACTIVE_VALUE).sum(axis=0)
    col_thresh = max(50, 0.15 * inactive_col.max())
    col_bands = _contiguous_segments(np.where(inactive_col >= col_thresh)[0], min_width=20)
    if len(col_bands) < 2:
        raise ValueError("Could not detect manifold column bands from find_active data")

    col_bands = sorted(col_bands, key=lambda s: s[1] - s[0], reverse=True)[:3]
    col_bands = sorted(col_bands, key=lambda s: s[0])
    wc = len(col_bands)
    x_sizes = [b[1] - b[0] for b in col_bands]
    x_wells = int(round(float(np.median(x_sizes))))
    col_origin = int(col_bands[0][0])
    if wc > 1:
        dx_wells = int(round(float(np.median([col_bands[i + 1][0] - col_bands[i][1] for i in range(wc - 1)]))))
    else:
        dx_wells = 0

    col_union = np.zeros(ncols, dtype=bool)
    for c0, c1 in col_bands:
        col_union[c0:c1] = True

    inside = values_2d[:, col_union]
    active_frac = (inside == LACEWING_ACTIVE_VALUE).mean(axis=1)
    row_thresh = max(0.35, 0.5 * float(np.percentile(active_frac, 75)))
    row_bands = _contiguous_segments(np.where(active_frac >= row_thresh)[0], min_width=4)
    if len(row_bands) < 5:
        raise ValueError("Could not detect manifold row bands from find_active data")

    y_sizes = [b[1] - b[0] for b in row_bands]
    y_wells = int(round(float(np.median(y_sizes))))
    row_origin = int(row_bands[0][0])
    wr = len(row_bands)
    if wr > 1:
        dy_wells = int(round(float(np.median([row_bands[i + 1][0] - row_bands[i][1] for i in range(wr - 1)]))))
    else:
        dy_wells = 0

    return {
        "n_wells": wr * wc,
        "Wr": wr,
        "Wc": wc,
        "col_origin": col_origin,
        "row_origin": row_origin,
        "x_wells": x_wells,
        "y_wells": y_wells,
        "dx_wells": dx_wells,
        "dy_wells": dy_wells,
        "temp_ref": 410,
        "_detected_col_bands": col_bands,
        "_detected_row_bands": row_bands,
    }


@dataclass
class CWellGeometry:
    """Firmware-style well grid parameters and derived pixel sizes."""

    n_wells: int
    Wr: int
    Wc: int
    dy_wells: int
    dx_wells: int
    temp_ref: int
    ROWS: int
    COLS: int
    x_wells: int
    y_wells: int
    n_pixel_well: int
    col_origin: int = 0
    row_origin: int = 0
    y_offset: int = 0
    use_firmware_assignment: bool = False

    @classmethod
    def from_layout(
        cls,
        layout: dict,
        rows: int = NROWS,
        cols: int = NCOLS,
    ) -> "CWellGeometry":
        wr = int(layout["Wr"])
        wc = int(layout["Wc"])
        dy = int(layout["dy_wells"])
        dx = int(layout["dx_wells"])
        col_origin = int(layout.get("col_origin", 0))
        row_origin = int(layout.get("row_origin", 0))
        y_offset = int(layout.get("y_offset", row_origin))

        if "x_wells" in layout and "y_wells" in layout:
            x_wells = int(layout["x_wells"])
            y_wells = int(layout["y_wells"])
        else:
            x_wells = int(round((cols - wc * dx) / wc))
            y_wells = int(round((rows - wr * dy) / wr))

        n_wells = int(layout.get("n_wells", wr * wc))
        if n_wells != wr * wc:
            raise ValueError(f"n_wells={n_wells} but Wr*Wc={wr * wc}")
        use_fw = _uses_firmware_assignment(layout)
        return cls(
            n_wells=n_wells,
            Wr=wr,
            Wc=wc,
            dy_wells=dy,
            dx_wells=dx,
            temp_ref=int(layout.get("temp_ref", 410)),
            ROWS=rows,
            COLS=cols,
            x_wells=x_wells,
            y_wells=y_wells,
            n_pixel_well=x_wells * y_wells,
            col_origin=col_origin,
            row_origin=row_origin,
            y_offset=y_offset,
            use_firmware_assignment=use_fw,
        )

    @classmethod
    def from_detected(cls, values_2d: np.ndarray, rows: int = NROWS, cols: int = NCOLS) -> "CWellGeometry":
        layout = detect_manifold_grid(values_2d)
        layout.setdefault("assignment", "origin")
        return cls.from_layout(layout, rows=rows, cols=cols)

    def well_index_map(self) -> np.ndarray:
        if self.use_firmware_assignment:
            return firmware_well_index_map(self, self.ROWS, self.COLS, self.y_offset)
        return self._origin_well_index_map()

    def _origin_well_index_map(self) -> np.ndarray:
        """Simple grid from col_origin/row_origin (manifold / auto-detect)."""
        out = np.zeros((self.ROWS, self.COLS), dtype=np.int32)
        for well_idx, (rs, re, cs, ce) in enumerate(self.well_spans_origin(), start=1):
            out[rs:re, cs:ce] = well_idx
        return out

    def well_spans_origin(self) -> List[Tuple[int, int, int, int]]:
        """(row_start, row_end, col_start, col_end) per well, row-major."""
        spans: List[Tuple[int, int, int, int]] = []
        stride_y = self.y_wells + self.dy_wells
        stride_x = self.x_wells + self.dx_wells
        for wr in range(self.Wr):
            for wc in range(self.Wc):
                row_start = self.row_origin + wr * stride_y
                row_end = row_start + self.y_wells
                col_start = self.col_origin + wc * stride_x
                col_end = col_start + self.x_wells
                spans.append((row_start, row_end, col_start, col_end))
        return spans

    def well_spans(self) -> List[Tuple[int, int, int, int]]:
        if self.use_firmware_assignment:
            return spans_from_well_index_map(self.well_index_map())
        return self.well_spans_origin()

    def gutter_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        if self.use_firmware_assignment:
            return self.well_index_map() == 0
        mask = np.ones(shape, dtype=bool)
        for rs, re, cs, ce in self.well_spans_origin():
            mask[rs:re, cs:ce] = False
        return mask

    def format_c_block(self, branch_name: str = "manifold") -> str:
        """Printable ``else if (n_wells == …)`` variable block for firmware."""
        lines = [
            f"    else if (n_wells == {self.n_wells})  // {branch_name}: {self.Wr} rows x {self.Wc} cols",
            "    {",
            "        // Variables",
            f"        Wr = {self.Wr};",
            f"        Wc = {self.Wc};",
            f"        dy_wells = {self.dy_wells};",
            f"        dx_wells = {self.dx_wells};",
            f"        x_wells = round((COLS-Wc*dx_wells)/Wc);   // -> {self.x_wells}",
            f"        y_wells = round((ROWS-Wr*dy_wells)/Wr);   // -> {self.y_wells}",
            f"        n_pixel_well = x_wells*y_wells;   // -> {self.n_pixel_well}",
            f"        temp_ref = {self.temp_ref};",
        ]
        if self.use_firmware_assignment:
            lines.append(f"        y_offset = {self.y_offset};")
        else:
            lines.append(f"        // col_origin = {self.col_origin}; row_origin = {self.row_origin};")
        lines.append("    }")
        return "\n".join(lines)

    def format_firmware_summary(self) -> str:
        """Human-readable summary of gutter bands used by firmware assignment."""
        dy_h = self.dy_wells // 2
        dx_h = self.dx_wells // 2
        sy = self.y_wells + self.dy_wells
        sx = self.x_wells + self.dx_wells
        return (
            f"Firmware assignment (n_wells==11||6): y_offset={self.y_offset}, "
            f"valid row band within each stride: [{dy_h}, {dy_h + self.y_wells}], "
            f"valid col band: [{dx_h}, {dx_h + self.x_wells}], "
            f"stride_y={sy}, stride_x={sx}"
        )


def print_c_well_geometry(geom: CWellGeometry, label: str = "") -> None:
    """Log derived sizes and copy-paste C blocks."""
    tag = f" ({label})" if label else ""
    print(f"\n{'=' * 60}")
    print(f"C well geometry{tag}: n_wells={geom.n_wells}, Wr={geom.Wr}, Wc={geom.Wc}")
    print(f"  ROWS={geom.ROWS}, COLS={geom.COLS}")
    if geom.use_firmware_assignment:
        print(f"  assignment: firmware (n_wells==11 || n_wells==6)")
        print(f"  {geom.format_firmware_summary()}")
    else:
        print(f"  assignment: origin grid")
        print(f"  origin: row={geom.row_origin}, col={geom.col_origin}")
    print(f"  dy_wells={geom.dy_wells}, dx_wells={geom.dx_wells}")
    print(f"  x_wells={geom.x_wells}, y_wells={geom.y_wells}, n_pixel_well={geom.n_pixel_well}")
    print(f"  temp_ref={geom.temp_ref}")
    print(geom.format_c_block(label or "layout"))
    if geom.use_firmware_assignment:
        print("\n  Well-index assignment (same for n_wells==6 and n_wells==11):")
        print(format_firmware_assignment_c())
    print(f"{'=' * 60}\n")


def per_well_active_stats(active_mask_2d: np.ndarray, geom: CWellGeometry) -> List[dict]:
    """Active pixel count per well using firmware or origin grid."""
    well_map = geom.well_index_map()
    stats = []
    for well_idx in range(1, geom.n_wells + 1):
        patch_mask = well_map == well_idx
        n_pix = int(patch_mask.sum())
        n_active = int(active_mask_2d[patch_mask].sum()) if n_pix else 0
        wr = (well_idx - 1) // geom.Wc
        wc = (well_idx - 1) % geom.Wc
        if n_pix:
            rs, cs = np.where(patch_mask)
            row_span = (int(rs.min()), int(rs.max()) + 1)
            col_span = (int(cs.min()), int(cs.max()) + 1)
        else:
            row_span = (0, 0)
            col_span = (0, 0)
        stats.append(
            {
                "well_idx": well_idx - 1,
                "channel": wr + 1,
                "column": wc + 1,
                "row_span": row_span,
                "col_span": col_span,
                "n_active": n_active,
                "n_pixels": n_pix,
                "pct": 100.0 * n_active / n_pix if n_pix else 0.0,
            }
        )
    return stats


def alignment_diagnostics(values_2d: np.ndarray, geom: CWellGeometry) -> dict:
    """Summarise how 400 / 1023 are split between wells and gutters."""
    values_2d = np.asarray(values_2d)
    gutter = geom.gutter_mask(values_2d.shape)
    wells = ~gutter
    active = values_2d == LACEWING_ACTIVE_VALUE
    inactive = values_2d == INACTIVE_VALUE
    return {
        "active_in_wells": int(active[wells].sum()),
        "active_in_gutters": int(active[gutter].sum()),
        "inactive_in_wells": int(inactive[wells].sum()),
        "inactive_in_gutters": int(inactive[gutter].sum()),
        "gutter_fraction_active": float(active[gutter].sum() / max(gutter.sum(), 1)),
    }


def load_find_active_mask(
    exp_path,
    nrows: int = NROWS,
    ncols: int = NCOLS,
    active_value: int = LACEWING_ACTIVE_VALUE,
    find_active_path=None,
):
    """
    Load the find-active pixel map from disk.

    Returns
    -------
    tuple
        (values_2d, active_mask_2d, path)
    """
    exp_path = Path(exp_path)
    path = Path(find_active_path) if find_active_path is not None else load.find_most_recent_find_active(exp_path)
    data_list = load.binary_file_read(path)
    n_pix = nrows * ncols
    if len(data_list) < n_pix:
        raise ValueError(
            f"{path.name} has {len(data_list)} words; expected at least {n_pix} for a {nrows}x{ncols} chip"
        )
    values_1d = np.asarray(data_list[:n_pix], dtype=np.uint16)
    values_2d = values_1d.reshape(nrows, ncols, order="F").reshape(nrows, ncols, order="C")
    active_mask_2d = values_2d == active_value
    return values_2d, active_mask_2d, path


def plot_find_active_pixels_plotly(
    values_2d,
    save_path=None,
    experiment_name="Experiment",
    source_file=None,
    figsize=(1600, 900),
    show=True,
    well_geometry: Optional[CWellGeometry] = None,
    plot_raw_values: bool = True,
):
    """
    Plot find_active map.

    Uses the same value scale as TITAN IMAGING when ``plot_raw_values=True``:
    1023 ≈ high (magenta), 400 ≈ mid (green), 511 ≈ temp grid (cyan).
    """
    values_2d = np.asarray(values_2d)
    active_mask_2d = values_2d == LACEWING_ACTIVE_VALUE
    n_active = int(active_mask_2d.sum())
    n_total = values_2d.size

    title = f"{experiment_name} - find_active.bin"
    if plot_raw_values:
        title = f"TITAN IMAGING — {experiment_name}"
    title += f"<br>{n_active:,} pixels with value {LACEWING_ACTIVE_VALUE} ({100.0 * n_active / n_total:.1f}%)"
    if well_geometry is not None:
        title += (
            f"<br>Grid: {well_geometry.Wr}×{well_geometry.Wc} wells "
            f"({well_geometry.y_wells}×{well_geometry.x_wells} px"
        )
        if well_geometry.use_firmware_assignment:
            title += f", firmware assign, y_offset={well_geometry.y_offset}"
        else:
            title += (
                f", origin row={well_geometry.row_origin} col={well_geometry.col_origin}"
            )
        title += f", gutter dy={well_geometry.dy_wells} dx={well_geometry.dx_wells})"
    if source_file is not None:
        title += f"<br>Source: {Path(source_file).name}"

    fig = go.Figure()
    if plot_raw_values:
        fig.add_trace(
            go.Heatmap(
                z=values_2d,
                zmin=0,
                zmax=2000,
                colorscale="Turbo",
                showscale=True,
                colorbar=dict(title="Value"),
            )
        )
    else:
        fig.add_trace(
            go.Heatmap(
                z=active_mask_2d.astype(int),
                colorscale="cividis",
                showscale=True,
                colorbar=dict(
                    title="Active",
                    tickmode="array",
                    tickvals=[0, 1],
                    ticktext=["Inactive", "Active"],
                ),
            )
        )

    if well_geometry is not None:
        shapes = []
        for rs, re, cs, ce in well_geometry.well_spans():
            shapes.append(
                dict(
                    type="rect",
                    x0=cs - 0.5,
                    x1=ce - 0.5,
                    y0=rs - 0.5,
                    y1=re - 0.5,
                    line=dict(color="white", width=1.5),
                    fillcolor="rgba(0,0,0,0)",
                )
            )
        fig.update_layout(shapes=shapes)

    fig.update_layout(
        title=title,
        xaxis_title="COLUMN",
        yaxis_title="ROW",
        height=figsize[1],
        width=figsize[0],
        font=dict(size=12),
        showlegend=False,
        margin=dict(l=60, r=60, t=120, b=60),
        autosize=False,
    )
    fig.update_xaxes(
        showgrid=False,
        scaleanchor="y",
        scaleratio=1.0,
        constrain="domain",
    )
    fig.update_yaxes(showgrid=False, constrain="domain", autorange="reversed")

    if show:
        fig.show()

    if save_path:
        try:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            safe_name = sanitize_filename(experiment_name)
            html_path = save_path / f"{safe_name}_find_active_pixels.html"
            print(f"Saving find_active pixels HTML to: {html_path}")
            fig.write_html(str(html_path))
            print(f"Saved: {html_path}")
        except Exception as exc:
            print(f"Error saving find_active pixels HTML: {exc}")

    return fig


def show_find_active_for_experiment(
    exp_path,
    save_path=None,
    experiment_name="Experiment",
    show=True,
    nrows: int = NROWS,
    ncols: int = NCOLS,
    active_value: int = LACEWING_ACTIVE_VALUE,
    well_layout: Optional[dict] = None,
    auto_detect_grid: bool = True,
    plot_raw_values: bool = True,
    n_wells: Optional[int] = None,
):
    """
    Load the newest find_active.bin in ``exp_path`` and display/save the map.

    Parameters
    ----------
    well_layout : dict, optional
        Fixed grid (``MANIFOLD_WELL_LAYOUT``, ``SIX_WELL_LAYOUT``, ``ELEVEN_WELL_LAYOUT``).
    auto_detect_grid : bool
        When True and ``n_wells`` is not 6/11, infer manifold grid from file.
    n_wells : int, optional
        If 6 or 11, use firmware assignment branch with ``SIX_WELL_LAYOUT`` /
        ``ELEVEN_WELL_LAYOUT`` (overrides auto-detect).
    """
    exp_path = Path(exp_path)
    values_2d, active_mask_2d, path = load_find_active_mask(
        exp_path, nrows=nrows, ncols=ncols, active_value=active_value
    )
    n_active = int(active_mask_2d.sum())
    unique_vals = np.unique(values_2d)
    print(f"Loaded find_active: {path.name}")
    print(f"  Pixels with value {active_value}: {n_active} / {active_mask_2d.size}")
    print(f"  Unique values: {unique_vals.tolist()}")

    geom = None
    if n_wells == 6:
        geom = CWellGeometry.from_layout(SIX_WELL_LAYOUT, rows=nrows, cols=ncols)
        print("Using n_wells=6 firmware layout (SIX_WELL_LAYOUT).")
    elif n_wells == 11:
        geom = CWellGeometry.from_layout(ELEVEN_WELL_LAYOUT, rows=nrows, cols=ncols)
        print("Using n_wells=11 firmware layout (ELEVEN_WELL_LAYOUT).")
    elif auto_detect_grid:
        try:
            geom = CWellGeometry.from_detected(values_2d, rows=nrows, cols=ncols)
            print("Auto-detected manifold grid from find_active file.")
        except Exception as exc:
            print(f"Warning: grid auto-detect failed ({exc}); falling back to well_layout.")
    if geom is None and well_layout is not None:
        geom = CWellGeometry.from_layout(well_layout, rows=nrows, cols=ncols)

    if geom is not None:
        print_c_well_geometry(geom, label=experiment_name)
        diag = alignment_diagnostics(values_2d, geom)
        print(
            "Alignment check (blank chip expects ~0 active pixels; value 400 in gutters = grid offset in firmware):"
        )
        print(
            f"  value {active_value} inside wells: {diag['active_in_wells']:,}  "
            f"in gutters: {diag['active_in_gutters']:,}  "
            f"({100.0 * diag['gutter_fraction_active']:.1f}% of gutter pixels are {active_value})"
        )
        print(
            f"  value {INACTIVE_VALUE} inside wells: {diag['inactive_in_wells']:,}  "
            f"in gutters: {diag['inactive_in_gutters']:,}"
        )
        if diag["active_in_gutters"] > diag["active_in_wells"]:
            print(
                "  >> Most value-400 pixels sit in gutters (vertical lanes cols ~52-80 and ~122-150)."
            )
            print(
                "     On a blank chip expect all 1023; green in TITAN IMAGING = 400. Update C dy/dx/origin to match auto-detected values."
            )
        stats = per_well_active_stats(active_mask_2d, geom)
        print("Per-well active pixels:")
        for s in stats:
            print(
                f"  Well {s['well_idx'] + 1:2d}  Ch{s['channel']:2d} Col{s['column']}  "
                f"rows[{s['row_span'][0]}:{s['row_span'][1]}] cols[{s['col_span'][0]}:{s['col_span'][1]}]  "
                f"active={s['n_active']:5d} / {s['n_pixels']:5d} ({s['pct']:.1f}%)"
            )

    fig = plot_find_active_pixels_plotly(
        values_2d,
        save_path=save_path,
        experiment_name=experiment_name,
        source_file=path,
        show=show,
        well_geometry=geom,
        plot_raw_values=plot_raw_values,
    )
    return fig, active_mask_2d, path


if __name__ == "__main__":
    for layout, label in (
        (SIX_WELL_LAYOUT, "6-well (firmware branch)"),
        (ELEVEN_WELL_LAYOUT, "11-well (firmware branch)"),
        (MANIFOLD_WELL_LAYOUT, "30-well manifold (origin grid)"),
    ):
        geom = CWellGeometry.from_layout(layout)
        print_c_well_geometry(geom, label=label)
        # Example pixel checks for firmware layouts
        if geom.use_firmware_assignment:
            samples = [(0, 0), (geom.dy_wells // 2, geom.dx_wells // 2), (50, 50), (100, 100)]
            print(f"  Sample firmware well indices for {label}:")
            for r, c in samples:
                w = firmware_assign_well(r, c, geom)
                print(f"    (row={r}, col={c}) -> well {w}")
