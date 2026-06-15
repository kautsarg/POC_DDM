from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

@dataclass
class WellOutline:
    """Holds the outline coordinates for a single well in chip coordinates."""
    well_index: int  # 0-based index
    polygon: List[Tuple[int, int]]  # (col, row) points in order, closed (first==last)
    row_span: Tuple[int, int]  # (row_start, row_end) inclusive/exclusive
    col_span: Tuple[int, int]  # (col_start, col_end) inclusive/exclusive


def _placement_spans(n_wells: int, chip_rows: int, chip_cols: int) -> List[Tuple[int, int, int, int]]:
    """
    Compute the placement spans (row_start, row_end, col_start, col_end) for each well.
    This mirrors the layout logic used in plot_wells_active_pixels_plotly.
    """
    spans: List[Tuple[int, int, int, int]] = []

    if n_wells == 1:
        spans.append((0, chip_rows, 0, chip_cols))
        return spans

    if n_wells == 2:
        # Two wells stacked vertically
        half_rows = chip_rows // 2
        spans.append((0, half_rows, 0, chip_cols))
        spans.append((half_rows, chip_rows, 0, chip_cols))
        return spans

    if n_wells in {4, 6, 10}:
        wells_per_row = 2
        rows_per_well = chip_rows // (n_wells // wells_per_row)
        cols_per_well = chip_cols // wells_per_row

        for well_idx in range(n_wells):
            row_start = (well_idx // wells_per_row) * rows_per_well
            col_start = (well_idx % wells_per_row) * cols_per_well
            row_end = row_start + rows_per_well
            col_end = col_start + cols_per_well
            spans.append((row_start, row_end, col_start, col_end))
        return spans

    raise ValueError(f"Unsupported number of wells: {n_wells}")


def get_well_outlines(exp, chip_rows: int = 290, chip_cols: int = 204, n_wells: int | None = None) -> List[WellOutline]:
    """
    Derive outline polygons for each well in chip coordinates.

    Parameters
    ----------
    exp : Experiment
        Experiment object with wells_list populated.
    chip_rows : int
        Total chip row count (default 290 for Titan).
    chip_cols : int
        Total chip column count (default 204 for Titan).
    n_wells : int | None
        If provided, force this well count. If None (default), use len(exp.wells_list).

    Returns
    -------
    List[WellOutline]
        One outline per well, with polygon coordinates in (col, row) order.
    """
    n_wells_effective = n_wells if n_wells is not None else len(exp.wells_list)
    spans = _placement_spans(n_wells_effective, chip_rows, chip_cols)

    outlines: List[WellOutline] = []
    for well_idx, (row_start, row_end, col_start, col_end) in enumerate(spans):
        # Define polygon in (col, row) with origin at top-left; close the polygon
        poly = [
            (col_start, row_start),
            (col_end, row_start),
            (col_end, row_end),
            (col_start, row_end),
            (col_start, row_start),
        ]
        outlines.append(
            WellOutline(
                well_index=well_idx,
                polygon=poly,
                row_span=(row_start, row_end),
                col_span=(col_start, col_end),
            )
        )

    return outlines


def get_inactive_well_outlines(exp, chip_rows: int = 290, chip_cols: int = 204, n_wells: int | None = None) -> List[WellOutline]:
    """
    Derive outline polygons for the inactive pixels of each well (bounding rectangle)
    in chip coordinates.

    Parameters
    ----------
    exp : Experiment
        Experiment object with wells_list populated.
    chip_rows : int
        Total chip row count (default 290 for Titan).
    chip_cols : int
        Total chip column count (default 204 for Titan).
    n_wells : int | None
        If provided, force this well count. If None (default), use len(exp.wells_list).

    Returns
    -------
    List[WellOutline]
        One outline per well, representing the bounding box of inactive pixels
        (where idx_active is False) in chip coordinates.
    """
    n_wells_effective = n_wells if n_wells is not None else len(exp.wells_list)
    spans = _placement_spans(n_wells_effective, chip_rows, chip_cols)

    outlines: List[WellOutline] = []
    for well_idx, (row_start, row_end, col_start, col_end) in enumerate(spans):
        if well_idx >= len(exp.wells_list):
            break
        well = exp.wells_list[well_idx]

        if not hasattr(well, "idx_active") or well.idx_active is None:
            continue

        try:
            active_mask_2d = well.idx_active.reshape(well.well_nrows, well.well_ncols, order="C")
        except Exception:
            # Fallback: try chip dimensions
            try:
                active_mask_2d = well.idx_active.reshape(row_end - row_start, col_end - col_start, order="C")
            except Exception:
                continue

        inactive_mask = ~active_mask_2d
        coords = np.argwhere(inactive_mask)
        if coords.size == 0:
            continue

        r_min, c_min = coords.min(axis=0)
        r_max, c_max = coords.max(axis=0) + 1  # exclusive

        # Translate to chip coordinates
        r0 = row_start + r_min
        r1 = row_start + r_max
        c0 = col_start + c_min
        c1 = col_start + c_max

        poly = [
            (int(c0), int(r0)),
            (int(c1), int(r0)),
            (int(c1), int(r1)),
            (int(c0), int(r1)),
            (int(c0), int(r0)),
        ]

        outlines.append(
            WellOutline(
                well_index=well_idx,
                polygon=poly,
                row_span=(r0, r1),
                col_span=(c0, c1),
            )
        )

    return outlines

