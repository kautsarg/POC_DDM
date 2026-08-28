import os
import sys
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

_MAIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_MAIN_DIR))
os.chdir(_MAIN_DIR)

import config

cdt = importlib.import_module("04_cross_dataset_training")

GROUP_NAME = "final_6_new"
EXP_FOLDER = "/vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_final"
CURVE_TYPE = "ori_curve_sg_p4_norm"
PC_TTP_ANCHOR = "min"
EXCLUDE_LABELS = ("PC", "NC-ALL")
THRESHOLD_FRAC = 0.2


def build_well_table(dataset_id, well_ids, Y_mapped):
    df = pd.DataFrame({"well_id": well_ids, "dataset_id": dataset_id, "label": Y_mapped})
    df["row_idx"] = np.arange(len(df))
    meta = df.groupby("well_id").agg(dataset_id=("dataset_id", "first"), label=("label", "first"))
    row_idx_map = df.groupby("well_id")["row_idx"].apply(np.array)
    return meta.join(row_idx_map.rename("row_idx")).reset_index()


def main():
    folder_names = config.CROSS_DATASET_GROUPS[GROUP_NAME]
    exp_paths = [Path(EXP_FOLDER, name) for name in folder_names]
    pc_ttp_cache_dir = (Path(EXP_FOLDER) / "cross_dataset_cv" / GROUP_NAME
                        / "curve_alignment_pc_ttp" / f"anchor_{PC_TTP_ANCHOR}" / "_cache_pc_ttp")

    print(f"Building PC-TTP-aligned pool for group={GROUP_NAME!r} curve_type={CURVE_TYPE!r} "
          f"(held_out_chip=None -- anchor/common_duration from all {len(folder_names)} chips)...")
    combined = cdt.combine_group_pc_aligned(
        exp_paths, GROUP_NAME, CURVE_TYPE, held_out_chip=None,
        anchor_method=PC_TTP_ANCHOR, anchor_pct=cdt.PC_TTP_ANCHOR_PCT_DEFAULT,
        pc_ttp_cache_dir=pc_ttp_cache_dir)
    assert combined is not None, "combine_group_pc_aligned returned None -- check GROUP_NAME/EXP_FOLDER/CURVE_TYPE."

    curves = combined["curves"]
    print(f"curves shape: {curves.shape}")

    thresh = THRESHOLD_FRAC * curves.max(axis=1, keepdims=True)
    crossing_frame = np.argmax(curves >= thresh, axis=1).astype(float)
    # argmax returns 0 for an all-False row (never crosses threshold) -- distinguish
    # that from a genuine frame-0 crossing so it doesn't silently pollute the stats.
    never_crosses = ~(curves >= thresh).any(axis=1)
    n_never = int(never_crosses.sum())
    if n_never:
        print(f"  [!] {n_never} / {len(curves)} pixels never cross {THRESHOLD_FRAC:.0%} of their own max -- excluded.")
    crossing_frame[never_crosses] = np.nan

    wells = build_well_table(combined["dataset_id"], combined["well_ids"], combined["Y_mapped"])
    wells = wells[~wells["label"].isin(EXCLUDE_LABELS)].reset_index(drop=True)
    wells["crossing_frame"] = [np.nanmean(crossing_frame[idx]) for idx in wells["row_idx"]]
    wells = wells.dropna(subset=["crossing_frame"])

    print(f"\n{'Target':<8} {'n_chips':>8} {'per-chip-mean std':>20} {'pooled std':>12} {'n_wells':>8}")
    per_target_chip_std = {}
    for label, grp in wells.groupby("label"):
        per_chip_mean = grp.groupby("dataset_id")["crossing_frame"].mean()
        chip_std = per_chip_mean.std(ddof=1) if len(per_chip_mean) > 1 else float("nan")
        pooled_std = grp["crossing_frame"].std(ddof=1) if len(grp) > 1 else float("nan")
        per_target_chip_std[label] = (chip_std, len(per_chip_mean))
        print(f"{label:<8} {len(per_chip_mean):>8} {chip_std:>20.2f} {pooled_std:>12.2f} {len(grp):>8}")

    valid_stds = [s for s, n in per_target_chip_std.values() if n >= 3 and not np.isnan(s)]
    if not valid_stds:
        print("\n[!] No target had >=3 chips with a usable std -- can't recommend S. Inspect the per-target table above.")
        return

    median_s = float(np.median(valid_stds))
    p75_s = float(np.percentile(valid_stds, 75))
    print(f"\nRecommended AUG_SHIFT_MAX_FRAMES:")
    print(f"  median across targets (>=3 chips each): {median_s:.1f} frames")
    print(f"  75th percentile across targets:          {p75_s:.1f} frames")
    print(f"  (prior expectation from PC-TTP shift overlays: ~100-200 frames)")


if __name__ == "__main__":
    main()
