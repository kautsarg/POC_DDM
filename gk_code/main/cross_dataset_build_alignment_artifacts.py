import os
import sys
import argparse
import importlib
from pathlib import Path

sys.path.insert(0, 'utils')
sys.path.insert(0, 'utils/model_training')
import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import-only, sibling script never modified.
cdt = importlib.import_module("04_cross_dataset_training")


class _Args:
    def __init__(self, curve_alignment):
        self.curve_alignment = curve_alignment


def build_and_save(exp_folder, group_name, curve_type, curve_alignment, pc_ttp_anchor):
    """Same fresh group-wide pool + _save_alignment_artifacts() call 04 makes inside
    `if args.train_full:` for pc_ttp alignment (or unconditionally for acquisition_start)
    -- just without the model-training tail that normally follows it."""
    folder_names = config.CROSS_DATASET_GROUPS[group_name]
    exp_paths = [Path(exp_folder, name) for name in folder_names]

    out_dir = Path(exp_folder) / "cross_dataset_cv" / group_name
    if curve_alignment == "pc_ttp":
        out_dir = out_dir / "curve_alignment_pc_ttp" / f"anchor_{pc_ttp_anchor}"

    if curve_alignment == "pc_ttp":
        pc_ttp_cache_dir = Path(exp_folder) / "cross_dataset_cv" / group_name / "_cache_pc_ttp"
        pc_ttp_cache_dir.mkdir(parents=True, exist_ok=True)
        combined = cdt.combine_group_pc_aligned(
            exp_paths, group_name, curve_type=curve_type, held_out_chip=None,
            anchor_method=pc_ttp_anchor, anchor_pct=cdt.PC_TTP_ANCHOR_PCT_DEFAULT,
            pc_ttp_cache_dir=pc_ttp_cache_dir)
    else:
        combined = cdt.combine_group(exp_paths, group_name, curve_type=curve_type)

    if combined is None:
        print(f"[!] {group_name}/{curve_type}: fewer than 2 usable datasets, nothing saved.")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    cdt._save_alignment_artifacts(combined, out_dir, curve_type, _Args(curve_alignment))
    return True


if __name__ == "__main__":
    print(f"\n{'='*70}\n[RUNNING] {os.path.basename(__file__)}\n{'='*70}\n")
    parser = argparse.ArgumentParser(
        description="Build and save just the resampler (+ pc_ttp recipe) for a group/curve_type "
                     "-- the artifact 08_cross_dataset_predict_new_chip.py needs to align a new "
                     "chip -- without training any models. No 04_cross_dataset_training.py changes.")
    parser.add_argument("--exp_folder", type=str, default=config.DEFAULT_EXP_FOLDER)
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--curve_type", type=str, nargs='+', required=True)
    parser.add_argument("--curve_alignment", type=str, choices=config.CURVE_ALIGNMENT_CHOICES, default="acquisition_start")
    parser.add_argument("--pc_ttp_anchor", type=str, choices=["min", "percentile"], default="min")
    args = parser.parse_args()

    group_names = list(config.CROSS_DATASET_GROUPS.keys())
    if args.task_id >= len(group_names):
        sys.exit(f"[!] task_id {args.task_id} out of bounds for {len(group_names)} groups.")
    group_name = group_names[args.task_id]
    print(f"Group: {group_name}")

    for curve_type in args.curve_type:
        print(f"\n=== {curve_type} ===")
        build_and_save(args.exp_folder, group_name, curve_type, args.curve_alignment, args.pc_ttp_anchor)
