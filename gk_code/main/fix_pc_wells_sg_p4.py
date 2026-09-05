import os
import sys
import argparse
import importlib.util
import joblib
from pathlib import Path

sys.path.insert(0, os.getcwd())
sys.path.insert(0, 'utils')
from safe_io import safe_joblib_dump
import config

_spec = importlib.util.spec_from_file_location(
    "curve_preprocessing_v6", str(Path.cwd() / "01_curve_preprocessing_v6.py"))
cp01 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cp01)

EXP_FOLDERS = [config.FINAL_EXP_FOLDER, config.FINAL_EXP_FOLDER + "_nc_subtract"]


def patch_one(exp_path, dry_run):
    save_path = exp_path / config.TRAINING_DATA_PATH
    if not save_path.exists():
        print(f"  [SKIP] {exp_path}: no {config.TRAINING_DATA_PATH}")
        return

    data = joblib.load(save_path)
    pc_wells = data.get("pc_wells")
    if not pc_wells or "curves" not in pc_wells:
        print(f"  [SKIP] {exp_path.name}: no 'pc_wells' snapshot (--drop_pc not used here).")
        return

    before = sorted(pc_wells["curves"].keys())
    if "ori_curves_sg_p4" in pc_wells["curves"] and "ori_curves_sg_p4_norm" in pc_wells["curves"]:
        print(f"  [OK]   {exp_path.name}: already patched -- {before}")
        return

    print(f"  [NEEDS PATCH] {exp_path.name}: pc_wells has {before}")
    if dry_run:
        return

    base = pc_wells["curves"]["ori_curves"]
    sg_p4, _ = cp01.sg_p4_denoise_curves(base)
    pc_wells["curves"]["ori_curves_sg_p4"] = sg_p4
    pc_wells["curves"]["ori_curves_sg_p4_norm"] = cp01.normalize_curves_minmax(sg_p4)
    data["pc_wells"] = pc_wells

    safe_joblib_dump(data, save_path, compress=3)
    after = sorted(pc_wells["curves"].keys())
    print(f"  [PATCHED] {exp_path.name}: {before} -> {after}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch missing sg_p4 variants into pc_wells snapshots.")
    parser.add_argument("--dry_run", action="store_true", help="Report only, write nothing.")
    args = parser.parse_args()

    for exp_folder in EXP_FOLDERS:
        print(f"\n=== {exp_folder} ===")
        exp_paths = sorted(p for p in Path(exp_folder).iterdir()
                           if p.is_dir() and p.name.startswith("D2026"))
        for exp_path in exp_paths:
            patch_one(exp_path, args.dry_run)
