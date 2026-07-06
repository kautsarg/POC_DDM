"""
Patch existing curve_for_training.joblib files to add a 'concentration' key.

Covers:
  - LAB_DDM_paper experiment folders  (reads FILE_CONC column from the source CSV)
  - LAB_DDM_paper *_aligned folders   (filters the parent concentration by the TTP keep mask,
                                        reconstructed by matching well_labels order)
  - LAB_OneToOne combo folders        (concentration is always None)
  - LAB_OneToOne *_aligned folders    (concentration is always None)

Any joblib that already has 'concentration' is skipped.
Run with: python adhoc_patch_concentration.py [--dry_run]
"""

import sys
import re
import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

import config
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("lab01b", Path(__file__).parent / "01b_lab_curve_preprocessing.py")
lab01b_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(lab01b_mod)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read_table(file_path):
    ext = Path(file_path).suffix.lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(file_path)
    return pd.read_csv(file_path)


def _patch(jl_path, concentration, dry_run):
    try:
        data = joblib.load(jl_path)
    except Exception as e:
        print(f"  [warn]  could not load {jl_path} ({e}) — skipping")
        return
    if "concentration" in data:
        print(f"  [skip]  already has 'concentration': {jl_path}")
        return
    if dry_run:
        val_repr = f"array(len={len(concentration)})" if concentration is not None else "None"
        print(f"  [dry]   would patch {jl_path}  →  concentration={val_repr}")
        return
    data["concentration"] = concentration
    joblib.dump(data, jl_path, compress=3)
    val_repr = f"array(len={len(concentration)})" if concentration is not None else "None"
    print(f"  [ok]    patched {jl_path}  →  concentration={val_repr}")


# ---------------------------------------------------------------------------
# LAB_DDM_paper (non-one-to-one)
# ---------------------------------------------------------------------------

def patch_lab_ddm_paper(dry_run):
    exp_folder = Path(config.LAB_EXP_FOLDER)
    for folder_name, csv_file in lab01b_mod.FILE_MAPPING.items():
        exp_path = exp_folder / folder_name
        jl_path  = exp_path / config.TRAINING_DATA_PATH
        if not jl_path.exists():
            continue

        conc_col = lab01b_mod.FILE_CONC.get(folder_name)
        if conc_col:
            df      = _read_table(exp_path / csv_file)
            concentration = df[conc_col].to_numpy()
        else:
            concentration = None

        print(f"LAB_DDM {folder_name}:")
        _patch(jl_path, concentration, dry_run)

        # aligned sibling
        aligned_path = exp_folder / (folder_name + "_aligned")
        aligned_jl   = aligned_path / config.TRAINING_DATA_PATH
        if not aligned_jl.exists():
            continue

        print(f"LAB_DDM {folder_name}_aligned:")
        if concentration is None:
            _patch(aligned_jl, None, dry_run)
        else:
            # Recover the TTP keep mask by matching the aligned well_labels against the
            # original order: aligned well_labels are a contiguous subsequence of the
            # original, so we find the matching row-indices in the original CSV.
            aligned_data = joblib.load(aligned_jl)
            aligned_labels = aligned_data["well_labels"]
            target_col = lab01b_mod.FILE_TARGET[folder_name]
            orig_labels = df[lab01b_mod.FILE_TARGET[folder_name]].to_numpy()

            # greedy left-to-right match (preserves insertion order, handles repeats)
            keep_idx = []
            orig_ptr = 0
            for al in aligned_labels:
                while orig_ptr < len(orig_labels) and orig_labels[orig_ptr] != al:
                    orig_ptr += 1
                if orig_ptr < len(orig_labels):
                    keep_idx.append(orig_ptr)
                    orig_ptr += 1
            aligned_concentration = concentration[keep_idx] if len(keep_idx) == len(aligned_labels) else None
            if aligned_concentration is None:
                print(f"  [warn]  could not reconstruct keep mask for {folder_name}_aligned — storing None")
            _patch(aligned_jl, aligned_concentration, dry_run)


# ---------------------------------------------------------------------------
# LAB_OneToOne combos (and their _aligned siblings)
# ---------------------------------------------------------------------------

def patch_one_to_one(dry_run):
    exp_folder = Path(config.LAB_1TO1_EXP_FOLDER)
    combos = lab01b_mod.discover_one_to_one_combos(str(exp_folder))
    for _, _, _, combo_name in combos:
        for suffix in ("", "_aligned"):
            jl_path = exp_folder / (combo_name + suffix) / config.TRAINING_DATA_PATH
            if not jl_path.exists():
                continue
            print(f"1to1 {combo_name}{suffix}:")
            _patch(jl_path, None, dry_run)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch concentration into existing curve_for_training.joblib files")
    parser.add_argument("--dry_run", action="store_true", help="Print what would be done without writing")
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN — no files will be modified]\n")

    print("=== LAB_DDM_paper ===")
    patch_lab_ddm_paper(args.dry_run)

    # print("\n=== LAB_OneToOne ===")
    # patch_one_to_one(args.dry_run)

    print("\nDone.")
