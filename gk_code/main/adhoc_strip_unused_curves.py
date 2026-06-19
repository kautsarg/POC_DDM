"""
adhoc_strip_unused_curves.py
=============================
One-off migration utility — NOT part of the numbered 01-08 pipeline.

Strips data from the shared curve_for_training_nonorm.joblib that's dead weight
under the default --curve_type ['ori_curve', 'ori_curve_avg'] (see joblib_redundancy.md
Change 1/Change 2):
  - 01's sigmoid-fit chain: sigmoid_curves, curves.{ori_curve_dydx,ori_dydx_avg,
    cleaned_std,cleaned_lowest}, idxs.{cleaned_idx,cleaned_lowest_idx}, window_size_1stder
  - 01's dead well_* fields: curves.{well_2d_bs_active,well_2d_nl_bs_active,
    well_temp_lin2d,well_2d_temp_npr,well_temp_mean_then_lin}
  - 02's unused curve variants: dataset/dataset_name/kinetic_features (and the matching
    feature-combination lists) trimmed down to just ori_curves/ori_curves_avg

Pure key deletion/filtering on the existing cache — no recomputation. Also folds in
a legacy separate preprocessed_curves_nonorm.joblib if 01/02 haven't been re-run yet
since the shared-file change. Idempotent — safe to re-run; already-stripped or
not-yet-built experiments are skipped.

Run once per dataset, after upgrading to the --compute_sigmoid_fits-gated 01/02, to
reclaim space on already-processed experiments without re-running them:

    python adhoc_strip_unused_curves.py --exp_folder /path/to/POC_DDM_dataset_folder
"""
import os
import argparse
from pathlib import Path
import joblib

import config


def strip_experiment(exp_path):
    """Strip the unused sigmoid-fit chain + dead well_* fields (01) and unused curve
    variants (02) from one experiment's shared joblib cache, in place."""
    save_path = exp_path / config.TRAINING_DATA_PATH
    legacy_save_path = exp_path / config.PREPROCESSED_CURVES_PATH

    state = {}
    if save_path.exists():
        state = joblib.load(save_path)
    if "curves" not in state and legacy_save_path.exists():
        # Pre-merge layout: 01 and 02 still wrote separate files — fold the legacy
        # file's keys in now so the result lands in the one shared file going forward.
        state.update(joblib.load(legacy_save_path))

    if "curves" not in state:
        print(f"  -> [SKIP] {exp_path.name}: no cached curve data found (run 01 first).")
        return

    changed = False

    # --- 01's part: sigmoid-fit chain ---
    if state.get("sigmoid_curves") or any(
            state["curves"].get(k) is not None
            for k in ("ori_curve_dydx", "ori_dydx_avg", "cleaned_std", "cleaned_lowest")):
        state["sigmoid_curves"] = {}
        for k in ["ori_curve_dydx", "ori_dydx_avg", "cleaned_std", "cleaned_lowest"]:
            state["curves"][k] = None
        if "idxs" in state:
            state["idxs"]["cleaned_idx"] = None
            state["idxs"]["cleaned_lowest_idx"] = None
        state["window_size_1stder"] = None
        changed = True

    # --- 01's part: dead well_* fields ---
    for k in ["well_2d_bs_active", "well_2d_nl_bs_active", "well_temp_lin2d",
              "well_2d_temp_npr", "well_temp_mean_then_lin"]:
        if state["curves"].pop(k, None) is not None:
            changed = True

    # --- 02's part: unused curve variants ---
    if "dataset_name" in state:
        keep_names = {"ori_curves", "ori_curves_avg"}
        dataset_name = list(state["dataset_name"])
        keep_idx = [i for i, n in enumerate(dataset_name) if n in keep_names]
        if len(keep_idx) < len(dataset_name):
            state["dataset_name"] = [dataset_name[i] for i in keep_idx]
            state["dataset"] = [state["dataset"][i] for i in keep_idx]
            if "kinetic_features" in state:
                state["kinetic_features"] = [state["kinetic_features"][i] for i in keep_idx]
            for key in ["linear_feature_combinations", "important_feature_combinations", "importance_dfs"]:
                if key in state:
                    state[key] = [state[key][i] for i in keep_idx]
            changed = True

    if not changed:
        print(f"  -> [SKIP] {exp_path.name}: already stripped, nothing to do.")
        return

    joblib.dump(state, save_path, compress=3)
    if legacy_save_path.exists() and legacy_save_path.resolve() != save_path.resolve():
        os.remove(legacy_save_path)
        print(f"  -> [STRIPPED+MERGED] {exp_path.name}: consolidated into {save_path}, removed {legacy_save_path}")
    else:
        print(f"  -> [STRIPPED] {exp_path.name}")


if __name__ == "__main__":
    print(f"\n{'='*70}\n[RUNNING] {os.path.basename(__file__)}\n{'='*70}\n")
    parser = argparse.ArgumentParser(
        description="One-off migration: strip the unused sigmoid-fit chain (01) and unused "
                     "curve variants (02) from already-cached experiments. Run once per dataset.")
    parser.add_argument("--exp_folder", type=str, default=config.DEFAULT_EXP_FOLDER)
    args = parser.parse_args()

    exp_paths = sorted([
        Path(args.exp_folder, name)
        for name in os.listdir(args.exp_folder)
        if os.path.isdir(os.path.join(args.exp_folder, name))
        and name not in config.EXCLUDED_FOLDERS
    ])

    for exp_path in exp_paths:
        strip_experiment(exp_path)
