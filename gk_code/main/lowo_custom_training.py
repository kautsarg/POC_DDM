import argparse
import importlib
import os
import types
from pathlib import Path

import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import config
cdt = importlib.import_module("04_cross_dataset_training")

GROUP_NAME = "final_6_new"
CURVE_TYPE = "ori_curve_sg_p4_norm"
PC_TTP_ANCHOR = "min"
TRAIN_CENTER_FRAC = 0.5
K_NEIGHBORS = 24

HELD_OUT_WELLS = {
    "D20260827_E00_C00_F4500KHz_U_DDM_01_final_final::1": "IAV",
    "D20260827_E00_C00_F4500KHz_U_DDM_02_final_final::1": "IBV",
    "D20260827_E00_C00_F4500KHz_U_DDM_03_final_final::1": "Kp",
    "D20260827_E00_C00_F4500KHz_U_DDM_04_final_final::1": "Cov",
    "D20260825_E00_C00_F4500KHz_U_DDM_05_01::1": "Hadv",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch_size", type=int, required=True,
                    help="GPU-dependent, matches f_lf_g13.sh's own auto-detection.")
    p.add_argument("--combo_tag", type=str, default="v1",
                    help="Label for this well combination; keeps output files "
                         "from different combinations from colliding.")
    p.add_argument("--model", type=str, required=True,
                    help="Model key, e.g. cnn_gru_dual_attn_recon -- same keys "
                         "f_lf_g13.sh's --models flag accepts.")
    p.add_argument("--mtl", action="store_true",
                    help="Only needed for the _mtl model variant -- mirrors "
                         "f_lf_g13.sh's --mtl flag (04's own args.mtl is read "
                         "directly by _derive_pool_labels/evaluate_outlier_filters).")
    p.add_argument("--dry_run", action="store_true",
                    help="Pool the data and build the split, print/verify it, "
                         "then exit before training -- no GPU time spent.")
    return p.parse_args()


def main():
    args_cli = parse_args()
    mode_str = f"lowo_{args_cli.combo_tag}"

    cdt.set_global_determinism(0, strict=True)

    folder_names = config.CROSS_DATASET_GROUPS[GROUP_NAME]
    exp_paths = [Path(config.FINAL_EXP_FOLDER, name) for name in folder_names]

    pc_ttp_cache_dir = Path(config.FINAL_EXP_FOLDER) / "cross_dataset_cv" / GROUP_NAME / "_cache_pc_ttp"
    pc_ttp_cache_dir.mkdir(parents=True, exist_ok=True)

    out_dir = (Path(config.FINAL_EXP_FOLDER) / "cross_dataset_cv" / GROUP_NAME
               / "curve_alignment_pc_ttp" / f"anchor_{PC_TTP_ANCHOR}")
    plot_dir = out_dir / f"model_performance_{CURVE_TYPE}"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"[LOWO] Pooling '{GROUP_NAME}' (held_out_chip=None -- no whole chip held "
          f"out, alignment stats computed from all {len(folder_names)} chips)...")
    combined = cdt.combine_group_pc_aligned(
        exp_paths, GROUP_NAME, curve_type=CURVE_TYPE, held_out_chip=None,
        anchor_method=PC_TTP_ANCHOR, anchor_pct=cdt.PC_TTP_ANCHOR_PCT_DEFAULT,
        pc_ttp_cache_dir=pc_ttp_cache_dir,
    )
    assert combined is not None, "combine_group_pc_aligned returned None -- check exp_folder/group/curve_type"

    keep_mask = cdt.is_amplifying_mask(combined["curves"])
    combined["features_df"][cdt.NOAMP_FILTER_NAME] = keep_mask.astype(int)
    print(f"  [*] {cdt.NOAMP_FILTER_NAME}: {int((~keep_mask).sum())}/{len(keep_mask)} "
          f"non-amplifying curves flagged for removal")

    args = types.SimpleNamespace(
        mtl=args_cli.mtl, train_center_frac=TRAIN_CENTER_FRAC, k_neighbors=K_NEIGHBORS,
        batch_size=args_cli.batch_size, curve_alignment="pc_ttp", pc_ttp_anchor=PC_TTP_ANCHOR,
    )
    cdt._save_alignment_artifacts(combined, out_dir, CURVE_TYPE, args, held_out_chip=None)

    encoder, y_full, X_candidates_clean, y_concentration, chip_id_encoded, conc_id_encoded = \
        cdt._derive_pool_labels(combined, args)
    print(f"  [*] Pool: {len(y_full)} samples, {len(encoder.classes_)} classes: {list(encoder.classes_)}")

    held_out_well_ids = list(HELD_OUT_WELLS.keys())
    test_idx = np.where(np.isin(combined["well_ids"], held_out_well_ids))[0]
    train_idx = np.where(~np.isin(combined["well_ids"], held_out_well_ids))[0]

    print(f"\n[LOWO] Held-out wells ({len(held_out_well_ids)}):")
    found_well_ids = set(combined["well_ids"][test_idx])
    all_ok = True
    for well_id, expected_target in HELD_OUT_WELLS.items():
        row_mask = combined["well_ids"] == well_id
        n_rows = int(row_mask.sum())
        actual_targets = sorted(set(combined["Y_mapped"][row_mask]))
        ok = well_id in found_well_ids and actual_targets == [expected_target]
        all_ok = all_ok and ok
        print(f"  [{'OK' if ok else 'MISMATCH'}] {well_id}  "
              f"expected={expected_target}  actual={actual_targets}  n_rows={n_rows}")
    assert len(test_idx) > 0, "No rows matched HELD_OUT_WELLS -- check well_ids format/spelling."
    assert all_ok, "One or more held-out wells didn't match their expected target -- check HELD_OUT_WELLS."
    print(f"\n  train={len(train_idx)}  test={len(test_idx)}  "
          f"({len(test_idx) / (len(train_idx) + len(test_idx)) * 100:.2f}% of pool held out)")

    if args_cli.dry_run:
        print("\n[DRY RUN] Split verified -- exiting before training (no GPU time spent).")
        return

    cdt._process_fold(
        fold_idx=0, total_folds=1, fold_label=mode_str,
        train_idx=train_idx, test_idx=test_idx,
        combined=combined, y_full=y_full, encoder=encoder, X_candidates_clean=X_candidates_clean,
        curve_type=CURVE_TYPE, models=[args_cli.model], outlier_filters=[cdt.NOAMP_FILTER_NAME],
        out_dir=out_dir, plot_dir=plot_dir, group_name=GROUP_NAME, total_count=len(y_full),
        lofo_results={}, mode_str=mode_str, args=args,
        y_concentration=y_concentration, chip_id_encoded=chip_id_encoded, conc_id_encoded=conc_id_encoded,
    )
    print(f"\n[DONE] LOWO run '{mode_str}' complete. Results under {out_dir}")


if __name__ == "__main__":
    main()
