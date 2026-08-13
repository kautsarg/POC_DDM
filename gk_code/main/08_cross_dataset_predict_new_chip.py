import os
import sys
import argparse
import importlib
from pathlib import Path

import numpy as np
import joblib

sys.path.insert(0, 'utils')
sys.path.insert(0, 'utils/model_training')
import config

# Import-only, sibling scripts never modified.
cdt = importlib.import_module("04_cross_dataset_training")
vis07 = importlib.import_module("07_attribution_vis_all")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

_SPATIAL_MODEL_HINT = "recon"


def load_new_chip_curves(exp_path, curve_type):
    """Like cdt.load_curve_data, but doesn't require a LABEL_MAPPINGS entry."""
    data_path = os.path.join(exp_path, config.TRAINING_DATA_PATH)
    if not os.path.exists(data_path):
        print(f"[!] '{data_path}' not found.")
        return None
    data = joblib.load(data_path)
    data = config.apply_well_exclusion(data, exp_path.name)
    dataset_name = list(data["dataset_name"])
    try:
        idx, _ = config.resolve_curve_dataset_idx(curve_type, dataset_name)
    except ValueError as e:
        print(f"[!] {e}")
        return None
    return {
        "curves": data["dataset"][idx],
        "timestamps": np.asarray(data["timestamps"], dtype=float),
        "Y_well_raw": np.asarray(data["Y_well"]),
    }


def align_new_chip(new_chip_path, out_dir, curve_type, curve_alignment, pc_ttp_anchor):
    """Returns (curves, resampler, Y_well_raw), or None on failure."""
    d = load_new_chip_curves(new_chip_path, curve_type)
    if d is None:
        return None

    resampler_path = out_dir / config.CROSS_DATASET_RESAMPLER_PATH.format(curve_type=curve_type)
    if not resampler_path.exists():
        print(f"[!] No saved resampler at {resampler_path}. Run 04_cross_dataset_training.py "
              f"--train_full for this group/curve_type first.")
        return None
    resampler = joblib.load(resampler_path)

    timestamps, curves = d["timestamps"], d["curves"]

    if curve_alignment == "pc_ttp":
        recipe_path = out_dir / config.CROSS_DATASET_PC_TTP_RECIPE_PATH.format(curve_type=curve_type)
        if not recipe_path.exists():
            print(f"[!] No saved pc_ttp recipe at {recipe_path}. Run 04 with "
                  f"--curve_alignment pc_ttp --train_full for this group/curve_type first.")
            return None
        recipe = joblib.load(recipe_path)

        pc = cdt.load_pc_wells_snapshot(new_chip_path, curve_type)
        if pc is None:
            print("[!] New chip has no PC snapshot -- can't compute its PC TTP for pc_ttp alignment.")
            return None
        new_ttp = cdt.compute_ct_for_curves(pc["curves"], pc["timestamps"])
        if new_ttp is None:
            print("[!] Could not fit a PC TTP for the new chip.")
            return None

        timestamps, curves, shift = cdt.shift_to_pc_ttp_anchor(timestamps, curves, new_ttp, recipe["anchor"])
        timestamps, curves = cdt.truncate_to_common_duration(timestamps, curves, recipe["common_duration"])
        print(f"  [PC-TTP align] new chip TTP={new_ttp:.2f}  anchor={recipe['anchor']:.2f}  shift={shift:.2f}")

    curves = resampler.transform(timestamps, curves)
    if curve_type.endswith("_norm") and curve_alignment == "pc_ttp":
        curves = cdt.normalize_curves_minmax(curves)

    return curves, resampler, d["Y_well_raw"]


if __name__ == "__main__":
    print(f"\n{'='*70}\n[RUNNING] {os.path.basename(__file__)}\n{'='*70}\n")
    parser = argparse.ArgumentParser(description="Predict on a new chip using a saved --train_full cross-dataset model")
    parser.add_argument("--exp_folder", type=str, default=config.DEFAULT_EXP_FOLDER,
                        help="Root containing the TRAINED group's cross_dataset_cv/ output.")
    parser.add_argument("--group", type=str, required=True,
                        help="Which CROSS_DATASET_GROUPS group's trained model/resampler/recipe to use.")
    parser.add_argument("--new_chip_folder", type=str, required=True,
                        help="Path to the new chip's own experiment folder (containing "
                             "curve_for_training.joblib, already produced by 01_curve_preprocessing_v6.py).")
    parser.add_argument("--curve_type", type=str, required=True,
                        help="Must match a curve_type the group was trained/--train_full'd with.")
    parser.add_argument("--model", type=str, required=True,
                        help="Model key of the saved .keras model to use (e.g. cnn_gru_dual). "
                             "Deep-learning curve models only -- KNN/manual-feature models and "
                             "cosine_recon/attn_recon (spatial neighbor-stack) models aren't supported here.")
    parser.add_argument("--outlier_filter", type=str, default="none",
                        help="Must match the outlier_filter the target model was saved under.")
    parser.add_argument("--mode", type=str, choices=["lofo", "random_split", "kfold"], default="lofo",
                        help="Must match the --mode used for the group's --train_full run.")
    parser.add_argument("--n_splits", type=int, default=5,
                        help="Must match --n_splits used for the group's --train_full run (only for --mode kfold).")
    parser.add_argument("--curve_alignment", type=str, choices=config.CURVE_ALIGNMENT_CHOICES,
                        default="acquisition_start",
                        help="Must match the --curve_alignment used for the group's --train_full run.")
    parser.add_argument("--pc_ttp_anchor", type=str, choices=["min", "percentile"], default="min",
                        help="Must match the --pc_ttp_anchor used for the group's --train_full run "
                             "(only used when --curve_alignment pc_ttp).")
    args = parser.parse_args()

    if _SPATIAL_MODEL_HINT in args.model:
        sys.exit(f"[!] '{args.model}' needs a spatial neighbor-stack input "
                  f"(build_neighbor_curve_stack) -- not supported by this script.")

    if args.group not in config.CROSS_DATASET_GROUPS:
        sys.exit(f"[!] Unknown group '{args.group}'. Known groups: {list(config.CROSS_DATASET_GROUPS.keys())}")

    out_dir = Path(args.exp_folder) / "cross_dataset_cv" / args.group
    if args.curve_alignment == "pc_ttp":
        out_dir = out_dir / "curve_alignment_pc_ttp" / f"anchor_{args.pc_ttp_anchor}"

    result = align_new_chip(Path(args.new_chip_folder), out_dir, args.curve_type,
                            args.curve_alignment, args.pc_ttp_anchor)
    if result is None:
        sys.exit(1)
    curves, resampler, Y_well_raw = result

    _mode_str = args.mode if args.mode != "kfold" else f"kfold{args.n_splits}"
    full_model_dir = out_dir / "model_interpretation" / f"full_data_{_mode_str}"
    results_file_path = out_dir / config.CROSS_DATASET_RESULT_PATH.format(mode=_mode_str, curve_type=args.curve_type)
    lofo_results = joblib.load(results_file_path) if results_file_path.exists() else {}
    class_names = lofo_results.get("full_data", {}).get("class_names")
    if class_names is None:
        print("[!] No class_names saved for this group's full_data run -- reporting raw integer class ids.")

    filter_key = "None" if args.outlier_filter.lower() == "none" else args.outlier_filter
    expected_seq_len = len(resampler.t_grid)
    models = vis07.load_saved_models(full_model_dir, filter_key, expected_seq_len,
                                     curve_type=args.curve_type, model_names=[args.model])
    if args.model not in models:
        sys.exit(f"[!] Could not load '{args.model}' from {full_model_dir} "
                 f"(filter={filter_key}, curve_type={args.curve_type}).")
    model = models[args.model]

    X = curves[..., None].astype(np.float32)
    out = model.predict(X, verbose=0)
    probs = out[0] if isinstance(out, list) else out
    pred_idx = np.argmax(probs, axis=1)
    pred_labels = [class_names[i] if class_names else int(i) for i in pred_idx]

    print(f"\n{'='*70}\nPredictions for {len(pred_idx)} pixels ({Path(args.new_chip_folder).name})\n{'='*70}")
    for i in range(len(pred_idx)):
        well = Y_well_raw[i] if i < len(Y_well_raw) else "?"
        print(f"  pixel {i:>5d} (well {well}): {pred_labels[i]}  "
              f"confidence={probs[i, pred_idx[i]]*100:.1f}%")

    out_path = Path(args.new_chip_folder) / f"predictions_{args.group}_{args.curve_type}_{args.model}.joblib"
    joblib.dump({
        "pred_idx": pred_idx, "pred_labels": pred_labels, "probs": probs,
        "class_names": class_names, "Y_well_raw": Y_well_raw,
        "group": args.group, "curve_type": args.curve_type, "model": args.model,
        "curve_alignment": args.curve_alignment,
    }, out_path, compress=3)
    print(f"\n  [*] Saved predictions -> {out_path}")
