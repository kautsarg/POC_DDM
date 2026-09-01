import os
import sys
import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

sys.path.insert(0, 'utils')
sys.path.insert(0, 'utils/model_training')
import config
from cross_dataset_result_io import load_partitioned

# Import-only, sibling scripts never modified.
cdt = importlib.import_module("04_cross_dataset_training")
vis07 = importlib.import_module("07_attribution_vis_all")
from model_utils import build_neighbor_curve_stack

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

_SPATIAL_MODEL_HINT = "recon"


def _is_spatial(model_key):
    return _SPATIAL_MODEL_HINT in model_key


def load_new_chip_curves(exp_path, curve_type, group_name=None):
    data_path = os.path.join(exp_path, config.TRAINING_DATA_PATH)
    if not os.path.exists(data_path):
        print(f"[!] '{data_path}' not found.")
        return None
    data = joblib.load(data_path)
    data = config.apply_well_exclusion(data, exp_path.name, group_name=group_name)
    dataset_name = list(data["dataset_name"])
    try:
        idx, _ = config.resolve_curve_dataset_idx(curve_type, dataset_name)
    except ValueError as e:
        print(f"[!] {e}")
        return None
    Y_well_raw = np.asarray(data["Y_well"])

    coords, well_ids = None, None
    if "metadata" in data:
        metadata_df = pd.DataFrame(data["metadata"])
        if {"pixel_row_idx", "pixel_col_idx"}.issubset(metadata_df.columns):
            coords = np.stack([
                metadata_df["pixel_row_idx"].values.astype(float),
                metadata_df["pixel_col_idx"].values.astype(float),
            ], axis=1)
            well_id_local = (metadata_df["well_id"].values if "well_id" in metadata_df.columns
                              else Y_well_raw)
            well_ids = np.array([f"{exp_path.name}::{w}" for w in well_id_local], dtype=object)

    return {
        "curves": data["dataset"][idx],
        "timestamps": np.asarray(data["timestamps"], dtype=float),
        "Y_well_raw": Y_well_raw,
        "coords": coords,
        "well_ids": well_ids,
    }


def _resolve_alignment_path(out_dir, path_template, curve_type, held_out_chip):
    if held_out_chip is not None:
        fold_path = config.cross_dataset_alignment_dir(out_dir, held_out_chip) / path_template.format(curve_type=curve_type)
        if fold_path.exists():
            return fold_path
    return out_dir / path_template.format(curve_type=curve_type)


def align_new_chip(new_chip_path, out_dir, curve_type, curve_alignment, pc_ttp_anchor, group_name=None,
                   held_out_chip=None, exclusion_group=None):
    d = load_new_chip_curves(new_chip_path, curve_type, group_name=exclusion_group if exclusion_group is not None else group_name)
    if d is None:
        return None

    resampler_path = _resolve_alignment_path(out_dir, config.CROSS_DATASET_RESAMPLER_PATH, curve_type, held_out_chip)
    if not resampler_path.exists():
        print(f"[!] No saved resampler at {resampler_path}. Run 04_cross_dataset_training.py "
              f"--train_full for this group/curve_type first.")
        return None
    resampler = joblib.load(resampler_path)

    timestamps, curves = d["timestamps"], d["curves"]

    pc = cdt.load_pc_wells_snapshot(new_chip_path, curve_type)
    pc_timestamps, pc_curves = (pc["timestamps"], pc["curves"]) if pc is not None else (None, None)

    if curve_alignment == "pc_ttp":
        recipe_path = _resolve_alignment_path(out_dir, config.CROSS_DATASET_PC_TTP_RECIPE_PATH, curve_type, held_out_chip)
        if not recipe_path.exists():
            print(f"[!] No saved pc_ttp recipe at {recipe_path}. Run 04 with "
                  f"--curve_alignment pc_ttp --train_full for this group/curve_type first.")
            return None
        recipe = joblib.load(recipe_path)

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

        pc_timestamps, pc_curves, _ = cdt.shift_to_pc_ttp_anchor(pc_timestamps, pc_curves, new_ttp, recipe["anchor"])
        pc_timestamps, pc_curves = cdt.truncate_to_common_duration(pc_timestamps, pc_curves, recipe["common_duration"])

    curves = resampler.transform(timestamps, curves)
    if curve_type.endswith("_norm") and curve_alignment == "pc_ttp":
        curves = cdt.normalize_curves_minmax(curves)

    pc_curves_aligned = None
    if pc_curves is not None:
        pc_curves_aligned = resampler.transform(pc_timestamps, pc_curves)
        if curve_type.endswith("_norm") and curve_alignment == "pc_ttp":
            pc_curves_aligned = cdt.normalize_curves_minmax(pc_curves_aligned)

    return curves, resampler, d["Y_well_raw"], pc_curves_aligned, d["coords"], d["well_ids"]


def _cls_layer(model):
    """The final classification Dense(softmax) layer -- named 'cls_out' for DANN/SupCon
    models, otherwise the model's last layer."""
    try:
        return model.get_layer('cls_out')
    except ValueError:
        return model.layers[-1]


def compute_embeddings(model, curves):
    """Runs curves through model up to (not including) its classification layer."""
    embed_model = tf.keras.Model(inputs=model.input, outputs=_cls_layer(model).input)
    return embed_model.predict(curves[..., None].astype(np.float32), verbose=0)


def _infer_k(model):
    """k (neighbour count) for a spatial model, read off its own input shape
    (None, k+1, T) -- no separately saved artifact needed."""
    return model.input_shape[1] - 1


def compute_embeddings_stack(model, stack):
    embed_model = tf.keras.Model(inputs=model.input, outputs=_cls_layer(model).input)
    return embed_model.predict(stack.astype(np.float32), verbose=0)


def _pc_mean_stack(pc_curves, k):
    mean_curve = pc_curves.mean(axis=0)
    return np.tile(mean_curve, (k + 1, 1))[None, ...]


def _build_training_pc_embeddings(model, exp_paths, out_dir, curve_type, curve_alignment, k=None,
                                  held_out_chip=None):
    resampler_path = _resolve_alignment_path(out_dir, config.CROSS_DATASET_RESAMPLER_PATH, curve_type, held_out_chip)
    resampler = joblib.load(resampler_path)

    recipe = None
    if curve_alignment == "pc_ttp":
        recipe_path = _resolve_alignment_path(out_dir, config.CROSS_DATASET_PC_TTP_RECIPE_PATH, curve_type, held_out_chip)
        recipe = joblib.load(recipe_path)

    pc_curves_all = []
    for exp_path in exp_paths:
        pc = cdt.load_pc_wells_snapshot(exp_path, curve_type)
        if pc is None:
            continue
        timestamps, curves = pc["timestamps"], pc["curves"]
        if curve_alignment == "pc_ttp":
            ttp = recipe["pc_ttp_per_chip"].get(exp_path.name)
            timestamps, curves, _ = cdt.shift_to_pc_ttp_anchor(timestamps, curves, ttp, recipe["anchor"])
            timestamps, curves = cdt.truncate_to_common_duration(timestamps, curves, recipe["common_duration"])
        curves = resampler.transform(timestamps, curves)
        if curve_type.endswith("_norm") and curve_alignment == "pc_ttp":
            curves = cdt.normalize_curves_minmax(curves)
        if k is not None:
            pc_curves_all.append(_pc_mean_stack(curves, k))
        else:
            pc_curves_all.append(curves)

    if k is not None:
        return compute_embeddings_stack(model, np.concatenate(pc_curves_all, axis=0))
    return compute_embeddings(model, np.concatenate(pc_curves_all, axis=0))


def reference_pc_embedding(model, model_key, exp_paths, out_dir, curve_type, filter_key,
                           curve_alignment, force_rerun=False, held_out_chip=None):
    align_dir = config.cross_dataset_alignment_dir(out_dir, held_out_chip)
    embed_path = align_dir / config.CROSS_DATASET_PC_EMBED_PATH.format(
        model=model_key, filter=filter_key, curve_type=curve_type)
    if not force_rerun and embed_path.exists():
        return joblib.load(embed_path)

    k = _infer_k(model) if _is_spatial(model_key) else None
    embeddings = _build_training_pc_embeddings(model, exp_paths, out_dir, curve_type, curve_alignment, k=k,
                                               held_out_chip=held_out_chip)
    mean_embed = embeddings.mean(axis=0)
    embed_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(mean_embed, embed_path, compress=3)
    print(f"  [*] Saved PC reference embedding -> {embed_path}")
    return mean_embed


def predict_new_chip(model, model_key, curves, coords, well_ids, pc_curves_aligned,
                     exp_paths, out_dir, curve_type, filter_key, curve_alignment,
                     pc_recenter=False, force_rerun=False, held_out_chip=None):
    is_spatial = _is_spatial(model_key)
    if is_spatial:
        k = _infer_k(model)
        X = build_neighbor_curve_stack(curves, coords, well_ids, k)
    else:
        X = curves[..., None].astype(np.float32)

    shift_norm = None
    if pc_recenter:
        if pc_curves_aligned is None or len(pc_curves_aligned) == 0:
            raise ValueError("pc_recenter requires the chip's own PC well curves, none found.")
        ref_embed = reference_pc_embedding(model, model_key, exp_paths, out_dir, curve_type,
                                           filter_key, curve_alignment, force_rerun=force_rerun,
                                           held_out_chip=held_out_chip)
        if is_spatial:
            new_chip_embed = compute_embeddings_stack(model, _pc_mean_stack(pc_curves_aligned, k))[0]
            embeddings = compute_embeddings_stack(model, X)
        else:
            new_chip_embed = compute_embeddings(model, pc_curves_aligned).mean(axis=0)
            embeddings = compute_embeddings(model, curves)
        shift = ref_embed - new_chip_embed
        shift_norm = float(np.linalg.norm(shift))
        out = _cls_layer(model)(embeddings + shift).numpy()
    else:
        out = model.predict(X, verbose=0)
    probs = out[0] if isinstance(out, list) else out
    return probs, shift_norm


if __name__ == "__main__":
    print(f"\n{'='*70}\n[RUNNING] {os.path.basename(__file__)}\n{'='*70}\n")
    parser = argparse.ArgumentParser(description="Predict on a new chip using a saved --train_full cross-dataset model")
    parser.add_argument("--exp_folder", type=str, default=config.DEFAULT_EXP_FOLDER,
                        help="Root containing the TRAINED group's cross_dataset_cv/ output.")
    parser.add_argument("--group", type=str, required=True,
                        help="Which CROSS_DATASET_GROUPS group's trained model/resampler/recipe to use.")
    parser.add_argument("--exclusion_group", type=str, default=None,
                        help="Which group's LOFO_EXCLUDE_WELL_MAPPING to apply to the NEW chip's own "
                             "wells (e.g. to drop its PC/NC/no-amp wells before prediction). Defaults "
                             "to --group. Set this separately when the new chip belongs to a different "
                             "group than the one the model was trained on (cross-group generalization "
                             "testing) -- --group must still name the TRAINED group so the saved "
                             "model/resampler/recipe can be found.")
    parser.add_argument("--new_chip_folder", type=str, required=True,
                        help="Path to the new chip's own experiment folder (containing "
                             "curve_for_training.joblib, already produced by 01_curve_preprocessing_v6.py).")
    parser.add_argument("--curve_type", type=str, required=True,
                        help="Must match a curve_type the group was trained/--train_full'd with.")
    parser.add_argument("--model", type=str, required=True,
                        help="Model key of the saved .keras model to use (e.g. cnn_gru_dual). "
                             "Deep-learning curve models only -- KNN/manual-feature models aren't "
                             "supported here. cosine_recon/attn_recon (spatial neighbor-stack) "
                             "models are supported, but need pixel_row_idx/pixel_col_idx metadata "
                             "for the new chip.")
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
    parser.add_argument("--pc_recenter", action="store_true",
                        help="Option-B PC-well feature-space recentering: shift this chip's "
                             "embeddings by (training-pool PC embedding - this chip's own PC "
                             "embedding) before the classification layer. Opt-in, default off.")
    parser.add_argument("--force_rerun", action="store_true",
                        help="Recompute the cached PC reference embedding even if already saved. "
                             "No effect without --pc_recenter.")
    args = parser.parse_args()

    if args.group not in config.CROSS_DATASET_GROUPS:
        sys.exit(f"[!] Unknown group '{args.group}'. Known groups: {list(config.CROSS_DATASET_GROUPS.keys())}")

    out_dir = Path(args.exp_folder) / "cross_dataset_cv" / args.group
    if args.curve_alignment == "pc_ttp":
        out_dir = out_dir / "curve_alignment_pc_ttp" / f"anchor_{args.pc_ttp_anchor}"

    result = align_new_chip(Path(args.new_chip_folder), out_dir, args.curve_type,
                            args.curve_alignment, args.pc_ttp_anchor, group_name=args.group,
                            exclusion_group=args.exclusion_group)
    if result is None:
        sys.exit(1)
    curves, resampler, Y_well_raw, pc_curves_aligned, coords, well_ids = result

    is_spatial = _is_spatial(args.model)
    if is_spatial and (coords is None or well_ids is None):
        sys.exit(f"[!] '{args.model}' needs pixel_row_idx/pixel_col_idx metadata for "
                 f"{args.new_chip_folder}, but none was found -- can't build a neighbor stack.")

    _mode_str = args.mode if args.mode != "kfold" else f"kfold{args.n_splits}"
    full_model_dir = out_dir / "model_interpretation" / f"full_data_{_mode_str}"
    lofo_results = load_partitioned(out_dir, _mode_str, args.curve_type)
    class_names = lofo_results.get("full_data", {}).get("class_names")
    if class_names is None:
        print("[!] No class_names saved for this group's full_data run -- reporting raw integer class ids.")

    filter_key = "None" if args.outlier_filter.lower() == "none" else args.outlier_filter

    if filter_key == cdt.NOAMP_FILTER_NAME:
        keep = cdt.is_amplifying_mask(curves)
        print(f"  [*] {cdt.NOAMP_FILTER_NAME}: {int((~keep).sum())}/{len(keep)} "
              f"non-amplifying curves removed from {args.new_chip_folder}")
        curves = curves[keep]
        Y_well_raw = Y_well_raw[keep]
        if coords is not None:
            coords = coords[keep]
        if well_ids is not None:
            well_ids = well_ids[keep]

    expected_seq_len = len(resampler.t_grid)
    models = vis07.load_saved_models(full_model_dir, filter_key, expected_seq_len,
                                     curve_type=args.curve_type, model_names=[args.model])
    if args.model not in models:
        sys.exit(f"[!] Could not load '{args.model}' from {full_model_dir} "
                 f"(filter={filter_key}, curve_type={args.curve_type}).")
    model = models[args.model]

    exp_paths = [Path(args.exp_folder, name) for name in config.CROSS_DATASET_GROUPS[args.group]]
    try:
        probs, shift_norm = predict_new_chip(
            model, args.model, curves, coords, well_ids, pc_curves_aligned,
            exp_paths, out_dir, args.curve_type, filter_key, args.curve_alignment,
            pc_recenter=args.pc_recenter, force_rerun=args.force_rerun)
    except ValueError as e:
        sys.exit(f"[!] {e}")
    if shift_norm is not None:
        print(f"  [PC-recenter] shift magnitude (L2): {shift_norm:.4f}")
    pred_idx = np.argmax(probs, axis=1)
    pred_labels = [class_names[i] if class_names else int(i) for i in pred_idx]

    print(f"\n{'='*70}\nPredictions for {len(pred_idx)} pixels ({Path(args.new_chip_folder).name})\n{'='*70}")
    for i in range(len(pred_idx)):
        well = Y_well_raw[i] if i < len(Y_well_raw) else "?"
        print(f"  pixel {i:>5d} (well {well}): {pred_labels[i]}  "
              f"confidence={probs[i, pred_idx[i]]*100:.1f}%")

    _recenter_tag = "_pc_recenter" if args.pc_recenter else ""
    out_path = Path(args.new_chip_folder) / f"predictions_{args.group}_{args.curve_type}_{args.model}{_recenter_tag}.joblib"
    joblib.dump({
        "pred_idx": pred_idx, "pred_labels": pred_labels, "probs": probs,
        "class_names": class_names, "Y_well_raw": Y_well_raw,
        "group": args.group, "curve_type": args.curve_type, "model": args.model,
        "curve_alignment": args.curve_alignment,
        "pc_recenter": args.pc_recenter, "pc_recenter_shift_norm": shift_norm,
    }, out_path, compress=3)
    print(f"\n  [*] Saved predictions -> {out_path}")
