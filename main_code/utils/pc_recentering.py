import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "utils", _ROOT / "utils" / "model_training"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import config
from chip import cross_dataset_training as cdt
from model_utils import build_neighbor_curve_stack, _frac_suffix
from cross_dataset_result_io import filter_token

_SPATIAL_MODEL_HINT = "attn_recon"


def is_spatial_model(model_key):
    return _SPATIAL_MODEL_HINT in model_key


def cls_layer(model):
    try:
        return model.get_layer('cls_out')
    except ValueError:
        return model.layers[-1]


def compute_embeddings(model, curves):
    embed_model = tf.keras.Model(inputs=model.input, outputs=cls_layer(model).input)
    return embed_model.predict(curves[..., None].astype(np.float32), verbose=0)


def compute_embeddings_stack(model, stack):
    embed_model = tf.keras.Model(inputs=model.input, outputs=cls_layer(model).input)
    return embed_model.predict(stack.astype(np.float32), verbose=0)


def infer_k(model):
    return model.input_shape[1] - 1


def pc_mean_stack(pc_curves, k):
    mean_curve = pc_curves.mean(axis=0)
    return np.tile(mean_curve, (k + 1, 1))[None, ...]


def recentered_predict(model, model_key, X_test, ref_embed, new_chip_embed):
    is_spatial = is_spatial_model(model_key)
    embeddings = (compute_embeddings_stack(model, X_test) if is_spatial
                  else compute_embeddings(model, X_test))
    shift = ref_embed - new_chip_embed
    probs = cls_layer(model)(embeddings + shift).numpy()
    return probs, float(np.linalg.norm(shift))


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


def align_new_chip(new_chip_path, out_dir, curve_type, group_name=None, held_out_chip=None):
    d = load_new_chip_curves(new_chip_path, curve_type, group_name=group_name)
    if d is None:
        return None

    resampler_path = _resolve_alignment_path(out_dir, config.CROSS_DATASET_RESAMPLER_PATH, curve_type, held_out_chip)
    if not resampler_path.exists():
        print(f"[!] No saved resampler at {resampler_path}.")
        return None
    resampler = joblib.load(resampler_path)

    recipe_path = _resolve_alignment_path(out_dir, config.CROSS_DATASET_PC_TTP_RECIPE_PATH, curve_type, held_out_chip)
    if not recipe_path.exists():
        print(f"[!] No saved pc_ttp recipe at {recipe_path}.")
        return None
    recipe = joblib.load(recipe_path)

    timestamps, curves = d["timestamps"], d["curves"]
    pc = cdt.load_pc_wells_snapshot(new_chip_path, curve_type)
    pc_timestamps, pc_curves = (pc["timestamps"], pc["curves"]) if pc is not None else (None, None)

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
    pc_curves_aligned = resampler.transform(pc_timestamps, pc_curves)
    if curve_type.endswith("_norm"):
        curves = cdt.normalize_curves_minmax(curves)
        pc_curves_aligned = cdt.normalize_curves_minmax(pc_curves_aligned)

    return curves, resampler, d["Y_well_raw"], pc_curves_aligned, d["coords"], d["well_ids"]


def _build_training_pc_embeddings(model, exp_paths, out_dir, curve_type, k, held_out_chip):
    resampler_path = _resolve_alignment_path(out_dir, config.CROSS_DATASET_RESAMPLER_PATH, curve_type, held_out_chip)
    resampler = joblib.load(resampler_path)
    recipe_path = _resolve_alignment_path(out_dir, config.CROSS_DATASET_PC_TTP_RECIPE_PATH, curve_type, held_out_chip)
    recipe = joblib.load(recipe_path)

    pc_curves_all = []
    for exp_path in exp_paths:
        pc = cdt.load_pc_wells_snapshot(exp_path, curve_type)
        if pc is None:
            continue
        timestamps, curves = pc["timestamps"], pc["curves"]
        ttp = recipe["pc_ttp_per_chip"].get(exp_path.name)
        timestamps, curves, _ = cdt.shift_to_pc_ttp_anchor(timestamps, curves, ttp, recipe["anchor"])
        timestamps, curves = cdt.truncate_to_common_duration(timestamps, curves, recipe["common_duration"])
        curves = resampler.transform(timestamps, curves)
        if curve_type.endswith("_norm"):
            curves = cdt.normalize_curves_minmax(curves)
        pc_curves_all.append(pc_mean_stack(curves, k) if k is not None else curves)

    if not pc_curves_all:
        return None
    stacked = np.concatenate(pc_curves_all, axis=0)
    return (compute_embeddings_stack(model, stacked) if k is not None
            else compute_embeddings(model, stacked))


def reference_pc_embedding(model, model_key, exp_paths, out_dir, curve_type, filter_key,
                           force_rerun=False, held_out_chip=None):
    align_dir = config.cross_dataset_alignment_dir(out_dir, held_out_chip)
    embed_path = align_dir / config.CROSS_DATASET_PC_EMBED_PATH.format(
        model=model_key, filter=filter_token(filter_key), curve_type=curve_type)
    if not force_rerun and embed_path.exists():
        return joblib.load(embed_path)

    k = infer_k(model) if is_spatial_model(model_key) else None
    embeddings = _build_training_pc_embeddings(model, exp_paths, out_dir, curve_type, k, held_out_chip)
    if embeddings is None:
        return None
    mean_embed = embeddings.mean(axis=0)

    embed_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(mean_embed, embed_path, compress=3)
    print(f"  [*] Saved PC reference embedding -> {embed_path}")
    return mean_embed


def predict_new_chip(model, model_key, curves, coords, well_ids, pc_curves_aligned,
                     exp_paths, out_dir, curve_type, filter_key,
                     pc_recenter=False, force_rerun=False, held_out_chip=None):
    is_spatial = is_spatial_model(model_key)
    if is_spatial:
        k = infer_k(model)
        X = build_neighbor_curve_stack(curves, coords, well_ids, k)
    else:
        X = curves[..., None].astype(np.float32)

    shift_norm = None
    if pc_recenter:
        if pc_curves_aligned is None or len(pc_curves_aligned) == 0:
            raise ValueError("pc_recenter requires the chip's own PC well curves, none found.")
        ref_embed = reference_pc_embedding(model, model_key, exp_paths, out_dir, curve_type,
                                           filter_key, force_rerun=force_rerun, held_out_chip=held_out_chip)
        if is_spatial:
            new_chip_embed = compute_embeddings_stack(model, pc_mean_stack(pc_curves_aligned, k))[0]
            embeddings = compute_embeddings_stack(model, X)
        else:
            new_chip_embed = compute_embeddings(model, pc_curves_aligned).mean(axis=0)
            embeddings = compute_embeddings(model, curves)
        shift = ref_embed - new_chip_embed
        shift_norm = float(np.linalg.norm(shift))
        out = cls_layer(model)(embeddings + shift).numpy()
    else:
        out = model.predict(X, verbose=0)
    probs = out[0] if isinstance(out, list) else out
    return probs, shift_norm


def load_saved_models(model_dir, filter_key, model_names, curve_type, train_center_frac=None):
    models = {}
    frac_tag = _frac_suffix(train_center_frac)
    for model_key in model_names:
        model_path = Path(model_dir) / f"{model_key}_{filter_key}_{curve_type}{frac_tag}_model.keras"
        if not model_path.exists():
            continue
        models[model_key] = tf.keras.models.load_model(model_path, compile=False)
    return models
