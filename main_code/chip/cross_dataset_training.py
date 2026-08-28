import os
import sys
import gc
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from joblib import Memory
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from kneed import KneeLocator

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "utils"))
sys.path.insert(0, str(_ROOT / "utils" / "model_training"))
sys.path.insert(0, str(_ROOT / "utils" / "outlier_detection"))

import config
from safe_io import safe_joblib_dump, safe_keras_save
from cross_dataset_result_io import save_partitioned, load_partitioned, filter_token
import sigmoid_fitting as sp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.keras.mixed_precision.set_global_policy('mixed_float16')
tf.config.optimizer.set_jit(True)

from model_utils import (evaluate_outlier_filters, plot_ml_results, set_global_determinism,
                          CurveResampler, build_well_stratified_random_split,
                          build_well_stratified_nfold_splits, build_neighbor_curve_stack)
from lstm_autoencoder_outlier import build_lstm_autoencoder
from pc_recentering import (cls_layer, compute_embeddings, compute_embeddings_stack,
                            infer_k, pc_mean_stack, recentered_predict, is_spatial_model)

PC_TTP_ANCHOR = "min"
PC_TTP_ANCHOR_PCT_DEFAULT = 10
K_NEIGHBORS = 24

MODEL_CHOICES = [
    'cnn_gru_dual', 'cnn_gru_dual_attn_recon',
    'cnn_gru_dual_supcon1', 'cnn_gru_dual_attn_recon_supcon1',
    'cnn_gru_dual_supcon3', 'cnn_gru_dual_attn_recon_supcon3',
    'cnn_gru_dual_dann', 'cnn_gru_dual_attn_recon_dann',
    'cnn_gru_dual_pc_recentering', 'cnn_gru_dual_attn_recon_pc_recentering',
]
_ENGINE_KEY = {
    'cnn_gru_dual': 'cnn_gru_dual',
    'cnn_gru_dual_attn_recon': 'cnn_gru_dual_attn_recon',
    'cnn_gru_dual_supcon1': 'cnn_gru_dual_supcon',
    'cnn_gru_dual_attn_recon_supcon1': 'cnn_gru_dual_attn_recon_supcon',
    'cnn_gru_dual_supcon3': 'cnn_gru_dual_supcon3',
    'cnn_gru_dual_attn_recon_supcon3': 'cnn_gru_dual_attn_recon_supcon3',
    'cnn_gru_dual_dann': 'cnn_gru_dual_dann',
    'cnn_gru_dual_attn_recon_dann': 'cnn_gru_dual_attn_recon_dann',
}
PC_RECENTER_SUFFIX = '_pc_recentering'

LOFO_AE_FILTER_NAME = "lofo_ae"
NOAMP_FILTER_NAME = "noamp_remove"


# ============================================================
# DATA LOADING
# ============================================================
def load_curve_data(exp_path, curve_type, group_name=None):
    data_path = os.path.join(exp_path, config.TRAINING_DATA_PATH)
    if not os.path.exists(data_path):
        print(f"  -> Skipping {exp_path.name}: '{data_path}' not found.")
        return None

    data = joblib.load(data_path)
    data = config.apply_well_exclusion(data, exp_path.name, group_name=group_name)
    dataset_name = list(data["dataset_name"])
    try:
        idx, _ = config.resolve_curve_dataset_idx(curve_type, dataset_name)
    except ValueError as e:
        print(f"  -> Skipping {exp_path.name}: {e}")
        return None

    label_mappings = config.get_label_mappings(exp_path)
    if exp_path.name not in label_mappings:
        print(f"  -> Skipping {exp_path.name}: no LABEL_MAPPINGS entry found (required for cross-dataset grouping).")
        return None
    mapping = label_mappings[exp_path.name]

    Y_well_raw = np.asarray(data["Y_well"])
    Y_mapped = np.array([mapping.get(w, w) for w in Y_well_raw])

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

    concentration_raw = (np.asarray(data["concentration"], dtype=object)
                         if data.get("concentration") is not None
                         else np.full(len(Y_well_raw), None, dtype=object))

    return {
        "curves": data["dataset"][idx],
        "features_df": data["kinetic_features"][idx].loc[:, ~data["kinetic_features"][idx].columns.duplicated()].reset_index(drop=True),
        "Y_well_raw": Y_well_raw,
        "Y_mapped": Y_mapped,
        "timestamps": np.asarray(data["timestamps"], dtype=float),
        "dataset_id": exp_path.name,
        "coords": coords,
        "well_ids": well_ids,
        "concentration_raw": concentration_raw,
    }


def load_pc_wells_snapshot(exp_path, curve_type):
    """Reads chip/preprocessing.py's drop_pc PC snapshot from curve_for_training.joblib."""
    data_path = os.path.join(exp_path, config.TRAINING_DATA_PATH)
    if not os.path.exists(data_path):
        return None
    data = joblib.load(data_path)
    pc = data.get("pc_wells")
    if not pc:
        print(f"  [!] {exp_path.name}: no 'pc_wells' snapshot.")
        return None
    resolved = config.CURVE_TYPE_ALIASES.get(curve_type, curve_type)
    if resolved not in pc["curves"]:
        print(f"  [!] {exp_path.name}: PC snapshot has no '{resolved}' variant "
              f"(available: {list(pc['curves'].keys())}).")
        return None
    return {
        "curves": np.asarray(pc["curves"][resolved]),
        "timestamps": np.asarray(data["timestamps"], dtype=float),
    }


def normalize_curves_minmax(curves):
    curves = np.asarray(curves, dtype=np.float64)
    row_min = curves.min(axis=1, keepdims=True)
    row_max = curves.max(axis=1, keepdims=True)
    denom = np.where(row_max - row_min == 0, 1, row_max - row_min)
    return (curves - row_min) / denom


def compute_ct_for_curves(curves, timestamps):
    """Ct fit on the mean curve across the batch (one fit on the average, since PC
    is a designed reference well, not a biological replicate set)."""
    curves = np.asarray(curves)
    invalid = np.isnan(curves).any(axis=1) | np.isinf(curves).any(axis=1)
    valid_curves = curves[~invalid]
    if len(valid_curves) == 0 or np.isfinite(timestamps).sum() < 3:
        return None
    mean_curve = valid_curves.mean(axis=0)
    try:
        feats = sp.extract_kinetic_parameters_original(timestamps, mean_curve)
        return float(feats["Ct"]) if "Ct" in feats and np.isfinite(feats["Ct"]) else None
    except Exception:
        return None


def _pc_ttp_for_one_chip(exp_path_str, curve_type):
    exp_path = Path(exp_path_str)
    pc = load_pc_wells_snapshot(exp_path, curve_type)
    if pc is None:
        return None
    return compute_ct_for_curves(pc["curves"], pc["timestamps"])


def pc_ttp_per_chip(exp_paths, curve_type, cache_dir):
    """{chip_name: mean PC Ct}, disk-cached under cache_dir (sigmoid-fitting is slow)."""
    cached_fn = Memory(str(cache_dir), verbose=0).cache(_pc_ttp_for_one_chip)
    ttp = {}
    for exp_path in exp_paths:
        ct = cached_fn(str(exp_path), curve_type)
        if ct is None:
            print(f"  [!] {exp_path.name}: no PC TTP available for curve_type={curve_type!r}.")
            continue
        ttp[exp_path.name] = ct
    return ttp


def shift_to_pc_ttp_anchor(timestamps, curves, chip_ttp, anchor):
    shift = max(chip_ttp - anchor, 0.0) if chip_ttp is not None else 0.0
    start_idx = min(int(np.searchsorted(timestamps, timestamps[0] + shift)), len(timestamps) - 1)
    return timestamps[start_idx:], curves[:, start_idx:], shift


def truncate_to_common_duration(timestamps, curves, common_duration):
    end_time = timestamps[0] + common_duration
    end_idx = min(int(np.searchsorted(timestamps, end_time)) + 1, len(timestamps))
    return timestamps[:end_idx], curves[:, :end_idx]


def align_parts_to_pc_ttp(parts, pc_ttp, held_out_chip, anchor_method, anchor_pct, verbose=True):
    train_ttps = [v for k, v in pc_ttp.items() if k != held_out_chip]
    if not train_ttps:
        raise ValueError("No PC TTP available for any training chip.")
    if anchor_method == "min":
        anchor = min(train_ttps)
    elif anchor_method == "percentile":
        anchor = float(np.percentile(train_ttps, anchor_pct))
    else:
        raise ValueError(f"Unknown anchor_method: {anchor_method!r}")

    aligned = []
    shifts = {}
    for p in parts:
        ttp = pc_ttp.get(p["dataset_id"])
        t2, c2, shift = shift_to_pc_ttp_anchor(p["timestamps"], p["curves"], ttp, anchor)
        shifts[p["dataset_id"]] = shift
        aligned.append({**p, "timestamps": t2, "curves": c2})

    train_lens = [p["timestamps"][-1] - p["timestamps"][0]
                  for p in aligned if p["dataset_id"] != held_out_chip]
    common_duration = min(train_lens)
    for p2 in aligned:
        p2["timestamps"], p2["curves"] = truncate_to_common_duration(
            p2["timestamps"], p2["curves"], common_duration)

    if verbose:
        print(f"  [PC-TTP align] anchor ({anchor_method}) = {anchor:.2f}"
              + (f" | held_out={held_out_chip}" if held_out_chip else ""))
        for name, s in shifts.items():
            flag = " <- CLIPPED (TTP below anchor)" if pc_ttp.get(name, anchor) < anchor else ""
            print(f"    {name}: shift={s:8.2f}{flag}")
        print(f"  [PC-TTP align] common_duration = {common_duration:.2f}")

    return aligned, anchor, common_duration


def combine_group(exp_paths, group_name, curve_type, held_out_chip, pc_ttp_cache_dir,
                  anchor_method=PC_TTP_ANCHOR, anchor_pct=PC_TTP_ANCHOR_PCT_DEFAULT):
    """Concatenates a group's chips onto one common time grid, zero-referenced at
    each chip's PC-well time-to-positivity (pc_ttp alignment, always on)."""
    parts = []
    for exp_path in exp_paths:
        d = load_curve_data(exp_path, curve_type, group_name=group_name)
        if d is None:
            continue
        parts.append(d)

    if len(parts) < 2:
        print(f"  -> Skipping group '{group_name}': fewer than 2 usable datasets found.")
        return None

    pc_ttp = pc_ttp_per_chip(exp_paths, curve_type, pc_ttp_cache_dir)
    parts, anchor, common_duration = align_parts_to_pc_ttp(parts, pc_ttp, held_out_chip, anchor_method, anchor_pct)

    timestamps_zeroed = [p["timestamps"] - p["timestamps"][0] for p in parts]
    resampler = CurveResampler.fit(timestamps_zeroed)
    print(f"  [*] Resampling curves onto common grid: {len(resampler.t_grid)} points, "
          f"duration={resampler.t_grid[-1]:.4g}")
    for p in parts:
        p["curves"] = resampler.transform(p["timestamps"], p["curves"])

    if all(p["coords"] is not None for p in parts):
        coords_combined = np.concatenate([p["coords"] for p in parts], axis=0)
        well_ids_combined = np.concatenate([p["well_ids"] for p in parts], axis=0)
    else:
        missing = [p["dataset_id"] for p in parts if p["coords"] is None]
        if missing:
            print(f"  [*] No pixel_row_idx/pixel_col_idx metadata for: {missing} -- "
                  f"cnn_gru_dual_attn_recon will be skipped for group '{group_name}'.")
        coords_combined, well_ids_combined = None, None

    combined_curves = np.concatenate([p["curves"] for p in parts], axis=0)
    if curve_type.endswith("_norm"):
        combined_curves = normalize_curves_minmax(combined_curves)
        print("  [*] Re-normalized aligned window to [0,1] per curve.")

    return {
        "curves": combined_curves,
        "features_df": pd.concat([p["features_df"] for p in parts], axis=0, ignore_index=True),
        "Y_mapped": np.concatenate([p["Y_mapped"] for p in parts], axis=0),
        "dataset_id": np.concatenate([np.full(len(p["Y_mapped"]), p["dataset_id"], dtype=object) for p in parts], axis=0),
        "dataset_names": [p["dataset_id"] for p in parts],
        "resampler": resampler,
        "coords": coords_combined,
        "well_ids": well_ids_combined,
        "concentration_raw": np.concatenate([p["concentration_raw"] for p in parts], axis=0),
        "pc_ttp_recipe": {
            "anchor": anchor, "anchor_method": anchor_method,
            "common_duration": common_duration, "pc_ttp_per_chip": pc_ttp,
            "held_out_chip": held_out_chip, "curve_type": curve_type,
        },
    }


# ============================================================
# SPLITS
# ============================================================
def build_lofo_splits(dataset_id):
    splits = {}
    for name in np.unique(dataset_id):
        test_idx = np.where(dataset_id == name)[0]
        train_idx = np.where(dataset_id != name)[0]
        splits[f"lofo_{name}"] = (train_idx, test_idx)
    return splits


def build_random_split(y, well_ids=None, test_size=0.1, random_state=0):
    if well_ids is not None:
        return build_well_stratified_random_split(y, well_ids, test_size=test_size, random_state=random_state)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(np.zeros(len(y)), y))
    return {"random_split": (train_idx, test_idx)}


def build_nfold_splits(y, well_ids=None, n_splits=5, random_state=0):
    if well_ids is not None:
        return build_well_stratified_nfold_splits(y, well_ids, n_splits=n_splits, random_state=random_state)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return {f"fold_{i}": (tr, te) for i, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y))}


def _save_alignment_artifacts(combined, out_dir, curve_type, held_out_chip=None):
    align_dir = config.cross_dataset_alignment_dir(out_dir, held_out_chip)
    align_dir.mkdir(parents=True, exist_ok=True)
    resampler_path = align_dir / config.CROSS_DATASET_RESAMPLER_PATH.format(curve_type=curve_type)
    safe_joblib_dump(combined["resampler"], resampler_path, compress=3)
    print(f"  [*] Saved curve resampler -> {resampler_path}")
    recipe_path = align_dir / config.CROSS_DATASET_PC_TTP_RECIPE_PATH.format(curve_type=curve_type)
    safe_joblib_dump(combined["pc_ttp_recipe"], recipe_path, compress=3)
    print(f"  [*] Saved pc_ttp alignment recipe -> {recipe_path}")


def is_amplifying_mask(curves):
    return curves[:, -1] >= curves[:, 0]


def _fit_lofo_ae_filter(combined, train_idx, out_dir, curve_type, fold_label, force_rerun=False):
    base_path = out_dir / config.CROSS_DATASET_LOFO_AE_PATH.format(fold_label=fold_label, curve_type=curve_type)
    model_path = base_path.with_name(f"{base_path.name}_model.keras")
    meta_path = base_path.with_name(f"{base_path.name}_meta.joblib")

    curves = combined["curves"]
    dataset_id = np.asarray(combined["dataset_id"])

    if not force_rerun and model_path.exists() and meta_path.exists():
        autoencoder = tf.keras.models.load_model(model_path)
        meta = joblib.load(meta_path)
        scaler, threshold = meta["scaler"], meta["threshold"]
    else:
        train_curves = curves[train_idx]
        invalid_train = np.isnan(train_curves).any(axis=1) | np.isinf(train_curves).any(axis=1)
        X_valid = train_curves[~invalid_train]

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_valid)
        timesteps = X_scaled.shape[1]

        set_global_determinism(0)
        autoencoder = build_lstm_autoencoder(timesteps)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        autoencoder.fit(X_scaled.reshape(-1, timesteps, 1), X_scaled.reshape(-1, timesteps, 1),
                         epochs=60, batch_size=2048, shuffle=True, callbacks=[early_stop], verbose=0)

        recon = autoencoder.predict(X_scaled.reshape(-1, timesteps, 1), batch_size=2048, verbose=0).reshape(X_scaled.shape)
        train_mse = np.mean((X_scaled - recon) ** 2, axis=1)

        sorted_mse = np.sort(train_mse)
        kneedle = KneeLocator(np.arange(len(sorted_mse)), sorted_mse, curve="convex", direction="increasing")
        threshold = float(sorted_mse[kneedle.knee]) if kneedle.knee is not None else float(np.percentile(train_mse, 95))

        base_path.parent.mkdir(parents=True, exist_ok=True)
        safe_keras_save(autoencoder, model_path)
        safe_joblib_dump({"scaler": scaler, "threshold": threshold, "timesteps": timesteps}, meta_path, compress=3)
        print(f"  [*] Saved LOFO-AE outlier model -> {model_path}")

    invalid_all = np.isnan(curves).any(axis=1) | np.isinf(curves).any(axis=1)
    keep_mask = np.zeros(len(curves), dtype=bool)
    if (~invalid_all).any():
        X_scaled_all = scaler.transform(curves[~invalid_all])
        timesteps = X_scaled_all.shape[1]
        recon_all = autoencoder.predict(X_scaled_all.reshape(-1, timesteps, 1), batch_size=2048, verbose=0).reshape(X_scaled_all.shape)
        mse_all = np.mean((X_scaled_all - recon_all) ** 2, axis=1)
        keep_mask[~invalid_all] = mse_all <= threshold

    train_chips = set(dataset_id[train_idx])
    print(f"  [LOFO-AE filter] fold={fold_label} threshold(train elbow)={threshold:.5f}")
    for chip in np.unique(dataset_id):
        chip_mask = dataset_id == chip
        n_total = int(chip_mask.sum())
        n_kept = int((keep_mask & chip_mask).sum())
        tag = "train" if chip in train_chips else "held-out"
        pct_removed = 100 * (n_total - n_kept) / n_total if n_total else 0.0
        print(f"    {chip} [{tag}]: kept {n_kept}/{n_total} ({pct_removed:.1f}% removed)")

    return keep_mask


def _derive_pool_labels(combined):
    encoder = LabelEncoder()
    y_full = encoder.fit_transform(combined["Y_mapped"])
    chip_id_encoded = LabelEncoder().fit_transform(combined["dataset_id"])
    return encoder, y_full, chip_id_encoded


# ============================================================
# PC-RECENTERING
# ============================================================
def _align_chip_pc_curves(exp_path, resampler, pc_ttp_recipe, curve_type):
    pc = load_pc_wells_snapshot(exp_path, curve_type)
    if pc is None:
        return None
    timestamps, curves = pc["timestamps"], pc["curves"]
    ttp = pc_ttp_recipe["pc_ttp_per_chip"].get(Path(exp_path).name)
    timestamps, curves, _ = shift_to_pc_ttp_anchor(timestamps, curves, ttp, pc_ttp_recipe["anchor"])
    timestamps, curves = truncate_to_common_duration(timestamps, curves, pc_ttp_recipe["common_duration"])
    curves = resampler.transform(timestamps, curves)
    if curve_type.endswith("_norm"):
        curves = normalize_curves_minmax(curves)
    return curves


def _chip_pc_embedding(model, is_spatial, k, exp_path, resampler, pc_ttp_recipe, curve_type):
    curves = _align_chip_pc_curves(exp_path, resampler, pc_ttp_recipe, curve_type)
    if curves is None:
        return None
    if is_spatial:
        return compute_embeddings_stack(model, pc_mean_stack(curves, k))[0]
    return compute_embeddings(model, curves).mean(axis=0)


def _reference_pc_embedding(model, model_key, exp_paths_train, out_dir, curve_type, filter_key,
                            resampler, pc_ttp_recipe, is_spatial, k, held_out_chip, force_rerun):
    align_dir = config.cross_dataset_alignment_dir(out_dir, held_out_chip)
    embed_path = align_dir / config.CROSS_DATASET_PC_EMBED_PATH.format(
        model=model_key, filter=filter_token(filter_key), curve_type=curve_type)
    if not force_rerun and embed_path.exists():
        return joblib.load(embed_path)

    pc_curves_all = []
    for exp_path in exp_paths_train:
        curves = _align_chip_pc_curves(exp_path, resampler, pc_ttp_recipe, curve_type)
        if curves is None:
            continue
        pc_curves_all.append(pc_mean_stack(curves, k) if is_spatial else curves)
    if not pc_curves_all:
        return None
    stacked = np.concatenate(pc_curves_all, axis=0)
    mean_embed = (compute_embeddings_stack(model, stacked) if is_spatial
                  else compute_embeddings(model, stacked)).mean(axis=0)

    embed_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(mean_embed, embed_path, compress=3)
    print(f"  [*] Saved PC reference embedding -> {embed_path}")
    return mean_embed


def _apply_pc_recentering(res_entry, y_full, test_idx, combined, exp_paths, held_out_chip,
                          lofo_model_dir, out_dir, curve_type, filter_key, pc_variants, force_rerun):
    for variant in pc_variants:
        base_model = variant[:-len(PC_RECENTER_SUFFIX)]
        base_model = _ENGINE_KEY.get(base_model, base_model)
        if base_model not in config.MODEL_KEY_MAP or variant not in config.MODEL_KEY_MAP:
            continue
        base_preds_key = config.MODEL_KEY_MAP[base_model][0]
        if base_preds_key not in res_entry:
            continue

        model_path = lofo_model_dir / f"{base_model}_{filter_key}_{curve_type}_model.keras"
        if not model_path.exists():
            print(f"  [SKIP] {variant}: base model file missing at {model_path}")
            continue

        model = tf.keras.models.load_model(model_path, compile=False)
        spatial = is_spatial_model(base_model)
        k = infer_k(model) if spatial else None

        exp_paths_train = [p for p in exp_paths if p.name != held_out_chip]
        ref_embed = _reference_pc_embedding(
            model, base_model, exp_paths_train, out_dir, curve_type, filter_key,
            combined["resampler"], combined["pc_ttp_recipe"], spatial, k,
            held_out_chip=held_out_chip, force_rerun=force_rerun)
        held_out_path = next((p for p in exp_paths if p.name == held_out_chip), None)
        new_chip_embed = (_chip_pc_embedding(model, spatial, k, held_out_path, combined["resampler"],
                                             combined["pc_ttp_recipe"], curve_type)
                          if held_out_path is not None else None)

        if ref_embed is None or new_chip_embed is None:
            print(f"  [SKIP] {variant}: could not compute PC embeddings for {held_out_chip}.")
            tf.keras.backend.clear_session()
            continue

        if spatial:
            X_test = build_neighbor_curve_stack(
                combined["curves"][test_idx].astype(np.float32, copy=False),
                combined["coords"][test_idx], combined["well_ids"][test_idx], k=k)
        else:
            X_test = combined["curves"][test_idx]

        probs, shift_norm = recentered_predict(model, base_model, X_test, ref_embed, new_chip_embed)
        pred = np.argmax(probs, axis=1)
        pk, pbk, ck = config.MODEL_KEY_MAP[variant]
        res_entry[pk] = [pred]
        res_entry[pbk] = [probs]
        res_entry[ck] = [np.unique(y_full)]
        print(f"  [PC-RECENTER] {variant} | filter={filter_key} | shift_norm={shift_norm:.4f}")
        tf.keras.backend.clear_session()


# ============================================================
# PER-FOLD ORCHESTRATION
# ============================================================
def _process_fold(fold_idx, total_folds, fold_label, train_idx, test_idx, combined, y_full, encoder,
                  curve_type, models, pc_variants, outlier_filters, out_dir, plot_dir, group_name,
                  total_count, lofo_results, mode_str, args, chip_id_encoded, exp_paths):
    progress_pct = ((fold_idx + 1) / total_folds) * 100
    print(f"\n{'='*75}")
    print(f"[{fold_idx+1}/{total_folds} | {progress_pct:.1f}%] FOLD: {fold_label} | train={len(train_idx)} test={len(test_idx)}")
    print(f"{'='*75}")

    cached_fold = lofo_results.get(fold_label, {})

    def checkpoint(updated_results, fold_label=fold_label):
        lofo_results[fold_label] = updated_results
        save_partitioned(lofo_results, out_dir, mode_str, curve_type, compress=3, models=models)

    lofo_model_dir = out_dir / "model_interpretation" / fold_label
    lofo_model_dir.mkdir(parents=True, exist_ok=True)

    if LOFO_AE_FILTER_NAME in outlier_filters:
        keep_mask = _fit_lofo_ae_filter(combined, train_idx, out_dir, curve_type, fold_label,
                                         force_rerun=args.force_rerun)
        combined["features_df"][LOFO_AE_FILTER_NAME] = keep_mask.astype(int)

    res = evaluate_outlier_filters(
        X_curves=combined["curves"],
        features_df=combined["features_df"],
        y_encoded=y_full,
        outlier_filters=outlier_filters,
        dataset_name=group_name,
        mode_name=fold_label,
        cached_results=cached_fold,
        models=models,
        checkpoint_fn=checkpoint,
        cv_splits=[(train_idx, test_idx)],
        save_model_dir=lofo_model_dir,
        save_model_curve_type=curve_type,
        coords=combined["coords"],
        well_ids=combined["well_ids"],
        k_neighbors=K_NEIGHBORS,
        chip_id_encoded=chip_id_encoded,
        batch_size=args.batch_size,
    )

    lofo_results[fold_label] = res
    lofo_results[fold_label]["class_names"] = [str(c) for c in encoder.classes_]

    if pc_variants and args.mode == "lofo":
        held_out_chip = fold_label[len("lofo_"):]
        for f in outlier_filters:
            res_entry = lofo_results[fold_label].get(f)
            if res_entry is None:
                continue
            _apply_pc_recentering(res_entry, y_full, test_idx, combined, exp_paths, held_out_chip,
                                  lofo_model_dir, out_dir, curve_type, f, pc_variants, args.force_rerun)

    # pc_variants' result keys only exist after _apply_pc_recentering above -- must be
    # included here or they're silently dropped from what's persisted to disk.
    save_partitioned(lofo_results, out_dir, mode_str, curve_type, compress=3, models=models + pc_variants)

    features_df_all = combined["features_df"]
    snapshot_path = lofo_model_dir / f"xai_data_{curve_type}.joblib"
    safe_joblib_dump({
        "X_curves_test": combined["curves"][test_idx].astype(np.float32),
        "features_df_test": features_df_all.iloc[test_idx].reset_index(drop=True),
        "y_test": y_full[test_idx],
        "timestamps": combined["resampler"].t_grid,
        "group_name": group_name,
        "fold_label": fold_label,
    }, snapshot_path, compress=3)
    print(f"  [XAI] Saved LOFO test snapshot -> {snapshot_path}")

    plot_ml_results(
        results_dict=lofo_results[fold_label],
        outlier_filters=outlier_filters,
        dataset_name=group_name,
        mode_name=fold_label,
        total_count=total_count,
        save_prefix=os.path.join(plot_dir, fold_label),
    )

    gc.collect()


def _clear_cached_results(lofo_results, models, pc_variants):
    keys_to_clear = set()
    for m in list(models) + list(pc_variants):
        if m in config.MODEL_KEY_MAP:
            pk, pbk, ck = config.MODEL_KEY_MAP[m]
            keys_to_clear.update([pk, pbk, ck, f'train_history_{m}_'])
    for fold_res in lofo_results.values():
        if not isinstance(fold_res, dict):
            continue
        for filter_res in fold_res.values():
            if not isinstance(filter_res, dict):
                continue
            for k in list(filter_res.keys()):
                if k in keys_to_clear:
                    del filter_res[k]


# ============================================================
# MAIN
# ============================================================
def main(argv=None):
    print(f"\n{'='*70}\n[RUNNING] chip/cross_dataset_training.py\n{'='*70}\n")
    parser = argparse.ArgumentParser(description="Cross-Dataset Leave-One-Folder-Out (LOFO) Training")
    parser.add_argument("--task_id", type=int, default=0, help="Array Job ID -> index into CROSS_DATASET_GROUPS")
    parser.add_argument("--exp_folder", type=str, default=config.DEFAULT_EXP_FOLDER)
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--curve_type", type=str, nargs='+',
                        choices=list(config.CURVE_TYPE_ALIASES.keys()),
                        default=list(config.CURVE_TYPE_ALIASES.keys()))
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--outlier_filter", type=str, nargs='+',
                        choices=["none", LOFO_AE_FILTER_NAME, NOAMP_FILTER_NAME]
                                + [f for f in config.OUTLIER_FILTERS if f is not None],
                        default=["none"])
    parser.add_argument("--models", type=str, nargs='+', choices=MODEL_CHOICES, default=MODEL_CHOICES)
    parser.add_argument("--train_full", action="store_true",
                        help="After LOFO, also train one model on ALL data (no holdout) and save to "
                             "model_interpretation/full_data_{mode}/.")
    parser.add_argument("--mode", type=str, choices=["lofo", "random_split", "kfold"], default="lofo")
    parser.add_argument("--n_splits", type=int, default=5, help="Fold count for --mode kfold.")
    parser.add_argument("--test_size", type=float, default=0.1, help="Test fraction for --mode random_split.")
    parser.add_argument("--held_out_chip", type=str, default=None,
                        help="Restrict --mode lofo to exactly this one held-out chip/folder.")
    args = parser.parse_args(argv)

    set_global_determinism(0, strict=True)

    group_names = list(config.CROSS_DATASET_GROUPS.keys())
    if args.task_id >= len(group_names):
        print(f"Task ID {args.task_id} is out of bounds for {len(group_names)} groups. Exiting.")
        sys.exit(0)

    group_name = group_names[args.task_id]
    folder_names = config.CROSS_DATASET_GROUPS[group_name]
    exp_paths = [Path(args.exp_folder, name) for name in folder_names]

    pc_ttp_cache_dir = Path(args.exp_folder) / "cross_dataset_cv" / group_name / "_cache_pc_ttp"
    pc_ttp_cache_dir.mkdir(parents=True, exist_ok=True)

    pc_variants = [m for m in args.models if m.endswith(PC_RECENTER_SUFFIX)]
    base_requested = [m for m in args.models if not m.endswith(PC_RECENTER_SUFFIX)]
    models_set = {_ENGINE_KEY[m] for m in base_requested}
    for v in pc_variants:
        models_set.add(_ENGINE_KEY[v[:-len(PC_RECENTER_SUFFIX)]])
    models = sorted(models_set)

    ordered = list(reversed(args.curve_type))
    for curve_type in ordered:
        print(f"\n\n{'#'*80}\nLOFO CROSS-DATASET CV FOR GROUP: {group_name} (curve_type: {curve_type})\nFolders: {folder_names}\n{'#'*80}")

        out_dir = Path(args.exp_folder) / "cross_dataset_cv" / group_name / "curve_alignment_pc_ttp" / f"anchor_{PC_TTP_ANCHOR}"
        plot_dir = out_dir / f"model_performance_{curve_type}"
        plot_dir.mkdir(parents=True, exist_ok=True)

        mode_str = args.mode if args.mode != "kfold" else f"kfold{args.n_splits}"
        outlier_filters = [None if f.lower() == "none" else f for f in args.outlier_filter]

        lofo_results = load_partitioned(out_dir, mode_str, curve_type)
        if args.force_rerun:
            print(f"  -> [FORCE RERUN] Clearing cached results for models: {models + pc_variants}")
            _clear_cached_results(lofo_results, models, pc_variants)

        pool_held_out_chips = list(folder_names) if args.mode == "lofo" else [None]
        if args.mode == "lofo" and args.held_out_chip:
            pool_held_out_chips = [c for c in pool_held_out_chips if c == args.held_out_chip]

        combined = None
        for held_out_chip in pool_held_out_chips:
            combined = combine_group(exp_paths, group_name, curve_type, held_out_chip, pc_ttp_cache_dir)
            if combined is None:
                continue

            if NOAMP_FILTER_NAME in outlier_filters:
                keep_mask = is_amplifying_mask(combined["curves"])
                combined["features_df"][NOAMP_FILTER_NAME] = keep_mask.astype(int)
                print(f"  [*] {NOAMP_FILTER_NAME}: {int((~keep_mask).sum())}/{len(keep_mask)} "
                      f"non-amplifying curves flagged for removal")

            _save_alignment_artifacts(combined, out_dir, curve_type, held_out_chip=held_out_chip)

            encoder, y_full, chip_id_encoded = _derive_pool_labels(combined)
            total_count = len(y_full)

            if args.mode == "lofo":
                cv_splits = build_lofo_splits(combined["dataset_id"])
                if held_out_chip is not None:
                    cv_splits = {k: v for k, v in cv_splits.items() if k == f"lofo_{held_out_chip}"}
                elif args.held_out_chip:
                    cv_splits = {k: v for k, v in cv_splits.items() if k == f"lofo_{args.held_out_chip}"}
            elif args.mode == "random_split":
                cv_splits = build_random_split(y_full, well_ids=combined["well_ids"], test_size=args.test_size)
            else:
                cv_splits = build_nfold_splits(y_full, well_ids=combined["well_ids"], n_splits=args.n_splits)

            total_folds = len(cv_splits)
            for fold_idx, (fold_label, (train_idx, test_idx)) in enumerate(reversed(list(cv_splits.items()))):
                _process_fold(fold_idx, total_folds, fold_label, train_idx, test_idx,
                              combined, y_full, encoder, curve_type, models, pc_variants,
                              outlier_filters, out_dir, plot_dir, group_name, total_count,
                              lofo_results, mode_str, args, chip_id_encoded, exp_paths)

        if args.train_full:
            combined = combine_group(exp_paths, group_name, curve_type, None, pc_ttp_cache_dir)
            if combined is not None:
                _save_alignment_artifacts(combined, out_dir, curve_type)
                if NOAMP_FILTER_NAME in outlier_filters:
                    keep_mask = is_amplifying_mask(combined["curves"])
                    combined["features_df"][NOAMP_FILTER_NAME] = keep_mask.astype(int)

                encoder, y_full, chip_id_encoded = _derive_pool_labels(combined)
                all_idx = np.arange(len(y_full))
                full_model_dir = out_dir / "model_interpretation" / f"full_data_{mode_str}"
                full_model_dir.mkdir(parents=True, exist_ok=True)
                print(f"\n{'='*75}\n[FULL DATA] Training on all {len(y_full)} samples (no holdout) | curve={curve_type}\n{'='*75}")

                if LOFO_AE_FILTER_NAME in outlier_filters:
                    keep_mask = _fit_lofo_ae_filter(combined, all_idx, out_dir, curve_type, "full_data",
                                                     force_rerun=args.force_rerun)
                    combined["features_df"][LOFO_AE_FILTER_NAME] = keep_mask.astype(int)

                res_full = evaluate_outlier_filters(
                    X_curves=combined["curves"],
                    features_df=combined["features_df"],
                    y_encoded=y_full,
                    outlier_filters=outlier_filters,
                    dataset_name=group_name,
                    mode_name="full_data",
                    cached_results=lofo_results.get("full_data", {}),
                    models=models,
                    checkpoint_fn=None,
                    cv_splits=[(all_idx, all_idx)],
                    save_model_dir=full_model_dir,
                    save_model_curve_type=curve_type,
                    coords=combined["coords"],
                    well_ids=combined["well_ids"],
                    k_neighbors=K_NEIGHBORS,
                    chip_id_encoded=chip_id_encoded,
                    batch_size=args.batch_size,
                )
                lofo_results["full_data"] = res_full
                lofo_results["full_data"]["class_names"] = [str(c) for c in encoder.classes_]
                save_partitioned(lofo_results, out_dir, mode_str, curve_type, compress=3, models=models)
                gc.collect()


if __name__ == "__main__":
    main()
