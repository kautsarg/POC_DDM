import os
import sys
import gc
import argparse
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from kneed import KneeLocator
sys.path.insert(0, 'utils')
from safe_io import safe_joblib_dump, safe_keras_save
from cross_dataset_result_io import save_partitioned, load_partitioned
import sigmoid_fitting as sp
sys.path.insert(0, 'utils/model_training')
from model_utils import (evaluate_outlier_filters, plot_ml_results, set_global_determinism,
                          CurveResampler, build_well_stratified_random_split, build_well_stratified_nfold_splits)
sys.path.insert(0, 'utils/02_outlier_detection')
from lstm_autoencoder_outlier import build_lstm_autoencoder
from model_utils_mtl import REG_SENTINEL as _MTL_REG_SENTINEL
from model_utils_supcon import (SUPCON_MODEL_KEYS, SUPCON_MTL_MODEL_KEYS,
                                BRANCH_SUPCON2_MODEL_KEYS, BRANCH_SUPCON2_MTL_MODEL_KEYS,
                                BRANCH_SUPCON3_MODEL_KEYS, BRANCH_SUPCON3_MTL_MODEL_KEYS,
                                CL_SUPCON_MTL_MODEL_KEYS,
                                CL_BRANCH_SUPCON2_MTL_MODEL_KEYS,
                                CL_BRANCH_SUPCON3_MTL_MODEL_KEYS,
                                STAGED_SUPCON_MODEL_KEYS, STAGED_BRANCH_SUPCON2_MODEL_KEYS,
                                STAGED_BRANCH_SUPCON3_MODEL_KEYS, ALL_STAGED_SUPCON_KEYS)
from model_utils_mtl import CL_MTL_MODEL_KEYS
from model_utils_rcfd import (RCFD_MODEL_KEYS, RCFD_SUPCON_MTL_MODEL_KEYS,
                               RCFD_BRANCH2_MTL_MODEL_KEYS, RCFD_BRANCH3_MTL_MODEL_KEYS)

import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.keras.mixed_precision.set_global_policy('mixed_float16')  # 2-3x speedup on A100 Tensor Cores
tf.config.optimizer.set_jit(True)                            # XLA JIT compilation

PC_TTP_ANCHOR_PCT_DEFAULT = 10


# ============================================================
# HELPERS
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

    return {
        "curves": data["dataset"][idx],
        "features_df": data["kinetic_features"][idx].loc[:, ~data["kinetic_features"][idx].columns.duplicated()].reset_index(drop=True),
        "Y_well_raw": Y_well_raw,
        "Y_mapped": Y_mapped,
        "timestamps": np.asarray(data["timestamps"], dtype=float),
        "dataset_id": exp_path.name,
        "coords": coords,
        "well_ids": well_ids,
        "concentration_raw": data.get("concentration", None),
    }


def combine_group(exp_paths, group_name, curve_type="ori_curve"):
    """Concatenate curve data for all folders in a group, validating they share one label mapping.

    Datasets may have different timestamp grids, so curves are first resampled
    onto a single common time grid (fitted across the whole group) via
    `CurveResampler` before concatenation.
    """
    parts = []
    ref_mapping = None
    for exp_path in exp_paths:
        d = load_curve_data(exp_path, curve_type, group_name=group_name)
        if d is None:
            continue
        if ref_mapping is None:
            ref_mapping = d["well_to_label"] if "well_to_label" in d else None
        parts.append(d)

    if len(parts) < 2:
        print(f"  -> Skipping group '{group_name}': fewer than 2 usable datasets found.")
        return None

    timestamps_zeroed = [p["timestamps"] - p["timestamps"][0] for p in parts]
    resampler = CurveResampler.fit(timestamps_zeroed)
    print(f"  [*] Resampling curves onto common grid: {len(resampler.t_grid)} points, "
          f"duration={resampler.t_grid[-1]:.4g}")
    for p in parts:
        p["curves"] = resampler.transform(p["timestamps"], p["curves"])

    # coords/well_ids: only meaningful if EVERY part in the group has them -- a partial
    # set would misalign with evaluate_outlier_filters' per-row mask/valid_class_mask
    # logic, so cnn_gru_dual_cosine_recon/cnn_gru_dual_attn_recon are skipped for the
    # whole group (not just the experiments missing metadata) when this happens.
    if all(p["coords"] is not None for p in parts):
        coords_combined = np.concatenate([p["coords"] for p in parts], axis=0)
        well_ids_combined = np.concatenate([p["well_ids"] for p in parts], axis=0)
    else:
        missing = [p["dataset_id"] for p in parts if p["coords"] is None]
        if missing:
            print(f"  [*] No pixel_row_idx/pixel_col_idx metadata for: {missing} -- "
                  f"cnn_gru_dual_cosine_recon/cnn_gru_dual_attn_recon will be skipped for "
                  f"group '{group_name}' (all-or-nothing across the group's folders).")
        coords_combined, well_ids_combined = None, None

    conc_parts = []
    for p in parts:
        raw = p.get("concentration_raw")
        conc_parts.append(np.asarray(raw, dtype=object) if raw is not None
                          else np.full(len(p["Y_mapped"]), None, dtype=object))

    return {
        "curves": np.concatenate([p["curves"] for p in parts], axis=0),
        "features_df": pd.concat([p["features_df"] for p in parts], axis=0, ignore_index=True),
        "Y_mapped": np.concatenate([p["Y_mapped"] for p in parts], axis=0),
        "dataset_id": np.concatenate([np.full(len(p["Y_mapped"]), p["dataset_id"], dtype=object) for p in parts], axis=0),
        "dataset_names": [p["dataset_id"] for p in parts],
        "resampler": resampler,
        "coords": coords_combined,
        "well_ids": well_ids_combined,
        "concentration_raw": np.concatenate(conc_parts, axis=0),
    }


# PC-TTP-anchored curve alignment (--curve_alignment pc_ttp)
def load_pc_wells_snapshot(exp_path, curve_type):
    """Reads 01's --drop_pc PC snapshot from curve_for_training.joblib's 'pc_wells' key."""
    data_path = os.path.join(exp_path, config.TRAINING_DATA_PATH)
    if not os.path.exists(data_path):
        return None
    data = joblib.load(data_path)
    pc = data.get("pc_wells")
    if not pc:
        print(f"  [!] {exp_path.name}: no 'pc_wells' snapshot (--drop_pc not used for this experiment?).")
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
    """Per-curve min-max to [0,1], copied from 01_curve_preprocessing_v6.py."""
    curves = np.asarray(curves, dtype=np.float64)
    row_min = curves.min(axis=1, keepdims=True)
    row_max = curves.max(axis=1, keepdims=True)
    denom = np.where(row_max - row_min == 0, 1, row_max - row_min)
    return (curves - row_min) / denom


def compute_ct_for_curves(curves, timestamps):
    """Ct fit on the mean curve across the batch (PC has no precomputed Ct) --
    one fit on the averaged curve instead of one fit per pixel then averaged, since
    PC is a designed reference well (uniform by construction, not a biological
    replicate set)."""
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
    """Front-truncate so chip_ttp aligns to anchor; shared with 08's inference path."""
    shift = max(chip_ttp - anchor, 0.0) if chip_ttp is not None else 0.0
    start_idx = min(int(np.searchsorted(timestamps, timestamps[0] + shift)), len(timestamps) - 1)
    return timestamps[start_idx:], curves[:, start_idx:], shift


def truncate_to_common_duration(timestamps, curves, common_duration):
    """Back-truncate to common_duration; shared with 08's inference path."""
    end_time = timestamps[0] + common_duration
    end_idx = min(int(np.searchsorted(timestamps, end_time)) + 1, len(timestamps))
    return timestamps[:end_idx], curves[:, :end_idx]


def align_parts_to_pc_ttp(parts, pc_ttp, held_out_chip, anchor_method, anchor_pct, verbose=True):
    """Returns (aligned_parts, anchor, common_duration)."""
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


def combine_group_pc_aligned(exp_paths, group_name, curve_type, held_out_chip,
                             anchor_method, anchor_pct, pc_ttp_cache_dir):
    """Like combine_group(), but zero-references each chip at its PC well's TTP."""
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
                  f"cnn_gru_dual_cosine_recon/cnn_gru_dual_attn_recon will be skipped for "
                  f"group '{group_name}' (all-or-nothing across the group's folders).")
        coords_combined, well_ids_combined = None, None

    conc_parts = []
    for p in parts:
        raw = p.get("concentration_raw")
        conc_parts.append(np.asarray(raw, dtype=object) if raw is not None
                          else np.full(len(p["Y_mapped"]), None, dtype=object))

    combined_curves = np.concatenate([p["curves"] for p in parts], axis=0)
    if curve_type.endswith("_norm"):
        # Re-normalize the truncated window, not the pre-truncation curve.
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
        "concentration_raw": np.concatenate(conc_parts, axis=0),
        "pc_ttp_recipe": {
            "anchor": anchor, "anchor_method": anchor_method,
            "common_duration": common_duration, "pc_ttp_per_chip": pc_ttp,
            "held_out_chip": held_out_chip, "curve_type": curve_type,
        },
    }


def build_lofo_splits(dataset_id):
    """Leave-one-folder-out: each fold holds out one whole dataset as the test set."""
    splits = {}
    for name in np.unique(dataset_id):
        test_idx = np.where(dataset_id == name)[0]
        train_idx = np.where(dataset_id != name)[0]
        splits[f"lofo_{name}"] = (train_idx, test_idx)
    return splits


def build_random_split(y, well_ids=None, test_size=0.1, random_state=0):
    """Stratified single random train/test split. Stratified by well_id when
    well_ids is available (one well = one label, so this also preserves label
    balance) instead of by label alone -- leak safety for cosine_recon/attn_recon
    comes from restricting model_utils.build_neighbor_curve_stack's input to one
    split side at a time, not from this split. Falls back to plain label-stratified
    otherwise (safe in that case since those models require well_ids and get
    skipped without it)."""
    if well_ids is not None:
        return build_well_stratified_random_split(y, well_ids, test_size=test_size, random_state=random_state)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(np.zeros(len(y)), y))
    return {"random_split": (train_idx, test_idx)}


def build_nfold_splits(y, well_ids=None, n_splits=5, random_state=0):
    """Stratified N-fold cross-validation splits — see build_random_split
    for the well-stratification rationale."""
    if well_ids is not None:
        return build_well_stratified_nfold_splits(y, well_ids, n_splits=n_splits, random_state=random_state)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return {f"fold_{i}": (tr, te) for i, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y))}


def _select_top_10_features(X_candidates_clean, y_full, idx, tag):
    mi_scores = mutual_info_classif(X_candidates_clean[idx], y_full[idx], random_state=0)
    top_10_idx = np.argsort(mi_scores)[-10:][::-1]
    feats = [config.LD_FEATURES[i] for i in top_10_idx]
    print(f"  [*] Selected Top 10 Features ({tag}): {feats}")
    return feats


def _save_alignment_artifacts(combined, out_dir, curve_type, args):
    """Saves the resampler, plus the pc_ttp recipe when curve_alignment is pc_ttp."""
    resampler_path = out_dir / config.CROSS_DATASET_RESAMPLER_PATH.format(curve_type=curve_type)
    safe_joblib_dump(combined["resampler"], resampler_path, compress=3)
    print(f"  [*] Saved curve resampler -> {resampler_path}")
    if args.curve_alignment == "pc_ttp":
        recipe_path = out_dir / config.CROSS_DATASET_PC_TTP_RECIPE_PATH.format(curve_type=curve_type)
        safe_joblib_dump(combined["pc_ttp_recipe"], recipe_path, compress=3)
        print(f"  [*] Saved pc_ttp alignment recipe -> {recipe_path}")


LOFO_AE_FILTER_NAME = "lofo_ae"


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


def _derive_pool_labels(combined, args):
    encoder = LabelEncoder()
    y_full = encoder.fit_transform(combined["Y_mapped"])
    chip_id_encoded = LabelEncoder().fit_transform(combined["dataset_id"])

    X_candidates = combined["features_df"][config.LD_FEATURES].values
    X_candidates_clean = np.nan_to_num(X_candidates, nan=0.0, posinf=0.0, neginf=0.0)

    y_concentration = None
    if args.mtl:
        raw_conc = combined.get("concentration_raw")
        if raw_conc is not None:
            _float_arr = np.array([float(v) if v is not None else np.nan for v in raw_conc], dtype=float)
            y_concentration = np.where(np.isnan(_float_arr) | (_float_arr == 0.0),
                                        _MTL_REG_SENTINEL, _float_arr)
        else:
            y_concentration = np.full(len(y_full), _MTL_REG_SENTINEL, dtype=float)
        _n_valid = int((y_concentration != _MTL_REG_SENTINEL).sum())
        print(f"  [MTL] Concentration loaded: {_n_valid} / {len(y_concentration)} samples "
              f"have non-sentinel concentration.")

    return encoder, y_full, X_candidates_clean, y_concentration, chip_id_encoded


def _process_fold(fold_idx, total_folds, fold_label, train_idx, test_idx,
                  combined, y_full, encoder, X_candidates_clean, curve_type, models,
                  outlier_filters, out_dir, plot_dir, group_name, total_count,
                  lofo_results, mode_str, args, y_concentration, chip_id_encoded):
    progress_pct = ((fold_idx + 1) / total_folds) * 100
    print(f"\n{'='*75}")
    print(f"[{fold_idx+1}/{total_folds} | {progress_pct:.1f}%] FOLD: {fold_label} | train={len(train_idx)} test={len(test_idx)}")
    print(f"{'='*75}")

    top_10_features = _select_top_10_features(X_candidates_clean, y_full, train_idx, f"{fold_label}, train-only")

    cached_fold = lofo_results.get(fold_label, {})

    def checkpoint(updated_results, fold_label=fold_label):
        lofo_results[fold_label] = updated_results
        save_partitioned(lofo_results, out_dir, mode_str, curve_type, compress=3)

    lofo_model_dir = out_dir / "model_interpretation" / fold_label
    lofo_model_dir.mkdir(parents=True, exist_ok=True)

    if LOFO_AE_FILTER_NAME in outlier_filters:
        keep_mask = _fit_lofo_ae_filter(combined, train_idx, out_dir, curve_type, fold_label,
                                         force_rerun=getattr(args, 'force_rerun', False))
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
        KFS=top_10_features,
        rerun_models=config.RERUN_MODELS,
        cv_splits=[(train_idx, test_idx)],
        save_model_dir=lofo_model_dir,
        save_model_curve_type=curve_type,
        coords=combined["coords"],
        well_ids=combined["well_ids"],
        k_neighbors=args.k_neighbors,
        multitask=args.mtl,
        y_concentration=y_concentration,
        chip_id_encoded=chip_id_encoded,
        cl_phase1_epochs=getattr(args, 'cl_phase1_epochs', None),
        batch_size=2048,
    )

    lofo_results[fold_label] = res

    # XAI metadata
    if "top_10_features" not in lofo_results[fold_label]:
        lofo_results[fold_label]["top_10_features"] = {}
    for f in outlier_filters:
        lofo_results[fold_label]["top_10_features"][str(f)] = top_10_features

    lofo_results[fold_label]["class_names"] = [str(c) for c in encoder.classes_]

    save_partitioned(lofo_results, out_dir, mode_str, curve_type, compress=3)

    features_df_all = combined["features_df"]
    X_man_train = np.nan_to_num(
        features_df_all.iloc[train_idx][top_10_features].values,
        nan=0.0, posinf=0.0, neginf=0.0,
    ).astype(np.float32)
    X_man_test = np.nan_to_num(
        features_df_all.iloc[test_idx][top_10_features].values,
        nan=0.0, posinf=0.0, neginf=0.0,
    ).astype(np.float32)
    snapshot_path = lofo_model_dir / f"xai_data_{curve_type}.joblib"
    safe_joblib_dump({
        "X_curves_test": combined["curves"][test_idx].astype(np.float32),
        "features_df_test": features_df_all.iloc[test_idx].reset_index(drop=True),
        "X_man_train": X_man_train,
        "X_man_test": X_man_test,
        "y_test": y_full[test_idx],
        "timestamps": combined["resampler"].t_grid,
        "top_10_features": top_10_features,
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


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print(f"\n{'='*70}\n[RUNNING] {os.path.basename(__file__)}\n{'='*70}\n")
    parser = argparse.ArgumentParser(description="Cross-Dataset Leave-One-Folder-Out (LOFO) Training")
    parser.add_argument("--task_id", type=int, default=0, help="Array Job ID -> index into CROSS_DATASET_GROUPS")
    parser.add_argument("--exp_folder", type=str, default=config.DEFAULT_EXP_FOLDER)
    parser.add_argument("--force_rerun", action="store_true", help="Recompute and overwrite even if presaved results already exist")
    parser.add_argument("--curve_type", type=str, nargs='+',
                        default=['ori_curve', 'ori_curve_norm',
                                 'ori_curve_avg', 'ori_curve_avg_norm',
                                 'ori_curve_wavelet_sym8', 'ori_curve_wavelet_sym8_norm',
                                 'ori_curve_wavelet_bior35', 'ori_curve_wavelet_bior35_norm',
                                 'ori_curve_sg_p4', 'ori_curve_sg_p4_norm'],
                        help="Which curve dataset(s) to train on. Accepts one or more values (e.g. 'ori_curve' 'ori_curve_norm').")
    parser.add_argument("--fast_mode", action="store_true",
                        help="Disable strict TF determinism (TF_CUDNN_DETERMINISTIC/enable_op_determinism) "
                             "for faster GRU/LSTM/Transformer training. RNG seeds are still set, but reruns "
                             "won't be bit-exact. Only affects this script.")
    parser.add_argument("--k_neighbors", type=int, default=24,
                        help="Neighbours per pixel (within the same well) for "
                             "cnn_gru_dual_cosine_recon/cnn_gru_dual_attn_recon's spatial reconstruction.")
    parser.add_argument("--mtl", action="store_true",
                        help="Train MTL models only (classification + concentration regression heads). "
                             "Results are merged into the same results joblib so standard models "
                             "do not need to be re-run.")
    parser.add_argument("--supcon", type=int, choices=[0, 1, 2, 3], default=0,
                        help="0=none, 1=original fused SupCon, 2=branch v2 (CNN+seq), "
                             "3=branch v3 (CNN+seq+fused). Results merged into same joblib.")
    parser.add_argument("--mtl_cl", action="store_true",
                        help="Use phase-decoupled curriculum learning for MTL.")
    parser.add_argument("--cl_phase1_epochs", type=int, default=None,
                        help="Fixed Phase 1 epochs for CL-MTL. If omitted, auto-detects convergence.")
    parser.add_argument("--condreg", action="store_true",
                        help="Train RCFD models (Regression-Conditioned Feature Dual). Implies --mtl.")
    parser.add_argument("--supcon_staged", action="store_true",
                        help="2-stage SupCon: Stage 1 SC-only until plateau, Stage 2 CE-only frozen backbone.")
    parser.add_argument("--dann", action="store_true",
                        help="Train domain-adversarial (chip-invariance) models instead. "
                             "Only --supcon 0 or 3 supported (plain or branch-v3 SupCon + DANN).")
    parser.add_argument("--coral", action="store_true",
                        help="Train Deep CORAL (chip-invariance via embedding covariance "
                             "alignment) models instead -- non-adversarial alternative to "
                             "--dann. Only --supcon 0 or 3 supported.")
    parser.add_argument("--outlier_filter", type=str, nargs='+', default=["none"],
                        help="Outlier filters to evaluate. Use 'none' for no filter. "
                             "'lofo_ae' fits an LSTM-AE on the current fold's train chips only "
                             "and scores the whole pool with it (fold-aware, unlike the "
                             "precomputed per-chip lstm_ae_* filters). "
                             "E.g. --outlier_filter none lstm_ae_glb_ds1_label_elbow")
    parser.add_argument("--models", type=str, nargs='+', default=None,
                        help="Filter model list by base name (e.g. 'cnn_gru_dual' 'cnn_gru_dual_attn_recon'). "
                             "Strips _supconN/_lc/_mtl suffixes before matching.")
    parser.add_argument("--rerun_models", type=str, nargs='+', default=None,
                        help="Restrict --force_rerun's cache-clearing to result keys matching these "
                             "model keys (substring match, so a base name also matches its supcon-"
                             "suffixed variants), instead of clearing every standard-track model. "
                             "Independent of --models -- e.g. --models cnn_gru_dual cnn_gru_dual_attn_recon "
                             "--rerun_models cnn_gru_dual_attn_recon --force_rerun trains/evaluates both "
                             "models but only force-clears+retrains cnn_gru_dual_attn_recon; cnn_gru_dual "
                             "still trains only if missing. No effect without --force_rerun.")
    parser.add_argument("--train_full", action="store_true",
                        help="After LOFO, train one final model on ALL data (no holdout) and save to "
                             "model_interpretation/full_data/. Metrics in result are train-set accuracy "
                             "(inflated) — use LOFO metrics for evaluation.")
    parser.add_argument("--mode", type=str, choices=["lofo", "random_split", "kfold"],
                        default="lofo",
                        help="Split strategy: lofo=leave-one-folder-out, "
                             "random_split=stratified single split, kfold=stratified N-fold.")
    parser.add_argument("--n_splits", type=int, default=5,
                        help="Number of folds for --mode kfold.")
    parser.add_argument("--test_size", type=float, default=0.1,
                        help="Test fraction for --mode random_split.")
    parser.add_argument("--curve_alignment", type=str,
                        choices=config.CURVE_ALIGNMENT_CHOICES,
                        default="acquisition_start",
                        help="How to zero-reference each chip's curves before resampling "
                             "onto a common grid. 'acquisition_start' (default, today's "
                             "behaviour): zero-reference at raw acquisition start. "
                             "'pc_ttp': zero-reference at each chip's PC well's "
                             "time-to-positivity, aligned to a shared anchor -- see "
                             "_brainstorming/20260813-lofo_domain_shift_combined_implementation_plan.md Part I.")
    parser.add_argument("--pc_ttp_anchor", type=str, choices=["min", "percentile"],
                        default="min",
                        help="Only used when --curve_alignment pc_ttp. Anchor statistic "
                             "across training chips' PC TTP -- 'min' (strict earliest) or "
                             "'percentile' (fixed 10th percentile, more robust to a single "
                             "outlier chip defining the anchor for everyone).")
    parser.add_argument("--lofo_limit", type=int, default=None,
                        help="Only run the first N LOFO folds instead of all of them "
                             "(e.g. 1 out of a 4-chip group) -- for quick iteration/testing. "
                             "Does not affect --train_full.")
    args = parser.parse_args()
    if args.mtl_cl:
        args.mtl = True  # --mtl_cl implies --mtl
    if args.condreg:
        args.mtl = True  # --condreg implies --mtl
    if getattr(args, 'supcon_staged', False) and args.mtl:
        sys.exit('[!] --supcon_staged is ST-only; incompatible with --mtl.')
    if getattr(args, 'supcon_staged', False) and args.supcon == 0:
        sys.exit('[!] --supcon_staged requires --supcon 1, 2, or 3.')
    if getattr(args, 'dann', False) and args.mtl:
        sys.exit('[!] --dann is ST-only; incompatible with --mtl.')
    if getattr(args, 'dann', False) and args.supcon not in (0, 3):
        sys.exit('[!] --dann only supports --supcon 0 or 3.')
    if getattr(args, 'coral', False) and args.mtl:
        sys.exit('[!] --coral is ST-only; incompatible with --mtl.')
    if getattr(args, 'coral', False) and args.supcon not in (0, 3):
        sys.exit('[!] --coral only supports --supcon 0 or 3.')
    if getattr(args, 'coral', False) and getattr(args, 'dann', False):
        sys.exit('[!] --coral and --dann are alternatives; pass at most one.')
    if args.rerun_models and not args.force_rerun:
        args.rerun_models = None  # --rerun_models has no effect without --force_rerun

    set_global_determinism(0, strict=not args.fast_mode)

    group_names = list(config.CROSS_DATASET_GROUPS.keys())
    if not group_names:
        print("CROSS_DATASET_GROUPS is empty in config.py. Define at least one group to run this script.")
        sys.exit(0)
    if args.task_id >= len(group_names):
        print(f"Task ID {args.task_id} is out of bounds for {len(group_names)} groups. Exiting.")
        sys.exit(0)

    group_name = group_names[args.task_id]
    folder_names = config.CROSS_DATASET_GROUPS[group_name]
    exp_paths = [Path(args.exp_folder, name) for name in folder_names]

    pc_ttp_cache_dir = None
    if args.curve_alignment == "pc_ttp":
        pc_ttp_cache_dir = Path(args.exp_folder) / "cross_dataset_cv" / group_name / "_cache_pc_ttp"
        pc_ttp_cache_dir.mkdir(parents=True, exist_ok=True)

    ordered = list(reversed(args.curve_type))
    for curve_type in ordered:
        print(f"\n\n{'#'*80}\nLOFO CROSS-DATASET CV FOR GROUP: {group_name} (curve_type: {curve_type})\nFolders: {folder_names}\n{'#'*80}")

        out_dir = Path(args.exp_folder) / "cross_dataset_cv" / group_name
        if args.curve_alignment == "pc_ttp":
            out_dir = out_dir / "curve_alignment_pc_ttp" / f"anchor_{args.pc_ttp_anchor}"
        plot_dir = out_dir / f"model_performance_{curve_type}"
        plot_dir.mkdir(parents=True, exist_ok=True)

        _mode_str = args.mode if args.mode != "kfold" else f"kfold{args.n_splits}"
        outlier_filters = [None if f.lower() == "none" else f for f in args.outlier_filter]

        # if getattr(args, 'supcon_staged', False):
        #     if args.supcon == 1:
        #         models = list(STAGED_SUPCON_MODEL_KEYS)
        #     elif args.supcon == 2:
        #         models = list(STAGED_BRANCH_SUPCON2_MODEL_KEYS)
        #     elif args.supcon == 3:
        #         models = list(STAGED_BRANCH_SUPCON3_MODEL_KEYS)
        # elif getattr(args, 'condreg', False):
        #     if args.supcon == 0:
        #         models = list(RCFD_MODEL_KEYS)
        #     elif args.supcon == 1:
        #         models = list(RCFD_SUPCON_MTL_MODEL_KEYS)
        #     elif args.supcon == 2:
        #         models = list(RCFD_BRANCH2_MTL_MODEL_KEYS)
        #     elif args.supcon == 3:
        #         models = list(RCFD_BRANCH3_MTL_MODEL_KEYS)
        # elif args.mtl and getattr(args, 'mtl_cl', False):
        #     if args.supcon == 0:
        #         models = list(CL_MTL_MODEL_KEYS)
        #     elif args.supcon == 1:
        #         models = list(CL_SUPCON_MTL_MODEL_KEYS)
        #     elif args.supcon == 2:
        #         models = list(CL_BRANCH_SUPCON2_MTL_MODEL_KEYS)
        #     elif args.supcon == 3:
        #         models = list(CL_BRANCH_SUPCON3_MTL_MODEL_KEYS)
        # elif args.mtl:
        #     if args.supcon == 1:
        #         models = list(SUPCON_MTL_MODEL_KEYS)
        #     elif args.supcon == 2:
        #         models = list(BRANCH_SUPCON2_MTL_MODEL_KEYS)
        #     elif args.supcon == 3:
        #         models = list(BRANCH_SUPCON3_MTL_MODEL_KEYS)
        #     else:
        #         models = {
        #             "cnn_mtl",
        #             "gru_mtl", "cnn_gru_dual_mtl",
        #             "transformer_mtl", "cnn_trans_dual_mtl",
        #             "cnn_gru_dual_cosine_recon_mtl", "cnn_gru_dual_attn_recon_mtl",
        #         }
        # else:
        #     if args.supcon == 1:
        #         models = list(SUPCON_MODEL_KEYS)
        #     elif args.supcon == 2:
        #         models = list(BRANCH_SUPCON2_MODEL_KEYS)
        #     elif args.supcon == 3:
        #         models = list(BRANCH_SUPCON3_MODEL_KEYS)
        #     else:
        #         models = [
        #             # "knn",
        #             "cnn", # "cnn_inc",
        #             # "cnn_lf",
        #             "gru", "cnn_gru_dual", # "cnn_gru_dual_inc",
        #             # "gru_lf",
        #             "transformer", "cnn_trans_dual", # "cnn_trans_dual_inc",
        #             # "trans_lf",
        #             "cnn_gru_dual_cosine_recon", "cnn_gru_dual_attn_recon",
        #             # "lstm_ae_clf",
        #             # "cnn_gru_gate", "cnn_gru_hadamard", "cnn_gru_crossattn", "cnn_gru_film",
        #             # "cnn_trans_gate", "cnn_trans_hadamard", "cnn_trans_crossattn", "cnn_trans_film",
        #         ]

        if getattr(args, 'supcon_staged', False):
            if args.supcon == 1:
                models = list(STAGED_SUPCON_MODEL_KEYS)
            elif args.supcon == 2:
                models = list(STAGED_BRANCH_SUPCON2_MODEL_KEYS)
            elif args.supcon == 3:
                models = list(STAGED_BRANCH_SUPCON3_MODEL_KEYS)

        # LABEL CONSOLIDATION (LC) MODELS
        elif getattr(args, 'lbl_conc', False):
            _sc_lc = {
                0: ['cnn_gru_dual_lc', 'cnn_gru_dual_cosine_recon_lc', 'cnn_gru_dual_attn_recon_lc'],
                1: ['cnn_gru_dual_supcon_lc', 'cnn_gru_dual_cosine_recon_supcon_lc', 'cnn_gru_dual_attn_recon_supcon_lc'],
                2: ['cnn_gru_dual_supcon2_lc', 'cnn_gru_dual_cosine_recon_supcon2_lc', 'cnn_gru_dual_attn_recon_supcon2_lc'],
                3: ['cnn_gru_dual_supcon3_lc', 'cnn_gru_dual_cosine_recon_supcon3_lc', 'cnn_gru_dual_attn_recon_supcon3_lc'],
            }
            models = _sc_lc[args.supcon]

        # CONDITIONAL REGRESSION (RCFD) MODELS
        elif getattr(args, 'condreg', False):
            if args.supcon == 0:
                models = ['cnn_rcfd_cgd', 'gru_rcfd_cgd', 'trans_rcfd_cgd']
            elif args.supcon == 1:
                models = ['cnn_rcfd_cgd_supcon_mtl', 'gru_rcfd_cgd_supcon_mtl', 'trans_rcfd_cgd_supcon_mtl']
            elif args.supcon == 2:
                models = ['cnn_rcfd_cgd_supcon2_mtl', 'gru_rcfd_cgd_supcon2_mtl', 'trans_rcfd_cgd_supcon2_mtl']
            elif args.supcon == 3:
                models = ['cnn_rcfd_cgd_supcon3_mtl', 'gru_rcfd_cgd_supcon3_mtl', 'trans_rcfd_cgd_supcon3_mtl']

        # DOMAIN-ADVERSARIAL (DANN) MODELS
        elif getattr(args, 'dann', False):
            models = (['cnn_gru_dual_supcon3_dann', 'cnn_gru_dual_attn_recon_supcon3_dann'] if args.supcon == 3
                     else ['cnn_gru_dual_dann', 'cnn_gru_dual_attn_recon_dann'])

        # DEEP CORAL MODELS
        elif getattr(args, 'coral', False):
            models = (['cnn_gru_dual_supcon3_coral', 'cnn_gru_dual_attn_recon_supcon3_coral'] if args.supcon == 3
                     else ['cnn_gru_dual_coral', 'cnn_gru_dual_attn_recon_coral'])

        # CURRICULUM LEARNING (CL) MTL MODELS
        elif args.mtl and getattr(args, 'mtl_cl', False):
            if args.supcon == 0:
                models = ['cnn_gru_dual_cl_mtl']
            elif args.supcon == 1:
                models = ['cnn_gru_dual_cl_supcon_mtl']
            elif args.supcon == 2:
                models = ['cnn_gru_dual_cl_supcon2_mtl']
            elif args.supcon == 3:
                models = ['cnn_gru_dual_cl_supcon3_mtl']

        # MULTI-TASK LEARNING (MTL) MODELS
        elif args.mtl:
            if args.supcon == 1:
                models = ['cnn_gru_dual_supcon_mtl', 'cnn_gru_dual_cosine_recon_supcon_mtl', 'cnn_gru_dual_attn_recon_supcon_mtl']
            elif args.supcon == 2:
                models = ['cnn_gru_dual_supcon2_mtl', 'cnn_gru_dual_cosine_recon_supcon2_mtl', 'cnn_gru_dual_attn_recon_supcon2_mtl']
            elif args.supcon == 3:
                models = ['cnn_gru_dual_supcon3_mtl', 'cnn_gru_dual_cosine_recon_supcon3_mtl', 'cnn_gru_dual_attn_recon_supcon3_mtl']
            else:
                models = ['cnn_gru_dual_mtl', 'cnn_gru_dual_cosine_recon_mtl', 'cnn_gru_dual_attn_recon_mtl']

        # SINGLE-TASK (ST) MODELS
        else:
            if args.supcon == 1:
                models = ['cnn_gru_dual_supcon', 'cnn_gru_dual_cosine_recon_supcon', 'cnn_gru_dual_attn_recon_supcon']
            elif args.supcon == 2:
                models = ['cnn_gru_dual_supcon2', 'cnn_gru_dual_cosine_recon_supcon2', 'cnn_gru_dual_attn_recon_supcon2']
            elif args.supcon == 3:
                models = ['cnn_gru_dual_supcon3', 'cnn_gru_dual_cosine_recon_supcon3', 'cnn_gru_dual_attn_recon_supcon3']
            else:
                models = ['knn', 'cnn_gru_dual', 'cnn_gru_dual_cosine_recon', 'cnn_gru_dual_attn_recon']

        if args.models:
            import re
            _req = set(args.models)
            _suffix_re = re.compile(r'_(supcon\d*|lc|staged|mtl)$')

            def _strip_variant_suffixes(name):
                # Repeatedly strip one suffix token at a time so compound suffixes
                # (e.g. "_supcon_lc", "_supcon3_staged") fully reduce to the base name,
                # regardless of order.
                while True:
                    stripped = _suffix_re.sub('', name)
                    if stripped == name:
                        return name
                    name = stripped

            models = [m for m in models
                      if m in _req or _strip_variant_suffixes(m) in _req]

        lofo_results = load_partitioned(out_dir, _mode_str, curve_type)
        if args.force_rerun:
            _mtl_result_keys = set()
            for _k, (_pk, _probk, _clsk) in config.MODEL_KEY_MAP.items():
                if _k in config._MTL_MODEL_KEYS:
                    _mtl_result_keys.update([_pk, _probk, _clsk,
                                             f'y_reg_preds_{_k}_', f'y_reg_trues_{_k}_'])
            _supcon_st_result_keys = set()
            _supcon_mtl_result_keys = set()
            for _k, (_pk, _probk, _clsk) in config.MODEL_KEY_MAP.items():
                if _k in config._SUPCON_MODEL_KEYS:
                    if 'mtl' in _k:
                        _supcon_mtl_result_keys.update([_pk, _probk, _clsk,
                                                        f'y_reg_preds_{_k}_', f'y_reg_trues_{_k}_'])
                    else:
                        _supcon_st_result_keys.update([_pk, _probk, _clsk])
            _bsc_st_keys  = (set(BRANCH_SUPCON2_MODEL_KEYS)     if args.supcon == 2
                             else set(BRANCH_SUPCON3_MODEL_KEYS) if args.supcon == 3 else set())
            _bsc_mtl_keys = (set(BRANCH_SUPCON2_MTL_MODEL_KEYS)     if args.supcon == 2
                             else set(BRANCH_SUPCON3_MTL_MODEL_KEYS) if args.supcon == 3 else set())
            _bsc_st_result_keys, _bsc_mtl_result_keys = set(), set()
            for _k, (_pk, _probk, _clsk) in config.MODEL_KEY_MAP.items():
                if _k in _bsc_st_keys:
                    _bsc_st_result_keys.update([_pk, _probk, _clsk])
                elif _k in _bsc_mtl_keys:
                    _bsc_mtl_result_keys.update([_pk, _probk, _clsk,
                                                 f'y_reg_preds_{_k}_', f'y_reg_trues_{_k}_'])
            _cl_base_result_keys  = set()
            _cl_supcon_result_keys = set()
            _cl_bsc2_result_keys  = set()
            _cl_bsc3_result_keys  = set()
            for _k, (_pk, _probk, _clsk) in config.MODEL_KEY_MAP.items():
                if _k in CL_MTL_MODEL_KEYS:
                    _cl_base_result_keys.update([_pk, _probk, _clsk, f'y_reg_preds_{_k}_', f'y_reg_trues_{_k}_'])
                elif _k in CL_SUPCON_MTL_MODEL_KEYS:
                    _cl_supcon_result_keys.update([_pk, _probk, _clsk, f'y_reg_preds_{_k}_', f'y_reg_trues_{_k}_'])
                elif _k in CL_BRANCH_SUPCON2_MTL_MODEL_KEYS:
                    _cl_bsc2_result_keys.update([_pk, _probk, _clsk, f'y_reg_preds_{_k}_', f'y_reg_trues_{_k}_'])
                elif _k in CL_BRANCH_SUPCON3_MTL_MODEL_KEYS:
                    _cl_bsc3_result_keys.update([_pk, _probk, _clsk, f'y_reg_preds_{_k}_', f'y_reg_trues_{_k}_'])
            _rcfd_result_keys = set()
            for _k, (_pk, _probk, _clsk) in config.MODEL_KEY_MAP.items():
                if _k in config._RCFD_MODEL_KEYS:
                    _rcfd_result_keys.update([_pk, _probk, _clsk,
                                              f'y_reg_preds_{_k}_', f'y_reg_trues_{_k}_'])
            _dann_result_keys = set()
            for _k, (_pk, _probk, _clsk) in config.MODEL_KEY_MAP.items():
                if _k in config._DANN_MODEL_KEYS:
                    _dann_result_keys.update([_pk, _probk, _clsk])
            _coral_result_keys = set()
            for _k, (_pk, _probk, _clsk) in config.MODEL_KEY_MAP.items():
                if _k in config._CORAL_MODEL_KEYS:
                    _coral_result_keys.update([_pk, _probk, _clsk])
            _staged_result_keys = set()
            _staged_sc_result_keys = set()
            for _k, (_pk, _probk, _clsk) in config.MODEL_KEY_MAP.items():
                if _k not in config._STAGED_SUPCON_MODEL_KEYS:
                    continue
                _staged_result_keys.update([_pk, _probk, _clsk])
                if 'supcon3_staged' in _k:   _k_staged_sc = 3
                elif 'supcon2_staged' in _k: _k_staged_sc = 2
                elif 'supcon_staged' in _k:  _k_staged_sc = 1
                else:                         _k_staged_sc = 0
                if _k_staged_sc == args.supcon:
                    _staged_sc_result_keys.update([_pk, _probk, _clsk])

            # Exact model-name membership (not substring match against result keys -- e.g.
            # "cnn_gru_dual" is a literal substring of "cnn_gru_dual_attn_recon"'s keys).
            _rerun_result_keys = set()
            if args.rerun_models:
                for _k, (_pk, _probk, _clsk) in config.MODEL_KEY_MAP.items():
                    if _k in args.rerun_models:
                        _rerun_result_keys.update([_pk, _probk, _clsk,
                                                   f'y_reg_preds_{_k}_', f'y_reg_trues_{_k}_'])

            _is_cl = getattr(args, 'mtl_cl', False)
            if getattr(args, 'supcon_staged', False):
                _which = f'Staged SupCon SC{args.supcon}'
            elif getattr(args, 'condreg', False):
                _which = f'RCFD SC{args.supcon}'
            elif getattr(args, 'dann', False):
                _which = f'DANN SC{args.supcon}'
            elif getattr(args, 'coral', False):
                _which = f'CORAL SC{args.supcon}'
            elif _is_cl and args.supcon == 0:
                _which = 'CL MTL'
            elif _is_cl and args.supcon == 1:
                _which = 'CL SupCon v1 MTL'
            elif _is_cl and args.supcon in (2, 3):
                _which = f'CL Branch SupCon v{args.supcon} MTL'
            elif args.supcon == 1 and args.mtl:
                _which = 'SupCon MTL'
            elif args.supcon == 1:
                _which = 'SupCon ST'
            elif args.supcon in (2, 3) and args.mtl:
                _which = f'Branch SupCon v{args.supcon} MTL'
            elif args.supcon in (2, 3):
                _which = f'Branch SupCon v{args.supcon} ST'
            elif args.mtl:
                _which = 'MTL'
            else:
                _which = 'standard'
            print(f"  -> [FORCE RERUN] Clearing {_which} cached results; preserving the rest.")
            for _fold_res in lofo_results.values():
                if not isinstance(_fold_res, dict):
                    continue
                for _filter_res in _fold_res.values():
                    if not isinstance(_filter_res, dict):
                        continue
                    for _rk in list(_filter_res.keys()):
                        if not isinstance(_rk, str):
                            continue
                        _is_mtl        = _rk in _mtl_result_keys
                        _is_supcon_st  = _rk in _supcon_st_result_keys
                        _is_supcon_mtl = _rk in _supcon_mtl_result_keys
                        _is_bsc_st     = _rk in _bsc_st_result_keys
                        _is_bsc_mtl    = _rk in _bsc_mtl_result_keys
                        _is_cl_base    = _rk in _cl_base_result_keys
                        _is_cl_supcon  = _rk in _cl_supcon_result_keys
                        _is_cl_bsc2    = _rk in _cl_bsc2_result_keys
                        _is_cl_bsc3    = _rk in _cl_bsc3_result_keys
                        _is_any_cl     = _is_cl_base or _is_cl_supcon or _is_cl_bsc2 or _is_cl_bsc3
                        _is_rcfd       = _rk in _rcfd_result_keys
                        _is_staged     = _rk in _staged_result_keys
                        _is_dann       = _rk in _dann_result_keys
                        _is_coral      = _rk in _coral_result_keys
                        _is_model_key  = any(_rk.startswith(p) for p in
                                             ('y_preds_AC_', 'y_probs_AC_', 'classes_AC_',
                                              'y_reg_preds_', 'y_reg_trues_'))
                        _mm = not args.rerun_models or _rk in _rerun_result_keys
                        _is_standard   = (_is_model_key and not _is_mtl and not _is_supcon_st
                                          and not _is_supcon_mtl and not _is_bsc_st and not _is_bsc_mtl
                                          and not _is_any_cl and not _is_rcfd and not _is_staged
                                          and not _is_dann and not _is_coral and _mm)
                        if _is_cl and args.supcon == 0 and _is_cl_base and _mm:
                            del _filter_res[_rk]
                        elif _is_cl and args.supcon == 1 and _is_cl_supcon and _mm:
                            del _filter_res[_rk]
                        elif _is_cl and args.supcon == 2 and _is_cl_bsc2 and _mm:
                            del _filter_res[_rk]
                        elif _is_cl and args.supcon == 3 and _is_cl_bsc3 and _mm:
                            del _filter_res[_rk]
                        elif args.supcon == 1 and args.mtl and _is_supcon_mtl and _mm:
                            del _filter_res[_rk]
                        elif args.supcon == 1 and not args.mtl and not getattr(args, 'supcon_staged', False) and _is_supcon_st and _mm:
                            del _filter_res[_rk]
                        elif args.supcon in (2, 3) and args.mtl and _is_bsc_mtl and _mm:
                            del _filter_res[_rk]
                        elif args.supcon in (2, 3) and not args.mtl and not getattr(args, 'supcon_staged', False) and _is_bsc_st and _mm:
                            del _filter_res[_rk]
                        elif args.mtl and args.supcon == 0 and _is_mtl and _mm:
                            del _filter_res[_rk]
                        elif args.supcon == 0 and not args.mtl and _is_standard:
                            del _filter_res[_rk]
                        elif getattr(args, 'condreg', False) and _is_rcfd and _mm:
                            del _filter_res[_rk]
                        elif getattr(args, 'dann', False) and _is_dann and _mm:
                            del _filter_res[_rk]
                        elif getattr(args, 'coral', False) and _is_coral and _mm:
                            del _filter_res[_rk]
                        elif getattr(args, 'supcon_staged', False) and _rk in _staged_sc_result_keys and _mm:
                            del _filter_res[_rk]

        def _build_pool(held_out_chip):
            if args.curve_alignment == "pc_ttp":
                return combine_group_pc_aligned(
                    exp_paths, group_name, curve_type=curve_type,
                    held_out_chip=held_out_chip, anchor_method=args.pc_ttp_anchor,
                    anchor_pct=PC_TTP_ANCHOR_PCT_DEFAULT, pc_ttp_cache_dir=pc_ttp_cache_dir)
            return combine_group(exp_paths, group_name, curve_type=curve_type)

        _lofo_pc_ttp = args.mode == "lofo" and args.curve_alignment == "pc_ttp"
        pool_held_out_chips = list(folder_names) if _lofo_pc_ttp else [None]
        if _lofo_pc_ttp and args.lofo_limit:
            pool_held_out_chips = pool_held_out_chips[:args.lofo_limit]

        combined = None
        for held_out_chip in pool_held_out_chips:
            combined = _build_pool(held_out_chip)
            if combined is None:
                continue

            if not _lofo_pc_ttp:
                _save_alignment_artifacts(combined, out_dir, curve_type, args)

            encoder, y_full, X_candidates_clean, y_concentration, chip_id_encoded = _derive_pool_labels(combined, args)
            total_count = len(y_full)

            if args.mode == "lofo":
                cv_splits = build_lofo_splits(combined["dataset_id"])
                if held_out_chip is not None:
                    cv_splits = {k: v for k, v in cv_splits.items() if k == f"lofo_{held_out_chip}"}
                elif args.lofo_limit:
                    cv_splits = dict(list(cv_splits.items())[:args.lofo_limit])
            elif args.mode == "random_split":
                cv_splits = build_random_split(y_full, well_ids=combined["well_ids"], test_size=args.test_size)
            else:
                cv_splits = build_nfold_splits(y_full, well_ids=combined["well_ids"], n_splits=args.n_splits)
            total_folds = len(cv_splits)
            for fold_idx, (fold_label, (train_idx, test_idx)) in enumerate(reversed(list(cv_splits.items()))):
                _process_fold(fold_idx, total_folds, fold_label, train_idx, test_idx,
                              combined, y_full, encoder, X_candidates_clean, curve_type, models,
                              outlier_filters, out_dir, plot_dir, group_name, total_count,
                              lofo_results, _mode_str, args, y_concentration, chip_id_encoded)

        if getattr(args, 'train_full', False):
            if _lofo_pc_ttp:
                combined = _build_pool(None)  # fresh group-wide pool
                if combined is not None:
                    _save_alignment_artifacts(combined, out_dir, curve_type, args)
            if combined is not None:
                encoder, y_full, X_candidates_clean, y_concentration, chip_id_encoded = _derive_pool_labels(combined, args)
                all_idx = np.arange(len(y_full))
                top_10_features = _select_top_10_features(X_candidates_clean, y_full, all_idx, "full_data")
                full_model_dir = out_dir / "model_interpretation" / f"full_data_{_mode_str}"
                full_model_dir.mkdir(parents=True, exist_ok=True)
                print(f"\n{'='*75}")
                print(f"[FULL DATA] Training on all {len(y_full)} samples (no holdout) | curve={curve_type}")
                print(f"{'='*75}")
                if LOFO_AE_FILTER_NAME in outlier_filters:
                    keep_mask = _fit_lofo_ae_filter(combined, all_idx, out_dir, curve_type, "full_data",
                                                     force_rerun=getattr(args, 'force_rerun', False))
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
                    KFS=top_10_features,
                    rerun_models=config.RERUN_MODELS,
                    cv_splits=[(all_idx, all_idx)],
                    save_model_dir=full_model_dir,
                    save_model_curve_type=curve_type,
                    coords=combined["coords"],
                    well_ids=combined["well_ids"],
                    k_neighbors=args.k_neighbors,
                    multitask=args.mtl,
                    y_concentration=y_concentration,
                    chip_id_encoded=chip_id_encoded,
                    cl_phase1_epochs=getattr(args, 'cl_phase1_epochs', None),
                    batch_size=2048,
                )
                lofo_results["full_data"] = res_full
                lofo_results["full_data"]["class_names"] = [str(c) for c in encoder.classes_]
                save_partitioned(lofo_results, out_dir, _mode_str, curve_type, compress=3)
                gc.collect()
