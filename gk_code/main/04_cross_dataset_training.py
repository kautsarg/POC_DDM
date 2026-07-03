import os
import sys
import gc
import argparse
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
sys.path.insert(0, 'utils/model_training')
from model_utils import evaluate_outlier_filters, plot_ml_results, set_global_determinism, CurveResampler

import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# set_global_determinism() is called inside __main__ after argparse, so --fast_mode
# can control strictness (see LOFO speed-up plan Change 1). Other scripts are unaffected.


# ============================================================
# HELPERS
# ============================================================
def load_curve_data(exp_path, curve_type):
    """Load the requested curve dataset, kinetic_features, and label-mapped Y_well for one experiment folder."""
    data_path = os.path.join(exp_path, config.TRAINING_DATA_PATH)
    if not os.path.exists(data_path):
        print(f"  -> Skipping {exp_path.name}: '{data_path}' not found.")
        return None

    data = joblib.load(data_path)
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

    # Spatial metadata for cnn_gru_dual_cosine_recon/cnn_gru_dual_attn_recon (see
    # model_utils.build_neighbor_curve_stack). Soft-optional, mirrors 03_main_training.py's
    # derivation -- well_id is made GLOBALLY unique (prefixed with this experiment's name)
    # since combine_group below concatenates rows from several experiments into one pool;
    # a bare per-experiment well_id (e.g. "0") would otherwise collide across experiments
    # and make neighbour-finding mix pixels from physically different wells/chips.
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
        d = load_curve_data(exp_path, curve_type)
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

    return {
        "curves": np.concatenate([p["curves"] for p in parts], axis=0),
        "features_df": pd.concat([p["features_df"] for p in parts], axis=0, ignore_index=True),
        "Y_mapped": np.concatenate([p["Y_mapped"] for p in parts], axis=0),
        "dataset_id": np.concatenate([np.full(len(p["Y_mapped"]), p["dataset_id"], dtype=object) for p in parts], axis=0),
        "dataset_names": [p["dataset_id"] for p in parts],
        "resampler": resampler,
        "coords": coords_combined,
        "well_ids": well_ids_combined,
    }


def build_lofo_splits(dataset_id):
    """Leave-one-folder-out: each fold holds out one whole dataset as the test set."""
    splits = {}
    for name in np.unique(dataset_id):
        test_idx = np.where(dataset_id == name)[0]
        train_idx = np.where(dataset_id != name)[0]
        splits[f"lofo_{name}"] = (train_idx, test_idx)
    return splits


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print(f"\n{'='*70}\n[RUNNING] {os.path.basename(__file__)}\n{'='*70}\n")
    parser = argparse.ArgumentParser(description="Cross-Dataset Leave-One-Folder-Out (LOFO) Training")
    parser.add_argument("--task_id", type=int, default=0, help="Array Job ID -> index into CROSS_DATASET_GROUPS")
    parser.add_argument("--exp_folder", type=str, default=config.DEFAULT_EXP_FOLDER)
    parser.add_argument("--force_rerun", action="store_true", help="Recompute and overwrite even if presaved results already exist")
    parser.add_argument("--curve_type", type=str, nargs='+', default=['ori_curve', 'ori_curve_avg', 'ori_curve_wavelet_sym8'], help="Which curve dataset(s) to train on. Accepts one or more values (e.g. 'ori_curve' 'ori_curve_avg').")
    parser.add_argument("--fast_mode", action="store_true",
                        help="Disable strict TF determinism (TF_CUDNN_DETERMINISTIC/enable_op_determinism) "
                             "for faster GRU/LSTM/Transformer training. RNG seeds are still set, but reruns "
                             "won't be bit-exact. Only affects this script.")
    parser.add_argument("--k_neighbors", type=int, default=24,
                        help="Neighbours per pixel (within the same well) for "
                             "cnn_gru_dual_cosine_recon/cnn_gru_dual_attn_recon's spatial reconstruction.")
    args = parser.parse_args()

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

    # for curve_type in args.curve_type:
    for curve_type in reversed(args.curve_type):
        print(f"\n\n{'#'*80}\nLOFO CROSS-DATASET CV FOR GROUP: {group_name} (curve_type: {curve_type})\nFolders: {folder_names}\n{'#'*80}")

        combined = combine_group(exp_paths, group_name, curve_type=curve_type)
        if combined is None:
            continue

        out_dir = Path(args.exp_folder) / "cross_dataset_cv" / group_name
        plot_dir = out_dir / f"model_performance_{curve_type}"
        plot_dir.mkdir(parents=True, exist_ok=True)

        resampler_path = out_dir / config.CROSS_DATASET_RESAMPLER_PATH.format(curve_type=curve_type)
        joblib.dump(combined["resampler"], resampler_path, compress=3)
        print(f"  [*] Saved curve resampler -> {resampler_path}")

        encoder = LabelEncoder()
        y_full = encoder.fit_transform(combined["Y_mapped"])
        total_count = len(y_full)

        # --- FEATURE SELECTION (MUTUAL INFORMATION) on the combined pool ---
        print(f"\n  [*] Calculating Mutual Information for Top 10 Features...")
        X_candidates = combined["features_df"][config.LD_FEATURES].values
        X_candidates_clean = np.nan_to_num(X_candidates, nan=0.0, posinf=0.0, neginf=0.0)
        mi_scores = mutual_info_classif(X_candidates_clean, y_full, random_state=0)
        top_10_idx = np.argsort(mi_scores)[-10:][::-1]
        top_10_features = [config.LD_FEATURES[i] for i in top_10_idx]
        print(f"  [*] Selected Top 10 Features: {top_10_features}")

        results_file_path = out_dir / config.CROSS_DATASET_RESULT_PATH.format(mode="lofo", curve_type=curve_type)
        outlier_filters = [None, 'lstm_ae_glb_ds1_label_elbow', 'spatial_knn_label_elbow', 'spatial_grid_label_elbow']
        # outlier_filters = [None]
        models = [
            "knn", "cnn",  "cnn_inc", 
            # "cnn_lf", 
            "gru", "cnn_gru_dual", "cnn_gru_dual_inc",
            #  "gru_lf", 
            "transformer", "cnn_trans_dual", "cnn_trans_dual_inc",
            #  "trans_lf", 

            # Spatial-reconstruction variants inspired GNN:
            # pixel_row_idx/pixel_col_idx -- see coords_full/well_ids_full above.
            "cnn_gru_dual_cosine_recon", "cnn_gru_dual_attn_recon",

            # # From pretained outlier unsupervised training encoder
            # "lstm_ae_clf",

            # # New gated dual-branch fusion models
            # "cnn_gru_gate", "cnn_gru_hadamard", "cnn_gru_crossattn", "cnn_gru_film",
            # "cnn_trans_gate", "cnn_trans_hadamard", "cnn_trans_crossattn", "cnn_trans_film",
        ]

        if args.force_rerun:
            print(f"  -> [FORCE RERUN] Ignoring presaved results at {results_file_path}. Recomputing everything...")
            lofo_results = {}
        else:
            lofo_results = joblib.load(results_file_path) if results_file_path.exists() else {}

        lofo_splits = build_lofo_splits(combined["dataset_id"])
        total_folds = len(lofo_splits)
        for fold_idx, (fold_label, (train_idx, test_idx)) in enumerate(lofo_splits.items()):
        # for fold_idx, (fold_label, (train_idx, test_idx)) in enumerate(reversed(list(lofo_splits.items()))):
            progress_pct = ((fold_idx + 1) / total_folds) * 100
            print(f"\n{'='*75}")
            print(f"[{fold_idx+1}/{total_folds} | {progress_pct:.1f}%] FOLD: {fold_label} | train={len(train_idx)} test={len(test_idx)}")
            print(f"{'='*75}")

            cached_fold = lofo_results.get(fold_label, {})

            def checkpoint(updated_results, fold_label=fold_label):
                lofo_results[fold_label] = updated_results
                joblib.dump(lofo_results, results_file_path, compress=3)

            lofo_model_dir = out_dir / "model_interpretation" / fold_label
            lofo_model_dir.mkdir(parents=True, exist_ok=True)

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
            )

            lofo_results[fold_label] = res

            # XAI metadata: folded directly into lofo_results (see joblib_redundancy.md
            # Change 4) instead of a separate model_interpretation_{curve_type}.joblib —
            # 07_attribution_vis_all now reads top_10_features from this same file.
            if "top_10_features" not in lofo_results[fold_label]:
                lofo_results[fold_label]["top_10_features"] = {}
            for f in outlier_filters:
                lofo_results[fold_label]["top_10_features"][str(f)] = top_10_features

            # encoder.classes_ isn't persisted anywhere else for this combined pool --
            # 06b_cross_dataset_prediction_report.py needs it to label confusion
            # matrices/class-metrics with real class names instead of integer ids.
            lofo_results[fold_label]["class_names"] = [str(c) for c in encoder.classes_]

            joblib.dump(lofo_results, results_file_path, compress=3)

            # Test-fold snapshot so 07 can run attribution without re-running combine_group.
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
            joblib.dump({
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
