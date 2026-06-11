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
from model_utils import evaluate_outlier_filters, plot_ml_results, set_global_determinism, CurveResampler

import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
set_global_determinism(0)


# ============================================================
# HELPERS
# ============================================================
def load_ori_curves(exp_path):
    """Load the 'ori_curves' dataset, kinetic_features, and label-mapped Y_well for one experiment folder."""
    data_path = os.path.join(exp_path, config.TRAINING_DATA_PATH)
    if not os.path.exists(data_path):
        print(f"  -> Skipping {exp_path.name}: '{data_path}' not found.")
        return None

    data = joblib.load(data_path)
    dataset_name = list(data["dataset_name"])
    if "ori_curves" not in dataset_name:
        print(f"  -> Skipping {exp_path.name}: 'ori_curves' not found.")
        return None
    idx = dataset_name.index("ori_curves")

    if exp_path.name not in config.LABEL_MAPPINGS:
        print(f"  -> Skipping {exp_path.name}: no LABEL_MAPPINGS entry found (required for cross-dataset grouping).")
        return None
    mapping = config.LABEL_MAPPINGS[exp_path.name]

    Y_well_raw = np.asarray(data["Y_well"])
    Y_mapped = np.array([mapping.get(w, w) for w in Y_well_raw])

    return {
        "curves": data["dataset"][idx],
        "features_df": data["kinetic_features"][idx].reset_index(drop=True),
        "Y_well_raw": Y_well_raw,
        "Y_mapped": Y_mapped,
        "well_to_label": mapping,
        "timestamps": np.asarray(data["timestamps"], dtype=float),
    }


def combine_group(exp_paths, group_name):
    """Concatenate ori_curves data for all folders in a group, validating they share one label mapping.

    Datasets may have different timestamp grids, so curves are first resampled
    onto a single common time grid (fitted across the whole group) via
    `CurveResampler` before concatenation.
    """
    parts = []
    ref_mapping = None
    for exp_path in exp_paths:
        d = load_ori_curves(exp_path)
        if d is None:
            continue
        if ref_mapping is None:
            ref_mapping = d["well_to_label"]
        elif d["well_to_label"] != ref_mapping:
            raise ValueError(
                f"[{group_name}] LABEL_MAPPINGS for '{exp_path.name}' differs from the first "
                f"dataset in the group. Cross-dataset CV requires an identical well->label mapping "
                f"across every folder in a CROSS_DATASET_GROUPS entry."
            )
        d["dataset_id"] = exp_path.name
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

    return {
        "curves": np.concatenate([p["curves"] for p in parts], axis=0),
        "features_df": pd.concat([p["features_df"] for p in parts], axis=0, ignore_index=True),
        "Y_mapped": np.concatenate([p["Y_mapped"] for p in parts], axis=0),
        "well_idx": np.concatenate([p["Y_well_raw"] for p in parts], axis=0),
        "dataset_id": np.concatenate([np.full(len(p["Y_mapped"]), p["dataset_id"], dtype=object) for p in parts], axis=0),
        "well_to_label": ref_mapping,
        "dataset_names": [p["dataset_id"] for p in parts],
        "resampler": resampler,
    }


def build_lofo_splits(dataset_id):
    """Leave-one-folder-out: each fold holds out one whole dataset as the test set."""
    splits = {}
    for name in np.unique(dataset_id):
        test_idx = np.where(dataset_id == name)[0]
        train_idx = np.where(dataset_id != name)[0]
        splits[f"lofo_{name}"] = (train_idx, test_idx)
    return splits


def build_well_cv_splits(well_idx, well_to_label):
    """
    Group well indices by their mapped target label, then build folds by
    pairing one well-index per target class together (cycling if class
    well-counts differ). Each fold's test set = every sample whose well
    index is one of the chosen indices, across ALL datasets in the group
    -> (n_targets x n_datasets) wells held out per fold.
    """
    label_to_wells = {}
    for w, label in well_to_label.items():
        label_to_wells.setdefault(label, []).append(w)
    for label in label_to_wells:
        label_to_wells[label] = sorted(label_to_wells[label])

    n_folds = max(len(wells) for wells in label_to_wells.values())

    splits = {}
    for fold in range(n_folds):
        test_wells = sorted({wells[fold % len(wells)] for wells in label_to_wells.values()})
        test_mask = np.isin(well_idx, test_wells)
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]
        splits[f"wellcv_fold{fold}_wells{test_wells}"] = (train_idx, test_idx)
    return splits


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Dataset Robustness Training (LOFO + Leave-One-Well-Out CV)")
    parser.add_argument("--task_id", type=int, default=0, help="Array Job ID -> index into CROSS_DATASET_GROUPS")
    parser.add_argument("--exp_folder", type=str, default=config.DEFAULT_EXP_FOLDER)
    parser.add_argument("--mode", type=str, default="both", choices=["lofo", "well_cv", "both"])
    parser.add_argument("--force_rerun", action="store_true", help="Recompute and overwrite even if presaved results already exist")
    args = parser.parse_args()

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

    print(f"\n\n{'#'*80}\nCROSS-DATASET CV FOR GROUP: {group_name}\nFolders: {folder_names}\n{'#'*80}")

    combined = combine_group(exp_paths, group_name)
    if combined is None:
        sys.exit(0)

    out_dir = Path(args.exp_folder) / "cross_dataset_cv" / group_name
    plot_dir = out_dir / "model_performance"
    plot_dir.mkdir(parents=True, exist_ok=True)

    resampler_path = out_dir / config.CROSS_DATASET_RESAMPLER_PATH
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

    # Each CV mode (lofo / wellcv) is checkpointed to its own joblib file so the
    # two robustness modes can be loaded and compared independently later.
    results_file_paths = {
        m: out_dir / config.CROSS_DATASET_RESULT_PATH.format(mode=m)
        for m in ("lofo", "wellcv")
    }

    all_ml_results = {}
    for m, path in results_file_paths.items():
        if args.force_rerun:
            print(f"  -> [FORCE RERUN] Ignoring presaved results at {path}. Recomputing everything...")
            all_ml_results[m] = {}
        else:
            all_ml_results[m] = joblib.load(path) if path.exists() else {}

    fold_specs = {}
    if args.mode in ("lofo", "both"):
        for fold_label, split in build_lofo_splits(combined["dataset_id"]).items():
            fold_specs[fold_label] = ("lofo", split)
    if args.mode in ("well_cv", "both"):
        for fold_label, split in build_well_cv_splits(combined["well_idx"], combined["well_to_label"]).items():
            fold_specs[fold_label] = ("wellcv", split)

    outlier_filters = [None, 'lstm_ae_glb_ds1_label_elbow', 'spatial_knn_label_elbow', 'spatial_grid_label_elbow'] # config.OUTLIER_FILTERS
    # models = ["knn", "cnn", "cnn_lf", "gru", "gru_lf", "transformer", "trans_lf", "cnn_gru_dual", "cnn_trans_dual"]
    models = ["cnn", "gru", "transformer", "cnn_gru_dual", "cnn_trans_dual"]

    total_folds = len(fold_specs)
    for fold_idx, (fold_label, (cv_mode, (train_idx, test_idx))) in enumerate(fold_specs.items()):
        progress_pct = ((fold_idx + 1) / total_folds) * 100
        print(f"\n{'='*75}")
        print(f"[{fold_idx+1}/{total_folds} | {progress_pct:.1f}%] FOLD: {fold_label} | train={len(train_idx)} test={len(test_idx)}")
        print(f"{'='*75}")

        results_file_path = results_file_paths[cv_mode]
        mode_results = all_ml_results[cv_mode]
        cached_fold = mode_results.get(fold_label, {})

        def checkpoint(updated_results, fold_label=fold_label, mode_results=mode_results, results_file_path=results_file_path):
            mode_results[fold_label] = updated_results
            joblib.dump(mode_results, results_file_path, compress=3)

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
        )

        mode_results[fold_label] = res
        joblib.dump(mode_results, results_file_path, compress=3)

        plot_ml_results(
            results_dict=mode_results[fold_label],
            outlier_filters=outlier_filters,
            dataset_name=group_name,
            mode_name=fold_label,
            total_count=total_count,
            save_prefix=os.path.join(plot_dir, fold_label)
        )

        gc.collect()
