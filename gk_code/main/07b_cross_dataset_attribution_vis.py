import os
import sys
import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

import config
sys.path.insert(0, "utils/model_training")
import model_utils_gated  # noqa: F401 -- registers custom layers for .keras deserialization

# Reuse 07's pure model-loading/XAI/plotting functions instead of duplicating
# them -- only the data-loading layer differs (the per-fold xai_data snapshot
# saved by 04, vs 07's single-experiment prepare_dataset()).
_attrib = importlib.import_module("07_attribution_vis_all")
load_saved_models            = _attrib.load_saved_models
extract_xai_artifacts        = _attrib.extract_xai_artifacts
plot_latent_tsne             = _attrib.plot_latent_tsne
plot_latent_saliency_heatmap = _attrib.plot_latent_saliency_heatmap
compute_kinetic_feature_cache = _attrib.compute_kinetic_feature_cache
plot_latent_feature_mapping  = _attrib.plot_latent_feature_mapping
merge_and_save_scores        = _attrib.merge_and_save_scores
set_global_determinism       = _attrib.set_global_determinism

# Reuse 06b's group/fold discovery -- same cross_dataset_cv/ layout.
_report06b = importlib.import_module("06b_cross_dataset_prediction_report")
get_group_paths = _report06b.get_group_paths
get_fold_labels = _report06b.get_fold_labels


def load_snapshot(group_dir, fold_label, curve_type):
    snapshot_path = group_dir / "model_interpretation" / fold_label / f"xai_data_{curve_type}.joblib"
    if not snapshot_path.exists():
        return None
    return joblib.load(snapshot_path)


def run_fold(group_dir, fold_label, curve_type, filter_keys, force_rerun=False, top_n=10):
    group_name = group_dir.name
    model_dir = group_dir / "model_interpretation" / fold_label

    snapshot = load_snapshot(group_dir, fold_label, curve_type)
    if snapshot is None:
        print(f"  -> [SKIP] {group_name}/{fold_label}: snapshot not found for curve_type={curve_type}.")
        return

    # The snapshot's X_curves_test is plain 2D (N, T) -- same as 07's own
    # prepare_dataset(), the saved .keras models require an explicit 3D
    # (batch, T, 1) input (create_cnn_model etc. all do Input(shape=(T, 1))
    # with no internal Reshape), so add the channel dim back here.
    X_full = snapshot["X_curves_test"].astype(np.float32)[..., None]
    X_man_full = snapshot.get("X_man_test")
    y_full = snapshot["y_test"]
    timestamps = snapshot["timestamps"]
    top_10_features = snapshot["top_10_features"]
    dataset_label = f"{group_name} | held-out: {fold_label}"

    out_dir = config.get_viz_dir(group_dir.parent.parent, "model_interpretation_cross_dataset_cv") / group_name / fold_label
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_size = min(512, len(X_full))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_full), size=batch_size, replace=False)
    X_batch = X_full[idx]
    X_man_batch = X_man_full[idx] if X_man_full is not None else None
    y_batch = y_full[idx]

    mean_curve = np.squeeze(np.mean(X_batch, axis=0))
    std_curve = np.squeeze(np.std(X_batch, axis=0))
    if mean_curve.ndim > 1:
        mean_curve = mean_curve.mean(axis=-1)
        std_curve = std_curve.mean(axis=-1)

    # Dataset-level (independent of model/filter) -- compute once per fold,
    # reused across whichever filters' models are actually found below.
    feat_cache = None

    # Accumulated across every filter_key processed below, then merged/saved once at
    # the end of this fold's run. curve_type is in the output filename (not just a
    # column) because --task_id flattens (group, fold, curve_type) onto the SLURM
    # array axis (see __main__ below) -- two different curve_types for the same fold
    # can run as separate, concurrent processes, and a single shared filename across
    # curve_types would race on the load->merge->dump cycle with no locking.
    score_dfs, profile_dfs = [], []

    for filter_key in filter_keys:
        filt_label = "none" if filter_key is None else str(filter_key)
        name_suffix = f"{filt_label}_{curve_type}"

        print(f"\n{'='*70}\n[*] {dataset_label} | filter={filt_label} | curve_type={curve_type}\n{'='*70}")

        models = load_saved_models(model_dir, str(filter_key), X_full.shape[1], curve_type=curve_type)
        if not models:
            print(f"  [-] No compatible saved models found for filter={filt_label}. Skipping.")
            continue

        tsne_path = out_dir / f"01_TSNE_{name_suffix}.png"
        saliency_paths = {m: out_dir / f"02_SALIENCY_{m}_{name_suffix}.png" for m in models}
        mapping_paths  = {m: out_dir / f"03_LATENT_MAPPING_{m}_{name_suffix}.png" for m in models}
        expected_outputs = [tsne_path] + list(saliency_paths.values()) + list(mapping_paths.values())
        if not force_rerun and all(p.exists() for p in expected_outputs):
            print(f"  [-] Skipping: outputs already exist (use --force_rerun to regenerate).")
            continue

        print(f"  -> Generating Central XAI Artifacts...")
        # lstm_ae_clf is never trained for cross-dataset LOFO (04 excludes it --
        # there's no single pretrained encoder that matches the pooled
        # multi-dataset curves), so unlike 07's per-experiment pipeline there's
        # no MinMaxScaler to load/apply here.
        artifacts = extract_xai_artifacts(models, X_batch, X_man_batch, lstm_ae_scaler=None)

        plot_latent_tsne(models, X_full, X_man_full, y_full, dataset_label, tsne_path)

        for model_name in models.keys():
            plot_latent_saliency_heatmap(
                artifacts, model_name, timestamps, top_10_features,
                mean_curve, std_curve, X_batch, y_batch, dataset_label,
                saliency_paths[model_name])

        if feat_cache is None:
            print(f"  -> Computing kinetic feature cache for {dataset_label} ...")
            feat_cache = compute_kinetic_feature_cache(X_batch, timestamps)
        feat_matrix, feat_sensitivity, feat_names = feat_cache

        for model_name in models.keys():
            result = plot_latent_feature_mapping(
                artifacts, model_name, X_batch, timestamps,
                feat_matrix, feat_sensitivity, feat_names,
                mean_curve, std_curve, dataset_label,
                mapping_paths[model_name], TOP_N=top_n,
                return_scores=True)
            if result is None:
                continue
            scores_df, profiles_df = result
            if not scores_df.empty:
                scores_df["model_name"] = model_name
                scores_df["filter_key"] = filt_label
                scores_df["curve_type"] = curve_type
                scores_df["group_name"] = group_name
                scores_df["fold_label"] = fold_label
                score_dfs.append(scores_df)
            if not profiles_df.empty:
                profiles_df["model_name"] = model_name
                profiles_df["filter_key"] = filt_label
                profiles_df["curve_type"] = curve_type
                profiles_df["group_name"] = group_name
                profiles_df["fold_label"] = fold_label
                profile_dfs.append(profiles_df)

        print(f"  [✓] Processed filter={filt_label}")
        tf.keras.backend.clear_session()

    if (score_dfs or profile_dfs) and feat_cache is not None:
        feat_matrix, feat_sensitivity, feat_names = feat_cache
        main_features = [f for f in config.XAI_KINETIC_FEATURE_GROUP.keys() if f in feat_names]
        curve_meta = {curve_type: {
            "mean_curve": mean_curve, "std_curve": std_curve,
            "timestamps": timestamps, "feat_names": main_features,
        }}
        scores_path = model_dir / f"latent_feature_scores_{curve_type}.joblib"
        merge_and_save_scores(
            pd.concat(score_dfs, ignore_index=True) if score_dfs else pd.DataFrame(),
            pd.concat(profile_dfs, ignore_index=True) if profile_dfs else pd.DataFrame(),
            curve_meta, scores_path,
            score_key_columns=["model_name", "filter_key", "curve_type", "group_name",
                              "fold_label", "branch", "latent_rank", "feature"],
            profile_key_columns=["model_name", "filter_key", "curve_type", "group_name",
                                "fold_label", "branch", "latent_rank"],
        )
        print(f"  [✓] Saved merged latent_feature_scores to {scores_path}")


if __name__ == "__main__":
    print(f"\n{'='*70}\n[RUNNING] {os.path.basename(__file__)}\n{'='*70}\n")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    parser = argparse.ArgumentParser(description="XAI Visualization Pipeline for Cross-Dataset LOFO Models")
    parser.add_argument("--task_id", type=int, default=None,
                        help="Array Job ID -- index into the flattened list of (group, fold, curve_type) combos (default: run all)")
    parser.add_argument("--exp_folder", type=str, default=config.DEFAULT_EXP_FOLDER)
    parser.add_argument("--group", type=str, default=None,
                        help="Restrict to a single group name under cross_dataset_cv/ (default: all discovered groups)")
    parser.add_argument("--filter_key", type=str, nargs="*",
                        default=[None, "lstm_ae_glb_ds1_label_elbow",
                                 "spatial_knn_label_elbow", "spatial_grid_label_elbow"],
                        help="kinetic_features column(s) used to select which saved models to interpret. Pass 'None' for baseline.")
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--curve_type", type=str, nargs="+", default=["ori_curve", "ori_curve_avg"])
    parser.add_argument("--top_n", type=int, default=10,
                        help="Unique-feature latent rows to show per branch in the latent->feature mapping plot")
    args = parser.parse_args()

    set_global_determinism(0)

    filter_keys = [None if f == "None" else f for f in args.filter_key]

    group_paths = get_group_paths(args.exp_folder)
    if args.group is not None:
        group_paths = [p for p in group_paths if p.name == args.group]

    pairs = []
    for group_dir in group_paths:
        for curve_type in args.curve_type:
            for fold_label in get_fold_labels(group_dir, curve_type):
                pairs.append((group_dir, fold_label, curve_type))

    if not pairs:
        print(f"No cross-dataset LOFO results found under {args.exp_folder}/cross_dataset_cv/. Exiting.")
        sys.exit(0)

    if args.task_id is not None:
        if args.task_id >= len(pairs):
            print(f"Task ID {args.task_id} is out of bounds for {len(pairs)} (group, fold, curve_type) combos. Exiting.")
            sys.exit(0)
        pairs = [pairs[args.task_id]]

    for group_dir, fold_label, curve_type in pairs:
        print(f"\n{'#'*80}\n[*] PROCESSING: {group_dir.name} | {fold_label} (curve_type: {curve_type})\n{'#'*80}")
        run_fold(group_dir, fold_label, curve_type, filter_keys, force_rerun=args.force_rerun, top_n=args.top_n)
