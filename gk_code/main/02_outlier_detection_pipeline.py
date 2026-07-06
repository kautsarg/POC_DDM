import os
import sys
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gc
import base64
import argparse
import joblib
from safe_io import safe_joblib_dump
import itertools
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import tensorflow as tf

import config

sys.path.insert(0, 'utils')
sys.path.insert(0, 'utils/02_outlier_detection')
sys.path.insert(0, 'utils/model_training')
import chip_utilities as utils
import sigmoid_fitting as sp
from msc_outlier import run_msc_pipeline
from amf_outlier import run_amf_pipeline
from mean_std_outlier import run_meanstd_pipeline
from knn_fingerprint_filter import run_knnfilter_pipeline
from autoencoder_outlier import run_autoencoder_pipeline
from lstm_autoencoder_outlier import run_lstm_autoencoder_pipeline
from cnn_autoencoder_outlier import run_cnn_autoencoder_pipeline
from autoencoder_outlier_per_well import run_autoencoder_per_well_pipeline
from lstm_autoencoder_outlier_per_well import run_lstm_autoencoder_per_well_pipeline
from cnn_autoencoder_outlier_per_well import run_cnn_autoencoder_per_well_pipeline
from spatial_consistency_outlier import run_spatial_consistency_knn_pipeline, run_spatial_consistency_grid_pipeline

from model_utils import set_global_determinism

WELL_CMAP = config.WELL_CMAP


# ====================================================================
# FEATURE EXTRACTION HELPERS
# ====================================================================

def _process_single_row(y, X):
    valid = np.isfinite(X) & np.isfinite(y)
    if np.sum(valid) < 3:
        return {}
    try:
        return sp.extract_kinetic_parameters_original(X, y)
    except Exception:
        return {}


def extract_kinetic_features(timestamps, curves, n_jobs=-1):
    features = Parallel(n_jobs=n_jobs)(
        delayed(_process_single_row)(y, timestamps) for y in curves
    )
    return pd.DataFrame(features)


def get_send(timestamps, curves_2d, send_n=(5, 10, 15, 20, 25)):
    dy_dx_list = Parallel(n_jobs=-1, backend="loky", batch_size='auto')(
        delayed(sp.calculate_first_derivative)(timestamps, y) for y in curves_2d
    )
    dy_dx = np.array(dy_dx_list)
    send_dict = {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        for n in send_n:
            send_dict[f"send_{n}"] = np.nanmean(dy_dx[:, -n:], axis=1)
            send_dict[f"send_abs_{n}"] = np.nanmean(np.abs(dy_dx[:, -n:]), axis=1)
    return send_dict


def build_kinetic_features(curves_2d, timestamps, metadata_df):
    """Extract kinetic parameters + Send derivatives + metadata for one dataset variant."""
    features_df = extract_kinetic_features(timestamps, curves_2d).reset_index(drop=True)
    add_features = get_send(timestamps, curves_2d)
    add_features["FFI"] = curves_2d[:, -1]
    add_features["F_range"] = curves_2d[:, -1] - curves_2d[:, 0]
    return pd.concat(
        [features_df, metadata_df.reset_index(drop=True), pd.DataFrame(add_features).reset_index(drop=True)],
        axis=1,
    )


# ====================================================================
# VISUALISATION HELPERS
# ====================================================================

def feature_boxplot(features_df, well_labels, feature_columns, target="well", title="", save_path=None):
    plot_data = features_df.copy()
    plot_data[target] = well_labels
    valid_features = [f for f in feature_columns if f in plot_data.columns]

    if not valid_features:
        print(f"Skipping {title}: None of the requested features exist.")
        return None

    n_features = len(valid_features)

    if n_features == 3:
        fig = plt.figure(figsize=(10, 8))
        axes = [fig.add_subplot(2, 2, i + 1) for i in range(3)]
        for i, feature in enumerate(valid_features):
            feature_data = plot_data.dropna(subset=[feature])
            if feature_data.empty:
                axes[i].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[i].transAxes)
            else:
                sns.boxplot(data=feature_data, x=target, y=feature, ax=axes[i])
            axes[i].set_title(feature, fontweight='bold')
            axes[i].grid(True, alpha=0.3, axis='y')

        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        f1, f2, f3 = valid_features
        unique_targets = np.unique(well_labels)
        palette = config.get_palette(unique_targets, config.WELL_COLOR_MAP if target == "well" else None)
        for val in unique_targets:
            subset = plot_data[plot_data[target] == val]
            ax4.scatter(subset[f1], subset[f2], subset[f3], label=f"Well {val}", color=palette[val], alpha=0.7, s=20)
        ax4.set_xlabel(f1, fontweight='bold')
        ax4.set_ylabel(f2, fontweight='bold')
        ax4.set_zlabel(f3, fontweight='bold')
        ax4.set_title("3D Feature Space", fontweight='bold')
        ax4.legend(title=target, bbox_to_anchor=(1.15, 1), loc='upper left')
    else:
        fig, axes = plt.subplots(1, n_features, figsize=(n_features * 4, 3))
        if n_features == 1:
            axes = [axes]
        for i, feature in enumerate(valid_features):
            feature_data = plot_data.dropna(subset=[feature])
            if feature_data.empty:
                axes[i].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[i].transAxes)
            else:
                sns.boxplot(data=feature_data, x=target, y=feature, ax=axes[i])
            axes[i].set_title(feature, fontweight='bold')
            axes[i].grid(True, alpha=0.3, axis='y')

    fig.suptitle(title, fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close(fig)
        return None
    else:
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor='white')
        plt.close(fig)
        buf.seek(0)
        return buf


def save_html_report(save_path, title, subtitle, img_buffers, flex_layout=False):
    width_style = ("width: 45%; min-width: 450px;" if flex_layout
                   else "width: 95%; max-width: 1600px; margin: 40px auto;")
    parts = [f"""
    <html>
    <head><title>{title}</title></head>
    <body style="font-family: Arial, sans-serif; background-color: #f4f4f9; text-align: center; margin: 0; padding: 20px;">
        <h1 style="color: #333; margin-bottom: 10px;">{title}</h1>
        {f'<p style="color: #666; margin-bottom: 40px;">{subtitle}</p>' if subtitle else ''}
    """]
    if flex_layout:
        parts.append("<div style='display: flex; flex-wrap: wrap; justify-content: center; gap: 20px;'>")
    for buf in img_buffers:
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        parts.append(f'''
        <div style="background: white; padding: 15px; box-shadow: 0px 4px 10px rgba(0,0,0,0.1); border-radius: 8px; {width_style}">
            <img src="data:image/png;base64,{img_b64}" style="width: 100%; height: auto;">
        </div>''')
        buf.close()
    if flex_layout:
        parts.append("</div>")
    parts.append("</body></html>")

    with open(save_path, "w") as f:
        f.write("".join(parts))


# ====================================================================
# PIPELINE STATE HELPERS
# ====================================================================

def _flush_and_save(features_to_concat, pipeline_state, dataset_name, unified_save_path):
    """Apply buffered feature DataFrames to pipeline_state and checkpoint to disk."""
    merged_any = False
    for i in range(len(dataset_name)):
        if features_to_concat[i]:
            pipeline_state["kinetic_features"][i] = pd.concat(
                [pipeline_state["kinetic_features"][i]] + features_to_concat[i], axis=1
            )
            features_to_concat[i].clear()
            merged_any = True
    if merged_any:
        safe_joblib_dump(pipeline_state, unified_save_path, compress=3)
        print("    [SAVED] Unified pipeline state updated.")


def _build_variant_lists(data):
    """Enumerate every curve variant currently present in data['curves']/data['sigmoid_curves']."""
    dataset_name = ["ori_curves"]
    dataset = [data["curves"]["ori_curves"]]   # same object — not a copy

    if "ori_curves_avg" in data["curves"]:
        dataset_name.append("ori_curves_avg")
        dataset.append(data["curves"]["ori_curves_avg"])

    if "ori_curves_norm" in data["curves"]:
        dataset_name.append("ori_curves_norm")
        dataset.append(data["curves"]["ori_curves_norm"])

    if "ori_curves_wavelet_sym8" in data["curves"]:
        dataset_name.append("ori_curves_wavelet_sym8")
        dataset.append(data["curves"]["ori_curves_wavelet_sym8"])

    for k, v in data["sigmoid_curves"].items():
        dataset_name.append(f"{k}_fitted_full")
        dataset.append(v["fitted_full"])
        dataset_name.append(f"{k}_fitted_stretched")
        dataset.append(v["fitted_stretched"])

    return dataset_name, dataset


def _load_or_build_state(exp_path, unified_save_path, force_rerun):
    """Load persisted pipeline state, or build a fresh one from preprocessed curves.
    Also detects curve variants added to "curves" since 02 last ran (e.g. a new
    --normalize_curves variant from 01/01b) and appends just those, without rebuilding
    or recomputing anything already cached."""
    pipeline_state = {}

    if os.path.exists(unified_save_path):
        print(f"  -> Found unified state at {unified_save_path}. Loading...")
        try:
            pipeline_state = joblib.load(unified_save_path)
            print("  -> Unified state loaded successfully.")
        except Exception as e:
            print(f"  -> [WARNING] Failed to load state ({e}). Starting fresh.")
            pipeline_state = {}

    if force_rerun and pipeline_state:
        # Only discard 02's OWN previously-derived keys — 01's keys (curves, sigmoid_curves,
        # timestamps, well_labels, metadata, ...) are untouched, so --force_rerun never
        # forces 01's raw chip-data reconstruction to re-run too.
        print("  -> [FORCE RERUN] Discarding 02's own cached dataset/kinetic_features "
              "(01's curves/metadata are kept). Recomputing 02's part...")
        for k in ("dataset", "dataset_name", "Y_well", "kinetic_features",
                  "linear_feature_combinations", "important_feature_combinations", "importance_dfs"):
            pipeline_state.pop(k, None)

    # 01 and 02 share one joblib (config.TRAINING_DATA_PATH) — 01 patches "curves"/
    # "sigmoid_curves"/"well_labels"/"timestamps"/"metadata" directly into the same
    # file 02 loads above, so no second file read is needed here. See
    # joblib_redundancy.md "01 + 02: one shared joblib, scripts stay separate".
    if "curves" not in pipeline_state:
        print(f"Skipping {exp_path.name} - run 01_curve_preprocessing_v6.py first "
              f"(no 'curves' key in '{config.TRAINING_DATA_PATH}').")
        sys.exit(0)

    expected_name, expected_arr = _build_variant_lists(pipeline_state)

    if "dataset" not in pipeline_state:
        pipeline_state.update({
            "dataset_name": expected_name,            # plain list, NOT np.array() — keeps array
            "dataset": expected_arr,                  # objects identical to curves[...] so joblib
            "Y_well": pipeline_state["well_labels"],   # dedupes them on disk
        })
    else:
        existing_name = list(pipeline_state["dataset_name"])
        new_vars = [(n, a) for n, a in zip(expected_name, expected_arr) if n not in existing_name]
        if new_vars:
            print(f"  -> Detected new curve variant(s) since last run: "
                  f"{[n for n, _ in new_vars]}. Appending without recomputing existing ones...")
            for n, a in new_vars:
                pipeline_state["dataset_name"].append(n)
                pipeline_state["dataset"].append(a)
            # Outlier-filter columns are computed per curve variant. Strip them from all existing
            # kinetic_features entries so the outlier pipeline reruns for every variant — the
            # new variant would otherwise be skipped because the checks use kinetic_features[0]
            # as the cache sentinel, and that entry already has all outlier columns from the
            # previous run.
            _OUTLIER_PREFIXES = (
                "knn_top_", "cnn_ae_", "lstm_ae_",
                "msc_label_", "amf_label_", "mean_std_label_",
                "spatial_knn_label_", "spatial_grid_label_",
            )
            kf_list = pipeline_state.get("kinetic_features")
            if kf_list:
                stripped = []
                for kf in kf_list:
                    drop_cols = [c for c in kf.columns if any(c.startswith(p) for p in _OUTLIER_PREFIXES)]
                    stripped.append(kf.drop(columns=drop_cols) if drop_cols else kf)
                pipeline_state["kinetic_features"] = stripped
                print(f"  -> Cleared outlier-filter columns from existing kinetic_features "
                      f"so all variants get (re)computed in the outlier pipeline.")

    return pipeline_state



def _ensure_kinetic_features(pipeline_state, unified_save_path):
    """Compute kinetic features for any dataset variants not already cached."""
    dataset_name = pipeline_state["dataset_name"]
    dataset = pipeline_state["dataset"]
    Y_well = pipeline_state["Y_well"]
    timestamps = pipeline_state["timestamps"]
    metadata_df = pd.DataFrame(pipeline_state["metadata"])   # derived on demand — not persisted

    cached = pipeline_state.get("kinetic_features")
    cache_valid = cached and len(cached[0]) == len(Y_well)

    if cache_valid and len(cached) == len(dataset_name):
        print("  -> Using cached kinetic features from unified state.")
        return

    if cache_valid and len(cached) < len(dataset_name):
        new_names = dataset_name[len(cached):]
        print(f"  -> Extracting kinetic features for new variant(s): {new_names}...")
        new_features = [build_kinetic_features(dataset[i], timestamps, metadata_df)
                         for i in range(len(cached), len(dataset_name))]
        pipeline_state["kinetic_features"] = list(cached) + new_features
        safe_joblib_dump(pipeline_state, unified_save_path, compress=3)
        return

    print("  -> Extracting initial kinetic features (CPU Bound)...")
    kinetic_features = [build_kinetic_features(c, timestamps, metadata_df) for c in dataset]

    # Drop legacy "avg_"-prefixed variants that may appear in older cache files.
    keep = [(n, d, f) for n, d, f in zip(dataset_name, dataset, kinetic_features)
            if not n.startswith("avg_")]
    if keep:
        names_k, data_k, feat_k = zip(*keep)
        pipeline_state["dataset_name"] = list(names_k)   # plain list, NOT np.array() —
        pipeline_state["dataset"] = list(data_k)         # preserves object identity with curves[...]
        pipeline_state["kinetic_features"] = list(feat_k)
    else:
        pipeline_state["dataset_name"] = np.array([])
        pipeline_state["dataset"] = np.array([])
        pipeline_state["kinetic_features"] = []

    safe_joblib_dump(pipeline_state, unified_save_path, compress=3)


# ====================================================================
# PIPELINE STEP FUNCTIONS
# ====================================================================

def _generate_boxplots(dataset_name, kinetic_features, Y_well, exp_path):
    print("\n=== GENERATING FEATURE BOXPLOTS ===")
    msc_features = ["Ct", "Cy0", "log_F0"]
    amf_features_all = ["Fm", "Fb", "Sc", "Cs", "Send", "Send_abs", "Send_fit", "Send_fit_abs"]

    msc_plot_path = exp_path / "msc_outlier"
    amf_plot_path = exp_path / "amf_outlier"
    msc_plot_path.mkdir(parents=True, exist_ok=True)
    amf_plot_path.mkdir(parents=True, exist_ok=True)

    for name, features_df in zip(dataset_name, kinetic_features):
        clean_title = name.replace("_", " ").title()
        feature_boxplot(features_df, Y_well, msc_features,
                        title=f"MSC Features: {clean_title}",
                        save_path=msc_plot_path / f"{name}_msc_features.png")
        feature_boxplot(features_df, Y_well, amf_features_all,
                        title=f"AMF Features: {clean_title}",
                        save_path=amf_plot_path / f"{name}_amf_features_boxplot.png")


def _compute_feature_analysis(pipeline_state, Y_well, colors, cmap, unified_save_path, save_plot_flag, reports_dir):
    """Compute correlation triplets + RF importances; results cached into pipeline_state.
    Only computes for dataset variants not yet covered (e.g. a new variant appended to
    dataset_name since the last run) — existing cached entries are reused untouched."""
    dataset_name = pipeline_state["dataset_name"]
    kinetic_features = pipeline_state["kinetic_features"]

    cached_linear = pipeline_state.get("linear_feature_combinations")
    cached_important = pipeline_state.get("important_feature_combinations")
    cached_importance_dfs = pipeline_state.get("importance_dfs")

    if cached_linear is not None and len(cached_linear) == len(dataset_name):
        print("  -> Using cached feature combinations and importance data from unified state.")
        return cached_linear, cached_important, cached_importance_dfs

    start = len(cached_linear) if cached_linear is not None else 0
    new_names = dataset_name[start:]
    new_kinetic_features = kinetic_features[start:]
    if start > 0:
        print(f"  -> Computing feature combinations/importances for new variant(s): {new_names}...")

    # --- Best correlated feature triplets ---
    print("\n=== EXTRACTING BEST LINEAR TRIPLETS (CORRELATION) ===")
    linear_feature_combinations = []
    heatmap_buffers = []

    for name, features_df in zip(new_names, new_kinetic_features):
        clean_title = name.replace("_", " ").title()
        numeric_df = features_df.select_dtypes(include=['number'])

        if numeric_df.shape[1] == 0:
            print(f"  -> [SKIP] No numeric features for {clean_title}")
            linear_feature_combinations.append([])
            continue

        corr = numeric_df.corr()
        if corr.empty:
            print(f"  -> [SKIP] Empty correlation matrix for {clean_title}")
            linear_feature_combinations.append([])
            continue

        corr_abs = corr.abs()
        valid_features = [f for f in corr_abs.columns if f not in config.EXCLUDED_FEATURES]

        cannot_pair_with = {f: set() for f in valid_features}
        for f in valid_features:
            for group in config.FEATURE_GROUPS:
                if f in group:
                    cannot_pair_with[f].update(group)

        best_triplet, max_score = [], -1
        for f1, f2, f3 in itertools.combinations(valid_features, 3):
            if (f2 in cannot_pair_with[f1] or f3 in cannot_pair_with[f1]
                    or f3 in cannot_pair_with[f2]):
                continue
            score = corr_abs.loc[f1, f2] + corr_abs.loc[f1, f3] + corr_abs.loc[f2, f3]
            if score > max_score:
                max_score = score
                best_triplet = [f1, f2, f3]

        linear_feature_combinations.append(best_triplet[:])
        print(f"  -> {clean_title} Best Triplet: {best_triplet} (Avg Inter-Corr: {max_score / 3:.3f})")

        if save_plot_flag:
            fig, ax = plt.subplots(figsize=(24, 20))
            sns.heatmap(corr.mask(corr_abs < 0.5, 0), cmap="coolwarm", center=0,
                        linewidths=0.5, cbar_kws={"shrink": .75}, ax=ax)
            ax.tick_params(axis='x', rotation=90, labelsize=8)
            ax.tick_params(axis='y', rotation=0, labelsize=8)
            plt.title(f"Correlation Matrix: {clean_title}", fontsize=22, fontweight='bold', pad=20)
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='white')
            buf.seek(0)
            heatmap_buffers.append(buf)
            plt.close(fig)
            gc.collect()

    if save_plot_flag and heatmap_buffers:
        save_html_report(reports_dir / "all_correlation_heatmaps.html",
                         "Experiment Correlation Heatmaps", None, heatmap_buffers)

        print("\n=== GENERATING 3D COMBINATION PLOTS ===")
        plot_3d_buffers = []
        for name, kf, combos in zip(new_names, new_kinetic_features, linear_feature_combinations):
            if not combos:
                continue
            clean_title = name.replace("_", " ").title()
            x_feat, y_feat, z_feat = combos[:3]
            corr_sub = kf[[x_feat, y_feat, z_feat]].corr(method='pearson').abs()
            avg_corr = (corr_sub.loc[x_feat, y_feat] + corr_sub.loc[x_feat, z_feat]
                        + corr_sub.loc[y_feat, z_feat]) / 3
            X_vals = kf[[x_feat, y_feat, z_feat]].replace([np.inf, -np.inf], np.nan).fillna(-9999).values

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_vals[:, 0], X_vals[:, 1], X_vals[:, 2],
                       c=colors, cmap=cmap, vmin=0, vmax=config.N_WELLS - 1, s=30, alpha=0.8, edgecolor='k')
            ax.set_title(f"{clean_title}\n[{x_feat}, {y_feat}, {z_feat}]\n"
                         f"Avg Inter-Correlation: {avg_corr:.3f}", fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel(x_feat, fontweight='bold', labelpad=10)
            ax.set_ylabel(y_feat, fontweight='bold', labelpad=10)
            ax.set_zlabel(z_feat, fontweight='bold', labelpad=15)
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            buf.seek(0)
            plot_3d_buffers.append(buf)
            plt.close(fig)
            gc.collect()

        save_html_report(reports_dir / "best_feature_combinations_3D.html",
                         "Best 3D Feature Combinations",
                         "Highest inter-correlated valid feature triplet for each experiment.",
                         plot_3d_buffers, flex_layout=True)

    # --- Random Forest feature importances ---
    print("\n=== EXTRACTING TOP 5 INDEPENDENT FEATURES (RANDOM FOREST) ===")
    importance_dfs, rf_buffers, important_feature_combinations = [], [], []

    for name, features_df in zip(new_names, new_kinetic_features):
        clean_title = name.replace("_", " ").title()
        numeric_df = (features_df.select_dtypes(include=['number'])
                      .replace([np.inf, -np.inf], np.nan).fillna(0))

        X_scaled = StandardScaler().fit_transform(numeric_df)
        rf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
        rf.fit(X_scaled, Y_well)

        importance_df = (pd.DataFrame({'feature': numeric_df.columns, 'importance': rf.feature_importances_})
                         .sort_values(by='importance', ascending=False))
        importance_dfs.append(importance_df)

        if save_plot_flag:
            fig, ax = plt.subplots(figsize=(12, 18))
            sns.barplot(data=importance_df.head(40), x='importance', y='feature', palette='viridis', ax=ax)
            ax.set_title(f"Random Forest Importances: {clean_title}", fontsize=18, fontweight='bold', pad=20)
            ax.set_xlabel("Importance Score", fontweight='bold')
            ax.set_ylabel("Features", fontweight='bold')
            ax.grid(axis='x', linestyle='--', alpha=0.6)
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='white')
            buf.seek(0)
            rf_buffers.append(buf)
            plt.close(fig)
            gc.collect()

        # Greedy selection: top-5 features with no within-group redundancy.
        valid_df = importance_df[~importance_df['feature'].isin(config.EXCLUDED_FEATURES)].copy()
        cannot_pair_with = {f: set() for f in valid_df['feature']}
        for f in valid_df['feature']:
            for group in config.FEATURE_GROUPS:
                if f in group:
                    cannot_pair_with[f].update(group)

        selected_features = []
        for _, row in valid_df.iterrows():
            if not any(row['feature'] in cannot_pair_with[sel] for sel in selected_features):
                selected_features.append(row['feature'])
            if len(selected_features) == 5:
                break

        important_feature_combinations.append(selected_features)
        print(f"  -> {clean_title}: {selected_features}")

    if save_plot_flag and rf_buffers:
        save_html_report(reports_dir / "all_feature_importances.html",
                         "Feature Importance Analysis", None, rf_buffers)

    linear_feature_combinations = (cached_linear or []) + linear_feature_combinations
    important_feature_combinations = (cached_important or []) + important_feature_combinations
    importance_dfs = (cached_importance_dfs or []) + importance_dfs

    pipeline_state["linear_feature_combinations"] = linear_feature_combinations
    pipeline_state["important_feature_combinations"] = important_feature_combinations
    pipeline_state["importance_dfs"] = importance_dfs
    safe_joblib_dump(pipeline_state, unified_save_path, compress=3)

    return linear_feature_combinations, important_feature_combinations, importance_dfs


def _generate_tsne_3d(dataset_name, important_feature_combinations, kinetic_features, colors, cmap, reports_dir):
    """t-SNE + 3D scatter for the top-5 independent features of each dataset variant."""
    print("\n=== GENERATING TSNE & 3D PLOTS FOR TOP 5 FEATURES ===")
    tsne_buffers = []
    for name, top_5, kf in zip(dataset_name, important_feature_combinations, kinetic_features):
        clean_title = name.replace("_", " ").title()
        top_3 = top_5[:3]
        df_top5 = kf[top_5].replace([np.inf, -np.inf], np.nan).fillna(0)
        X_tsne = TSNE(n_components=2, random_state=0).fit_transform(df_top5.values)

        fig = plt.figure(figsize=(22, 9))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.scatter(df_top5[top_3[0]], df_top5[top_3[1]], df_top5[top_3[2]],
                    c=colors, cmap=cmap, vmin=0, vmax=config.N_WELLS - 1, s=40, alpha=0.8, edgecolor='k')
        ax1.set_title(f"{clean_title}\n(Top 3 Features: {', '.join(top_3)})",
                      fontsize=14, fontweight='bold', pad=15)

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.scatter(X_tsne[:, 0], X_tsne[:, 1],
                    c=colors, cmap=cmap, vmin=0, vmax=config.N_WELLS - 1, s=40, alpha=0.8, edgecolor='k')
        ax2.set_title(f"{clean_title}\n(t-SNE on All 5: {', '.join(top_5)})",
                      fontsize=14, fontweight='bold', pad=15)
        ax2.grid(True, linestyle='--', alpha=0.6)

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        tsne_buffers.append(buf)
        plt.close(fig)
        gc.collect()

    save_html_report(reports_dir / "independent_features_tsne_and_3d.html",
                     "Top 5 Independent Features Dimensionality Reduction",
                     "Visualizing the highest importance features after removing mathematical redundancies.",
                     tsne_buffers)


ALL_FILTERS = frozenset({"lstm_ae", "spatial_knn", "spatial_grid"})


def _run_outlier_pipelines(exp_path, pipeline_state, unified_save_path,
                            linear_feature_combinations, important_feature_combinations,
                            ae_configs, knn_filter_config, spatial_knn_configs, spatial_grid_configs,
                            downsample_factor, save_plot_flag, filters=ALL_FILTERS):
    """Run all outlier detection sub-pipelines, skipping columns already computed."""
    print("\n=== RUNNING OUTLIER DETECTION PIPELINES ===")

    dataset_name = pipeline_state["dataset_name"]
    dataset = pipeline_state["dataset"]
    kinetic_features = pipeline_state["kinetic_features"]
    Y_well = pipeline_state["Y_well"]
    metadata_df = pd.DataFrame(pipeline_state["metadata"])   # derived on demand — not persisted
    ref_curves = dataset[0]

    has_spatial_info = {"pixel_row_idx", "pixel_col_idx"}.issubset(metadata_df.columns)
    if not has_spatial_info:
        print("  -> [INFO] No 'pixel_row_idx'/'pixel_col_idx' in metadata_df "
              "(experiment has no spatial chip layout) — spatial consistency "
              "filters (KNN/Grid) will be skipped.")

    if "kinetic_features" not in pipeline_state:
        print("  -> [ERROR] kinetic_features missing from pipeline_state. Aborting.")
        sys.exit(1)

    features_to_concat = [[] for _ in range(len(dataset_name))]

    def flush():
        _flush_and_save(features_to_concat, pipeline_state, dataset_name, unified_save_path)

    # AE methods run on all standard curve variants (raw, avg, wavelet).
    ae_names = [n for n in dataset_name
                if n in ('ori_curves', 'ori_curves_avg', 'ori_curves_wavelet_sym8')]
    ae_dataset = [dataset[list(dataset_name).index(n)] for n in ae_names]
    ae_idx = [list(dataset_name).index(n) for n in ae_names]

    msc_configs = [
        ("msc_linear_0.001", 0.001, linear_feature_combinations),
        ("msc_baseline_0.001", 0.001, [["Ct", "Cy0", "log_F0"]] * len(dataset_name)),
    ]
    amf_configs = [
        ("amf_important", important_feature_combinations),
        ("amf_send_5", [["Fm", "Fb", "Sc", "Cs", "send_5"]] * len(dataset_name)),
    ]
    mean_std_configs = []  # Add (label, num_std) tuples to enable

    def _missing_ae(prefix_fn):
        """Return ae_configs entries whose column is absent from any AE dataset's features."""
        return [pct for pct in ae_configs
                if any(prefix_fn(pct) not in kinetic_features[idx].columns for idx in ae_idx)]

    def _missing_global(label_fn):
        """Return configs whose label column is absent from dataset[0]'s features."""
        return [pct for pct in knn_filter_config if label_fn(pct) not in kinetic_features[0].columns]

    # --- CNN AutoEncoder (Global) --- [DISABLED]
    # missing_cnn_glb = _missing_ae(lambda p: f"cnn_ae_glb_ds{downsample_factor}_label_{p}")
    # if missing_cnn_glb:
    #     extracted_dfs = run_cnn_autoencoder_pipeline(
    #         ae_names, ae_dataset, Y_well, ref_curves,
    #         str(exp_path / "ae_outlier"), missing_cnn_glb,
    #         save_plot=save_plot_flag, downsample_factor=downsample_factor, per_well=False)
    #     for i, name in enumerate(ae_names):
    #         features_to_concat[list(dataset_name).index(name)].append(extracted_dfs[i])
    #     flush()
    # else:
    #     print("  -> [SKIP] CNN AutoEncoder (Global): Already calculated.")

    # --- LSTM AutoEncoder (Global) ---
    if "lstm_ae" not in filters:
        print("  -> [DISABLED] LSTM AutoEncoder (Global): not in --filters.")
    else:
        missing_lstm_glb = _missing_ae(lambda p: f"lstm_ae_glb_ds{downsample_factor}_label_{p}")
        if missing_lstm_glb:
            # save_encoder_dir: the global (per_well=False) encoder is also saved standalone
            # here, reusable later as a pretrained backbone for model_utils.py's "lstm_ae_clf"
            # classifier (see 03_main_training.py) instead of training a fresh LSTM from scratch.
            extracted_dfs = run_lstm_autoencoder_pipeline(
                ae_names, ae_dataset, Y_well, ref_curves,
                str(exp_path / "ae_per_well_outlier"), missing_lstm_glb,
                save_plot=save_plot_flag, downsample_factor=downsample_factor, per_well=False,
                save_encoder_dir=str(exp_path / "pretrained_encoders"))
            for i, name in enumerate(ae_names):
                features_to_concat[list(dataset_name).index(name)].append(extracted_dfs[i])
            flush()
        else:
            print("  -> [SKIP] LSTM AutoEncoder (Global): Already calculated.")

    # --- KNN Filter --- [DISABLED]
    # missing_knn = [pct for pct in knn_filter_config
    #                if f"knn_top_{pct}" not in kinetic_features[0].columns]
    # if missing_knn:
    #     extracted_dfs = run_knnfilter_pipeline(
    #         dataset_name, dataset, Y_well, ref_curves,
    #         str(exp_path / "knnfilter_outlier"), missing_knn, save_plot=save_plot_flag)
    #     for i in range(len(dataset_name)):
    #         features_to_concat[i].append(extracted_dfs[i])
    #     flush()
    # else:
    #     print("  -> [SKIP] KNN Filter: Already calculated.")

    # --- Spatial Consistency Filter (KNN neighbors) ---
    if "spatial_knn" not in filters:
        print("  -> [DISABLED] Spatial Consistency Filter (KNN): not in --filters.")
    else:
        missing_spatial_knn = [pct for pct in spatial_knn_configs
                               if f"spatial_knn_label_{pct}" not in kinetic_features[0].columns]
        if not has_spatial_info:
            print("  -> [SKIP] Spatial Consistency Filter (KNN): No pixel coordinates available.")
        elif missing_spatial_knn:
            extracted_dfs = run_spatial_consistency_knn_pipeline(
                dataset_name, dataset, Y_well, ref_curves, metadata_df,
                str(exp_path / "spatial_knn_outlier"), missing_spatial_knn,
                k_neighbors=config.SPATIAL_CONSISTENCY_KNN_K, save_plot=save_plot_flag)
            for i in range(len(dataset_name)):
                features_to_concat[i].append(extracted_dfs[i])
            flush()
        else:
            print("  -> [SKIP] Spatial Consistency Filter (KNN): Already calculated.")

    # --- Spatial Consistency Filter (Grid neighbors) ---
    if "spatial_grid" not in filters:
        print("  -> [DISABLED] Spatial Consistency Filter (Grid): not in --filters.")
    else:
        missing_spatial_grid = [pct for pct in spatial_grid_configs
                                if f"spatial_grid_label_{pct}" not in kinetic_features[0].columns]
        if not has_spatial_info:
            print("  -> [SKIP] Spatial Consistency Filter (Grid): No pixel coordinates available.")
        elif missing_spatial_grid:
            extracted_dfs = run_spatial_consistency_grid_pipeline(
                dataset_name, dataset, Y_well, ref_curves, metadata_df,
                str(exp_path / "spatial_grid_outlier"), missing_spatial_grid,
                window=config.SPATIAL_CONSISTENCY_GRID_WINDOW, save_plot=save_plot_flag)
            for i in range(len(dataset_name)):
                features_to_concat[i].append(extracted_dfs[i])
            flush()
        else:
            print("  -> [SKIP] Spatial Consistency Filter (Grid): Already calculated.")

    # --- MSC Filter --- [DISABLED]
    # msc_to_run, msc_skip = [], []
    # for exp_label, p_val, feats in msc_configs:
    #     (msc_to_run if f"msc_label_{exp_label}" not in kinetic_features[0].columns
    #      else msc_skip).append((exp_label, p_val, feats))
    # for exp_label, _, __ in msc_skip:
    #     print(f"  -> [SKIP] MSC [{exp_label}]: Already calculated.")
    # if msc_to_run:
    #     for exp_label, p_val, feats in msc_to_run:
    #         extracted_dfs = run_msc_pipeline(
    #             exp_label, p_val, ref_curves, str(exp_path / "msc_outlier"),
    #             dataset_name, kinetic_features, dataset, Y_well, feats, save_plot=False)
    #         for i in range(len(dataset_name)):
    #             features_to_concat[i].append(extracted_dfs[i])
    #     flush()

    # --- AMF Filter --- [DISABLED]
    # amf_to_run, amf_skip = [], []
    # for exp_label, feats in amf_configs:
    #     (amf_to_run if f"amf_label_{exp_label}" not in kinetic_features[0].columns
    #      else amf_skip).append((exp_label, feats))
    # for exp_label, _ in amf_skip:
    #     print(f"  -> [SKIP] AMF [{exp_label}]: Already calculated.")
    # if amf_to_run:
    #     for exp_label, feats in amf_to_run:
    #         extracted_dfs = run_amf_pipeline(
    #             exp_label, feats, ref_curves, str(exp_path / "amf_outlier"),
    #             dataset_name, kinetic_features, dataset, Y_well, save_plot=False)
    #         for i in range(len(dataset_name)):
    #             features_to_concat[i].append(extracted_dfs[i])
    #     flush()

    # --- Mean/Std Filter --- [DISABLED]
    # mean_std_to_run, mean_std_skip = [], []
    # for exp_label, num_std in mean_std_configs:
    #     (mean_std_to_run if f"mean_std_label_{exp_label}" not in kinetic_features[0].columns
    #      else mean_std_skip).append((exp_label, num_std))
    # for exp_label, _ in mean_std_skip:
    #     print(f"  -> [SKIP] Mean/Std [{exp_label}]: Already calculated.")
    # if mean_std_to_run:
    #     for exp_label, num_std in mean_std_to_run:
    #         extracted_dfs = run_meanstd_pipeline(
    #             exp_label, num_std, ref_curves, str(exp_path / "meanstd_outlier"),
    #             dataset_name, dataset, Y_well, kinetic_features[0].index, save_plot=False)
    #         for i in range(len(dataset_name)):
    #             features_to_concat[i].append(extracted_dfs[i])
    #     flush()

    # Safety flush for anything not yet checkpointed.
    flush()


# ====================================================================
# MASTER PIPELINE
# ====================================================================

def run_pipeline(exp_path, force_rerun=False, save_plot_flag=False, filters=ALL_FILTERS):
    """End-to-end outlier detection pipeline for a single experiment folder."""
    exp_path = Path(exp_path)
    print(f"\n\n{'#'*80}\nSTARTING MASTER PIPELINE FOR: {exp_path.name}\n{'#'*80}")

    unified_save_path = exp_path / config.TRAINING_DATA_PATH
    reports_dir = config.get_viz_dir(exp_path, "visual_reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load or build initial state from preprocessed curves.
    pipeline_state = _load_or_build_state(exp_path, unified_save_path, force_rerun)

    assert len(pipeline_state["dataset_name"]) == len(pipeline_state["dataset"]), \
        "dataset_name / dataset length mismatch"

    # 3. Ensure kinetic features are computed for all variants.
    _ensure_kinetic_features(pipeline_state, unified_save_path)

    dataset_name = pipeline_state["dataset_name"]
    kinetic_features = pipeline_state["kinetic_features"]
    Y_well = pipeline_state["Y_well"]
    colors = Y_well
    cmap = WELL_CMAP

    # 4. Optional boxplots.
    if save_plot_flag:
        _generate_boxplots(dataset_name, kinetic_features, Y_well, exp_path)

    # 5. Correlation triplets + RF feature importances.
    linear_feature_combinations, important_feature_combinations, _ = _compute_feature_analysis(
        pipeline_state, Y_well, colors, cmap, unified_save_path, save_plot_flag, reports_dir
    )

    # 6. Optional t-SNE / 3D visualisations.
    if save_plot_flag:
        _generate_tsne_3d(dataset_name, important_feature_combinations, kinetic_features,
                          colors, cmap, reports_dir)

    # 7. Outlier detection.
    _run_outlier_pipelines(
        exp_path, pipeline_state, unified_save_path,
        linear_feature_combinations, important_feature_combinations,
        ae_configs=["elbow", 90, 95],
        knn_filter_config=[0.85, 0.90, 0.95],
        spatial_knn_configs=["elbow", 90, 95],
        spatial_grid_configs=["elbow", 90, 95],
        downsample_factor=config.AE_DOWNSAMPLE_FACTOR,
        save_plot_flag=save_plot_flag,
        filters=filters,
    )

    print(f"\nExperiment {exp_path.name} finished gracefully!")
    tf.keras.backend.clear_session()
    gc.collect()


# ====================================================================
# ENTRY POINT
# ====================================================================

if __name__ == "__main__":
    print(f"\n{'='*70}\n[RUNNING] {os.path.basename(__file__)}\n{'='*70}\n")
    parser = argparse.ArgumentParser(description="Outlier Detection Pipeline")
    parser.add_argument("--task_id", type=int, default=0, help="Array Job ID")
    parser.add_argument("--exp_folder", type=str, default=config.DEFAULT_EXP_FOLDER)
    parser.add_argument("--force_rerun", action="store_true",
                        help="Recompute and overwrite even if a presaved unified state already exists")
    parser.add_argument("--fast_mode", action="store_true",
                        help="Disable strict TF determinism (TF_CUDNN_DETERMINISTIC/enable_op_determinism) "
                             "for faster GRU/LSTM/Transformer training. RNG seeds are still set, but reruns "
                             "won't be bit-exact. Only affects this script.")
    parser.add_argument("--filters", nargs="*",
                        choices=sorted(ALL_FILTERS),
                        default=sorted(ALL_FILTERS),
                        help="Which outlier filters to run. Default: all "
                             f"({', '.join(sorted(ALL_FILTERS))}). "
                             "Pass specific names to run only those, or bare --filters for none.")

    args = parser.parse_args()

    set_global_determinism(0, strict=not args.fast_mode)
    
    exp_paths = sorted([
        Path(args.exp_folder, name)
        for name in os.listdir(args.exp_folder)
        if os.path.isdir(os.path.join(args.exp_folder, name))
        and name not in config.EXCLUDED_FOLDERS
    ])

    if args.task_id >= len(exp_paths):
        print(f"Task ID {args.task_id} is out of bounds for {len(exp_paths)} folders. Exiting.")
        sys.exit(0)

    exp_path = exp_paths[args.task_id]

    saved_viz = getattr(config, "SAVED_VIZ", [])
    save_plot_flag = bool(saved_viz) and any(s in str(exp_path) for s in saved_viz)

    run_pipeline(exp_path, force_rerun=args.force_rerun, save_plot_flag=save_plot_flag,
                 filters=set(args.filters))
