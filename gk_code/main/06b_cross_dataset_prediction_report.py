import os
import re
import sys
import argparse
import importlib
from pathlib import Path

import numpy as np
import joblib

from sklearn.metrics import accuracy_score

import config

# Reuse 06's pure rendering/HTML helpers instead of duplicating them -- only the
# data-loading layer differs (flat fold_label -> filter -> res_entry dict + the
# per-fold xai_data snapshot saved by 04, vs 06's single-experiment-folder layout).
_report = importlib.import_module("06_model_prediction_report")
render_overview_chart    = _report.render_overview_chart
render_cm_plot           = _report.render_cm_plot
render_confidence_plot   = _report.render_confidence_plot
render_roc_pr_plot       = _report.render_roc_pr_plot
build_class_metrics_html = _report.build_class_metrics_html
build_tabbed_html        = _report.build_tabbed_html
_panel                   = _report._panel
_buf_to_img_html         = _report._buf_to_img_html
_fig_to_buf              = _report._fig_to_buf

MODEL_KEY_MAP   = config.MODEL_KEY_MAP
MODEL_PRINT_MAP = config.MODEL_PRINT_MAP

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# DISCOVERY
# ============================================================

def get_group_paths(exp_folder):
    """Lists group folders under exp_folder/cross_dataset_cv/ (04's output root).
    Discovered from disk rather than config.CROSS_DATASET_GROUPS, since groups
    already trained on disk can predate config edits (e.g. a group renamed or
    removed from CROSS_DATASET_GROUPS after it was run)."""
    cv_root = Path(exp_folder) / "cross_dataset_cv"
    if not cv_root.exists():
        return []
    return sorted([
        p for p in cv_root.iterdir()
        if p.is_dir() and not p.name.startswith("old_version")
    ])


def get_fold_labels(group_dir, curve_type, mode_str):
    """Fold labels = top-level keys of the 04 results dict for this curve_type/mode."""
    results_path = find_results_path(group_dir, mode_str, curve_type)
    if results_path is None:
        return []
    try:
        lofo_results = joblib.load(results_path)
    except Exception:
        return []
    return sorted(lofo_results.keys())


def find_results_path(group_dir, mode, curve_type):
    """CROSS_DATASET_RESULT_PATH's word order changed at some point in config.py
    (cross_dataset_classification_performances_... vs the older
    classification_performances_cross_dataset_...) and some already-trained
    groups (e.g. init_oneplex_nc_subtract) predate the rename and were never
    regenerated under the new name -- so try both, current convention first."""
    candidates = [
        config.CROSS_DATASET_RESULT_PATH.format(mode=mode, curve_type=curve_type),
        f"classification_performances_cross_dataset_{mode}_{curve_type}.joblib",
    ]
    for name in candidates:
        p = group_dir / name
        if p.exists():
            return p
    return None


def _infer_n_classes(lofo_results, outlier_filter):
    """Older LOFO runs (predating the class_names addition to 04) don't have
    class_names saved -- fall back to the highest label seen across ANY fold's
    cached y_trues_ for this filter (+1), since a single fold's held-out dataset
    may not contain every global class."""
    max_label = -1
    for fold_entry in lofo_results.values():
        res_entry = fold_entry.get(outlier_filter)
        if not res_entry or "y_trues_" not in res_entry:
            continue
        for yt in res_entry["y_trues_"]:
            if len(yt):
                max_label = max(max_label, int(np.max(yt)))
    return max_label + 1


# ============================================================
# FOLD-LOCAL SPLIT RECONSTRUCTION
# ============================================================

def compute_filtered_fold_split(y_test_fold, features_df_test, outlier_filter):
    """Mirrors 06's compute_filtered_splits, but for the single fixed LOFO
    train/test split (the held-out dataset) instead of a re-derived k-fold:
    applies the outlier_filter mask within the already-known test fold. The
    other half of evaluate_outlier_filters' masking -- removing globally-rare
    classes across the full combined pool -- happened once at training time and
    isn't redone here; a mismatch against the cached y_trues_ (checked by the
    caller) catches the rare edge case where that would matter."""
    if outlier_filter is None:
        mask = np.ones(len(y_test_fold), dtype=bool)
    elif outlier_filter in features_df_test.columns:
        mask = (features_df_test[outlier_filter] == 1).fillna(False).values
    else:
        return None
    local_idx = np.where(mask)[0]
    return local_idx, y_test_fold[mask]


# ============================================================
# CURVE PLOT (n_classes × 2 grid -- LOFO pools multiple datasets/wells into one
# held-out fold, so "well" isn't a stable axis here; class fills that role
# instead, mirroring 06's render_curve_plot's n_wells × 2, one-subplot-per-group
# structure exactly, just with class instead of well as the row dimension)
# ============================================================

def render_fold_curve_plot(curves, result, y_true_by_local):
    """curves (snapshot's X_curves_test) is plain 2D (N, T) -- unlike the saved
    .keras models, which require an explicit 3D (batch, T, 1) input, the raw
    curve arrays in curve_for_training.joblib / the xai_data snapshot never
    carry a channel dim, so no squeeze is needed here.

    y_true_by_local: true label indexed by position in `curves` (built by the
    caller from local_idx/y_masked) -- result["y_true_all"] is indexed by
    position-within-the-filtered-subset instead, so it can't be indexed directly
    with correct_idx/wrong_idx (those are positions in `curves`).
    """
    correct_idx = result["correct_idx"]
    wrong_idx   = result["wrong_idx"]
    class_names = result["class_names"]
    n_classes   = len(class_names)

    fig, axes = plt.subplots(n_classes, 2, figsize=(8, 2.5 * n_classes),
                             squeeze=False, sharey="row")

    for cls_idx, cls_name in enumerate(class_names):
        cc = correct_idx[y_true_by_local[correct_idx] == cls_idx]
        ww = wrong_idx[y_true_by_local[wrong_idx] == cls_idx]
        ax_ok, ax_bad = axes[cls_idx, 0], axes[cls_idx, 1]

        if len(cc) > 0:
            ax_ok.plot(curves[cc].T, color="#27ae60", alpha=0.2, linewidth=0.8)
        ax_ok.set_title(f"{cls_name} – Correct (n={len(cc)})", fontsize=9, fontweight="bold")

        if len(ww) > 0:
            ax_bad.plot(curves[ww].T, color="#e74c3c", alpha=0.35, linewidth=0.8)
        ax_bad.set_title(f"{cls_name} – Wrong (n={len(ww)})", fontsize=9, fontweight="bold")

        for ax in (ax_ok, ax_bad):
            ax.set_xlabel("Time index", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)
        ax_ok.set_ylabel("Signal", fontsize=8)

    plt.tight_layout()
    return _fig_to_buf(fig)


# ============================================================
# MAIN PROCESSOR -- one HTML report per (group, fold, curve_type, filter)
# ============================================================

def process_fold(exp_folder, group_dir, fold_label, outlier_filter, curve_type, force_rerun, mode_str):
    group_name = group_dir.name
    out_dir = config.get_viz_dir(Path(exp_folder), "model_performance_viz_cross_dataset_cv") / group_name
    filter_tag = "none" if outlier_filter is None else re.sub(r"[^A-Za-z0-9._-]+", "_", outlier_filter)
    out_path = out_dir / f"{fold_label}__{curve_type}__{filter_tag}.html"

    if out_path.exists() and not force_rerun:
        print(f"  -> [SKIP] Report already exists: {out_path}")
        return

    results_path = find_results_path(group_dir, mode_str, curve_type)
    if results_path is None:
        print(f"  -> [SKIP] {group_name}/{fold_label}: no '{mode_str}' results file for curve_type={curve_type}.")
        return

    snapshot_path = group_dir / "model_interpretation" / fold_label / f"xai_data_{curve_type}.joblib"
    if not snapshot_path.exists():
        print(f"  -> [SKIP] {group_name}/{fold_label}: snapshot not found ({snapshot_path}).")
        return

    print(f"\n{'#'*80}\nCROSS-DATASET MODEL PREDICTION REPORT: {group_name} | {fold_label} "
          f"(curve_type: {curve_type}, filter: {outlier_filter})\n{'#'*80}")

    try:
        lofo_results = joblib.load(results_path)
        snapshot     = joblib.load(snapshot_path)
    except Exception as e:
        print(f"  -> [SKIP] failed to load results/snapshot ({e}).")
        return

    if fold_label not in lofo_results or outlier_filter not in lofo_results[fold_label]:
        print(f"  -> [SKIP] No results for fold={fold_label} filter={outlier_filter}.")
        return
    res_entry = lofo_results[fold_label][outlier_filter]

    curves            = snapshot["X_curves_test"]
    y_test_fold       = snapshot["y_test"]
    features_df_test  = snapshot["features_df_test"]

    split_info = compute_filtered_fold_split(y_test_fold, features_df_test, outlier_filter)
    if split_info is None:
        print(f"  -> [SKIP] Could not reconstruct fold split (filter={outlier_filter}).")
        return
    local_idx, y_masked = split_info

    y_trues_cached = res_entry.get("y_trues_")
    if not y_trues_cached or y_masked.shape != np.asarray(y_trues_cached[0]).shape \
            or not np.array_equal(y_masked, np.asarray(y_trues_cached[0])):
        print(f"  -> [SKIP] {group_name}/{fold_label}: reconstructed fold split doesn't "
              f"match cached y_trues_ (filter={outlier_filter}). Skipping.")
        return

    class_names = lofo_results[fold_label].get("class_names")
    if class_names is None:
        n_classes = _infer_n_classes(lofo_results, outlier_filter)
        class_names = [f"Class {i}" for i in range(n_classes)]
        print(f"  -> [!] No saved class_names for this group (predates the 04 fix) -- "
              f"showing integer class IDs instead of label names.")

    # ---- Collect per-model data (single fold -> single "split") ----
    model_results = []
    for m, (preds_key, probs_key, _) in MODEL_KEY_MAP.items():
        if preds_key not in res_entry:
            continue

        preds_list = res_entry[preds_key]
        probs_list = res_entry.get(probs_key)
        if not preds_list:
            continue

        y_pred = np.asarray(preds_list[0])
        if y_pred.shape != y_masked.shape:
            print(f"  -> [WARNING] {m}: shape mismatch vs reconstructed fold. Skipping.")
            continue

        y_prob_all = None
        if probs_list:
            prob0 = np.asarray(probs_list[0])
            if prob0.ndim == 2:
                y_prob_all = prob0

        mask = y_masked == y_pred
        model_results.append({
            "key":         m,
            "name":        MODEL_PRINT_MAP.get(m, m),
            "acc":         accuracy_score(y_masked, y_pred) * 100,
            "std":         0.0,
            "fold_accs":   [accuracy_score(y_masked, y_pred)],
            "correct_idx": local_idx[mask],
            "wrong_idx":   local_idx[~mask],
            "y_true_all":  y_masked,
            "y_pred_all":  y_pred,
            "y_prob_all":  y_prob_all,
            "class_names": class_names,
        })
        print(f"  -> Processed: {MODEL_PRINT_MAP.get(m, m)} (acc={model_results[-1]['acc']:.1f}%)")

    if not model_results:
        print(f"  -> No model results found for {group_name}/{fold_label}. Skipping.")
        return

    # ---- Build tabs (mirrors 06_model_prediction_report's structure, minus
    # the per-fold-stability tab -- this report IS a single LOFO fold, fold
    # stability ACROSS folds belongs in the comparison notebook instead) ----
    tabs = []

    overview_content = ""
    buf = render_overview_chart(model_results)
    if buf:
        overview_content += _panel(
            "Model Accuracy Comparison",
            _buf_to_img_html(buf, style="height:auto;max-width:750px;"))
    tabs.append(("overview", "Overview", f'<div class="panel-row">{overview_content}</div>'))

    # y_true_by_local: true label indexed by position in `curves` -- see
    # render_fold_curve_plot's docstring for why result["y_true_all"] alone isn't usable.
    y_true_by_local = np.full(len(curves), -1, dtype=y_masked.dtype)
    y_true_by_local[local_idx] = y_masked
    curves_content = "".join(
        _panel(
            f'{r["name"]}<br><span style="font-size:11px;color:#666;">Acc: {r["acc"]:.1f}%</span>',
            _buf_to_img_html(render_fold_curve_plot(curves, r, y_true_by_local), style="height:auto;max-width:500px;")
        )
        for r in model_results
    )
    tabs.append(("curves", "Curves", f'<div class="panel-row">{curves_content}</div>'))

    cm_content = "".join(
        _panel(
            f'{r["name"]}<br><span style="font-size:11px;color:#666;">Acc: {r["acc"]:.1f}%</span>',
            _buf_to_img_html(render_cm_plot(r), style="height:auto;max-width:480px;")
        )
        for r in model_results
    )
    tabs.append(("cm", "Confusion Matrix", f'<div class="panel-row">{cm_content}</div>'))

    metrics_content = "".join(build_class_metrics_html(r) for r in model_results)
    tabs.append(("metrics", "Class Metrics", metrics_content))

    conf_content = ""
    for r in model_results:
        buf = render_confidence_plot(r)
        if buf:
            conf_content += _panel(r["name"], _buf_to_img_html(buf, style="height:auto;max-width:480px;"))
    if conf_content:
        tabs.append(("confidence", "Confidence", f'<div class="panel-row">{conf_content}</div>'))

    roc_content = ""
    for r in model_results:
        buf = render_roc_pr_plot(r)
        if buf:
            roc_content += _panel(r["name"], _buf_to_img_html(buf, style="height:auto;max-width:700px;"))
    if roc_content:
        tabs.append(("roc", "ROC / PR", f'<div class="panel-row">{roc_content}</div>'))

    os.makedirs(out_dir, exist_ok=True)
    filter_label = outlier_filter if outlier_filter else "None (Baseline)"
    mode_label = {"lofo": "LOFO", "random_split": "Random Split"}.get(mode_str, mode_str)
    title = (f"Cross-Dataset {mode_label} Report: {group_name} | Fold: {fold_label} "
             f"| Curve: {curve_type} | Filter: {filter_label}")
    build_tabbed_html(title, tabs, out_path)
    print(f"  -> [SAVED] {out_path}")


if __name__ == "__main__":
    print(f"\n{'='*70}\n[RUNNING] {os.path.basename(__file__)}\n{'='*70}\n")
    parser = argparse.ArgumentParser(description="Cross-Dataset LOFO Prediction Visualization Report")
    parser.add_argument("--task_id", type=int, default=None,
                        help="Array Job ID -- index into the flattened list of (group, fold) pairs (default: run all)")
    parser.add_argument("--exp_folder", type=str, default=config.DEFAULT_EXP_FOLDER)
    parser.add_argument("--group", type=str, default=None,
                        help="Restrict to a single group name under cross_dataset_cv/ (default: all discovered groups)")
    parser.add_argument("--force_rerun", action="store_true")
    parser.add_argument("--outlier_filter", type=str, nargs="*",
                        default=[None, "lstm_ae_glb_ds1_label_elbow",
                                 "spatial_knn_label_elbow", "spatial_grid_label_elbow"],
                        help="Outlier filter(s) to visualize. Pass 'None' for baseline.")
    parser.add_argument("--curve_type", type=str, nargs="+", default=["ori_curve", "ori_curve_avg"])
    parser.add_argument("--mode", type=str, choices=["lofo", "random_split", "kfold"],
                        default="lofo",
                        help="Which 04_cross_dataset_training.py --mode's results to report on: "
                             "lofo=leave-one-folder-out, random_split=stratified single split, "
                             "kfold=stratified N-fold.")
    parser.add_argument("--n_splits", type=int, default=5,
                        help="Must match the --n_splits used for the corresponding 04 kfold run "
                             "(only used when --mode kfold).")
    args = parser.parse_args()

    outlier_filters = [None if f == "None" else f for f in args.outlier_filter]
    mode_str = args.mode if args.mode != "kfold" else f"kfold{args.n_splits}"

    group_paths = get_group_paths(args.exp_folder)
    if args.group is not None:
        group_paths = [p for p in group_paths if p.name == args.group]

    # Flatten (group, fold) pairs across curve_types present for each group, so
    # --task_id matches the array-job convention used by 03/04/06/07.
    pairs = []
    for group_dir in group_paths:
        for curve_type in args.curve_type:
            for fold_label in get_fold_labels(group_dir, curve_type, mode_str):
                pairs.append((group_dir, fold_label, curve_type))

    if not pairs:
        print(f"No cross-dataset '{mode_str}' results found under {args.exp_folder}/cross_dataset_cv/. Exiting.")
        sys.exit(0)

    if args.task_id is not None:
        if args.task_id >= len(pairs):
            print(f"Task ID {args.task_id} is out of bounds for {len(pairs)} (group, fold, curve_type) combos. Exiting.")
            sys.exit(0)
        pairs = [pairs[args.task_id]]

    for group_dir, fold_label, curve_type in pairs:
        for outlier_filter in outlier_filters:
            process_fold(args.exp_folder, group_dir, fold_label, outlier_filter, curve_type,
                         args.force_rerun, mode_str)
