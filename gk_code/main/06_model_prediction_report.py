import os
import re
import sys
import argparse
from pathlib import Path
sys.path.insert(0, 'utils')
sys.path.insert(0, 'utils/model_training')
from html_utils import _fig_to_buf, _buf_to_img_html, _panel, build_tabbed_html
from pipeline_utils import get_exp_paths, check_task_id
from model_utils_mtl import REG_SENTINEL as _MTL_REG_SENTINEL
from model_utils import build_well_stratified_random_split, build_well_stratified_nfold_splits

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_recall_fscore_support,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)

import config
from scipy import stats as _scipy_stats

# Moved to config.py (shared with 08_statistical_comparison.py).
MODEL_KEY_MAP = config.MODEL_KEY_MAP
MODEL_PRINT_MAP = config.MODEL_PRINT_MAP




def compute_filtered_splits(y_full, features_df, outlier_filter, n_splits, well_ids=None):
    """Recreate the deterministic mask + rare-class filter + split used by evaluate_outlier_filters
    so cached predictions/y_trues_ can be mapped back to original curve indices.

    well_ids (if given) makes this stratified by well, mirroring evaluate_outlier_filters'
    well-stratified default (see model_utils.build_well_stratified_random_split) -- without it,
    a reconstructed split for any run that was actually trained with well-stratified splitting
    would silently diverge from what the model actually saw."""
    if outlier_filter is None:
        mask = np.ones(len(y_full), dtype=bool)
    elif outlier_filter in features_df.columns:
        mask = (features_df[outlier_filter] == 1).fillna(False).values
    else:
        return None

    global_idx = np.where(mask)[0]
    y_masked = y_full[mask]
    well_ids_masked = well_ids[mask] if well_ids is not None else None

    unique_classes, class_counts = np.unique(y_masked, return_counts=True)
    rare_classes = unique_classes[class_counts < 2]
    if len(rare_classes) > 0:
        valid = ~np.isin(y_masked, rare_classes)
        global_idx = global_idx[valid]
        y_masked = y_masked[valid]
        if well_ids_masked is not None:
            well_ids_masked = well_ids_masked[valid]

    n_classes = len(np.unique(y_masked))
    if n_classes < 2 or len(y_masked) < 2 * n_classes:
        return None

    if well_ids_masked is not None:
        # Sized/clamped by well count (the stratify key), not label count -- see
        # model_utils.evaluate_outlier_filters' matching well_ids_m branch.
        n_wells = len(np.unique(well_ids_masked))
        calculated_test_size = max(len(y_masked) * 0.10, n_wells) / len(y_masked)
        if n_splits == 1:
            splits = list(build_well_stratified_random_split(
                y_masked, well_ids_masked, test_size=calculated_test_size).values())
        else:
            actual_splits = min(n_splits, n_wells)
            splits = list(build_well_stratified_nfold_splits(
                y_masked, well_ids_masked, n_splits=actual_splits).values())
    else:
        calculated_test_size = max(int(len(y_masked) * 0.10), n_classes)
        if n_splits == 1:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=calculated_test_size, random_state=0)
        else:
            min_class_count = np.min(class_counts[~np.isin(unique_classes, rare_classes)])
            actual_splits = min(n_splits, min_class_count)
            splitter = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=0)

        splits = list(splitter.split(np.zeros((len(y_masked), 1)), y_masked))

    return global_idx, y_masked, splits


# ============================================================
# RENDERING FUNCTIONS
# ============================================================

def render_overview_chart(model_results):
    """Horizontal bar chart of all models sorted by accuracy with ± std error bars."""
    if not model_results:
        return None
    sorted_r = sorted(model_results, key=lambda r: r["acc"])
    names = [r["name"] for r in sorted_r]
    accs  = [r["acc"]  for r in sorted_r]
    stds  = [r["std"]  for r in sorted_r]

    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.55 * len(names))))
    colors = ["#27ae60" if a >= 80 else "#e67e22" if a >= 60 else "#e74c3c" for a in accs]
    bars = ax.barh(names, accs, xerr=stds, color=colors, alpha=0.85, capsize=4,
                   error_kw={"elinewidth": 1.5})
    max_std = max(stds) if stds else 0
    ax.set_xlabel("Accuracy (%)", fontsize=10)
    ax.set_xlim(0, 113)
    for bar, acc, std in zip(bars, accs, stds):
        ax.text(bar.get_width() + max_std + 1,
                bar.get_y() + bar.get_height() / 2,
                f"{acc:.1f} ± {std:.1f}%", va="center", fontsize=8)
    ax.set_title("Model Accuracy Comparison", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return _fig_to_buf(fig)


def render_fold_stability_plot(model_results):
    """Strip chart of per-fold accuracies (mean shown as thick bar). Only rendered when n_splits > 1."""
    multi = [r for r in model_results if len(r["fold_accs"]) > 1]
    if not multi:
        return None

    names = [r["name"] for r in multi]
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(names)), 4))
    rng = np.random.default_rng(0)
    for i, r in enumerate(multi):
        fold_pct = np.array(r["fold_accs"]) * 100
        jitter = rng.uniform(-0.15, 0.15, len(fold_pct))
        ax.scatter(np.full(len(fold_pct), i) + jitter, fold_pct,
                   alpha=0.7, s=45, zorder=3, color="#3498db")
        ax.plot([i - 0.3, i + 0.3], [r["acc"], r["acc"]],
                color="#2c3e50", linewidth=2.5, zorder=4)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy per fold (%)", fontsize=10)
    ax.set_title("Per-Fold Accuracy Stability", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return _fig_to_buf(fig)


def render_curve_plot(curves, Y_well_raw, label_map, result):
    """n_wells × 2 grid: correctly predicted (green) | wrongly predicted (red)."""
    unique_wells = np.unique(Y_well_raw)
    n_wells = len(unique_wells)
    correct_idx = result["correct_idx"]
    wrong_idx   = result["wrong_idx"]

    fig, axes = plt.subplots(n_wells, 2, figsize=(8, 2.5 * n_wells),
                             squeeze=False, sharey="row")
    correct_wells = Y_well_raw[correct_idx]
    wrong_wells   = Y_well_raw[wrong_idx]

    for i, well in enumerate(unique_wells):
        well_title = f"Well {well}"
        try:
            wk = int(well)
        except (TypeError, ValueError):
            wk = well
        if wk in label_map:
            well_title += f" ({label_map[wk]})"

        wc = correct_idx[correct_wells == well]
        ww = wrong_idx[wrong_wells == well]
        ax_ok, ax_bad = axes[i, 0], axes[i, 1]

        if len(wc) > 0:
            ax_ok.plot(curves[wc].T, color="#27ae60", alpha=0.2, linewidth=0.8)
        ax_ok.set_title(f"{well_title} – Correct (n={len(wc)})", fontsize=9, fontweight="bold")

        if len(ww) > 0:
            ax_bad.plot(curves[ww].T, color="#e74c3c", alpha=0.4, linewidth=0.8)
        ax_bad.set_title(f"{well_title} – Wrong (n={len(ww)})", fontsize=9, fontweight="bold")

        for ax in (ax_ok, ax_bad):
            ax.set_xlabel("Time index", fontsize=8)
            ax.set_ylabel("Signal", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)

    plt.tight_layout()
    return _fig_to_buf(fig)


def render_cm_plot(result):
    """Confusion matrix heatmap: raw counts + row-normalised percentage per cell."""
    y_true = result["y_true_all"]
    y_pred = result["y_pred_all"]
    class_names = result["class_names"]
    n_classes = len(class_names)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(max(4, n_classes * 1.4), max(3.5, n_classes * 1.1)))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for r in range(n_classes):
        for c in range(n_classes):
            ax.text(c, r, f"{cm[r, c]}\n({cm_norm[r, c]*100:.0f}%)",
                    ha="center", va="center", fontsize=8,
                    color="white" if cm_norm[r, c] > 0.5 else "black")

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    plt.tight_layout()
    return _fig_to_buf(fig)


def render_confidence_plot(result):
    """Histogram of max softmax confidence: correct predictions (green) vs wrong (red)."""
    y_prob = result["y_prob_all"]
    if y_prob is None or y_prob.ndim != 2 or y_prob.shape[1] < 2:
        return None

    y_true = result["y_true_all"]
    y_pred = result["y_pred_all"]
    max_conf = y_prob.max(axis=1)
    correct_mask = y_true == y_pred
    n_correct = int(correct_mask.sum())
    n_wrong   = int((~correct_mask).sum())

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bins = np.linspace(0, 1, 21)
    if n_correct:
        ax.hist(max_conf[correct_mask], bins=bins, alpha=0.6, color="#27ae60",
                label=f"Correct (n={n_correct})", density=True)
        ax.axvline(max_conf[correct_mask].mean(), color="#1e8449",
                   linestyle="--", linewidth=1.5,
                   label=f"Correct mean: {max_conf[correct_mask].mean():.2f}")
    if n_wrong:
        ax.hist(max_conf[~correct_mask], bins=bins, alpha=0.6, color="#e74c3c",
                label=f"Wrong (n={n_wrong})", density=True)
        ax.axvline(max_conf[~correct_mask].mean(), color="#922b21",
                   linestyle="--", linewidth=1.5,
                   label=f"Wrong mean: {max_conf[~correct_mask].mean():.2f}")

    ax.set_xlabel("Max predicted probability", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title("Prediction Confidence Distribution", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return _fig_to_buf(fig)


def render_roc_pr_plot(result):
    """ROC (left) and Precision-Recall (right) curves, one line per class (one-vs-rest)."""
    y_prob = result["y_prob_all"]
    if y_prob is None or y_prob.ndim != 2 or y_prob.shape[1] < 2:
        return None

    y_true      = result["y_true_all"]
    class_names = result["class_names"]
    n_classes   = len(class_names)

    if y_prob.shape[1] != n_classes:
        return None

    if n_classes == 2:
        y_bin_list = [y_true]
        prob_list  = [y_prob[:, 1]]
        labels     = [class_names[1]]
    else:
        y_binarized = label_binarize(y_true, classes=list(range(n_classes)))
        y_bin_list  = [y_binarized[:, i] for i in range(n_classes)]
        prob_list   = [y_prob[:, i]      for i in range(n_classes)]
        labels      = class_names

    cmap = plt.cm.tab10(np.linspace(0, 0.9, len(labels)))
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(10, 4))

    for y_bin_i, prob_i, label, color in zip(y_bin_list, prob_list, labels, cmap):
        if len(np.unique(y_bin_i)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_bin_i, prob_i)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color=color, linewidth=1.5, label=f"{label} (AUC={roc_auc:.2f})")

        prec, rec, _ = precision_recall_curve(y_bin_i, prob_i)
        ap = average_precision_score(y_bin_i, prob_i)
        ax_pr.plot(rec, prec, color=color, linewidth=1.5, label=f"{label} (AP={ap:.2f})")

    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax_roc.set_xlabel("False Positive Rate", fontsize=9)
    ax_roc.set_ylabel("True Positive Rate", fontsize=9)
    ax_roc.set_title("ROC Curve (OvR)", fontsize=10, fontweight="bold")
    ax_roc.legend(fontsize=7.5)
    ax_roc.grid(alpha=0.3)

    ax_pr.set_xlabel("Recall", fontsize=9)
    ax_pr.set_ylabel("Precision", fontsize=9)
    ax_pr.set_title("Precision-Recall Curve (OvR)", fontsize=10, fontweight="bold")
    ax_pr.legend(fontsize=7.5)
    ax_pr.grid(alpha=0.3)

    plt.tight_layout()
    return _fig_to_buf(fig)


# ============================================================
# CLASS METRICS HTML TABLE
# ============================================================

def build_class_metrics_html(result):
    """HTML table: per-class N / Precision / Recall / F1, plus macro and weighted totals."""
    y_true      = result["y_true_all"]
    y_pred      = result["y_pred_all"]
    class_names = result["class_names"]
    n_classes   = len(class_names)
    labels      = list(range(n_classes))

    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)

    cell_base = "padding:6px 14px;border:1px solid #ddd;text-align:center;"

    def _th(cells):
        return "<tr>" + "".join(
            f"<th style='{cell_base}background:#2c3e50;color:white;'>{c}</th>"
            for c in cells) + "</tr>"

    def _td(cells, bg="white", bold=False):
        fw = "font-weight:bold;" if bold else ""
        return "<tr>" + "".join(
            f"<td style='{cell_base}background:{bg};{fw}'>{c}</td>"
            for c in cells) + "</tr>"

    header = _th(["Class", "N (support)", "Precision", "Recall", "F1"])
    body   = ""
    for i, name in enumerate(class_names):
        bg = "#f2f3f4" if i % 2 == 0 else "white"
        body += _td([name, int(sup[i]),
                     f"{prec[i]:.3f}", f"{rec[i]:.3f}", f"{f1[i]:.3f}"], bg=bg)
    total_n = int(sup.sum())
    body += _td(["Macro avg",    total_n, f"{prec_m:.3f}", f"{rec_m:.3f}", f"{f1_m:.3f}"],
                bg="#d6eaf8", bold=True)
    body += _td(["Weighted avg", total_n, f"{prec_w:.3f}", f"{rec_w:.3f}", f"{f1_w:.3f}"],
                bg="#d6eaf8", bold=True)

    return (
        f'<div style="margin-bottom:28px;">'
        f'<h3 style="margin:0 0 6px 0;font-size:14px;">{result["name"]}'
        f'  <span style="font-size:12px;color:#666;font-weight:normal;">'
        f'Acc: {result["acc"]:.1f}% ± {result["std"]:.1f}%</span></h3>'
        f'<table style="border-collapse:collapse;font-size:13px;">'
        f'<thead>{header}</thead>'
        f'<tbody>{body}</tbody>'
        f'</table></div>'
    )


# ============================================================
# MTL REGRESSION HELPERS
# ============================================================

def _mtl_regression_metrics(reg_preds_list, reg_trues_list):
    """Compute per-fold and aggregated regression metrics; mask out sentinel values."""
    fold_rows = []
    for fold_i, (preds, trues) in enumerate(zip(reg_preds_list, reg_trues_list)):
        preds = np.asarray(preds, dtype=float)
        trues = np.asarray(trues, dtype=float)
        valid = trues != _MTL_REG_SENTINEL
        n_valid = int(valid.sum())
        if n_valid < 2:
            fold_rows.append({"fold": fold_i, "n_valid": n_valid,
                               "mae": np.nan, "rmse": np.nan, "r2": np.nan, "pearson_r": np.nan})
            continue
        p, t = preds[valid], trues[valid]
        mae  = float(np.mean(np.abs(p - t)))
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        pr, _ = _scipy_stats.pearsonr(t, p)
        fold_rows.append({"fold": fold_i, "n_valid": n_valid,
                          "mae": mae, "rmse": rmse, "r2": r2, "pearson_r": float(pr)})
    metrics_by_fold = fold_rows
    valid_rows = [r for r in fold_rows if np.isfinite(r["mae"])]
    agg = {}
    for k in ("mae", "rmse", "r2", "pearson_r"):
        vals = [r[k] for r in valid_rows]
        agg[k + "_mean"] = float(np.mean(vals)) if vals else float("nan")
        agg[k + "_std"]  = float(np.std(vals))  if vals else float("nan")
    return metrics_by_fold, agg


def render_mtl_scatter(mtl_reg_list, class_names):
    """Predicted vs actual scatter (all folds concatenated), colored by class."""
    if not mtl_reg_list:
        return None
    n_models = len(mtl_reg_list)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5), squeeze=False)
    cmap = plt.cm.tab10

    for ax, rd in zip(axes[0], mtl_reg_list):
        all_preds, all_trues, all_cls = [], [], []
        for fold_i, (preds, trues) in enumerate(zip(rd["reg_preds"], rd["reg_trues"])):
            preds = np.asarray(preds, dtype=float)
            trues = np.asarray(trues, dtype=float)
            valid = trues != _MTL_REG_SENTINEL
            if valid.sum() == 0:
                continue
            all_preds.append(preds[valid])
            all_trues.append(trues[valid])
            cls_fold = np.asarray(rd.get("y_trues_folds", [[]] * (fold_i + 1))[fold_i])
            all_cls.append(cls_fold[valid] if len(cls_fold) == len(preds) else np.zeros(valid.sum(), dtype=int))
        if not all_preds:
            ax.set_visible(False)
            continue
        P = np.concatenate(all_preds)
        T = np.concatenate(all_trues)
        C = np.concatenate(all_cls)

        for ci in np.unique(C):
            mask = C == ci
            label = class_names[int(ci)] if int(ci) < len(class_names) else str(ci)
            ax.scatter(T[mask], P[mask], c=[cmap(int(ci) / max(len(class_names), 1))],
                       alpha=0.5, s=20, label=label)
        lo, hi = min(T.min(), P.min()), max(T.max(), P.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        _, agg = _mtl_regression_metrics(rd["reg_preds"], rd["reg_trues"])
        ax.set_title(f"{rd['name']}\nR²={agg['r2_mean']:.3f}  MAE={agg['mae_mean']:.3f}", fontsize=9)
        ax.set_xlabel("True concentration", fontsize=8)
        ax.set_ylabel("Predicted concentration", fontsize=8)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    return _fig_to_buf(fig)


def render_mtl_metrics_table(mtl_reg_list):
    """HTML table: per-model regression metrics (mean ± std across folds)."""
    cell = "padding:6px 12px;border:1px solid #ddd;text-align:center;"
    th_s = f"background:#2c3e50;color:white;{cell}"
    header = "".join(f"<th style='{th_s}'>{c}</th>"
                     for c in ["Model", "Folds", "n valid (mean)", "MAE", "RMSE", "R²", "Pearson r"])
    rows_html = ""
    for i, rd in enumerate(mtl_reg_list):
        fold_rows, agg = _mtl_regression_metrics(rd["reg_preds"], rd["reg_trues"])
        n_valid_mean = np.mean([r["n_valid"] for r in fold_rows])
        bg = "#f2f3f4" if i % 2 == 0 else "white"
        def _fmt(m, s): return f"{m:.3f} ± {s:.3f}" if np.isfinite(m) else "—"
        cells = [
            rd["name"], len(rd["reg_preds"]), f"{n_valid_mean:.0f}",
            _fmt(agg["mae_mean"], agg["mae_std"]),
            _fmt(agg["rmse_mean"], agg["rmse_std"]),
            _fmt(agg["r2_mean"], agg["r2_std"]),
            _fmt(agg["pearson_r_mean"], agg["pearson_r_std"]),
        ]
        rows_html += "<tr>" + "".join(f"<td style='{cell}background:{bg};'>{c}</td>" for c in cells) + "</tr>"
    return (f'<table style="border-collapse:collapse;font-size:13px;">'
            f'<thead><tr>{header}</tr></thead>'
            f'<tbody>{rows_html}</tbody></table>')


# ============================================================
# MAIN EXPERIMENT PROCESSOR
# ============================================================

def process_experiment(exp_path, mode, outlier_filter, n_splits, force_rerun, curve_type="ori_curve"):
    out_dir = config.get_viz_dir(exp_path.parent, "model_performance_viz")

    mode_key   = mode.strip().title()
    filter_tag = "none" if outlier_filter is None else re.sub(r"[^A-Za-z0-9._-]+", "_", outlier_filter)
    out_path   = out_dir / f"{exp_path.name}__{curve_type}__{mode_key}__{filter_tag}__nsplits{n_splits}.html"

    if out_path.exists() and not force_rerun:
        print(f"  -> [SKIP] Report already exists: {out_path}")
        return

    results_path = os.path.join(
        exp_path, config.TRAINING_10FOLD_RESULT_PATH if n_splits > 1 else config.TRAINING_RESULT_PATH)
    state_path = os.path.join(exp_path, config.TRAINING_DATA_PATH)

    if not os.path.exists(results_path):
        print(f"  -> Skipping {exp_path.name}: '{os.path.basename(results_path)}' not found.")
        return
    if not os.path.exists(state_path):
        print(f"  -> Skipping {exp_path.name}: '{config.TRAINING_DATA_PATH}' not found.")
        return

    print(f"\n{'#'*80}\nGENERATING MODEL PREDICTION REPORT FOR: {exp_path.name} (curve_type: {curve_type})\n{'#'*80}")

    try:
        all_ml_results = joblib.load(results_path)
        state          = joblib.load(state_path)
    except Exception as e:
        print(f"  -> Skipping {exp_path.name}: failed to load results/state ({e}).")
        return

    dataset_name      = list(state["dataset_name"])
    dataset           = state["dataset"]
    Y_well_raw        = np.asarray(state["Y_well"])
    kinetic_features  = state["kinetic_features"]

    # well_ids: same derivation as 03_main_training.py, needed so compute_filtered_splits
    # can reconstruct a well-stratified split matching what evaluate_outlier_filters actually
    # trained on (see model_utils.build_well_stratified_random_split).
    well_ids_full = None
    if "metadata" in state:
        metadata_df = pd.DataFrame(state["metadata"])
        if {"pixel_row_idx", "pixel_col_idx"}.issubset(metadata_df.columns):
            well_ids_full = (metadata_df["well_id"].values if "well_id" in metadata_df.columns
                              else Y_well_raw.copy())

    try:
        curve_idx, resolved_name = config.resolve_curve_dataset_idx(curve_type, dataset_name)
    except ValueError as e:
        print(f"  -> [SKIP] {e}")
        return
    curves      = dataset[curve_idx]
    features_df = kinetic_features[curve_idx]

    label_mappings = config.get_label_mappings(exp_path)
    Y_well_mapped = list(Y_well_raw)
    if exp_path.name in label_mappings:
        mapping = label_mappings[exp_path.name]
        Y_well_mapped = [mapping.get(w, w) for w in Y_well_raw]

    encoder = LabelEncoder()
    y_full  = encoder.fit_transform(Y_well_mapped)

    clean_title = resolved_name.replace("_", " ").title()

    if clean_title not in all_ml_results or mode_key not in all_ml_results[clean_title]:
        print(f"  -> No '{mode_key}' results found for {exp_path.name}. Skipping.")
        return

    results_dict = all_ml_results[clean_title][mode_key]
    if outlier_filter not in results_dict:
        print(f"  -> Filter '{outlier_filter}' not found in '{mode_key}' results. Skipping.")
        return

    res_entry  = results_dict[outlier_filter]
    split_info = compute_filtered_splits(y_full, features_df, outlier_filter, n_splits,
                                          well_ids=well_ids_full)
    if split_info is None:
        print(f"  -> Could not reconstruct splits (filter={outlier_filter}). Skipping.")
        return
    global_idx, y_masked, splits = split_info

    label_map   = label_mappings.get(exp_path.name, {})
    class_names = [str(c) for c in encoder.classes_]

    # ---- Collect per-model data ----
    model_results = []
    for m, (preds_key, probs_key, _) in MODEL_KEY_MAP.items():
        if preds_key not in res_entry:
            continue

        y_trues_list = res_entry["y_trues_"]
        preds_list   = res_entry[preds_key]
        probs_list   = res_entry.get(probs_key)

        correct_idx_list, wrong_idx_list = [], []
        y_true_list, y_pred_list, y_prob_list = [], [], []
        fold_accs, mismatch = [], False

        for fold_i, (_, test_idx_local) in enumerate(splits):
            if fold_i >= len(preds_list) or fold_i >= len(y_trues_list):
                break

            y_true_fold   = y_masked[test_idx_local]
            y_pred_fold   = np.asarray(preds_list[fold_i])
            y_true_cached = np.asarray(y_trues_list[fold_i])

            if y_true_fold.shape != y_true_cached.shape or not np.array_equal(y_true_fold, y_true_cached):
                mismatch = True
                break

            fold_accs.append(accuracy_score(y_true_fold, y_pred_fold))
            y_true_list.append(y_true_fold)
            y_pred_list.append(y_pred_fold)

            if probs_list is not None and fold_i < len(probs_list):
                prob_fold = np.asarray(probs_list[fold_i])
                if prob_fold.ndim == 2:
                    y_prob_list.append(prob_fold)

            gidx = global_idx[test_idx_local]
            mask = y_true_fold == y_pred_fold
            correct_idx_list.append(gidx[mask])
            wrong_idx_list.append(gidx[~mask])

        if mismatch or not fold_accs:
            print(f"  -> [WARNING] {m}: split mismatch (try a different --n_splits). Skipping.")
            continue

        y_prob_all = np.concatenate(y_prob_list) if y_prob_list else None
        model_results.append({
            "key":         m,
            "name":        MODEL_PRINT_MAP.get(m, m),
            "acc":         np.mean(fold_accs) * 100,
            "std":         np.std(fold_accs) * 100,
            "fold_accs":   fold_accs,
            "correct_idx": np.concatenate(correct_idx_list),
            "wrong_idx":   np.concatenate(wrong_idx_list),
            "y_true_all":  np.concatenate(y_true_list),
            "y_pred_all":  np.concatenate(y_pred_list),
            "y_prob_all":  y_prob_all,
            "class_names": class_names,
        })
        print(f"  -> Processed: {MODEL_PRINT_MAP.get(m, m)} (acc={np.mean(fold_accs)*100:.1f}%)")

    if not model_results:
        print(f"  -> No model results found for {exp_path.name}. Skipping.")
        return

    # ---- Build tab contents ----
    tabs = []

    # Tab 1: Overview — bar chart + fold stability (if k-fold)
    overview_content = ""
    buf = render_overview_chart(model_results)
    if buf:
        overview_content += _panel(
            "Model Accuracy Comparison",
            _buf_to_img_html(buf, style="height:auto;max-width:750px;"))
    buf = render_fold_stability_plot(model_results)
    if buf:
        overview_content += _panel(
            "Per-Fold Stability",
            _buf_to_img_html(buf, style="height:auto;max-width:750px;"))
    tabs.append(("overview", "Overview", f'<div class="panel-row">{overview_content}</div>'))

    # Tab 2: Curves — correct / wrong per well, one panel per model
    curves_content = "".join(
        _panel(
            f'{r["name"]}'
            f'<br><span style="font-size:11px;color:#666;">Acc: {r["acc"]:.1f}% ± {r["std"]:.1f}%</span>',
            _buf_to_img_html(render_curve_plot(curves, Y_well_raw, label_map, r),
                             style="height:auto;max-width:600px;")
        )
        for r in model_results
    )
    tabs.append(("curves", "Curves", f'<div class="panel-row">{curves_content}</div>'))

    # Tab 3: Confusion Matrix — one panel per model
    cm_content = "".join(
        _panel(
            f'{r["name"]}'
            f'<br><span style="font-size:11px;color:#666;">Acc: {r["acc"]:.1f}%</span>',
            _buf_to_img_html(render_cm_plot(r), style="height:auto;max-width:480px;")
        )
        for r in model_results
    )
    tabs.append(("cm", "Confusion Matrix", f'<div class="panel-row">{cm_content}</div>'))

    # Tab 4: Class Metrics — HTML tables (precision / recall / F1 / support + totals)
    metrics_content = "".join(build_class_metrics_html(r) for r in model_results)
    tabs.append(("metrics", "Class Metrics", metrics_content))

    # Tab 5: Confidence — max softmax histogram per model
    conf_content = ""
    for r in model_results:
        buf = render_confidence_plot(r)
        if buf:
            conf_content += _panel(r["name"],
                                   _buf_to_img_html(buf, style="height:auto;max-width:480px;"))
    if conf_content:
        tabs.append(("confidence", "Confidence", f'<div class="panel-row">{conf_content}</div>'))

    # Tab 6: ROC / PR — OvR curves per model
    roc_content = ""
    for r in model_results:
        buf = render_roc_pr_plot(r)
        if buf:
            roc_content += _panel(r["name"],
                                  _buf_to_img_html(buf, style="height:auto;max-width:700px;"))
    if roc_content:
        tabs.append(("roc", "ROC / PR", f'<div class="panel-row">{roc_content}</div>'))

    # Tab 7: Regression — MTL concentration prediction (auto-detected by key presence)
    mtl_reg_list = []
    for r in model_results:
        mk = r["key"]
        reg_preds_key = f'y_reg_preds_{mk}_'
        reg_trues_key = f'y_reg_trues_{mk}_'
        if reg_preds_key in res_entry:
            entry = {
                "name": r["name"], "key": mk,
                "reg_preds": res_entry[reg_preds_key],
                "reg_trues": res_entry[reg_trues_key],
                "y_trues_folds": res_entry.get("y_trues_", []),
            }
            mtl_reg_list.append(entry)
    if mtl_reg_list:
        reg_content = render_mtl_metrics_table(mtl_reg_list)
        buf = render_mtl_scatter(mtl_reg_list, class_names)
        if buf:
            reg_content += _panel("Predicted vs Actual Concentration",
                                   _buf_to_img_html(buf, style="height:auto;max-width:100%;"))
        tabs.append(("regression", "Regression (MTL)", reg_content))

    # ---- Write output ----
    os.makedirs(out_dir, exist_ok=True)
    filter_label = outlier_filter if outlier_filter else "None (Baseline)"
    title = (f"Model Report: {exp_path.name} | Curve: {curve_type} "
             f"| Mode: {mode_key} | Filter: {filter_label}")
    build_tabbed_html(title, tabs, out_path)
    print(f"  -> [SAVED] {out_path}")


if __name__ == "__main__":
    print(f"\n{'='*70}\n[RUNNING] {os.path.basename(__file__)}\n{'='*70}\n")
    parser = argparse.ArgumentParser(description="Static Model Prediction Visualization Report")
    parser.add_argument("--task_id", type=int, default=0, help="Array Job ID")
    parser.add_argument("--exp_folder", type=str, default=config.DEFAULT_EXP_FOLDER)
    parser.add_argument("--force_rerun", action="store_true",
                        help="Regenerate the HTML reports even if they already exist")
    parser.add_argument("--mode", type=str, default="Native", help="'Native' or 'Reference'")
    parser.add_argument("--outlier_filter", type=str, nargs="*",
                        default=config.OUTLIER_FILTERS,
                        help="Outlier filter(s) to visualize. Pass 'None' for baseline. Default: config.OUTLIER_FILTERS.")
    parser.add_argument("--n_splits", type=int, default=1,
                        help="Must match the --n_splits used for the corresponding 03 training run")
    parser.add_argument("--curve_type", type=str, nargs="+",
                        default=["ori_curve", "ori_curve_avg", "ori_curve_wavelet_sym8",
                                 "ori_curve_wavelet_bior35", "ori_curve_sg_p4"],
                        help="Which curve dataset(s) to report on")
    args = parser.parse_args()

    outlier_filters = [None if f == "None" else f for f in args.outlier_filter]

    exp_paths = get_exp_paths(args.exp_folder)
    check_task_id(args.task_id, exp_paths)
    exp_path = exp_paths[args.task_id]

    for curve_type in args.curve_type:
        for outlier_filter in outlier_filters:
            process_experiment(exp_path, args.mode, outlier_filter, args.n_splits,
                               args.force_rerun, curve_type=curve_type)
