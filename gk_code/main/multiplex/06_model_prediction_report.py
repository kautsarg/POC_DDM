"""Multiplex multi-label model prediction report.

Adapted from main/06.  Key structural differences vs main/06:
  - Result file: flat  {filter: res_entry}  (no Native/Reference/curve_type nesting)
  - y_trues_ / y_preds_ / y_probs_ are 2D (N_test, n_targets), not 1D encoded
  - Metrics: exact-match accuracy, hamming loss, per-label F1 (not single-class accuracy)
  - No --mode, no --curve_type (flat lab data has one curve variant, one training mode)
"""
import os
import re
import sys
import gc
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, hamming_loss, f1_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)
from scipy import stats as _scipy_stats

_THIS          = Path(__file__).resolve()
_MULTIPLEX_DIR = _THIS.parent
_MAIN_DIR      = _MULTIPLEX_DIR.parent

sys.path.insert(0, str(_MAIN_DIR))
sys.path.insert(0, str(_MAIN_DIR / "utils"))
sys.path.insert(0, str(_MAIN_DIR / "utils" / "model_training"))
sys.path.insert(0, str(_MULTIPLEX_DIR))
sys.path.insert(0, str(_MULTIPLEX_DIR / "utils" / "model_training"))

import config_multiplex as config
from html_utils import _fig_to_buf, _buf_to_img_html, _panel, build_tabbed_html
from model_utils_mtl import REG_SENTINEL as _REG_SENTINEL
from model_utils_multilabel import (
    ML_MODEL_KEY_MAP,
    ML_MODEL_PRINT_MAP,
    encode_multilabel_for_training,
)


# ======================================================================
# SPLIT RECONSTRUCTION
# Mirrors evaluate_outlier_filters_ml's split logic exactly so cached
# fold predictions can be mapped back to original curve indices.
# ======================================================================

def compute_ml_filtered_splits(y_binary, y_combo_int, features_df, outlier_filter, n_splits):
    """Reconstruct mask + splits from the cached training run.

    Returns (global_idx, y_bin_f, y_comb_f, splits) or None if skipped.
    global_idx[te] gives original array indices for test-fold te.
    """
    if outlier_filter is None:
        mask = np.ones(len(y_combo_int), dtype=bool)
    elif outlier_filter in features_df.columns:
        mask = features_df[outlier_filter].fillna(False).astype(bool).values
    else:
        return None

    global_idx = np.where(mask)[0]
    y_bin_f    = y_binary[mask]
    y_comb_f   = y_combo_int[mask]

    counts      = np.bincount(y_comb_f)
    rare_combos = np.where(counts < 2)[0]
    if len(rare_combos):
        keep       = ~np.isin(y_comb_f, rare_combos)
        global_idx = global_idx[keep]
        y_bin_f    = y_bin_f[keep]
        y_comb_f   = y_comb_f[keep]
        counts     = np.bincount(y_comb_f)

    if len(y_comb_f) < 4:
        return None

    n_total   = len(y_comb_f)
    test_size = max(int(n_total * 0.10), len(np.unique(y_comb_f)))
    min_cnt   = int(np.min(counts[counts > 0]))

    if n_splits == 1:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
    else:
        splitter = StratifiedKFold(
            n_splits=min(n_splits, min_cnt), shuffle=True, random_state=0)

    splits = list(splitter.split(y_bin_f, y_comb_f))
    return global_idx, y_bin_f, y_comb_f, splits


# ======================================================================
# RENDERING — OVERVIEW
# ======================================================================

def render_ml_overview_chart(model_results):
    """Horizontal bar chart: exact-match accuracy and F1-macro sorted by exact acc."""
    if not model_results:
        return None
    sorted_r = sorted(model_results, key=lambda r: r["exact_acc"])
    names  = [r["name"]       for r in sorted_r]
    accs   = [r["exact_acc"]  for r in sorted_r]
    stds   = [r["exact_std"]  for r in sorted_r]
    f1s    = [r["f1_macro"]   for r in sorted_r]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, max(3.0, 0.55 * len(names))))

    colors = ["#27ae60" if a >= 80 else "#e67e22" if a >= 60 else "#e74c3c" for a in accs]
    bars = ax1.barh(names, accs, xerr=stds, color=colors, alpha=0.85, capsize=4,
                    error_kw={"elinewidth": 1.5})
    ax1.set_xlabel("Exact-match Accuracy (%)", fontsize=10)
    ax1.set_xlim(0, 115)
    max_std = max(stds) if stds else 0
    for bar, acc, std in zip(bars, accs, stds):
        ax1.text(bar.get_width() + max_std + 1,
                 bar.get_y() + bar.get_height() / 2,
                 f"{acc:.1f}±{std:.1f}%", va="center", fontsize=8)
    ax1.set_title("Exact-Match Accuracy", fontsize=11, fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)

    colors2 = ["#2980b9" if f >= 0.8 else "#e67e22" if f >= 0.6 else "#e74c3c" for f in f1s]
    ax2.barh(names, [f * 100 for f in f1s], color=colors2, alpha=0.85)
    ax2.set_xlabel("F1 Macro (%)", fontsize=10)
    ax2.set_xlim(0, 115)
    for i, f in enumerate(f1s):
        ax2.text(f * 100 + 1, i, f"{f*100:.1f}%", va="center", fontsize=8)
    ax2.set_title("F1 Macro", fontsize=11, fontweight="bold")
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels([])
    ax2.grid(axis="x", alpha=0.3)

    plt.suptitle("Model Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return _fig_to_buf(fig)


def render_ml_fold_stability(model_results):
    """Strip chart of per-fold exact-match accuracies (only when n_splits > 1)."""
    multi = [r for r in model_results if len(r["fold_accs"]) > 1]
    if not multi:
        return None

    names = [r["name"] for r in multi]
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(names)), 4))
    rng = np.random.default_rng(0)
    for i, r in enumerate(multi):
        fold_pct = np.array(r["fold_accs"])  # already in %
        jitter = rng.uniform(-0.15, 0.15, len(fold_pct))
        ax.scatter(np.full(len(fold_pct), i) + jitter, fold_pct,
                   alpha=0.7, s=45, zorder=3, color="#3498db")
        ax.plot([i - 0.3, i + 0.3], [r["exact_acc"], r["exact_acc"]],
                color="#2c3e50", linewidth=2.5, zorder=4)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Exact-match accuracy per fold (%)", fontsize=10)
    ax.set_title("Per-Fold Stability", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return _fig_to_buf(fig)


# ======================================================================
# RENDERING — CURVES (correct vs wrong by combination)
# ======================================================================

def render_ml_curve_plot(curves, combo_labels, r):
    """2-col grid (correct | wrong) per unique combination label."""
    combos  = np.unique(combo_labels)
    correct = r["correct_idx"]
    wrong   = r["wrong_idx"]

    fig, axes = plt.subplots(len(combos), 2, figsize=(8, 2.5 * len(combos)),
                             squeeze=False, sharey="row")
    for i, combo in enumerate(combos):
        cmask_c = combo_labels[correct] == combo if len(correct) else np.array([], dtype=bool)
        cmask_w = combo_labels[wrong]   == combo if len(wrong)   else np.array([], dtype=bool)
        wc = correct[cmask_c] if len(correct) else np.array([], dtype=int)
        ww = wrong[cmask_w]   if len(wrong)   else np.array([], dtype=int)

        ax_ok, ax_bad = axes[i, 0], axes[i, 1]
        if len(wc):
            ax_ok.plot(curves[wc].T, color="#27ae60", alpha=0.25, linewidth=0.8)
        ax_ok.set_title(f"{combo} – Correct (n={len(wc)})", fontsize=9, fontweight="bold")

        if len(ww):
            ax_bad.plot(curves[ww].T, color="#e74c3c", alpha=0.4, linewidth=0.8)
        ax_bad.set_title(f"{combo} – Wrong (n={len(ww)})", fontsize=9, fontweight="bold")

        for ax in (ax_ok, ax_bad):
            ax.set_xlabel("Time index", fontsize=8)
            ax.set_ylabel("Signal", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)

    plt.tight_layout()
    return _fig_to_buf(fig)


# ======================================================================
# RENDERING — PER-LABEL CONFUSION MATRICES
# ======================================================================

def render_label_cm_grid(r, target_names):
    """One 2×2 confusion matrix per target gene, in a row."""
    n = len(target_names)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
    y_true = r["y_true_all"]  # (N, n_targets)
    y_pred = r["y_pred_all"]  # (N, n_targets)

    for j, tname in enumerate(target_names):
        cm = confusion_matrix(y_true[:, j], y_pred[:, j], labels=[0, 1])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        ax = axes[0, j]
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for rr in range(2):
            for cc in range(2):
                ax.text(cc, rr, f"{cm[rr, cc]}\n({cm_norm[rr, cc]*100:.0f}%)",
                        ha="center", va="center", fontsize=9,
                        color="white" if cm_norm[rr, cc] > 0.5 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["neg", "pos"]); ax.set_yticklabels(["neg", "pos"])
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)
        ax.set_title(tname, fontsize=10, fontweight="bold")

    plt.suptitle(f"{r['name']} — Per-Label Confusion Matrices", fontsize=11, fontweight="bold")
    plt.tight_layout()
    return _fig_to_buf(fig)


# ======================================================================
# RENDERING — PER-LABEL ROC + PR
# ======================================================================

def render_label_roc_pr(r, target_names):
    """ROC and Precision-Recall curves, one line per target gene."""
    y_prob = r["y_prob_all"]
    y_true = r["y_true_all"]
    if y_prob is None:
        return None

    cmap = plt.cm.tab10(np.linspace(0, 0.9, len(target_names)))
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4))

    for j, (tname, color) in enumerate(zip(target_names, cmap)):
        yt = y_true[:, j]
        yp = y_prob[:, j]
        if len(np.unique(yt)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yt, yp)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color=color, linewidth=1.5,
                    label=f"{tname} (AUC={roc_auc:.2f})")

        prec, rec, _ = precision_recall_curve(yt, yp)
        ap = average_precision_score(yt, yp)
        ax_pr.plot(rec, prec, color=color, linewidth=1.5,
                   label=f"{tname} (AP={ap:.2f})")

    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax_roc.set_xlabel("False Positive Rate", fontsize=9)
    ax_roc.set_ylabel("True Positive Rate", fontsize=9)
    ax_roc.set_title("ROC (per label)", fontsize=10, fontweight="bold")
    ax_roc.legend(fontsize=8)
    ax_roc.grid(alpha=0.3)

    ax_pr.set_xlabel("Recall", fontsize=9)
    ax_pr.set_ylabel("Precision", fontsize=9)
    ax_pr.set_title("Precision-Recall (per label)", fontsize=10, fontweight="bold")
    ax_pr.legend(fontsize=8)
    ax_pr.grid(alpha=0.3)

    plt.suptitle(f"{r['name']}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    return _fig_to_buf(fig)


# ======================================================================
# RENDERING — CONFIDENCE (max sigmoid)
# ======================================================================

def render_ml_confidence(r):
    """Max sigmoid score distribution: exact-match correct (green) vs wrong (red)."""
    y_prob = r["y_prob_all"]
    if y_prob is None:
        return None

    y_true = r["y_true_all"]
    y_pred = r["y_pred_all"]
    max_conf     = y_prob.max(axis=1)
    correct_mask = np.all(y_true == y_pred, axis=1)
    n_correct    = int(correct_mask.sum())
    n_wrong      = int((~correct_mask).sum())

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bins = np.linspace(0, 1, 21)
    if n_correct:
        ax.hist(max_conf[correct_mask], bins=bins, alpha=0.6, color="#27ae60",
                label=f"Exact match (n={n_correct})", density=True)
        ax.axvline(max_conf[correct_mask].mean(), color="#1e8449", linestyle="--",
                   linewidth=1.5, label=f"mean: {max_conf[correct_mask].mean():.2f}")
    if n_wrong:
        ax.hist(max_conf[~correct_mask], bins=bins, alpha=0.6, color="#e74c3c",
                label=f"Wrong (n={n_wrong})", density=True)
        ax.axvline(max_conf[~correct_mask].mean(), color="#922b21", linestyle="--",
                   linewidth=1.5, label=f"mean: {max_conf[~correct_mask].mean():.2f}")

    ax.set_xlabel("Max sigmoid probability", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title(f"{r['name']} — Prediction Confidence", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return _fig_to_buf(fig)


# ======================================================================
# CLASS METRICS HTML TABLE (per label)
# ======================================================================

def build_label_metrics_html(r, target_names):
    """HTML table with per-label P/R/F1 and macro/micro/samples aggregates."""
    y_true = r["y_true_all"]
    y_pred = r["y_pred_all"]

    cell = "padding:6px 14px;border:1px solid #ddd;text-align:center;"
    th_s = f"background:#2c3e50;color:white;{cell}"

    header = "".join(f"<th style='{th_s}'>{c}</th>"
                     for c in ["Label", "N pos / total", "Precision", "Recall", "F1"])

    rows_html = ""
    for j, tname in enumerate(target_names):
        bg = "#f2f3f4" if j % 2 == 0 else "white"
        prec, rec, f1, sup = precision_recall_fscore_support(
            y_true[:, j], y_pred[:, j], average="binary", zero_division=0)
        n_pos = int(y_true[:, j].sum())
        rows_html += ("<tr>"
                      + "".join(f"<td style='{cell}background:{bg};'>{v}</td>"
                                for v in [tname, f"{n_pos}/{len(y_true)}",
                                          f"{prec:.3f}", f"{rec:.3f}", f"{f1:.3f}"])
                      + "</tr>")

    def _agg_row(label, avg, bg):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0)
        return ("<tr>" + "".join(f"<td style='{cell}background:{bg};font-weight:bold;'>{v}</td>"
                                 for v in [label, "—", f"{p:.3f}", f"{r:.3f}", f"{f:.3f}"])
                + "</tr>")

    rows_html += _agg_row("Macro avg",    "macro",    "#d6eaf8")
    rows_html += _agg_row("Micro avg",    "micro",    "#d6eaf8")
    rows_html += _agg_row("Samples avg",  "samples",  "#d6eaf8")

    exact = float(accuracy_score(y_true, y_pred)) * 100
    hl    = float(hamming_loss(y_true, y_pred))

    return (
        f'<div style="margin-bottom:28px;">'
        f'<h3 style="margin:0 0 6px 0;font-size:14px;">{r["name"]}'
        f'  <span style="font-size:12px;color:#666;font-weight:normal;">'
        f'Exact: {r["exact_acc"]:.1f}% ± {r["exact_std"]:.1f}%  |  '
        f'Hamming: {hl:.4f}</span></h3>'
        f'<table style="border-collapse:collapse;font-size:13px;">'
        f'<thead><tr>{header}</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table></div>'
    )


# ======================================================================
# MTL REGRESSION (RCFD models) — same helpers as main/06
# ======================================================================

def _mtl_regression_metrics(reg_preds_list, reg_trues_list):
    fold_rows = []
    for fold_i, (preds, trues) in enumerate(zip(reg_preds_list, reg_trues_list)):
        preds = np.asarray(preds, dtype=float)
        trues = np.asarray(trues, dtype=float)
        valid = trues != _REG_SENTINEL
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
    valid_rows = [r for r in fold_rows if np.isfinite(r["mae"])]
    agg = {}
    for k in ("mae", "rmse", "r2", "pearson_r"):
        vals = [r[k] for r in valid_rows]
        agg[k + "_mean"] = float(np.mean(vals)) if vals else float("nan")
        agg[k + "_std"]  = float(np.std(vals))  if vals else float("nan")
    return fold_rows, agg


def render_mtl_scatter(mtl_reg_list, target_names):
    if not mtl_reg_list:
        return None
    n_models = len(mtl_reg_list)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5), squeeze=False)
    cmap = plt.cm.tab10

    for ax, rd in zip(axes[0], mtl_reg_list):
        all_preds, all_trues = [], []
        for preds, trues in zip(rd["reg_preds"], rd["reg_trues"]):
            preds = np.asarray(preds, dtype=float)
            trues = np.asarray(trues, dtype=float)
            valid = trues != _REG_SENTINEL
            if valid.sum() == 0:
                continue
            all_preds.append(preds[valid])
            all_trues.append(trues[valid])
        if not all_preds:
            ax.set_visible(False)
            continue
        P = np.concatenate(all_preds)
        T = np.concatenate(all_trues)
        ax.scatter(T, P, alpha=0.5, s=20, color=cmap(0))
        lo, hi = min(T.min(), P.min()), max(T.max(), P.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        _, agg = _mtl_regression_metrics(rd["reg_preds"], rd["reg_trues"])
        ax.set_title(f"{rd['name']}\nR²={agg['r2_mean']:.3f}  MAE={agg['mae_mean']:.3f}", fontsize=9)
        ax.set_xlabel("True concentration", fontsize=8)
        ax.set_ylabel("Predicted concentration", fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    return _fig_to_buf(fig)


def render_mtl_metrics_table(mtl_reg_list):
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
        rows_html += ("<tr>"
                      + "".join(f"<td style='{cell}background:{bg};'>{c}</td>" for c in cells)
                      + "</tr>")
    return (f'<table style="border-collapse:collapse;font-size:13px;">'
            f'<thead><tr>{header}</tr></thead>'
            f'<tbody>{rows_html}</tbody></table>')


# ======================================================================
# RENDERING — COMBINATION FREQUENCY (train vs test)
# ======================================================================

def render_ml_combination_freq_chart(well_labels, global_idx, splits):
    """Grouped bar: average per-combination count in train vs test across folds."""
    filtered_labels = np.asarray(well_labels)[global_idx]
    combos = sorted(np.unique(filtered_labels))

    train_counts = {c: [] for c in combos}
    test_counts  = {c: [] for c in combos}
    for tr, te in splits:
        tr_lbl = filtered_labels[tr]
        te_lbl = filtered_labels[te]
        for c in combos:
            train_counts[c].append(int(np.sum(tr_lbl == c)))
            test_counts[c].append(int(np.sum(te_lbl == c)))

    avg_train = [np.mean(train_counts[c]) for c in combos]
    avg_test  = [np.mean(test_counts[c])  for c in combos]

    x = np.arange(len(combos))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(combos)), 4))
    ax.bar(x - w / 2, avg_train, w, label="Train (avg/fold)", color="#3498db", alpha=0.85)
    ax.bar(x + w / 2, avg_test,  w, label="Test  (avg/fold)", color="#e67e22", alpha=0.85)
    for xi, (nt, nv) in enumerate(zip(avg_train, avg_test)):
        ax.text(xi - w / 2, nt + 0.3, f"{nt:.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + w / 2, nv + 0.3, f"{nv:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(combos, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Sample count", fontsize=10)
    ax.set_title("Combination Frequency: Train vs Test Split", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return _fig_to_buf(fig)


# ======================================================================
# MAIN EXPERIMENT PROCESSOR
# ======================================================================

def process_experiment(exp_path, outlier_filter, n_splits, force_rerun):
    """Generate an HTML report for one (experiment, outlier_filter) combination."""
    out_dir = exp_path / "model_performance_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    filter_tag = "none" if outlier_filter is None else re.sub(r"[^A-Za-z0-9._-]+", "_", outlier_filter)
    out_path   = out_dir / f"ml_report__{filter_tag}__nsplits{n_splits}.html"

    if out_path.exists() and not force_rerun:
        print(f"  -> [SKIP] Report already exists: {out_path}")
        return

    results_path = exp_path / (config.TRAINING_10FOLD_RESULT_PATH
                               if n_splits > 1 else config.TRAINING_RESULT_PATH)
    data_path    = exp_path / config.TRAINING_DATA_PATH

    if not results_path.exists():
        print(f"  -> Skipping {exp_path.name}: '{results_path.name}' not found.")
        return
    if not data_path.exists():
        print(f"  -> Skipping {exp_path.name}: '{data_path.name}' not found.")
        return

    print(f"\n{'#'*80}\nGENERATING ML REPORT: {exp_path.name} | filter={filter_tag}\n{'#'*80}")

    try:
        cached_results = joblib.load(results_path)
        data           = joblib.load(data_path)
    except Exception as e:
        print(f"  -> Skipping {exp_path.name}: failed to load files ({e}).")
        return

    curves      = data["curves"]["ori_curves"]
    label_lists = data["label_lists"]
    all_targets = data["all_targets"]
    well_labels = data["well_labels"]    # combo strings, e.g. "VIM_NDM"

    # Kinetic features for outlier filter mask reconstruction
    kf_dict     = data.get("kinetic_features", {})
    features_df = kf_dict.get(exp_path.name)
    if features_df is None or len(features_df) != len(curves):
        features_df = pd.DataFrame(index=range(len(curves)))

    y_binary, y_combo_int, _, _ = encode_multilabel_for_training(label_lists, all_targets)

    if outlier_filter not in cached_results:
        print(f"  -> No results for filter '{outlier_filter}'. Skipping.")
        return

    res_entry  = cached_results[outlier_filter]
    split_info = compute_ml_filtered_splits(
        y_binary, y_combo_int, features_df, outlier_filter, n_splits)
    if split_info is None:
        print(f"  -> Could not reconstruct splits. Skipping.")
        return
    global_idx, y_bin_f, y_comb_f, splits = split_info

    # ---- Collect per-model results ----
    model_results = []
    for m, (preds_key, probs_key, classes_key) in ML_MODEL_KEY_MAP.items():
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

            y_true_fold   = y_bin_f[test_idx_local]          # (N_test, n_targets)
            y_pred_fold   = np.asarray(preds_list[fold_i])   # (N_test, n_targets)
            y_true_cached = np.asarray(y_trues_list[fold_i]) # (N_test, n_targets)

            if y_true_fold.shape != y_true_cached.shape or not np.array_equal(y_true_fold, y_true_cached):
                mismatch = True
                break

            exact_acc_fold = accuracy_score(y_true_fold, y_pred_fold)
            fold_accs.append(exact_acc_fold)
            y_true_list.append(y_true_fold)
            y_pred_list.append(y_pred_fold)

            if probs_list is not None and fold_i < len(probs_list):
                prob_fold = np.asarray(probs_list[fold_i])
                if prob_fold.ndim == 2:
                    y_prob_list.append(prob_fold)

            gidx          = global_idx[test_idx_local]
            exact_correct = np.all(y_true_fold == y_pred_fold, axis=1)
            correct_idx_list.append(gidx[exact_correct])
            wrong_idx_list.append(gidx[~exact_correct])

        if mismatch or not fold_accs:
            print(f"  -> [WARNING] {m}: split mismatch or no folds. Skipping.")
            continue

        y_true_cat = np.concatenate(y_true_list)  # (N_total, n_targets)
        y_pred_cat = np.concatenate(y_pred_list)
        y_prob_cat = np.concatenate(y_prob_list) if y_prob_list else None

        f1_macros  = [f1_score(yt, yp, average="macro",   zero_division=0)
                      for yt, yp in zip(y_true_list, y_pred_list)]
        f1_samples = [f1_score(yt, yp, average="samples", zero_division=0)
                      for yt, yp in zip(y_true_list, y_pred_list)]

        model_results.append({
            "key":         m,
            "name":        ML_MODEL_PRINT_MAP.get(m, m),
            "exact_acc":   float(np.mean(fold_accs)) * 100,
            "exact_std":   float(np.std(fold_accs))  * 100,
            "f1_macro":    float(np.mean(f1_macros)),
            "f1_samples":  float(np.mean(f1_samples)),
            "fold_accs":   [fa * 100 for fa in fold_accs],
            "correct_idx": np.concatenate(correct_idx_list) if correct_idx_list else np.array([], dtype=int),
            "wrong_idx":   np.concatenate(wrong_idx_list)   if wrong_idx_list   else np.array([], dtype=int),
            "y_true_all":  y_true_cat,
            "y_pred_all":  y_pred_cat,
            "y_prob_all":  y_prob_cat,
            "target_names": all_targets,
        })
        print(f"  -> {ML_MODEL_PRINT_MAP.get(m, m):20}  "
              f"exact={np.mean(fold_accs)*100:.1f}%  f1_macro={np.mean(f1_macros)*100:.1f}%")

    if not model_results:
        print(f"  -> No model results found for filter '{outlier_filter}'. Skipping.")
        return

    # ---- Build tabs ----
    tabs = []

    # Tab 1: Overview
    ov = ""
    buf = render_ml_overview_chart(model_results)
    if buf:
        ov += _panel("Model Comparison", _buf_to_img_html(buf, style="height:auto;max-width:900px;"))
    buf = render_ml_fold_stability(model_results)
    if buf:
        ov += _panel("Per-Fold Stability", _buf_to_img_html(buf, style="height:auto;max-width:750px;"))
    tabs.append(("overview", "Overview", f'<div class="panel-row">{ov}</div>'))

    # Tab 2: Combination frequency (train vs test distribution)
    buf = render_ml_combination_freq_chart(well_labels, global_idx, splits)
    if buf:
        tabs.append(("combo_freq", "Combo Freq",
                     _panel("Label Combination Frequency: Train vs Test",
                            _buf_to_img_html(buf, style="height:auto;max-width:650px;"))))

    # Tab 3: Curves
    curves_content = "".join(
        _panel(f'{r["name"]}<br><span style="font-size:11px;color:#666;">'
               f'Exact: {r["exact_acc"]:.1f}% ± {r["exact_std"]:.1f}%</span>',
               _buf_to_img_html(render_ml_curve_plot(curves, well_labels, r),
                                style="height:auto;max-width:600px;"))
        for r in model_results
    )
    tabs.append(("curves", "Curves", f'<div class="panel-row">{curves_content}</div>'))

    # Tab 4: Per-label confusion matrices
    cm_content = "".join(
        _panel(f'{r["name"]}<br><span style="font-size:11px;color:#666;">'
               f'Exact: {r["exact_acc"]:.1f}%</span>',
               _buf_to_img_html(render_label_cm_grid(r, all_targets),
                                style="height:auto;max-width:700px;"))
        for r in model_results
    )
    tabs.append(("cm", "Label CMs", f'<div class="panel-row">{cm_content}</div>'))

    # Tab 5: Label metrics table
    metrics_content = "".join(build_label_metrics_html(r, all_targets) for r in model_results)
    tabs.append(("metrics", "Label Metrics", metrics_content))

    # Tab 6: Confidence
    conf_content = ""
    for r in model_results:
        buf = render_ml_confidence(r)
        if buf:
            conf_content += _panel(r["name"],
                                   _buf_to_img_html(buf, style="height:auto;max-width:480px;"))
    if conf_content:
        tabs.append(("conf", "Confidence", f'<div class="panel-row">{conf_content}</div>'))

    # Tab 7: ROC / PR per label
    roc_content = ""
    for r in model_results:
        buf = render_label_roc_pr(r, all_targets)
        if buf:
            roc_content += _panel(r["name"],
                                  _buf_to_img_html(buf, style="height:auto;max-width:750px;"))
    if roc_content:
        tabs.append(("roc", "ROC / PR", f'<div class="panel-row">{roc_content}</div>'))

    # Tab 8: Regression (RCFD models only — auto-detected)
    mtl_reg_list = []
    for r in model_results:
        mk = r["key"]
        reg_preds_key = f'y_reg_preds_{mk}_'
        reg_trues_key = f'y_reg_trues_{mk}_'
        if reg_preds_key in res_entry:
            mtl_reg_list.append({
                "name": r["name"], "key": mk,
                "reg_preds": res_entry[reg_preds_key],
                "reg_trues": res_entry[reg_trues_key],
            })
    if mtl_reg_list:
        reg_content = render_mtl_metrics_table(mtl_reg_list)
        buf = render_mtl_scatter(mtl_reg_list, all_targets)
        if buf:
            reg_content += _panel("Predicted vs Actual Concentration",
                                   _buf_to_img_html(buf, style="height:auto;max-width:100%;"))
        tabs.append(("regression", "Regression (RCFD)", reg_content))

    # ---- Write HTML ----
    filter_label = outlier_filter if outlier_filter else "None (Baseline)"
    title = (f"ML Report: {exp_path.name}  |  Filter: {filter_label}  |  n_splits={n_splits}")
    build_tabbed_html(title, tabs, out_path)
    print(f"  -> [SAVED] {out_path}")
    gc.collect()


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    print(f"\n{'='*70}\n[RUNNING] {os.path.basename(__file__)}\n{'='*70}\n")
    parser = argparse.ArgumentParser(description="Multiplex Multi-Label Model Report")
    parser.add_argument("--task_id",     type=int, default=0, help="Array Job ID")
    parser.add_argument("--exp_folder",  type=str, default=config.LAB_MULTIPLEX_FOLDER)
    parser.add_argument("--n_splits",    type=int, default=1,
                        help="Must match --n_splits used for the corresponding 03 run")
    parser.add_argument("--force_rerun", action="store_true",
                        help="Regenerate HTML reports even if they already exist")
    parser.add_argument("--outlier_filter", type=str, nargs="*",
                        default=config.OUTLIER_FILTERS,
                        help="Filter(s) to report on. Pass 'None' for baseline. "
                             "Default: all filters in config.OUTLIER_FILTERS.")
    args = parser.parse_args()

    outlier_filters = [None if f == "None" else f for f in args.outlier_filter]

    subdirs   = sorted([d for d in Path(args.exp_folder).iterdir() if d.is_dir()])
    task_dirs = [d for d in subdirs if d.name in config.FILE_MAPPING]

    if not task_dirs:
        print(f"No valid task directories found in {args.exp_folder}.")
        sys.exit(0)

    exp_path = task_dirs[args.task_id % len(task_dirs)]
    print(f"Generating reports for: {exp_path.name}")

    for outlier_filter in outlier_filters:
        process_experiment(exp_path, outlier_filter, args.n_splits, args.force_rerun)
