import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon, chi2 as chi2_dist, rankdata
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

import config
from html_utils import _fig_to_buf, _buf_to_img_html, _panel, build_tabbed_html


_NEMENYI_Q = {
    2: 1.960, 3: 2.344, 4: 2.569,  5: 2.728,
    6: 2.850, 7: 2.948, 8: 3.031,  9: 3.102,
    10: 3.163, 11: 3.219, 12: 3.268, 13: 3.313,
    14: 3.354, 15: 3.391, 20: 3.615,
}

MODEL_KEY_MAP = {k: v[0] for k, v in config.MODEL_KEY_MAP.items()}
MODEL_PRINT_MAP = config.MODEL_PRINT_MAP
FILTER_PRINT_MAP = config.FILTER_PRINT_MAP
CURVE_PRINT_MAP = config.CURVE_PRINT_MAP
METRIC_LABEL = config.METRIC_LABEL


def _compute_metric(y_true, y_pred, metric):
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "macro_f1":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    if metric == "mcc":
        return float(matthews_corrcoef(y_true, y_pred))
    raise ValueError(f"Unknown metric: {metric}")


def load_exp_data(exp_path, n_splits, mode, curve_type_args, metric="accuracy"):
    """{dataset_name: {filter_key: {model_key: {fold_metrics, mean_metric,
    y_true_all, y_pred_all, n_folds}}}} or None if the results file is missing."""
    fname = (config.TRAINING_10FOLD_RESULT_PATH if n_splits > 1
             else config.TRAINING_RESULT_PATH)
    results_path = Path(exp_path) / fname

    if not results_path.exists():
        print(f"  [SKIP] {Path(exp_path).name}: '{fname}' not found.")
        return None

    raw = joblib.load(results_path)
    mode_key = mode.strip().title()
    out = {}

    for ct_arg in curve_type_args:
        dataset_name = config.CURVE_TYPE_ALIASES.get(ct_arg, ct_arg)
        if dataset_name not in raw:
            continue
        if mode_key not in raw[dataset_name]:
            continue

        mode_results = raw[dataset_name][mode_key]
        out[dataset_name] = {}

        for filter_key, filter_results in mode_results.items():
            if "y_trues_" not in filter_results:
                continue
            y_trues_ = filter_results["y_trues_"]
            n_folds = len(y_trues_)
            if n_folds == 0:
                continue

            out[dataset_name][filter_key] = {}

            for model_key, preds_key in MODEL_KEY_MAP.items():
                if preds_key not in filter_results:
                    continue
                y_preds_ = filter_results[preds_key]
                if len(y_preds_) != n_folds:
                    continue

                fold_metrics, y_true_parts, y_pred_parts = [], [], []
                for fi in range(n_folds):
                    y_true = np.asarray(y_trues_[fi])
                    y_pred = np.asarray(y_preds_[fi])
                    if len(y_true) == 0 or len(y_true) != len(y_pred):
                        continue
                    fold_metrics.append(_compute_metric(y_true, y_pred, metric))
                    y_true_parts.append(y_true)
                    y_pred_parts.append(y_pred)

                if not fold_metrics:
                    continue

                out[dataset_name][filter_key][model_key] = {
                    "fold_metrics": fold_metrics,
                    "mean_metric":  float(np.mean(fold_metrics)),
                    "y_true_all":   np.concatenate(y_true_parts),
                    "y_pred_all":   np.concatenate(y_pred_parts),
                    "n_folds":      len(fold_metrics),
                }

    return out if out else None


def _get_value(exp_data, compare_axis, fixed, cond):
    """Mean metric for one (exp_data, condition) cell, or NaN if missing."""
    if compare_axis == "models":
        entry = (exp_data
                 .get(fixed["curve_type"], {})
                 .get(fixed["filter"], {})
                 .get(cond))
    elif compare_axis == "outlier_filters":
        entry = (exp_data
                 .get(fixed["curve_type"], {})
                 .get(cond, {})
                 .get(fixed["model"]))
    elif compare_axis == "curve_types":
        entry = (exp_data
                 .get(cond, {})
                 .get(fixed["filter"], {})
                 .get(fixed["model"]))
    else:
        return np.nan
    return entry["mean_metric"] if entry is not None else np.nan


def _get_preds(exp_data, compare_axis, fixed, cond):
    """(y_true_all, y_pred_all) for McNemar, or (None, None) if missing."""
    if compare_axis == "models":
        entry = (exp_data
                 .get(fixed["curve_type"], {})
                 .get(fixed["filter"], {})
                 .get(cond))
    elif compare_axis == "outlier_filters":
        entry = (exp_data
                 .get(fixed["curve_type"], {})
                 .get(cond, {})
                 .get(fixed["model"]))
    elif compare_axis == "curve_types":
        entry = (exp_data
                 .get(cond, {})
                 .get(fixed["filter"], {})
                 .get(fixed["model"]))
    else:
        entry = None
    if entry is None:
        return None, None
    return entry["y_true_all"], entry["y_pred_all"]


def build_accuracy_matrix(exp_data_list, compare_axis, fixed, all_conditions):
    """(n_experiments x n_conditions) mean-metric matrix. Conditions with ANY
    missing value across experiments are dropped (Friedman requires complete blocks).
    Returns (matrix, valid_condition_keys, exp_names)."""
    n_exp = len(exp_data_list)
    n_cond = len(all_conditions)

    raw = np.full((n_exp, n_cond), np.nan)
    for i, (_, exp_data) in enumerate(exp_data_list):
        for j, cond in enumerate(all_conditions):
            raw[i, j] = _get_value(exp_data, compare_axis, fixed, cond)

    valid_mask = ~np.any(np.isnan(raw), axis=0)
    if not np.any(valid_mask):
        return None, [], [n for n, _ in exp_data_list]

    dropped = [all_conditions[j] for j in range(n_cond) if not valid_mask[j]]
    if dropped:
        print(f"  [WARN] Dropping {len(dropped)} conditions with missing data in >=1 folder.")

    return (raw[:, valid_mask],
            [all_conditions[j] for j in range(n_cond) if valid_mask[j]],
            [n for n, _ in exp_data_list])


def holm_correction(p_values, alpha=0.05):
    """Holm-Bonferroni step-down correction. Returns (reject, adjusted_p)."""
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=bool), np.array([])

    p = np.asarray(p_values, dtype=float)
    sorted_idx = np.argsort(p)

    adjusted_sorted = np.zeros(n)
    prev = 0.0
    for k, idx in enumerate(sorted_idx):
        adjusted_sorted[k] = max(prev, min(1.0, p[idx] * (n - k)))
        prev = adjusted_sorted[k]

    adjusted = np.zeros(n)
    for k, idx in enumerate(sorted_idx):
        adjusted[idx] = adjusted_sorted[k]

    return adjusted < alpha, adjusted


def nemenyi_cd(k, n_exp, alpha=0.05):
    """CD = q_alpha * sqrt(k(k+1)/(6N))."""
    available = sorted(_NEMENYI_Q.keys())
    q = _NEMENYI_Q.get(k, _NEMENYI_Q[min(available, key=lambda x: abs(x - k))])
    return q * np.sqrt(k * (k + 1) / (6 * n_exp))


def _rank_biserial(matrix_col_a, matrix_col_b, w_stat):
    """Signed rank-biserial correlation; r > 0 means A tends higher than B."""
    n = len(matrix_col_a)
    n_total = n * (n + 1) / 2
    magnitude = 1.0 - 2.0 * w_stat / n_total
    direction = np.sign(np.mean(matrix_col_a - matrix_col_b))
    return float(direction * magnitude) if direction != 0 else 0.0


def run_friedman_posthoc(matrix, condition_keys, condition_names, baseline_idx, alpha=0.05):
    """Friedman test on (n_exp x k) matrix, then pairwise Wilcoxon with Holm correction."""
    n_exp, k = matrix.shape

    if n_exp < 2:
        return {"error": f"Need >=2 experiment folders (got {n_exp})."}
    if k < 2:
        return {"error": "Need >=2 conditions."}

    try:
        friedman_stat, friedman_p = friedmanchisquare(*matrix.T)
    except Exception as e:
        return {"error": f"Friedman test failed: {e}"}

    def rank_row(row):
        return k + 1 - rankdata(row, method="average")

    ranks = np.apply_along_axis(rank_row, 1, matrix)
    avg_ranks = ranks.mean(axis=0)

    pairs = list(combinations(range(k), 2))
    raw_p, w_stats, eff_sizes = [], [], []

    for i, j in pairs:
        diff = matrix[:, i] - matrix[:, j]
        if np.all(diff == 0):
            raw_p.append(1.0); w_stats.append(np.nan); eff_sizes.append(0.0)
            continue
        try:
            w, p = wilcoxon(matrix[:, i], matrix[:, j],
                            alternative="two-sided", zero_method="pratt")
            raw_p.append(float(p))
            w_stats.append(float(w))
            eff_sizes.append(_rank_biserial(matrix[:, i], matrix[:, j], w))
        except Exception:
            raw_p.append(1.0); w_stats.append(np.nan); eff_sizes.append(0.0)

    reject, adj_p = holm_correction(raw_p, alpha=alpha)

    pval_mat = np.full((k, k), np.nan)
    adj_mat  = np.full((k, k), np.nan)
    eff_mat  = np.full((k, k), np.nan)
    rej_mat  = np.full((k, k), False)

    for idx, (i, j) in enumerate(pairs):
        pval_mat[i, j] = pval_mat[j, i] = raw_p[idx]
        adj_mat[i, j]  = adj_mat[j, i]  = adj_p[idx]
        eff_mat[i, j]  = eff_sizes[idx]
        eff_mat[j, i]  = -eff_sizes[idx]
        rej_mat[i, j]  = rej_mat[j, i]  = reject[idx]

    return {
        "n_exp":         n_exp,
        "k":             k,
        "friedman_stat": friedman_stat,
        "friedman_p":    friedman_p,
        "friedman_sig":  friedman_p < alpha,
        "avg_ranks":     avg_ranks,
        "cd":            nemenyi_cd(k, n_exp, alpha=alpha),
        "condition_names": condition_names,
        "condition_keys":  condition_keys,
        "pval_mat":      pval_mat,
        "adj_mat":       adj_mat,
        "eff_mat":       eff_mat,
        "rej_mat":       rej_mat,
        "mean_values":   matrix.mean(axis=0),
        "std_values":    matrix.std(axis=0, ddof=1 if n_exp > 1 else 0),
        "matrix":        matrix,
        "baseline_idx":  baseline_idx,
    }


def run_mcnemar_pairwise(exp_data, compare_axis, fixed, condition_keys, condition_names,
                         baseline_idx, alpha=0.05):
    """McNemar's test on per-sample predictions for the N=1-folder fallback."""
    k = len(condition_keys)
    pairs = list(combinations(range(k), 2))
    raw_p, chi2_vals = [], []

    for i, j in pairs:
        y_true_i, y_pred_i = _get_preds(exp_data, compare_axis, fixed, condition_keys[i])
        y_true_j, y_pred_j = _get_preds(exp_data, compare_axis, fixed, condition_keys[j])

        if y_pred_i is None or y_pred_j is None or len(y_pred_i) != len(y_pred_j):
            raw_p.append(np.nan); chi2_vals.append(np.nan)
            continue

        correct_i = (y_pred_i == y_true_i)
        correct_j = (y_pred_j == y_true_j)
        b = int(np.sum(correct_i & ~correct_j))
        c = int(np.sum(~correct_i & correct_j))

        if b + c == 0:
            raw_p.append(1.0); chi2_vals.append(0.0)
        else:
            chi2_val = (abs(b - c) - 1) ** 2 / (b + c)
            raw_p.append(float(1 - chi2_dist.cdf(chi2_val, df=1)))
            chi2_vals.append(float(chi2_val))

    valid = [p for p in raw_p if not np.isnan(p)]
    if valid:
        reject_valid, adj_valid = holm_correction(valid, alpha=alpha)
    else:
        reject_valid, adj_valid = np.array([], dtype=bool), np.array([])

    pval_mat = np.full((k, k), np.nan)
    adj_mat  = np.full((k, k), np.nan)
    rej_mat  = np.full((k, k), False)

    vi = 0
    for idx, (i, j) in enumerate(pairs):
        if not np.isnan(raw_p[idx]):
            pval_mat[i, j] = pval_mat[j, i] = raw_p[idx]
            adj_mat[i, j]  = adj_mat[j, i]  = adj_valid[vi]
            rej_mat[i, j]  = rej_mat[j, i]  = reject_valid[vi]
            vi += 1

    mean_values = np.zeros(k)
    for j, ck in enumerate(condition_keys):
        yt, yp = _get_preds(exp_data, compare_axis, fixed, ck)
        if yt is not None:
            mean_values[j] = _compute_metric(yt, yp, "accuracy")

    return {
        "n_exp":         1,
        "k":             k,
        "friedman_stat": np.nan,
        "friedman_p":    np.nan,
        "friedman_sig":  False,
        "avg_ranks":     np.full(k, np.nan),
        "cd":            np.nan,
        "condition_names": condition_names,
        "condition_keys":  condition_keys,
        "pval_mat":      pval_mat,
        "adj_mat":       adj_mat,
        "eff_mat":       np.full((k, k), np.nan),
        "rej_mat":       rej_mat,
        "mean_values":   mean_values,
        "std_values":    np.zeros(k),
        "matrix":        mean_values.reshape(1, -1),
        "baseline_idx":  baseline_idx,
    }


def plot_cd_diagram(avg_ranks, cd, condition_names, baseline_idx=None,
                    title="Critical Difference Diagram"):
    """Demsar-style CD diagram; x-axis = mean rank (1 = best, left)."""
    k = len(avg_ranks)
    sort_idx = np.argsort(avg_ranks)
    sorted_ranks = avg_ranks[sort_idx]
    sorted_names  = [condition_names[i] for i in sort_idx]
    sorted_is_bl  = [(i == baseline_idx) for i in sort_idx]

    fig_w = max(9, k * 1.3)
    fig, ax = plt.subplots(figsize=(fig_w, max(4, k * 0.55 + 1.5)))
    ax.axis("off")
    ax.set_xlim(0.3, k + 0.7)
    ax.set_ylim(-0.5, 3.0)

    ax.plot([1, k], [2.5, 2.5], "k-", lw=1.5)
    for r in range(1, k + 1):
        ax.plot([r, r], [2.5, 2.62], "k-", lw=1.2)
        ax.text(r, 2.72, str(r), ha="center", va="bottom", fontsize=9)
    ax.text((1 + k) / 2, 3.0, "Mean Rank  (lower = better)", ha="center",
            va="bottom", fontsize=10, fontweight="bold")

    cd_mid = k - cd / 2
    ax.annotate("", xy=(k, 2.6), xytext=(k - cd, 2.6),
                arrowprops=dict(arrowstyle="<->", color="#e74c3c", lw=1.8))
    ax.text(k - cd / 2, 2.75, f"CD={cd:.2f}", ha="center", va="bottom",
            fontsize=8, color="#e74c3c")

    n_labels = len(sort_idx)
    y_positions = np.linspace(0, 2.0, n_labels)

    for plot_i, (orig_i, rank, name, is_bl) in enumerate(
            zip(sort_idx, sorted_ranks, sorted_names, sorted_is_bl)):
        color = "#e74c3c" if is_bl else "#2c3e50"
        weight = "bold" if is_bl else "normal"
        y_text = y_positions[plot_i]

        ax.plot([rank, rank], [2.5, y_text + 0.12], color=color, lw=0.9, alpha=0.7)
        ax.plot(rank, 2.5, "o", color=color, ms=6 + 2 * is_bl, zorder=5)
        ax.text(rank, y_text, ("★ " if is_bl else "") + name,
                ha="center", va="top", fontsize=8, color=color, fontweight=weight)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=4)
    plt.tight_layout()
    return fig


def plot_pvalue_heatmap(adj_mat, rej_mat, condition_names, baseline_idx=None,
                        alpha=0.05, metric_name="Accuracy", title=None):
    """Lower-triangular Holm-corrected p-value heatmap."""
    k = len(condition_names)
    if title is None:
        title = f"Pairwise adj. p-values  ({metric_name}, Holm-corrected)"

    fig, ax = plt.subplots(figsize=(max(5, k * 0.85), max(4, (k - 1) * 0.8)))

    for i in range(1, k):
        for j in range(i):
            p = adj_mat[i, j]
            sig = bool(rej_mat[i, j])
            is_bl = (i == baseline_idx or j == baseline_idx)

            if np.isnan(p):
                facecolor, text = "#d5d8dc", "N/A"
            elif sig:
                facecolor = "#1a9c5c" if is_bl else "#abebc6"
                text = f"{p:.3f}*"
            else:
                facecolor = "#c0392b" if is_bl else "#f5b7b1"
                text = f"{p:.3f}"

            rect = plt.Rectangle([j, k - i - 1], 1, 1, facecolor=facecolor,
                                  edgecolor="white", lw=1.5)
            ax.add_patch(rect)
            if is_bl and baseline_idx is not None:
                ax.add_patch(plt.Rectangle([j, k - i - 1], 1, 1, facecolor="none",
                                           edgecolor="#2c3e50", lw=2.5))
            fs = max(6, 11 - k)
            ax.text(j + 0.5, k - i - 0.5, text, ha="center", va="center",
                    fontsize=fs, fontweight="bold" if sig else "normal")

    ax.set_xlim(0, k - 1)
    ax.set_ylim(0, k - 1)
    ax.set_xticks(np.arange(k - 1) + 0.5)
    ax.set_xticklabels(condition_names[:-1], rotation=30, ha="right", fontsize=9)
    ax.set_yticks(np.arange(k - 1) + 0.5)
    ax.set_yticklabels(condition_names[1:][::-1], fontsize=9)
    ax.set_title(f"{title}\n* = sig. after Holm correction (alpha={alpha})", fontsize=10)
    plt.tight_layout()
    return fig


def plot_accuracy_boxplot(matrix, condition_names, metric_name="Accuracy", baseline_idx=None):
    """Boxplot per condition; baseline outlined in red; mean shown as star."""
    k = matrix.shape[1]
    fig, ax = plt.subplots(figsize=(max(6, k), 5))

    colors = ["#f5cba7" if j == baseline_idx else "#aed6f1" for j in range(k)]
    data = [(matrix[:, j] * 100).tolist() for j in range(k)]

    bp = ax.boxplot(data, patch_artist=True, tick_labels=condition_names,
                    medianprops=dict(color="#2c3e50", lw=2))
    for j, (patch, color) in enumerate(zip(bp["boxes"], colors)):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        if j == baseline_idx:
            patch.set_edgecolor("#e74c3c")
            patch.set_linewidth(2.5)

    for j in range(k):
        m = matrix[:, j].mean() * 100
        ax.scatter(j + 1, m, marker="*", s=110, zorder=5,
                   color="#e74c3c" if j == baseline_idx else "#2c3e50")

    ax.set_xticklabels(condition_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(f"{metric_name} (%)")
    ax.set_title(f"{metric_name} across experiment folders (★ = mean)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def _summary_table_html(stats, metric_name, alpha):
    """Sortable HTML table: rank, condition, mean+/-std, avg Friedman rank, vs baseline."""
    k = stats["k"]
    baseline_idx = stats.get("baseline_idx")
    sort_by_mean = np.argsort(-stats["mean_values"])

    header = ("<tr>"
              "<th>Rank</th><th>Condition</th>"
              f"<th>Mean {metric_name} (%)</th>"
              "<th>Friedman Avg Rank</th>"
              "<th>vs Baseline (adj. p)</th>"
              "<th>Effect size (r)</th>"
              "</tr>")
    rows = ""
    for rank, orig_i in enumerate(sort_by_mean):
        name  = stats["condition_names"][orig_i]
        mean_ = stats["mean_values"][orig_i] * 100
        std_  = stats["std_values"][orig_i] * 100
        fr    = stats["avg_ranks"][orig_i]
        is_bl = (orig_i == baseline_idx)

        bg    = " background:#fff3cd;" if is_bl else ""
        bold  = "font-weight:bold;" if is_bl else ""
        label = f"★ {name}" if is_bl else name
        fr_txt = f"{fr:.2f}" if not np.isnan(fr) else "—"

        if baseline_idx is not None and not is_bl:
            p   = stats["adj_mat"][orig_i, baseline_idx]
            sig = bool(stats["rej_mat"][orig_i, baseline_idx])
            eff = stats["eff_mat"][orig_i, baseline_idx]

            if np.isnan(p):
                vs_bl = "N/A"
                eff_txt = "N/A"
            else:
                color = "#27ae60" if (sig and eff > 0) else ("#e74c3c" if (sig and eff < 0) else "#666")
                vs_bl = (f'<span style="color:{color};font-weight:{"bold" if sig else "normal"}">'
                         f'{p:.3f}{"*" if sig else ""}</span>')
                eff_txt = f"N/A" if np.isnan(eff) else f"{eff:+.3f}"
        else:
            vs_bl   = "— (baseline)" if is_bl else "—"
            eff_txt = "—"

        rows += (f'<tr style="{bg}{bold}">'
                 f"<td>{rank + 1}</td><td>{label}</td>"
                 f"<td>{mean_:.2f} ± {std_:.2f}%</td>"
                 f"<td>{fr_txt}</td>"
                 f"<td>{vs_bl}</td>"
                 f"<td>{eff_txt}</td>"
                 "</tr>")

    return (
        '<div style="overflow-x:auto;margin:12px 0">'
        f'<table><thead>{header}</thead><tbody>{rows}</tbody></table>'
        '</div>'
    )


def _methodology_note(n_exp, test_type, alpha):
    if test_type == "friedman":
        body = (
            f"<b>Method:</b> Friedman test across <b>{n_exp} independent experiment folders</b> "
            "(each folder contributes one mean metric value per condition, "
            "so observations are truly independent). "
            "Post-hoc pairwise comparison: Wilcoxon signed-rank test with "
            "<b>Holm-Bonferroni</b> correction (controls FWER at α=" + str(alpha) + "). "
            "Effect size: <b>rank-biserial correlation r</b> (r=+1: A always wins; "
            "r=0: chance; r=−1: B always wins). "
            "CD threshold: Nemenyi critical difference at α=0.05 (Demšar 2006). "
            "★ = baseline condition. * = significant after Holm correction."
        )
    elif test_type == "mcnemar":
        body = (
            f"<b>Method:</b> McNemar's test with continuity correction "
            "(N=1 folder, so cross-experiment generalisation cannot be assessed). "
            "Input: per-sample binary outcomes (correct/wrong) concatenated over all test folds. "
            "χ² = (|b−c|−1)² / (b+c), df=1. "
            "<b>Holm-Bonferroni</b> correction for multiple comparisons (α=" + str(alpha) + "). "
            "★ = baseline condition."
        )
    else:
        body = "Statistical method: N/A."

    return (
        '<div style="margin:12px 0;padding:10px 14px;background:#eaf4fb;'
        'border-left:4px solid #2980b9;font-size:12px;border-radius:0 4px 4px 0">'
        + body + "</div>"
    )


def _baseline_summary(stats, metric_name, alpha):
    baseline_idx = stats.get("baseline_idx")
    if baseline_idx is None:
        return ""

    baseline_name = stats["condition_names"][baseline_idx]
    better, worse = [], []

    for j, name in enumerate(stats["condition_names"]):
        if j == baseline_idx:
            continue
        p   = stats["adj_mat"][j, baseline_idx]
        sig = bool(stats["rej_mat"][j, baseline_idx])
        eff = stats["eff_mat"][j, baseline_idx]
        if sig and not np.isnan(p):
            (better if eff > 0 else worse).append(
                f"<b>{name}</b> (r={eff:+.2f})")

    lines = [
        f"<b>vs Baseline ({baseline_name})  —  Holm-corrected at α={alpha}:</b>",
        "Significantly better: " + (", ".join(better) if better else "none"),
        "Significantly worse:  " + (", ".join(worse)  if worse  else "none"),
    ]
    return (
        '<div style="margin:12px 0;padding:10px 14px;background:#eafaf1;'
        'border-left:4px solid #27ae60;font-size:13px;border-radius:0 4px 4px 0">'
        + "<br>".join(lines) + "</div>"
    )


def build_tab_content(stats, figs, test_type, metric_name, alpha):
    """Assemble HTML for one comparison tab."""
    content = _methodology_note(stats.get("n_exp", "?"), test_type, alpha)

    if "error" in stats:
        return content + f'<p style="color:#e74c3c"><b>Error:</b> {stats["error"]}</p>'

    if test_type == "friedman":
        sig = stats["friedman_sig"]
        sc  = "#27ae60" if sig else "#e74c3c"
        content += (
            '<div style="padding:8px 12px;margin:8px 0;background:#f8f9fa;'
            'border-radius:6px;font-size:13px">'
            f'<b>Friedman test:</b> χ²({stats["k"]-1}) = {stats["friedman_stat"]:.3f}, '
            f'p = <span style="color:{sc};font-weight:bold">{stats["friedman_p"]:.4f}</span> '
            f'({"significant" if sig else "not significant"} at α={alpha})'
            "</div>"
        )

    content += "<h3 style='margin:16px 0 6px;font-size:14px'>Summary Table</h3>"
    content += _summary_table_html(stats, metric_name, alpha)
    content += _baseline_summary(stats, metric_name, alpha)

    panels = ""
    if "boxplot" in figs:
        buf = _fig_to_buf(figs["boxplot"])
        panels += _panel("Distribution", _buf_to_img_html(buf, style="height:auto;max-width:700px;"))
    if "cd" in figs:
        buf = _fig_to_buf(figs["cd"])
        panels += _panel("CD Diagram", _buf_to_img_html(buf, style="height:auto;max-width:700px;"))
    if "pvalue" in figs:
        buf = _fig_to_buf(figs["pvalue"])
        panels += _panel("P-value Heatmap", _buf_to_img_html(buf, style="height:auto;max-width:600px;"))
    if panels:
        content += f'<div class="panel-row">{panels}</div>'

    return content


def run_comparison(exp_data_list, compare_axis, all_conditions, condition_print_map,
                   fixed, baseline_key, metric, alpha):
    """Build matrix -> test -> plots -> HTML content. Returns (stats, figs, test_type, tab_html)."""
    matrix, condition_keys, exp_names = build_accuracy_matrix(
        exp_data_list, compare_axis, fixed, all_conditions)

    if matrix is None or len(condition_keys) < 2:
        html = '<p style="color:#e74c3c">Insufficient data for comparison.</p>'
        return None, {}, "none", html

    condition_names = [condition_print_map.get(c, str(c)) for c in condition_keys]

    baseline_idx = (condition_keys.index(baseline_key)
                    if baseline_key in condition_keys else None)

    metric_name = METRIC_LABEL.get(metric, metric)
    n_exp = matrix.shape[0]

    if n_exp >= 2:
        stats = run_friedman_posthoc(
            matrix, condition_keys, condition_names, baseline_idx, alpha=alpha)
        test_type = "friedman"
    else:
        single_exp_data = exp_data_list[0][1]
        stats = run_mcnemar_pairwise(
            single_exp_data, compare_axis, fixed,
            condition_keys, condition_names, baseline_idx, alpha=alpha)
        test_type = "mcnemar"

    figs = {}
    figs["boxplot"] = plot_accuracy_boxplot(
        matrix, condition_names, metric_name=metric_name, baseline_idx=baseline_idx)

    if n_exp >= 2 and "error" not in stats:
        figs["cd"] = plot_cd_diagram(
            stats["avg_ranks"], stats["cd"], condition_names,
            baseline_idx=baseline_idx,
            title=f"CD Diagram — {compare_axis.replace('_', ' ').title()} ({metric_name})")

    if "error" not in stats:
        figs["pvalue"] = plot_pvalue_heatmap(
            stats["adj_mat"], stats["rej_mat"], condition_names,
            baseline_idx=baseline_idx, alpha=alpha, metric_name=metric_name)

    html = build_tab_content(stats, figs, test_type, metric_name, alpha)
    return stats, figs, test_type, html
