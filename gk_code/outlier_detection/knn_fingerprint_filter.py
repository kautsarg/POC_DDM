#!/usr/bin/env python3
"""
KNN Fingerprint Filter for chip dPCR data.

Idea
----
Each chip dPCR amplification curve carries a "molecular fingerprint" — a
characteristic shape that identifies the target. Some curves in the raw dataset
do show amplification but their shape is noisy / atypical, hurting downstream
classification.

This script uses out-of-fold KNN probabilities to score every curve by
*fingerprint clarity*: the fraction of its k nearest neighbors that share its
true class. High score = typical curve shape; low score = atypical / messy.

We then keep the top-N curves per class (class-balanced) and save the filtered
dataset for use as a cleaner training set for the downstream Transformer.

Default input: 3Plex_raw.csv (14478 chip curves with HAdv / IAV / IBV labels).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import gc
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from outlier_utils import *

DEFAULT_INPUT = (
    "/Users/zz2820/Documents/DDM_on_chip/"
    "titan-processing-costanza-main/3Plex_raw.csv"
)
DEFAULT_OUTPUT = "/Users/zz2820/Documents/DDM_on_chip/knn_filter_output"


# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="KNN-based fingerprint-clarity filter for chip dPCR curves",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", type=str, default=DEFAULT_INPUT,
                   help="CSV with `Target` column + numeric curve columns")
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT,
                   help="Where to write filtered CSVs, plots, summaries")
    p.add_argument("--n_per_class", type=int, nargs="+", default=[500, 800],
                   help="Top-N samples to keep per class (multiple allowed)")
    p.add_argument("--k", type=int, default=20, help="KNN n_neighbors")
    p.add_argument("--cv_folds", type=int, default=5,
                   help="Folds for out-of-fold KNN scoring")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--target_col", type=str, default="Target",
                   help="Column with class labels (string)")
    p.add_argument("--scores_csv", type=str, default=None,
                   help="Path to existing fingerprint_scores.csv to skip recomputation")
    return p.parse_args()


# ─── Core: scoring & selection ───────────────────────────────────────────────
def get_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    """Numeric curve columns: everything except known metadata."""
    meta = {target_col, "Target", "Target_cat", "Channel", "PrimerMix",
            "Assay", "Conc", "Exp_id", "MeltPeaks", "Unnamed: 0"}
    return [c for c in df.columns if c not in meta]


def compute_fingerprint_scores(
    X: np.ndarray, y: np.ndarray, k: int, cv_folds: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Out-of-fold KNN predictions; score = P(true class | x)."""
    knn = KNeighborsClassifier(n_neighbors=k)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    y_proba = cross_val_predict(knn, X, y, cv=skf, method="predict_proba")
    y_pred = y_proba.argmax(axis=1)
    scores = y_proba[np.arange(len(y)), y]
    return scores, y_pred, y_proba


def select_top_n_per_class(
    scores: np.ndarray, y: np.ndarray, n_per_class: int, n_classes: int,
) -> np.ndarray:
    """Indices of top-N samples per class, sorted by score (desc)."""
    keep = []
    for c in range(n_classes):
        cls_mask = y == c
        masked_scores = np.where(cls_mask, scores, -np.inf)
        n_take = min(n_per_class, int(cls_mask.sum()))
        top = np.argsort(masked_scores)[::-1][:n_take]
        keep.extend(top.tolist())
    return np.array(sorted(keep))


# ─── Plotting ────────────────────────────────────────────────────────────────
def plot_score_distribution(
    scores: np.ndarray, y: np.ndarray, classes: list[str],
    save_path: Path,
) -> None:
    """Histogram of fingerprint scores, one panel per class."""
    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 4),
                             sharey=True)
    if len(classes) == 1:
        axes = [axes]
    for i, (ax, name) in enumerate(zip(axes, classes)):
        s = scores[y == i]
        ax.hist(s, bins=40, edgecolor="black", alpha=0.7,
                color=f"C{i}")
        ax.axvline(s.mean(), color="red", ls="--",
                   label=f"mean={s.mean():.3f}")
        ax.axvline(np.median(s), color="orange", ls=":",
                   label=f"median={np.median(s):.3f}")
        ax.set_title(f"{name}  (n={(y==i).sum()})")
        ax.set_xlabel("Fingerprint clarity score")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Count")
    fig.suptitle("Fingerprint clarity score per class (out-of-fold KNN)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_score_with_thresholds(
    scores: np.ndarray, y: np.ndarray, classes: list[str],
    threshold_per_class: dict[str, float], save_path: Path, n_per_class: int,
) -> None:
    """Score histogram with cutoff line for the top-N selection."""
    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 4),
                             sharey=True)
    if len(classes) == 1:
        axes = [axes]
    for i, (ax, name) in enumerate(zip(axes, classes)):
        s = scores[y == i]
        thresh = threshold_per_class[name]
        ax.hist(s[s >= thresh], bins=30, edgecolor="black",
                color="steelblue", alpha=0.85, label="kept")
        ax.hist(s[s < thresh], bins=30, edgecolor="black",
                color="lightcoral", alpha=0.7, label="removed")
        ax.axvline(thresh, color="black", ls="--", lw=2,
                   label=f"cutoff={thresh:.3f}")
        ax.set_title(f"{name}  kept={(s>=thresh).sum()}/{len(s)}")
        ax.set_xlabel("Fingerprint clarity score")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Count")
    fig.suptitle(f"Top-{n_per_class}/class selection")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mean_curves_kept_vs_removed(
    X: np.ndarray, y: np.ndarray, keep_mask: np.ndarray,
    classes: list[str], save_path: Path,
) -> None:
    """Mean ± std curve, kept vs removed, per class."""
    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 4),
                             sharey=True)
    if len(classes) == 1:
        axes = [axes]
    t = np.arange(X.shape[1])
    for i, (ax, name) in enumerate(zip(axes, classes)):
        cls_mask = y == i
        kept = cls_mask & keep_mask
        rem  = cls_mask & ~keep_mask
        if kept.sum() > 0:
            mu_k = X[kept].mean(axis=0)
            sd_k = X[kept].std(axis=0)
            ax.plot(t, mu_k, color="steelblue", lw=2,
                    label=f"kept (n={kept.sum()})")
            ax.fill_between(t, mu_k - sd_k, mu_k + sd_k, color="steelblue",
                            alpha=0.25)
        if rem.sum() > 0:
            mu_r = X[rem].mean(axis=0)
            sd_r = X[rem].std(axis=0)
            ax.plot(t, mu_r, color="lightcoral", lw=2,
                    label=f"removed (n={rem.sum()})")
            ax.fill_between(t, mu_r - sd_r, mu_r + sd_r, color="lightcoral",
                            alpha=0.25)
        ax.set_title(name)
        ax.set_xlabel("Time index")
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Fluorescence")
    fig.suptitle("Mean ± std curve: kept vs removed")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sample_curves(
    X: np.ndarray, y: np.ndarray, keep_mask: np.ndarray,
    classes: list[str], save_path: Path, n_show: int = 8, seed: int = 42,
) -> None:
    """Random sample curves: kept (top row) vs removed (bottom row), per class."""
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, len(classes), figsize=(5 * len(classes), 6),
                             sharex=True, sharey=True)
    if len(classes) == 1:
        axes = axes.reshape(2, 1)
    t = np.arange(X.shape[1])
    for i, name in enumerate(classes):
        cls_mask = y == i
        kept_idx = np.where(cls_mask & keep_mask)[0]
        rem_idx  = np.where(cls_mask & ~keep_mask)[0]
        if len(kept_idx):
            sel = rng.choice(kept_idx, min(n_show, len(kept_idx)), replace=False)
            for s in sel:
                axes[0, i].plot(t, X[s], color="steelblue", alpha=0.6, lw=0.8)
        axes[0, i].set_title(f"{name} — KEPT")
        if len(rem_idx):
            sel = rng.choice(rem_idx, min(n_show, len(rem_idx)), replace=False)
            for s in sel:
                axes[1, i].plot(t, X[s], color="lightcoral", alpha=0.6, lw=0.8)
        axes[1, i].set_title(f"{name} — REMOVED")
        axes[1, i].set_xlabel("Time index")
    axes[0, 0].set_ylabel("Fluorescence")
    axes[1, 0].set_ylabel("Fluorescence")
    fig.suptitle("Random sample curves: kept (top) vs removed (bottom)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_amplitude_range_distribution(
    X: np.ndarray, y: np.ndarray, keep_mask: np.ndarray,
    classes: list[str], save_path: Path,
) -> None:
    """Amplitude-range histograms (max - min), kept vs removed, per class."""
    ranges = X.max(axis=1) - X.min(axis=1)
    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 4),
                             sharey=True)
    if len(classes) == 1:
        axes = [axes]
    for i, (ax, name) in enumerate(zip(axes, classes)):
        cls_mask = y == i
        kept = ranges[cls_mask & keep_mask]
        rem  = ranges[cls_mask & ~keep_mask]
        if len(rem):
            ax.hist(rem, bins=40, color="lightcoral", alpha=0.7,
                    edgecolor="black",
                    label=f"removed (μ={rem.mean():.3f})")
        if len(kept):
            ax.hist(kept, bins=40, color="steelblue", alpha=0.7,
                    edgecolor="black",
                    label=f"kept (μ={kept.mean():.3f})")
        ax.set_title(name)
        ax.set_xlabel("Amplitude range (max − min)")
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Count")
    fig.suptitle("Amplitude-range distribution: kept vs removed")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── Per-N analysis pipeline ─────────────────────────────────────────────────
def analyze_one_n(
    df: pd.DataFrame, X: np.ndarray, y: np.ndarray, scores: np.ndarray,
    classes: list[str], n_per_class: int, args: argparse.Namespace,
    output_root: Path,
) -> dict:
    """Run filter + analysis for one N value, return summary dict."""
    n_classes = len(classes)
    keep_idx = select_top_n_per_class(scores, y, n_per_class, n_classes)
    keep_mask = np.zeros(len(df), dtype=bool)
    keep_mask[keep_idx] = True

    sub_dir = output_root / f"top{n_per_class}_per_class"
    sub_dir.mkdir(parents=True, exist_ok=True)

    # Save the filtered CSV (preserves original columns)
    out_csv = output_root / f"3plex_raw_knn_top{n_per_class}.csv"
    df.iloc[keep_idx].reset_index(drop=True).to_csv(out_csv, index=False)

    # Per-class statistics
    threshold_per_class = {}
    per_class_stats = {}
    for c, name in enumerate(classes):
        cls_mask = y == c
        kept = cls_mask & keep_mask
        kept_scores = scores[kept]
        per_class_stats[name] = {
            "n_total":     int(cls_mask.sum()),
            "n_kept":      int(kept.sum()),
            "n_removed":   int(cls_mask.sum() - kept.sum()),
            "score_min":   float(kept_scores.min()),
            "score_max":   float(kept_scores.max()),
            "score_mean":  float(kept_scores.mean()),
            "score_std":   float(kept_scores.std()),
        }
        threshold_per_class[name] = float(kept_scores.min())

    # Sanity check: KNN CV accuracy on filtered set
    knn_check = KNeighborsClassifier(n_neighbors=args.k)
    y_pred_filt = cross_val_predict(
        knn_check, X[keep_idx], y[keep_idx],
        cv=StratifiedKFold(n_splits=args.cv_folds, shuffle=True,
                           random_state=args.seed),
    )
    acc_filt = accuracy_score(y[keep_idx], y_pred_filt) * 100
    cm_filt = confusion_matrix(y[keep_idx], y_pred_filt)

    # Plots
    plot_score_with_thresholds(
        scores, y, classes, threshold_per_class,
        sub_dir / "01_score_with_cutoffs.png", n_per_class)
    plot_mean_curves_kept_vs_removed(
        X, y, keep_mask, classes, sub_dir / "02_mean_curves.png")
    plot_sample_curves(
        X, y, keep_mask, classes,
        sub_dir / "03_sample_curves.png", seed=args.seed)
    plot_amplitude_range_distribution(
        X, y, keep_mask, classes, sub_dir / "04_amplitude_range.png")

    summary = {
        "n_per_class":          n_per_class,
        "n_kept_total":         int(keep_mask.sum()),
        "n_original":           int(len(df)),
        "kept_pct":             float(keep_mask.mean() * 100),
        "filtered_csv":         str(out_csv),
        "per_class":            per_class_stats,
        "score_thresholds":     threshold_per_class,
        "knn_cv_acc_filtered":  float(acc_filt),
        "confusion_matrix":     cm_filt.tolist(),
        "classes":              classes,
    }

    # Plain-text report
    lines = [
        f"=== KNN Fingerprint Filter — top-{n_per_class} per class ===",
        f"Input  : {args.input}",
        f"Output : {out_csv}",
        f"",
        f"Total kept       : {keep_mask.sum()} / {len(df)}  "
        f"({keep_mask.mean()*100:.2f}%)",
        f"KNN CV acc (k={args.k}, {args.cv_folds}-fold) on filtered: "
        f"{acc_filt:.2f}%",
        f"",
        f"Per-class:",
    ]
    for name, st in per_class_stats.items():
        lines.append(
            f"  {name:>6s}: kept {st['n_kept']:>5d} / {st['n_total']:>5d}  "
            f"score range [{st['score_min']:.3f}, {st['score_max']:.3f}]  "
            f"mean={st['score_mean']:.3f}  std={st['score_std']:.3f}"
        )
    lines.append("")
    lines.append("Confusion matrix on filtered set (rows=true, cols=pred):")
    lines.append("       " + "  ".join(f"{c:>6s}" for c in classes))
    for i, name in enumerate(classes):
        row = "  ".join(f"{v:>6d}" for v in cm_filt[i])
        lines.append(f"  {name:>4s}  {row}")
    report = "\n".join(lines) + "\n"
    (sub_dir / "summary.txt").write_text(report)
    (sub_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(report)
    return summary

def run_knnfilter_pipeline(dataset_names, dataset_curves, Y_well, ref_curves, knn_plot_path, keep_pcts=[0.25, 0.50, 0.75], save_plot=True):
    os.makedirs(knn_plot_path, exist_ok=True)
    
    le = LabelEncoder()
    y = le.fit_transform(Y_well)
    classes = list(le.classes_)

    results_dfs = []

    for i, (name, curves) in enumerate(zip(dataset_names, dataset_curves)):
        clean_title = name.replace("_", " ").title()
        print(f"  -> Running KNN Filtering for {clean_title}...")

        invalid_mask = np.isnan(curves).any(axis=1) | np.isinf(curves).any(axis=1)
        valid_mask = ~invalid_mask
        
        X_valid = curves[valid_mask]
        y_valid = y[valid_mask]
        
        full_scores = np.full(len(y), -1.0)
        full_proba = np.zeros((len(y), len(classes)))
        
        if len(X_valid) > 5:
            scores_valid, _, y_proba_valid = compute_fingerprint_scores(X_valid, y_valid, k=20, cv_folds=5, seed=0)
            
            full_scores[valid_mask] = scores_valid
    
            valid_classes = np.unique(y_valid)
            valid_indices = np.where(valid_mask)[0]
            
            for col_idx, class_idx in enumerate(valid_classes):
                full_proba[valid_indices, class_idx] = y_proba_valid[:, col_idx]

        results_filter = {f"knn_top_{pct}": np.full(len(y), -1) for pct in keep_pcts}
        results_filter["knn_proba"] = full_proba
        
        for c, c_name in enumerate(classes):
            class_mask = (y == c)
            class_indices = np.where(class_mask)[0]
            class_scores = full_scores[class_mask]
            
            ranks = np.argsort(np.argsort(-class_scores)) + 1
            n_elements = len(class_scores)

            for pct in keep_pcts:
                cutoff_rank = int(np.floor(n_elements * pct))
                
                if cutoff_rank < 1 and n_elements > 0:
                    cutoff_rank = 1
                    
                keep_in_class_mask = ranks <= cutoff_rank
                
                valid_keep_mask = keep_in_class_mask & (class_scores != -1.0)
                
                global_keep_indices = class_indices[valid_keep_mask]
                results_filter[f"knn_top_{pct}"][global_keep_indices] = 1

        # Build DataFrame
        filtered_dict = {k: v for k, v in results_filter.items() if k != "knn_proba"}
        new_features = pd.DataFrame(filtered_dict)
        results_dfs.append(new_features)
        
        # --- NEW PLOTTING BLOCK ---
        if save_plot:
            unique_wells = np.unique(Y_well)
            for pct in keep_pcts:
                html = init_html_report(
                    title=f"KNN Fingerprint Outliers: {clean_title}", 
                    subtitle=f"Kept Top {pct * 100}% of Curves per Well"
                )
                
                for well in unique_wells:
                    well_mask = (Y_well == well)
                    is_outlier = (new_features.loc[well_mask, f"knn_top_{pct}"] == -1).fillna(False).values
                    
                    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                    
                    # Ref Normal / Outlier
                    axes[0,0].plot(ref_curves[well_mask][~is_outlier].T, c="blue", alpha=0.3, rasterized=True)
                    axes[0,0].set_title(f"Well {well} - Ref Normal")
                    axes[0,1].plot(ref_curves[well_mask][is_outlier].T, c="red", alpha=0.5, rasterized=True)
                    axes[0,1].set_title(f"Well {well} - Ref Outliers")
                    
                    # Curr Normal / Outlier
                    axes[1,0].plot(curves[well_mask][~is_outlier].T, c="green", alpha=0.3, rasterized=True)
                    axes[1,0].set_title(f"Curr Normal")
                    axes[1,1].plot(curves[well_mask][is_outlier].T, c="red", alpha=0.5, rasterized=True)
                    axes[1,1].set_title(f"Curr Outliers")
                    
                    plt.tight_layout()
                    html += f"<div style='background: white; padding: 10px; border-radius: 8px; width: 30%; min-width: 400px;'><img src='data:image/png;base64,{fig_to_base64(fig)}' width='100%'></div>"
                    
                html += "</div></body></html>"
                save_path = os.path.join(knn_plot_path, f"{name}_knn_top_{pct}.html")
                with open(save_path, "w") as f: 
                    f.write(html)
                    
        gc.collect()
    
    return results_dfs

# ─── Main ────────────────────────────────────────────────────────────────────
def main() -> int:
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.input}")
    df = pd.read_csv(args.input)
    feature_cols = get_feature_columns(df, args.target_col)
    if not feature_cols:
        raise ValueError("No numeric feature columns found.")
    X = df[feature_cols].values.astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df[args.target_col].values)
    classes = list(le.classes_)

    print(f"Shape         : X={X.shape}, y={y.shape}")
    print(f"Classes       : {classes}")
    print(f"Class counts  : "
          f"{dict(zip(classes, np.bincount(y).tolist()))}")
    print(f"Feature cols  : {len(feature_cols)} (first={feature_cols[0]}, "
          f"last={feature_cols[-1]})")
    print()

    # Step 1: out-of-fold scores (computed once, or loaded from existing file)
    if args.scores_csv is not None:
        print(f"Loading existing fingerprint scores from: {args.scores_csv}")
        scores_df = pd.read_csv(args.scores_csv)
        scores = scores_df["score"].values.astype(np.float32)
        baseline_acc = None
        print("(Scores loaded — skipping out-of-fold score computation)")
        print("Baseline KNN CV accuracy on full data: (see previous run)")
    else:
        print(f"Computing fingerprint clarity scores  "
              f"(k={args.k}, {args.cv_folds}-fold CV)...")
        scores, y_pred, _ = compute_fingerprint_scores(
            X, y, k=args.k, cv_folds=args.cv_folds, seed=args.seed)
        baseline_acc = accuracy_score(y, y_pred) * 100
        print(f"Baseline KNN CV accuracy on full data: {baseline_acc:.2f}%")
    print()
    print("Per-class score statistics:")
    for c, name in enumerate(classes):
        s = scores[y == c]
        print(f"  {name:>6s}: mean={s.mean():.3f}  median={np.median(s):.3f}  "
              f"std={s.std():.3f}  <0.5={(s<0.5).mean()*100:.1f}%")
    print()

    # Save the once-per-run score plot
    plot_score_distribution(
        scores, y, classes, output_root / "score_distribution.png")

    # Persist scores so users can rerun analyses without recomputing
    pd.DataFrame({
        "row_index": np.arange(len(df)),
        args.target_col: df[args.target_col].values,
        "true_class": y,
        "score": scores,
    }).to_csv(output_root / "fingerprint_scores.csv", index=False)

    # Step 2: per-N filter + analysis
    summaries = {}
    for n in args.n_per_class:
        print("=" * 70)
        print(f"Top-{n} per class")
        print("=" * 70)
        summaries[n] = analyze_one_n(
            df, X, y, scores, classes, n, args, output_root)

    # Run-level summary
    run_summary = {
        "timestamp":          datetime.now().isoformat(timespec="seconds"),
        "input":              args.input,
        "output_dir":         str(output_root),
        "k":                  args.k,
        "cv_folds":           args.cv_folds,
        "seed":               args.seed,
        "classes":            classes,
        "class_counts":       {classes[c]: int((y == c).sum())
                               for c in range(len(classes))},
        "baseline_knn_acc":   float(baseline_acc) if baseline_acc is not None else None,
        "n_values":           args.n_per_class,
        "summaries":          summaries,
    }
    (output_root / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2))
    print(f"\nAll outputs written to: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
