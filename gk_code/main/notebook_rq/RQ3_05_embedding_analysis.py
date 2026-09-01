import os
import sys
import time
import argparse
import importlib
from pathlib import Path

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

_MAIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_MAIN_DIR))
sys.path.insert(0, str(_MAIN_DIR / "utils"))
sys.path.insert(0, str(_MAIN_DIR / "utils" / "model_training"))
os.chdir(_MAIN_DIR)

import config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

pred08 = importlib.import_module("08_cross_dataset_predict_new_chip")
vis07 = importlib.import_module("07_attribution_vis_all")
b06 = importlib.import_module("06b_cross_dataset_prediction_report")

# Same constants as RQ3_02_loco_results.ipynb's final_6_new comparison.
EXP_FOLDER = config.FINAL_EXP_FOLDER
GROUP_NAME = "final_6_new"
CURVE_TYPE = "ori_curve_sg_p4_norm"
FILTER_KEY = "noamp_remove"
CURVE_ALIGNMENT = "pc_ttp"
PC_TTP_ANCHOR = "min"
TRAIN_CENTER_FRAC = 0.5

# Copied from RQ3_02_loco_results.ipynb's currently-active RQ3_2_BASE_MODELS /
# PC_RECENTER_BASES (knn/coral/dann_conc commented out there too) -- kept as two
# separate lists here in case they diverge later, update both if that notebook changes.
BASE_MODELS = [
    "cnn_gru_dual", "cnn_gru_dual_attn_recon",
    "cnn_gru_dual_attn_recon_dann", "cnn_gru_dual_attn_recon_supcon3",
    "cnn_gru_dual_attn_recon_aug", "cnn_gru_dual_attn_recon_mtl",
]
PC_RECENTER_BASES = [
    "cnn_gru_dual", "cnn_gru_dual_attn_recon",
    "cnn_gru_dual_attn_recon_dann", "cnn_gru_dual_attn_recon_supcon3",
    "cnn_gru_dual_attn_recon_aug", "cnn_gru_dual_attn_recon_mtl",
]
ALL_MODELS = BASE_MODELS + [f"{m}_pc_recenter" for m in PC_RECENTER_BASES]

OUT_DIR = Path(__file__).resolve().parent / "RQ3_05_embedding_analysis"

_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def short_name(chip_name):
    return chip_name.split("_U_", 1)[1]


def group_out_dir():
    base_dir = Path(EXP_FOLDER) / "cross_dataset_cv" / GROUP_NAME
    return b06.alignment_dir(base_dir, CURVE_ALIGNMENT, PC_TTP_ANCHOR)


def mapped_labels(chip_name, Y_well_raw):
    mapping = config.LABEL_MAPPINGS.get(chip_name)
    if mapping is None:
        return np.asarray(Y_well_raw)
    return np.array([mapping.get(w, w) for w in Y_well_raw])


def load_fold(group_dir, chips, held_out, rng, batch_n):
    """Aligns + subsamples all chips once for this fold (shared across every model)."""
    resampler_path = pred08._resolve_alignment_path(
        group_dir, config.CROSS_DATASET_RESAMPLER_PATH, CURVE_TYPE, held_out)
    if not resampler_path.exists():
        print(f"  [!] fold {held_out}: no resampler at {resampler_path}, skipping fold.")
        return None, None
    resampler = joblib.load(resampler_path)

    aligned = {}
    for chip in chips:
        chip_path = Path(EXP_FOLDER, chip)
        result = pred08.align_new_chip(chip_path, group_dir, CURVE_TYPE, CURVE_ALIGNMENT, PC_TTP_ANCHOR,
                                       group_name=GROUP_NAME, held_out_chip=held_out)
        if result is None:
            print(f"  [!] fold {held_out}: could not align {chip}, dropping it from this fold's plots.")
            continue
        curves, _, Y_well_raw, pc_curves_aligned, coords, well_ids = result
        n = min(batch_n, len(curves))
        idx = rng.choice(len(curves), size=n, replace=False)
        aligned[chip] = dict(
            curves=curves[idx], coords=(coords[idx] if coords is not None else None),
            well_ids=(well_ids[idx] if well_ids is not None else None),
            y_labels=mapped_labels(chip, Y_well_raw)[idx], pc_curves=pc_curves_aligned)
    return aligned, resampler


def model_embeddings(model, base_model, aligned):
    """(embeddings, own_pc_embed) per chip, for this fold's already-aligned+subsampled data."""
    is_spatial = pred08._is_spatial(base_model)
    k = pred08._infer_k(model) if is_spatial else None
    per_chip_emb, per_chip_own_pc = {}, {}
    for chip, d in aligned.items():
        if is_spatial:
            if d["coords"] is None or d["well_ids"] is None:
                continue
            stack = pred08.build_neighbor_curve_stack(d["curves"], d["coords"], d["well_ids"], k)
            emb = pred08.compute_embeddings_stack(model, stack)
            own_pc = (pred08.compute_embeddings_stack(model, pred08._pc_mean_stack(d["pc_curves"], k))[0]
                     if d["pc_curves"] is not None and len(d["pc_curves"]) else None)
        else:
            emb = pred08.compute_embeddings(model, d["curves"])
            own_pc = (pred08.compute_embeddings(model, d["pc_curves"]).mean(axis=0)
                     if d["pc_curves"] is not None and len(d["pc_curves"]) else None)
        per_chip_emb[chip] = emb
        per_chip_own_pc[chip] = own_pc
    return per_chip_emb, per_chip_own_pc, k


def pc_recenter(per_chip_emb, per_chip_own_pc, ref_embed):
    recentered = {}
    for chip, emb in per_chip_emb.items():
        own = per_chip_own_pc.get(chip)
        if own is None:
            print(f"  [!] {chip}: no PC embedding for recentering, keeping raw.")
            recentered[chip] = emb
            continue
        recentered[chip] = emb + (ref_embed - own)
    return recentered


def _scatter_train_held(ax, xy, labels, is_held, color_map):
    train_mask = ~is_held
    for lab, color in color_map.items():
        m = train_mask & (labels == lab)
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.35, color=color, marker="o", linewidths=0)
    for lab, color in color_map.items():
        m = is_held & (labels == lab)
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=55, alpha=0.9, color=color, marker="X",
                      edgecolors="black", linewidths=0.6, zorder=5)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_model_grid(model_key, fold_results, chip_colors, seed, save_path):
    n_folds = len(fold_results)
    fig, axes = plt.subplots(n_folds, 2, figsize=(11, 4.3 * n_folds), facecolor="white", squeeze=False)

    all_targets = sorted({lab for fr in fold_results for lab in np.unique(fr["labels"])})
    target_colors = {t: _PALETTE[i % len(_PALETTE)] for i, t in enumerate(all_targets)}

    for row, fr in enumerate(fold_results):
        ts = TSNE(n_components=2, random_state=seed, init="pca", perplexity=30).fit_transform(fr["embeddings"])
        is_held = fr["chip_ids"] == fr["held_out"]

        _scatter_train_held(axes[row, 0], ts, fr["labels"], is_held, target_colors)
        _scatter_train_held(axes[row, 1], ts, fr["chip_ids"], is_held, chip_colors)
        axes[row, 0].set_ylabel(short_name(fr["held_out"]), fontsize=10, fontweight="bold")

    axes[0, 0].set_title("Coloured by target", fontsize=11, fontweight="bold")
    axes[0, 1].set_title("Coloured by chip", fontsize=11, fontweight="bold")

    target_handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=c, markersize=7, label=str(t))
                      for t, c in target_colors.items()]
    chip_handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=c, markersize=7,
                           label=short_name(chip)) for chip, c in chip_colors.items()]
    shape_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="gray", markersize=6, label="training chip"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="gray", markeredgecolor="black",
              markersize=8, label="held-out chip"),
    ]
    fig.legend(handles=target_handles, title="target", fontsize=7, title_fontsize=8,
              loc="upper left", bbox_to_anchor=(1.0, 0.98), frameon=False)
    fig.legend(handles=chip_handles, title="chip", fontsize=7, title_fontsize=8,
              loc="upper left", bbox_to_anchor=(1.0, 0.62), frameon=False)
    fig.legend(handles=shape_handles, title="marker", fontsize=7, title_fontsize=8,
              loc="upper left", bbox_to_anchor=(1.0, 0.28), frameon=False)

    fig.suptitle(f"{model_key} -- {GROUP_NAME}  (row = held-out fold)", fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(batch_n, seed):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chips = list(config.CROSS_DATASET_GROUPS[GROUP_NAME])
    chip_colors = {chip: _PALETTE[i % len(_PALETTE)] for i, chip in enumerate(chips)}
    exp_paths_all = [Path(EXP_FOLDER, c) for c in chips]
    group_dir = group_out_dir()
    rng = np.random.default_rng(seed)

    fold_results = {m: [] for m in ALL_MODELS}

    for held_out in chips:
        t0 = time.perf_counter()
        print(f"\n{'-'*70}\n[fold] held out = {held_out}\n{'-'*70}")
        aligned, resampler = load_fold(group_dir, chips, held_out, rng, batch_n)
        if aligned is None or len(aligned) < 2:
            print(f"  [!] fold {held_out}: fewer than 2 chips aligned, skipping fold.")
            continue

        model_dir = group_dir / "model_interpretation" / f"lofo_{held_out}"
        keras_models = vis07.load_saved_models(model_dir, FILTER_KEY, len(resampler.t_grid),
                                               curve_type=CURVE_TYPE, model_names=BASE_MODELS,
                                               train_center_frac=TRAIN_CENTER_FRAC)

        for base_model in BASE_MODELS:
            model = keras_models.get(base_model)
            if model is None:
                print(f"  [!] fold {held_out}: missing {base_model}.keras, skipping.")
                continue

            per_chip_emb, per_chip_own_pc, _ = model_embeddings(model, base_model, aligned)
            if len(per_chip_emb) < 2:
                continue

            def _stack(emb_by_chip):
                cs = list(emb_by_chip.keys())
                return dict(
                    held_out=held_out,
                    embeddings=np.concatenate([emb_by_chip[c] for c in cs], axis=0),
                    labels=np.concatenate([aligned[c]["y_labels"] for c in cs], axis=0),
                    chip_ids=np.concatenate([np.full(len(emb_by_chip[c]), c) for c in cs]))

            fold_results[base_model].append(_stack(per_chip_emb))

            if base_model in PC_RECENTER_BASES:
                ref_embed = pred08.reference_pc_embedding(model, base_model, exp_paths_all, group_dir,
                                                          CURVE_TYPE, FILTER_KEY, CURVE_ALIGNMENT,
                                                          held_out_chip=held_out)
                recentered = pc_recenter(per_chip_emb, per_chip_own_pc, ref_embed)
                fold_results[f"{base_model}_pc_recenter"].append(_stack(recentered))

        tf.keras.backend.clear_session()
        print(f"  [+] fold {held_out} done in {time.perf_counter()-t0:.1f}s")

    for model_key in ALL_MODELS:
        frs = fold_results[model_key]
        if not frs:
            print(f"  [!] {model_key}: no fold produced results, skipping plot.")
            continue
        save_path = OUT_DIR / f"{model_key}_tsne_grid.png"
        plot_model_grid(model_key, frs, chip_colors, seed, save_path)
        print(f"  [+] saved -> {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-model 6(fold) x 2(target/chip) t-SNE embedding grid for the "
                    "final_6_new LOFO comparison (RQ3_02_loco_results.ipynb's model list).")
    parser.add_argument("--batch_n", type=int, default=800, metavar="N",
                        help="Samples drawn per chip for the t-SNE (default: 800).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\n{'='*70}\n[RUNNING] RQ3_05_embedding_analysis.py\n{'='*70}\n")
    run(args.batch_n, args.seed)
    print(f"\n{'='*70}\n[DONE] RQ3_05_embedding_analysis.py\n{'='*70}\n")
