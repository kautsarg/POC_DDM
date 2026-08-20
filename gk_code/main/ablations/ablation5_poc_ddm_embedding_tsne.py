import os
import sys
import argparse
import importlib
import joblib
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

_MAIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_MAIN_DIR))
sys.path.insert(0, str(_MAIN_DIR / "utils"))
sys.path.insert(0, str(_MAIN_DIR / "utils" / "model_training"))

import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

os.chdir(_MAIN_DIR)
cdt = importlib.import_module("04_cross_dataset_training")
vis07 = importlib.import_module("07_attribution_vis_all")
pred08 = importlib.import_module("08_cross_dataset_predict_new_chip")

EXP_FOLDER = "/vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_final_nc_subtract"
GROUP_NAME = "final_4_chip_clean_nn"
DATASETS = [
    "D20260806_E00_C00_F4500KHz_U_DDM_01_06",
    "D20260807_E00_C00_F4500KHz_U_DDM_02_07",
    "D20260808_E00_C00_F4500KHz_U_DDM_03_01",
    "D20260810_E00_C00_F4500KHz_U_DDM_04_01",
]
MODEL_KEYS = ['cnn_gru_dual', 'cnn_gru_dual_attn_recon']
CURVE_TYPES = ['ori_curve_norm', 'ori_curve_sg_p4_norm']
CURVE_ALIGNMENT = "pc_ttp"
PC_TTP_ANCHOR = "min"
FILTER_KEY = "noamp_remove"
PC_LABEL = "PC"

_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def short_name(folder):
    return folder.split('_U_', 1)[1]


def group_out_dir():
    d = Path(EXP_FOLDER) / "cross_dataset_cv" / GROUP_NAME
    if CURVE_ALIGNMENT == "pc_ttp":
        d = d / "curve_alignment_pc_ttp" / f"anchor_{PC_TTP_ANCHOR}"
    return d


def mapped_labels(exp_path, Y_well_raw):
    label_mappings = config.get_label_mappings(exp_path)
    mapping = label_mappings.get(exp_path.name)
    if mapping is None:
        return np.asarray(Y_well_raw)
    return np.array([mapping.get(w, w) for w in Y_well_raw])


def chip_embeddings(model, model_key, curve_type, chip_path, out_dir, batch_n, seed):
    result = pred08.align_new_chip(chip_path, out_dir, curve_type, CURVE_ALIGNMENT,
                                   PC_TTP_ANCHOR, group_name=GROUP_NAME)
    if result is None:
        print(f"  [!] {chip_path.name}: could not align, skipping.")
        return None
    curves, resampler, Y_well_raw, pc_curves_aligned, coords, well_ids = result
    y_labels = mapped_labels(chip_path, Y_well_raw)

    is_spatial = pred08._is_spatial(model_key)
    if is_spatial and (coords is None or well_ids is None):
        print(f"  [!] {chip_path.name}: no spatial metadata, skipping (needed for {model_key}).")
        return None

    rng = np.random.default_rng(seed)
    n = min(batch_n, len(curves))
    idx = rng.choice(len(curves), size=n, replace=False)

    if is_spatial:
        k = pred08._infer_k(model)
        X_full = pred08.build_neighbor_curve_stack(curves, coords, well_ids, k)
        embeddings = pred08.compute_embeddings_stack(model, X_full[idx])
        own_pc_embed = (pred08.compute_embeddings_stack(model, pred08._pc_mean_stack(pc_curves_aligned, k))[0]
                       if pc_curves_aligned is not None and len(pc_curves_aligned) else None)
    else:
        embeddings = pred08.compute_embeddings(model, curves[idx])
        own_pc_embed = (pred08.compute_embeddings(model, pc_curves_aligned).mean(axis=0)
                       if pc_curves_aligned is not None and len(pc_curves_aligned) else None)

    return dict(embeddings=embeddings, y_labels=y_labels[idx], own_pc_embed=own_pc_embed)


def pc_recenter(embeddings, chip_ids, ref_embed, own_pc_embeds):
    recentered = embeddings.copy()
    for chip, own_embed in own_pc_embeds.items():
        chip_mask = chip_ids == chip
        if own_embed is None:
            print(f"  [!] {chip}: no PC embedding -- not recentered.")
            continue
        recentered[chip_mask] = embeddings[chip_mask] + (ref_embed - own_embed)
    return recentered


def plot_tsne_grid(emb_raw, emb_recentered, y_labels, chip_ids, title, save_path, seed):
    classes = sorted(np.unique(y_labels))
    chips = sorted(np.unique(chip_ids))
    class_color = {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(classes)}
    chip_color = {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(chips)}

    ts_raw = TSNE(n_components=2, random_state=seed, init='pca', perplexity=30).fit_transform(emb_raw)
    ts_rec = TSNE(n_components=2, random_state=seed, init='pca', perplexity=30).fit_transform(emb_recentered)

    fig, axes = plt.subplots(2, 2, figsize=(13, 12), facecolor='white')

    def _scatter(ax, xy, labels, color_map, legend_title):
        for lab in sorted(color_map):
            m = labels == lab
            ax.scatter(xy[m, 0], xy[m, 1], s=8, alpha=0.6, color=color_map[lab], label=str(lab))
        ax.legend(title=legend_title, fontsize=7, markerscale=1.5, framealpha=0.85, loc='best')
        ax.set_xticks([])
        ax.set_yticks([])

    _scatter(axes[0, 0], ts_raw, y_labels, class_color, "target")
    axes[0, 0].set_title("Raw embedding — colour by target", fontsize=10, fontweight='bold')
    _scatter(axes[0, 1], ts_raw, chip_ids, chip_color, "chip")
    axes[0, 1].set_title("Raw embedding — colour by chip", fontsize=10, fontweight='bold')
    _scatter(axes[1, 0], ts_rec, y_labels, class_color, "target")
    axes[1, 0].set_title("PC-recentered embedding — colour by target", fontsize=10, fontweight='bold')
    _scatter(axes[1, 1], ts_rec, chip_ids, chip_color, "chip")
    axes[1, 1].set_title("PC-recentered embedding — colour by chip", fontsize=10, fontweight='bold')

    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def discover_lofo_dirs(group_dir):
    model_root = group_dir / "model_interpretation"
    if not model_root.is_dir():
        return []
    return sorted(p.name for p in model_root.iterdir() if p.is_dir() and p.name.startswith("lofo_"))


def run_one(model_key, curve_type, batch_n, seed, save_dir, model_dir_name, held_out_chip=None):
    group_dir = group_out_dir()
    model_dir = group_dir / "model_interpretation" / model_dir_name
    resampler_path = group_dir / config.CROSS_DATASET_RESAMPLER_PATH.format(curve_type=curve_type)
    if not resampler_path.exists():
        print(f"  [!] Missing resampler, skipping: {resampler_path}")
        return
    resampler = joblib.load(resampler_path)

    models = vis07.load_saved_models(model_dir, FILTER_KEY, len(resampler.t_grid),
                                     curve_type=curve_type, model_names=[model_key])
    model = models.get(model_key)
    if model is None:
        print(f"  [!] Missing model, skipping: {model_dir}/{model_key}_{FILTER_KEY}_{curve_type}_model.keras")
        return

    exp_paths = [Path(EXP_FOLDER, name) for name in DATASETS]
    per_chip = {}
    for chip_path in exp_paths:
        res = chip_embeddings(model, model_key, curve_type, chip_path, group_dir, batch_n, seed)
        if res is not None:
            per_chip[short_name(chip_path.name)] = res

    if len(per_chip) < 2:
        print("  [!] Fewer than 2 chips produced embeddings, skipping plot.")
        tf.keras.backend.clear_session()
        return

    if model_dir_name == "full_data_lofo":
        ref_embed = pred08.reference_pc_embedding(model, model_key, exp_paths, group_dir, curve_type,
                                                  FILTER_KEY, CURVE_ALIGNMENT)
    else:
        # bypass reference_pc_embedding's cache: it's keyed by model_key/curve_type only, not fold
        k = pred08._infer_k(model) if pred08._is_spatial(model_key) else None
        ref_embed = pred08._build_training_pc_embeddings(
            model, exp_paths, group_dir, curve_type, CURVE_ALIGNMENT, k=k).mean(axis=0)

    embeddings = np.concatenate([v["embeddings"] for v in per_chip.values()], axis=0)
    y_labels = np.concatenate([v["y_labels"] for v in per_chip.values()], axis=0)
    chip_ids = np.concatenate([np.full(len(v["embeddings"]), chip) for chip, v in per_chip.items()])
    own_pc_embeds = {chip: v["own_pc_embed"] for chip, v in per_chip.items()}

    emb_rec = pc_recenter(embeddings, chip_ids, ref_embed, own_pc_embeds)
    tf.keras.backend.clear_session()

    save_path = save_dir / f"{model_key}_{curve_type}_{model_dir_name}_tsne.png"
    title = f"{model_key} | {curve_type} | {GROUP_NAME} | {model_dir_name}"
    if held_out_chip:
        title += f" (held out: {held_out_chip})"
    plot_tsne_grid(embeddings, emb_rec, y_labels, chip_ids, title, save_path, seed)
    print(f"  [+] saved -> {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="2x2 t-SNE (target/chip x raw/PC-recentered) of the cross-dataset "
                    "embedding, per (model, curve_type) combination -- for the full_data_lofo "
                    "model and every available LOFO held-out-fold model.")
    parser.add_argument("--batch_n", type=int, default=800, metavar="N",
                        help="Samples drawn per chip for the t-SNE (default: 800).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\n{'='*70}\n[RUNNING] ablation5_poc_ddm_embedding_tsne.py\n{'='*70}\n")

    save_dir = group_out_dir() / "embedding_tsne"
    save_dir.mkdir(parents=True, exist_ok=True)
    lofo_dirs = discover_lofo_dirs(group_out_dir())
    print(f"Found {len(lofo_dirs)} LOFO fold dir(s): {lofo_dirs}")

    for curve_type in CURVE_TYPES:
        for model_key in MODEL_KEYS:
            print(f"\n{'-'*70}\n[{model_key} | {curve_type}]\n{'-'*70}")
            run_one(model_key, curve_type, args.batch_n, args.seed, save_dir, "full_data_lofo")

            for lofo_dir in lofo_dirs:
                held_out = short_name(lofo_dir[len("lofo_"):])
                run_one(model_key, curve_type, args.batch_n, args.seed, save_dir, lofo_dir,
                       held_out_chip=held_out)

    print(f"\n{'='*70}\n[DONE] ablation5_poc_ddm_embedding_tsne.py\n{'='*70}\n")
