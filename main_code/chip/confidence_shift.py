import os
import re
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "utils"))
sys.path.insert(0, str(_ROOT / "utils" / "model_training"))

import config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

from pc_recentering import align_new_chip, load_saved_models, is_spatial_model, predict_new_chip
from cross_dataset_result_io import load_partitioned, frac_suffix

GROUP_NAME = "final_6_new"
CURVE_TYPE = "ori_curve_sg_p4_norm"
FILTER_KEY = "noamp_remove"
TRAIN_CENTER_FRAC = 0.5
CV_MODE_STR = "kfold5"
LOFO_MODE_STR = "lofo"

CV_MODELS = ['knn', 'cnn_gru_dual', 'cnn_gru_dual_attn_recon']
BASE_MODELS = [
    "cnn_gru_dual", "cnn_gru_dual_attn_recon",
    "cnn_gru_dual_attn_recon_dann", "cnn_gru_dual_attn_recon_supcon3",
    "cnn_gru_dual_attn_recon_aug", "cnn_gru_dual_attn_recon_mtl",
]
LOCO_MODELS = BASE_MODELS + [f"{m}_pc_recenter" for m in BASE_MODELS]

OUT_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "chip" / "RQ3_06_confidence_shift"
PC_RECENTER_CACHE_NAME = "pc_recenter_confidence_cache.joblib"


def short_name(chip_name):
    m = re.search(r'DDM_0(\d)', chip_name)
    return f'Chip 0{m.group(1)}' if m else chip_name.split('_U_', 1)[1]


def group_out_dir(exp_folder):
    return Path(exp_folder) / "cross_dataset_cv" / GROUP_NAME / "curve_alignment_pc_ttp" / "anchor_min"


def cv_group_dir(exp_folder):
    return Path(exp_folder) / "cross_dataset_cv" / GROUP_NAME


def cv_confidence(cv_results):
    correct_all, conf_all = [], []
    for fold_entry in cv_results.values():
        res_entry = fold_entry.get(FILTER_KEY)
        if res_entry is None:
            continue
        y_true = np.concatenate(res_entry["y_trues_"])
        for model in CV_MODELS:
            preds_key, probs_key, _ = config.MODEL_KEY_MAP.get(model, (None, None, None))
            if preds_key is None or preds_key not in res_entry or probs_key not in res_entry:
                continue
            y_pred = np.concatenate(res_entry[preds_key])
            y_prob = np.concatenate(res_entry[probs_key])
            correct_all.append(y_true == y_pred)
            conf_all.append(y_prob.max(axis=1))
    if not correct_all:
        return None, None
    return np.concatenate(correct_all), np.concatenate(conf_all)


def loco_base_confidence(lofo_results, model, chip_name):
    res_entry = lofo_results.get(f"lofo_{chip_name}", {}).get(FILTER_KEY)
    if res_entry is None:
        return None, None
    preds_key, probs_key, _ = config.MODEL_KEY_MAP.get(model, (None, None, None))
    if preds_key is None or preds_key not in res_entry or probs_key not in res_entry:
        return None, None
    y_true = np.concatenate(res_entry["y_trues_"])
    y_pred = np.concatenate(res_entry[preds_key])
    y_prob = np.concatenate(res_entry[probs_key])
    return y_true == y_pred, y_prob.max(axis=1)


def _cache_path():
    return OUT_DIR / PC_RECENTER_CACHE_NAME


def _load_cache():
    path = _cache_path()
    if not path.exists():
        return {}
    try:
        return joblib.load(path)
    except Exception:
        return {}


def _save_cache(cache):
    joblib.dump(cache, _cache_path(), compress=3)


def _mtime(path):
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def _pc_recenter_signature(model_path, out_dir, held_out_chip):
    align_dir = config.cross_dataset_alignment_dir(out_dir, held_out_chip)
    return (_mtime(model_path),
           _mtime(align_dir / config.CROSS_DATASET_RESAMPLER_PATH.format(curve_type=CURVE_TYPE)),
           _mtime(align_dir / config.CROSS_DATASET_PC_TTP_RECIPE_PATH.format(curve_type=CURVE_TYPE)))


def loco_pc_recenter_confidence(base_model, chip_path, exp_paths_all, class_names, cache, out_dir):
    chip_name = chip_path.name
    model_dir = out_dir / "model_interpretation" / f"lofo_{chip_name}"
    model_path = model_dir / f"{base_model}_{FILTER_KEY}_{CURVE_TYPE}{frac_suffix(TRAIN_CENTER_FRAC)}_model.keras"
    if not model_path.exists():
        return None, None

    cache_key = (base_model, chip_name)
    sig = _pc_recenter_signature(model_path, out_dir, chip_name)
    cached = cache.get(cache_key)
    if cached is not None and cached["sig"] == sig:
        return cached["correct"], cached["conf"]

    align_result = align_new_chip(chip_path, out_dir, CURVE_TYPE, group_name=GROUP_NAME, held_out_chip=chip_name)
    if align_result is None:
        return None, None
    curves, resampler, Y_well_raw, pc_curves_aligned, coords, well_ids = align_result

    loaded = load_saved_models(model_dir, FILTER_KEY, [base_model], CURVE_TYPE, train_center_frac=TRAIN_CENTER_FRAC)
    model = loaded.get(base_model)
    if model is None:
        return None, None
    if is_spatial_model(base_model) and (coords is None or well_ids is None):
        return None, None

    mapping = config.LABEL_MAPPINGS[chip_name]
    y_true_str = np.array([mapping.get(w, w) for w in Y_well_raw])
    valid = y_true_str != "PC"
    exp_paths_train = [p for p in exp_paths_all if p.name != chip_name]

    try:
        probs, _ = predict_new_chip(model, base_model, curves, coords, well_ids, pc_curves_aligned,
                                    exp_paths_train, out_dir, CURVE_TYPE, FILTER_KEY,
                                    pc_recenter=True, force_rerun=True, held_out_chip=chip_name)
    except ValueError:
        del model, loaded
        tf.keras.backend.clear_session()
        return None, None

    pred_str = np.array(class_names)[np.argmax(probs, axis=1)]
    correct = pred_str[valid] == y_true_str[valid]
    conf = probs[valid].max(axis=1)

    cache[cache_key] = {"sig": sig, "correct": correct, "conf": conf}
    _save_cache(cache)

    del model, loaded
    tf.keras.backend.clear_session()
    return correct, conf


def _plot_confidence_ax(ax, correct, conf, title):
    if correct is None or len(correct) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=7, color="gray",
               transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        if title:
            ax.set_title(title, fontsize=7.5)
        return

    bins = np.linspace(0, 1, 21)
    n_correct, n_wrong = int(correct.sum()), int((~correct).sum())
    if n_correct:
        c = conf[correct]
        ax.hist(c, bins=bins, alpha=0.6, color="#27ae60", density=True, label=f"Correct (n={n_correct})")
        ax.axvline(c.mean(), color="#1e8449", linestyle="--", linewidth=1.2)
    if n_wrong:
        w = conf[~correct]
        ax.hist(w, bins=bins, alpha=0.6, color="#e74c3c", density=True, label=f"Wrong (n={n_wrong})")
        ax.axvline(w.mean(), color="#922b21", linestyle="--", linewidth=1.2)
    ax.set_xlim(0, 1)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5, loc="upper left")
    if title:
        ax.set_title(title, fontsize=7.5)


def run(exp_folder):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chips = list(config.CROSS_DATASET_GROUPS[GROUP_NAME])
    exp_paths_all = [Path(exp_folder, c) for c in chips]

    print("Loading CV (in-distribution) results...")
    cv_results = load_partitioned(cv_group_dir(exp_folder), CV_MODE_STR, CURVE_TYPE, train_center_frac=TRAIN_CENTER_FRAC)
    cv_correct, cv_conf = cv_confidence(cv_results)

    out_dir = group_out_dir(exp_folder)
    lofo_results = load_partitioned(out_dir, LOFO_MODE_STR, CURVE_TYPE, train_center_frac=TRAIN_CENTER_FRAC)
    pc_cache = _load_cache()

    columns = ["CV (in-dist.)"] + LOCO_MODELS
    n_rows, n_cols = len(chips), len(columns)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.4 * n_rows), squeeze=False)

    for row, chip in enumerate(chips):
        t0 = time.perf_counter()
        print(f"\n{'-'*70}\n[row] held out = {chip}\n{'-'*70}")
        fold_entry = lofo_results.get(f"lofo_{chip}", {})
        class_names = fold_entry.get("class_names")

        _plot_confidence_ax(axes[row, 0], cv_correct, cv_conf, columns[0] if row == 0 else None)
        axes[row, 0].set_ylabel(short_name(chip), fontsize=9, fontweight="bold")

        for col, model in enumerate(LOCO_MODELS, start=1):
            if model in BASE_MODELS:
                correct, conf = loco_base_confidence(lofo_results, model, chip)
            elif class_names is None:
                correct, conf = None, None
            else:
                base_model = model[:-len("_pc_recenter")]
                correct, conf = loco_pc_recenter_confidence(
                    base_model, Path(exp_folder, chip), exp_paths_all, class_names, pc_cache, out_dir)
            _plot_confidence_ax(axes[row, col], correct, conf, model if row == 0 else None)

        print(f"  [+] row {chip} done in {time.perf_counter()-t0:.1f}s")

    fig.suptitle(f"Softmax Confidence -- In-Distribution (CV) vs. Distribution Shift (LOCO) | {GROUP_NAME}",
                fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout()
    save_path = OUT_DIR / "confidence_shift_grid.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n[+] saved -> {save_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Softmax confidence: in-distribution CV vs LOCO.")
    parser.add_argument("--exp_folder", type=str, default=config.DEFAULT_EXP_FOLDER)
    args = parser.parse_args(argv)

    print(f"\n{'='*70}\n[RUNNING] chip/confidence_shift.py\n{'='*70}\n")
    run(args.exp_folder)


if __name__ == "__main__":
    main()
