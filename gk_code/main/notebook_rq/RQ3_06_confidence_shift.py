import os
import re
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

_MAIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_MAIN_DIR))
sys.path.insert(0, str(_MAIN_DIR / "utils"))
sys.path.insert(0, str(_MAIN_DIR / "utils" / "model_training"))
os.chdir(_MAIN_DIR)

import config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

pred08 = importlib.import_module("08_cross_dataset_predict_new_chip")
vis07  = importlib.import_module("07_attribution_vis_all")
b06    = importlib.import_module("06b_cross_dataset_prediction_report")
rio    = importlib.import_module("cross_dataset_result_io")

# Same constants as RQ3_01_cv_results.ipynb / RQ3_02_loco_results.ipynb's final_6_new comparison.
EXP_FOLDER = config.FINAL_EXP_FOLDER
GROUP_NAME = "final_6_new"
CURVE_TYPE = "ori_curve_sg_p4_norm"
FILTER_KEY = "noamp_remove"
CURVE_ALIGNMENT = "pc_ttp"
PC_TTP_ANCHOR = "min"
TRAIN_CENTER_FRAC = 0.5
CV_MODE_STR = "kfold5"
LOFO_MODE_STR = "lofo"

CV_MODELS = ['knn', 'cnn_gru_dual', 'cnn_gru_dual_attn_recon']

# Copied from RQ3_02_loco_results.ipynb's currently-active RQ3_2_BASE_MODELS /
# PC_RECENTER_BASES (knn/coral/dann_conc commented out there too) -- update both
# here and in RQ3_05_embedding_analysis.py if that notebook's lists change.
BASE_MODELS = [
    "cnn_gru_dual", "cnn_gru_dual_attn_recon",
    "cnn_gru_dual_attn_recon_dann", "cnn_gru_dual_attn_recon_supcon3",
    "cnn_gru_dual_attn_recon_aug", "cnn_gru_dual_attn_recon_mtl",
]
PC_RECENTER_BASES = list(BASE_MODELS)
LOCO_MODELS = BASE_MODELS + [f"{m}_pc_recenter" for m in PC_RECENTER_BASES]

OUT_DIR = Path(__file__).resolve().parent / "RQ3_06_confidence_shift"
PC_RECENTER_CACHE_NAME = "pc_recenter_confidence_cache.joblib"


def short_name(chip_name):
    m = re.search(r'DDM_0(\d)', chip_name)
    return f'Chip 0{m.group(1)}' if m else chip_name.split('_U_', 1)[1]


def group_out_dir():
    base_dir = Path(EXP_FOLDER) / "cross_dataset_cv" / GROUP_NAME
    return b06.alignment_dir(base_dir, CURVE_ALIGNMENT, PC_TTP_ANCHOR)


def cv_group_dir():
    return Path(EXP_FOLDER) / "cross_dataset_cv" / GROUP_NAME


def load_cv_results():
    group_dir = cv_group_dir()
    legacy_path = b06.find_results_path(group_dir, CV_MODE_STR, CURVE_TYPE)
    return b06.load_partitioned(group_dir, CV_MODE_STR, CURVE_TYPE, legacy_path=legacy_path,
                                train_center_frac=TRAIN_CENTER_FRAC)


def load_lofo_results():
    out_dir = group_out_dir()
    legacy_path = b06.find_results_path(out_dir, LOFO_MODE_STR, CURVE_TYPE)
    return b06.load_partitioned(out_dir, LOFO_MODE_STR, CURVE_TYPE, legacy_path=legacy_path,
                                train_center_frac=TRAIN_CENTER_FRAC)


def cv_confidence(cv_results):
    """Pooled (correct, max softmax prob) across every CV_MODELS x fold -- CV has no
    held-out chip, so this single in-distribution reference is reused in every row."""
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
    """(correct, max softmax prob) for a BASE_MODELS model on one held-out chip, straight
    from 04_cross_dataset_training.py's own cached lofo results -- no model loading."""
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


def loco_pc_recenter_confidence(base_model, chip_path, exp_paths_all, class_names, cache):
    """(correct, max softmax prob) for base_model's PC-recentered predictions on one
    held-out chip -- mirrors RQ3_02_loco_results.ipynb's lofo_pc_recenter_accuracy, but
    keeps the full per-pixel probs (that notebook's own cache only keeps accuracy + cm)."""
    out_dir = group_out_dir()
    chip_name = chip_path.name
    model_dir = out_dir / "model_interpretation" / f"lofo_{chip_name}"
    model_path = model_dir / f"{base_model}_{FILTER_KEY}_{CURVE_TYPE}{rio.frac_suffix(TRAIN_CENTER_FRAC)}_model.keras"
    if not model_path.exists():
        return None, None

    cache_key = (base_model, chip_name)
    sig = _pc_recenter_signature(model_path, out_dir, chip_name)
    cached = cache.get(cache_key)
    if cached is not None and cached["sig"] == sig:
        return cached["correct"], cached["conf"]

    align_result = pred08.align_new_chip(chip_path, out_dir, CURVE_TYPE, CURVE_ALIGNMENT, PC_TTP_ANCHOR,
                                         group_name=GROUP_NAME, held_out_chip=chip_name)
    if align_result is None:
        return None, None
    curves, resampler, Y_well_raw, pc_curves_aligned, coords, well_ids = align_result

    loaded = vis07.load_saved_models(model_dir, FILTER_KEY, len(resampler.t_grid), curve_type=CURVE_TYPE,
                                     model_names=[base_model], train_center_frac=TRAIN_CENTER_FRAC)
    model = loaded.get(base_model)
    if model is None:
        return None, None
    if pred08._is_spatial(base_model) and (coords is None or well_ids is None):
        return None, None

    mapping = config.LABEL_MAPPINGS[chip_name]
    y_true_str = np.array([mapping.get(w, w) for w in Y_well_raw])
    valid = y_true_str != "PC"
    exp_paths_train = [p for p in exp_paths_all if p.name != chip_name]

    try:
        probs, _ = pred08.predict_new_chip(model, base_model, curves, coords, well_ids, pc_curves_aligned,
                                           exp_paths_train, out_dir, CURVE_TYPE, FILTER_KEY, CURVE_ALIGNMENT,
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


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chips = list(config.CROSS_DATASET_GROUPS[GROUP_NAME])
    exp_paths_all = [Path(EXP_FOLDER, c) for c in chips]

    print("Loading CV (in-distribution) results...")
    cv_correct, cv_conf = cv_confidence(load_cv_results())

    lofo_results = load_lofo_results()
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
                    base_model, Path(EXP_FOLDER, chip), exp_paths_all, class_names, pc_cache)
            _plot_confidence_ax(axes[row, col], correct, conf, model if row == 0 else None)

        print(f"  [+] row {chip} done in {time.perf_counter()-t0:.1f}s")

    fig.suptitle(f"Softmax Confidence -- In-Distribution (CV) vs. Distribution Shift (LOCO) | {GROUP_NAME}",
                fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout()
    save_path = OUT_DIR / "confidence_shift_grid.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n[+] saved -> {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Softmax confidence under distribution shift: in-distribution (RQ3.1 "
                    "kfold5 CV, 1 pooled column) vs LOCO (RQ3.2's 12-model comparison, "
                    "1 row per held-out chip). Saves one grid figure.")
    parser.parse_args()

    print(f"\n{'='*70}\n[RUNNING] RQ3_06_confidence_shift.py\n{'='*70}\n")
    run()
    print(f"\n{'='*70}\n[DONE] RQ3_06_confidence_shift.py\n{'='*70}\n")
