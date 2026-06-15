import os
import re
import argparse
import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score

import config

MODEL_KEY_MAP = {
    "cnn": ("y_preds_AC_", "y_probs_AC_", "classes_AC_"),
    "lstm": ("y_preds_AC_lstm_", "y_probs_AC_lstm_", "classes_AC_lstm_"),
    "gru": ("y_preds_AC_gru_", "y_probs_AC_gru_", "classes_AC_gru_"),
    "rnn": ("y_preds_AC_rnn_", "y_probs_AC_rnn_", "classes_AC_rnn_"),
    "transformer": ("y_preds_AC_trans_", "y_probs_AC_trans_", "classes_AC_trans_"),
    "rf": ("y_preds_AC_rf_", "y_probs_AC_rf_", "classes_AC_rf_"),
    "knn": ("y_preds_AC_kNN_", "y_probs_AC_kNN_", "classes_AC_kNN_"),
    "ffi": ("y_preds_FFI_", "y_probs_FFI_", "classes_FFI_"),
    "cnn_lf": ("y_preds_AC_cnn_lf_", "y_probs_AC_cnn_lf_", "classes_AC_cnn_lf_"),
    "lstm_lf": ("y_preds_AC_lstm_lf_", "y_probs_AC_lstm_lf_", "classes_AC_lstm_lf_"),
    "trans_lf": ("y_preds_AC_trans_lf_", "y_probs_AC_trans_lf_", "classes_AC_trans_lf_"),
    "gru_lf": ("y_preds_AC_gru_lf_", "y_probs_AC_gru_lf_", "classes_AC_gru_lf_"),
    "cnn_gru_dual": ("y_preds_AC_cnn_gru_dual_", "y_probs_AC_cnn_gru_dual_", "classes_AC_cnn_gru_dual_"),
    "cnn_trans_dual": ("y_preds_AC_cnn_trans_dual_", "y_probs_AC_cnn_trans_dual_", "classes_AC_cnn_trans_dual_"),
}

MODEL_PRINT_MAP = {
    "cnn": "CNN (ACA)", "lstm": "LSTM (ACA)", "gru": "GRU (ACA)",
    "rnn": "RNN (ACA)", "transformer": "Trans (ACA)", "rf": "RF (ACA)",
    "knn": "KNN (ACA)", "ffi": "LR (FFI)",
    "cnn_lf": "CNN LF", "lstm_lf": "LSTM LF", "trans_lf": "Trans LF", "gru_lf": "GRU LF",
    "cnn_gru_dual": "CNN+GRU Dual", "cnn_trans_dual": "CNN+Tr Dual",
}


def get_exp_paths(exp_folder):
    return sorted([
        Path(exp_folder, name) for name in os.listdir(exp_folder)
        if os.path.isdir(os.path.join(exp_folder, name)) and name not in config.EXCLUDED_FOLDERS
    ])


def compute_filtered_splits(y_full, features_df, outlier_filter, n_splits):
    """Recreate the deterministic mask + rare-class filter + split used by evaluate_outlier_filters
    so cached predictions/y_trues_ can be mapped back to original curve indices."""
    if outlier_filter is None:
        mask = np.ones(len(y_full), dtype=bool)
    elif outlier_filter in features_df.columns:
        mask = (features_df[outlier_filter] == 1).fillna(False).values
    else:
        return None

    global_idx = np.where(mask)[0]
    y_masked = y_full[mask]

    unique_classes, class_counts = np.unique(y_masked, return_counts=True)
    rare_classes = unique_classes[class_counts < 2]
    if len(rare_classes) > 0:
        valid = ~np.isin(y_masked, rare_classes)
        global_idx = global_idx[valid]
        y_masked = y_masked[valid]

    n_classes = len(np.unique(y_masked))
    if n_classes < 2 or len(y_masked) < 2 * n_classes:
        return None

    calculated_test_size = max(int(len(y_masked) * 0.10), n_classes)
    if n_splits == 1:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=calculated_test_size, random_state=0)
    else:
        min_class_count = np.min(class_counts[~np.isin(unique_classes, rare_classes)])
        actual_splits = min(n_splits, min_class_count)
        splitter = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=0)

    splits = list(splitter.split(np.zeros((len(y_masked), 1)), y_masked))
    return global_idx, y_masked, splits


def render_model_plot(curves, Y_well_raw, label_map, correct_idx, wrong_idx, header_lines):
    """n_wells x 2 grid (correctly predicted | wrongly predicted) for a single model."""
    unique_wells = np.unique(Y_well_raw)
    n_wells = len(unique_wells)

    fig, axes = plt.subplots(n_wells, 2, figsize=(8, 2.5 * n_wells), squeeze=False, sharey="row")

    correct_wells = Y_well_raw[correct_idx]
    wrong_wells = Y_well_raw[wrong_idx]

    for i, well in enumerate(unique_wells):
        well_title = f"Well {well}"
        try:
            well_key = int(well)
        except (TypeError, ValueError):
            well_key = well
        if well_key in label_map:
            well_title += f" ({label_map[well_key]})"

        well_correct = correct_idx[correct_wells == well]
        well_wrong = wrong_idx[wrong_wells == well]

        ax_ok, ax_bad = axes[i, 0], axes[i, 1]

        if len(well_correct) > 0:
            ax_ok.plot(curves[well_correct].T, color="#27ae60", alpha=0.2, linewidth=0.8)
        ax_ok.set_title(f"{well_title} - Correct (n={len(well_correct)})", fontsize=9, fontweight="bold")

        if len(well_wrong) > 0:
            ax_bad.plot(curves[well_wrong].T, color="#e74c3c", alpha=0.4, linewidth=0.8)
        ax_bad.set_title(f"{well_title} - Wrong (n={len(well_wrong)})", fontsize=9, fontweight="bold")

        for ax in (ax_ok, ax_bad):
            ax.set_xlabel("Time index", fontsize=8)
            ax.set_ylabel("Signal", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)

    fig.suptitle("\n".join(header_lines), fontweight="bold", fontsize=11, y=1.0)
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def build_html(title, panel_titles, img_buffers, save_path):
    html = f"""<html>
<head><title>{title}</title></head>
<body style="font-family: Arial, sans-serif; background-color: #f4f4f9; margin: 0; padding: 20px;">
    <h1 style="color: #333;">{title}</h1>
    <div style="display: flex; flex-wrap: nowrap; overflow-x: auto; gap: 20px; padding-bottom: 20px; align-items: flex-start;">
"""
    for panel_title, buf in zip(panel_titles, img_buffers):
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        html += f"""
        <div style="flex: 0 0 auto; background: white; padding: 10px; box-shadow: 0px 4px 10px rgba(0,0,0,0.1); border-radius: 8px;">
            <h3 style="text-align:center; margin: 0 0 10px 0;">{panel_title}</h3>
            <img src="data:image/png;base64,{img_b64}" style="height: auto; max-width: 600px;">
        </div>
        """
    html += "    </div>\n</body></html>"

    with open(save_path, "w") as f:
        f.write(html)


def process_experiment(exp_path, mode, outlier_filter, n_splits, force_rerun, curve_type="ori_curve"):
    out_dir = config.get_viz_dir(exp_path.parent, "model_performance_viz")

    mode_key = mode.strip().title()
    filter_tag = "none" if outlier_filter is None else re.sub(r"[^A-Za-z0-9._-]+", "_", outlier_filter)
    out_path = out_dir / f"{exp_path.name}__{curve_type}__{mode_key}__{filter_tag}__nsplits{n_splits}.html"

    if out_path.exists() and not force_rerun:
        print(f"  -> [SKIP] Report already exists: {out_path}")
        return

    results_path = os.path.join(exp_path, config.TRAINING_10FOLD_RESULT_PATH if n_splits > 1 else config.TRAINING_RESULT_PATH)
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
        state = joblib.load(state_path)
    except Exception as e:
        print(f"  -> Skipping {exp_path.name}: failed to load results/state ({e}).")
        return

    dataset_name = list(state["dataset_name"])
    dataset = state["dataset"]
    Y_well_raw = np.asarray(state["Y_well"])
    kinetic_features = state["kinetic_features"]

    try:
        curve_idx, resolved_name = config.resolve_curve_dataset_idx(curve_type, dataset_name)
    except ValueError as e:
        print(f"  -> [SKIP] {e}")
        return
    curves = dataset[curve_idx]
    features_df = kinetic_features[curve_idx]

    Y_well_mapped = list(Y_well_raw)
    if exp_path.name in config.LABEL_MAPPINGS:
        mapping = config.LABEL_MAPPINGS[exp_path.name]
        Y_well_mapped = [mapping.get(w, w) for w in Y_well_raw]

    encoder = LabelEncoder()
    y_full = encoder.fit_transform(Y_well_mapped)

    clean_title = resolved_name.replace("_", " ").title()

    if clean_title not in all_ml_results or mode_key not in all_ml_results[clean_title]:
        print(f"  -> No '{mode_key}' results found for {exp_path.name}. Skipping.")
        return

    results_dict = all_ml_results[clean_title][mode_key]
    if outlier_filter not in results_dict:
        print(f"  -> Filter '{outlier_filter}' not found in '{mode_key}' results for {exp_path.name}. Skipping.")
        return

    res_entry = results_dict[outlier_filter]

    split_info = compute_filtered_splits(y_full, features_df, outlier_filter, n_splits)
    if split_info is None:
        print(f"  -> Could not reconstruct train/test splits for {exp_path.name} (filter={outlier_filter}). Skipping.")
        return
    global_idx, y_masked, splits = split_info

    label_map = config.LABEL_MAPPINGS.get(exp_path.name, {})

    img_buffers, panel_titles = [], []

    for m, (preds_key, _, _) in MODEL_KEY_MAP.items():
        if preds_key not in res_entry:
            continue

        y_trues_list = res_entry["y_trues_"]
        preds_list = res_entry[preds_key]

        correct_idx_list, wrong_idx_list, fold_accs = [], [], []
        mismatch = False

        for fold_i, (_, test_idx_local) in enumerate(splits):
            if fold_i >= len(preds_list) or fold_i >= len(y_trues_list):
                break

            y_true_fold = y_masked[test_idx_local]
            y_pred_fold = np.asarray(preds_list[fold_i])
            y_true_cached = np.asarray(y_trues_list[fold_i])

            if y_true_fold.shape != y_true_cached.shape or not np.array_equal(y_true_fold, y_true_cached):
                mismatch = True
                break

            fold_accs.append(accuracy_score(y_true_fold, y_pred_fold))

            global_test_idx = global_idx[test_idx_local]
            correct_mask = (y_true_fold == y_pred_fold)
            correct_idx_list.append(global_test_idx[correct_mask])
            wrong_idx_list.append(global_test_idx[~correct_mask])

        if mismatch or not fold_accs:
            print(f"  -> [WARNING] {m}: reconstructed splits do not match cached results "
                  f"(try a different --n_splits). Skipping this model.")
            continue

        correct_idx = np.concatenate(correct_idx_list)
        wrong_idx = np.concatenate(wrong_idx_list)

        acc = np.mean(fold_accs) * 100
        std = np.std(fold_accs) * 100

        header_lines = [
            MODEL_PRINT_MAP.get(m, m),
            f"Acc: {acc:.2f}% ± {std:.2f}%",
            f"Correct: {len(correct_idx)}  |  Wrong: {len(wrong_idx)}",
        ]

        print(f"  -> Rendering: {MODEL_PRINT_MAP.get(m, m)} ({acc:.2f}% +/- {std:.2f}%)")
        img_buffers.append(render_model_plot(curves, Y_well_raw, label_map, correct_idx, wrong_idx, header_lines))
        panel_titles.append("<br>".join(header_lines))

    if not img_buffers:
        print(f"  -> No model results found for {exp_path.name} (mode={mode_key}, filter={outlier_filter}). Skipping.")
        return

    os.makedirs(out_dir, exist_ok=True)
    filter_label = outlier_filter if outlier_filter else "None (Baseline)"
    title = f"Model Prediction Visualization: {exp_path.name} | Curve: {curve_type} | Mode: {mode_key} | Filter: {filter_label}"
    build_html(title, panel_titles, img_buffers, out_path)
    print(f"  -> [SAVED] {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Static Model Prediction Visualization Report")
    parser.add_argument("--exp_folder", type=str, default=config.DEFAULT_EXP_FOLDER)
    parser.add_argument("--force_rerun", action="store_true", help="Regenerate the HTML reports even if they already exist")
    parser.add_argument("--mode", type=str, default="Reference", help="'Native' or 'Reference'")
    parser.add_argument("--outlier_filter", type=str, default=None, help="Outlier filter column to visualize (default: None / baseline)")
    parser.add_argument("--n_splits", type=int, default=1, help="Must match the --n_splits used for the corresponding 03 training run")
    parser.add_argument("--curve_type", type=str, nargs="+", default=["ori_curve", "ori_curve_avg"], help="Which curve dataset(s) to report on (e.g. 'ori_curve', 'ori_curve_avg', or a raw dataset_name entry)")
    args = parser.parse_args()

    for exp_path in get_exp_paths(args.exp_folder):
        for curve_type in args.curve_type:
            process_experiment(exp_path, args.mode, args.outlier_filter, args.n_splits, args.force_rerun, curve_type=curve_type)
