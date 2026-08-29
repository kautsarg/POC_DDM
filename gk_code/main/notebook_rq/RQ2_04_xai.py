import os
import sys
import re
import argparse
import joblib
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

_MAIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_MAIN_DIR))
sys.path.insert(0, str(_MAIN_DIR / "utils"))
sys.path.insert(0, str(_MAIN_DIR / "utils" / "model_training"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from cross_dataset_result_io import frac_suffix

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from saliency import (extract_dual_saliency, plot_per_label_saliency_heatmap,
                      compute_kinetic_feature_cache, plot_latent_feature_mapping)

GROUP_NAME = "final_6_new"
EXP_FOLDER = config.FINAL_EXP_FOLDER
DATASETS = config.CROSS_DATASET_GROUPS[GROUP_NAME]
# MODEL_KEYS = ['cnn_gru_dual', 'cnn_gru_dual_attn_recon']
# CURVE_TYPES = ['ori_curve_norm', 'ori_curve_sg_p4_norm']
MODEL_KEYS = ['cnn_gru_dual_attn_recon']
CURVE_TYPES = ['ori_curve_sg_p4_norm']
EXTRACT_N_DIMS = 100


def short_name(folder_name):
    m = re.search(r'DDM_0(\d)', folder_name)
    return f'Chip 0{m.group(1)}' if m else folder_name.split('_U_', 1)[1]


def filter_datasets(dataset_name, dataset, kinetic_features):
    filtered_names, filtered_dataset, filtered_features = [], [], []
    for name, data, features in zip(dataset_name, dataset, kinetic_features):
        if not name.startswith("avg_") and not name.startswith("original_fitted_stretched"):
            filtered_names.append(name)
            filtered_dataset.append(data)
            filtered_features.append(features)
    return filtered_names, filtered_dataset, filtered_features


def load_dataset_curves(exp_path, curve_type):
    data_path = os.path.join(exp_path, config.TRAINING_DATA_PATH)
    if not os.path.exists(data_path):
        print(f"  [SKIP] {exp_path.name}: '{data_path}' not found.")
        return None
    training_data = joblib.load(data_path)

    dataset_name = training_data["dataset_name"]
    dataset = training_data["dataset"]
    kinetic_features = training_data["kinetic_features"]
    Y_well = training_data["Y_well"]
    dataset_name, dataset, kinetic_features = filter_datasets(dataset_name, dataset, kinetic_features)

    label_mappings = config.get_label_mappings(exp_path)
    if exp_path.name in label_mappings:
        mapping = label_mappings[exp_path.name]
        Y_well = [mapping.get(w, w) for w in Y_well]

    encoder = LabelEncoder()
    y_full = encoder.fit_transform(Y_well)

    target_name = config.CURVE_TYPE_ALIASES.get(curve_type, curve_type)
    if target_name not in dataset_name:
        print(f"  [SKIP] curve_type '{curve_type}' not found in this dataset's curve variants.")
        return None
    idx = dataset_name.index(target_name)
    curves_2d = np.nan_to_num(dataset[idx], nan=0.0, posinf=0.0, neginf=0.0)

    unique_classes, counts = np.unique(y_full, return_counts=True)
    rare = unique_classes[counts < 2]
    if len(rare):
        keep = ~np.isin(y_full, rare)
        curves_2d, y_full = curves_2d[keep], y_full[keep]

    return curves_2d, y_full, list(encoder.classes_)


def predict_labels(model, X_batch, batch_size=256):
    is_ns = (not isinstance(model.input, (list, tuple))
             and hasattr(model.input, 'name')
             and 'neighbor_stack' in model.input.name)
    if is_ns:
        k_plus_1 = model.input.shape[1]
        x = np.repeat(X_batch[:, np.newaxis, :, 0], k_plus_1, axis=1)
    else:
        x = X_batch
    out = model.predict(x, batch_size=batch_size, verbose=0)
    out = out[0] if isinstance(out, (list, tuple)) else out
    return np.argmax(out, axis=1)


def run_one(exp_path, model_key, curve_type, n_dims, top_n, batch_n, seed):
    model_path = exp_path / "ablations/model_interpretation" / f"{model_key}_None_{curve_type}_model.keras"

    if not model_path.is_file():
        print(f"  [!] Missing model, skipping: {model_path}")
        return

    loaded = load_dataset_curves(exp_path, curve_type)
    if loaded is None:
        return
    curves_2d, y_full, class_names = loaded

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(curves_2d), size=min(batch_n, len(curves_2d)), replace=False)
    X_batch = curves_2d[idx][:, :, np.newaxis].astype(np.float32)
    y_true = y_full[idx]
    timestamps = np.arange(X_batch.shape[1])

    model = tf.keras.models.load_model(model_path, compile=False)
    y_pred = predict_labels(model, X_batch)
    art = extract_dual_saliency(model, X_batch, n_dims=EXTRACT_N_DIMS)

    out_dir = exp_path / "xai_saliency"
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_matrix, feat_sensitivity, feat_names = compute_kinetic_feature_cache(X_batch, timestamps)

    chip_label = short_name(exp_path.name)
    for c_idx, c_name in enumerate(class_names):
        save_path = out_dir / f"{model_key}_{curve_type}_{c_name}.png"
        saved = plot_per_label_saliency_heatmap(
            art, X_batch, y_true, y_pred, c_idx, c_name,
            timestamps=timestamps, n_dims=n_dims, save_path=save_path,
            show_std=False,
        )
        print(f"    {c_name}: {'saved -> ' + str(save_path) if saved else 'skipped'}")

        mask = (y_true == c_idx) & (y_pred == c_idx)
        mean_curve = X_batch[mask, :, 0].mean(0) if mask.any() else X_batch[:, :, 0].mean(0)
        mapping_path = out_dir / f"{model_key}_{curve_type}_{c_name}_latent_feature_mapping.png"
        plot_latent_feature_mapping(
            art, model_key, timestamps,
            feat_matrix, feat_sensitivity, feat_names,
            mean_curve, f"{chip_label} | {c_name}", mapping_path,
            TOP_N=top_n, sample_mask=mask,
        )

    tf.keras.backend.clear_session()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"cnn_gru_dual_attn_recon saliency + latent-feature-mapping plots across "
                    f"{GROUP_NAME}'s {len(DATASETS)} chips (RQ2 -- per-chip, non-LOFO models).")
    parser.add_argument("--n_dims", type=int, default=25, metavar="N",
                        help="Top N latent dims per branch to show in the saliency heatmap (default: 25).")
    parser.add_argument("--top_n", type=int, default=5, metavar="N",
                        help="Top N unique-feature latent dims per branch to show in the "
                             "latent-feature-mapping plot (default: 5).")
    parser.add_argument("--batch_n", type=int, default=512, metavar="N",
                        help="Number of samples to draw for gradient computation (default: 512).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for the sample batch (default: 42).")
    args = parser.parse_args()

    print(f"\n{'='*70}\n[RUNNING] RQ2_04_xai.py\n{'='*70}\n")

    for name in DATASETS:
        exp_path = Path(EXP_FOLDER) / name
        for model_key in MODEL_KEYS:
            for curve_type in CURVE_TYPES:
                print(f"\n{'-'*70}\n[{short_name(name)} | {model_key} | {curve_type}]\n{'-'*70}")
                run_one(exp_path, model_key, curve_type, args.n_dims, args.top_n, args.batch_n, args.seed)

    print(f"\n{'='*70}\n[DONE] RQ2_04_xai.py\n{'='*70}\n")
