"""
Ablation 1: cnn_gru_dual saliency heatmaps.

Same dual-branch latent-gradient saliency technique as
adhoc/ttp_modulation/main_saliency.py (extract_dual_saliency +
plot_per_label_saliency_heatmap, imported directly from
adhoc/ttp_modulation/utils/saliency.py) -- reused unmodified since
create_cnn_gru_dual in main/utils/model_training/model_utils.py is
architecturally identical to adhoc/ttp_modulation/utils/model.py's version
(Flatten CNN branch + last Bidirectional GRU branch), so find_flatten_layer/
find_bidirectional_recurrent_layer locate the same landmarks.

Loads the cnn_gru_dual_None_ori_curve_model.keras already saved by
ablation1_model_comparison.py under each LAB dataset's
ablations/model_interpretation/, draws a random sample batch from that
dataset's ori_curve data (same preprocessing as ablation1_model_comparison.py:
filter_datasets + label mapping + rare-class exclusion, matching what the
baseline (filter=None) training run saw), and saves one heatmap PNG per class
to ablations/xai_saliency/.
"""
import os
import sys
import argparse
import importlib.util
import joblib
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

_MAIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_MAIN_DIR))
sys.path.insert(0, str(_MAIN_DIR / "utils"))
sys.path.insert(0, str(_MAIN_DIR / "utils" / "model_training"))

import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

_TTP_SALIENCY_PATH = _MAIN_DIR.parent / "adhoc" / "ttp_modulation" / "utils" / "saliency.py"
_spec = importlib.util.spec_from_file_location("ttp_saliency", str(_TTP_SALIENCY_PATH))
ttp_saliency = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ttp_saliency)
extract_dual_saliency = ttp_saliency.extract_dual_saliency
plot_per_label_saliency_heatmap = ttp_saliency.plot_per_label_saliency_heatmap

MODEL_KEY = 'cnn_gru_dual'
CURVE_TYPE = 'ori_curve'
LAB_DATASETS = ['01_ACA_qdPCR', '02_AMCA_qdLAMP', '03_AMCA_qdPCR']


def filter_datasets(dataset_name, dataset, kinetic_features):
    filtered_names, filtered_dataset, filtered_features = [], [], []
    for name, data, features in zip(dataset_name, dataset, kinetic_features):
        if not name.startswith("avg_") and not name.startswith("original_fitted_stretched"):
            filtered_names.append(name)
            filtered_dataset.append(data)
            filtered_features.append(features)
    return filtered_names, filtered_dataset, filtered_features


def load_dataset_curves(exp_path, curve_type=CURVE_TYPE):
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

    # Mirrors evaluate_outlier_filters' baseline (filter=None) rare-class exclusion --
    # the saved cnn_gru_dual model never saw classes with <2 samples.
    unique_classes, counts = np.unique(y_full, return_counts=True)
    rare = unique_classes[counts < 2]
    if len(rare):
        keep = ~np.isin(y_full, rare)
        curves_2d, y_full = curves_2d[keep], y_full[keep]

    return curves_2d, y_full, list(encoder.classes_)


def run_one(exp_path, n_dims, batch_n, seed):
    model_path = (exp_path / "ablations" / "model_interpretation"
                  / f"{MODEL_KEY}_None_{CURVE_TYPE}_model.keras")
    if not model_path.is_file():
        print(f"  [!] Missing model, skipping: {model_path}")
        return

    loaded = load_dataset_curves(exp_path)
    if loaded is None:
        return
    curves_2d, y_full, class_names = loaded

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(curves_2d), size=min(batch_n, len(curves_2d)), replace=False)
    X_batch = curves_2d[idx][:, :, np.newaxis].astype(np.float32)
    y_true = y_full[idx]

    model = tf.keras.models.load_model(model_path, compile=False)
    y_pred = np.argmax(model.predict(X_batch, batch_size=256, verbose=0), axis=1)
    art = extract_dual_saliency(model, X_batch, n_dims=n_dims)

    out_dir = exp_path / "ablations" / "xai_saliency"
    out_dir.mkdir(parents=True, exist_ok=True)
    for c_idx, c_name in enumerate(class_names):
        save_path = out_dir / f"{MODEL_KEY}_{CURVE_TYPE}_{c_name}.png"
        saved = plot_per_label_saliency_heatmap(
            art, X_batch, y_true, y_pred, c_idx, c_name,
            timestamps=np.arange(X_batch.shape[1]), n_dims=n_dims, save_path=save_path,
            show_std=False,
        )
        print(f"    {c_name}: {'saved -> ' + str(save_path) if saved else 'skipped'}")
    tf.keras.backend.clear_session()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="cnn_gru_dual saliency heatmaps for Ablation 1's 3 LAB datasets.")
    parser.add_argument("--n_dims", type=int, default=25, metavar="N",
                        help="Top N latent dims per branch to show (default: 25).")
    parser.add_argument("--batch_n", type=int, default=512, metavar="N",
                        help="Number of samples to draw for gradient computation (default: 512).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for the sample batch (default: 42).")
    args = parser.parse_args()

    print(f"\n{'='*70}\n[RUNNING] ablation1_cnn_gru_dual_saliency.py\n{'='*70}\n")

    for name in LAB_DATASETS:
        exp_path = Path(config.LAB_EXP_FOLDER) / name
        print(f"\n{'-'*70}\n[{name}]\n{'-'*70}")
        run_one(exp_path, args.n_dims, args.batch_n, args.seed)

    print(f"\n{'='*70}\n[DONE] ablation1_cnn_gru_dual_saliency.py\n{'='*70}\n")
