import argparse
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


from inference import preprocess_test, MODEL_DIR
from utils.preprocess import filter_to_pair, model_name, saliency_name
from utils.saliency import extract_dual_saliency, plot_per_label_saliency_heatmap


XAI_DIR = ROOT / 'xai'


def _predict(model, X, batch_size=256):
    return np.argmax(model.predict(X, batch_size=batch_size, verbose=0), axis=1)


def _run_one(model, X_batch, y_true, class_names, class_labels, n_dims, save_paths):
    y_pred = _predict(model, X_batch)
    art = extract_dual_saliency(model, X_batch, n_dims=n_dims)
    for c_idx, c_name, save_path in zip(class_labels, class_names, save_paths):
        saved = plot_per_label_saliency_heatmap(
            art, X_batch, y_true, y_pred, c_idx, c_name,
            timestamps=np.arange(X_batch.shape[1]),
            n_dims=n_dims, save_path=save_path,
        )
        print(f'    {c_name}: {"saved -> " + str(save_path) if saved else "skipped"}')
    tf.keras.backend.clear_session()


def run_saliency(dataset_file, dataset_name, one_to_one, ttp_aligned, normalised, n_dims, batch_n, seed):
    print(f"\n[{dataset_name}] saliency")
    curves, well_labels, _, manifest, _ = preprocess_test(
        dataset_file, dataset_name, one_to_one, ttp_aligned, normalised)
    rng = np.random.default_rng(seed)

    if one_to_one:
        for label1, label2 in manifest['pairs']:
            pair_curves, pair_labels = filter_to_pair(curves, well_labels, label1, label2)
            if len(pair_curves) == 0:
                print(f"  [!] {label1} vs {label2}: no matching rows, skipping")
                continue

            mpath = MODEL_DIR / model_name(dataset_name, one_to_one, ttp_aligned, normalised, label1, label2)
            if not mpath.is_file():
                print(f"  [!] Missing model, skipping: {mpath}")
                continue

            print(f"  {label1} vs {label2}")
            idx = rng.choice(len(pair_curves), size=min(batch_n, len(pair_curves)), replace=False)
            X_batch = pair_curves[idx][:, :, np.newaxis].astype(np.float32)
            y_true = (pair_labels[idx] == label2).astype(int)  # label1->0, label2->1 (matches training's sort order)

            model = tf.keras.models.load_model(mpath, compile=False)
            save_paths = [
                XAI_DIR / saliency_name(dataset_name, one_to_one, ttp_aligned, normalised, name, label1, label2)
                for name in (label1, label2)
            ]
            _run_one(model, X_batch, y_true, [label1, label2], [0, 1], n_dims, save_paths)

    else:
        mpath = MODEL_DIR / model_name(dataset_name, one_to_one, ttp_aligned, normalised)
        if not mpath.is_file():
            print(f"  [!] Missing model, skipping: {mpath}")
            return

        classes = manifest['classes']
        idx = rng.choice(len(curves), size=min(batch_n, len(curves)), replace=False)
        X_batch = curves[idx][:, :, np.newaxis].astype(np.float32)
        y_true = np.array([classes.index(l) for l in well_labels[idx]])

        model = tf.keras.models.load_model(mpath, compile=False)
        save_paths = [
            XAI_DIR / saliency_name(dataset_name, one_to_one, ttp_aligned, normalised, name)
            for name in classes
        ]
        _run_one(model, X_batch, y_true, classes, list(range(len(classes))), n_dims, save_paths)


def parse_args():
    ap = argparse.ArgumentParser(
        description='Per-label saliency heatmaps (View A, normalised) for saved CNN-GRU dual models.')
    ap.add_argument('--dataset', dest='datasets', metavar='FILE', action='append', required=True, help='Path to a dataset CSV or Excel file (same file used to train the models). May be repeated.')
    ap.add_argument('--one_to_one', action='store_true', default=False, help='Same flag as main.py.')
    ap.add_argument('--normalise_curve', action='store_true', default=False, help='Same flag as main.py.')
    ap.add_argument('--ttp_aligned', action='store_true', default=False, help='Same flag as main.py.')
    ap.add_argument('--n_dims', type=int, default=25, metavar='N', help='Top N latent dims per branch to show (default: 25).')
    ap.add_argument('--batch_n', type=int, default=512, metavar='N', help='Number of samples to draw for gradient computation (default: 512).')
    ap.add_argument('--seed', type=int, default=42, help='Random seed for the sample batch (default: 42).')
    return ap.parse_args()


def main():
    args = parse_args()
    XAI_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_path in args.datasets:
        dataset_file = Path(dataset_path)
        if not dataset_file.is_file():
            print(f"[!] File not found: {dataset_file}", file=sys.stderr)
            continue

        dataset_name = dataset_file.stem
        try:
            run_saliency(
                dataset_file, dataset_name,
                args.one_to_one, args.ttp_aligned, args.normalise_curve,
                args.n_dims, args.batch_n, args.seed,
            )
        except FileNotFoundError as e:
            print(f"[!] {e}", file=sys.stderr)
            continue

    print("\nDone. Outputs in:", XAI_DIR)


if __name__ == '__main__':
    main()
