import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


from utils.preprocess import load_raw_data, normalise_curves, build_ttp_aligned_curves, model_name, manifest_name, load_manifest, result_name


RESULT_DIR = ROOT / 'result'
MODEL_DIR = ROOT / 'models'


def preprocess_test(dataset_file, model_dataset_name, one_to_one, ttp_aligned, normalised):
    # Load the test set and apply the exact same preprocessing as training, reusing the
    # saved TTP-alignment cutoff (it's batch-dependent, so it can't be recomputed on the test set).

    curves, timestamps, well_labels = load_raw_data(dataset_file)
    orig_row = np.arange(len(curves))

    manifest_path = MODEL_DIR / manifest_name(model_dataset_name, one_to_one, ttp_aligned, normalised)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No manifest at {manifest_path} — run main.py with matching --one_to_one/--ttp_aligned/"
            f"--normalise_curve flags for dataset '{model_dataset_name}' first."
        )
    manifest = load_manifest(manifest_path)

    if ttp_aligned:
        ttp = manifest['ttp_align']
        curves, timestamps, well_labels, keep, _, _ = build_ttp_aligned_curves(
            curves, timestamps, well_labels, min_ttp=ttp['min_ttp'], max_ttp=ttp['max_ttp'])
        orig_row = orig_row[keep]

    if normalised:
        curves = normalise_curves(curves)

    return curves, well_labels, orig_row, manifest, timestamps


def run_inference(dataset_file, dataset_name, model_dataset_name, one_to_one, ttp_aligned, normalised):
    # Raw, independent predictions from every saved model — no combining/voting across pairs.

    print(f"\n[{dataset_name}] inference (models: {model_dataset_name})")
    curves, well_labels, orig_row, manifest, _ = preprocess_test(
        dataset_file, model_dataset_name, one_to_one, ttp_aligned, normalised)
    X = curves[:, :, np.newaxis].astype(np.float32)

    out = {'orig_row': orig_row, 'true_label': well_labels}

    if one_to_one:
        for label1, label2 in manifest['pairs']:
            col = f'{label1}_vs_{label2}'
            mpath = MODEL_DIR / model_name(model_dataset_name, one_to_one, ttp_aligned, normalised, label1, label2)
            if not mpath.is_file():
                print(f"  [!] Missing model, skipping: {mpath}")
                continue

            model = tf.keras.models.load_model(mpath, compile=False)
            probs = model.predict(X, verbose=0)
            pred_idx = np.argmax(probs, axis=1)
            classes = np.array([label1, label2])
            out[f'{col}_pred'] = classes[pred_idx]
            out[f'{col}_prob'] = probs[np.arange(len(probs)), pred_idx]
            tf.keras.backend.clear_session()
            print(f"  {col}: predicted {len(pred_idx)} rows")

    else:
        mpath = MODEL_DIR / model_name(model_dataset_name, one_to_one, ttp_aligned, normalised)
        if not mpath.is_file():
            raise FileNotFoundError(f"Missing model: {mpath}")

        model = tf.keras.models.load_model(mpath, compile=False)
        probs = model.predict(X, verbose=0)
        pred_idx = np.argmax(probs, axis=1)
        classes = np.array(manifest['classes'])
        out['pred'] = classes[pred_idx]
        out['prob'] = probs[np.arange(len(probs)), pred_idx]
        tf.keras.backend.clear_session()
        print(f"  predicted {len(pred_idx)} rows")

    return pd.DataFrame(out)


def parse_args():
    ap = argparse.ArgumentParser(
        description='TTP modulation inference: raw predictions from saved models (no CV, no training).')
    ap.add_argument('--dataset', dest='datasets', metavar='FILE', action='append', required=True, help='Path to a test dataset CSV or Excel file. May be repeated.')
    ap.add_argument('--model_dataset', metavar='NAME', default=None, help='Stem of the training dataset whose saved models/manifest to use (default: derived from each --dataset filename).')
    ap.add_argument('--one_to_one', action='store_true', default=False, help='Use the per-pair binary models (same flag as main.py).')
    ap.add_argument('--normalise_curve', action='store_true', default=False, help='Min-max normalise each curve to [0, 1] (same flag as main.py).')
    ap.add_argument('--ttp_aligned', action='store_true', default=False, help='Align curves to TTP using the saved training-time cutoff (same flag as main.py).')
    return ap.parse_args()


def main():
    args = parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_path in args.datasets:
        dataset_file = Path(dataset_path)
        if not dataset_file.is_file():
            print(f"[!] File not found: {dataset_file}", file=sys.stderr)
            continue

        dataset_name = dataset_file.stem
        model_dataset_name = args.model_dataset or dataset_name

        try:
            df = run_inference(
                dataset_file, dataset_name, model_dataset_name,
                args.one_to_one, args.ttp_aligned, args.normalise_curve,
            )
        except FileNotFoundError as e:
            print(f"[!] {e}", file=sys.stderr)
            continue

        stem = result_name(dataset_name + '_inference', args.one_to_one, args.ttp_aligned, args.normalise_curve).replace('.html', '.csv')
        out_path = RESULT_DIR / stem
        df.to_csv(out_path, index=False)
        print(f"  Predictions saved → {out_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
