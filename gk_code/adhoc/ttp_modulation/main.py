import argparse
import sys
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


from utils.preprocess import load_raw_data, normalise_curves, build_ttp_aligned_curves, get_pairs, filter_to_pair, preprocessed_name, result_name, save_preprocessed
from utils.train import run_cv
from utils.report import generate_html, save_result


PREPROC_DIR = ROOT / 'preprocessed_dataset'
RESULT_DIR = ROOT / 'result'


def preprocess(dataset_file, dataset_name, one_to_one, ttp_aligned, normalised):
    # Load and preprocess data (TTP alignment, normalisation, and one-to-one splitting)

    curves, timestamps, well_labels = load_raw_data(dataset_file)

    if ttp_aligned:
        print("  Aligning to TTP …")
        curves, timestamps, well_labels, _ = build_ttp_aligned_curves(
            curves, timestamps, well_labels)

    if normalised:
        curves = normalise_curves(curves)

    results = []
    if one_to_one:
        pairs = get_pairs(well_labels)
        for label1, label2 in pairs:
            pair_curves, pair_labels = filter_to_pair(curves, well_labels, label1, label2)
            fname = preprocessed_name(dataset_name, one_to_one, ttp_aligned, normalised, label1, label2)
            cache = PREPROC_DIR / fname
            save_preprocessed(pair_curves, pair_labels, cache)
            results.append((pair_curves, pair_labels, f'{label1} vs {label2}', cache))
    else:
        fname = preprocessed_name(dataset_name, one_to_one, ttp_aligned, normalised)
        cache = PREPROC_DIR / fname
        save_preprocessed(curves, well_labels, cache)
        results.append((curves, well_labels, None, cache))

    return results


def run_experiment(dataset_file, dataset_name, one_to_one, ttp_aligned, normalised, n_folds):
    # Run CNN + GRU dual model training and evaluation on preprocesed data

    label_parts = []
    if one_to_one: label_parts.append('one-to-one')
    if ttp_aligned:label_parts.append('TTP-aligned')
    if normalised: label_parts.append('normalised')
    combo_label = ', '.join(label_parts) if label_parts else 'baseline (full · unaligned · raw)'

    print(f"\n[{dataset_name}] {combo_label}")
    print("  Preprocessing …")
    items = preprocess(dataset_file, dataset_name, one_to_one, ttp_aligned, normalised)

    if one_to_one:
        pair_results_agg = []
        all_true_pool, all_pred_pool = [], []
        fold_accs_all = []

        for curves, well_labels, pair_label, _ in items:
            print(f"  Training pair: {pair_label}")
            res = run_cv(curves, well_labels, n_folds=n_folds)
            pair_results_agg.append({
                'label': pair_label,
                'mean_acc': res['mean_acc'],
                'std_acc': res['std_acc'],
                'cm': res['cm'],
                'classes': res['classes'],
            })
            all_true_pool.extend(res['y_true'].tolist())
            all_pred_pool.extend(res['y_pred'].tolist())
            fold_accs_all.extend(res['fold_accs'])

        all_true = np.array(all_true_pool)
        all_pred = np.array(all_pred_pool)
        agg_result = {
            'fold_accs': fold_accs_all,
            'mean_acc': float(np.mean([r['mean_acc'] for r in pair_results_agg])),
            'std_acc': float(np.std([r['mean_acc'] for r in pair_results_agg])),
            'precision': precision_score(all_true, all_pred, average='macro', zero_division=0),
            'recall': recall_score(all_true, all_pred, average='macro', zero_division=0),
            'f1': f1_score(all_true, all_pred, average='macro', zero_division=0),
        }
        return agg_result, pair_results_agg

    else:
        curves, well_labels, _, _ = items[0]
        print("  Training full multi-class …")
        res = run_cv(curves, well_labels, n_folds=n_folds)
        return res, None

def parse_args():
    ap = argparse.ArgumentParser(
        description='TTP alignment experiment runner (CNN-GRU dual, stratified K-fold CV)')
    ap.add_argument('--dataset', dest='datasets', metavar='FILE', action='append', required=True, help='Path to a dataset CSV or Excel file. May be repeated.')
    ap.add_argument('--one_to_one', action='store_true', default=False, help='Binary classifiers for every label pair.')
    ap.add_argument('--normalise_curve', action='store_true', default=False, help='Min-max normalise each curve to [0, 1].')
    ap.add_argument('--ttp_aligned', action='store_true', default=False, help='Align curves to their TTP before training.')
    ap.add_argument('--n_cv_folds', type=int, default=5, metavar='N', help='Number of stratified CV folds (default: 5).')
    return ap.parse_args()


def main():
    args = parse_args()

    PREPROC_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    one_to_one = args.one_to_one
    ttp_aligned = args.ttp_aligned
    normalised = args.normalise_curve

    label_parts = []
    if one_to_one: label_parts.append('one-to-one')
    if ttp_aligned: label_parts.append('TTP-aligned')
    if normalised: label_parts.append('normalised')
    combo_label = ', '.join(label_parts) or 'baseline'

    for dataset_path in args.datasets:
        dataset_file = Path(dataset_path)
        if not dataset_file.is_file():
            print(f"[!] File not found: {dataset_file}", file=sys.stderr)
            continue

        dataset_name = dataset_file.stem

        res, pair_res = run_experiment(
            dataset_file, dataset_name,
            one_to_one, ttp_aligned, normalised,
            args.n_cv_folds,
        )

        combo_results = [{
            'label': combo_label,
            'flags': {
                'one_to_one': one_to_one,
                'ttp_aligned': ttp_aligned,
                'normalised': normalised
            },
            'result': res,
            'pair_results': pair_res,
        }]

        stem = result_name(dataset_name, one_to_one, ttp_aligned, normalised).replace('.html', '')
        save_result(res, pair_res, RESULT_DIR / f'{stem}.json')
        generate_html(dataset_name, combo_results, RESULT_DIR / f'{stem}.html')

    print("\nDone.")


if __name__ == '__main__':
    main()
