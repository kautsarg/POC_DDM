import re
import json
import itertools
import numpy as np
import pandas as pd
from pathlib import Path


def _numeric_key(col):
    nums = re.findall(r'\d+\.?\d*', str(col))
    return float(nums[0]) if nums else 0.0


def load_raw_data(dataset_file, target_col='Target'):
    path = Path(dataset_file)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    print(f"  Loading: {path.name}")

    df = pd.read_excel(path) if path.suffix.lower() in ('.xlsx', '.xls') else pd.read_csv(path)

    cycle_cols = [c for c in df.columns if str(c).startswith('Cycle') or re.match(r'^\d+(\.\d+)?$', str(c))]
    cycle_cols = sorted(cycle_cols, key=_numeric_key)

    timestamps = np.array([_numeric_key(c) for c in cycle_cols])
    curves = df[cycle_cols].to_numpy(dtype=np.float64)
    well_labels = df[target_col].to_numpy()

    return curves, timestamps, well_labels


def normalise_curves(curves):
    curves = np.asarray(curves, dtype=np.float64)
    lo = curves.min(axis=1, keepdims=True)
    hi = curves.max(axis=1, keepdims=True)
    denom = np.where(hi - lo == 0, 1.0, hi - lo)

    return (curves - lo) / denom


def _compute_ttp(curves, threshold_frac=0.1):
    max_v = curves.max(axis=1, keepdims=True)
    base = curves[:, [0]]
    thr  = base + (max_v - base) * threshold_frac

    return np.argmax(curves >= thr, axis=1)


def _tradeoff_curve(ttp_idx, well_labels, n_cycles):
    unique_ttps = sorted(set(ttp_idx.tolist()))
    targets = sorted(set(well_labels))
    min_ttp = min(unique_ttps)
    totals = {t: int((well_labels == t).sum()) for t in targets}
    cycles_left, min_ret = [], []
    for max_ttp in unique_ttps:
        keep = ttp_idx <= max_ttp
        cycles_left.append(n_cycles - (max_ttp - min_ttp))
        min_ret.append(min(100 * int(((well_labels == t) & keep).sum()) / totals[t] for t in targets))

    return np.array(unique_ttps), np.array(cycles_left), np.array(min_ret)


def _knee_max_ttp(unique_ttps, cycles_left, min_ret):
    x = (cycles_left - cycles_left.min()) / (cycles_left.max() - cycles_left.min() + 1e-9)
    y = (min_ret - min_ret.min()) / (min_ret.max() - min_ret.min() + 1e-9)
    p1, p2 = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    d = p2 - p1
    d /= np.linalg.norm(d) + 1e-9
    pts = np.stack([x, y], axis=1)
    proj = p1 + np.outer((pts - p1) @ d, d)

    return int(unique_ttps[np.argmax(np.linalg.norm(pts - proj, axis=1))])


def build_ttp_aligned_curves(curves, timestamps, well_labels, threshold_frac=0.1, min_ttp=None, max_ttp=None):
    # min_ttp/max_ttp: reuse a previously computed cutoff (e.g. at inference time) instead of
    # deriving one from this batch, since the knee cutoff is batch-dependent and a saved model
    # expects the exact curve length it was trained on.
    ttp_idx = _compute_ttp(curves, threshold_frac)
    n_cycles = curves.shape[1]

    if min_ttp is None or max_ttp is None:
        min_ttp = int(ttp_idx.min())
        unique_ttps, cycles_left, min_ret = _tradeoff_curve(ttp_idx, well_labels, n_cycles)
        max_ttp = _knee_max_ttp(unique_ttps, cycles_left, min_ret)

    keep = (ttp_idx >= min_ttp) & (ttp_idx <= max_ttp)
    shifts = ttp_idx[keep] - min_ttp
    final_len = n_cycles - (max_ttp - min_ttp)

    aligned = np.empty((keep.sum(), final_len))
    for i, (curve, s) in enumerate(zip(curves[keep], shifts)):
        aligned[i] = curve[s:s + final_len]

    aligned -= aligned[:, [0]]  # rebaselining so all curves start at 0

    return aligned, timestamps[:final_len], well_labels[keep], keep, min_ttp, max_ttp


def get_pairs(well_labels):
    return list(itertools.combinations(sorted(np.unique(well_labels)), 2))


def filter_to_pair(curves, well_labels, label1, label2):
    mask = np.isin(well_labels, [label1, label2])

    return curves[mask], well_labels[mask]


def _safe(s):
    return str(s).replace(' ', '_').replace('/', '-')


def preprocessed_name(dataset, one_to_one, ttp_aligned, normalised, label1=None, label2=None):
    alignment = 'aligned' if ttp_aligned else 'unaligned'
    norm = 'normalised' if normalised  else 'unnormalised'
    if one_to_one and label1 is not None:
        return f"{dataset}_onetone_{_safe(label1)}_vs_{_safe(label2)}_{alignment}_{norm}.csv"
    mode = 'onetone' if one_to_one else 'full'

    return f"{dataset}_{mode}_{alignment}_{norm}.csv"


def result_name(dataset, one_to_one, ttp_aligned, normalised):
    alignment = 'aligned' if ttp_aligned else 'unaligned'
    norm = 'normalised' if normalised else 'unnormalised'
    mode = 'onetone' if one_to_one else 'full'

    return f"{dataset}_{mode}_{alignment}_{norm}.html"


def model_name(dataset, one_to_one, ttp_aligned, normalised, label1=None, label2=None):
    name = preprocessed_name(dataset, one_to_one, ttp_aligned, normalised, label1, label2)
    return name[:-4] + '.keras'


def manifest_name(dataset, one_to_one, ttp_aligned, normalised):
    alignment = 'aligned' if ttp_aligned else 'unaligned'
    norm = 'normalised' if normalised else 'unnormalised'
    mode = 'onetone' if one_to_one else 'full'

    return f"{dataset}_{mode}_{alignment}_{norm}_manifest.json"


def save_manifest(manifest, path):
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2)


def load_manifest(path):
    with open(path) as f:
        return json.load(f)


def save_preprocessed(curves, well_labels, path):
    cols = {f'cycle_{i}': curves[:, i] for i in range(curves.shape[1])}
    cols['Label'] = well_labels
    pd.DataFrame(cols).to_csv(path, index=False)


def load_preprocessed(path):
    df = pd.read_csv(path)
    labels = df['Label'].to_numpy()
    curves = df.drop(columns=['Label']).to_numpy(dtype=np.float64)

    return curves, labels
