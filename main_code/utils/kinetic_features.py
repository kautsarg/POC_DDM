import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import sigmoid_fitting as sp


def _process_single_row(y, X):
    valid = np.isfinite(X) & np.isfinite(y)
    if np.sum(valid) < 3:
        return {}
    try:
        return sp.extract_kinetic_parameters_original(X, y)
    except Exception:
        return {}


def extract_kinetic_features(timestamps, curves, n_jobs=-1):
    features = Parallel(n_jobs=n_jobs)(
        delayed(_process_single_row)(y, timestamps) for y in curves
    )
    return pd.DataFrame(features)


def get_send(timestamps, curves_2d, send_n=(5, 10, 15, 20, 25)):
    dy_dx_list = Parallel(n_jobs=-1, backend="loky", batch_size='auto')(
        delayed(sp.calculate_first_derivative)(timestamps, y) for y in curves_2d
    )
    dy_dx = np.array(dy_dx_list)
    send_dict = {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        for n in send_n:
            send_dict[f"send_{n}"] = np.nanmean(dy_dx[:, -n:], axis=1)
            send_dict[f"send_abs_{n}"] = np.nanmean(np.abs(dy_dx[:, -n:]), axis=1)
    return send_dict


def build_kinetic_features(curves_2d, timestamps, metadata_df):
    features_df = extract_kinetic_features(timestamps, curves_2d).reset_index(drop=True)
    add_features = get_send(timestamps, curves_2d)
    add_features["FFI"] = curves_2d[:, -1]
    add_features["F_range"] = curves_2d[:, -1] - curves_2d[:, 0]
    return pd.concat(
        [features_df, metadata_df.reset_index(drop=True), pd.DataFrame(add_features).reset_index(drop=True)],
        axis=1,
    )
