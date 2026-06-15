import numpy as np


def change_point_detection(time_min, linearized_signal):
    """
    Detect a single change point from a linearized signal.

    Parameters
    ----------
    time_min : array-like
        Time axis in minutes.
    linearized_signal : array-like
        Linearized signal values aligned with time_min.

    Returns
    -------
    dict
        {
            "found_change_point": bool,
            "change_idx": int or None,
            "change_time_min": float or None,
            "change_value": float or None,
            "change_score": float or None,
        }
    """
    t = np.asarray(time_min, dtype=float)
    y = np.asarray(linearized_signal, dtype=float)

    n = int(min(t.size, y.size))
    if n < 3:
        return {
            "found_change_point": False,
            "change_idx": None,
            "change_time_min": None,
            "change_value": None,
            "change_score": None,
        }

    t = t[:n]
    y = y[:n]

    dy = np.diff(y)
    idx_local = int(np.argmax(np.abs(dy)))
    idx_cp = idx_local + 1  # diff index maps to right-side sample

    return {
        "found_change_point": True,
        "change_idx": idx_cp,
        "change_time_min": float(t[idx_cp]),
        "change_value": float(y[idx_cp]),
        "change_score": float(np.abs(dy[idx_local])),
    }

