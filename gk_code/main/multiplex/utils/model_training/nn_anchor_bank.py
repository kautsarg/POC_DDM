import numpy as np


def build_nn_anchor_bank(X_curves, y_binary, n_targets=3, max_per_target=500):
    """Extract per-target single-target reference curves for NN anchor loss.

    Parameters
    ----------
    X_curves       : (N, T) float array — normalised curve time-series
    y_binary       : (N, n_targets) int array — multi-label binary targets
    n_targets      : int
    max_per_target : int — cap reference bank size to bound GPU memory.
        Distance matrix shape is (batch, N_ref, T); with batch=512, T=45,
        N_ref=500: ~46 MB/channel. Without cap (N_ref≈7000): ~630 MB/channel.

    Returns
    -------
    banks : list of n_targets float32 arrays, each (≤max_per_target, T)
        Only strictly single-target wells are kept. Subsampling is seeded
        (seed=42) for reproducibility. No concentration information is used.
    """
    X = np.asarray(X_curves, dtype=np.float32)
    Y = np.asarray(y_binary, dtype=np.int32)
    single_mask = Y.sum(axis=1) == 1
    rng = np.random.default_rng(42)
    banks = []
    for j in range(n_targets):
        mask_j = single_mask & (Y[:, j] == 1)
        curves_j = X[mask_j]
        if len(curves_j) > max_per_target:
            idx = rng.choice(len(curves_j), max_per_target, replace=False)
            curves_j = curves_j[idx]
        banks.append(curves_j)
    return banks
