import numpy as np
from scipy.signal import convolve
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

from titan_v4 import functions
import titan_v4.load_functions as load


def generate_params(frame_3d, gain_3d, vref, nrows=290, ncols=204):
    """
    Generates parameter arrays for linearisation based on input 3D data and gain/vref values.

    Parameters:
        frame_3d (np.ndarray): A 3D array of input data.
        gain_y (np.ndarray): A 3D array containing initial (tau_i) and final (tau_f) gain values to be extracted.
        vref (float): A reference voltage value used to calculate the initial and final voltage values (V_i, V_f).

    Returns:
        params (np.ndarray): A 3D array with shape (nrows, ncols, 4), containing parameters: A, B, C, and D.
    """

    params = np.zeros((nrows, ncols, 4))  # Initialise params with shape (nrows, ncols, 4)
    # print(f"DEBUG: shape of gain_3d {gain_3d.shape}")

    if gain_3d.shape[2] == 4:
        c_min = np.min(frame_3d)
        C = np.zeros((nrows, ncols, 1))
        C.fill(c_min)
    elif gain_3d.shape[2] == 8:
        C = gain_3d[:, :, [7]]
        # print(f"DEBUG: found gain file with 8 parameters")
        # print(f"DEBUG: C shape is {C.shape}")
    else:
        raise f"RAISED: Unexpected number of samples in gain file = {gain_3d.shape[2]}. Expect 4 or 8."
    A = 2000 + C  # Calculate A from C (statistically equivalent to fitting)
    dv = 0.058594  # Default gain interval 58.594 mV
    # Extract tau_i and tau_f (y-axis) from gain, assuming the array is composed of 4 samples
    tau_i = gain_3d[:, :, [1]]
    tau_f = gain_3d[:, :, [3]]
    # Calculate V_i and V_f (x-axis)
    V_i = vref
    V_f = vref + dv

    log_div = np.log((tau_i - C) / A) / np.log((tau_f - C) / A)  # Intermediate operation
    D = (V_i - (log_div * V_f)) / (1 - log_div)  # Calculate D
    B = - (1 / (V_f - D)) * np.log((tau_f - C) / A)  # Calculate B

    # Replace NaN or infinite values in B and D
    D[np.isnan(D) | np.isinf(D)] = 0
    B[np.isnan(B) | np.isinf(B)] = 0

    # Assign the generated parameters
    params[:, :, 0] = A.reshape(nrows, ncols)
    params[:, :, 1] = B.reshape(nrows, ncols)
    params[:, :, 2] = C.reshape(nrows, ncols)
    params[:, :, 3] = D.reshape(nrows, ncols)

    return params


def linearise(frame_3d, params, idx_active_gain):
    """
    Applies the 4-parameter pixel-by-pixel linearisation process to a 3D dataset.
    Note: A, B, C, D must be all positive.

    Parameters:
        frame_3d (np.ndarray): A 3D array of input data to be linearised.
        params (np.ndarray): A 3D array of parameters with shape (nrows, ncols, 4), where each (nrows, ncols) slice
                                corresponds to:
                                - A (scale factor, approx)
                                - B (exponent factor, approx)
                                - C (minimum threshold, approx)
                                - D (offset value, approx)

    Returns:
        frame_3d_lin (np.ndarray): A 3D array of linearised data with the same shape as frame_3d.
    """
    frame_3d_lin = np.zeros_like(frame_3d)  # Copy input dataset to new array

    # Read parameters and broadcast to correct size
    A = np.broadcast_to(params[:, :, 0][:, :, np.newaxis], frame_3d_lin.shape)
    B = np.broadcast_to(params[:, :, 1][:, :, np.newaxis], frame_3d_lin.shape)
    C = np.broadcast_to(params[:, :, 2][:, :, np.newaxis], frame_3d_lin.shape)
    D = np.broadcast_to(params[:, :, 3][:, :, np.newaxis], frame_3d_lin.shape)
    idx_active_gain_3d = np.broadcast_to(idx_active_gain.reshape(frame_3d.shape[0], frame_3d.shape[1])[:, :, np.newaxis], frame_3d_lin.shape)

    # Mask incorrect values
    mask = (idx_active_gain_3d
            & (A > 300)  # Exponentials scaled with a smaller number shouldn't be possible
            & (B > 1)  # Slow exponentials most likely represent inactive pixels
            & (frame_3d > C))  # Values smaller than the minimum possible (C)

    # Linearisation
    frame_3d_lin[mask] = (-(1 / B[mask]) * np.log((frame_3d[mask] - C[mask]) / A[mask])) + D[mask]
    frame_3d_lin = np.nan_to_num(frame_3d_lin, nan=0)  # Replaces NaNs

    idx_active_lin2d = ~np.any(frame_3d_lin == 0, axis=2)
    # print(f"DEBUG: number of outliers/n_pixel due to A: {(A < 300).sum()/(290*204)}; B: {(B<1).sum()/(290*204)}; "
    #       f"C: {(frame_3d < C).sum()/(290*204)} -- "
    #       f"Number of pixels inactive due to linearisation but active in gain: {(~idx_active_lin2d.reshape(-1) & idx_active_gain).sum()}")
    return frame_3d_lin, idx_active_lin2d.reshape(-1)