import os
from pathlib import Path
from model_utils import set_global_determinism
set_global_determinism(0)

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from cycler import cycler

# ==========================================
# GLOBAL CONFIGURATION
# ==========================================

# Default path for the HPC cluster
DEFAULT_EXP_FOLDER = "/rds/general/user/gk225/home/Run Data/POC_DDM_dataset/"

# Global Experiment Parameters
N_WELLS = 10
N_A_TYPE = "v04"

# Preprocessing Window Sizes
WINDOW_SIZE_ORI = 50
WINDOW_SIZE_1STDER = 200

# Plotting Optimizations (Downsampling for Bokeh)
PLOT_DOWNSAMPLE_STEP = 1
PLOT_DECIMAL_PRECISION = 4

# AutoEncoder Downsample Factor
AE_DOWNSAMPLE_FACTOR = 1


mpl_colors = [
    (0.00, 0.45, 0.70), (0.90, 0.60, 0.00), (0.35, 0.70, 0.90), 
    (0.00, 0.60, 0.50), (0.95, 0.90, 0.25), (0.80, 0.40, 0.70), 
    (0.20, 0.13, 0.53), (0.87, 0.80, 0.47), (0.27, 0.67, 0.60), (0.65, 0.65, 0.65)
]
plt.rcParams['axes.prop_cycle'] = cycler(color=[to_hex(i) for i in mpl_colors])

EXCLUDED_FEATURES = [
    'TH', 'msc_mahal_dist', 'msc_label_0.001', 'amf_label_Send', 
    'amf_label_Send_abs', 'amf_label_Send_fit', 'amf_label_Send_fit_abs', 
    'mean_std_outlier_label_3sigma',
    "pixel_row_idx", "pixel_col_idx", "temp_group_idx", "num_active_pixels_in_temp_group", "well_temp_lin2d_mean", "well_2d_temp_npr_mean"
]

FEATURE_GROUPS = [
    ['F0', 'log_F0'],
    ['send_5', 'send_10', 'send_15', 'send_20', 'send_25', 'send_abs_5', 'send_abs_10', 'send_abs_15', 'send_abs_20', 'send_abs_25', 'Send', 'Send_abs', 'Send_fit', 'Send_fit_abs'],
    ['F_max', 'Fm', 'FFI', 'F_range'],
    ['Ct', 'ct_idx'],
    ['Cy0', 'Cs', 'As', 'Sc'],
    ['first_half_distance', 'A1', 'xs', 'xms'],
    ['second_half_distance', 'A2', 'xms', 'xe'],
    ['threshold_distance', 'xs', 'xe'],
    ['peak_shifting_distance', 'xp1', 'xp2'],
    ['distance_asymmetry_index', 'first_half_distance', 'second_half_distance'],
    ['area_asymmetry_index', 'A1', 'A2'],
    ['peak_asymmetry_index', 'd2y_xp1', 'd2y_xp2'],
    ['xms', 'y_xms', 'dy_xms'],
    ['xp1', 'y_xp1', 'dy_xp1', 'd2y_xp1'],
    ['xp2', 'y_xp2', 'dy_xp2', 'd2y_xp2'],
    ['xs', 'y_xs'],
    ['xe', 'y_xe'],
    ['amplitude', 'y_xs', 'y_xe'],
    ['y_xms', 'y_xs', 'y_xe', 'y_xp1', 'y_xp2', 'F_max', 'Fm', 'FFI', 'F_range']
]