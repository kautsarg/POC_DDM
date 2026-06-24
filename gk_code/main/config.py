import os
import sys
from pathlib import Path

# model_utils.py lives alongside this file's package; ensure it is importable
# regardless of the caller's own sys.path setup.
sys.path.insert(0, str(Path(__file__).resolve().parent / "utils" / "model_training"))

from model_utils import set_global_determinism
set_global_determinism(0)

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, ListedColormap
from cycler import cycler

# ==========================================
# GLOBAL CONFIGURATION
# ==========================================
# Used by: every pipeline stage (01-08) and light_pipeline/* via
# `--exp_folder` CLI defaults; get_viz_dir() is used by 02, 05, 06, 07, 08,
# and light_pipeline/attribution_vis.py to mirror an experiment folder under
# the visualisation output root.

LAB_ROOT_FOLDER = "/vol/bitbucket/gk225/POC_DDM_datasets"
HPC_ROOT_FOLDER = "/rds/general/user/gk225/home/POC_DDM_datasets/POC_DDM_datasets"
LOCAL_ROOT_FOLDER = "/Users/kautsarg/Documents/Final Project/Run Data"

# Pick whichever root actually exists on this machine, in priority order.
if os.path.exists(LAB_ROOT_FOLDER):
    BASE_FOLDER = LAB_ROOT_FOLDER
elif os.path.exists(HPC_ROOT_FOLDER):
    BASE_FOLDER = HPC_ROOT_FOLDER
elif os.path.exists(LOCAL_ROOT_FOLDER):
    BASE_FOLDER = LOCAL_ROOT_FOLDER
else:
    BASE_FOLDER = LAB_ROOT_FOLDER

VIZ_BASE_FOLDER = BASE_FOLDER.replace("POC_DDM_datasets", "POC_DDM_viz")

DEFAULT_EXP_FOLDER = os.path.join(BASE_FOLDER, "POC_DDM_chip_init")
LAB_EXP_FOLDER = os.path.join(BASE_FOLDER, "LAB_DDM_paper")
MULTI_EXP_FOLDER = os.path.join(BASE_FOLDER, "POC_DDM_multi")

def get_viz_dir(path, subdir):
    """Mirror `path` (a folder under BASE_FOLDER) under VIZ_BASE_FOLDER and append `subdir`."""
    rel = Path(path).relative_to(BASE_FOLDER)
    return Path(VIZ_BASE_FOLDER) / rel / subdir
EXCLUDED_FOLDERS = ['.DS_Store', 'model_interpretation', 'model_interpretation_old', 'outlier_visualisation', 'outlier_visualisation_old', 'cross_dataset_cv', 'model_performance_viz', 'model_performance_viz_old', 'outlier_visualisation', 'outlier_visualisation_old']

# ==========================================
# CURVE TYPE RESOLUTION
# ==========================================
# Used by: 03_main_training.py, 08_statistical_comparison.py (CURVE_TYPE_ALIASES);
# resolve_curve_dataset_idx() is used by 04, 05, 06, 07, model_for_xai.py to
# resolve a `--curve_type` CLI value to its index in the joblib dataset list.

CURVE_TYPE_ALIASES = {
    "ori_curve": "ori_curves",
    "ori_curve_avg": "ori_curves_avg",
}

def resolve_curve_dataset_idx(curve_type, dataset_name_list):
    """Resolve a `--curve_type` value to (index, dataset_name) within `dataset_name_list`."""
    target = CURVE_TYPE_ALIASES.get(curve_type, curve_type)
    if target not in dataset_name_list:
        raise ValueError(f"curve_type '{curve_type}' (resolved to '{target}') not found in dataset_name {list(dataset_name_list)}")
    return list(dataset_name_list).index(target), target

# ==========================================
# EXPERIMENT PARAMETERS
# ==========================================
# Used by: 01_curve_preprocessing_v6.py (N_WELLS, N_A_TYPE);
# 02_outlier_detection_pipeline.py, 07_attribution_vis_all.py (N_WELLS only);
# light_pipeline/00_light-pipeline.py, light_pipeline/01_light-curve_preprocessing.py.
N_WELLS = 10
N_A_TYPE = "v06"

# ==========================================
# PREPROCESSING WINDOW SIZES
# ==========================================
# Used by: 01_curve_preprocessing_v6.py (moving-average window sizes for
# ori_curves_avg / derivative smoothing).
WINDOW_SIZE_ORI = 50
WINDOW_SIZE_1STDER = 200

# ==========================================
# PLOTTING / DOWNSAMPLING
# ==========================================
# Used by: 01_curve_preprocessing_v6.py (Bokeh plot downsampling/rounding).
PLOT_DOWNSAMPLE_STEP = 1
PLOT_DECIMAL_PRECISION = 4

# ==========================================
# AUTOENCODER OUTLIER DETECTION
# ==========================================
# Used by: 02_outlier_detection_pipeline.py and light_pipeline/00, /02, /03
# (downsample factor for the CNN/LSTM autoencoder outlier filters).
AE_DOWNSAMPLE_FACTOR = 1

# ==========================================
# SPATIAL CONSISTENCY OUTLIER FILTERS
# ==========================================
# Used by: 02_outlier_detection_pipeline.py (spatial kNN / grid consistency
# outlier filters).
SPATIAL_CONSISTENCY_KNN_K = 24       # Number of nearest neighbors (by pixel coordinate distance)
SPATIAL_CONSISTENCY_GRID_WINDOW = 2  # Grid half-width -> (2*window+1)^2 neighborhood (5x5)

# ==========================================
# OUTLIER FILTER LABELS
# ==========================================
# Used by: 03_main_training.py, 05_outlier_visualization_report.py,
# 08_statistical_comparison.py — iterated over to train/report/compare each
# outlier filter (None = no filtering / baseline).
# All outlier filter labels produced by 02_outlier_detection_pipeline.py
OUTLIER_FILTERS = [
    None,

    'msc_label_msc_linear_0.001',
    # 'msc_label_msc_baseline_0.001',

    'amf_label_amf_important',
    # 'amf_label_amf_send_5',

    'knn_top_0.95',
    # 'knn_top_0.9',
    # 'knn_top_0.85',

    # f'cnn_ae_pw_ds{AE_DOWNSAMPLE_FACTOR}_label_elbow',
    # # f'cnn_ae_pw_ds{AE_DOWNSAMPLE_FACTOR}_label_95',
    # # f'cnn_ae_pw_ds{AE_DOWNSAMPLE_FACTOR}_label_90',

    f'cnn_ae_glb_ds{AE_DOWNSAMPLE_FACTOR}_label_elbow',
    # f'cnn_ae_glb_ds{AE_DOWNSAMPLE_FACTOR}_label_95',
    # f'cnn_ae_glb_ds{AE_DOWNSAMPLE_FACTOR}_label_90',

    # f'lstm_ae_pw_ds{AE_DOWNSAMPLE_FACTOR}_label_elbow',
    # # f'lstm_ae_pw_ds{AE_DOWNSAMPLE_FACTOR}_label_95',
    # # f'lstm_ae_pw_ds{AE_DOWNSAMPLE_FACTOR}_label_90',

    f'lstm_ae_glb_ds{AE_DOWNSAMPLE_FACTOR}_label_elbow',
    # f'lstm_ae_glb_ds{AE_DOWNSAMPLE_FACTOR}_label_95',
    # f'lstm_ae_glb_ds{AE_DOWNSAMPLE_FACTOR}_label_90',

    'spatial_knn_label_elbow',
    # 'spatial_knn_label_90',
    # 'spatial_knn_label_95',

    'spatial_grid_label_elbow',
    # 'spatial_grid_label_90',
    # 'spatial_grid_label_95',
]

# ==========================================
# RESULT / CACHE FILE PATHS
# ==========================================
# Used by: PREPROCESSED_CURVES_PATH (01, 02); TRAINING_DATA_PATH (02, 03, 04,
# 05, 06, 07, light_pipeline/attribution_vis.py, model_for_xai.py,
# resampling_check.py); TRAINING_RESULT_PATH / TRAINING_10FOLD_RESULT_PATH
# (03, 06, 08); CROSS_DATASET_RESULT_PATH (04); CROSS_DATASET_RESAMPLER_PATH
# (04, resampling_check.py).
PREPROCESSED_CURVES_PATH = 'preprocessed_curves.joblib'
TRAINING_DATA_PATH = 'curve_for_training.joblib'
TRAINING_RESULT_PATH = 'classification_performances.joblib'
TRAINING_10FOLD_RESULT_PATH = 'classification_performances_10fold.joblib'
CROSS_DATASET_RESULT_PATH = 'cross_dataset_classification_performances_{mode}_{curve_type}.joblib'
CROSS_DATASET_RESAMPLER_PATH = 'cross_dataset_resampler_classification_performances_{curve_type}.joblib'

# ==========================================
# MATPLOTLIB GLOBAL STYLE
# ==========================================
# Used by: every script that does `import config` and then creates a
# matplotlib figure — sets the default colour cycle at import time. No
# direct references elsewhere by name.
mpl_colors = [
    (0.00, 0.45, 0.70), (0.90, 0.60, 0.00), (0.35, 0.70, 0.90),
    (0.00, 0.60, 0.50), (0.95, 0.90, 0.25), (0.80, 0.40, 0.70),
    (0.20, 0.13, 0.53), (0.87, 0.80, 0.47), (0.27, 0.67, 0.60), (0.65, 0.65, 0.65)
]
plt.rcParams['axes.prop_cycle'] = cycler(color=[to_hex(i) for i in mpl_colors])

# ==========================================
# UNIFIED VISUALISATION COLOUR PALETTE
# ==========================================
# Used by: get_palette() — 02_outlier_detection_pipeline.py,
# resampling_check.py, utils/model_training/model_utils.py.
# WELL_COLORS / WELL_CMAP — 02, 07_attribution_vis_all.py,
# light_pipeline/attribution_vis.py, utils/01_curve_preprocessing/
# curve_preprocessing_plots.py, utils/02_outlier_detection/chip_utilities.py,
# utils/02_outlier_detection/knn_fingerprint_filter.py.
# WELL_COLOR_MAP — 02_outlier_detection_pipeline.py.
# FILTER_COLORS — utils/model_training/model_utils.py.
# MODEL_COLORS / CURVE_COLORS are reserved fixed-colour maps for get_palette's
# `fixed_map` argument; not currently wired into any call site.
#
# Colour-blind-safe, high-contrast palette (same colours as the global mpl
# cycle above) exposed as hex strings so seaborn/plotly categorical plots
# (e.g. grouped bar charts comparing models or curve types) can reuse it,
# keeping colours consistent and non-overlapping across all figures.
VIZ_PALETTE = [to_hex(c) for c in mpl_colors]

# Fixed colour per model architecture so the same model always has the same
# colour across line plots, bar charts, and the interactive HTML report.
MODEL_COLORS = {
    "CNN": VIZ_PALETTE[0],
    "LSTM": VIZ_PALETTE[1],
    "GRU": VIZ_PALETTE[2],
    "RNN": VIZ_PALETTE[3],
    "Transformer": VIZ_PALETTE[4],
    "RandomForest": VIZ_PALETTE[5],
    "KNN": VIZ_PALETTE[6],
    "LR (FFI)": VIZ_PALETTE[7],
}

# Fixed colour per curve type so the same curve type always has the same
# colour across grouped bar charts comparing outlier filters.
CURVE_COLORS = {
    "Ori Curves": VIZ_PALETTE[0],
    "Original Fitted Full": VIZ_PALETTE[1],
    "Cleaned Std Fitted Full": VIZ_PALETTE[2],
    "Cleaned Std Fitted Stretched": VIZ_PALETTE[3],
    "Cleaned Lowest Fitted Full": VIZ_PALETTE[4],
    "Cleaned Lowest Fitted Stretched": VIZ_PALETTE[5],
}


def get_palette(categories, fixed_map=None):
    """
    Build a {category: hex_color} map from VIZ_PALETTE for a grouped/hue bar
    chart. Categories present in `fixed_map` (e.g. MODEL_COLORS, CURVE_COLORS)
    keep their fixed colour so the same category is always the same colour
    across every figure; any other categories get the remaining VIZ_PALETTE
    colours, in order, so no two categories share a colour.
    """
    fixed_map = fixed_map or {}
    remaining = [c for c in VIZ_PALETTE if c not in fixed_map.values()]
    palette = {}
    for category in categories:
        if category in fixed_map:
            palette[category] = fixed_map[category]
        else:
            palette[category] = remaining.pop(0) if remaining else VIZ_PALETTE[len(palette) % len(VIZ_PALETTE)]
    return palette



WELL_COLORS = VIZ_PALETTE[:N_WELLS]
WELL_COLOR_MAP = dict(enumerate(WELL_COLORS))
WELL_CMAP = ListedColormap(WELL_COLORS)
FILTER_COLORS = get_palette([f for f in OUTLIER_FILTERS if f is not None])
FILTER_COLORS[None] = '#888888'

# ==========================================
# MODEL / FILTER / CURVE DISPLAY MAPS
# ==========================================
# Used by: 06_model_prediction_report.py and 08_statistical_comparison.py to
# look up the joblib key prefixes written by utils/model_training/model_utils.py
# (MODEL_KEY_MAP), and to render short human-readable labels in HTML reports
# (MODEL_PRINT_MAP, FILTER_PRINT_MAP, CURVE_PRINT_MAP, METRIC_LABEL).

# model key -> (y_preds_ key, y_probs_ key, classes_ key) written by
# evaluate_outlier_filters() in utils/model_training/model_utils.py.
MODEL_KEY_MAP = {
    "cnn":          ("y_preds_AC_",            "y_probs_AC_",            "classes_AC_"),
    "lstm":         ("y_preds_AC_lstm_",        "y_probs_AC_lstm_",       "classes_AC_lstm_"),
    "gru":          ("y_preds_AC_gru_",         "y_probs_AC_gru_",        "classes_AC_gru_"),
    "rnn":          ("y_preds_AC_rnn_",         "y_probs_AC_rnn_",        "classes_AC_rnn_"),
    "transformer":  ("y_preds_AC_trans_",       "y_probs_AC_trans_",      "classes_AC_trans_"),
    "rf":           ("y_preds_AC_rf_",          "y_probs_AC_rf_",         "classes_AC_rf_"),
    "knn":          ("y_preds_AC_kNN_",         "y_probs_AC_kNN_",        "classes_AC_kNN_"),
    "ffi":          ("y_preds_FFI_",            "y_probs_FFI_",           "classes_FFI_"),
    "cnn_lf":       ("y_preds_AC_cnn_lf_",      "y_probs_AC_cnn_lf_",     "classes_AC_cnn_lf_"),
    "lstm_lf":      ("y_preds_AC_lstm_lf_",     "y_probs_AC_lstm_lf_",    "classes_AC_lstm_lf_"),
    "trans_lf":     ("y_preds_AC_trans_lf_",    "y_probs_AC_trans_lf_",   "classes_AC_trans_lf_"),
    "gru_lf":       ("y_preds_AC_gru_lf_",      "y_probs_AC_gru_lf_",     "classes_AC_gru_lf_"),
    "cnn_gru_dual": ("y_preds_AC_cnn_gru_dual_","y_probs_AC_cnn_gru_dual_","classes_AC_cnn_gru_dual_"),
    "cnn_trans_dual":("y_preds_AC_cnn_trans_dual_","y_probs_AC_cnn_trans_dual_","classes_AC_cnn_trans_dual_"),
    "lstm_ae_clf":  ("y_preds_AC_lstm_ae_clf_",  "y_probs_AC_lstm_ae_clf_", "classes_AC_lstm_ae_clf_"),
    "cnn_gru_gate":        ("y_preds_AC_cnn_gru_gate_",        "y_probs_AC_cnn_gru_gate_",        "classes_AC_cnn_gru_gate_"),
    "cnn_gru_hadamard":    ("y_preds_AC_cnn_gru_hadamard_",    "y_probs_AC_cnn_gru_hadamard_",    "classes_AC_cnn_gru_hadamard_"),
    "cnn_gru_crossattn":   ("y_preds_AC_cnn_gru_crossattn_",   "y_probs_AC_cnn_gru_crossattn_",   "classes_AC_cnn_gru_crossattn_"),
    "cnn_gru_film":        ("y_preds_AC_cnn_gru_film_",        "y_probs_AC_cnn_gru_film_",        "classes_AC_cnn_gru_film_"),
    "cnn_trans_gate":      ("y_preds_AC_cnn_trans_gate_",      "y_probs_AC_cnn_trans_gate_",      "classes_AC_cnn_trans_gate_"),
    "cnn_trans_hadamard":  ("y_preds_AC_cnn_trans_hadamard_",  "y_probs_AC_cnn_trans_hadamard_",  "classes_AC_cnn_trans_hadamard_"),
    "cnn_trans_crossattn": ("y_preds_AC_cnn_trans_crossattn_", "y_probs_AC_cnn_trans_crossattn_", "classes_AC_cnn_trans_crossattn_"),
    "cnn_trans_film":      ("y_preds_AC_cnn_trans_film_",      "y_probs_AC_cnn_trans_film_",      "classes_AC_cnn_trans_film_"),
    # PyTorch/PyG spatial GNN (03b_gnn_spatial_training.py) -- predictions-only entry;
    # no .keras file is saved for it, so 07's XAI pipeline (TF/Keras-specific) simply
    # never finds a model to load and skips it, same as any other missing model name.
    "gnn_gat":             ("y_preds_AC_gnn_gat_",             "y_probs_AC_gnn_gat_",             "classes_AC_gnn_gat_"),
    "gnn_gcn":             ("y_preds_AC_gnn_gcn_",             "y_probs_AC_gnn_gcn_",             "classes_AC_gnn_gcn_"),
}

MODEL_PRINT_MAP = {
    "cnn": "CNN (ACA)", "lstm": "LSTM (ACA)", "gru": "GRU (ACA)",
    "rnn": "RNN (ACA)", "transformer": "Trans (ACA)", "rf": "RF (ACA)",
    "knn": "KNN (ACA)", "ffi": "LR (FFI)",
    "cnn_lf": "CNN LF", "lstm_lf": "LSTM LF", "trans_lf": "Trans LF", "gru_lf": "GRU LF",
    "cnn_gru_dual": "CNN+GRU Dual", "cnn_trans_dual": "CNN+Tr Dual",
    "lstm_ae_clf": "LSTM-AE Clf",
    "cnn_gru_gate": "CNN+GRU Gate", "cnn_gru_hadamard": "CNN+GRU Hadamard",
    "cnn_gru_crossattn": "CNN+GRU CoAttn", "cnn_gru_film": "CNN+GRU FiLM",
    "cnn_trans_gate": "CNN+Tr Gate", "cnn_trans_hadamard": "CNN+Tr Hadamard",
    "cnn_trans_crossattn": "CNN+Tr CoAttn", "cnn_trans_film": "CNN+Tr FiLM",
    "gnn_gat": "GNN (GAT)", "gnn_gcn": "GNN (GCN)",
}

FILTER_PRINT_MAP = {
    None:                             "None (Baseline)",
    "msc_label_msc_linear_0.001":     "MSC",
    "amf_label_amf_important":        "AMF",
    "knn_top_0.95":                   "kNN-95%",
    "cnn_ae_glb_ds1_label_elbow":     "CNN-AE",
    "lstm_ae_glb_ds1_label_elbow":    "LSTM-AE",
    "spatial_knn_label_elbow":        "Spatial-kNN",
    "spatial_grid_label_elbow":       "Spatial-Grid",
}

CURVE_PRINT_MAP = {
    "ori_curves":     "Ori Curves (raw)",
    "ori_curves_avg": "Ori Curves (avg)",
}

METRIC_LABEL = {"accuracy": "Accuracy", "macro_f1": "Macro-F1", "mcc": "MCC"}

# ==========================================
# FEATURE SELECTION
# ==========================================
# Used by: EXCLUDED_FEATURES, FEATURE_GROUPS — 02_outlier_detection_pipeline.py.
# CURVE_SPLIT, IMPORTANCE_METRICS — not currently referenced outside config.py
# (reserved for future feature-importance reporting).
EXCLUDED_FEATURES = [
    "pixel_row_idx", "pixel_col_idx", "temp_group_idx", "num_active_pixels_in_temp_group",
    "well_temp_lin2d_mean", "well_2d_temp_npr_mean",
    "5p_sigmoid_fitted", "5p_sigmoid_fitted_dydx",
    "Send", "Send_abs", "Send_fit", "Send_fit_abs",
    "cnn_ae_pw_ds1_label_elbow", "cnn_ae_pw_ds1_label_90", "cnn_ae_pw_ds1_label_95",
    "cnn_ae_glb_ds1_label_elbow", "cnn_ae_glb_ds1_label_90", "cnn_ae_glb_ds1_label_95",
    "lstm_ae_pw_ds1_label_elbow", "lstm_ae_pw_ds1_label_90", "lstm_ae_pw_ds1_label_95",
    "lstm_ae_glb_ds1_label_elbow", "lstm_ae_glb_ds1_label_90", "lstm_ae_glb_ds1_label_95",
    "knn_top_0.85", "knn_top_0.9", "knn_top_0.95",
    "spatial_knn_label_elbow", "spatial_knn_label_90", "spatial_knn_label_95",
    "spatial_grid_label_elbow", "spatial_grid_label_90", "spatial_grid_label_95",
    "msc_mahal_dist",
    "msc_mahal_dist_msc_linear_0.001", "msc_label_msc_linear_0.001",
    "msc_mahal_dist_msc_baseline_0.001", "msc_label_msc_baseline_0.001",
    "amf_label_amf_important", "amf_label_amf_send_5",
    "fit_rmse", "fit_r2",
]

FEATURE_GROUPS = [
    # A. Baseline / Noise
    ['F0', 'log_F0', 'baseline_mean', 'baseline_std', 'baseline_slope', 'snr_peak', 'snr_xms'],

    # B. End / Plateau behavior
    ['F_max', 'F_max_ori', 'Fm', 'FFI', 'F_range', 'plateau_mean', 'plateau_std', 'plateau_slope', 'overshoot_index'],

    # C. Early kinetic onset
    ['Ct', 'Ct_ori', 'ct_idx', 'ct_idx_ori', 'lag_time', 't10'],

    # D. Mid-rise timing
    ['t50', 'xms'],

    # E. Rise dynamics
    ['dy_xms', 'max_accel', 'accel_fwhm', 'rise_time_10_90', 'rise_time_20_80'],

    # F. Threshold window
    ['xs', 'xe', 'threshold_distance'],

    # G. First-half kinetics
    ['first_half_distance', 'A1'],

    # H. Second-half kinetics
    ['second_half_distance', 'A2'],

    # I. Asymmetry
    ['distance_asymmetry_index', 'area_asymmetry_index', 'peak_asymmetry_index'],

    # J. Inflection geometry (fit)
    ['Cy0', 'Cy0_ori', 'Cs', 'Sc', 'As'],

    # K. Peak-shift & curvature
    ['xp1', 'xp2', 'peak_shifting_distance', 'd2y_xp1', 'd2y_xp2'],

    # L. Signal amplitude / critical Y values
    ['amplitude', 'y_xs', 'y_xe', 'y_xms', 'y_xp1', 'y_xp2'],

    # M. Tail slope / drift
    ['send_5', 'send_10', 'send_15', 'send_20', 'send_25',
     'send_abs_5', 'send_abs_10', 'send_abs_15', 'send_abs_20', 'send_abs_25',
     'Send', 'Send_abs', 'Send_fit', 'Send_fit_abs'],
]

CURVE_SPLIT = {
    'original_curves': ['ori_curves'],
    'fitted_curves': ['original_fitted_full', 'original_fitted_stretched'],
    'preprocessed_curves': ['cleaned_std_fitted_full', 'cleaned_std_fitted_stretched'],
    'preprocessed_stretched_curves': ['cleaned_lowest_fitted_full', 'cleaned_lowest_fitted_stretched'],
}

IMPORTANCE_METRICS = [
    "rf_importance",
    "anova_score",
    "silhouette_score",
    "mutual_info_score",
    "kruskal_wallis_score"
]


XAI_KINETIC_FEATURE_GROUP = {
    # ---------------------------------------------------------
    # 1. Threshold & Main Timing (When reaction hits 20% target)
    # ---------------------------------------------------------
    "Ct": [                    
        "Ct_ori",       # Fit-based 20% crossing
        "ct_idx",       # Array index of fit crossing
        "ct_idx_ori",   # Array index of original crossing
        "Cy0",          # Fit-based inflection point timing
        "Cy0_ori",      # Original curve inflection timing
    ],

    # ---------------------------------------------------------
    # 2. Onset Timing (When the reaction FIRST leaves the noise floor)
    # ---------------------------------------------------------
    "lag_time": [          
        "xs",           # Start time via derivative threshold crossing
        "xp1",          # Time of maximum acceleration (2nd deriv peak)
        "t10"           # 10% normalized amplitude crossing
    ],

    # ---------------------------------------------------------
    # 3. Midpoint Timing (When the reaction is moving fastest)
    # ---------------------------------------------------------
    "xms": [            # Time of max slope (empirical)
        "Cs",           # Fit-based center of symmetry
        "t50"           # 50% normalized amplitude crossing
    ],

    # ---------------------------------------------------------
    # 4. Saturation Timing (When the reaction flattens out)
    # ---------------------------------------------------------
    "xe": [             # End time via derivative threshold crossing
        "xp2",          # Time of maximum deceleration (2nd deriv valley)
        "t90"           # 90% normalized amplitude crossing
    ],

    # ---------------------------------------------------------
    # 5. Reaction Duration (Length of the exponential phase)
    # ---------------------------------------------------------
    "rise_time_10_90": [
        "rise_time_20_80",       # 20% to 80% duration
        "threshold_distance",    # Distance between xe and xs
        "first_half_distance",   # Distance between xms and xs
        "second_half_distance",  # Distance between xe and xms
        "peak_shifting_distance" # Distance between deceleration and acceleration peaks
    ],

    # ---------------------------------------------------------
    # 6. Maximum Reaction Speed (1st Derivative / Slopes)
    # ---------------------------------------------------------
    "dy_xms": [         # Empirical maximum slope
        "Sc",           # Fit-based slope parameter
        "dy_xp1",       # 1st derivative value at max acceleration
        "dy_xp2",       # 1st derivative value at max deceleration
        "TH"            # The calculated derivative threshold value
    ],

    # ---------------------------------------------------------
    # 7A. Positive Acceleration (Kicking Off)
    # ---------------------------------------------------------
    "max_accel": [      # Absolute highest positive acceleration
        "d2y_xp1"       # 2nd derivative value at positive peak
    ],

    # ---------------------------------------------------------
    # 7B. Negative Acceleration (Hitting the Brakes)
    # ---------------------------------------------------------
    "min_accel": [      # Absolute lowest negative acceleration
        "d2y_xp2"       # 2nd derivative value at negative peak (deceleration)
    ],

    # ---------------------------------------------------------
    # 8. Magnitude & Final Yield (Top of the curve)
    # ---------------------------------------------------------
    "amplitude": [      # Absolute change (y_xe - y_xs)
        "F_max",        # Absolute max of fitted curve
        "F_max_ori",    # Absolute max of original curve
        "Fm",           # Fit-based max parameter
        "plateau_mean", # Mean value of the final flat region
        "y_xe",         # Y-value at end-time threshold
        "y_xp2"         # Y-value at max deceleration
    ],

    # ---------------------------------------------------------
    # 9. Baseline / Noise Floor (Bottom of the curve)
    # ---------------------------------------------------------
    "baseline_mean": [
        "Fb",           # Fit-based minimum parameter
        "F0",           # Literal first y-value
        "log_F0",       # Log of literal first y-value
        "y_xs"          # Y-value at start-time threshold
    ],

    # ---------------------------------------------------------
    # 10. Intermediate Signal States
    # ---------------------------------------------------------
    "y_xms": [          # Y-value precisely at max slope
        "y_xp1"         # Y-value precisely at max acceleration
    ],

    # ---------------------------------------------------------
    # 11. Curve Area / Integration
    # ---------------------------------------------------------
    "auc_norm": [       # Area under curve (normalized by duration)
        "auc",          # Raw area under curve
        "A1",           # Area of the 1st half derivative
        "A2"            # Area of the 2nd half derivative
    ],

    # ---------------------------------------------------------
    # 12. Curve Shape & Asymmetry
    # ---------------------------------------------------------
    "area_asymmetry_index": [ 
        "As",                       # Fit-based asymmetry parameter
        "distance_asymmetry_index", # Asymmetry based on timing distances
        "peak_asymmetry_index",     # Asymmetry based on 2nd derivative peak heights
        "accel_fwhm"                # Full width at half max of the acceleration peak
    ],

    # ---------------------------------------------------------
    # 13. Signal Noise & Physical Stability
    # ---------------------------------------------------------
    "snr_peak": [       # Signal-to-Noise against max peak
        "snr_xms",      # Signal-to-Noise against max slope point
        "baseline_std", # Standard deviation of the pre-reaction sensor noise
        "plateau_std"   # Standard deviation of the post-reaction sensor noise
    ],

    # ---------------------------------------------------------
    # 14A. Pre-Reaction Drift (Sensor Settling)
    # ---------------------------------------------------------
    "baseline_slope": [], 

    # ---------------------------------------------------------
    # 14B. Post-Reaction Drift (Reagent Exhaustion/Decay)
    # ---------------------------------------------------------
    "plateau_slope": [
        "Send",         # Mean derivative of last 5 points
        "Send_abs",     # Absolute mean derivative of last 5 points
        "Send_fit",     # Mean derivative of last 5 points (fitted)
        "Send_fit_abs"  # Absolute mean deriv of last 5 points (fitted)
    ],

    # ---------------------------------------------------------
    # 16. Overshoot (Curve dips/rebounds past its plateau)
    # ---------------------------------------------------------
    "overshoot_index": [],
}

# ==========================================
# VISUALISATION OVERRIDES
# ==========================================
# Used by: 01_curve_preprocessing_v6.py, 02_outlier_detection_pipeline.py via
# `getattr(config, "SAVED_VIZ", [])` — manual allow-list of dataset names to
# always (re)generate plots for; currently empty.
SAVED_VIZ = [
    # "D20250808_E00_C00_F4500KHz_U_Sample_7",
    # "D20250820_E00_C00_F4500KHz_U_Sample_18_wet_3"
]

# ==========================================
# FEATURE SUBSET FOR LATE FUSION
# ==========================================
# Used by: 03_main_training.py, 04_cross_dataset_training.py,
# 07_attribution_vis_all.py, model_for_xai.py — fixed top-level kinetic
# feature subset fed into the late-fusion (LF) model branch.
LD_FEATURES = [
    'Fm', 'Fb', 'Sc', 'Cs', 'As', 'xms', 'xs', 'xe', 'xp1', 'xp2', 'TH',
    'y_xms', 'y_xs', 'y_xe', 'y_xp1', 'y_xp2', 'amplitude', 'dy_xms',
    'dy_xp1', 'dy_xp2', 'd2y_xp1', 'd2y_xp2', 'threshold_distance',
    'first_half_distance', 'second_half_distance', 'distance_asymmetry_index',
    'peak_shifting_distance', 'A1', 'A2', 'area_asymmetry_index',
    'peak_asymmetry_index', 'Ct', 'Cy0', 'F_max', 'log_F0', 'F0', 'Send',
    'FFI', 'F_range'
]

# ==========================================
# MODEL RERUN OVERRIDES
# ==========================================
# Used by: 03_main_training.py, 04_cross_dataset_training.py — list of model
# keys to force-retrain even if cached results exist; currently empty.
RERUN_MODELS = [
    # "cnn_lf"
]

# ==========================================
# DATASET LABEL MAPPINGS
# ==========================================
# Used by: 01_curve_preprocessing_v6.py, 03_main_training.py,
# 04_cross_dataset_training.py, 05_outlier_visualization_report.py,
# 06_model_prediction_report.py, 07_attribution_vis_all.py,
# model_for_xai.py, resampling_check.py — maps each experiment folder's raw
# well index to its class label (per-dataset, since well layout varies).
LABEL_MAPPINGS = {
	'D20260320_E00_C00_F4500KHz_U_Elena_steap_cv': {
		0: 'S',
		1: 'C',
		2: 'C',
		3: 'S',
		4: 'S',
		5: 'C',
		6: 'C',
		7: 'S',
		8: 'NC-S',
		9: 'NC-C',
	},
    'D20260522_E00_C00_F4500KHz_U_manifold_test_05': {
		0: 'Target',
		1: 'NC',
		2: 'NC',
		3: 'Target',
		4: 'Target',
		5: 'NC',
		6: 'NC',
		7: 'Target',
		8: 'Target',
		9: 'NC',
	},
    'D20260608_E00_C00_F4500KHz_U_norm_temp_04':{
        0: 'Target',
		1: 'NC',
		2: 'NC',
		3: 'Target',
		4: 'Target',
		5: 'NC',
		6: 'NC',
		7: 'Target',
		8: 'Target',
		9: 'NC',
    },
    'D20260609_E00_C00_F4500KHz_U_norm_temp_read_06':{
        0: 'Target',
		1: 'NC',
		2: 'NC',
		3: 'Target',
		4: 'Target',
		5: 'NC',
		6: 'NC',
		7: 'Target',
		8: 'Target',
		9: 'NC',
    },
    'D20260609_E00_C00_F4500KHz_U_norm_temp_read_07':{
        0: 'Target',
		1: 'NC',
		2: 'NC',
		3: 'Target',
		4: 'Target',
		5: 'NC',
		6: 'NC',
		7: 'Target',
		8: 'Target',
		9: 'NC',
    },
    'D20260609_E00_C00_F4500KHz_U_norm_temp_ready_08':{
        0: 'Target',
		1: 'NC',
		2: 'NC',
		3: 'Target',
		4: 'Target',
		5: 'NC',
		6: 'NC',
		7: 'Target',
		8: 'Target',
		9: 'NC',
    },
    'D20260611_E00_C00_F4500KHz_U_lambda_test_manifold_01': {
		0: 'Conc-01',
		1: 'Conc-02',
		2: 'Conc-02',
		3: 'Conc-01',
		4: 'Conc-01',
		5: 'Conc-02',
		6: 'Conc-02',
		7: 'Conc-01',
		8: 'NC-Conc-01',
		9: 'NC-Conc-02',
	},
}

def get_label_mappings(exp_path):
    """Picks LABEL_MAPPINGS based on exp_path's dataset folder (e.g. 'POC_DDM_multi'
    vs 'POC_DDM_multi_nc_subtract'). The nc_subtract preprocessing collapses
    per-target NCs (e.g. 'NC-S'/'NC-C') into a single 'NC' class, so every label
    starting with 'NC-' is collapsed to plain 'NC' — otherwise NCs for different
    targets get conflated under LABEL_MAPPINGS' per-target NC labels.
    """
    if Path(exp_path).parent.name.endswith("nc_subtract"):
        return {
            dataset: {
                idx: ('NC' if label.startswith('NC-') else label)
                for idx, label in mapping.items()
            }
            for dataset, mapping in LABEL_MAPPINGS.items()
        }
    return LABEL_MAPPINGS

# ==========================================
# CROSS-DATASET ROBUSTNESS CV (04)
# ==========================================
# Used by: 04_cross_dataset_training.py, resampling_check.py — groups of
# experiment folders that share a label mapping, combined for leave-one-
# folder-out (LOFO) cross-validation.
CROSS_DATASET_GROUPS = {
    # 'group_name': ['exp_folder_1', 'exp_folder_2', ...],
    'init_oneplex_v6': ['D20260608_E00_C00_F4500KHz_U_norm_temp_04', 'D20260609_E00_C00_F4500KHz_U_norm_temp_read_06', 'D20260609_E00_C00_F4500KHz_U_norm_temp_read_07', 'D20260609_E00_C00_F4500KHz_U_norm_temp_ready_08'],
    'init_oneplex_nc_subtract': ['D20260608_E00_C00_F4500KHz_U_norm_temp_04', 'D20260609_E00_C00_F4500KHz_U_norm_temp_read_06', 'D20260609_E00_C00_F4500KHz_U_norm_temp_read_07', 'D20260609_E00_C00_F4500KHz_U_norm_temp_ready_08'],
}


