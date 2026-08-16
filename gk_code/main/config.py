import os
import sys
from pathlib import Path
import numpy as np

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
HPC_ROOT_FOLDER = "/rds/general/user/gk225/home/POC_DDM_datasets"
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
LAB_1TO1_EXP_FOLDER = os.path.join(BASE_FOLDER, "LAB_OneToOne")
MULTI_EXP_FOLDER = os.path.join(BASE_FOLDER, "POC_DDM_multi")
FINAL_EXP_FOLDER = os.path.join(BASE_FOLDER, "POC_DDM_final")

def get_viz_dir(path, subdir):
    """Mirror `path` (a folder under BASE_FOLDER) under VIZ_BASE_FOLDER and append `subdir`."""
    rel = Path(path).relative_to(BASE_FOLDER)
    return Path(VIZ_BASE_FOLDER) / rel / subdir
EXCLUDED_FOLDERS = ['.DS_Store', 'model_interpretation', 'model_interpretation_old', 'outlier_visualisation', 'outlier_visualisation_old', 'cross_dataset_cv', 'model_performance_viz', 'model_performance_viz_old',
                    '1_area', '2_range', '3_range_filtered']

# ==========================================
# CURVE TYPE RESOLUTION
# ==========================================
# Used by: 03_main_training.py, 08_statistical_comparison.py (CURVE_TYPE_ALIASES);
# resolve_curve_dataset_idx() is used by 04, 05, 06, 07, model_for_xai.py to
# resolve a `--curve_type` CLI value to its index in the joblib dataset list.

CURVE_TYPE_ALIASES = {
    "ori_curve":                      "ori_curves",
    "ori_curve_avg":                  "ori_curves_avg",
    "ori_curve_norm":                 "ori_curves_norm",
    "ori_curve_avg_norm":             "ori_curves_avg_norm",
    "ori_curve_wavelet_sym8":         "ori_curves_wavelet_sym8",
    "ori_curve_wavelet_sym8_norm":    "ori_curves_wavelet_sym8_norm",
    "ori_curve_wavelet_bior35":       "ori_curves_wavelet_bior35",
    "ori_curve_wavelet_bior35_norm":  "ori_curves_wavelet_bior35_norm",
    "ori_curve_sg_p4":                "ori_curves_sg_p4",
    "ori_curve_sg_p4_norm":           "ori_curves_sg_p4_norm",
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
# (03, 06, 08); CROSS_DATASET_RESULT_PATH (legacy single-file layout, kept as
# a read-only fallback in utils/cross_dataset_result_io.py for groups trained
# before the per-model split -- see CROSS_DATASET_RESULT_PATH_PER_MODEL, the
# current layout written by 04 and read by 04/06b/08); CROSS_DATASET_RESAMPLER_PATH
# (04, resampling_check.py).
PREPROCESSED_CURVES_PATH = 'preprocessed_curves.joblib'
TRAINING_DATA_PATH = 'curve_for_training.joblib'
TRAINING_RESULT_PATH = 'classification_performances.joblib'
TRAINING_10FOLD_RESULT_PATH = 'classification_performances_10fold.joblib'
CROSS_DATASET_RESULT_PATH = 'cross_dataset_classification_performances_{mode}_{curve_type}.joblib'
CROSS_DATASET_RESULT_PATH_PER_MODEL = 'cross_dataset_classification_performances_{mode}_{curve_type}_{model}.joblib'
CROSS_DATASET_RESAMPLER_PATH = 'cross_dataset_resampler_classification_performances_{curve_type}.joblib'
CURVE_ALIGNMENT_CHOICES = ["acquisition_start", "pc_ttp"]
CROSS_DATASET_PC_TTP_RECIPE_PATH = 'cross_dataset_pc_ttp_recipe_{curve_type}.joblib'
CROSS_DATASET_LOFO_AE_PATH = 'lofo_ae_filter/{fold_label}_{curve_type}'
CROSS_DATASET_PC_EMBED_PATH = 'pc_reference_embedding_{model}_{filter}_{curve_type}.joblib'

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
    "cnn_lstm_dual":("y_preds_AC_cnn_lstm_dual_","y_probs_AC_cnn_lstm_dual_","classes_AC_cnn_lstm_dual_"),
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
    # Spatial neighbour-reconstruction variants of cnn_gru_dual (03_main_training.py) --
    # see model_utils.build_neighbor_curve_stack/reconstruct_curves_cosine/
    # create_cnn_gru_dual_attn_recon_model. Same single-curve-input result shape as
    # cnn_gru_dual itself; only the curve fed into the classifier differs.
    "cnn_gru_dual_cosine_recon": ("y_preds_AC_cnn_gru_dual_cosine_recon_", "y_probs_AC_cnn_gru_dual_cosine_recon_", "classes_AC_cnn_gru_dual_cosine_recon_"),
    "cnn_gru_dual_attn_recon":   ("y_preds_AC_cnn_gru_dual_attn_recon_",   "y_probs_AC_cnn_gru_dual_attn_recon_",   "classes_AC_cnn_gru_dual_attn_recon_"),
    # MTL variants — same joblib as standard models; keys distinguished by _mtl_ infix.
    "cnn_mtl":                       ("y_preds_AC_cnn_mtl_",                        "y_probs_AC_cnn_mtl_",                        "classes_AC_cnn_mtl_"),
    "lstm_mtl":                      ("y_preds_AC_lstm_mtl_",                       "y_probs_AC_lstm_mtl_",                       "classes_AC_lstm_mtl_"),
    "gru_mtl":                       ("y_preds_AC_gru_mtl_",                        "y_probs_AC_gru_mtl_",                        "classes_AC_gru_mtl_"),
    "rnn_mtl":                       ("y_preds_AC_rnn_mtl_",                        "y_probs_AC_rnn_mtl_",                        "classes_AC_rnn_mtl_"),
    "transformer_mtl":               ("y_preds_AC_trans_mtl_",                      "y_probs_AC_trans_mtl_",                      "classes_AC_trans_mtl_"),
    "cnn_gru_dual_mtl":              ("y_preds_AC_cnn_gru_dual_mtl_",               "y_probs_AC_cnn_gru_dual_mtl_",               "classes_AC_cnn_gru_dual_mtl_"),
    "cnn_trans_dual_mtl":            ("y_preds_AC_cnn_trans_dual_mtl_",             "y_probs_AC_cnn_trans_dual_mtl_",             "classes_AC_cnn_trans_dual_mtl_"),
    "cnn_gru_dual_cosine_recon_mtl":     ("y_preds_AC_cnn_gru_dual_cosine_recon_mtl_",      "y_probs_AC_cnn_gru_dual_cosine_recon_mtl_",      "classes_AC_cnn_gru_dual_cosine_recon_mtl_"),
    "cnn_gru_dual_attn_recon_mtl":       ("y_preds_AC_cnn_gru_dual_attn_recon_mtl_",        "y_probs_AC_cnn_gru_dual_attn_recon_mtl_",        "classes_AC_cnn_gru_dual_attn_recon_mtl_"),
    "cnn_gru_dual_cosine_recon_supcon":     ("y_preds_AC_cnn_gru_dual_cosine_recon_supcon_",     "y_probs_AC_cnn_gru_dual_cosine_recon_supcon_",     "classes_AC_cnn_gru_dual_cosine_recon_supcon_"),
    "cnn_gru_dual_attn_recon_supcon":       ("y_preds_AC_cnn_gru_dual_attn_recon_supcon_",       "y_probs_AC_cnn_gru_dual_attn_recon_supcon_",       "classes_AC_cnn_gru_dual_attn_recon_supcon_"),
    "cnn_gru_dual_cosine_recon_supcon_mtl": ("y_preds_AC_cnn_gru_dual_cosine_recon_supcon_mtl_", "y_probs_AC_cnn_gru_dual_cosine_recon_supcon_mtl_", "classes_AC_cnn_gru_dual_cosine_recon_supcon_mtl_"),
    "cnn_gru_dual_attn_recon_supcon_mtl":   ("y_preds_AC_cnn_gru_dual_attn_recon_supcon_mtl_",   "y_probs_AC_cnn_gru_dual_attn_recon_supcon_mtl_",   "classes_AC_cnn_gru_dual_attn_recon_supcon_mtl_"),
    "cnn_lf_mtl":                    ("y_preds_AC_cnn_lf_mtl_",                     "y_probs_AC_cnn_lf_mtl_",                     "classes_AC_cnn_lf_mtl_"),
    "gru_lf_mtl":                    ("y_preds_AC_gru_lf_mtl_",                     "y_probs_AC_gru_lf_mtl_",                     "classes_AC_gru_lf_mtl_"),
    "trans_lf_mtl":                  ("y_preds_AC_trans_lf_mtl_",                   "y_probs_AC_trans_lf_mtl_",                   "classes_AC_trans_lf_mtl_"),
    "lstm_lf_mtl":                   ("y_preds_AC_lstm_lf_mtl_",                    "y_probs_AC_lstm_lf_mtl_",                    "classes_AC_lstm_lf_mtl_"),
    "cnn_lstm_dual_mtl":             ("y_preds_AC_cnn_lstm_dual_mtl_",              "y_probs_AC_cnn_lstm_dual_mtl_",              "classes_AC_cnn_lstm_dual_mtl_"),
    "cnn_gru_gate_mtl":              ("y_preds_AC_cnn_gru_gate_mtl_",               "y_probs_AC_cnn_gru_gate_mtl_",               "classes_AC_cnn_gru_gate_mtl_"),
    "cnn_gru_hadamard_mtl":          ("y_preds_AC_cnn_gru_hadamard_mtl_",           "y_probs_AC_cnn_gru_hadamard_mtl_",           "classes_AC_cnn_gru_hadamard_mtl_"),
    "cnn_gru_crossattn_mtl":         ("y_preds_AC_cnn_gru_crossattn_mtl_",          "y_probs_AC_cnn_gru_crossattn_mtl_",          "classes_AC_cnn_gru_crossattn_mtl_"),
    "cnn_gru_film_mtl":              ("y_preds_AC_cnn_gru_film_mtl_",               "y_probs_AC_cnn_gru_film_mtl_",               "classes_AC_cnn_gru_film_mtl_"),
    "cnn_trans_gate_mtl":            ("y_preds_AC_cnn_trans_gate_mtl_",             "y_probs_AC_cnn_trans_gate_mtl_",             "classes_AC_cnn_trans_gate_mtl_"),
    "cnn_trans_hadamard_mtl":        ("y_preds_AC_cnn_trans_hadamard_mtl_",         "y_probs_AC_cnn_trans_hadamard_mtl_",         "classes_AC_cnn_trans_hadamard_mtl_"),
    "cnn_trans_crossattn_mtl":       ("y_preds_AC_cnn_trans_crossattn_mtl_",        "y_probs_AC_cnn_trans_crossattn_mtl_",        "classes_AC_cnn_trans_crossattn_mtl_"),
    "cnn_trans_film_mtl":            ("y_preds_AC_cnn_trans_film_mtl_",             "y_probs_AC_cnn_trans_film_mtl_",             "classes_AC_cnn_trans_film_mtl_"),
    # SupCon ST models
    "cnn_supcon":               ("y_preds_AC_cnn_supcon_",               "y_probs_AC_cnn_supcon_",               "classes_AC_cnn_supcon_"),
    "gru_supcon":               ("y_preds_AC_gru_supcon_",               "y_probs_AC_gru_supcon_",               "classes_AC_gru_supcon_"),
    "transformer_supcon":       ("y_preds_AC_trans_supcon_",             "y_probs_AC_trans_supcon_",             "classes_AC_trans_supcon_"),
    "cnn_gru_dual_supcon":      ("y_preds_AC_cnn_gru_dual_supcon_",      "y_probs_AC_cnn_gru_dual_supcon_",      "classes_AC_cnn_gru_dual_supcon_"),
    "cnn_trans_dual_supcon":    ("y_preds_AC_cnn_trans_dual_supcon_",    "y_probs_AC_cnn_trans_dual_supcon_",    "classes_AC_cnn_trans_dual_supcon_"),
    # SupCon MTL models
    "cnn_supcon_mtl":           ("y_preds_AC_cnn_supcon_mtl_",           "y_probs_AC_cnn_supcon_mtl_",           "classes_AC_cnn_supcon_mtl_"),
    "gru_supcon_mtl":           ("y_preds_AC_gru_supcon_mtl_",           "y_probs_AC_gru_supcon_mtl_",           "classes_AC_gru_supcon_mtl_"),
    "transformer_supcon_mtl":   ("y_preds_AC_trans_supcon_mtl_",         "y_probs_AC_trans_supcon_mtl_",         "classes_AC_trans_supcon_mtl_"),
    "cnn_gru_dual_supcon_mtl":  ("y_preds_AC_cnn_gru_dual_supcon_mtl_",  "y_probs_AC_cnn_gru_dual_supcon_mtl_",  "classes_AC_cnn_gru_dual_supcon_mtl_"),
    "cnn_trans_dual_supcon_mtl":("y_preds_AC_cnn_trans_dual_supcon_mtl_","y_probs_AC_cnn_trans_dual_supcon_mtl_","classes_AC_cnn_trans_dual_supcon_mtl_"),
    # Branch SupCon v2 ST (2 heads: CNN + seq branch)
    "cnn_gru_dual_supcon2":      ("y_preds_AC_cnn_gru_dual_supcon2_",      "y_probs_AC_cnn_gru_dual_supcon2_",      "classes_AC_cnn_gru_dual_supcon2_"),
    "cnn_trans_dual_supcon2":    ("y_preds_AC_cnn_trans_dual_supcon2_",    "y_probs_AC_cnn_trans_dual_supcon2_",    "classes_AC_cnn_trans_dual_supcon2_"),
    "cnn_gru_dual_cosine_recon_supcon2":     ("y_preds_AC_cnn_gru_dual_cosine_recon_supcon2_",     "y_probs_AC_cnn_gru_dual_cosine_recon_supcon2_",     "classes_AC_cnn_gru_dual_cosine_recon_supcon2_"),
    "cnn_gru_dual_attn_recon_supcon2":       ("y_preds_AC_cnn_gru_dual_attn_recon_supcon2_",       "y_probs_AC_cnn_gru_dual_attn_recon_supcon2_",       "classes_AC_cnn_gru_dual_attn_recon_supcon2_"),
    # Branch SupCon v3 ST (3 heads: CNN + seq + fused)
    "cnn_gru_dual_supcon3":      ("y_preds_AC_cnn_gru_dual_supcon3_",      "y_probs_AC_cnn_gru_dual_supcon3_",      "classes_AC_cnn_gru_dual_supcon3_"),
    "cnn_trans_dual_supcon3":    ("y_preds_AC_cnn_trans_dual_supcon3_",    "y_probs_AC_cnn_trans_dual_supcon3_",    "classes_AC_cnn_trans_dual_supcon3_"),
    "cnn_gru_dual_cosine_recon_supcon3":     ("y_preds_AC_cnn_gru_dual_cosine_recon_supcon3_",     "y_probs_AC_cnn_gru_dual_cosine_recon_supcon3_",     "classes_AC_cnn_gru_dual_cosine_recon_supcon3_"),
    "cnn_gru_dual_attn_recon_supcon3":       ("y_preds_AC_cnn_gru_dual_attn_recon_supcon3_",       "y_probs_AC_cnn_gru_dual_attn_recon_supcon3_",       "classes_AC_cnn_gru_dual_attn_recon_supcon3_"),
    # Branch SupCon v2 MTL
    "cnn_gru_dual_supcon2_mtl":  ("y_preds_AC_cnn_gru_dual_supcon2_mtl_",  "y_probs_AC_cnn_gru_dual_supcon2_mtl_",  "classes_AC_cnn_gru_dual_supcon2_mtl_"),
    "cnn_trans_dual_supcon2_mtl":("y_preds_AC_cnn_trans_dual_supcon2_mtl_","y_probs_AC_cnn_trans_dual_supcon2_mtl_","classes_AC_cnn_trans_dual_supcon2_mtl_"),
    "cnn_gru_dual_cosine_recon_supcon2_mtl": ("y_preds_AC_cnn_gru_dual_cosine_recon_supcon2_mtl_", "y_probs_AC_cnn_gru_dual_cosine_recon_supcon2_mtl_", "classes_AC_cnn_gru_dual_cosine_recon_supcon2_mtl_"),
    "cnn_gru_dual_attn_recon_supcon2_mtl":   ("y_preds_AC_cnn_gru_dual_attn_recon_supcon2_mtl_",   "y_probs_AC_cnn_gru_dual_attn_recon_supcon2_mtl_",   "classes_AC_cnn_gru_dual_attn_recon_supcon2_mtl_"),
    # Branch SupCon v3 MTL
    "cnn_gru_dual_supcon3_mtl":  ("y_preds_AC_cnn_gru_dual_supcon3_mtl_",  "y_probs_AC_cnn_gru_dual_supcon3_mtl_",  "classes_AC_cnn_gru_dual_supcon3_mtl_"),
    "cnn_trans_dual_supcon3_mtl":("y_preds_AC_cnn_trans_dual_supcon3_mtl_","y_probs_AC_cnn_trans_dual_supcon3_mtl_","classes_AC_cnn_trans_dual_supcon3_mtl_"),
    "cnn_gru_dual_cosine_recon_supcon3_mtl": ("y_preds_AC_cnn_gru_dual_cosine_recon_supcon3_mtl_", "y_probs_AC_cnn_gru_dual_cosine_recon_supcon3_mtl_", "classes_AC_cnn_gru_dual_cosine_recon_supcon3_mtl_"),
    "cnn_gru_dual_attn_recon_supcon3_mtl":   ("y_preds_AC_cnn_gru_dual_attn_recon_supcon3_mtl_",   "y_probs_AC_cnn_gru_dual_attn_recon_supcon3_mtl_",   "classes_AC_cnn_gru_dual_attn_recon_supcon3_mtl_"),
    # CL (Phase-Decoupled) MTL variants — base + supcon v1/v2/v3
    "cnn_gru_dual_cl_mtl":           ("y_preds_AC_cnn_gru_dual_cl_mtl_",           "y_probs_AC_cnn_gru_dual_cl_mtl_",           "classes_AC_cnn_gru_dual_cl_mtl_"),
    "cnn_trans_dual_cl_mtl":         ("y_preds_AC_cnn_trans_dual_cl_mtl_",         "y_probs_AC_cnn_trans_dual_cl_mtl_",         "classes_AC_cnn_trans_dual_cl_mtl_"),
    "cnn_gru_dual_cl_supcon_mtl":    ("y_preds_AC_cnn_gru_dual_cl_supcon_mtl_",    "y_probs_AC_cnn_gru_dual_cl_supcon_mtl_",    "classes_AC_cnn_gru_dual_cl_supcon_mtl_"),
    "cnn_trans_dual_cl_supcon_mtl":  ("y_preds_AC_cnn_trans_dual_cl_supcon_mtl_",  "y_probs_AC_cnn_trans_dual_cl_supcon_mtl_",  "classes_AC_cnn_trans_dual_cl_supcon_mtl_"),
    "cnn_gru_dual_cl_supcon2_mtl":   ("y_preds_AC_cnn_gru_dual_cl_supcon2_mtl_",   "y_probs_AC_cnn_gru_dual_cl_supcon2_mtl_",   "classes_AC_cnn_gru_dual_cl_supcon2_mtl_"),
    "cnn_trans_dual_cl_supcon2_mtl": ("y_preds_AC_cnn_trans_dual_cl_supcon2_mtl_", "y_probs_AC_cnn_trans_dual_cl_supcon2_mtl_", "classes_AC_cnn_trans_dual_cl_supcon2_mtl_"),
    "cnn_gru_dual_cl_supcon3_mtl":   ("y_preds_AC_cnn_gru_dual_cl_supcon3_mtl_",   "y_probs_AC_cnn_gru_dual_cl_supcon3_mtl_",   "classes_AC_cnn_gru_dual_cl_supcon3_mtl_"),
    "cnn_trans_dual_cl_supcon3_mtl": ("y_preds_AC_cnn_trans_dual_cl_supcon3_mtl_", "y_probs_AC_cnn_trans_dual_cl_supcon3_mtl_", "classes_AC_cnn_trans_dual_cl_supcon3_mtl_"),
    # RCFD (Regression-Conditioned Feature Dual) — base + supcon v1/v2/v3
    "cnn_rcfd_cgd":   ("y_preds_AC_cnn_rcfd_cgd_",   "y_probs_AC_cnn_rcfd_cgd_",   "classes_AC_cnn_rcfd_cgd_"),
    "cnn_rcfd_ctd":   ("y_preds_AC_cnn_rcfd_ctd_",   "y_probs_AC_cnn_rcfd_ctd_",   "classes_AC_cnn_rcfd_ctd_"),
    "gru_rcfd_cgd":   ("y_preds_AC_gru_rcfd_cgd_",   "y_probs_AC_gru_rcfd_cgd_",   "classes_AC_gru_rcfd_cgd_"),
    "gru_rcfd_ctd":   ("y_preds_AC_gru_rcfd_ctd_",   "y_probs_AC_gru_rcfd_ctd_",   "classes_AC_gru_rcfd_ctd_"),
    "trans_rcfd_cgd": ("y_preds_AC_trans_rcfd_cgd_", "y_probs_AC_trans_rcfd_cgd_", "classes_AC_trans_rcfd_cgd_"),
    "trans_rcfd_ctd": ("y_preds_AC_trans_rcfd_ctd_", "y_probs_AC_trans_rcfd_ctd_", "classes_AC_trans_rcfd_ctd_"),
    "cnn_rcfd_attn_recon":   ("y_preds_AC_cnn_rcfd_attn_recon_",   "y_probs_AC_cnn_rcfd_attn_recon_",   "classes_AC_cnn_rcfd_attn_recon_"),
    "gru_rcfd_attn_recon":   ("y_preds_AC_gru_rcfd_attn_recon_",   "y_probs_AC_gru_rcfd_attn_recon_",   "classes_AC_gru_rcfd_attn_recon_"),
    "trans_rcfd_attn_recon": ("y_preds_AC_trans_rcfd_attn_recon_", "y_probs_AC_trans_rcfd_attn_recon_", "classes_AC_trans_rcfd_attn_recon_"),
    "cnn_rcfd_cgd_supcon_mtl":   ("y_preds_AC_cnn_rcfd_cgd_supcon_mtl_",   "y_probs_AC_cnn_rcfd_cgd_supcon_mtl_",   "classes_AC_cnn_rcfd_cgd_supcon_mtl_"),
    "cnn_rcfd_ctd_supcon_mtl":   ("y_preds_AC_cnn_rcfd_ctd_supcon_mtl_",   "y_probs_AC_cnn_rcfd_ctd_supcon_mtl_",   "classes_AC_cnn_rcfd_ctd_supcon_mtl_"),
    "gru_rcfd_cgd_supcon_mtl":   ("y_preds_AC_gru_rcfd_cgd_supcon_mtl_",   "y_probs_AC_gru_rcfd_cgd_supcon_mtl_",   "classes_AC_gru_rcfd_cgd_supcon_mtl_"),
    "gru_rcfd_ctd_supcon_mtl":   ("y_preds_AC_gru_rcfd_ctd_supcon_mtl_",   "y_probs_AC_gru_rcfd_ctd_supcon_mtl_",   "classes_AC_gru_rcfd_ctd_supcon_mtl_"),
    "trans_rcfd_cgd_supcon_mtl": ("y_preds_AC_trans_rcfd_cgd_supcon_mtl_", "y_probs_AC_trans_rcfd_cgd_supcon_mtl_", "classes_AC_trans_rcfd_cgd_supcon_mtl_"),
    "trans_rcfd_ctd_supcon_mtl": ("y_preds_AC_trans_rcfd_ctd_supcon_mtl_", "y_probs_AC_trans_rcfd_ctd_supcon_mtl_", "classes_AC_trans_rcfd_ctd_supcon_mtl_"),
    "cnn_rcfd_cgd_supcon2_mtl":   ("y_preds_AC_cnn_rcfd_cgd_supcon2_mtl_",   "y_probs_AC_cnn_rcfd_cgd_supcon2_mtl_",   "classes_AC_cnn_rcfd_cgd_supcon2_mtl_"),
    "cnn_rcfd_ctd_supcon2_mtl":   ("y_preds_AC_cnn_rcfd_ctd_supcon2_mtl_",   "y_probs_AC_cnn_rcfd_ctd_supcon2_mtl_",   "classes_AC_cnn_rcfd_ctd_supcon2_mtl_"),
    "gru_rcfd_cgd_supcon2_mtl":   ("y_preds_AC_gru_rcfd_cgd_supcon2_mtl_",   "y_probs_AC_gru_rcfd_cgd_supcon2_mtl_",   "classes_AC_gru_rcfd_cgd_supcon2_mtl_"),
    "gru_rcfd_ctd_supcon2_mtl":   ("y_preds_AC_gru_rcfd_ctd_supcon2_mtl_",   "y_probs_AC_gru_rcfd_ctd_supcon2_mtl_",   "classes_AC_gru_rcfd_ctd_supcon2_mtl_"),
    "trans_rcfd_cgd_supcon2_mtl": ("y_preds_AC_trans_rcfd_cgd_supcon2_mtl_", "y_probs_AC_trans_rcfd_cgd_supcon2_mtl_", "classes_AC_trans_rcfd_cgd_supcon2_mtl_"),
    "trans_rcfd_ctd_supcon2_mtl": ("y_preds_AC_trans_rcfd_ctd_supcon2_mtl_", "y_probs_AC_trans_rcfd_ctd_supcon2_mtl_", "classes_AC_trans_rcfd_ctd_supcon2_mtl_"),
    "cnn_rcfd_cgd_supcon3_mtl":   ("y_preds_AC_cnn_rcfd_cgd_supcon3_mtl_",   "y_probs_AC_cnn_rcfd_cgd_supcon3_mtl_",   "classes_AC_cnn_rcfd_cgd_supcon3_mtl_"),
    "cnn_rcfd_ctd_supcon3_mtl":   ("y_preds_AC_cnn_rcfd_ctd_supcon3_mtl_",   "y_probs_AC_cnn_rcfd_ctd_supcon3_mtl_",   "classes_AC_cnn_rcfd_ctd_supcon3_mtl_"),
    "gru_rcfd_cgd_supcon3_mtl":   ("y_preds_AC_gru_rcfd_cgd_supcon3_mtl_",   "y_probs_AC_gru_rcfd_cgd_supcon3_mtl_",   "classes_AC_gru_rcfd_cgd_supcon3_mtl_"),
    "gru_rcfd_ctd_supcon3_mtl":   ("y_preds_AC_gru_rcfd_ctd_supcon3_mtl_",   "y_probs_AC_gru_rcfd_ctd_supcon3_mtl_",   "classes_AC_gru_rcfd_ctd_supcon3_mtl_"),
    "trans_rcfd_cgd_supcon3_mtl": ("y_preds_AC_trans_rcfd_cgd_supcon3_mtl_", "y_probs_AC_trans_rcfd_cgd_supcon3_mtl_", "classes_AC_trans_rcfd_cgd_supcon3_mtl_"),
    "trans_rcfd_ctd_supcon3_mtl": ("y_preds_AC_trans_rcfd_ctd_supcon3_mtl_", "y_probs_AC_trans_rcfd_ctd_supcon3_mtl_", "classes_AC_trans_rcfd_ctd_supcon3_mtl_"),
    # Label Consolidation (LC) — pure ST, combined label+conc target; 4 SC variants × 3 architectures
    "cnn_gru_dual_lc":                  ("y_preds_AC_cnn_gru_dual_lc_",                  "y_probs_AC_cnn_gru_dual_lc_",                  "classes_AC_cnn_gru_dual_lc_"),
    "cnn_gru_dual_cosine_recon_lc":     ("y_preds_AC_cnn_gru_dual_cosine_recon_lc_",     "y_probs_AC_cnn_gru_dual_cosine_recon_lc_",     "classes_AC_cnn_gru_dual_cosine_recon_lc_"),
    "cnn_gru_dual_attn_recon_lc":       ("y_preds_AC_cnn_gru_dual_attn_recon_lc_",       "y_probs_AC_cnn_gru_dual_attn_recon_lc_",       "classes_AC_cnn_gru_dual_attn_recon_lc_"),
    "cnn_gru_dual_supcon_lc":           ("y_preds_AC_cnn_gru_dual_supcon_lc_",           "y_probs_AC_cnn_gru_dual_supcon_lc_",           "classes_AC_cnn_gru_dual_supcon_lc_"),
    "cnn_gru_dual_cosine_recon_supcon_lc": ("y_preds_AC_cnn_gru_dual_cosine_recon_supcon_lc_", "y_probs_AC_cnn_gru_dual_cosine_recon_supcon_lc_", "classes_AC_cnn_gru_dual_cosine_recon_supcon_lc_"),
    "cnn_gru_dual_attn_recon_supcon_lc":   ("y_preds_AC_cnn_gru_dual_attn_recon_supcon_lc_",   "y_probs_AC_cnn_gru_dual_attn_recon_supcon_lc_",   "classes_AC_cnn_gru_dual_attn_recon_supcon_lc_"),
    "cnn_gru_dual_supcon2_lc":          ("y_preds_AC_cnn_gru_dual_supcon2_lc_",          "y_probs_AC_cnn_gru_dual_supcon2_lc_",          "classes_AC_cnn_gru_dual_supcon2_lc_"),
    "cnn_gru_dual_cosine_recon_supcon2_lc": ("y_preds_AC_cnn_gru_dual_cosine_recon_supcon2_lc_", "y_probs_AC_cnn_gru_dual_cosine_recon_supcon2_lc_", "classes_AC_cnn_gru_dual_cosine_recon_supcon2_lc_"),
    "cnn_gru_dual_attn_recon_supcon2_lc":   ("y_preds_AC_cnn_gru_dual_attn_recon_supcon2_lc_",   "y_probs_AC_cnn_gru_dual_attn_recon_supcon2_lc_",   "classes_AC_cnn_gru_dual_attn_recon_supcon2_lc_"),
    "cnn_gru_dual_supcon3_lc":          ("y_preds_AC_cnn_gru_dual_supcon3_lc_",          "y_probs_AC_cnn_gru_dual_supcon3_lc_",          "classes_AC_cnn_gru_dual_supcon3_lc_"),
    "cnn_gru_dual_cosine_recon_supcon3_lc": ("y_preds_AC_cnn_gru_dual_cosine_recon_supcon3_lc_", "y_probs_AC_cnn_gru_dual_cosine_recon_supcon3_lc_", "classes_AC_cnn_gru_dual_cosine_recon_supcon3_lc_"),
    "cnn_gru_dual_attn_recon_supcon3_lc":   ("y_preds_AC_cnn_gru_dual_attn_recon_supcon3_lc_",   "y_probs_AC_cnn_gru_dual_attn_recon_supcon3_lc_",   "classes_AC_cnn_gru_dual_attn_recon_supcon3_lc_"),
    # Staged SupCon ST — SC1
    "cnn_gru_dual_supcon_staged":               ("y_preds_AC_cnn_gru_dual_supcon_staged_",               "y_probs_AC_cnn_gru_dual_supcon_staged_",               "classes_AC_cnn_gru_dual_supcon_staged_"),
    "cnn_gru_dual_cosine_recon_supcon_staged":  ("y_preds_AC_cnn_gru_dual_cosine_recon_supcon_staged_",  "y_probs_AC_cnn_gru_dual_cosine_recon_supcon_staged_",  "classes_AC_cnn_gru_dual_cosine_recon_supcon_staged_"),
    "cnn_gru_dual_attn_recon_supcon_staged":    ("y_preds_AC_cnn_gru_dual_attn_recon_supcon_staged_",    "y_probs_AC_cnn_gru_dual_attn_recon_supcon_staged_",    "classes_AC_cnn_gru_dual_attn_recon_supcon_staged_"),
    # SC2 staged
    "cnn_gru_dual_supcon2_staged":              ("y_preds_AC_cnn_gru_dual_supcon2_staged_",              "y_probs_AC_cnn_gru_dual_supcon2_staged_",              "classes_AC_cnn_gru_dual_supcon2_staged_"),
    "cnn_gru_dual_cosine_recon_supcon2_staged": ("y_preds_AC_cnn_gru_dual_cosine_recon_supcon2_staged_", "y_probs_AC_cnn_gru_dual_cosine_recon_supcon2_staged_", "classes_AC_cnn_gru_dual_cosine_recon_supcon2_staged_"),
    "cnn_gru_dual_attn_recon_supcon2_staged":   ("y_preds_AC_cnn_gru_dual_attn_recon_supcon2_staged_",   "y_probs_AC_cnn_gru_dual_attn_recon_supcon2_staged_",   "classes_AC_cnn_gru_dual_attn_recon_supcon2_staged_"),
    # SC3 staged
    "cnn_gru_dual_supcon3_staged":              ("y_preds_AC_cnn_gru_dual_supcon3_staged_",              "y_probs_AC_cnn_gru_dual_supcon3_staged_",              "classes_AC_cnn_gru_dual_supcon3_staged_"),
    "cnn_gru_dual_cosine_recon_supcon3_staged": ("y_preds_AC_cnn_gru_dual_cosine_recon_supcon3_staged_", "y_probs_AC_cnn_gru_dual_cosine_recon_supcon3_staged_", "classes_AC_cnn_gru_dual_cosine_recon_supcon3_staged_"),
    "cnn_gru_dual_attn_recon_supcon3_staged":   ("y_preds_AC_cnn_gru_dual_attn_recon_supcon3_staged_",   "y_probs_AC_cnn_gru_dual_attn_recon_supcon3_staged_",   "classes_AC_cnn_gru_dual_attn_recon_supcon3_staged_"),
    # CCGD arch-poc ST (from 03g, available in 03 single-task mode)
    "ccgd_arch_poc_st":     ("y_preds_AC_ccgd_arch_poc_st_",     "y_probs_AC_ccgd_arch_poc_st_",     "classes_AC_ccgd_arch_poc_st_"),
    "ccgd_arch_poc_st_sc1": ("y_preds_AC_ccgd_arch_poc_st_sc1_", "y_probs_AC_ccgd_arch_poc_st_sc1_", "classes_AC_ccgd_arch_poc_st_sc1_"),
    "ccgd_arch_poc_st_sc2": ("y_preds_AC_ccgd_arch_poc_st_sc2_", "y_probs_AC_ccgd_arch_poc_st_sc2_", "classes_AC_ccgd_arch_poc_st_sc2_"),
    "ccgd_arch_poc_st_sc3": ("y_preds_AC_ccgd_arch_poc_st_sc3_", "y_probs_AC_ccgd_arch_poc_st_sc3_", "classes_AC_ccgd_arch_poc_st_sc3_"),
    # Domain-adversarial (DANN)
    "cnn_gru_dual_dann":                  ("y_preds_AC_cnn_gru_dual_dann_",                  "y_probs_AC_cnn_gru_dual_dann_",                  "classes_AC_cnn_gru_dual_dann_"),
    "cnn_gru_dual_attn_recon_dann":       ("y_preds_AC_cnn_gru_dual_attn_recon_dann_",       "y_probs_AC_cnn_gru_dual_attn_recon_dann_",       "classes_AC_cnn_gru_dual_attn_recon_dann_"),
    "cnn_gru_dual_supcon3_dann":          ("y_preds_AC_cnn_gru_dual_supcon3_dann_",          "y_probs_AC_cnn_gru_dual_supcon3_dann_",          "classes_AC_cnn_gru_dual_supcon3_dann_"),
    "cnn_gru_dual_attn_recon_supcon3_dann": ("y_preds_AC_cnn_gru_dual_attn_recon_supcon3_dann_", "y_probs_AC_cnn_gru_dual_attn_recon_supcon3_dann_", "classes_AC_cnn_gru_dual_attn_recon_supcon3_dann_"),
    # Deep CORAL (non-adversarial domain alignment)
    "cnn_gru_dual_coral":                  ("y_preds_AC_cnn_gru_dual_coral_",                  "y_probs_AC_cnn_gru_dual_coral_",                  "classes_AC_cnn_gru_dual_coral_"),
    "cnn_gru_dual_attn_recon_coral":       ("y_preds_AC_cnn_gru_dual_attn_recon_coral_",       "y_probs_AC_cnn_gru_dual_attn_recon_coral_",       "classes_AC_cnn_gru_dual_attn_recon_coral_"),
    "cnn_gru_dual_supcon3_coral":          ("y_preds_AC_cnn_gru_dual_supcon3_coral_",          "y_probs_AC_cnn_gru_dual_supcon3_coral_",          "classes_AC_cnn_gru_dual_supcon3_coral_"),
    "cnn_gru_dual_attn_recon_supcon3_coral": ("y_preds_AC_cnn_gru_dual_attn_recon_supcon3_coral_", "y_probs_AC_cnn_gru_dual_attn_recon_supcon3_coral_", "classes_AC_cnn_gru_dual_attn_recon_supcon3_coral_"),
}
# _inc variants: same models with inception smoothing front-end. Cache keys get _inc_ suffix
# so results coexist with the baseline in the same joblib without overwriting each other.
# rf/knn/ffi/gnn_* are non-Keras; attn_recon has incompatible input shape; MTL/SupCon models never
# use inception smoothing.
_MTL_MODEL_KEYS = {
    "cnn_mtl", "lstm_mtl", "gru_mtl", "rnn_mtl", "transformer_mtl",
    "cnn_gru_dual_mtl", "cnn_trans_dual_mtl", "cnn_lstm_dual_mtl",
    "cnn_gru_dual_cosine_recon_mtl", "cnn_gru_dual_attn_recon_mtl",
    "cnn_lf_mtl", "gru_lf_mtl", "trans_lf_mtl", "lstm_lf_mtl",
    "cnn_gru_gate_mtl", "cnn_gru_hadamard_mtl", "cnn_gru_crossattn_mtl", "cnn_gru_film_mtl",
    "cnn_trans_gate_mtl", "cnn_trans_hadamard_mtl", "cnn_trans_crossattn_mtl", "cnn_trans_film_mtl",
}
_SUPCON_MODEL_KEYS = {
    "cnn_supcon", "gru_supcon", "transformer_supcon",
    "cnn_gru_dual_supcon", "cnn_trans_dual_supcon",
    "cnn_gru_dual_cosine_recon_supcon", "cnn_gru_dual_attn_recon_supcon",
    "cnn_supcon_mtl", "gru_supcon_mtl", "transformer_supcon_mtl",
    "cnn_gru_dual_supcon_mtl", "cnn_trans_dual_supcon_mtl",
    "cnn_gru_dual_cosine_recon_supcon_mtl", "cnn_gru_dual_attn_recon_supcon_mtl",
}
# Branch SupCon keys kept separate from _SUPCON_MODEL_KEYS so --supcon force_rerun
# does not accidentally clear branch supcon results and vice versa.
_BRANCH_SUPCON_MODEL_KEYS = {
    "cnn_gru_dual_supcon2",  "cnn_trans_dual_supcon2",
    "cnn_gru_dual_cosine_recon_supcon2", "cnn_gru_dual_attn_recon_supcon2",
    "cnn_gru_dual_supcon3",  "cnn_trans_dual_supcon3",
    "cnn_gru_dual_cosine_recon_supcon3", "cnn_gru_dual_attn_recon_supcon3",
    "cnn_gru_dual_supcon2_mtl", "cnn_trans_dual_supcon2_mtl",
    "cnn_gru_dual_cosine_recon_supcon2_mtl", "cnn_gru_dual_attn_recon_supcon2_mtl",
    "cnn_gru_dual_supcon3_mtl", "cnn_trans_dual_supcon3_mtl",
    "cnn_gru_dual_cosine_recon_supcon3_mtl", "cnn_gru_dual_attn_recon_supcon3_mtl",
}
_CL_MTL_MODEL_KEYS = {
    "cnn_gru_dual_cl_mtl", "cnn_trans_dual_cl_mtl",
    "cnn_gru_dual_cl_supcon_mtl", "cnn_trans_dual_cl_supcon_mtl",
    "cnn_gru_dual_cl_supcon2_mtl", "cnn_trans_dual_cl_supcon2_mtl",
    "cnn_gru_dual_cl_supcon3_mtl", "cnn_trans_dual_cl_supcon3_mtl",
}
_RCFD_MODEL_KEYS = {
    "cnn_rcfd_cgd", "cnn_rcfd_ctd", "gru_rcfd_cgd", "gru_rcfd_ctd",
    "trans_rcfd_cgd", "trans_rcfd_ctd",
    "cnn_rcfd_attn_recon", "gru_rcfd_attn_recon", "trans_rcfd_attn_recon",
    "cnn_rcfd_cgd_supcon_mtl", "cnn_rcfd_ctd_supcon_mtl",
    "gru_rcfd_cgd_supcon_mtl", "gru_rcfd_ctd_supcon_mtl",
    "trans_rcfd_cgd_supcon_mtl", "trans_rcfd_ctd_supcon_mtl",
    "cnn_rcfd_cgd_supcon2_mtl", "cnn_rcfd_ctd_supcon2_mtl",
    "gru_rcfd_cgd_supcon2_mtl", "gru_rcfd_ctd_supcon2_mtl",
    "trans_rcfd_cgd_supcon2_mtl", "trans_rcfd_ctd_supcon2_mtl",
    "cnn_rcfd_cgd_supcon3_mtl", "cnn_rcfd_ctd_supcon3_mtl",
    "gru_rcfd_cgd_supcon3_mtl", "gru_rcfd_ctd_supcon3_mtl",
    "trans_rcfd_cgd_supcon3_mtl", "trans_rcfd_ctd_supcon3_mtl",
}
_LC_MODEL_KEYS = {
    "cnn_gru_dual_lc", "cnn_gru_dual_cosine_recon_lc", "cnn_gru_dual_attn_recon_lc",
    "cnn_gru_dual_supcon_lc", "cnn_gru_dual_cosine_recon_supcon_lc", "cnn_gru_dual_attn_recon_supcon_lc",
    "cnn_gru_dual_supcon2_lc", "cnn_gru_dual_cosine_recon_supcon2_lc", "cnn_gru_dual_attn_recon_supcon2_lc",
    "cnn_gru_dual_supcon3_lc", "cnn_gru_dual_cosine_recon_supcon3_lc", "cnn_gru_dual_attn_recon_supcon3_lc",
}
_STAGED_SUPCON_MODEL_KEYS = {
    "cnn_gru_dual_supcon_staged", "cnn_gru_dual_cosine_recon_supcon_staged", "cnn_gru_dual_attn_recon_supcon_staged",
    "cnn_gru_dual_supcon2_staged", "cnn_gru_dual_cosine_recon_supcon2_staged", "cnn_gru_dual_attn_recon_supcon2_staged",
    "cnn_gru_dual_supcon3_staged", "cnn_gru_dual_cosine_recon_supcon3_staged", "cnn_gru_dual_attn_recon_supcon3_staged",
}
_CCGD_ST_KEYS = {"ccgd_arch_poc_st", "ccgd_arch_poc_st_sc1", "ccgd_arch_poc_st_sc2", "ccgd_arch_poc_st_sc3"}
_DANN_MODEL_KEYS = {
    "cnn_gru_dual_dann", "cnn_gru_dual_attn_recon_dann",
    "cnn_gru_dual_supcon3_dann", "cnn_gru_dual_attn_recon_supcon3_dann",
}
_CORAL_MODEL_KEYS = {
    "cnn_gru_dual_coral", "cnn_gru_dual_attn_recon_coral",
    "cnn_gru_dual_supcon3_coral", "cnn_gru_dual_attn_recon_supcon3_coral",
}
_NO_INC = ({"rf", "knn", "ffi", "gnn_gat", "gnn_gcn", "cnn_gru_dual_attn_recon"}
           | _MTL_MODEL_KEYS | _SUPCON_MODEL_KEYS | _BRANCH_SUPCON_MODEL_KEYS
           | _CL_MTL_MODEL_KEYS | _RCFD_MODEL_KEYS | _LC_MODEL_KEYS
           | _STAGED_SUPCON_MODEL_KEYS | _CCGD_ST_KEYS | _DANN_MODEL_KEYS
           | _CORAL_MODEL_KEYS)
MODEL_KEY_MAP.update({
    f"{m}_inc": tuple(k.rstrip("_") + "_inc_" for k in keys)
    for m, keys in list(MODEL_KEY_MAP.items()) if m not in _NO_INC
})

MODEL_PRINT_MAP = {
    "cnn": "CNN (ACA)", "lstm": "LSTM (ACA)", "gru": "GRU (ACA)",
    "rnn": "RNN (ACA)", "transformer": "Trans (ACA)", "rf": "RF (ACA)",
    "knn": "KNN (ACA)", "ffi": "LR (FFI)",
    "cnn_lf": "CNN LF", "lstm_lf": "LSTM LF", "trans_lf": "Trans LF", "gru_lf": "GRU LF",
    "cnn_gru_dual": "CNN+GRU Dual", "cnn_trans_dual": "CNN+Tr Dual", "cnn_lstm_dual": "CNN+LSTM Dual",
    "lstm_ae_clf": "LSTM-AE Clf",
    "cnn_gru_gate": "CNN+GRU Gate", "cnn_gru_hadamard": "CNN+GRU Hadamard",
    "cnn_gru_crossattn": "CNN+GRU CoAttn", "cnn_gru_film": "CNN+GRU FiLM",
    "cnn_trans_gate": "CNN+Tr Gate", "cnn_trans_hadamard": "CNN+Tr Hadamard",
    "cnn_trans_crossattn": "CNN+Tr CoAttn", "cnn_trans_film": "CNN+Tr FiLM",
    "gnn_gat": "GNN (GAT)", "gnn_gcn": "GNN (GCN)",
    "cnn_gru_dual_cosine_recon": "CNN+GRU CosRecon", "cnn_gru_dual_attn_recon": "CNN+GRU AttnRecon",
    # MTL
    "cnn_mtl": "CNN MTL", "lstm_mtl": "LSTM MTL", "gru_mtl": "GRU MTL",
    "rnn_mtl": "RNN MTL", "transformer_mtl": "Trans MTL",
    "cnn_gru_dual_mtl": "CNN+GRU Dual MTL", "cnn_trans_dual_mtl": "CNN+Tr Dual MTL",
    "cnn_lstm_dual_mtl": "CNN+LSTM Dual MTL",
    "cnn_gru_dual_cosine_recon_mtl": "CNN+GRU CosRecon MTL",
    "cnn_gru_dual_attn_recon_mtl": "CNN+GRU AttnRecon MTL",
    "cnn_lf_mtl": "CNN LF MTL", "gru_lf_mtl": "GRU LF MTL",
    "trans_lf_mtl": "Trans LF MTL", "lstm_lf_mtl": "LSTM LF MTL",
    "cnn_gru_gate_mtl": "CNN+GRU Gate MTL", "cnn_gru_hadamard_mtl": "CNN+GRU Hadamard MTL",
    "cnn_gru_crossattn_mtl": "CNN+GRU CoAttn MTL", "cnn_gru_film_mtl": "CNN+GRU FiLM MTL",
    "cnn_trans_gate_mtl": "CNN+Tr Gate MTL", "cnn_trans_hadamard_mtl": "CNN+Tr Hadamard MTL",
    "cnn_trans_crossattn_mtl": "CNN+Tr CoAttn MTL", "cnn_trans_film_mtl": "CNN+Tr FiLM MTL",
    # SupCon ST
    "cnn_supcon": "CNN SupCon", "gru_supcon": "GRU SupCon",
    "transformer_supcon": "Trans SupCon",
    "cnn_gru_dual_supcon": "CNN+GRU Dual SupCon", "cnn_trans_dual_supcon": "CNN+Tr Dual SupCon",
    # SupCon MTL
    "cnn_supcon_mtl": "CNN SupCon MTL", "gru_supcon_mtl": "GRU SupCon MTL",
    "transformer_supcon_mtl": "Trans SupCon MTL",
    "cnn_gru_dual_supcon_mtl": "CNN+GRU Dual SupCon MTL",
    "cnn_trans_dual_supcon_mtl": "CNN+Tr Dual SupCon MTL",
    # Branch SupCon v1 ST
    "cnn_gru_dual_cosine_recon_supcon":     "CNN+GRU CosRecon SC",
    "cnn_gru_dual_attn_recon_supcon":       "CNN+GRU AttnRecon SC",
    # Branch SupCon v1 MTL
    "cnn_gru_dual_cosine_recon_supcon_mtl": "CNN+GRU CosRecon SC MTL",
    "cnn_gru_dual_attn_recon_supcon_mtl":   "CNN+GRU AttnRecon SC MTL",
    # Branch SupCon v2 ST (2 heads: CNN + seq branch)
    "cnn_gru_dual_supcon2":  "CNN+GRU Dual SC2",  "cnn_trans_dual_supcon2":  "CNN+Tr Dual SC2",
    "cnn_gru_dual_cosine_recon_supcon2": "CNN+GRU CosRecon SC2",
    "cnn_gru_dual_attn_recon_supcon2":   "CNN+GRU AttnRecon SC2",
    # Branch SupCon v3 ST (3 heads: CNN + seq + fused)
    "cnn_gru_dual_supcon3":  "CNN+GRU Dual SC3",  "cnn_trans_dual_supcon3":  "CNN+Tr Dual SC3",
    "cnn_gru_dual_cosine_recon_supcon3": "CNN+GRU CosRecon SC3",
    "cnn_gru_dual_attn_recon_supcon3":   "CNN+GRU AttnRecon SC3",
    # Branch SupCon v2 MTL
    "cnn_gru_dual_supcon2_mtl": "CNN+GRU Dual SC2 MTL", "cnn_trans_dual_supcon2_mtl": "CNN+Tr Dual SC2 MTL",
    "cnn_gru_dual_cosine_recon_supcon2_mtl": "CNN+GRU CosRecon SC2 MTL",
    "cnn_gru_dual_attn_recon_supcon2_mtl":   "CNN+GRU AttnRecon SC2 MTL",
    # Branch SupCon v3 MTL
    "cnn_gru_dual_supcon3_mtl": "CNN+GRU Dual SC3 MTL", "cnn_trans_dual_supcon3_mtl": "CNN+Tr Dual SC3 MTL",
    "cnn_gru_dual_cosine_recon_supcon3_mtl": "CNN+GRU CosRecon SC3 MTL",
    "cnn_gru_dual_attn_recon_supcon3_mtl":   "CNN+GRU AttnRecon SC3 MTL",
    # CL MTL (curriculum learning)
    "cnn_gru_dual_cl_mtl":       "CNN+GRU Dual CL",      "cnn_trans_dual_cl_mtl":       "CNN+Tr Dual CL",
    "cnn_gru_dual_cl_supcon_mtl":"CNN+GRU Dual CL SC",   "cnn_trans_dual_cl_supcon_mtl":"CNN+Tr Dual CL SC",
    "cnn_gru_dual_cl_supcon2_mtl":"CNN+GRU Dual CL SC2", "cnn_trans_dual_cl_supcon2_mtl":"CNN+Tr Dual CL SC2",
    "cnn_gru_dual_cl_supcon3_mtl":"CNN+GRU Dual CL SC3", "cnn_trans_dual_cl_supcon3_mtl":"CNN+Tr Dual CL SC3",
    # RCFD (Regression-Conditioned Feature Dual)
    "cnn_rcfd_cgd":   "CNN RCFD CGD",    "cnn_rcfd_ctd":   "CNN RCFD CTD",
    "gru_rcfd_cgd":   "GRU RCFD CGD",    "gru_rcfd_ctd":   "GRU RCFD CTD",
    "trans_rcfd_cgd": "Trans RCFD CGD",  "trans_rcfd_ctd": "Trans RCFD CTD",
    "cnn_rcfd_attn_recon":   "CNN RCFD AttnRecon",
    "gru_rcfd_attn_recon":   "GRU RCFD AttnRecon",
    "trans_rcfd_attn_recon": "Trans RCFD AttnRecon",
    "cnn_rcfd_cgd_supcon_mtl":   "CNN RCFD CGD SC1",   "cnn_rcfd_ctd_supcon_mtl":   "CNN RCFD CTD SC1",
    "gru_rcfd_cgd_supcon_mtl":   "GRU RCFD CGD SC1",   "gru_rcfd_ctd_supcon_mtl":   "GRU RCFD CTD SC1",
    "trans_rcfd_cgd_supcon_mtl": "Trans RCFD CGD SC1", "trans_rcfd_ctd_supcon_mtl": "Trans RCFD CTD SC1",
    "cnn_rcfd_cgd_supcon2_mtl":   "CNN RCFD CGD SC2",   "cnn_rcfd_ctd_supcon2_mtl":   "CNN RCFD CTD SC2",
    "gru_rcfd_cgd_supcon2_mtl":   "GRU RCFD CGD SC2",   "gru_rcfd_ctd_supcon2_mtl":   "GRU RCFD CTD SC2",
    "trans_rcfd_cgd_supcon2_mtl": "Trans RCFD CGD SC2", "trans_rcfd_ctd_supcon2_mtl": "Trans RCFD CTD SC2",
    "cnn_rcfd_cgd_supcon3_mtl":   "CNN RCFD CGD SC3",   "cnn_rcfd_ctd_supcon3_mtl":   "CNN RCFD CTD SC3",
    "gru_rcfd_cgd_supcon3_mtl":   "GRU RCFD CGD SC3",   "gru_rcfd_ctd_supcon3_mtl":   "GRU RCFD CTD SC3",
    "trans_rcfd_cgd_supcon3_mtl": "Trans RCFD CGD SC3", "trans_rcfd_ctd_supcon3_mtl": "Trans RCFD CTD SC3",
    # Label Consolidation (LC) — pure ST, combined label+conc target
    "cnn_gru_dual_lc":                       "CNN+GRU LC",
    "cnn_gru_dual_cosine_recon_lc":          "CNN+GRU CosRecon LC",
    "cnn_gru_dual_attn_recon_lc":            "CNN+GRU AttnRecon LC",
    "cnn_gru_dual_supcon_lc":                "CNN+GRU LC SC1",
    "cnn_gru_dual_cosine_recon_supcon_lc":   "CNN+GRU CosRecon LC SC1",
    "cnn_gru_dual_attn_recon_supcon_lc":     "CNN+GRU AttnRecon LC SC1",
    "cnn_gru_dual_supcon2_lc":               "CNN+GRU LC SC2",
    "cnn_gru_dual_cosine_recon_supcon2_lc":  "CNN+GRU CosRecon LC SC2",
    "cnn_gru_dual_attn_recon_supcon2_lc":    "CNN+GRU AttnRecon LC SC2",
    "cnn_gru_dual_supcon3_lc":               "CNN+GRU LC SC3",
    "cnn_gru_dual_cosine_recon_supcon3_lc":  "CNN+GRU CosRecon LC SC3",
    "cnn_gru_dual_attn_recon_supcon3_lc":    "CNN+GRU AttnRecon LC SC3",
    # Staged SupCon ST
    "cnn_gru_dual_supcon_staged":               "CNN+GRU SC1 Staged",
    "cnn_gru_dual_cosine_recon_supcon_staged":  "CNN+GRU CosRecon SC1 Staged",
    "cnn_gru_dual_attn_recon_supcon_staged":    "CNN+GRU AttnRecon SC1 Staged",
    "cnn_gru_dual_supcon2_staged":              "CNN+GRU SC2 Staged",
    "cnn_gru_dual_cosine_recon_supcon2_staged": "CNN+GRU CosRecon SC2 Staged",
    "cnn_gru_dual_attn_recon_supcon2_staged":   "CNN+GRU AttnRecon SC2 Staged",
    "cnn_gru_dual_supcon3_staged":              "CNN+GRU SC3 Staged",
    "cnn_gru_dual_cosine_recon_supcon3_staged": "CNN+GRU CosRecon SC3 Staged",
    "cnn_gru_dual_attn_recon_supcon3_staged":   "CNN+GRU AttnRecon SC3 Staged",
    # CCGD arch-poc ST
    "ccgd_arch_poc_st":     "CCGD ST",
    "ccgd_arch_poc_st_sc1": "CCGD SC1 ST",
    "ccgd_arch_poc_st_sc2": "CCGD SC2 ST",
    "ccgd_arch_poc_st_sc3": "CCGD SC3 ST",
    "cnn_gru_dual_dann": "CNN+GRU Dual DANN",
    "cnn_gru_dual_attn_recon_dann": "CNN+GRU AttnRecon DANN",
    "cnn_gru_dual_supcon3_dann": "CNN+GRU Dual SC3 DANN",
    "cnn_gru_dual_attn_recon_supcon3_dann": "CNN+GRU AttnRecon SC3 DANN",
    "cnn_gru_dual_coral": "CNN+GRU Dual CORAL",
    "cnn_gru_dual_attn_recon_coral": "CNN+GRU AttnRecon CORAL",
    "cnn_gru_dual_supcon3_coral": "CNN+GRU Dual SC3 CORAL",
    "cnn_gru_dual_attn_recon_supcon3_coral": "CNN+GRU AttnRecon SC3 CORAL",
}
# _inc print names: append " (Inc)" so reports distinguish them from baseline variants.
MODEL_PRINT_MAP.update({
    f"{m}_inc": f"{label} (Inc)"
    for m, label in list(MODEL_PRINT_MAP.items()) if m not in _NO_INC
})

FILTER_PRINT_MAP = {
    None:                             "None (Baseline)",
    "msc_label_msc_linear_0.001":     "MSC",
    "amf_label_amf_important":        "AMF",
    "knn_top_0.95":                   "kNN-95%",
    "cnn_ae_glb_ds1_label_elbow":     "CNN-AE",
    "lstm_ae_glb_ds1_label_elbow":    "LSTM-AE",
    "spatial_knn_label_elbow":        "Spatial-kNN",
    "spatial_grid_label_elbow":       "Spatial-Grid",
    "lofo_ae":                        "LOFO LSTM-AE",
}

CURVE_PRINT_MAP = {
    "ori_curves":              "Ori Curves (raw)",
    "ori_curves_avg":          "Ori Curves (avg)",
    "ori_curves_wavelet_sym8": "Wavelet sym8",
    "ori_curves_wavelet_bior35": "Wavelet bior3.5",
    "ori_curves_sg_p4":          "SG p=4 (auto-w)",
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
    'D20260624_E00_C00_F4500KHz_U_UTI_Trial_01_neg_test': {
		0: 'Negative-Test',
		1: 'Negative-Test',
		2: 'RealTarget-2',
		3: 'NC-RealTarget-2',
		4: 'Negative-Test',
		5: 'Negative-Test',
		6: 'Negative-Test',
		7: 'Negative-Test',
		8: 'RealTarget',
		9: 'NC-RealTarget',
	},
    'D20260731_E00_C00_F4500KHz_U_DDM_01_01': {
        0: 'IAV',
        1: 'IAV',
        2: 'IAV',
        3: 'IBV',
        4: 'Kp',
        5: 'Cov',
        6: 'Hadv',
        7: 'Hadv',
        8: 'PC',
        9: 'NC-ALL',
    },
    'D20260731_E00_C00_F4500KHz_U_DDM_01_04': {
        0: 'IAV',
        1: 'IAV',
        2: 'IAV',
        3: 'IBV',
        4: 'Kp',
        5: 'Cov',
        6: 'Hadv',
        7: 'Hadv',
        8: 'PC',
        9: 'NC-ALL',
    },
    'D20260806_E00_C00_F4500KHz_U_DDM_01_06': {
        0: 'IAV',
        1: 'IAV',
        2: 'IAV',
        3: 'IBV',
        4: 'Kp',
        5: 'Cov',
        6: 'Hadv',
        7: 'Hadv',
        8: 'PC',
        9: 'NC-ALL',
    },
    'D20260804_E00_C00_F4500KHz_U_DDM_02_01': {
        0: 'IBV',
        1: 'IBV',
        2: 'IBV',
        3: 'IAV',
        4: 'IAV',
        5: 'Kp',
        6: 'Cov',
        7: 'Hadv',
        8: 'PC',
        9: 'NC-ALL',
    },
    'D20260804_E00_C00_F4500KHz_U_DDM_02_03': {
        0: 'IBV',
        1: 'IBV',
        2: 'IBV',
        3: 'IAV',
        4: 'IAV',
        5: 'Kp',
        6: 'Cov',
        7: 'Hadv',
        8: 'PC',
        9: 'NC-ALL',
    },
    'D20260807_E00_C00_F4500KHz_U_DDM_02_07': {
        0: 'IBV',
        1: 'IBV',
        2: 'IBV',
        3: 'IAV',
        4: 'IAV',
        5: 'Kp',
        6: 'Cov',
        7: 'Hadv',
        8: 'PC',
        9: 'NC-ALL',
    },
    'D20260804_E00_C00_F4500KHz_U_DDM_02_04': {
        0: 'IBV',
        1: 'IBV',
        2: 'IBV',
        3: 'IAV',
        4: 'IAV',
        5: 'Kp',
        6: 'Cov',
        7: 'Hadv',
        8: 'PC',
        9: 'NC-ALL',
    },
    'D20260808_E00_C00_F4500KHz_U_DDM_03_01': {
        0: 'Kp',
        1: 'Kp',
        2: 'Kp',
        3: 'IAV',
        4: 'IBV',
        5: 'IBV',
        6: 'Cov',
        7: 'Hadv',
        8: 'PC',
        9: 'NC-ALL',
    },
    'D20260810_E00_C00_F4500KHz_U_DDM_04_01': {
        0: 'Cov',
        1: 'Cov',
        2: 'Cov',
        3: 'IAV',
        4: 'IBV',
        5: 'Kp',
        6: 'Kp',
        7: 'Hadv',
        8: 'PC',
        9: 'NC-ALL',
    }
}

CONC_MAPPINGS = {
    'D20260731_E00_C00_F4500KHz_U_DDM_01_01': {
        0: 1000000,
        1: 100000,
        2: 10000,
        3: 10000,
        4: 100000,
        5: 1000000,
        6: 1000000,
        7: 100000,
        8: 0,
        9: 0,
    },
    'D20260731_E00_C00_F4500KHz_U_DDM_01_04': {
        0: 1000000,
        1: 100000,
        2: 10000,
        3: 10000,
        4: 100000,
        5: 1000000,
        6: 1000000,
        7: 100000,
        8: 0,
        9: 0,
    },
    'D20260806_E00_C00_F4500KHz_U_DDM_01_06': {
        0: 1000000,
        1: 100000,
        2: 10000,
        3: 10000,
        4: 100000,
        5: 1000000,
        6: 1000000,
        7: 100000,
        8: 0,
        9: 0,
    },
    'D20260804_E00_C00_F4500KHz_U_DDM_02_01': {
        0: 1000000,
        1: 100000,
        2: 10000,
        3: 1000000,
        4: 100000,
        5: 10000,
        6: 100000,
        7: 10000,
        8: 0,
        9: 0,
    },
    'D20260804_E00_C00_F4500KHz_U_DDM_02_03': {
        0: 1000000,
        1: 100000,
        2: 10000,
        3: 1000000,
        4: 100000,
        5: 10000,
        6: 100000,
        7: 10000,
        8: 0,
        9: 0,
    },
    'D20260807_E00_C00_F4500KHz_U_DDM_02_07': {
        0: 1000000,
        1: 100000,
        2: 10000,
        3: 1000000,
        4: 100000,
        5: 10000,
        6: 100000,
        7: 10000,
        8: 0,
        9: 0,
    },
    'D20260804_E00_C00_F4500KHz_U_DDM_02_04': {
        0: 1000000,
        1: 100000,
        2: 10000,
        3: 1000000,
        4: 100000,
        5: 10000,
        6: 100000,
        7: 10000,
        8: 0,
        9: 0,
    },
    'D20260808_E00_C00_F4500KHz_U_DDM_03_01': {
        0: 1000000,
        1: 100000,
        2: 10000,
        3: 10000,
        4: 1000000,
        5: 100000,
        6: 10000,
        7: 1000000,
        8: 0,
        9: 0,
    },
    'D20260810_E00_C00_F4500KHz_U_DDM_04_01': {
        0: 1000000,
        1: 100000,
        2: 10000,
        3: 1000000,
        4: 10000,
        5: 1000000,
        6: 100000,
        7: 100000,
        8: 0,
        9: 0,
    }
    
}

EXCLUDE_WELL_MAPPING = {
    'D20260806_E00_C00_F4500KHz_U_DDM_01_06': [8, 9],
    'D20260807_E00_C00_F4500KHz_U_DDM_02_07': [6, 7, 8, 9],
    'D20260808_E00_C00_F4500KHz_U_DDM_03_01': [6, 8, 9],
    'D20260810_E00_C00_F4500KHz_U_DDM_04_01': [8, 9],
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


def get_conc_array(folder_name, Y_well):
    """Expand CONC_MAPPINGS[folder_name] to a per-sample object array aligned to Y_well.

    Returns None if folder_name is not in CONC_MAPPINGS or all mapped values are
    empty / None — so datasets with no entry behave identically to before.
    Empty string '' maps to None and becomes REG_SENTINEL in 03's float-conversion pass.
    """
    mapping = CONC_MAPPINGS.get(folder_name, {})
    if not mapping:
        return None
    arr = np.array(
        [(v if (v is not None and v != '') else None)
         for v in (mapping.get(int(w), None) for w in Y_well)],
        dtype=object,
    )
    return None if all(x is None for x in arr) else arr

def apply_well_exclusion(training_data, exp_folder_name, group_name=None):
    if group_name is not None:
        source_name = f"LOFO_EXCLUDE_WELL_MAPPING[{group_name!r}]"
        excluded = LOFO_EXCLUDE_WELL_MAPPING.get(group_name, {}).get(exp_folder_name, [])
    else:
        source_name = "EXCLUDE_WELL_MAPPING"
        excluded = EXCLUDE_WELL_MAPPING.get(exp_folder_name, [])
    if not excluded:
        return training_data
    Y_well_raw = np.asarray(training_data["Y_well"])
    keep_idx = np.where(~np.isin(Y_well_raw, excluded))[0]
    def _fmt_conc(v):
        try:
            return f'{float(v):.0e}'
        except (TypeError, ValueError):
            return v

    label_map = LABEL_MAPPINGS.get(exp_folder_name, {})
    conc_map = CONC_MAPPINGS.get(exp_folder_name, {})
    y_label = {w: label_map.get(w, '?') for w in excluded}
    y_concentration = {w: _fmt_conc(conc_map.get(w, '?')) for w in excluded}
    print(f"  [*] {source_name}: dropping {len(Y_well_raw) - len(keep_idx)} "
          f"samples from wells {excluded} (y_label={y_label}, y_concentration={y_concentration}) "
          f"for {exp_folder_name}")

    training_data["dataset"] = [d[keep_idx] for d in training_data["dataset"]]
    training_data["kinetic_features"] = [
        df.iloc[keep_idx].reset_index(drop=True) for df in training_data["kinetic_features"]
    ]
    training_data["Y_well"] = Y_well_raw[keep_idx]
    if training_data.get("concentration") is not None:
        training_data["concentration"] = np.asarray(training_data["concentration"], dtype=object)[keep_idx]
    if training_data.get("metadata") is not None:
        training_data["metadata"] = {k: np.asarray(v)[keep_idx] for k, v in training_data["metadata"].items()}
    return training_data


# ==========================================
# CROSS-DATASET LOFO CV (04)
# ==========================================
CROSS_DATASET_GROUPS = {
    # 'group_name': ['exp_folder_1', 'exp_folder_2', ...],
    'init_oneplex_v6': ['D20260608_E00_C00_F4500KHz_U_norm_temp_04', 'D20260609_E00_C00_F4500KHz_U_norm_temp_read_06', 'D20260609_E00_C00_F4500KHz_U_norm_temp_read_07', 'D20260609_E00_C00_F4500KHz_U_norm_temp_ready_08'],
    'init_oneplex_nc_subtract': ['D20260608_E00_C00_F4500KHz_U_norm_temp_04', 'D20260609_E00_C00_F4500KHz_U_norm_temp_read_06', 'D20260609_E00_C00_F4500KHz_U_norm_temp_read_07', 'D20260609_E00_C00_F4500KHz_U_norm_temp_ready_08'],
    'final_chip_1_2': ['D20260731_E00_C00_F4500KHz_U_DDM_01_01', 'D20260804_E00_C00_F4500KHz_U_DDM_02_01', 'D20260804_E00_C00_F4500KHz_U_DDM_02_04', 'D20260731_E00_C00_F4500KHz_U_DDM_01_04', 'D20260804_E00_C00_F4500KHz_U_DDM_02_03', 'D20260806_E00_C00_F4500KHz_U_DDM_01_06', 'D20260807_E00_C00_F4500KHz_U_DDM_02_07'],
    'final_4_chip_clean_nn': ['D20260806_E00_C00_F4500KHz_U_DDM_01_06', 'D20260807_E00_C00_F4500KHz_U_DDM_02_07', 'D20260808_E00_C00_F4500KHz_U_DDM_03_01', 'D20260810_E00_C00_F4500KHz_U_DDM_04_01'],
    'final_4_chip_cleanv2_nn': ['D20260806_E00_C00_F4500KHz_U_DDM_01_06', 'D20260807_E00_C00_F4500KHz_U_DDM_02_07', 'D20260808_E00_C00_F4500KHz_U_DDM_03_01', 'D20260810_E00_C00_F4500KHz_U_DDM_04_01'],
    'final_4_chip_cov_hadv_iav': ['D20260806_E00_C00_F4500KHz_U_DDM_01_06', 'D20260807_E00_C00_F4500KHz_U_DDM_02_07', 'D20260808_E00_C00_F4500KHz_U_DDM_03_01', 'D20260810_E00_C00_F4500KHz_U_DDM_04_01']
}

LOFO_EXCLUDE_WELL_MAPPING = {
    'final_4_chip_clean_nn': { #no PC, no NC, no no-amp
        'D20260806_E00_C00_F4500KHz_U_DDM_01_06': [8, 9],
        'D20260807_E00_C00_F4500KHz_U_DDM_02_07': [6, 7, 8, 9],
        'D20260808_E00_C00_F4500KHz_U_DDM_03_01': [6, 8, 9],
        'D20260810_E00_C00_F4500KHz_U_DDM_04_01': [8, 9],
    },
    'final_4_chip_cleanv2_nn': { #no PC, no NC, no no-amp, no small-amp
        'D20260806_E00_C00_F4500KHz_U_DDM_01_06': [8, 9],
        'D20260807_E00_C00_F4500KHz_U_DDM_02_07': [5, 6, 7, 8, 9],
        'D20260808_E00_C00_F4500KHz_U_DDM_03_01': [6, 8, 9],
        'D20260810_E00_C00_F4500KHz_U_DDM_04_01': [4, 5, 6, 8, 9],
    },
    'final_4_chip_cov_hadv_iav': {  #no PC, no NC, no no-amp, no small-amp, no Kp, no IBV
            'D20260806_E00_C00_F4500KHz_U_DDM_01_06': [3, 4, 8, 9],
            'D20260807_E00_C00_F4500KHz_U_DDM_02_07': [0, 1, 2, 5, 6, 7, 8, 9],
            'D20260808_E00_C00_F4500KHz_U_DDM_03_01': [0, 1, 2, 4, 5, 6, 8, 9],
            'D20260810_E00_C00_F4500KHz_U_DDM_04_01': [4, 5, 6, 8, 9],
        }
}

