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
# DEFAULT_EXP_FOLDER = "/rds/general/user/gk225/home/Run Data/POC_DDM_dataset/"
# DEFAULT_EXP_FOLDER = "/Users/kautsarg/Documents/Final Project/Run Data/trial test data"
LAB_ROOT_FOLDER = "/vol/bitbucket/gk225/POC_DDM_datasets"
HPC_ROOT_FOLDER = "/rds/general/user/gk225/home/POC_DDM_datasets/POC_DDM_datasets"
LOCAL_ROOT_FOLDER = "/Users/kautsarg/Documents/Final Project/Run Data"

BASE_FOLDER = LAB_ROOT_FOLDER

DEFAULT_EXP_FOLDER = os.path.join(BASE_FOLDER, "POC_DDM_chip_init")
LAB_EXP_FOLDER = os.path.join(BASE_FOLDER, "LAB_DDM_paper")
MULTI_EXP_FOLDER = os.path.join(BASE_FOLDER, "POC_DDM_multi")

# Global Experiment Parameters
N_WELLS = 10
N_A_TYPE = "v06"

# Preprocessing Window Sizes
WINDOW_SIZE_ORI = 50
WINDOW_SIZE_1STDER = 200

# Plotting Optimizations (Downsampling for Bokeh)
PLOT_DOWNSAMPLE_STEP = 1
PLOT_DECIMAL_PRECISION = 4

# AutoEncoder Downsample Factor
AE_DOWNSAMPLE_FACTOR = 1

# Spatial Consistency Outlier Filters
SPATIAL_CONSISTENCY_KNN_K = 24       # Number of nearest neighbors (by pixel coordinate distance)
SPATIAL_CONSISTENCY_GRID_WINDOW = 2  # Grid half-width -> (2*window+1)^2 neighborhood (5x5)

# All outlier filter labels produced by 02_par-outlier_detection_pipeline.py
# (None = no filtering / baseline)
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

# File paths
PREPROCESSED_CURVES_PATH = 'preprocessed_curves_nonorm.joblib'
TRAINING_DATA_PATH = 'curve_for_training_nonorm.joblib'
TRAINING_RESULT_PATH = 'classification_performances_nonorm.joblib'
TRAINING_10FOLD_RESULT_PATH = 'classification_performances_10fold_nonorm.joblib'
CROSS_DATASET_RESULT_PATH = 'classification_performances_cross_dataset_{mode}.joblib'
CROSS_DATASET_RESAMPLER_PATH = 'classification_performances_cross_dataset_resampler.joblib'

# File paths
# PREPROCESSED_CURVES_PATH = 'preprocessed_curves.joblib'
# TRAINING_DATA_PATH = 'curve_for_training.joblib'
# TRAINING_RESULT_PATH = 'classification_performances.joblib'
# TRAINING_10FOLD_RESULT_PATH = 'classification_performances_10fold.joblib'

mpl_colors = [
    (0.00, 0.45, 0.70), (0.90, 0.60, 0.00), (0.35, 0.70, 0.90), 
    (0.00, 0.60, 0.50), (0.95, 0.90, 0.25), (0.80, 0.40, 0.70), 
    (0.20, 0.13, 0.53), (0.87, 0.80, 0.47), (0.27, 0.67, 0.60), (0.65, 0.65, 0.65)
]
plt.rcParams['axes.prop_cycle'] = cycler(color=[to_hex(i) for i in mpl_colors])

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
    "amf_label_amf_important", "amf_label_amf_send_5"
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

SAVED_VIZ = [
    "D20250808_E00_C00_F4500KHz_U_Sample_7",
    "D20250820_E00_C00_F4500KHz_U_Sample_18_wet_3"
]

LD_FEATURES = [
    'Fm', 'Fb', 'Sc', 'Cs', 'As', 'xms', 'xs', 'xe', 'xp1', 'xp2', 'TH', 
    'y_xms', 'y_xs', 'y_xe', 'y_xp1', 'y_xp2', 'amplitude', 'dy_xms', 
    'dy_xp1', 'dy_xp2', 'd2y_xp1', 'd2y_xp2', 'threshold_distance', 
    'first_half_distance', 'second_half_distance', 'distance_asymmetry_index', 
    'peak_shifting_distance', 'A1', 'A2', 'area_asymmetry_index', 
    'peak_asymmetry_index', 'Ct', 'Cy0', 'F_max', 'log_F0', 'F0', 'Send', 
    'FFI', 'F_range'
]

RERUN_MODELS = [
    # "cnn_lf"
]

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
    'D20260320_E00_C00_F4500KHz_U_Elena_steap_cv_nc_subtract': {
		0: 'S',
		1: 'C',
		2: 'C',
		3: 'S',
		4: 'S',
		5: 'C',
		6: 'C',
		7: 'S',
		8: 'NC',
		9: 'NC',
	},
    'D20260522_E00_C00_F4500KHz_U_manifold_test_05': {
		0: 'Target',
		1: 'NC-Target',
		2: 'NC-Target',
		3: 'Target',
		4: 'Target',
		5: 'NC-Target',
		6: 'NC-Target',
		7: 'Target',
		8: 'Target',
		9: 'NC-Target',
	},
    'D20260522_E00_C00_F4500KHz_U_manifold_test_05_nc_subtract': {
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
		1: 'NC-Target',
		2: 'NC-Target',
		3: 'Target',
		4: 'Target',
		5: 'NC-Target',
		6: 'NC-Target',
		7: 'Target',
		8: 'Target',
		9: 'NC-Target',
    },
    'D20260608_E00_C00_F4500KHz_U_norm_temp_04_nc_subtract':{
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
		1: 'NC-Target',
		2: 'NC-Target',
		3: 'Target',
		4: 'Target',
		5: 'NC-Target',
		6: 'NC-Target',
		7: 'Target',
		8: 'Target',
		9: 'NC-Target',
    },
    'D20260609_E00_C00_F4500KHz_U_norm_temp_read_06_nc_subtract':{
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
		1: 'NC-Target',
		2: 'NC-Target',
		3: 'Target',
		4: 'Target',
		5: 'NC-Target',
		6: 'NC-Target',
		7: 'Target',
		8: 'Target',
		9: 'NC-Target',
    },
    'D20260609_E00_C00_F4500KHz_U_norm_temp_read_07_nc_subtract':{
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
		1: 'NC-Target',
		2: 'NC-Target',
		3: 'Target',
		4: 'Target',
		5: 'NC-Target',
		6: 'NC-Target',
		7: 'Target',
		8: 'Target',
		9: 'NC-Target',
    },
    'D20260609_E00_C00_F4500KHz_U_norm_temp_ready_08_nc_subtract':{
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
    'D20260611_E00_C00_F4500KHz_U_lambda_test_manifold_01_nc_subtract': {
		0: 'Conc-01',
		1: 'Conc-02',
		2: 'Conc-02',
		3: 'Conc-01',
		4: 'Conc-01',
		5: 'Conc-02',
		6: 'Conc-02',
		7: 'Conc-01',
		8: 'NC',
		9: 'NC',
	},
}

# ==========================================
# CROSS-DATASET ROBUSTNESS CV (03b)
# ==========================================
# Manually defined groups of experiment folders to combine for cross-dataset
# CV (see 03b_par-cross_dataset_training.py). All folders within a group MUST
# share an IDENTICAL well-index -> label mapping in LABEL_MAPPINGS above,
# since the well-based CV folds rely on that mapping being consistent.
CROSS_DATASET_GROUPS = {
    # 'group_name': ['exp_folder_1', 'exp_folder_2', ...],
    'init_oneplex_nc_subtract': ['D20260608_E00_C00_F4500KHz_U_norm_temp_04_nc_subtract', 'D20260609_E00_C00_F4500KHz_U_norm_temp_read_06_nc_subtract', 'D20260609_E00_C00_F4500KHz_U_norm_temp_read_07_nc_subtract', 'D20260609_E00_C00_F4500KHz_U_norm_temp_ready_08_nc_subtract']
}

EXCLUDED_FOLDERS = ['.DS_Store', 'model_interpretation', 'model_interpretation_old', 'outlier_visualisation', 'outlier_visualisation_old', 'cross_dataset_cv', 'model_performance_viz', 'model_performance_viz_old', 'outlier_visualisation', 'outlier_visualisation_old']