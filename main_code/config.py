import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "utils" / "model_training"))
from model_utils import set_global_determinism
set_global_determinism(0)

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, ListedColormap
from cycler import cycler

# ==========================================
# PATHS
# ==========================================
DATASET_ROOT_FOLDER = "/vol/bitbucket/gk225/POC_DDM_datasets"
BASE_FOLDER = DATASET_ROOT_FOLDER

DEFAULT_EXP_FOLDER = os.path.join(BASE_FOLDER, "POC_DDM_final")
FINAL_EXP_FOLDER = DEFAULT_EXP_FOLDER
LAB_EXP_FOLDER = os.path.join(BASE_FOLDER, "LAB_DDM_paper")
LAB_MULTIPLEX_FOLDER = os.path.join(BASE_FOLDER, "LAB_Multiplex")

# ==========================================
# DATASET SCOPE
# ==========================================
LAB_DATASETS_IN_SCOPE = ['01_ACA_qdPCR', '02_AMCA_qdLAMP', '03_AMCA_qdPCR',
                         '12_ACA_qdPCR_multiplex_balanced']

CROSS_DATASET_GROUPS = {
    'final_6_new': ['D20260827_E00_C00_F4500KHz_U_DDM_01_final_final', 'D20260827_E00_C00_F4500KHz_U_DDM_02_final_final', 'D20260827_E00_C00_F4500KHz_U_DDM_03_final_final', 'D20260827_E00_C00_F4500KHz_U_DDM_04_final_final', 'D20260825_E00_C00_F4500KHz_U_DDM_05_01', 'D20260825_E00_C00_F4500KHz_U_DDM_06_02'],
}

LOFO_EXCLUDE_WELL_MAPPING = {
    'final_6_new': {
        'D20260827_E00_C00_F4500KHz_U_DDM_01_final_final': [8, 9],
        'D20260827_E00_C00_F4500KHz_U_DDM_02_final_final': [8, 9],
        'D20260827_E00_C00_F4500KHz_U_DDM_03_final_final': [8, 9],
        'D20260827_E00_C00_F4500KHz_U_DDM_04_final_final': [8, 9],
        'D20260825_E00_C00_F4500KHz_U_DDM_05_01': [8, 9],
        'D20260825_E00_C00_F4500KHz_U_DDM_06_02': [6, 8, 9],
    },
}

LABEL_MAPPINGS = {
    'D20260827_E00_C00_F4500KHz_U_DDM_01_final_final': {
        0: 'IAV', 1: 'IAV', 2: 'IAV', 3: 'IBV', 4: 'Kp',
        5: 'Cov', 6: 'Hadv', 7: 'Hadv', 8: 'PC', 9: 'NC-ALL',
    },
    'D20260827_E00_C00_F4500KHz_U_DDM_02_final_final': {
        0: 'IBV', 1: 'IBV', 2: 'IBV', 3: 'IAV', 4: 'IAV',
        5: 'Kp', 6: 'Cov', 7: 'Hadv', 8: 'PC', 9: 'NC-ALL',
    },
    'D20260827_E00_C00_F4500KHz_U_DDM_03_final_final': {
        0: 'Kp', 1: 'Kp', 2: 'Kp', 3: 'IAV', 4: 'IBV',
        5: 'IBV', 6: 'Cov', 7: 'Hadv', 8: 'PC', 9: 'NC-ALL',
    },
    'D20260827_E00_C00_F4500KHz_U_DDM_04_final_final': {
        0: 'Cov', 1: 'Cov', 2: 'Cov', 3: 'IAV', 4: 'IBV',
        5: 'Kp', 6: 'Kp', 7: 'Hadv', 8: 'PC', 9: 'NC-ALL',
    },
    'D20260825_E00_C00_F4500KHz_U_DDM_05_01': {
        0: 'Hadv', 1: 'Hadv', 2: 'Hadv', 3: 'IAV', 4: 'IBV',
        5: 'Kp', 6: 'Cov', 7: 'Cov', 8: 'PC', 9: 'NC-ALL',
    },
    'D20260825_E00_C00_F4500KHz_U_DDM_06_02': {
        0: 'IAV', 1: 'IBV', 2: 'Kp', 3: 'Cov', 4: 'Hadv',
        5: 'Cov', 6: 'IBV', 7: 'Hadv', 8: 'PC', 9: 'NC-ALL',
    },
}

CONC_MAPPINGS = {
    'D20260827_E00_C00_F4500KHz_U_DDM_01_final_final': {
        0: 1000000, 1: 100000, 2: 10000, 3: 10000, 4: 100000,
        5: 1000000, 6: 1000000, 7: 100000, 8: 0, 9: 0,
    },
    'D20260827_E00_C00_F4500KHz_U_DDM_02_final_final': {
        0: 1000000, 1: 100000, 2: 10000, 3: 1000000, 4: 100000,
        5: 10000, 6: 100000, 7: 10000, 8: 0, 9: 0,
    },
    'D20260827_E00_C00_F4500KHz_U_DDM_03_final_final': {
        0: 1000000, 1: 100000, 2: 10000, 3: 10000, 4: 1000000,
        5: 100000, 6: 10000, 7: 1000000, 8: 0, 9: 0,
    },
    'D20260827_E00_C00_F4500KHz_U_DDM_04_final_final': {
        0: 1000000, 1: 100000, 2: 10000, 3: 1000000, 4: 10000,
        5: 1000000, 6: 100000, 7: 100000, 8: 0, 9: 0,
    },
    'D20260825_E00_C00_F4500KHz_U_DDM_05_01': {
        0: 1000000, 1: 100000, 2: 10000, 3: 100000, 4: 1000000,
        5: 10000, 6: 1000000, 7: 100000, 8: 0, 9: 0,
    },
    'D20260825_E00_C00_F4500KHz_U_DDM_06_02': {
        0: 10000, 1: 100000, 2: 1000000, 3: 10000, 4: 10000,
        5: 100000, 6: 10000, 7: 10000, 8: 0, 9: 0,
    },
}

EXCLUDE_WELL_MAPPING = {
    'D20260827_E00_C00_F4500KHz_U_DDM_01_final_final': [8, 9],
    'D20260827_E00_C00_F4500KHz_U_DDM_02_final_final': [8, 9],
    'D20260827_E00_C00_F4500KHz_U_DDM_03_final_final': [8, 9],
    'D20260827_E00_C00_F4500KHz_U_DDM_04_final_final': [8, 9],
    'D20260825_E00_C00_F4500KHz_U_DDM_05_01': [8, 9],
    'D20260825_E00_C00_F4500KHz_U_DDM_06_02': [8, 9],
}

# ==========================================
# MODEL / FILTER SCOPE
# ==========================================
AE_DOWNSAMPLE_FACTOR = 1
OUTLIER_FILTERS = [
    None,
    f'lstm_ae_glb_ds{AE_DOWNSAMPLE_FACTOR}_label_elbow',
    f'lstm_ae_glb_ds{AE_DOWNSAMPLE_FACTOR}_label_90',
    f'lstm_ae_glb_ds{AE_DOWNSAMPLE_FACTOR}_label_95',
]

# model key -> (y_preds_ key, y_probs_ key, classes_ key), written by
# evaluate_outlier_filters() in utils/model_training/model_utils.py.
MODEL_KEY_MAP = {
    "cnn":         ("y_preds_AC_",       "y_probs_AC_",       "classes_AC_"),
    "gru":         ("y_preds_AC_gru_",   "y_probs_AC_gru_",   "classes_AC_gru_"),
    "transformer": ("y_preds_AC_trans_", "y_probs_AC_trans_", "classes_AC_trans_"),
    "knn":         ("y_preds_AC_kNN_",   "y_probs_AC_kNN_",   "classes_AC_kNN_"),
    "cnn_gru_dual": ("y_preds_AC_cnn_gru_dual_", "y_probs_AC_cnn_gru_dual_", "classes_AC_cnn_gru_dual_"),
    "cnn_gru_dual_attn_recon": (
        "y_preds_AC_cnn_gru_dual_attn_recon_",
        "y_probs_AC_cnn_gru_dual_attn_recon_",
        "classes_AC_cnn_gru_dual_attn_recon_",
    ),
    "cnn_gru_dual_supcon": (
        "y_preds_AC_cnn_gru_dual_supcon_",
        "y_probs_AC_cnn_gru_dual_supcon_",
        "classes_AC_cnn_gru_dual_supcon_",
    ),
    "cnn_gru_dual_attn_recon_supcon": (
        "y_preds_AC_cnn_gru_dual_attn_recon_supcon_",
        "y_probs_AC_cnn_gru_dual_attn_recon_supcon_",
        "classes_AC_cnn_gru_dual_attn_recon_supcon_",
    ),
    "cnn_gru_dual_supcon3": (
        "y_preds_AC_cnn_gru_dual_supcon3_",
        "y_probs_AC_cnn_gru_dual_supcon3_",
        "classes_AC_cnn_gru_dual_supcon3_",
    ),
    "cnn_gru_dual_attn_recon_supcon3": (
        "y_preds_AC_cnn_gru_dual_attn_recon_supcon3_",
        "y_probs_AC_cnn_gru_dual_attn_recon_supcon3_",
        "classes_AC_cnn_gru_dual_attn_recon_supcon3_",
    ),
    "cnn_gru_dual_dann": (
        "y_preds_AC_cnn_gru_dual_dann_",
        "y_probs_AC_cnn_gru_dual_dann_",
        "classes_AC_cnn_gru_dual_dann_",
    ),
    "cnn_gru_dual_attn_recon_dann": (
        "y_preds_AC_cnn_gru_dual_attn_recon_dann_",
        "y_probs_AC_cnn_gru_dual_attn_recon_dann_",
        "classes_AC_cnn_gru_dual_attn_recon_dann_",
    ),
    "cnn_gru_dual_attn_recon_aug": (
        "y_preds_AC_cnn_gru_dual_attn_recon_aug_",
        "y_probs_AC_cnn_gru_dual_attn_recon_aug_",
        "classes_AC_cnn_gru_dual_attn_recon_aug_",
    ),
    "cnn_gru_dual_attn_recon_mtl": (
        "y_preds_AC_cnn_gru_dual_attn_recon_mtl_",
        "y_probs_AC_cnn_gru_dual_attn_recon_mtl_",
        "classes_AC_cnn_gru_dual_attn_recon_mtl_",
    ),
}

MODEL_PRINT_MAP = {
    "cnn": "CNN (ACA)",
    "gru": "GRU (ACA)",
    "transformer": "Trans (ACA)",
    "knn": "KNN (ACA)",
    "cnn_gru_dual": "CNN+GRU Dual",
    "cnn_gru_dual_attn_recon": "CNN+GRU AttnRecon",
    "cnn_gru_dual_supcon": "CNN+GRU Dual SupCon",
    "cnn_gru_dual_attn_recon_supcon": "CNN+GRU AttnRecon SC",
    "cnn_gru_dual_supcon3": "CNN+GRU Dual SC3",
    "cnn_gru_dual_attn_recon_supcon3": "CNN+GRU AttnRecon SC3",
    "cnn_gru_dual_dann": "CNN+GRU Dual DANN",
    "cnn_gru_dual_attn_recon_dann": "CNN+GRU AttnRecon DANN",
    "cnn_gru_dual_attn_recon_aug": "CNN+GRU AttnRecon Aug",
    "cnn_gru_dual_attn_recon_mtl": "CNN+GRU AttnRecon MTL",
}

# ==========================================
# INTERNAL / UTILITY CONFIG
# ==========================================
EXCLUDED_FOLDERS = ['.DS_Store', 'model_interpretation', 'model_interpretation_old',
                    'outlier_visualisation', 'outlier_visualisation_old', 'cross_dataset_cv',
                    'model_performance_viz', 'model_performance_viz_old',
                    '1_area', '2_range', '3_range_filtered']

CURVE_TYPE_ALIASES = {
    "ori_curve":            "ori_curves",
    "ori_curve_norm":       "ori_curves_norm",
    "ori_curve_sg_p4":      "ori_curves_sg_p4",
    "ori_curve_sg_p4_norm": "ori_curves_sg_p4_norm",
}


def resolve_curve_dataset_idx(curve_type, dataset_name_list):
    target = CURVE_TYPE_ALIASES.get(curve_type, curve_type)
    if target not in dataset_name_list:
        raise ValueError(f"curve_type '{curve_type}' (resolved to '{target}') not found in dataset_name {list(dataset_name_list)}")
    return list(dataset_name_list).index(target), target


N_WELLS = 10
N_A_TYPE = "v06"
WINDOW_SIZE_ORI = 50

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


def cross_dataset_alignment_dir(out_dir, held_out_chip=None):
    out_dir = Path(out_dir)
    if held_out_chip is None:
        return out_dir
    return out_dir / "model_interpretation" / f"lofo_{held_out_chip}"


mpl_colors = [
    (0.00, 0.45, 0.70), (0.90, 0.60, 0.00), (0.35, 0.70, 0.90),
    (0.00, 0.60, 0.50), (0.95, 0.90, 0.25), (0.80, 0.40, 0.70),
    (0.20, 0.13, 0.53), (0.87, 0.80, 0.47), (0.27, 0.67, 0.60), (0.65, 0.65, 0.65)
]
plt.rcParams['axes.prop_cycle'] = cycler(color=[to_hex(i) for i in mpl_colors])
VIZ_PALETTE = [to_hex(c) for c in mpl_colors]


def get_palette(categories, fixed_map=None):
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

FILTER_PRINT_MAP = {
    None: "None (Baseline)",
    "lstm_ae_glb_ds1_label_elbow": "LSTM-AE",
    "lofo_ae": "LOFO LSTM-AE",
    "noamp_remove": "Non-Amplifying Removed",
}

CURVE_PRINT_MAP = {
    "ori_curves": "Ori Curves (raw)",
    "ori_curves_sg_p4": "SG p=4 (auto-w)",
}

METRIC_LABEL = {"accuracy": "Accuracy", "macro_f1": "Macro-F1", "mcc": "MCC"}

SAVED_VIZ = []

LD_FEATURES = [
    'Fm', 'Fb', 'Sc', 'Cs', 'As', 'xms', 'xs', 'xe', 'xp1', 'xp2', 'TH',
    'y_xms', 'y_xs', 'y_xe', 'y_xp1', 'y_xp2', 'amplitude', 'dy_xms',
    'dy_xp1', 'dy_xp2', 'd2y_xp1', 'd2y_xp2', 'threshold_distance',
    'first_half_distance', 'second_half_distance', 'distance_asymmetry_index',
    'peak_shifting_distance', 'A1', 'A2', 'area_asymmetry_index',
    'peak_asymmetry_index', 'Ct', 'Cy0', 'F_max', 'log_F0', 'F0', 'Send',
    'FFI', 'F_range'
]

# Used by utils/saliency.py's plot_latent_feature_mapping to correlate ranked
# latent saliency dimensions against interpretable kinetic-curve features.
XAI_KINETIC_FEATURE_GROUP = {
    "Ct": ["Ct_ori", "ct_idx", "ct_idx_ori", "Cy0", "Cy0_ori"],
    "lag_time": ["xs", "xp1", "t10"],
    "xms": ["Cs", "t50"],
    "xe": ["xp2", "t90"],
    "rise_time_10_90": ["rise_time_20_80", "threshold_distance", "first_half_distance",
                        "second_half_distance", "peak_shifting_distance"],
    "dy_xms": ["Sc", "dy_xp1", "dy_xp2", "TH"],
    "max_accel": ["d2y_xp1"],
    "min_accel": ["d2y_xp2"],
    "amplitude": ["F_max", "F_max_ori", "Fm", "plateau_mean", "y_xe", "y_xp2"],
    "baseline_mean": ["Fb", "F0", "log_F0", "y_xs"],
    "y_xms": ["y_xp1"],
    "auc_norm": ["auc", "A1", "A2"],
    "area_asymmetry_index": ["As", "distance_asymmetry_index", "peak_asymmetry_index", "accel_fwhm"],
    "baseline_slope": [],
    "plateau_slope": ["Send", "Send_abs", "Send_fit", "Send_fit_abs"],
}


def get_label_mappings(exp_path):
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
