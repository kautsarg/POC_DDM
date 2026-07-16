import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils" / "model_training"))

from config import *  # noqa: F401,F403  inherit all base constants

# ==========================================
# MULTIPLEX DATASET PATHS
# ==========================================

LAB_MULTIPLEX_FOLDER = os.path.join(BASE_FOLDER, "LAB_Multiplex")

FILE_MAPPING = {
    '01_ACA_qdPCR': 'dPCR_Dataset_Multiplex.csv',
}

FILE_CONC = {
    '01_ACA_qdPCR': 'Conc',
}

FILE_TARGET = {
    '01_ACA_qdPCR': 'LoadedPanels',
}

# Separator used to split multi-target combination strings ("VIM_NDM" -> ["VIM","NDM"])
MULTIPLEX_SEPARATOR = '_'

# Individual target genes present in the dataset (sorted for deterministic binarization)
MULTIPLEX_TARGETS = ['KPC', 'NDM', 'VIM']

# ==========================================
# RESULT FILE PATHS (separate from main/ to avoid collisions)
# ==========================================

TRAINING_DATA_PATH         = 'curve_for_training_ml.joblib'
TRAINING_RESULT_PATH       = 'classification_performances_ml.joblib'
TRAINING_10FOLD_RESULT_PATH = 'classification_performances_ml_10fold.joblib'

# ==========================================
# OUTLIER FILTERS FOR MULTIPLEX
# Spatial filters listed for structural consistency; they produce no column
# on flat-CSV lab data (no row/col coordinates) and are gracefully skipped.
# ==========================================

OUTLIER_FILTERS = [
    None,
    f'lstm_ae_glb_ds{AE_DOWNSAMPLE_FACTOR}_label_elbow',
    'spatial_knn_label_elbow',
    'spatial_grid_label_elbow',
]

# ==========================================
# MULTIPLEX MODEL KEYS
# 12 standard + SupCon ST variants (no spatial coords → cosine_recon/attn_recon skipped at runtime)
# 4 RCFD variants (gru_rcfd_cgd family with concentration regression head)
# ==========================================

MULTIPLEX_MODELS = [
    # Single-branch standard ST
    'cnn', 'gru', 'transformer',
    # Single-branch SupCon SC1
    'cnn_supcon', 'gru_supcon', 'transformer_supcon',
    # Dual-branch CNN+GRU: base + SC1/2/3
    'cnn_gru_dual',
    'cnn_gru_dual_supcon', 'cnn_gru_dual_supcon2', 'cnn_gru_dual_supcon3',
    # Dual-branch CNN+Trans: base + SC1/2/3
    'cnn_trans_dual',
    'cnn_trans_dual_supcon', 'cnn_trans_dual_supcon2', 'cnn_trans_dual_supcon3',
    # RCFD — GRU early encoder, CNN+GRU dual
    'gru_rcfd_cgd',
    'gru_rcfd_cgd_supcon_mtl', 'gru_rcfd_cgd_supcon2_mtl', 'gru_rcfd_cgd_supcon3_mtl',
    # RCFD — GRU early encoder, CNN+Trans dual
    'gru_rcfd_ctd',
    'gru_rcfd_ctd_supcon_mtl', 'gru_rcfd_ctd_supcon2_mtl', 'gru_rcfd_ctd_supcon3_mtl',
    # RCFD — Transformer early encoder, CNN+GRU dual
    'trans_rcfd_cgd',
    'trans_rcfd_cgd_supcon_mtl', 'trans_rcfd_cgd_supcon2_mtl', 'trans_rcfd_cgd_supcon3_mtl',
    # RCFD — Transformer early encoder, CNN+Trans dual
    'trans_rcfd_ctd',
    'trans_rcfd_ctd_supcon_mtl', 'trans_rcfd_ctd_supcon2_mtl', 'trans_rcfd_ctd_supcon3_mtl',
]

# ML_MODEL_KEY_MAP and ML_MODEL_PRINT_MAP are defined in
# utils/model_training/model_utils_multilabel.py — import from there.
