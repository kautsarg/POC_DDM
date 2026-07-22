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
    '02_ACA_qdPCR_balanced': 'dPCR_Dataset_Mixture.csv',

}

FILE_CONC = {
    '01_ACA_qdPCR': 'Conc',
    '02_ACA_qdPCR_balanced': 'Conc',
}

FILE_TARGET = {
    '01_ACA_qdPCR': 'LoadedPanels',
    '02_ACA_qdPCR_balanced': 'LoadedPanels',
}

# Separator used to split multi-target combination strings ("VIM_NDM" -> ["VIM","NDM"])
MULTIPLEX_SEPARATOR = '_'

# Individual target genes present in the dataset (sorted for deterministic binarization)
MULTIPLEX_TARGETS = ['KPC', 'NDM', 'VIM']

# ==========================================
# RESULT FILE PATHS (separate from main/ to avoid collisions)
# ==========================================

TRAINING_DATA_PATH          = 'curve_for_training_ml.joblib'
TRAINING_RESULT_PATH        = 'classification_performances_ml.joblib'
TRAINING_10FOLD_RESULT_PATH = 'classification_performances_ml_10fold.joblib'

# Per-group result files — one per parallel-job flag (keeps SLURM jobs race-free)
RESULT_FILE_BY_FLAG = {
    'default':    'classification_performances_ml.joblib',
    'cross_attn': 'classification_performances_ml_cross_attn.joblib',
    'cattn_v2':   'classification_performances_ml_cattn_v2.joblib',
    'auxdet':     'classification_performances_ml_auxdet.joblib',
    'quercon':    'classification_performances_ml_quercon.joblib',
    'condreg':    'classification_performances_ml_condreg.joblib',
    'crf':              'classification_performances_ml_crf.joblib',
    'source_sep':       'classification_performances_ml_source_sep.joblib',
    'source_sep_precon': 'classification_performances_ml_source_sep_precon.joblib',
}
RESULT_10FOLD_FILE_BY_FLAG = {
    k: v.replace('.joblib', '_10fold.joblib')
    for k, v in RESULT_FILE_BY_FLAG.items()
}


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
# MULTIPLEX MODEL KEYS (128 total)
# ML_MODEL_KEY_MAP and ML_MODEL_PRINT_MAP are defined in
# utils/model_training/model_utils_multilabel.py — import from there.
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
    # Dual-branch CNN+GRU cross-attn: base + SC1/2/3
    'cnn_gru_dual_cross_attn',
    'cnn_gru_dual_cross_attn_supcon', 'cnn_gru_dual_cross_attn_supcon2', 'cnn_gru_dual_cross_attn_supcon3',
    # Dual-branch CNN+Trans cross-attn: base + SC1/2/3
    'cnn_trans_dual_cross_attn',
    'cnn_trans_dual_cross_attn_supcon', 'cnn_trans_dual_cross_attn_supcon2', 'cnn_trans_dual_cross_attn_supcon3',
    # v2 ablation — DeepKV: SC0-3
    'cnn_gru_dual_cross_attn_deepkv',
    'cnn_gru_dual_cross_attn_deepkv_supcon', 'cnn_gru_dual_cross_attn_deepkv_supcon2', 'cnn_gru_dual_cross_attn_deepkv_supcon3',
    # v2 ablation — DeepHead: SC0-3
    'cnn_gru_dual_cross_attn_deephead',
    'cnn_gru_dual_cross_attn_deephead_supcon', 'cnn_gru_dual_cross_attn_deephead_supcon2', 'cnn_gru_dual_cross_attn_deephead_supcon3',
    # v2 combined — CGD: SC0-3
    'cnn_gru_dual_cross_attn_v2',
    'cnn_gru_dual_cross_attn_v2_supcon', 'cnn_gru_dual_cross_attn_v2_supcon2', 'cnn_gru_dual_cross_attn_v2_supcon3',
    # v2 combined — CTD: SC0-3
    'cnn_trans_dual_cross_attn_v2',
    'cnn_trans_dual_cross_attn_v2_supcon', 'cnn_trans_dual_cross_attn_v2_supcon2', 'cnn_trans_dual_cross_attn_v2_supcon3',
    # v2 AuxDet — CGD: SC0-3
    'cnn_gru_dual_cross_attn_v2_auxdet',
    'cnn_gru_dual_cross_attn_v2_auxdet_supcon', 'cnn_gru_dual_cross_attn_v2_auxdet_supcon2', 'cnn_gru_dual_cross_attn_v2_auxdet_supcon3',
    # v2 QuerCon — CGD: QuerCon + backbone SC0-3
    'cnn_gru_dual_cross_attn_v2_quercon',
    'cnn_gru_dual_cross_attn_v2_quercon_supcon', 'cnn_gru_dual_cross_attn_v2_quercon_supcon2', 'cnn_gru_dual_cross_attn_v2_quercon_supcon3',
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
    # CRF-MRF (Option A: full-state 8-class NLL) SC0-3
    'cnn_gru_dual_crf_mrf',
    'cnn_gru_dual_crf_mrf_supcon', 'cnn_gru_dual_crf_mrf_supcon2', 'cnn_gru_dual_crf_mrf_supcon3',
    'cnn_trans_dual_crf_mrf',
    'cnn_trans_dual_crf_mrf_supcon', 'cnn_trans_dual_crf_mrf_supcon2', 'cnn_trans_dual_crf_mrf_supcon3',
    # CRF-chain (Option B: linear-chain) SC0-3
    'cnn_gru_dual_crf_chain',
    'cnn_gru_dual_crf_chain_supcon', 'cnn_gru_dual_crf_chain_supcon2', 'cnn_gru_dual_crf_chain_supcon3',
    'cnn_trans_dual_crf_chain',
    'cnn_trans_dual_crf_chain_supcon', 'cnn_trans_dual_crf_chain_supcon2', 'cnn_trans_dual_crf_chain_supcon3',
    # Source separation pretrained encoder — legacy (SC0-1, no phase suffix)
    'cnn_gru_source_sep',
    'cnn_gru_source_sep_supcon',
    'cnn_gru_source_sep_crf',
    'cnn_gru_source_sep_crf_supcon',
    # Source sep Family A — standard encoder, SC0-3, explicit phase suffix
    'cnn_gru_source_sep_p2',      'cnn_gru_source_sep_crf_p2',
    'cnn_gru_source_sep_p3',      'cnn_gru_source_sep_crf_p3',
    'cnn_gru_source_sep_supcon_p3',   'cnn_gru_source_sep_crf_supcon_p3',
    'cnn_gru_source_sep_supcon2_p3',  'cnn_gru_source_sep_crf_supcon2_p3',
    'cnn_gru_source_sep_supcon3_p3',  'cnn_gru_source_sep_crf_supcon3_p3',
    # Source sep Family B — SC-specific precon encoder, SC1-3, p2+p3
    'cnn_gru_source_sep_precon_supcon_p2',    'cnn_gru_source_sep_precon_crf_supcon_p2',
    'cnn_gru_source_sep_precon_supcon2_p2',   'cnn_gru_source_sep_precon_crf_supcon2_p2',
    'cnn_gru_source_sep_precon_supcon3_p2',   'cnn_gru_source_sep_precon_crf_supcon3_p2',
    'cnn_gru_source_sep_precon_supcon_p3',    'cnn_gru_source_sep_precon_crf_supcon_p3',
    'cnn_gru_source_sep_precon_supcon2_p3',   'cnn_gru_source_sep_precon_crf_supcon2_p3',
    'cnn_gru_source_sep_precon_supcon3_p3',   'cnn_gru_source_sep_precon_crf_supcon3_p3',
    # CAttn-V2 + CRF-MRF: flat/factored/bilinear × CGD SC0-3
    'cnn_gru_dual_cross_attn_v2_crf_flat',
    'cnn_gru_dual_cross_attn_v2_crf_flat_supcon', 'cnn_gru_dual_cross_attn_v2_crf_flat_supcon2', 'cnn_gru_dual_cross_attn_v2_crf_flat_supcon3',
    'cnn_gru_dual_cross_attn_v2_crf_factored',
    'cnn_gru_dual_cross_attn_v2_crf_factored_supcon', 'cnn_gru_dual_cross_attn_v2_crf_factored_supcon2', 'cnn_gru_dual_cross_attn_v2_crf_factored_supcon3',
    'cnn_gru_dual_cross_attn_v2_crf_bilinear',
    'cnn_gru_dual_cross_attn_v2_crf_bilinear_supcon', 'cnn_gru_dual_cross_attn_v2_crf_bilinear_supcon2', 'cnn_gru_dual_cross_attn_v2_crf_bilinear_supcon3',
    # CAttn-V2 + CRF-MRF: flat/factored/bilinear × CTD SC0-3
    'cnn_trans_dual_cross_attn_v2_crf_flat',
    'cnn_trans_dual_cross_attn_v2_crf_flat_supcon', 'cnn_trans_dual_cross_attn_v2_crf_flat_supcon2', 'cnn_trans_dual_cross_attn_v2_crf_flat_supcon3',
    'cnn_trans_dual_cross_attn_v2_crf_factored',
    'cnn_trans_dual_cross_attn_v2_crf_factored_supcon', 'cnn_trans_dual_cross_attn_v2_crf_factored_supcon2', 'cnn_trans_dual_cross_attn_v2_crf_factored_supcon3',
    'cnn_trans_dual_cross_attn_v2_crf_bilinear',
    'cnn_trans_dual_cross_attn_v2_crf_bilinear_supcon', 'cnn_trans_dual_cross_attn_v2_crf_bilinear_supcon2', 'cnn_trans_dual_cross_attn_v2_crf_bilinear_supcon3',
]


def _flag_for_model(key):
    if 'source_sep_precon' in key: return 'source_sep_precon'
    if 'source_sep' in key: return 'source_sep'
    if 'crf'        in key: return 'crf'
    if 'quercon'    in key: return 'quercon'
    if 'auxdet'     in key: return 'auxdet'
    if any(x in key for x in ('deepkv', 'deephead', '_v2')): return 'cattn_v2'
    if 'cross_attn' in key: return 'cross_attn'
    if 'rcfd'       in key: return 'condreg'
    return 'default'


MODEL_FLAG_MAP = {k: _flag_for_model(k) for k in MULTIPLEX_MODELS}
