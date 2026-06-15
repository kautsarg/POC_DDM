"""
04_par-gnn_spatial_training.py
================================
Spatial GNN training pipeline — mirrors the flow of 02/03 so results drop
straight into the SAME result file and the SAME nested format used by 03.

Two GNN variants are added as if they were two more "models" in the
existing leaderboard:
    "gnn_2stage"  → Model A: BiGRU-CNN → GNN (two-stage + fine-tune)
    "gnn_e2e"     → Model B: End-to-End GNN (trained from scratch)

Result integration (identical structure to 03):
    all_ml_results[clean_title][mode_key][filter_f] = res_entry
res_entry gains:
    y_preds_AC_gnn_2stage_ / y_probs_AC_gnn_2stage_ / classes_AC_gnn_2stage_
    y_preds_AC_gnn_e2e_    / y_probs_AC_gnn_e2e_    / classes_AC_gnn_e2e_

Same result file selection as 03:
    n_splits > 1 → config.TRAINING_10FOLD_RESULT_PATH
    else         → config.TRAINING_RESULT_PATH

Same outlier_filters loop and Native/Reference modes as 03.

Usage (array job, one folder per task — identical to 03):
    python 04_par-gnn_spatial_training.py --task_id 0 --exp_folder /data/exps
    python 04_par-gnn_spatial_training.py --task_id 0 --n_splits 10
    python 04_par-gnn_spatial_training.py --task_id 0 --gnn_models gnn_e2e
    python 04_par-gnn_spatial_training.py --task_id 0 --modes Native,Reference
"""

import os
import sys
import gc
import time
import argparse
import warnings
import joblib
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, '../main')
sys.path.insert(0, '../main/utils/model_training')
import config
import model_utils
from model_utils import plot_ml_results, set_global_determinism

# --- PyTorch / PyG (only needed for the GNN itself) ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from torch_geometric.nn import GCNConv, GATConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

warnings.filterwarnings("ignore", category=UserWarning)
set_global_determinism(0)
torch.manual_seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# HYPERPARAMETERS
# ============================================================
LATENT_DIM     = 64
GNN_HIDDEN_DIM = 128
GNN_LAYERS     = 2
KNN_K          = 6
DROPOUT        = 0.3

# Two-stage (Model A)
STAGE2_EPOCHS  = 60      # GNN-only
STAGE3_EPOCHS  = 20      # fine-tune end-to-end
LR_STAGE2      = 1e-3
LR_STAGE3_ENC  = 1e-5    # tiny LR for pretrained encoder
LR_STAGE3_GNN  = 1e-4

# End-to-end (Model B)
E2E_EPOCHS     = 80
LR_E2E         = 5e-4

# GNN model keys (mirrors model_key_map in model_utils.evaluate_outlier_filters)
GNN_MODEL_KEY_MAP = {
    "gnn_2stage": ("y_preds_AC_gnn_2stage_", "y_probs_AC_gnn_2stage_", "classes_AC_gnn_2stage_"),
    "gnn_e2e":    ("y_preds_AC_gnn_e2e_",    "y_probs_AC_gnn_e2e_",    "classes_AC_gnn_e2e_"),
}
GNN_MODEL_PRINT_MAP = {
    "gnn_2stage": "GNN 2-Stage",
    "gnn_e2e":    "GNN E2E",
}


# ============================================================
# PATCH plot_ml_results SO IT DETECTS THE GNN MODELS
# ============================================================
# plot_ml_results auto-detects models via a hard-coded list of
# "y_preds_AC_..." keys. We wrap it to append the GNN detections so the
# GNN rows appear automatically on the same accuracy/leaderboard plots.
_ORIG_PLOT_ML_RESULTS = model_utils.plot_ml_results


def plot_ml_results_with_gnn(results_dict, outlier_filters, dataset_name, mode_name, total_count, save_prefix=None):
    """Call original plotting, then append GNN-only leaderboard lines/plot."""
    base = _ORIG_PLOT_ML_RESULTS(results_dict, outlier_filters, dataset_name, mode_name, total_count, save_prefix)

    # Detect GNN keys present in any filter entry
    sample_res = next((results_dict[f] for f in outlier_filters if f in results_dict), None)
    if not sample_res:
        return base

    gnn_methods = []
    for m, (pk, _, _) in GNN_MODEL_KEY_MAP.items():
        if pk in sample_res:
            gnn_methods.append((GNN_MODEL_PRINT_MAP[m], pk))
    if not gnn_methods:
        return base

    # Console leaderboard for the GNN rows (mirrors the print block in plot_ml_results)
    print(f"\n  🧬 GNN Leaderboard for {mode_name}: {dataset_name}")
    print("  " + "-" * 105)
    gnn_rows = []
    for title, m_key in gnn_methods:
        for f in outlier_filters:
            if f not in results_dict:
                continue
            res = results_dict[f]
            if m_key not in res:
                continue
            fold_accs = [accuracy_score(yt, yp) * 100 for yt, yp in zip(res["y_trues_"], res[m_key])]
            if fold_accs:
                filt = str(f) if f is not None else "Baseline (None)"
                gnn_rows.append((np.mean(fold_accs), np.std(fold_accs), dataset_name, mode_name, title, filt))
    gnn_rows.sort(key=lambda x: x[0], reverse=True)
    for i, (acc, std, d, mo, meth, filt) in enumerate(gnn_rows):
        print(f"  {i+1:2d}. {acc:6.2f}% ± {std:5.2f}% | Data: {d[:15]:<15} | Model: {meth[:20]:<20} | Filter: {filt[:30]}")
    print("  " + "-" * 105 + "\n")

    return (base or []) + gnn_rows


# ============================================================
# HELPERS (identical to 03)
# ============================================================
def get_exp_paths(exp_folder):
    return sorted([
        Path(exp_folder, name)
        for name in os.listdir(exp_folder)
        if (os.path.isdir(os.path.join(exp_folder, name)) and name not in config.EXCLUDED_FOLDERS)
    ])


def load_training_data(exp_path):
    data_path = os.path.join(exp_path, config.TRAINING_DATA_PATH)
    if not os.path.exists(data_path):
        print(f"  -> Skipping {exp_path.name}: '{data_path}' not found.")
        sys.exit(0)
    return joblib.load(data_path)


def filter_datasets(dataset_name, dataset, kinetic_features):
    filtered_names, filtered_dataset, filtered_features = [], [], []
    for name, data, features in zip(dataset_name, dataset, kinetic_features):
        if not name.startswith("avg_") and not name.startswith("original_fitted_stretched"):
            filtered_names.append(name)
            filtered_dataset.append(data)
            filtered_features.append(features)
    return filtered_names, filtered_dataset, filtered_features


def load_or_init_results(results_file_path):
    if os.path.exists(results_file_path):
        return joblib.load(results_file_path)
    return {}


def make_checkpoint_fn(all_ml_results, results_file_path, clean_title, mode_key):
    def _checkpoint(updated_results):
        all_ml_results[clean_title][mode_key] = updated_results
        joblib.dump(all_ml_results, results_file_path, compress=3)
    return _checkpoint


# ============================================================
# MODULE 1: TEMPORAL ENCODER (mirrors 03 BiGRU-CNN dual)
# ============================================================
class TemporalEncoder(nn.Module):
    """1D encoder: CNN branch + BiGRU branch → fused (N, latent_dim)."""

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        half, quar = latent_dim // 2, latent_dim // 4

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.cnn_proj = nn.Linear(64, half)

        self.bigru = nn.GRU(input_size=1, hidden_size=quar, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=DROPOUT)
        self.gru_proj = nn.Linear(half, half)

        self.fusion = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.LayerNorm(latent_dim), nn.ReLU(),
        )

    def forward(self, x):                      # x: (N, T)
        cnn_out = F.relu(self.cnn_proj(self.cnn(x.unsqueeze(1))))      # (N, half)
        _, h_n = self.bigru(x.unsqueeze(-1))
        gru_out = F.relu(self.gru_proj(torch.cat([h_n[-2], h_n[-1]], dim=-1)))  # (N, half)
        return self.fusion(torch.cat([cnn_out, gru_out], dim=-1))     # (N, latent_dim)


# ============================================================
# MODULE 2: GNN HEAD + END-TO-END MODEL
# ============================================================
class SpatialGNNHead(nn.Module):
    """Two-layer GAT/GCN head → (N, num_classes) logits."""

    def __init__(self, in_features, hidden_dim, num_classes,
                 num_layers=GNN_LAYERS, dropout=DROPOUT, conv_type="gat"):
        super().__init__()
        assert HAS_PYG, "torch_geometric required"
        self.dropout = dropout
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        dims = [in_features] + [hidden_dim] * num_layers
        for i in range(num_layers):
            in_d, out_d = dims[i], dims[i + 1]
            is_last = (i == num_layers - 1)
            if conv_type == "gat":
                self.convs.append(GATConv(
                    in_d, out_d // 4 if not is_last else out_d,
                    heads=4 if not is_last else 1, concat=not is_last, dropout=dropout))
            else:
                self.convs.append(GCNConv(in_d, out_d))
            self.norms.append(nn.LayerNorm(out_d))
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x = F.dropout(F.relu(norm(conv(x, edge_index))), p=self.dropout, training=self.training)
        return self.classifier(x)


class EndToEndSpatialGNN(nn.Module):
    """Encoder + GNN in one module (gradients flow into the encoder)."""

    def __init__(self, num_classes, latent_dim=LATENT_DIM,
                 gnn_hidden=GNN_HIDDEN_DIM, gnn_layers=GNN_LAYERS, conv_type="gat"):
        super().__init__()
        self.encoder = TemporalEncoder(latent_dim)
        self.gnn = SpatialGNNHead(latent_dim, gnn_hidden, num_classes, gnn_layers, conv_type=conv_type)

    def forward(self, curves, edge_index):
        return self.gnn(self.encoder(curves), edge_index)


# ============================================================
# MODULE 3: GRAPH CONSTRUCTION
# ============================================================
def build_knn_edge_index(coords, k=KNN_K):
    """Symmetric kNN edge index (2, E) from (N, 2) coords."""
    n = len(coords)
    k_actual = min(k, n - 1)
    nbrs = NearestNeighbors(n_neighbors=k_actual + 1).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    src, dst = [], []
    for i, row in enumerate(indices):
        for j in row[1:]:
            src += [i, j]; dst += [j, i]      # undirected
    return torch.unique(torch.tensor([src, dst], dtype=torch.long), dim=1)


def build_subset_graphs(curves_subset, coords_subset, well_ids_subset, y_subset, k=KNN_K):
    """
    Build per-well graph dicts from an already-masked subset.
    Each: {curves (N,T), edge_index (2,E), y (N,), global_idx (N,)}.
    global_idx maps each node back to its row in the subset.
    """
    graphs = []
    for well in np.unique(well_ids_subset):
        wm = (well_ids_subset == well)
        if int(np.sum(wm)) < k + 1:
            continue
        graphs.append({
            "curves": torch.tensor(curves_subset[wm], dtype=torch.float32),
            "edge_index": build_knn_edge_index(coords_subset[wm], k),
            "y": torch.tensor(y_subset[wm], dtype=torch.long),
            "global_idx": np.where(wm)[0],
        })
    return graphs


# ============================================================
# MODULE 4: GNN TRAIN / PREDICT
# ============================================================
def _set_requires_grad(model, flag):
    for p in model.parameters():
        p.requires_grad = flag


def _train_step(parts, optimizer, criterion, graphs):
    for p in parts:
        p.train()
    total = 0.0
    for g in graphs:
        optimizer.zero_grad()
        x, ei, y = g["curves"].to(DEVICE), g["edge_index"].to(DEVICE), g["y"].to(DEVICE)
        logits = parts[0](x, ei) if len(parts) == 1 else parts[1](parts[0](x), ei)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p_ for m in parts for p_ in m.parameters()], 1.0)
        optimizer.step()
        total += loss.item()
    return total / max(len(graphs), 1)


@torch.no_grad()
def _predict_graphs(parts, graphs, n_classes):
    """Predict over graphs; return (preds, probs, global_idx) concatenated."""
    for p in parts:
        p.eval()
    preds, probs, idxs = [], [], []
    for g in graphs:
        x, ei = g["curves"].to(DEVICE), g["edge_index"].to(DEVICE)
        logits = parts[0](x, ei) if len(parts) == 1 else parts[1](parts[0](x), ei)
        prob = F.softmax(logits, dim=-1).cpu().numpy()
        probs.append(prob); preds.append(prob.argmax(axis=1)); idxs.append(g["global_idx"])
    if not preds:
        return np.array([]), np.zeros((0, n_classes)), np.array([], dtype=int)
    return np.concatenate(preds), np.concatenate(probs), np.concatenate(idxs)


def train_gnn_2stage(train_graphs, n_classes, pretrained_encoder=None):
    """Model A: freeze encoder → train GNN → fine-tune end-to-end."""
    encoder = TemporalEncoder(LATENT_DIM).to(DEVICE)
    gnn = SpatialGNNHead(LATENT_DIM, GNN_HIDDEN_DIM, n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    if pretrained_encoder is not None:
        encoder.load_state_dict(pretrained_encoder, strict=False)

    # Stage 2: GNN only
    _set_requires_grad(encoder, False)
    _set_requires_grad(gnn, True)
    opt = Adam(gnn.parameters(), lr=LR_STAGE2, weight_decay=1e-4)
    sch = CosineAnnealingLR(opt, T_max=STAGE2_EPOCHS, eta_min=1e-5)
    for _ in range(STAGE2_EPOCHS):
        _train_step((encoder, gnn), opt, criterion, train_graphs)
        sch.step()

    # Stage 3: end-to-end fine-tune (tiny encoder LR)
    _set_requires_grad(encoder, True)
    opt = Adam([
        {"params": encoder.parameters(), "lr": LR_STAGE3_ENC},
        {"params": gnn.parameters(), "lr": LR_STAGE3_GNN},
    ], weight_decay=1e-4)
    sch = CosineAnnealingLR(opt, T_max=STAGE3_EPOCHS, eta_min=1e-6)
    for _ in range(STAGE3_EPOCHS):
        _train_step((encoder, gnn), opt, criterion, train_graphs)
        sch.step()

    return (encoder, gnn)


def train_gnn_e2e(train_graphs, n_classes):
    """Model B: end-to-end from scratch."""
    model = EndToEndSpatialGNN(n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    opt = Adam(model.parameters(), lr=LR_E2E, weight_decay=1e-4)
    sch = CosineAnnealingLR(opt, T_max=E2E_EPOCHS, eta_min=1e-6)
    for _ in range(E2E_EPOCHS):
        _train_step((model,), opt, criterion, train_graphs)
        sch.step()
    return (model,)


# ============================================================
# MODULE 5: EVALUATE GNN ACROSS OUTLIER FILTERS
#           (mirrors model_utils.evaluate_outlier_filters)
# ============================================================
def evaluate_gnn_outlier_filters(
    X_curves, features_df, y_encoded, coords, well_ids,
    outlier_filters, dataset_name, mode_name,
    cached_results=None, gnn_models=("gnn_2stage", "gnn_e2e"),
    n_splits=1, checkpoint_fn=None, k=KNN_K, pretrained_encoder=None, rerun_models=(),
):
    """
    Same contract & res_entry structure as model_utils.evaluate_outlier_filters,
    but trains the GNN variants.

    Extra inputs:
        coords    : (N, 2)  pixel [row, col] per sample
        well_ids  : (N,)    well id per sample (graphs are per-well)

    Strategy (inductive to avoid label leakage):
        Train graphs are built from TRAIN nodes only. Prediction graphs are
        built from ALL nodes (so test nodes have full neighbourhoods), then we
        slice out test rows to match res_entry["y_trues_"].
    """
    results_dict = cached_results.copy() if cached_results is not None else {}
    total_filters = len(outlier_filters)

    for idx, f in enumerate(outlier_filters):
        filter_name = f if f else "None (Baseline)"
        pct = ((idx + 1) / total_filters) * 100
        print(f"  -> Testing Filter [{idx+1}/{total_filters} | {pct:.1f}%]: {filter_name}")

        res_entry = results_dict.get(f, {})

        # --- Mask (identical logic to the 1D path) ---
        if f is None:
            mask = np.ones(len(y_encoded), dtype=bool)
        elif f in features_df.columns:
            mask = (features_df[f] == 1).fillna(False).values
        else:
            print(f"     [Warning] {f} not found. Skipping.")
            continue

        X_AC = np.nan_to_num(X_curves[mask], nan=0.0, posinf=0.0, neginf=0.0)
        y_true_all = y_encoded[mask]
        coords_m = coords[mask]
        wells_m = well_ids[mask]

        # Drop rare classes (<2), same as 1D path
        uniq, counts = np.unique(y_true_all, return_counts=True)
        rare = uniq[counts < 2]
        if len(rare) > 0:
            keep = ~np.isin(y_true_all, rare)
            X_AC, y_true_all = X_AC[keep], y_true_all[keep]
            coords_m, wells_m = coords_m[keep], wells_m[keep]

        n_classes = len(np.unique(y_true_all))
        if n_classes < 2 or len(y_true_all) < 2 * n_classes:
            print(f"     [Warning] Insufficient classes/samples. Skipping.")
            continue

        # --- Splits (identical to 1D path) ---
        test_size = max(int(len(y_true_all) * 0.10), n_classes)
        if n_splits == 1:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
        else:
            min_cc = np.min(counts[~np.isin(uniq, rare)])
            splitter = StratifiedKFold(n_splits=min(n_splits, int(min_cc)), shuffle=True, random_state=0)
        splits = list(splitter.split(X_AC, y_true_all))

        if "y_trues_" not in res_entry:
            res_entry["y_trues_"] = [y_true_all[test_idx] for _, test_idx in splits]
            res_entry["mask_count"] = int(np.sum(mask))

        # --- Train each GNN variant ---
        for m in gnn_models:
            preds_key, probs_key, classes_key = GNN_MODEL_KEY_MAP[m]
            print_name = f"{GNN_MODEL_PRINT_MAP[m]:<11}"

            if (preds_key in res_entry) and (m not in rerun_models):
                fold_accs = [accuracy_score(yt, yp) for yt, yp in zip(res_entry["y_trues_"], res_entry[preds_key])]
                acc, std = np.mean(fold_accs) * 100, np.std(fold_accs) * 100
                print(f"     [CACHE HIT] {m.upper()} | {acc:5.2f}% ± {std:5.2f}%")
                continue

            preds, probs, classes_list = [], [], []
            t0 = time.perf_counter()

            for train_idx, test_idx in splits:
                train_mask = np.zeros(len(y_true_all), dtype=bool)
                train_mask[train_idx] = True

                # Prediction graphs: all nodes (full neighbourhoods for test nodes)
                graphs_all = build_subset_graphs(X_AC, coords_m, wells_m, y_true_all, k=k)
                # Training graphs: train nodes only (no test labels in loss)
                train_graphs = build_subset_graphs(
                    X_AC[train_mask], coords_m[train_mask], wells_m[train_mask], y_true_all[train_mask], k=k
                )

                if len(train_graphs) == 0 or len(graphs_all) == 0:
                    preds.append(np.zeros(len(test_idx), dtype=int))
                    probs.append(np.zeros((len(test_idx), n_classes)))
                    classes_list.append(np.unique(y_encoded))
                    continue

                if m == "gnn_2stage":
                    parts = train_gnn_2stage(train_graphs, n_classes, pretrained_encoder)
                else:
                    parts = train_gnn_e2e(train_graphs, n_classes)

                pred_all, prob_all, idx_all = _predict_graphs(parts, graphs_all, n_classes)

                # Scatter back to full-length arrays, then slice test rows
                pred_full = np.zeros(len(y_true_all), dtype=int)
                prob_full = np.zeros((len(y_true_all), n_classes))
                pred_full[idx_all] = pred_all
                prob_full[idx_all] = prob_all

                preds.append(pred_full[test_idx])
                probs.append(prob_full[test_idx])
                classes_list.append(np.unique(y_encoded))

                del parts
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            res_entry[preds_key] = preds
            res_entry[probs_key] = probs
            res_entry[classes_key] = classes_list
            results_dict[f] = res_entry
            if checkpoint_fn is not None:
                checkpoint_fn(results_dict)

            fold_accs = [accuracy_score(yt, yp) for yt, yp in zip(res_entry["y_trues_"], preds)]
            acc, std = np.mean(fold_accs) * 100, np.std(fold_accs) * 100
            dur = time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - t0)))
            print(f"     [+] {mode_name}-{dataset_name}-{filter_name[:30]} | {print_name} | {acc:5.2f}% ± {std:5.2f}% | Duration: {dur}")

        results_dict[f] = res_entry

    return results_dict


# ============================================================
# MODULE 6: PRETRAINED ENCODER EXPORT (Keras → PyTorch)
# ============================================================
def export_temporal_encoder_state(keras_model_path):
    """Best-effort CNN/GRU weight transfer from a 03 Keras model → encoder state."""
    try:
        import tensorflow as tf
    except ImportError:
        return None
    if not os.path.exists(keras_model_path):
        return None

    keras_model = tf.keras.models.load_model(keras_model_path)
    enc = TemporalEncoder(LATENT_DIM)
    state = enc.state_dict()

    transfer_map = {"conv1d": "cnn.0", "conv1d_1": "cnn.3", "bidirectional": "bigru"}
    transferred = 0
    for layer in keras_model.layers:
        prefix = next((v for kk, v in transfer_map.items() if kk in layer.name), None)
        if prefix is None:
            continue
        for w in layer.get_weights():
            wt = torch.tensor(np.array(w))
            if wt.ndim == 3:                 # Keras conv (k,in,out) → PyTorch (out,in,k)
                wt = wt.permute(2, 1, 0)
            for key, tensor in state.items():
                if key.startswith(prefix) and tensor.shape == wt.shape:
                    state[key] = wt
                    transferred += 1
                    break
    print(f"    -> Encoder export: {transferred} tensors from {os.path.basename(keras_model_path)}")
    return state if transferred > 0 else None


# ============================================================
# MAIN (mirrors 03)
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN Spatial Training Pipeline")
    parser.add_argument("--task_id", type=int, default=0, help="Array Job ID")
    parser.add_argument("--exp_folder", type=str, default=config.DEFAULT_EXP_FOLDER)
    parser.add_argument("--n_splits", type=int, default=1)
    parser.add_argument("--k", type=int, default=KNN_K, help="kNN neighbours per pixel")
    parser.add_argument("--gnn_models", type=str, default="gnn_2stage,gnn_e2e",
                        help="Comma-separated GNN variants: gnn_2stage, gnn_e2e")
    parser.add_argument("--modes", type=str, default="Reference",
                        help="Comma-separated modes: Native, Reference")
    args = parser.parse_args()

    if not HAS_PYG:
        print("[FATAL] torch_geometric not installed. pip install torch-geometric")
        sys.exit(1)

    gnn_models = tuple(m.strip() for m in args.gnn_models.split(",") if m.strip())
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    exp_paths = get_exp_paths(args.exp_folder)
    if args.task_id >= len(exp_paths):
        print(f"Task ID {args.task_id} out of bounds for {len(exp_paths)} folders. Exiting.")
        sys.exit(0)

    exp_path = exp_paths[args.task_id]
    print(f"\n\n{'#'*80}\nGNN SPATIAL TRAINING FOR: {exp_path.name}\n{'#'*80}")

    # --- SAME result file selection as 03 ---
    if args.n_splits > 1:
        results_file_path = os.path.join(exp_path, config.TRAINING_10FOLD_RESULT_PATH)
    else:
        results_file_path = os.path.join(exp_path, config.TRAINING_RESULT_PATH)
    model_plot_path = os.path.join(exp_path, "model_performance")
    os.makedirs(model_plot_path, exist_ok=True)

    # --- Load data (identical to 03) ---
    training_data = load_training_data(exp_path)
    dataset_name = training_data["dataset_name"]
    dataset = training_data["dataset"]
    kinetic_features = training_data["kinetic_features"]
    Y_well = training_data["Y_well"]

    # Spatial metadata (required for GNN; produced by 01, carried by 02)
    metadata_df = training_data.get("metadata_df", None)
    if metadata_df is None or "pixel_row_idx" not in getattr(metadata_df, "columns", []):
        print("[FATAL] metadata_df with pixel_row_idx/pixel_col_idx missing. Re-run 01/02.")
        sys.exit(1)
    pixel_row = metadata_df["pixel_row_idx"].values
    pixel_col = metadata_df["pixel_col_idx"].values
    well_ids = metadata_df["well_id"].values if "well_id" in metadata_df.columns else np.array(Y_well).copy()
    coords_full = np.stack([pixel_row.astype(float), pixel_col.astype(float)], axis=1)

    # Filter Clean data only (identical to 03)
    dataset_name, dataset, kinetic_features = filter_datasets(dataset_name, dataset, kinetic_features)

    # Label mapping (identical to 03)
    if hasattr(config, "LABEL_MAPPINGS") and exp_path.name in config.LABEL_MAPPINGS:
        print(f"  [*] Applying custom target label mapping for: {exp_path.name}")
        mapping = config.LABEL_MAPPINGS[exp_path.name]
        Y_well = [mapping.get(w, w) for w in Y_well]
    else:
        print(f"  [*] No custom mapping for {exp_path.name}. Retaining default well labels.")

    encoder_le = LabelEncoder()
    y_full = encoder_le.fit_transform(Y_well)

    # Same outlier_filters list as 03
    outlier_filters = [
        None,
        # 'msc_label_msc_linear_0.001',
        # 'amf_label_amf_important',
        # 'knn_top_0.95',
    ]
    print(f"[*] Found {len(outlier_filters)-1} Dynamic Outlier Filters to test.")

    all_ml_results = load_or_init_results(results_file_path)

    total_samples = len(y_full)
    trained_curve = dataset[0].copy()    # reference curve, same as 03

    # Optional pretrained encoder from 03's cnn_gru_dual model (Model A only)
    pretrained_state = None
    if "gnn_2stage" in gnn_models:
        for cand in ["cnn_gru_dual", "bigru"]:
            kp = os.path.join(exp_path, "model_interpretation", f"{cand}_None_model.keras")
            pretrained_state = export_temporal_encoder_state(kp)
            if pretrained_state is not None:
                break
        if pretrained_state is None:
            print("  [*] No pretrained encoder found — Model A starts encoder from scratch.")

    # Reuse cached reference baseline across datasets (mirrors 03)
    shared_ref_baseline = None
    for ct in all_ml_results:
        if "Reference" in all_ml_results[ct] and None in all_ml_results[ct]["Reference"]:
            shared_ref_baseline = all_ml_results[ct]["Reference"][None]
            break

    total_datasets = len(dataset_name)

    for idx, (name, features_df, curves_2d) in enumerate(zip(dataset_name, kinetic_features, dataset)):
        if name != "ori_curves":
            continue

        clean_title = name.replace("_", " ").title()
        pct = ((idx + 1) / total_datasets) * 100
        print(f"\n{'='*75}\n[{idx+1}/{total_datasets} | {pct:.1f}%] Processing Dataset: {clean_title}\n{'='*75}")

        if clean_title not in all_ml_results:
            all_ml_results[clean_title] = {}

        # Top-10 MI features (parity with 03; GNN doesn't consume KFS)
        # X_candidates = np.nan_to_num(features_df[config.LD_FEATURES].values, nan=0.0, posinf=0.0, neginf=0.0)
        # mi_scores = mutual_info_classif(X_candidates, y_full, random_state=0)
        # top_10 = [config.LD_FEATURES[i] for i in np.argsort(mi_scores)[-10:][::-1]]
        # print(f"  [*] Top 10 Features: {top_10}")

        # ---------------- NATIVE ----------------
        # if "Native" in modes:
        #     print(f"\n  [MODE] NATIVE TRAINING (GNN)")
        #     cached_native = all_ml_results[clean_title].get("Native", {})
        #     checkpoint_native = make_checkpoint_fn(all_ml_results, results_file_path, clean_title, "Native")

        #     res_native = evaluate_gnn_outlier_filters(
        #         X_curves=curves_2d, features_df=features_df, y_encoded=y_full,
        #         coords=coords_full, well_ids=well_ids,
        #         outlier_filters=outlier_filters, dataset_name=clean_title, mode_name="Native",
        #         cached_results=cached_native, gnn_models=gnn_models, n_splits=args.n_splits,
        #         checkpoint_fn=checkpoint_native, k=args.k, pretrained_encoder=pretrained_state,
        #         rerun_models=config.RERUN_MODELS,
        #     )
        #     all_ml_results[clean_title]["Native"] = res_native
        #     joblib.dump(all_ml_results, results_file_path, compress=3)

        #     prefix_native = os.path.join(model_plot_path, f"{name}_Native")
        #     plot_ml_results_with_gnn(res_native, outlier_filters, clean_title, "Native Training", total_samples, save_prefix=prefix_native)

        # ---------------- REFERENCE ----------------
        if "Reference" in modes:
            print(f"\n  [MODE] REFERENCE TRAINING (GNN)")
            cached_ref = all_ml_results[clean_title].get("Reference", {})
            if shared_ref_baseline is not None and None not in cached_ref:
                cached_ref[None] = shared_ref_baseline

            checkpoint_ref = make_checkpoint_fn(all_ml_results, results_file_path, clean_title, "Reference")

            res_ref = evaluate_gnn_outlier_filters(
                X_curves=trained_curve, features_df=features_df, y_encoded=y_full,
                coords=coords_full, well_ids=well_ids,
                outlier_filters=outlier_filters, dataset_name=clean_title, mode_name="Reference",
                cached_results=cached_ref, gnn_models=gnn_models, n_splits=args.n_splits,
                checkpoint_fn=checkpoint_ref, k=args.k, pretrained_encoder=pretrained_state,
                rerun_models=config.RERUN_MODELS,
            )
            if shared_ref_baseline is None and None in res_ref:
                shared_ref_baseline = res_ref[None]

            all_ml_results[clean_title]["Reference"] = res_ref
            joblib.dump(all_ml_results, results_file_path, compress=3)

            # prefix_ref = os.path.join(model_plot_path, f"{name}_Reference")
            # plot_ml_results_with_gnn(res_ref, outlier_filters, clean_title, "Reference Training", total_samples, save_prefix=prefix_ref)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n  [✓] GNN pipeline complete for {exp_path.name}")