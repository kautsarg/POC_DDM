"""
03b_gnn_spatial_training.py
============================
Spatial GNN training, ported from gk_code/legacy/04_par-gnn_spatial_training.py.

Retry of that model, scoped down per explicit request:
  - E2E only: the legacy script's "gnn_2stage" (pretrained-encoder + fine-tune) variant
    is dropped entirely -- this script trains the encoder+GNN jointly, from scratch,
    only. This also removes the pretrained-encoder plumbing/Keras-weight-export code
    and simplifies _train_step/_predict_graphs to operate on a single model instead of
    a (encoder, gnn) tuple pair. Two e2e variants are trained as separate leaderboard
    entries, distinguished only by the GNN head's conv type: "gnn_gat" (GATConv,
    attention-weighted neighbour aggregation) and "gnn_gcn" (GCNConv, the simpler
    fixed-weight convolution) -- see GNN_MODEL_NAMES/GNN_CONV_TYPE.
  - Native mode only: trains on each requested curve_type's OWN curves (mirroring
    03_main_training.py's Native mode) -- the legacy script's Reference mode (always
    training on ori_curves regardless of which curve_type is being reported on) is
    dropped, along with its shared_ref_baseline reuse logic.
  - Ported to the current curve_for_training.joblib format: the legacy script read a
    "metadata_df" key that doesn't exist anymore (it was never persisted directly --
    02_outlier_detection_pipeline.py derives metadata_df on demand from a "metadata"
    dict at load time, the same pattern reused here). Also switched from the legacy
    script's bespoke --modes/hardcoded "ori_curves"-only CLI to 03's current
    --curve_type/CURVE_TYPE_ALIASES convention, and added --force_rerun (the legacy
    script always loaded+merged into existing results, with no overwrite option).
    --force_rerun here only forces gnn_gat/gnn_gcn to recompute -- unlike 03's
    --force_rerun, it never wipes all_ml_results wholesale, since this file is shared
    with 03 and holds every other model's results for the same dataset too.
  - Optimization: added an early-stopping + validation-loss loop (held-out validation
    wells, built as separate small graphs via the same build_subset_graphs used for
    train/test -- inductive, not transductive label-masking within one graph, which
    would have been more "correct" GNN practice but meaningfully more complex to get
    right; this is an approximation good enough for "is training still improving").
    Mirrors the EarlyStopping/ReduceLROnPlateau pattern model_utils.py already uses
    for the Keras models, ported to PyTorch -- previously this ran a fixed
    E2E_EPOCHS regardless of convergence.

Result integration (identical structure to 03 -- same result file, same nested
format, so 06/06b/08's reports pick these models up automatically once added to
config.MODEL_KEY_MAP/MODEL_PRINT_MAP):
    all_ml_results[clean_title]["Native"][filter_f] = res_entry
res_entry gains, per variant:
    y_preds_AC_gnn_gat_ / y_probs_AC_gnn_gat_ / classes_AC_gnn_gat_
    y_preds_AC_gnn_gcn_ / y_probs_AC_gnn_gcn_ / classes_AC_gnn_gcn_

Same result file selection as 03:
    n_splits > 1 -> config.TRAINING_10FOLD_RESULT_PATH
    else         -> config.TRAINING_RESULT_PATH

07's XAI/saliency pipeline is TensorFlow/Keras-specific (gradient extraction via
tf.GradientTape against a loaded .keras model) and out of scope here -- this script
doesn't save a model file for 07 to find, so neither variant will appear there. They
DO appear in 06/06b's prediction reports and 08's statistical comparison once wired
into config.MODEL_KEY_MAP, since those only read the predictions already in the
results joblib.

Usage (array job, one folder per task -- identical to 03):
    python 03b_gnn_spatial_training.py --task_id 0 --exp_folder /data/exps
    python 03b_gnn_spatial_training.py --task_id 0 --n_splits 10
    python 03b_gnn_spatial_training.py --task_id 0 --curve_type ori_curve
    python 03b_gnn_spatial_training.py --task_id 0 --force_rerun
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
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, 'utils/model_training')
import config
from model_utils import plot_ml_results, set_global_determinism

# --- PyTorch / PyG (only needed for the GNN itself) ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    from torch_geometric.nn import GCNConv, GATConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

warnings.filterwarnings("ignore", category=UserWarning)
torch.manual_seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# HYPERPARAMETERS
# ============================================================
LATENT_DIM     = 64
GNN_HIDDEN_DIM = 128
GNN_LAYERS     = 2
KNN_K          = 24
DROPOUT        = 0.3

# Matched to model_utils.py's Keras models (e.g. cnn_gru_dual, the gated dual-branch
# models, lstm_ae_clf): epochs=500, 10% stratified validation split, EarlyStopping
# patience=100 with restore_best_weights, ReduceLROnPlateau(factor=0.5, patience=30,
# min_lr=1e-5) -- patience > LR-reduce patience so the LR gets a chance to drop before
# training stops, same reasoning as model_utils.py's comment on this. LR itself
# matches cnn_gru_dual's Adam(1e-3) since this encoder is the same CNN+GRU shape.
E2E_EPOCHS         = 500    # max epochs; early stopping usually cuts this short
LR_E2E             = 1e-3
VAL_FRACTION       = 0.1    # held out from the train split, for early stopping only
PATIENCE           = 100    # epochs with no val-loss improvement before stopping
LR_REDUCE_FACTOR   = 0.5
LR_REDUCE_PATIENCE = 30
LR_MIN             = 1e-5

# Two e2e variants, distinguished only by the GNN head's conv type -- "e2e" is no
# longer part of the name (the legacy script's other variant, gnn_2stage, is gone, so
# there's nothing left to distinguish "e2e" from).
GNN_MODEL_NAMES = ("gnn_gat", "gnn_gcn")
GNN_CONV_TYPE = {"gnn_gat": "gat", "gnn_gcn": "gcn"}
GNN_MODEL_KEY_MAP = {
    "gnn_gat": ("y_preds_AC_gnn_gat_", "y_probs_AC_gnn_gat_", "classes_AC_gnn_gat_"),
    "gnn_gcn": ("y_preds_AC_gnn_gcn_", "y_probs_AC_gnn_gcn_", "classes_AC_gnn_gcn_"),
}
GNN_MODEL_PRINT_MAP = {"gnn_gat": "GNN (GAT)", "gnn_gcn": "GNN (GCN)"}


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
    """1D encoder: CNN branch + BiGRU branch -> fused (N, latent_dim)."""

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
    """Two-layer GAT/GCN head -> (N, num_classes) logits."""

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
# (single model, not a (encoder, gnn) tuple -- only e2e exists now)
# ============================================================
def _train_step(model, optimizer, criterion, graphs):
    """Accumulates gradients across ALL graphs before a single optimizer step per
    epoch -- NOT one step per graph. One graph is built per well (build_subset_graphs),
    and one well = one reaction = one label (see evaluate_gnn_outlier_filters' well_ids
    fallback to Y_well when there's no separate well_id column -- that's the normal
    case here, by design, not a data gap). A step-per-graph loop would therefore push
    the whole model toward that one well's single class every step, then immediately
    toward the next well's different class on the very next step -- closer to
    alternating single-class SGD across classes than real multi-class training, and
    in practice this was the reason the GNN variants underperformed every other model
    on real (single-class-per-well) data despite training fine on synthetic graphs
    that happened to mix multiple classes per graph. Accumulating first means each
    epoch's one step reflects a balanced gradient across every well/class at once.
    """
    model.train()
    optimizer.zero_grad()
    total = 0.0
    for g in graphs:
        x, ei, y = g["curves"].to(DEVICE), g["edge_index"].to(DEVICE), g["y"].to(DEVICE)
        logits = model(x, ei)
        loss = criterion(logits, y)
        loss.backward()
        total += loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return total / max(len(graphs), 1)


@torch.no_grad()
def _validate_step(model, criterion, graphs):
    model.eval()
    total, n = 0.0, 0
    for g in graphs:
        x, ei, y = g["curves"].to(DEVICE), g["edge_index"].to(DEVICE), g["y"].to(DEVICE)
        logits = model(x, ei)
        total += criterion(logits, y).item() * len(y)
        n += len(y)
    return total / max(n, 1)


@torch.no_grad()
def _predict_graphs(model, graphs, n_classes):
    """Predict over graphs; return (preds, probs, global_idx) concatenated."""
    model.eval()
    preds, probs, idxs = [], [], []
    for g in graphs:
        x, ei = g["curves"].to(DEVICE), g["edge_index"].to(DEVICE)
        logits = model(x, ei)
        prob = F.softmax(logits, dim=-1).cpu().numpy()
        probs.append(prob); preds.append(prob.argmax(axis=1)); idxs.append(g["global_idx"])
    if not preds:
        return np.array([]), np.zeros((0, n_classes)), np.array([], dtype=int)
    return np.concatenate(preds), np.concatenate(probs), np.concatenate(idxs)


def train_gnn_e2e(fit_graphs, val_graphs, n_classes, conv_type="gat",
                  max_epochs=E2E_EPOCHS, patience=PATIENCE):
    """End-to-end from scratch, with early stopping + LR scheduling on held-out
    validation wells (separate small graphs, see build_subset_graphs callers below --
    inductive approximation of "is training still improving", not a transductive
    in-graph label mask).

    conv_type: "gat" or "gcn" -- which GNN head SpatialGNNHead/EndToEndSpatialGNN uses.

    Mirrors model_utils.py's EarlyStopping(patience=100, restore_best_weights=True) +
    ReduceLROnPlateau(factor=0.5, patience=30, min_lr=1e-5) exactly, including its
    "no val split -> no callbacks at all" fallback: with no validation graphs (e.g.
    too few wells/samples to split one off), this runs the full max_epochs at a
    constant LR with no early stopping, same as model_utils.py's Keras branches do
    when _val_split_ok is False.
    """
    model = EndToEndSpatialGNN(n_classes, conv_type=conv_type).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    opt = Adam(model.parameters(), lr=LR_E2E, weight_decay=1e-4)

    if not val_graphs:
        for _ in range(max_epochs):
            _train_step(model, opt, criterion, fit_graphs)
        return model

    sch = ReduceLROnPlateau(opt, mode="min", factor=LR_REDUCE_FACTOR,
                            patience=LR_REDUCE_PATIENCE, min_lr=LR_MIN)
    best_val_loss, best_state, epochs_no_improve = float("inf"), None, 0

    for _ in range(max_epochs):
        _train_step(model, opt, criterion, fit_graphs)
        val_loss = _validate_step(model, criterion, val_graphs)
        sch.step(val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ============================================================
# MODULE 5: EVALUATE GNN ACROSS OUTLIER FILTERS
#           (mirrors model_utils.evaluate_outlier_filters)
# ============================================================
def evaluate_gnn_outlier_filters(
    X_curves, features_df, y_encoded, coords, well_ids,
    outlier_filters, dataset_name, mode_name,
    cached_results=None, n_splits=1, checkpoint_fn=None, k=KNN_K, rerun_models=(),
    max_epochs=E2E_EPOCHS, patience=PATIENCE,
):
    """
    Same contract & res_entry structure as model_utils.evaluate_outlier_filters,
    but trains the e2e spatial GNN.

    Extra inputs:
        coords    : (N, 2)  pixel [row, col] per sample
        well_ids  : (N,)    well id per sample (graphs are per-well)

    Strategy (inductive to avoid label leakage):
        Train graphs are built from TRAIN nodes only (further split into fit/val for
        early stopping). Prediction graphs are built from ALL nodes (so test nodes
        have full neighbourhoods), then we slice out test rows to match
        res_entry["y_trues_"].
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

        # Two independent models (GAT head, GCN head) -- each gets its own cache-hit
        # check/checkpoint, so a crash partway through doesn't lose the other one's
        # already-finished folds, and re-running with one in rerun_models doesn't
        # force-retrain the other.
        for m in GNN_MODEL_NAMES:
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
                # Further split train_idx into fit/val (for early stopping only -- val
                # nodes never appear in graphs_all's loss computation, just here).
                # Same pattern as model_utils.py's _val_split_ok block: just try the
                # stratified split and fall back to no validation split on ValueError
                # (raised when a class is too sparse in this fold to stratify), rather
                # than a separate manual pre-check.
                fit_idx, val_idx = train_idx, np.array([], dtype=int)
                y_train = y_true_all[train_idx]
                try:
                    fit_pos, val_pos = train_test_split(
                        np.arange(len(train_idx)), test_size=VAL_FRACTION,
                        stratify=y_train, random_state=0)
                    fit_idx, val_idx = train_idx[fit_pos], train_idx[val_pos]
                except ValueError:
                    pass

                fit_mask = np.zeros(len(y_true_all), dtype=bool)
                fit_mask[fit_idx] = True
                val_mask = np.zeros(len(y_true_all), dtype=bool)
                val_mask[val_idx] = True

                # Prediction graphs: all nodes (full neighbourhoods for test nodes)
                graphs_all = build_subset_graphs(X_AC, coords_m, wells_m, y_true_all, k=k)
                # Fit graphs: fit nodes only (no val/test labels in the training loss)
                fit_graphs = build_subset_graphs(
                    X_AC[fit_mask], coords_m[fit_mask], wells_m[fit_mask], y_true_all[fit_mask], k=k)
                # Val graphs: held out from fit, used only to decide when to stop
                val_graphs = build_subset_graphs(
                    X_AC[val_mask], coords_m[val_mask], wells_m[val_mask], y_true_all[val_mask], k=k
                ) if val_idx.size > 0 else []

                if len(fit_graphs) == 0 or len(graphs_all) == 0:
                    preds.append(np.zeros(len(test_idx), dtype=int))
                    probs.append(np.zeros((len(test_idx), n_classes)))
                    classes_list.append(np.unique(y_encoded))
                    continue

                model = train_gnn_e2e(fit_graphs, val_graphs, n_classes, conv_type=GNN_CONV_TYPE[m],
                                      max_epochs=max_epochs, patience=patience)
                pred_all, prob_all, idx_all = _predict_graphs(model, graphs_all, n_classes)

                # Scatter back to full-length arrays, then slice test rows
                pred_full = np.zeros(len(y_true_all), dtype=int)
                prob_full = np.zeros((len(y_true_all), n_classes))
                pred_full[idx_all] = pred_all
                prob_full[idx_all] = prob_all

                preds.append(pred_full[test_idx])
                probs.append(prob_full[test_idx])
                classes_list.append(np.unique(y_encoded))

                del model
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
# MAIN (mirrors 03's current CLI/loop conventions)
# ============================================================
if __name__ == "__main__":
    print(f"\n{'='*70}\n[RUNNING] {os.path.basename(__file__)}\n{'='*70}\n")
    parser = argparse.ArgumentParser(description="GNN Spatial Training Pipeline (e2e, Native mode only)")
    parser.add_argument("--task_id", type=int, default=0, help="Array Job ID")
    parser.add_argument("--exp_folder", type=str, default=config.DEFAULT_EXP_FOLDER)
    parser.add_argument("--n_splits", type=int, default=1)
    parser.add_argument("--k", type=int, default=KNN_K, help="kNN neighbours per pixel")
    parser.add_argument("--curve_type", type=str, nargs="+", default=["ori_curve", "ori_curve_avg"],
                        help="Which curve variant(s) to train on (e.g. 'ori_curve' 'ori_curve_avg')")
    parser.add_argument("--force_rerun", action="store_true",
                        help="Recompute and overwrite even if presaved results already exist")
    parser.add_argument("--max_epochs", type=int, default=E2E_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE,
                        help="Early-stopping patience (epochs with no val-loss improvement)")
    args = parser.parse_args()

    if not HAS_PYG:
        print("[FATAL] torch_geometric not installed. pip install torch-geometric")
        sys.exit(1)

    set_global_determinism(0)
    torch.manual_seed(0)

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

    # --- Load data (current curve_for_training.joblib format) ---
    training_data = load_training_data(exp_path)
    dataset_name = training_data["dataset_name"]
    dataset = training_data["dataset"]
    kinetic_features = training_data["kinetic_features"]
    Y_well = training_data["Y_well"]

    # Spatial metadata: "metadata" is a dict/records structure, NOT a DataFrame --
    # 02_outlier_detection_pipeline.py derives metadata_df from it the same way on
    # demand rather than persisting the DataFrame itself (see its line ~280/540).
    if "metadata" not in training_data:
        print(f"  -> Skipping {exp_path.name}: no 'metadata' in {config.TRAINING_DATA_PATH}. Re-run 01/02.")
        sys.exit(0)
    metadata_df = pd.DataFrame(training_data["metadata"])
    if not {"pixel_row_idx", "pixel_col_idx"}.issubset(metadata_df.columns):
        print(f"  -> Skipping {exp_path.name}: no pixel_row_idx/pixel_col_idx in metadata. Re-run 01/02.")
        sys.exit(0)
    pixel_row = metadata_df["pixel_row_idx"].values
    pixel_col = metadata_df["pixel_col_idx"].values
    # well_id is computed from the RAW (pre label-mapping) Y_well -- this is the
    # physical/spatial grouping for graph topology, deliberately independent of
    # however config.LABEL_MAPPINGS later buckets labels for the classification
    # target itself (e.g. merging several raw classes into one).
    well_ids = metadata_df["well_id"].values if "well_id" in metadata_df.columns else np.array(Y_well).copy()
    coords_full = np.stack([pixel_row.astype(float), pixel_col.astype(float)], axis=1)

    # Filter Clean data only (identical to 03)
    dataset_name, dataset, kinetic_features = filter_datasets(dataset_name, dataset, kinetic_features)

    # Label mapping (identical to 03)
    label_mappings = config.get_label_mappings(exp_path)
    if exp_path.name in label_mappings:
        print(f"  [*] Applying custom target label mapping for: {exp_path.name}")
        mapping = label_mappings[exp_path.name]
        Y_well = [mapping.get(w, w) for w in Y_well]
    else:
        print(f"  [*] No custom mapping for {exp_path.name}. Retaining default well labels.")

    encoder_le = LabelEncoder()
    y_full = encoder_le.fit_transform(Y_well)

    # Same curated outlier_filters list 03/04 currently use.
    outlier_filters = [None, 'lstm_ae_glb_ds1_label_elbow', 'spatial_knn_label_elbow', 'spatial_grid_label_elbow']
    print(f"[*] Found {len(outlier_filters)-1} Dynamic Outlier Filters to test.")

    # Unlike 03_main_training.py's --force_rerun (which owns every model in this file and
    # can safely wipe all_ml_results wholesale), 03b shares this same results file with 03
    # -- it only owns the gnn_gat/gnn_gcn entries within it. Wiping the dict here would also
    # discard every other already-trained model's results for this dataset. So --force_rerun
    # always loads the existing dict and instead forces just the GNN models' cache-hit check
    # to bypass (evaluate_gnn_outlier_filters' rerun_models param already does this per-model,
    # only overwriting that model's keys in res_entry -- see GNN_MODEL_KEY_MAP usage there).
    all_ml_results = load_or_init_results(results_file_path)
    rerun_models = set(config.RERUN_MODELS)
    if args.force_rerun:
        print(f"  -> [FORCE RERUN] Forcing GNN models ({', '.join(GNN_MODEL_NAMES)}) to recompute; "
              f"other models' cached results in {results_file_path} are left untouched.")
        rerun_models |= set(GNN_MODEL_NAMES)

    total_samples = len(y_full)
    total_datasets = len(dataset_name)

    # Dataset names that correspond to the requested curve_types (same alias
    # resolution 03_main_training.py uses).
    _target_names = {config.CURVE_TYPE_ALIASES.get(ct, ct) for ct in args.curve_type}

    for idx, (name, features_df, curves_2d) in enumerate(zip(dataset_name, kinetic_features, dataset)):
        if name not in _target_names:
            continue

        clean_title = name.replace("_", " ").title()
        pct = ((idx + 1) / total_datasets) * 100
        print(f"\n{'='*75}\n[{idx+1}/{total_datasets} | {pct:.1f}%] Processing Dataset: {clean_title}\n{'='*75}")

        if clean_title not in all_ml_results:
            all_ml_results[clean_title] = {}

        print(f"\n  [MODE] NATIVE TRAINING (GNN, e2e only)")
        cached_native = all_ml_results[clean_title].get("Native", {})
        checkpoint_native = make_checkpoint_fn(all_ml_results, results_file_path, clean_title, "Native")

        res_native = evaluate_gnn_outlier_filters(
            X_curves=curves_2d, features_df=features_df, y_encoded=y_full,
            coords=coords_full, well_ids=well_ids,
            outlier_filters=outlier_filters, dataset_name=clean_title, mode_name="Native",
            cached_results=cached_native, n_splits=args.n_splits,
            checkpoint_fn=checkpoint_native, k=args.k, rerun_models=rerun_models,
            max_epochs=args.max_epochs, patience=args.patience,
        )
        all_ml_results[clean_title]["Native"] = res_native
        joblib.dump(all_ml_results, results_file_path, compress=3)

        prefix_native = os.path.join(model_plot_path, f"{name}_Native")
        plot_ml_results(
            results_dict=all_ml_results[clean_title]["Native"],
            outlier_filters=outlier_filters,
            dataset_name=clean_title,
            mode_name="Native Training",
            total_count=total_samples,
            save_prefix=prefix_native,
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n  [✓] GNN pipeline complete for {exp_path.name}")
