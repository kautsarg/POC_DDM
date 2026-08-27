import os
# ====================================================================
# SUPPRESS TENSORFLOW C++ WARNINGS (Must be before TF import)
# ====================================================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=INFO, 1=WARN, 2=ERROR, 3=FATAL

import gc
import time
import hashlib
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    StratifiedShuffleSplit, StratifiedKFold, train_test_split
)
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from safe_io import safe_keras_save
import tensorflow as tf
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
tf.get_logger().setLevel('ERROR')

from scikeras.wrappers import KerasClassifier

import model_utils_gated
from model_utils_supcon import (
    create_cnn_gru_dual_supcon_model,
    create_cnn_gru_dual_attn_recon_supcon_model,
    create_cnn_gru_dual_supcon3_model,
    create_cnn_gru_dual_attn_recon_supcon3_model,
    SUPCON_MODEL_KEYS, BRANCH_SUPCON3_MODEL_KEYS,
)
from model_utils_dann import (
    DANN_MODEL_KEYS, GRLLambdaSchedule,
    create_cnn_gru_dual_dann_model, create_cnn_gru_dual_attn_recon_dann_model,
)

# ====================================================================
# GPU SETUP & VERIFICATION
# ====================================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"\n[*] SUCCESS: TensorFlow is utilizing the GPU -> {gpus[0].name}\n")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
else:
    print("\n[!] WARNING: No GPU found. TensorFlow will run on the CPU.")
    print("    Ensure you have installed: pip install tensorflow-macos tensorflow-metal\n")

# ====================================================================
# GLOBAL DETERMINISM SETUP
# ====================================================================
def set_global_determinism(seed=0, strict=True):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if strict:
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        try:
            tf.config.experimental.enable_op_determinism()
        except AttributeError:
            pass


# ====================================================================
# CURVE RESAMPLING (for combining datasets with different timestamp grids)
# ====================================================================
class CurveResampler:

    def __init__(self, t_grid):
        self.t_grid = np.asarray(t_grid, dtype=float)

    @classmethod
    def fit(cls, timestamps_list, n_points=None):
        """timestamps_list: list of 1D arrays, each starting at t=0 (i.e. t - t[0])."""
        duration = min(t[-1] for t in timestamps_list)
        if n_points is None:
            n_points = max(int(np.sum(t <= duration)) for t in timestamps_list)
        return cls(np.linspace(0, duration, n_points))

    def transform(self, t_raw, curves):
        """curves: (N, T) array sharing timestamps t_raw -> (N, len(t_grid))."""
        t_zeroed = np.asarray(t_raw, dtype=float) - t_raw[0]
        return np.array([np.interp(self.t_grid, t_zeroed, c) for c in curves])

# ====================================================================
# DUAL MODEL
# ====================================================================

def _build_cnn_gru_dual_branches(input_curve):
    # 1. Local Feature Branch (CNN)
    c = tf.keras.layers.Conv1D(16, 5, activation='relu')(input_curve)
    c = tf.keras.layers.Conv1D(8, 3, activation='relu')(c)
    c = tf.keras.layers.Flatten()(c)
    cnn_emb = tf.keras.layers.Dense(32, activation='relu')(c)

    # 2. Global Feature Branch (BiGRU)
    g = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, return_sequences=True))(input_curve)
    g = tf.keras.layers.LayerNormalization()(g)
    g = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(16))(g)
    g = tf.keras.layers.Dropout(0.2)(g)
    gru_emb = tf.keras.layers.Dense(32, activation='relu')(g)

    # 3. Fusion
    merged = tf.keras.layers.Concatenate()([cnn_emb, gru_emb])
    z = tf.keras.layers.Dense(64, activation='relu')(merged)
    z = tf.keras.layers.Dropout(0.2)(z)
    return z


def create_cnn_gru_dual_model(input_size_curve, output_size, inception_smoothing=False):
    """
    Dual-branch architecture combining CNN (Local) and BiGRU (Global)
    using only the raw curve as input.
    """
    input_curve = tf.keras.layers.Input(shape=(input_size_curve, 1), name="curve_input")
    x = model_utils_gated.inception_smoothing_block(input_curve) if inception_smoothing else input_curve
    z = _build_cnn_gru_dual_branches(x)
    outputs = tf.keras.layers.Dense(output_size, activation='softmax')(z)

    model = tf.keras.models.Model(inputs=input_curve, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# ====================================================================
# SPATIAL NEIGHBOR RECONSTRUCTION (cnn_gru_dual_attn_recon)
# ====================================================================


def build_neighbor_curve_stack(curves, coords, well_ids, k):
    n, t = curves.shape
    stack = np.empty((n, k + 1, t), dtype=curves.dtype)

    for well in np.unique(well_ids):
        well_idx = np.where(well_ids == well)[0]
        well_coords = coords[well_idx]
        well_curves = curves[well_idx]
        n_well = len(well_idx)

        k_actual = min(k, n_well - 1)
        if k_actual > 0:
            nbrs = NearestNeighbors(n_neighbors=k_actual + 1).fit(well_coords)
            _, neighbor_pos = nbrs.kneighbors(well_coords)  # (n_well, k_actual+1), includes self

        for local_i, global_i in enumerate(well_idx):
            stack[global_i, 0] = well_curves[local_i]

            if k_actual > 0:
                chosen = neighbor_pos[local_i]
                chosen = chosen[chosen != local_i][:k]
            else:
                chosen = np.array([], dtype=int)

            if len(chosen) < k:
                if len(chosen) == 0:
                    chosen = np.full(k, local_i, dtype=int)          # well of size 1: repeat self
                else:
                    reps = int(np.ceil(k / len(chosen)))
                    chosen = np.tile(chosen, reps)[:k]

            stack[global_i, 1:] = well_curves[chosen]

    return stack


@tf.keras.utils.register_keras_serializable(package="model_utils")
class _QuerySlice(tf.keras.layers.Layer):
    """Extracts the first timestep (index 0) as the query: (N, k+1, D) → (N, 1, D)."""
    def call(self, x):
        return x[:, 0:1, :]


@tf.keras.utils.register_keras_serializable(package="model_utils")
class _AttnScores(tf.keras.layers.Layer):
    """Scaled dot-product scores: Q @ K^T / sqrt(d). Needs attn_dim in config for reload."""
    def __init__(self, attn_dim, **kwargs):
        super().__init__(**kwargs)
        self.attn_dim = attn_dim

    def call(self, inputs):
        q, k = inputs
        return tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(self.attn_dim, tf.float32))

    def get_config(self):
        return {**super().get_config(), "attn_dim": self.attn_dim}


@tf.keras.utils.register_keras_serializable(package="model_utils")
class _WeightedRecon(tf.keras.layers.Layer):
    """Attention-weighted reconstruction: weights @ stack → (N, 1, T)."""
    def call(self, inputs):
        weights, stack = inputs
        return tf.matmul(weights, stack)


def create_cnn_gru_dual_attn_recon_model(k_plus_1, input_size_curve, output_size, attn_dim=16):
    stack_input = tf.keras.layers.Input(shape=(k_plus_1, input_size_curve), name="neighbor_stack_input")

    per_curve_encoder = tf.keras.Sequential([
        tf.keras.layers.Reshape((input_size_curve, 1)),
        tf.keras.layers.Conv1D(16, 5, activation='relu', padding='same'),
        tf.keras.layers.Conv1D(8, 3, activation='relu', padding='same'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(attn_dim, activation='relu'),
    ], name="per_curve_encoder")
    embeddings = tf.keras.layers.TimeDistributed(per_curve_encoder)(stack_input)   # (N, k+1, attn_dim)

    query = _QuerySlice()(embeddings)                                              # (N, 1, attn_dim)
    scores = _AttnScores(attn_dim)([query, embeddings])                           # (N, 1, k+1)
    attn_weights = tf.keras.layers.Softmax(axis=-1, name="attn_weights")(scores)  # (N, 1, k+1)

    reconstructed = _WeightedRecon()([attn_weights, stack_input])                 # (N, 1, T)
    reconstructed = tf.keras.layers.Reshape((input_size_curve, 1))(reconstructed)  # (N, T, 1)

    z = _build_cnn_gru_dual_branches(reconstructed)
    outputs = tf.keras.layers.Dense(output_size, activation='softmax')(z)

    model = tf.keras.models.Model(inputs=stack_input, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# ====================================================================
# NEURAL NETWORK SETUP
# ====================================================================
class KerasModelWrapper(KerasClassifier):
    pass

# 1. 1D CNN
def create_cnn_model(input_size, output_size, kernel_size_1=5, kernel_size_2=3, inception_smoothing=False):
    inputs = tf.keras.layers.Input(shape=(input_size, 1))
    inp = model_utils_gated.inception_smoothing_block(inputs) if inception_smoothing else inputs
    x = tf.keras.layers.Conv1D(16, kernel_size_1, activation='relu')(inp)
    x = tf.keras.layers.Conv1D(8, kernel_size_2, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(output_size, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 2. GRU (Bidirectional + Gradient Clipping)
def create_gru_model(input_size, output_size, inception_smoothing=False):
    inputs = tf.keras.layers.Input(shape=(input_size, 1))
    inp = model_utils_gated.inception_smoothing_block(inputs) if inception_smoothing else inputs
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, return_sequences=True))(inp)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(16))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(output_size, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 3. Transformer
def create_transformer_model(input_size, output_size, head_size=32, num_heads=2, ff_dim=32, num_blocks=2, dropout=0.1, inception_smoothing=False):
    inputs = tf.keras.layers.Input(shape=(input_size, 1))
    inp = model_utils_gated.inception_smoothing_block(inputs) if inception_smoothing else inputs

    # Downsampling
    x = tf.keras.layers.Conv1D(filters=head_size, kernel_size=5, strides=2, padding="same", activation="relu")(inp)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, padding="same")(x)

    new_seq_len = x.shape[1]
    positions = tf.range(start=0, limit=x.shape[1], delta=1)
    pos_embedding = tf.keras.layers.Embedding(input_dim=new_seq_len, output_dim=head_size)(positions)
    x = x + pos_embedding

    for _ in range(num_blocks):
        attn_output = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
        attn_output = tf.keras.layers.Dropout(dropout)(attn_output)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

        ffn_output = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
        ffn_output = tf.keras.layers.Dropout(dropout)(ffn_output)
        ffn_output = tf.keras.layers.Dense(head_size)(ffn_output)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

    x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(output_size, activation="softmax")(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ====================================================================
# MODULE 1: MODEL EVALUATION FUNCTION (WITH PROBABILITIES)
# ====================================================================
def _remap_global_splits(global_splits, mask, valid_mask=None):
    kept_global = np.where(mask)[0]
    if valid_mask is not None:
        kept_global = kept_global[valid_mask]
    pos_lookup = {g: i for i, g in enumerate(kept_global)}

    remapped = []
    for train_idx, test_idx in global_splits:
        local_train = np.array([pos_lookup[g] for g in train_idx if g in pos_lookup], dtype=int)
        local_test = np.array([pos_lookup[g] for g in test_idx if g in pos_lookup], dtype=int)
        remapped.append((local_train, local_test))
    return remapped


def build_well_stratified_random_split(y, well_ids, test_size=0.1, random_state=0):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(np.zeros(len(y)), well_ids))
    return {"random_split": (train_idx, test_idx)}


def build_well_stratified_nfold_splits(y, well_ids, n_splits=5, random_state=0):
    """N-fold CV splits, stratified by well_id — see build_well_stratified_random_split."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return {f"fold_{i}": (tr, te)
            for i, (tr, te) in enumerate(skf.split(np.zeros(len(y)), well_ids))}


_XAI_SAVE_NAME = {
    'cnn': 'cnn',
    'gru': 'bigru',
    'transformer': 'transformer',
    'cnn_gru_dual': 'cnn_gru_dual',
    'cnn_gru_dual_attn_recon': 'cnn_gru_dual_attn_recon',
}
_XAI_SAVE_NAME.update({k: k for k in SUPCON_MODEL_KEYS})
_XAI_SAVE_NAME.update({k: k for k in BRANCH_SUPCON3_MODEL_KEYS})
_XAI_SAVE_NAME.update({k: k for k in DANN_MODEL_KEYS})


def evaluate_outlier_filters(
    X_curves, features_df, y_encoded, outlier_filters, dataset_name, mode_name,
    cached_results=None, models=["cnn_gru_dual"], n_splits=1,
    checkpoint_fn=None, cv_splits=None,
    save_model_dir=None, save_model_curve_type="ori_curve",
    coords=None, well_ids=None, k_neighbors=24,
    chip_id_encoded=None,
    batch_size=512,
):
    if save_model_dir is not None:
        Path(save_model_dir).mkdir(parents=True, exist_ok=True)

    _bs = batch_size
    results_dict = cached_results.copy() if cached_results is not None else {}
    total_filters = len(outlier_filters)
    models = [m.lower() for m in models]

    import config
    model_key_map  = config.MODEL_KEY_MAP
    model_print_map = config.MODEL_PRINT_MAP

    _SPATIAL_RECON_MODELS = (
        "cnn_gru_dual_attn_recon",
        "cnn_gru_dual_attn_recon_supcon",
        "cnn_gru_dual_attn_recon_supcon3",
        "cnn_gru_dual_attn_recon_dann",
    )

    for idx, f in enumerate(outlier_filters):
        filter_name = f if f else 'None (Baseline)'
        filter_pct = ((idx + 1) / total_filters) * 100
        print(f"  -> Testing Filter [{idx+1}/{total_filters} | {filter_pct:.1f}%]: {filter_name}")

        res_entry = results_dict.get(f, {})

        if f is None:
            mask = np.ones(len(y_encoded), dtype=bool)
        elif f in features_df.columns:
            mask = (features_df[f] == 1).fillna(False).values
        else:
            print(f"     [Warning] {f} not found in dataset. Skipping.")
            continue

        X_AC = np.nan_to_num(X_curves[mask], nan=0.0, posinf=0.0, neginf=0.0)
        y_true = y_encoded[mask]

        coords_m = coords[mask] if coords is not None else None
        well_ids_m = well_ids[mask] if well_ids is not None else None
        chip_id_m = chip_id_encoded[mask] if chip_id_encoded is not None else None

        unique_classes, class_counts = np.unique(y_true, return_counts=True)
        rare_classes = unique_classes[class_counts < 2]

        valid_class_mask = None
        if len(rare_classes) > 0:
            valid_class_mask = ~np.isin(y_true, rare_classes)
            X_AC = X_AC[valid_class_mask]
            y_true = y_true[valid_class_mask]
            if coords_m is not None:
                coords_m = coords_m[valid_class_mask]
                well_ids_m = well_ids_m[valid_class_mask]
            if chip_id_m is not None:
                chip_id_m = chip_id_m[valid_class_mask]

        n_classes = len(np.unique(y_true))

        if n_classes < 2 or len(y_true) < 2 * n_classes:
            print(f"     [Warning] Insufficient classes or samples. Skipping.")
            continue

        if cv_splits is not None:
            splits = _remap_global_splits(cv_splits, mask, valid_class_mask)
            splits = [(tr, te) for tr, te in splits if len(tr) > 0 and len(te) > 0]
            if not splits:
                print(f"     [Warning] No samples remain for this filter under the given CV splits. Skipping.")
                continue
        elif well_ids_m is not None:
            n_wells = len(np.unique(well_ids_m))
            calculated_test_size = max(len(y_true) * 0.10, n_wells) / len(y_true)
            if n_splits == 1:
                splits = list(build_well_stratified_random_split(
                    y_true, well_ids_m, test_size=calculated_test_size).values())
            else:
                actual_splits = min(n_splits, n_wells)
                splits = list(build_well_stratified_nfold_splits(
                    y_true, well_ids_m, n_splits=actual_splits).values())
        else:
            calculated_test_size = max(int(len(y_true) * 0.10), n_classes)

            if n_splits == 1:
                splitter = StratifiedShuffleSplit(n_splits=1, test_size=calculated_test_size, random_state=0)
            else:
                min_class_count = np.min(class_counts[~np.isin(unique_classes, rare_classes)])
                actual_splits = min(n_splits, min_class_count)
                splitter = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=0)

            splits = list(splitter.split(X_AC, y_true))

        current_mask_count = int(np.sum(mask))
        cached_mask_count = res_entry.get("mask_count")
        if cached_mask_count is not None and cached_mask_count != current_mask_count:
            print(f"     [Warning] Cached results for filter '{filter_name}' were built from "
                  f"{cached_mask_count} filtered samples; current data has {current_mask_count} "
                  f"(likely changed upstream). Discarding stale cache for this filter.")
            res_entry = {}

        _split_signature = tuple(hashlib.md5(np.sort(test_index).tobytes()).hexdigest() for _, test_index in splits)
        _had_cached_predictions = any(k.startswith('y_preds_') for k in res_entry)
        if res_entry.get("_split_signature") != _split_signature:
            if _had_cached_predictions:
                print(f"     [Warning] Cached results for filter '{filter_name}' were built from a "
                      f"different train/test split (splitter changed upstream, or this cache predates "
                      f"split tracking). Discarding stale cache for this filter.")
            res_entry = {}
        res_entry["_split_signature"] = _split_signature

        res_entry["y_trues_"] = [y_true[test_index] for _, test_index in splits]
        res_entry["well_ids_test_"] = ([well_ids_m[test_index] for _, test_index in splits]
                                        if well_ids_m is not None else None)

        res_entry["mask_count"] = current_mask_count
        res_entry["y_true_count"] = len(y_true)

        # --- Spatial neighbour reconstruction setup ---
        _wanted_recon = [m for m in models if m in _SPATIAL_RECON_MODELS]
        _recon_unavailable = bool(_wanted_recon) and (coords_m is None or well_ids_m is None)
        if _recon_unavailable:
            print(f"     [SKIP] {', '.join(_wanted_recon)}: no coords/well_ids provided "
                  f"(pass coords=/well_ids= to evaluate_outlier_filters). Skipping for this filter.")

        for m in models:
            tf.keras.backend.clear_session()
            gc.collect()
            if m not in model_key_map: continue
            _base_m = m

            if _base_m in _SPATIAL_RECON_MODELS and _recon_unavailable:
                continue 

            preds_key, probs_key, classes_key = model_key_map[m]
            print_name = f"{model_print_map[m]:<11}"

            # --- CHECK CACHE ---
            if preds_key in res_entry:
                _xai_file_missing = (
                    save_model_dir is not None and m in _XAI_SAVE_NAME
                    and not (Path(save_model_dir) / f"{_XAI_SAVE_NAME[m]}_{f}_{save_model_curve_type}_model.keras").exists()
                )
                if not _xai_file_missing:
                    fold_accs = [accuracy_score(yt, yp) for yt, yp in zip(res_entry["y_trues_"], res_entry[preds_key])]
                    acc = np.mean(fold_accs) * 100
                    std = np.std(fold_accs) * 100

                    print(f"     [CACHE HIT] {m.upper()} cached result found. Skipping training.")
                    print(f"     [+] {mode_name}-{dataset_name}-{filter_name[:30]} | {print_name} | {acc:5.2f}% ± {std:5.2f}% | Duration: Cached")
                    continue
                print(f"     [XAI-RETRAIN] {m.upper()} cached but .keras missing — retraining to save model.")

            # --- TRAIN NEW MODEL ---
            preds, probs, classes_list = [], [], []
            histories = []
            start_time = time.perf_counter()

            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                if _base_m in _SPATIAL_RECON_MODELS:
                    X_train_curve = build_neighbor_curve_stack(
                        X_AC[train_idx].astype(np.float32, copy=False),
                        coords_m[train_idx], well_ids_m[train_idx], k=k_neighbors)
                    X_test_curve = build_neighbor_curve_stack(
                        X_AC[test_idx].astype(np.float32, copy=False),
                        coords_m[test_idx], well_ids_m[test_idx], k=k_neighbors)
                else:
                    X_train_curve, X_test_curve = X_AC[train_idx], X_AC[test_idx]
                y_train = y_true[train_idx]

                _do_xai_save = (save_model_dir is not None and fold_idx == 0
                                and m in _XAI_SAVE_NAME)

                _val_split_ok = False
                if _base_m != "knn":
                    try:
                        _tr_sub, _val_sub = train_test_split(
                            np.arange(len(y_train)), test_size=0.1, stratify=y_train, random_state=0)
                        y_train_fit = y_train[_tr_sub]
                        y_val = y_train[_val_sub]
                        X_train_curve_fit = X_train_curve[_tr_sub]
                        X_val_curve = X_train_curve[_val_sub]
                        _val_split_ok = True
                    except ValueError:
                        pass
                if not _val_split_ok:
                    y_train_fit = y_train
                    X_train_curve_fit = X_train_curve

                _fit_callbacks = [
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=30, min_lr=1e-5),
                ] if _val_split_ok else []

                if _base_m == "cnn_gru_dual":
                    tf.keras.backend.clear_session()
                    model = create_cnn_gru_dual_model(X_train_curve.shape[1], n_classes)
                    epochs = 500

                    if _val_split_ok:
                        _hist = model.fit(X_train_curve_fit, y_train_fit,
                                 validation_data=(X_val_curve, y_val),
                                 epochs=epochs, batch_size=_bs, shuffle=True, verbose=0,
                                 callbacks=_fit_callbacks)
                        histories.append(_hist.history)
                    else:
                        _hist = model.fit(X_train_curve, y_train, epochs=epochs, batch_size=_bs, shuffle=True, verbose=0)
                        histories.append(_hist.history)

                    if _do_xai_save:
                        _xai_path = Path(save_model_dir) / f"{_XAI_SAVE_NAME[m]}_{f}_{save_model_curve_type}_model.keras"
                        safe_keras_save(model, _xai_path)
                        print(f"     [XAI] Saved {_XAI_SAVE_NAME[m]} -> {_xai_path}")

                    prob = model.predict(X_test_curve, verbose=0)
                    pred = np.argmax(prob, axis=1)
                    cls = np.unique(y_encoded)

                    preds.append(pred)
                    probs.append(prob)
                    classes_list.append(cls)

                elif _base_m == "cnn_gru_dual_attn_recon":
                    tf.keras.backend.clear_session()
                    model = create_cnn_gru_dual_attn_recon_model(
                        X_train_curve.shape[1], X_train_curve.shape[2], n_classes)
                    epochs = 500

                    if _val_split_ok:
                        _hist = model.fit(X_train_curve_fit, y_train_fit,
                                 validation_data=(X_val_curve, y_val),
                                 epochs=epochs, batch_size=_bs, shuffle=True, verbose=0,
                                 callbacks=_fit_callbacks)
                        histories.append(_hist.history)
                    else:
                        _hist = model.fit(X_train_curve, y_train, epochs=epochs, batch_size=_bs, shuffle=True, verbose=0)
                        histories.append(_hist.history)

                    if _do_xai_save:
                        _xai_path = Path(save_model_dir) / f"{m}_{f}_{save_model_curve_type}_model.keras"
                        safe_keras_save(model, _xai_path)
                        print(f"     [XAI] Saved {m} -> {_xai_path}")

                    prob = model.predict(X_test_curve, verbose=0)
                    pred = np.argmax(prob, axis=1)
                    cls = np.unique(y_encoded)

                    preds.append(pred)
                    probs.append(prob)
                    classes_list.append(cls)

                elif _base_m in DANN_MODEL_KEYS:
                    if chip_id_m is None:
                        print(f"     [SKIP] {m}: no chip_id_encoded provided.")
                        continue
                    tf.keras.backend.clear_session()
                    is_attn_recon = _base_m == 'cnn_gru_dual_attn_recon_dann'
                    n_chips = len(np.unique(chip_id_m))
                    chip_train_all = chip_id_m[train_idx]

                    if is_attn_recon:
                        k_plus_1, T = X_train_curve.shape[1], X_train_curve.shape[2]
                        model = create_cnn_gru_dual_attn_recon_dann_model(k_plus_1, T, n_classes, n_chips)
                    else:
                        T = X_train_curve.shape[1]
                        model = create_cnn_gru_dual_dann_model(T, n_classes, n_chips)
                    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
                                 metrics=['accuracy'], jit_compile=False)
                    epochs = 500

                    _dann_es = tf.keras.callbacks.EarlyStopping(monitor='val_cls_ce', mode='min', patience=100, restore_best_weights=True)
                    _dann_rlrp = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_cls_ce', mode='min', factor=0.5, patience=30, min_lr=1e-5)

                    if _val_split_ok:
                        chip_train_fit = chip_train_all[_tr_sub]
                        chip_val = chip_train_all[_val_sub]
                        steps_per_epoch = max(int(np.ceil(len(X_train_curve_fit) / _bs)), 1)
                        _grl_cb = GRLLambdaSchedule(model.grl, total_steps=epochs * steps_per_epoch)
                        _hist = model.fit(X_train_curve_fit,
                                 {'cls_out': y_train_fit, 'chip_out': chip_train_fit},
                                 validation_data=(X_val_curve, {'cls_out': y_val, 'chip_out': chip_val}),
                                 epochs=epochs, batch_size=_bs, shuffle=True, verbose=0,
                                 callbacks=[_dann_es, _dann_rlrp, _grl_cb])
                        histories.append(_hist.history)
                    else:
                        steps_per_epoch = max(int(np.ceil(len(X_train_curve) / _bs)), 1)
                        _grl_cb = GRLLambdaSchedule(model.grl, total_steps=epochs * steps_per_epoch)
                        _hist = model.fit(X_train_curve,
                                 {'cls_out': y_train, 'chip_out': chip_train_all},
                                 epochs=epochs, batch_size=_bs, shuffle=True, verbose=0,
                                 callbacks=[_grl_cb])
                        histories.append(_hist.history)

                    if _do_xai_save:
                        _xai_path = Path(save_model_dir) / f"{m}_{f}_{save_model_curve_type}_model.keras"
                        safe_keras_save(model, _xai_path)
                        print(f"     [XAI] Saved {m} -> {_xai_path}")

                    dann_outputs = model.predict(X_test_curve, verbose=0)
                    cls_prob = dann_outputs[0]
                    pred = np.argmax(cls_prob, axis=1)
                    cls = np.unique(y_encoded)

                    preds.append(pred)
                    probs.append(cls_prob)
                    classes_list.append(cls)

                    tf.keras.backend.clear_session()

                elif _base_m in SUPCON_MODEL_KEYS:
                    tf.keras.backend.clear_session()
                    T = X_train_curve.shape[1]
                    if _base_m == 'cnn_gru_dual_supcon':
                        model = create_cnn_gru_dual_supcon_model(T, n_classes); epochs = 500
                    elif _base_m == 'cnn_gru_dual_attn_recon_supcon':
                        model = create_cnn_gru_dual_attn_recon_supcon_model(
                            X_train_curve.shape[1], X_train_curve.shape[2], n_classes); epochs = 500
                    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0))

                    if _val_split_ok:
                        _hist = model.fit(
                            X_train_curve_fit,
                            {'cls_out': y_train_fit},
                            validation_data=(X_val_curve, {'cls_out': y_val}),
                            epochs=epochs, batch_size=_bs, shuffle=True, verbose=0,
                            callbacks=_fit_callbacks)
                        histories.append(_hist.history)
                    else:
                        _hist = model.fit(X_train_curve, {'cls_out': y_train},
                                  epochs=epochs, batch_size=_bs, shuffle=True, verbose=0)
                        histories.append(_hist.history)

                    if _do_xai_save:
                        _xai_path = Path(save_model_dir) / f"{m}_{f}_{save_model_curve_type}_model.keras"
                        safe_keras_save(model, _xai_path)
                        print(f"     [XAI] Saved {m} -> {_xai_path}")

                    cls_prob, _proj = model.predict(X_test_curve, verbose=0)
                    pred = np.argmax(cls_prob, axis=1)
                    cls  = np.unique(y_encoded)

                    preds.append(pred)
                    probs.append(cls_prob)
                    classes_list.append(cls)
                    tf.keras.backend.clear_session()

                elif _base_m in BRANCH_SUPCON3_MODEL_KEYS:
                    tf.keras.backend.clear_session()
                    T = X_train_curve.shape[1]
                    if _base_m == 'cnn_gru_dual_supcon3':
                        model = create_cnn_gru_dual_supcon3_model(T, n_classes)
                    elif _base_m == 'cnn_gru_dual_attn_recon_supcon3':
                        model = create_cnn_gru_dual_attn_recon_supcon3_model(
                            X_train_curve.shape[1], X_train_curve.shape[2], n_classes)
                    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0))
                    epochs = 500
                    if _val_split_ok:
                        _hist = model.fit(X_train_curve_fit, {'cls_out': y_train_fit},
                                  validation_data=(X_val_curve, {'cls_out': y_val}),
                                  epochs=epochs, batch_size=_bs, shuffle=True, verbose=0,
                                  callbacks=_fit_callbacks)
                        histories.append(_hist.history)
                    else:
                        _hist = model.fit(X_train_curve, {'cls_out': y_train},
                                  epochs=epochs, batch_size=_bs, shuffle=True, verbose=0)
                        histories.append(_hist.history)
                    if _do_xai_save:
                        _xai_path = Path(save_model_dir) / f"{m}_{f}_{save_model_curve_type}_model.keras"
                        safe_keras_save(model, _xai_path)
                        print(f"     [XAI] Saved {m} -> {_xai_path}")
                    raw_out  = model.predict(X_test_curve, verbose=0)
                    cls_prob = raw_out[0]
                    pred = np.argmax(cls_prob, axis=1)
                    cls  = np.unique(y_encoded)
                    preds.append(pred); probs.append(cls_prob); classes_list.append(cls)
                    tf.keras.backend.clear_session()

                else:
                    if _base_m == "cnn": clf = KerasModelWrapper(model=create_cnn_model, model__input_size=X_train_curve.shape[1], model__output_size=n_classes, epochs=1000, batch_size=_bs, shuffle=True, verbose=False, random_state=0)
                    elif _base_m == "gru": clf = KerasModelWrapper(model=create_gru_model, model__input_size=X_train_curve.shape[1], model__output_size=n_classes, epochs=500, batch_size=_bs, shuffle=True, verbose=False, random_state=0)
                    elif _base_m == "transformer": clf = KerasModelWrapper(model=create_transformer_model, model__input_size=X_train_curve.shape[1], model__output_size=n_classes, epochs=500, batch_size=_bs, shuffle=True, verbose=False, random_state=0)
                    elif _base_m == "knn": clf = KNeighborsClassifier(n_neighbors=10)
                    else:
                        raise ValueError(f"Model '{m}' is not properly defined in the training loop.")

                    if _val_split_ok:
                        clf.fit(X_train_curve_fit, y_train_fit,
                               validation_data=(X_val_curve, y_val),
                               callbacks=_fit_callbacks)
                        if hasattr(clf, 'history_') and clf.history_:
                            histories.append(clf.history_)
                    else:
                        clf.fit(X_train_curve, y_train)
                        if hasattr(clf, 'history_') and clf.history_:
                            histories.append(clf.history_)

                    if _do_xai_save and _base_m in ['cnn', 'gru', 'transformer']:
                        _xai_path = Path(save_model_dir) / f"{_XAI_SAVE_NAME[m]}_{f}_{save_model_curve_type}_model.keras"
                        safe_keras_save(clf.model_, _xai_path)
                        print(f"     [XAI] Saved {_XAI_SAVE_NAME[m]} -> {_xai_path}")

                    preds.append(clf.predict(X_test_curve))
                    probs.append(clf.predict_proba(X_test_curve))
                    classes_list.append(clf.classes_)

                    if _base_m in ["cnn", "gru", "transformer"]:
                        tf.keras.backend.clear_session()

            end_time = time.perf_counter()
            duration = end_time - start_time
            formatted_time = time.strftime("%H:%M:%S", time.gmtime(int(duration)))

            res_entry[preds_key] = preds
            res_entry[probs_key] = probs
            res_entry[classes_key] = classes_list
            if histories:
                res_entry[f'train_history_{_base_m}_'] = histories

            results_dict[f] = res_entry
            if checkpoint_fn is not None:
                checkpoint_fn(results_dict)

            fold_accs = [accuracy_score(yt, yp) for yt, yp in zip(res_entry["y_trues_"], preds)]
            acc = np.mean(fold_accs) * 100
            std = np.std(fold_accs) * 100

            print(f"     [+] {mode_name}-{dataset_name}-{filter_name[:30]} | {print_name} | {acc:5.2f}% ± {std:5.2f}% | Duration: {formatted_time}")

        results_dict[f] = res_entry

    return results_dict


# ====================================================================
# MODULE 2: VISUALIZATION FUNCTIONS
# ====================================================================
def plot_ml_results(results_dict, outlier_filters, dataset_name, mode_name, total_count, save_prefix=None):
    import config

    present_filters = [f for f in outlier_filters if f in results_dict]
    filter_labels = [str(f) if f is not None else "No Filter" for f in present_filters]

    filter_palette = config.get_palette(present_filters, config.FILTER_COLORS)
    colors = [filter_palette[f] for f in present_filters]

    sample_res = next((results_dict[f] for f in outlier_filters if f in results_dict), None)

    method_info = []
    if sample_res:
        if 'y_preds_AC_kNN_' in sample_res:
            method_info.append(('kNN (ACA)', 'y_preds_AC_kNN_'))
        if 'y_preds_AC_trans_' in sample_res:
            method_info.append(('Transformer (ACA)', 'y_preds_AC_trans_'))
        if 'y_preds_AC_gru_' in sample_res:
            method_info.append(('Gated Recurrent Unit (ACA)', 'y_preds_AC_gru_'))
        if 'y_preds_AC_' in sample_res:
            method_info.append(('Convolutional Neural Network (ACA)', 'y_preds_AC_'))
        if 'y_preds_AC_cnn_gru_dual_' in sample_res:
            method_info.append(('CNN + GRU Dual', 'y_preds_AC_cnn_gru_dual_'))
        if 'y_preds_AC_cnn_gru_dual_attn_recon_' in sample_res:
            method_info.append(('CNN+GRU Dual (Attn Recon)', 'y_preds_AC_cnn_gru_dual_attn_recon_'))
        _already_added = {_key for _, _key in method_info}
        for _mname, (_pk, _, _) in config.MODEL_KEY_MAP.items():
            if _pk in sample_res and _pk not in _already_added:
                _label = config.MODEL_PRINT_MAP.get(_mname, _mname)
                method_info.append((_label, _pk))
                _already_added.add(_pk)

    if not method_info:
        print(f"  [Warning] No model data found in results dict to plot for {dataset_name}.")
        return []

    fig_acc, axes = plt.subplots(len(method_info), 1, figsize=(14, 6 * len(method_info)))
    if len(method_info) == 1:
        axes = [axes]

    for ax, (title, m_key) in zip(axes, method_info):
        means, stds = [], []
        base_mean = 0

        for f in outlier_filters:
            if f not in results_dict: continue
            res = results_dict[f]

            if m_key not in res:
                means.append(0)
                stds.append(0)
                continue

            fold_accs = [accuracy_score(yt, yp) * 100 for yt, yp in zip(res['y_trues_'], res[m_key])]
            if len(fold_accs) > 0:
                m_val = np.mean(fold_accs)
                s_val = np.std(fold_accs)
            else:
                m_val = 0.0
                s_val = 0.0

            means.append(m_val)
            stds.append(s_val)

            if f is None:
                base_mean = m_val

        bar_labels = [f'{m:.1f}%' for m in means]
        bars = ax.bar(filter_labels, means, yerr=stds, color=colors, edgecolor='black', alpha=0.8, capsize=5)

        ax.axhline(y=base_mean, color='red', linestyle='--', linewidth=2, label=f'Baseline ({base_mean:.1f}%)')
        ax.bar_label(bars, labels=bar_labels, padding=5, fontsize=10, fontweight='bold')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0, 115)
        ax.set_xticks(range(len(filter_labels)))
        ax.set_xticklabels(filter_labels, rotation=15, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.legend(loc='upper right')

    fig_acc.suptitle(f"Model Accuracies | {mode_name}: {dataset_name}", fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()

    if save_prefix:
        acc_path = f"{save_prefix}_accuracies.png"
        fig_acc.savefig(acc_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig_acc)

    # --- 2. PLOT DATA COMPOSITION ---
    n_normals = [results_dict[f]['mask_count'] for f in outlier_filters if f in results_dict]
    n_outliers = [total_count - n for n in n_normals]

    fig_comp, ax_comp = plt.subplots(figsize=(12, 6))
    ax_comp.bar(filter_labels, n_normals, color=colors, edgecolor='black', alpha=0.8, label='Normal')
    ax_comp.bar(filter_labels, n_outliers, bottom=n_normals, color='#ffcccc', edgecolor='black', alpha=0.6, label='Outlier')

    for i in range(len(filter_labels)):
        if n_normals[i] > 0:
            ax_comp.text(i, n_normals[i]/2, f'{(n_normals[i]/total_count)*100:.1f}%', ha='center', color='white', fontweight='bold')
        if n_outliers[i] > 0:
            ax_comp.text(i, n_normals[i] + (n_outliers[i]/2), f'{(n_outliers[i]/total_count)*100:.1f}%', ha='center', color='darkred', fontweight='bold')

    ax_comp.set_title(f"Data Composition | {mode_name}: {dataset_name}", fontsize=14, fontweight='bold')
    ax_comp.set_ylabel("Number of Samples")
    ax_comp.set_xticks(range(len(filter_labels)))
    ax_comp.set_xticklabels(filter_labels, rotation=15, ha='right')
    plt.tight_layout()

    if save_prefix:
        comp_path = f"{save_prefix}_composition.png"
        fig_comp.savefig(comp_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig_comp)

   # --- 3. CONSOLE LEADERBOARD PRINT ---
    all_results = []
    for title, m_key in method_info:
        for f in outlier_filters:
            if f not in results_dict: continue
            res = results_dict[f]
            if m_key not in res: continue

            fold_accs = [accuracy_score(yt, yp) * 100 for yt, yp in zip(res['y_trues_'], res[m_key])]
            if not fold_accs:
                continue
            mean_acc = np.mean(fold_accs)
            std_acc  = np.std(fold_accs)
            filt_name = str(f) if f is not None else "Baseline (None)"
            all_results.append((mean_acc, std_acc, dataset_name, mode_name, title, filt_name))

    all_results.sort(key=lambda x: x[0], reverse=True)

    print(f"\n  🏆 Top Combinations for {mode_name}: {dataset_name}")
    print("  " + "-" * 105)
    for i, (acc, std_acc, d_name, m_name, method, filt) in enumerate(all_results):
        print(f"  {i+1:2d}. {acc:6.2f}% ± {std_acc:5.2f}% "
              f"| Data: {d_name[:15]:<15} | Model: {method[:20]:<20} | Filter: {filt[:30]}")
    print("  " + "-" * 105 + "\n")

    return all_results
