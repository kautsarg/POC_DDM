import os
# ====================================================================
# SUPPRESS TENSORFLOW C++ WARNINGS (Must be before TF import)
# ====================================================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=INFO, 1=WARN, 2=ERROR, 3=FATAL

import time
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import tensorflow as tf
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
tf.get_logger().setLevel('ERROR')

from scikeras.wrappers import KerasClassifier

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
def set_global_determinism(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        pass


# ====================================================================
# NEURAL NETWORK SETUP
# ====================================================================
class KerasModelWrapper(KerasClassifier):
    pass

# 1. 1D CNN
def create_cnn_model(input_size, output_size, kernel_size_1=5, kernel_size_2=3): 
    inputs = tf.keras.layers.Input(shape=(input_size, 1))
    x = tf.keras.layers.Conv1D(16, kernel_size_1, activation='relu')(inputs)
    x = tf.keras.layers.Conv1D(8, kernel_size_2, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(output_size, activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# 2. LSTM (Long Short-Term Memory)
def create_lstm_model(input_size, output_size):
    inputs = tf.keras.layers.Input(shape=(input_size, 1))
    x = tf.keras.layers.LSTM(32, return_sequences=True)(inputs)
    x = tf.keras.layers.LSTM(16)(x)
    x = tf.keras.layers.Dense(output_size, activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# 3. GRU (Gated Recurrent Unit)
def create_gru_model(input_size, output_size):
    inputs = tf.keras.layers.Input(shape=(input_size, 1))
    x = tf.keras.layers.GRU(32, return_sequences=True)(inputs)
    x = tf.keras.layers.GRU(16)(x)
    x = tf.keras.layers.Dense(output_size, activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# 4. Simple RNN
def create_rnn_model(input_size, output_size):
    inputs = tf.keras.layers.Input(shape=(input_size, 1))
    x = tf.keras.layers.SimpleRNN(32, return_sequences=True)(inputs)
    x = tf.keras.layers.SimpleRNN(16)(x)
    x = tf.keras.layers.Dense(output_size, activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# 5. Transformer (1D Time Series Encoder)
def create_transformer_model(input_size, output_size, head_size=32, num_heads=2, ff_dim=32, num_blocks=2, dropout=0.1):
    inputs = tf.keras.layers.Input(shape=(input_size, 1))
    x = inputs
    
    for _ in range(num_blocks):
        # Multi-Head Attention Block
        attn_output = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
        attn_output = tf.keras.layers.Dropout(dropout)(attn_output)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

        # Feed Forward Block
        ffn_output = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
        ffn_output = tf.keras.layers.Dropout(dropout)(ffn_output)
        ffn_output = tf.keras.layers.Dense(inputs.shape[-1])(ffn_output)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

    # Global average pooling over the sequence dimension
    x = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(output_size, activation="softmax")(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model


# ====================================================================
# MODULE 1: MODEL EVALUATION FUNCTION (WITH PROBABILITIES)
# ====================================================================
def evaluate_outlier_filters(
    X_curves, features_df, y_encoded, outlier_filters, dataset_name, mode_name,
    cached_results=None, models=["cnn", "lstm", "gru", "rnn", "transformer", "rf", "knn", "ffi"], n_splits=1,
    checkpoint_fn=None
):
    X_FFI_full = X_curves[:, [-1]]
    
    results_dict = cached_results.copy() if cached_results is not None else {}
    total_filters = len(outlier_filters)
    
    # Normalize model names to lowercase for robust matching
    models = [m.lower() for m in models]

    for idx, f in enumerate(outlier_filters):
        filter_name = f if f else 'None (Baseline)'
        filter_pct = ((idx + 1) / total_filters) * 100
        print(f"  -> Testing Filter [{idx+1}/{total_filters} | {filter_pct:.1f}%]: {filter_name}")
        
        # Per-model cache detection (do not skip the whole filter)
        res_cached = results_dict.get(f, {})
        skip_models = set()

        if res_cached:
            if "y_preds_AC_" in res_cached:        skip_models.add("cnn")
            if "y_preds_AC_lstm_" in res_cached:   skip_models.add("lstm")
            if "y_preds_AC_gru_" in res_cached:    skip_models.add("gru")
            if "y_preds_AC_rnn_" in res_cached:    skip_models.add("rnn")
            if "y_preds_AC_trans_" in res_cached:  skip_models.add("transformer")
            if "y_preds_AC_rf_" in res_cached:     skip_models.add("rf")
            if "y_preds_AC_kNN_" in res_cached:    skip_models.add("knn")
            if "y_preds_FFI_" in res_cached:       skip_models.add("ffi")

        if skip_models:
            print(f"     [CACHE HIT] Skipping cached models: {sorted(skip_models)}")
        
        if f is None:
            mask = np.ones(len(y_encoded), dtype=bool)
        elif f in features_df.columns:
            mask = (features_df[f] == 1).fillna(False).values
        else:
            print(f"     [Warning] {f} not found in dataset. Skipping.")
            continue

        X_AC = np.nan_to_num(X_curves[mask], nan=0.0, posinf=0.0, neginf=0.0)
        X_FFI = np.nan_to_num(X_FFI_full[mask], nan=0.0, posinf=0.0, neginf=0.0)
        y_true = y_encoded[mask]

        unique_classes, class_counts = np.unique(y_true, return_counts=True)
        rare_classes = unique_classes[class_counts < 2]

        if len(rare_classes) > 0:
            valid_class_mask = ~np.isin(y_true, rare_classes)
            X_AC = X_AC[valid_class_mask]
            X_FFI = X_FFI[valid_class_mask]
            y_true = y_true[valid_class_mask]

        n_classes = len(np.unique(y_true))
        
        if n_classes < 2:
            print(f"     [Warning] Not enough classes left to train after filtering. Skipping.")
            continue

        if len(y_true) < 2 * n_classes:
            print(f"     [Warning] Too few samples left ({len(y_true)}) to stratify {n_classes} classes. Skipping.")
            continue

        calculated_test_size = max(int(len(y_true) * 0.10), n_classes)

        if n_splits == 1:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=calculated_test_size, random_state=0)
        elif n_splits > 1:
            min_class_count = np.min(class_counts[~np.isin(unique_classes, rare_classes)])
            actual_splits = min(n_splits, min_class_count)
            
            if actual_splits < n_splits:
                print(f"     [Warning] Reduced n_splits from {n_splits} to {actual_splits} due to class imbalance.")
                
            splitter = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=0)
        else:
            raise ValueError("n_splits must be 1 or greater.")

        # Initialize lists
        y_trues_ = []
        y_preds_AC_, y_probs_AC_, classes_AC_ = [], [], []
        y_preds_AC_lstm_, y_probs_AC_lstm_, classes_AC_lstm_ = [], [], []
        y_preds_AC_gru_, y_probs_AC_gru_, classes_AC_gru_ = [], [], []
        y_preds_AC_rnn_, y_probs_AC_rnn_, classes_AC_rnn_ = [], [], []
        y_preds_AC_trans_, y_probs_AC_trans_, classes_AC_trans_ = [], [], []
        y_preds_AC_rf_, y_probs_AC_rf_, classes_AC_rf_ = [], [], []
        y_preds_AC_kNN_, y_probs_AC_kNN_, classes_AC_kNN_ = [], [], []
        y_preds_FFI_, y_probs_FFI_, classes_FFI_ = [], [], []

        def _checkpoint_partial():
            if checkpoint_fn is None:
                return
            res_entry_partial = {
                "y_trues_": y_trues_,
                "mask_count": np.sum(mask) 
            }
            if "cnn" in models:
                res_entry_partial.update({"y_preds_AC_": y_preds_AC_, "y_probs_AC_": y_probs_AC_, "classes_AC_": classes_AC_})
            if "lstm" in models:
                res_entry_partial.update({"y_preds_AC_lstm_": y_preds_AC_lstm_, "y_probs_AC_lstm_": y_probs_AC_lstm_, "classes_AC_lstm_": classes_AC_lstm_})
            if "gru" in models:
                res_entry_partial.update({"y_preds_AC_gru_": y_preds_AC_gru_, "y_probs_AC_gru_": y_probs_AC_gru_, "classes_AC_gru_": classes_AC_gru_})
            if "rnn" in models:
                res_entry_partial.update({"y_preds_AC_rnn_": y_preds_AC_rnn_, "y_probs_AC_rnn_": y_probs_AC_rnn_, "classes_AC_rnn_": classes_AC_rnn_})
            if "transformer" in models:
                res_entry_partial.update({"y_preds_AC_trans_": y_preds_AC_trans_, "y_probs_AC_trans_": y_probs_AC_trans_, "classes_AC_trans_": classes_AC_trans_})
            if "rf" in models:
                res_entry_partial.update({"y_preds_AC_rf_": y_preds_AC_rf_, "y_probs_AC_rf_": y_probs_AC_rf_, "classes_AC_rf_": classes_AC_rf_})
            if "knn" in models:
                res_entry_partial.update({"y_preds_AC_kNN_": y_preds_AC_kNN_, "y_probs_AC_kNN_": y_probs_AC_kNN_, "classes_AC_kNN_": classes_AC_kNN_})
            if "ffi" in models:
                res_entry_partial.update({"y_preds_FFI_": y_preds_FFI_, "y_probs_FFI_": y_probs_FFI_, "classes_FFI_": classes_FFI_})
            results_dict[f] = res_entry_partial
            checkpoint_fn(results_dict)

        splits = splitter.split(X_AC, y_true)
        
        for train_index, test_index in splits:
            X_AC_train, X_AC_test = X_AC[train_index], X_AC[test_index]
            X_FFI_train, X_FFI_test = X_FFI[train_index], X_FFI[test_index]
            y_train, y_test = y_true[train_index], y_true[test_index]
            y_trues_.append(y_test)

            # --- Convolutional Neural Network (CNN) ---
            if "cnn" in models and "cnn" not in skip_models:
                start_time = time.perf_counter()
                clf_AC = KerasModelWrapper(model=create_cnn_model,
                                   model__input_size=X_AC.shape[1],
                                   model__output_size=len(np.unique(y_encoded)), 
                                   epochs=1000, 
                                   batch_size=512, 
                                   shuffle=True, 
                                   verbose=False,
                                   random_state=0)
                clf_AC.fit(X_AC_train, y_train)
                
                pred_AC = clf_AC.predict(X_AC_test)
                prob_AC = clf_AC.predict_proba(X_AC_test)
                
                y_preds_AC_.append(pred_AC)
                y_probs_AC_.append(prob_AC)
                classes_AC_.append(clf_AC.classes_)
                
                cnn_acc = accuracy_score(y_test, pred_AC) * 100

                end_time = time.perf_counter()
                duration = end_time - start_time
                formatted_time = time.strftime("%H:%M:%S", time.gmtime(int(duration)))

                print(f"     [+] {mode_name}-{dataset_name}-{filter_name[:30]} | CNN (ACA)   | {cnn_acc:5.2f}%   | Duration: {formatted_time}")
                tf.keras.backend.clear_session()
                _checkpoint_partial()
                
            # --- Long Short-Term Memory (LSTM) ---
            if "lstm" in models and "lstm" not in skip_models:
                start_time = time.perf_counter()
                clf_lstm = KerasModelWrapper(model=create_lstm_model,
                                   model__input_size=X_AC.shape[1],
                                   model__output_size=len(np.unique(y_encoded)), 
                                   epochs=500,
                                   batch_size=512, 
                                   shuffle=True, 
                                   verbose=False,
                                   random_state=0)
                clf_lstm.fit(X_AC_train, y_train)
                
                pred_lstm = clf_lstm.predict(X_AC_test)
                prob_lstm = clf_lstm.predict_proba(X_AC_test)
                
                y_preds_AC_lstm_.append(pred_lstm)
                y_probs_AC_lstm_.append(prob_lstm)
                classes_AC_lstm_.append(clf_lstm.classes_)
                
                lstm_acc = accuracy_score(y_test, pred_lstm) * 100

                end_time = time.perf_counter()
                duration = end_time - start_time
                formatted_time = time.strftime("%H:%M:%S", time.gmtime(int(duration)))

                print(f"     [+] {mode_name}-{dataset_name}-{filter_name[:30]} | LSTM (ACA)  | {lstm_acc:5.2f}%   | Duration: {formatted_time}")
                tf.keras.backend.clear_session()
                _checkpoint_partial()
                
            # --- Gated Recurrent Unit (GRU) ---
            if "gru" in models and "gru" not in skip_models:
                start_time = time.perf_counter()
                clf_gru = KerasModelWrapper(model=create_gru_model,
                                   model__input_size=X_AC.shape[1],
                                   model__output_size=len(np.unique(y_encoded)), 
                                   epochs=500, 
                                   batch_size=512, 
                                   shuffle=True, 
                                   verbose=False,
                                   random_state=0)
                clf_gru.fit(X_AC_train, y_train)
                
                pred_gru = clf_gru.predict(X_AC_test)
                prob_gru = clf_gru.predict_proba(X_AC_test)
                
                y_preds_AC_gru_.append(pred_gru)
                y_probs_AC_gru_.append(prob_gru)
                classes_AC_gru_.append(clf_gru.classes_)
                
                gru_acc = accuracy_score(y_test, pred_gru) * 100

                end_time = time.perf_counter()
                duration = end_time - start_time
                formatted_time = time.strftime("%H:%M:%S", time.gmtime(int(duration)))

                print(f"     [+] {mode_name}-{dataset_name}-{filter_name[:30]} | GRU (ACA)   | {gru_acc:5.2f}%   | Duration: {formatted_time}")
                tf.keras.backend.clear_session()
                _checkpoint_partial()

            # --- Simple Recurrent Neural Network (RNN) ---
            if "rnn" in models and "rnn" not in skip_models:
                start_time = time.perf_counter()
                clf_rnn = KerasModelWrapper(model=create_rnn_model,
                                   model__input_size=X_AC.shape[1],
                                   model__output_size=len(np.unique(y_encoded)), 
                                   epochs=500, 
                                   batch_size=512, 
                                   shuffle=True, 
                                   verbose=False,
                                   random_state=0)
                clf_rnn.fit(X_AC_train, y_train)
                
                pred_rnn = clf_rnn.predict(X_AC_test)
                prob_rnn = clf_rnn.predict_proba(X_AC_test)
                
                y_preds_AC_rnn_.append(pred_rnn)
                y_probs_AC_rnn_.append(prob_rnn)
                classes_AC_rnn_.append(clf_rnn.classes_)
                
                rnn_acc = accuracy_score(y_test, pred_rnn) * 100

                end_time = time.perf_counter()
                duration = end_time - start_time
                formatted_time = time.strftime("%H:%M:%S", time.gmtime(int(duration)))

                print(f"     [+] {mode_name}-{dataset_name}-{filter_name[:30]} | RNN (ACA)   | {rnn_acc:5.2f}%   | Duration: {formatted_time}")
                tf.keras.backend.clear_session()
                _checkpoint_partial()

            # --- Transformer ---
            if "transformer" in models and "transformer" not in skip_models:
                start_time = time.perf_counter()
                clf_trans = KerasModelWrapper(model=create_transformer_model,
                                   model__input_size=X_AC.shape[1],
                                   model__output_size=len(np.unique(y_encoded)), 
                                   epochs=500, 
                                   batch_size=512, 
                                   shuffle=True, 
                                   verbose=False,
                                   random_state=0)
                clf_trans.fit(X_AC_train, y_train)
                
                pred_trans = clf_trans.predict(X_AC_test)
                prob_trans = clf_trans.predict_proba(X_AC_test)
                
                y_preds_AC_trans_.append(pred_trans)
                y_probs_AC_trans_.append(prob_trans)
                classes_AC_trans_.append(clf_trans.classes_)
                
                trans_acc = accuracy_score(y_test, pred_trans) * 100

                end_time = time.perf_counter()
                duration = end_time - start_time
                formatted_time = time.strftime("%H:%M:%S", time.gmtime(int(duration)))

                print(f"     [+] {mode_name}-{dataset_name}-{filter_name[:30]} | Trans (ACA) | {trans_acc:5.2f}%   | Duration: {formatted_time}")
                tf.keras.backend.clear_session()
                _checkpoint_partial()

            # --- Random Forest (AC) ---
            if "rf" in models and "rf" not in skip_models:
                start_time = time.perf_counter()
                clf_AC_rf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
                clf_AC_rf.fit(X_AC_train, y_train)
                
                pred_rf = clf_AC_rf.predict(X_AC_test)
                prob_rf = clf_AC_rf.predict_proba(X_AC_test) 
                
                y_preds_AC_rf_.append(pred_rf)
                y_probs_AC_rf_.append(prob_rf)
                classes_AC_rf_.append(clf_AC_rf.classes_)
                
                rf_acc = accuracy_score(y_test, pred_rf) * 100

                end_time = time.perf_counter()
                duration = end_time - start_time
                formatted_time = time.strftime("%H:%M:%S", time.gmtime(int(duration)))

                print(f"     [+] {mode_name}-{dataset_name}-{filter_name[:30]} | RF (ACA)    | {rf_acc:5.2f}%   | Duration: {formatted_time}")
                _checkpoint_partial()

            # --- K-Nearest Neighbors (AC) ---
            if "knn" in models and "knn" not in skip_models:
                start_time = time.perf_counter()
                clf_AC_kNN = KNeighborsClassifier(n_neighbors=10)
                clf_AC_kNN.fit(X_AC_train, y_train)
                
                pred_kNN = clf_AC_kNN.predict(X_AC_test)
                prob_kNN = clf_AC_kNN.predict_proba(X_AC_test) 
                
                y_preds_AC_kNN_.append(pred_kNN)
                y_probs_AC_kNN_.append(prob_kNN)
                classes_AC_kNN_.append(clf_AC_kNN.classes_)
                
                knn_acc = accuracy_score(y_test, pred_kNN) * 100

                end_time = time.perf_counter()
                duration = end_time - start_time
                formatted_time = time.strftime("%H:%M:%S", time.gmtime(int(duration)))

                print(f"     [+] {mode_name}-{dataset_name}-{filter_name[:30]} | KNN (ACA)   | {knn_acc:5.2f}%   | Duration: {formatted_time}")
                _checkpoint_partial()

            # --- Logistic Regression (FFI) ---
            if "ffi" in models and "ffi" not in skip_models:
                start_time = time.perf_counter()
                clf_FFI = LogisticRegression(max_iter=1000)
                clf_FFI.fit(X_FFI_train, y_train)
                
                pred_FFI = clf_FFI.predict(X_FFI_test)
                prob_FFI = clf_FFI.predict_proba(X_FFI_test) 
                
                y_preds_FFI_.append(pred_FFI)
                y_probs_FFI_.append(prob_FFI)
                classes_FFI_.append(clf_FFI.classes_)
                
                lr_acc = accuracy_score(y_test, pred_FFI) * 100

                end_time = time.perf_counter()
                duration = end_time - start_time
                formatted_time = time.strftime("%H:%M:%S", time.gmtime(int(duration)))

                print(f"     [+] {mode_name}-{dataset_name}-{filter_name[:30]} | LR (FFI)    | {lr_acc:5.2f}%")
                _checkpoint_partial()
            
        # Dynamically build the results entry based on trained models
        res_entry = {
            "y_trues_": y_trues_,
            "mask_count": np.sum(mask) 
        }
        if "cnn" in models:
            res_entry.update({"y_preds_AC_": y_preds_AC_, "y_probs_AC_": y_probs_AC_, "classes_AC_": classes_AC_})
        if "lstm" in models:
            res_entry.update({"y_preds_AC_lstm_": y_preds_AC_lstm_, "y_probs_AC_lstm_": y_probs_AC_lstm_, "classes_AC_lstm_": classes_AC_lstm_})
        if "gru" in models:
            res_entry.update({"y_preds_AC_gru_": y_preds_AC_gru_, "y_probs_AC_gru_": y_probs_AC_gru_, "classes_AC_gru_": classes_AC_gru_})
        if "rnn" in models:
            res_entry.update({"y_preds_AC_rnn_": y_preds_AC_rnn_, "y_probs_AC_rnn_": y_probs_AC_rnn_, "classes_AC_rnn_": classes_AC_rnn_})
        if "transformer" in models:
            res_entry.update({"y_preds_AC_trans_": y_preds_AC_trans_, "y_probs_AC_trans_": y_probs_AC_trans_, "classes_AC_trans_": classes_AC_trans_})
        if "rf" in models:
            res_entry.update({"y_preds_AC_rf_": y_preds_AC_rf_, "y_probs_AC_rf_": y_probs_AC_rf_, "classes_AC_rf_": classes_AC_rf_})
        if "knn" in models:
            res_entry.update({"y_preds_AC_kNN_": y_preds_AC_kNN_, "y_probs_AC_kNN_": y_probs_AC_kNN_, "classes_AC_kNN_": classes_AC_kNN_})
        if "ffi" in models:
            res_entry.update({"y_preds_FFI_": y_preds_FFI_, "y_probs_FFI_": y_probs_FFI_, "classes_FFI_": classes_FFI_})
            
        results_dict[f] = res_entry

    return results_dict


# ====================================================================
# MODULE 2: VISUALIZATION FUNCTIONS
# ====================================================================
def plot_ml_results(results_dict, outlier_filters, dataset_name, mode_name, total_count, save_prefix=None):
    filter_labels = [str(f) if f is not None else "No Filter" for f in outlier_filters if f in results_dict]
    
    colors = []
    for f in outlier_filters:
        if f not in results_dict: continue
        if f is None: colors.append('#888888')
        elif 'mean_' in f: colors.append('#ff7f0e')
        elif 'amf_' in f: colors.append('#2ca02c')
        else: colors.append('#9467bd')

    # Dynamically determine which models were evaluated
    sample_res = next((results_dict[f] for f in outlier_filters if f in results_dict), None)
    
    method_info = []
    if sample_res:
        if 'y_preds_FFI_' in sample_res:
            method_info.append(('Logistic Regression (FFI)', 'y_preds_FFI_'))
        if 'y_preds_AC_kNN_' in sample_res:
            method_info.append(('kNN (ACA)', 'y_preds_AC_kNN_'))
        if 'y_preds_AC_rf_' in sample_res:
            method_info.append(('Random Forest (ACA)', 'y_preds_AC_rf_'))
        if 'y_preds_AC_trans_' in sample_res:
            method_info.append(('Transformer (ACA)', 'y_preds_AC_trans_'))
        if 'y_preds_AC_rnn_' in sample_res:
            method_info.append(('Simple RNN (ACA)', 'y_preds_AC_rnn_'))
        if 'y_preds_AC_gru_' in sample_res:
            method_info.append(('Gated Recurrent Unit (ACA)', 'y_preds_AC_gru_'))
        if 'y_preds_AC_lstm_' in sample_res:
            method_info.append(('Long Short-Term Memory (ACA)', 'y_preds_AC_lstm_'))
        if 'y_preds_AC_' in sample_res:
            method_info.append(('Convolutional Neural Network (ACA)', 'y_preds_AC_'))

    if not method_info:
        print(f"  [Warning] No model data found in results dict to plot for {dataset_name}.")
        return []

    # --- 1. PLOT ACCURACIES ---
    # Adjust figure height based on the number of models actually plotted
    fig_acc, axes = plt.subplots(len(method_info), 1, figsize=(14, 6 * len(method_info)))
    if len(method_info) == 1:
        axes = [axes]
    
    for ax, (title, m_key) in zip(axes, method_info):
        means, stds = [], []
        base_mean = 0

        for f in outlier_filters:
            if f not in results_dict: continue
            res = results_dict[f]
            
            fold_accs = [accuracy_score(yt, yp) * 100 for yt, yp in zip(res['y_trues_'], res[m_key])]
            m_val = np.mean(fold_accs)
            s_val = np.std(fold_accs)
            
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
    print(f"\n  🏆 Top Combinations for {mode_name}: {dataset_name}")
    print("  " + "-"*95)
    
    all_results = []
    for title, m_key in method_info:
        for f in outlier_filters:
            if f not in results_dict: continue
            res = results_dict[f]
            fold_accs = [accuracy_score(yt, yp) * 100 for yt, yp in zip(res['y_trues_'], res[m_key])]
            mean_acc = np.mean(fold_accs)
            filt_name = str(f) if f is not None else "Baseline (None)"
            
            all_results.append((mean_acc, dataset_name, mode_name, title, filt_name))
            
    all_results.sort(key=lambda x: x[0], reverse=True)
    
    for i, (acc, d_name, m_name, method, filt) in enumerate(all_results):
        print(f"  {i+1}. {acc:6.2f}% | Data: {d_name[:15]:<15} | Model: {method[:20]:<20} | Filter: {filt[:30]}")
    print("  " + "-"*95 + "\n")
    
    return all_results