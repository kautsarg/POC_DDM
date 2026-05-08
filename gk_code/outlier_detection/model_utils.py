import os
# ====================================================================
# SUPPRESS TENSORFLOW C++ WARNINGS (Must be before TF import)
# ====================================================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=INFO, 1=WARN, 2=ERROR, 3=FATAL

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import tensorflow as tf
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
tf.get_logger().setLevel('ERROR')

from scikeras.wrappers import KerasClassifier

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
class myWrapper(KerasClassifier):
    pass

def create_model(input_size, output_size, kernel_size_1=5, kernel_size_2=3): 
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


# ====================================================================
# MODULE 1: MODEL EVALUATION FUNCTION (NaN-SAFE & MIN-CLASS SAFE)
# ====================================================================
def evaluate_outlier_filters(X_curves, features_df, y_encoded, outlier_filters, dataset_name, mode_name):
    X_FFI_full = X_curves[:, [-1]]
    results_dict = {}
    
    total_filters = len(outlier_filters)

    for idx, f in enumerate(outlier_filters):
        filter_name = f if f else 'None (Baseline)'
        filter_pct = ((idx + 1) / total_filters) * 100
        print(f"  -> Testing Filter [{idx+1}/{total_filters} | {filter_pct:.1f}%]: {filter_name}")
        
        # 1. Generate robust mask
        if f is None:
            mask = np.ones(len(y_encoded), dtype=bool)
        elif f in features_df.columns:
            mask = (features_df[f] == 1).fillna(False).values
        else:
            print(f"     [Warning] {f} not found in dataset. Skipping.")
            continue

        # Sanitize NaN and Inf values to 0.0 to prevent training crashes
        X_AC = np.nan_to_num(X_curves[mask], nan=0.0, posinf=0.0, neginf=0.0)
        X_FFI = np.nan_to_num(X_FFI_full[mask], nan=0.0, posinf=0.0, neginf=0.0)
        y_true = y_encoded[mask]

        # --- SAFEGUARD 1: Remove classes with fewer than 2 samples ---
        unique_classes, class_counts = np.unique(y_true, return_counts=True)
        rare_classes = unique_classes[class_counts < 2]

        if len(rare_classes) > 0:
            valid_class_mask = ~np.isin(y_true, rare_classes)
            X_AC = X_AC[valid_class_mask]
            X_FFI = X_FFI[valid_class_mask]
            y_true = y_true[valid_class_mask]

        n_classes = len(np.unique(y_true))
        
        # --- SAFEGUARD 2: Check if enough classes remain ---
        if n_classes < 2:
            print(f"     [Warning] Not enough classes left to train after filtering. Skipping.")
            continue

        # --- THE FIX: Dynamic Test Sizing ---
        # We need at least 1 sample per class in Train AND Test (min 2 * n_classes)
        if len(y_true) < 2 * n_classes:
            print(f"     [Warning] Too few samples left ({len(y_true)}) to stratify {n_classes} classes. Skipping.")
            continue

        # Calculate absolute test size: 10%, but NEVER less than the number of classes
        calculated_test_size = max(int(len(y_true) * 0.10), n_classes)

        # Move SSS *inside* the loop so it can use the dynamic size
        sss = StratifiedShuffleSplit(n_splits=1, test_size=calculated_test_size, random_state=0)

        y_trues_, y_preds_AC_, y_preds_AC_kNN_, y_preds_FFI_ = [], [], [], []

        # 2. Train / Test Split
        splits = sss.split(X_AC, y_true)
        
        for train_index, test_index in splits:
            X_AC_train, X_AC_test = X_AC[train_index], X_AC[test_index]
            X_FFI_train, X_FFI_test = X_FFI[train_index], X_FFI[test_index]
            y_train, y_test = y_true[train_index], y_true[test_index]
            y_trues_.append(y_test)

            # --- Neural Network (AC) ---
            # ... (The rest of your model training code remains exactly the same below here) ...
            clf_AC = myWrapper(model=create_model,
                               model__input_size=X_AC.shape[1],
                               model__output_size=len(np.unique(y_encoded)), # Keep original output size to prevent shape errors
                               epochs=1000, 
                               batch_size=512, 
                               shuffle=True, 
                               verbose=False,
                               random_state=0)
            clf_AC.fit(X_AC_train, y_train)
            pred_AC = clf_AC.predict(X_AC_test)
            y_preds_AC_.append(pred_AC)
            
            cnn_acc = accuracy_score(y_test, pred_AC) * 100
            print(f"     [+] {mode_name}-{dataset_name}-{filter_name[:30]} | CNN (ACA) | {cnn_acc:5.2f}%")
            tf.keras.backend.clear_session()

            # --- K-Nearest Neighbors (AC) ---
            clf_AC_kNN = KNeighborsClassifier(n_neighbors=10)
            clf_AC_kNN.fit(X_AC_train, y_train)
            pred_kNN = clf_AC_kNN.predict(X_AC_test)
            y_preds_AC_kNN_.append(pred_kNN)
            
            knn_acc = accuracy_score(y_test, pred_kNN) * 100
            print(f"     [+] {mode_name}-{dataset_name}-{filter_name[:30]} | KNN (ACA) | {knn_acc:5.2f}%")

            # --- Logistic Regression (FFI) ---
            clf_FFI = LogisticRegression(max_iter=1000)
            clf_FFI.fit(X_FFI_train, y_train)
            pred_FFI = clf_FFI.predict(X_FFI_test)
            y_preds_FFI_.append(pred_FFI)
            
            lr_acc = accuracy_score(y_test, pred_FFI) * 100
            print(f"     [+] {mode_name}-{dataset_name}-{filter_name[:30]} | LR (FFI)  | {lr_acc:5.2f}%")
            
        # Store results for this filter
        results_dict[f] = {
            "y_trues_": y_trues_, "y_preds_AC_": y_preds_AC_,
            "y_preds_AC_kNN_": y_preds_AC_kNN_, "y_preds_FFI_": y_preds_FFI_,
            "mask_count": np.sum(mask) # Track original mask count
        }

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

    method_info = [
        ('Logistic Regression (FFI)', 'y_preds_FFI_'),
        ('kNN (ACA)', 'y_preds_AC_kNN_'),
        ('Convolutional Neural Network (ACA)', 'y_preds_AC_')
    ]

    # --- 1. PLOT ACCURACIES ---
    fig_acc, axes = plt.subplots(len(method_info), 1, figsize=(14, 18))
    
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