import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import tensorflow as tf
from .model import create_cnn_gru_dual


def run_cv(curves, well_labels, n_folds=5, epochs=500, batch_size=512, seed=0):
    # Stratified K-fold CV with CNN-GRU dual model

    le = LabelEncoder()
    y = le.fit_transform(well_labels)
    X = curves[:, :, np.newaxis].astype(np.float32)
    n_classes = len(le.classes_)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_accs = []
    all_true, all_pred = [], []

    for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"    Fold {fold_i + 1}/{n_folds}  "
              f"(train={len(tr_idx)}, val={len(val_idx)})")

        model = create_cnn_gru_dual(X.shape[1], n_classes)
        cb = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True, verbose=0),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=30, min_lr=1e-5, verbose=0),
        ]

        model.fit(X[tr_idx], y[tr_idx], validation_data=(X[val_idx], y[val_idx]), epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=cb, verbose=0)

        y_pred = np.argmax(model.predict(X[val_idx], verbose=0), axis=1)
        fold_accs.append(accuracy_score(y[val_idx], y_pred) * 100)
        all_true.extend(y[val_idx].tolist())
        all_pred.extend(y_pred.tolist())

        tf.keras.backend.clear_session()

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    return {
        'fold_accs': fold_accs,
        'mean_acc': float(np.mean(fold_accs)),
        'std_acc': float(np.std(fold_accs)),
        'cm': confusion_matrix(all_true, all_pred, labels=list(range(n_classes))),
        'classes': le.classes_,
        'y_true': all_true,
        'y_pred': all_pred,
        'precision': precision_score(all_true, all_pred, average='macro', zero_division=0),
        'recall': recall_score(all_true, all_pred, average='macro', zero_division=0),
        'f1': f1_score(all_true, all_pred, average='macro', zero_division=0),
    }
