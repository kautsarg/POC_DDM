import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit

from titan.load_and_preprocessing import titan_load_and_preprocessing

from model import MODEL_BUILDERS, train_and_evaluate

EXP_PATH = Path("/Users/kautsarg/Documents/Final Project/Run Data/trial test data/D20250808_E00_C00_F4500KHz_U_Sample_7")
MODELS_TO_RUN = list(MODEL_BUILDERS.keys()) 
# by default will run all models: [cnn, gru, cnn_gru_dual, transformer, cnn_transformer_dual]
# or you can specify it like [cnn_gru_dual] to run only that model

def load_and_reconstruct(exp_path, n_wells=10, n_a_type="v04"):
    print(f"\n[1/3] Loading experiment data from: {exp_path.name}")
    start_time = time.time()

    exp = titan_load_and_preprocessing(
        exp_path,
        n_wells=n_wells,
        start_type="temperature",
        n_a_type=n_a_type,
        print_status=False
    )

    curves_data = None
    well_ids = []

    print(f"      -> Reconstructing curves for {len(exp.wells_list)} wells...")
    for well in exp.wells_list:
        temp_x = well.well_2d_bs_active.copy()
        temp_x = np.swapaxes(temp_x, 0, 1)

        if curves_data is None:
            curves_data = temp_x
        else:
            curves_data = np.vstack((curves_data, temp_x))

        well_ids.append(temp_x.shape[0])

    Y_well = np.array([label for label, count in enumerate(well_ids) for _ in range(count)])

    elapsed = time.time() - start_time
    print(f"      -> Success. Total curves: {len(Y_well)}. ({elapsed:.2f}s)")
    return curves_data, Y_well

def run_classification(curves, Y_well, models_to_run):
    """
    Scales curves, splits train/test, and trains+tests each requested model.
    """
    print(f"\n[2/3] Preparing data and train/test split")

    encoder = LabelEncoder()
    y_full = encoder.fit_transform(Y_well)
    n_classes = len(np.unique(y_full))

    scaler = MinMaxScaler()
    curves_scaled = scaler.fit_transform(curves)
    X_full = curves_scaled.reshape((curves_scaled.shape[0], curves_scaled.shape[1], 1))

    test_size = max(int(len(y_full) * 0.10), n_classes)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
    train_idx, test_idx = next(splitter.split(X_full, y_full))

    X_train, X_test = X_full[train_idx], X_full[test_idx]
    y_train, y_test = y_full[train_idx], y_full[test_idx]

    print(f"      -> Train: {len(train_idx)} curves | Test: {len(test_idx)} curves | Classes: {n_classes}")

    print(f"\n[3/3] Training and testing models: {models_to_run}")
    results = {}
    for model_name in models_to_run:
        print(f"\n  -> {model_name.upper()}")
        accuracy, duration = train_and_evaluate(model_name, X_train, y_train, X_test, y_test, n_classes)
        results[model_name] = accuracy
        print(f"     Accuracy: {accuracy * 100:.2f}% | Duration: {duration:.2f}s")

    return results

if __name__ == "__main__":
    if EXP_PATH.exists():
        global_start = time.time()

        # 1. Load Data
        curve_data, well_labels = load_and_reconstruct(EXP_PATH)

        # 2. Build, Train & Test Models
        results = run_classification(curve_data, well_labels, MODELS_TO_RUN)

        # 3. Summary
        print(f"\nSummary:")
        for model_name, accuracy in sorted(results.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {model_name:<22}: {accuracy * 100:.2f}%")

        total_elapsed = time.time() - global_start
        print(f"\nPipeline finished in {total_elapsed:.2f}s.")
    else:
        print(f"Error: Path {EXP_PATH} does not exist.")
