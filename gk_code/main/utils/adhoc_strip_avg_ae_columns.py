"""Ad-hoc one-off script.

Strips AE-derived columns (cnn_ae_*, lstm_ae_*) from the cached
'ori_curves_avg' entry of kinetic_features in each experiment's unified
pipeline state (curve_for_training_nonorm.joblib), so that re-running
02_outlier_detection_pipeline.py (without --force_rerun) will recompute
the autoencoder outlier filters for ori_curves_avg instead of seeing
the previously-propagated columns and skipping.

Usage:
    python adhoc_strip_avg_ae_columns.py            # dry run, no writes
    python adhoc_strip_avg_ae_columns.py --apply    # actually modify files
"""
import sys
from pathlib import Path
import joblib

DATASETS_ROOT = Path("/vol/bitbucket/gk225/POC_DDM_datasets/POC_DDM_multi")
STATE_FILENAME = "curve_for_training_nonorm.joblib"
AE_PREFIXES = ("cnn_ae_", "lstm_ae_")

apply_changes = "--apply" in sys.argv

for state_path in sorted(DATASETS_ROOT.rglob(STATE_FILENAME)):
    print("-" * 80)
    try:
        state = joblib.load(state_path)
    except Exception as e:
        print(f"[SKIP] {state_path}: failed to load ({e})")
        continue

    dataset_name = list(state.get("dataset_name", []))
    kinetic_features = state.get("kinetic_features")
    if "ori_curves_avg" not in dataset_name or kinetic_features is None:
        print(f"[SKIP] {state_path}: no ori_curves_avg entry or missing kinetic_features")
        continue

    avg_idx = dataset_name.index("ori_curves_avg")
    df = kinetic_features[avg_idx]
    ae_cols = [c for c in df.columns if c.startswith(AE_PREFIXES)]
    if not ae_cols:
        print(f"[OK]    {state_path}: no AE columns on ori_curves_avg")
        continue

    print(f"[STRIP] {state_path}: dropping {ae_cols}")
    if apply_changes:
        kinetic_features[avg_idx] = df.drop(columns=ae_cols)
        state["kinetic_features"] = kinetic_features
        joblib.dump(state, state_path, compress=3)

if not apply_changes:
    print("\nDry run only. Re-run with --apply to write changes.")
