import os
import sys
import argparse
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, "../outlier_detection")
import config
from cnn_autoencoder_outlier import run_cnn_autoencoder_pipeline


def get_exp_paths(exp_folder):
    return sorted(
        [
            Path(exp_folder, name)
            for name in os.listdir(exp_folder)
            if os.path.isdir(os.path.join(exp_folder, name)) and name not in [".DS_Store"]
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Light Outlier Detection (CNN AE Global Only)")
    parser.add_argument("--task_id", type=int, default=0, help="Array Job ID")
    parser.add_argument("--exp_folder", type=str, default=config.DEFAULT_EXP_FOLDER)
    parser.add_argument("--out_subdir", type=str, default="light_pipeline", help="Output subfolder under each experiment")
    args = parser.parse_args()

    exp_paths = get_exp_paths(args.exp_folder)
    if args.task_id >= len(exp_paths):
        print(f"Task ID {args.task_id} is out of bounds for {len(exp_paths)} folders. Exiting.")
        sys.exit(0)

    exp_path = exp_paths[args.task_id]
    out_dir = exp_path / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    in_path = out_dir / "light_preprocessed_curves.joblib"
    if not in_path.exists():
        print(f"Skipping {exp_path.name}: '{in_path}' not found.")
        sys.exit(0)

    out_path = out_dir / "light_training_data.joblib"
    if out_path.exists():
        print(f"Cache hit: {exp_path}")
        print("  -> Light outlier data already exists. Skipping.")
        sys.exit(0)

    print(f"\n\n{'#'*80}\nSTARTING LIGHT OUTLIER PIPELINE FOR: {exp_path.name}\n{'#'*80}")

    data = joblib.load(in_path)
    curves = data["curves"]["ori_curves"]
    timestamps = data["timestamps"]
    y_well = data["well_labels"]

    dataset_name = ["ori_curves"]
    dataset = [curves]

    ae_results = run_cnn_autoencoder_pipeline(
        dataset_names=dataset_name,
        dataset_curves=dataset,
        Y_well=y_well,
        ref_curves=curves,
        ae_plot_path=str(out_dir / "cnn_ae_light"),
        threshold_percentiles=["elbow", 95],
        epochs=60,
        batch_size=128,
        save_plot=False,
        downsample_factor=config.AE_DOWNSAMPLE_FACTOR,
        per_well=False,
    )

    save_data = {
        "dataset_name": np.array(dataset_name),
        "dataset": np.array(dataset, dtype=object),
        "outlier_features": ae_results,
        "Y_well": y_well,
        "timestamps": timestamps,
    }

    joblib.dump(save_data, out_path, compress=3)
    print(f"  -> Saved light training data to {out_path}")
    print("  -> Light outlier pipeline complete.\n")


if __name__ == "__main__":
    main()
