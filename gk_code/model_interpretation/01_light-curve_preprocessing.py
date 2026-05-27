import os
import sys
import argparse
from pathlib import Path

import numpy as np
import joblib

sys.path.insert(0, "..")
from titan.load_and_preprocessing import titan_load_and_preprocessing

sys.path.insert(0, "../outlier_detection")
import config


def get_exp_paths(exp_folder):
    return sorted(
        [
            Path(exp_folder, name)
            for name in os.listdir(exp_folder)
            if os.path.isdir(os.path.join(exp_folder, name)) and name not in [".DS_Store"]
        ]
    )


def reconstruct_data(exp_data, attr_str):
    vstacked = None
    well_ids = []

    for well in exp_data.wells_list:
        temp_x = getattr(well, attr_str).copy()
        temp_x = np.swapaxes(temp_x, 0, 1)

        if vstacked is None:
            vstacked = temp_x
        else:
            vstacked = np.vstack((vstacked, temp_x))

        well_ids.append(temp_x.shape[0])

    x_time = exp_data.wells_list[0].time
    y_well = []
    for label, count in enumerate(well_ids):
        y_well.extend([label] * count)

    return x_time, np.array(y_well), vstacked


def main():
    parser = argparse.ArgumentParser(description="Light Curve Preprocessing (Original Curves Only)")
    parser.add_argument("--task_id", type=int, default=0, help="Array Job ID")
    parser.add_argument("--exp_folder", type=str, default=config.DEFAULT_EXP_FOLDER, help="Path to experiment datasets")
    parser.add_argument("--out_subdir", type=str, default="light_pipeline", help="Output subfolder under each experiment")
    args = parser.parse_args()

    exp_paths = get_exp_paths(args.exp_folder)
    if args.task_id >= len(exp_paths):
        print(f"Task ID {args.task_id} is out of bounds for {len(exp_paths)} folders. Exiting.")
        sys.exit(0)

    exp_path = exp_paths[args.task_id]
    out_dir = exp_path / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    save_path = out_dir / "light_preprocessed_curves.joblib"
    if save_path.exists():
        print(f"Cache hit: {exp_path}")
        print("  -> Light preprocessing already exists. Skipping.")
        sys.exit(0)

    print(f"Processing Experiment (Light): {exp_path}")

    exp = titan_load_and_preprocessing(
        exp_path,
        n_wells=config.N_WELLS,
        start_type="temperature",
        end_time_min=60,
        n_a_type=config.N_A_TYPE,
        print_status=False,
        plt_gain_calib=False,
        save_gain_calib=False,
    )

    x_time, y_well, curves_2d = reconstruct_data(exp, attr_str="well_2d_bs_active")

    baseline_value = 0.0
    if np.min(curves_2d) < 0:
        baseline_value = float(-np.min(curves_2d) + 1e-9)
        curves_2d = curves_2d + baseline_value

    save_data = {
        "curves": {"ori_curves": curves_2d},
        "timestamps": x_time,
        "well_labels": y_well,
        "baseline_value": baseline_value,
    }

    joblib.dump(save_data, save_path, compress=3)
    print(f"  -> Saved light curves to {save_path}")
    print("  -> Light preprocessing complete.\n")


if __name__ == "__main__":
    main()
