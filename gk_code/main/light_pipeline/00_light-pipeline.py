import os
import sys
import gc
import argparse
from pathlib import Path

import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, "../../..")
from titan_v4.load_and_preprocessing import titan_load_and_preprocessing

sys.path.insert(0, "..")
sys.path.insert(0, "../utils/02_outlier_detection")
sys.path.insert(0, "../utils/model_training")
import config
from cnn_autoencoder_outlier import run_cnn_autoencoder_pipeline
from model_utils import evaluate_outlier_filters, set_global_determinism

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
set_global_determinism(0)


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


def run_step1(exp_path, out_dir):
    save_path = out_dir / "light_preprocessed_curves.joblib"
    if save_path.exists():
        print("  -> Step 1 cache hit. Skipping light preprocessing.")
        return save_path

    print("  -> Step 1: Original curve extraction")
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
    return save_path


def run_step2(exp_path, out_dir, in_path):
    out_path = out_dir / "light_training_data.joblib"
    if out_path.exists():
        print("  -> Step 2 cache hit. Skipping light outlier detection.")
        return out_path

    print("  -> Step 2: Global CNN AE outlier detection")
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
        threshold_percentiles=["elbow"],
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
    return out_path


def load_or_init_results(results_file_path):
    if results_file_path.exists():
        return joblib.load(results_file_path)
    return {}


def make_checkpoint_fn(all_ml_results, results_file_path, clean_title, mode_key):
    def _checkpoint(updated_results):
        all_ml_results[clean_title][mode_key] = updated_results
        joblib.dump(all_ml_results, results_file_path, compress=3)
    return _checkpoint


def run_step3(exp_path, out_dir, training_data_path):
    results_file_path = out_dir / "light_training_results.joblib"

    training_data = joblib.load(training_data_path)

    dataset_name = training_data["dataset_name"]
    dataset = training_data["dataset"]
    outlier_features = training_data["outlier_features"]
    y_well = training_data["Y_well"]

    encoder = LabelEncoder()
    y_full = encoder.fit_transform(y_well)

    outlier_filters = [
        None,
        f"cnn_ae_glb_ds{config.AE_DOWNSAMPLE_FACTOR}_label_elbow",
    ]

    print(f"[*] Found {len(outlier_filters)-1} Light Outlier Filters to test.")

    all_ml_results = load_or_init_results(results_file_path)

    total_datasets = len(dataset_name)
    trained_curve = dataset[0].copy()

    shared_ref_baseline = None
    for ct in all_ml_results:
        if "Light" in all_ml_results[ct] and None in all_ml_results[ct]["Light"]:
            shared_ref_baseline = all_ml_results[ct]["Light"][None]
            print("  [*] Found cached Light Baseline. Will skip redundant baseline training for all datasets.")
            break

    for idx, (name, features_df) in enumerate(zip(dataset_name, outlier_features)):
        clean_title = str(name).replace("_", " ").title()
        progress_pct = ((idx + 1) / total_datasets) * 100

        print(f"\n{'='*75}")
        print(f"[{idx+1}/{total_datasets} | {progress_pct:.1f}%] Processing Dataset: {clean_title}")
        print(f"{'='*75}")

        if clean_title not in all_ml_results:
            all_ml_results[clean_title] = {}

        cached_ref = all_ml_results[clean_title].get("Light", {})

        if shared_ref_baseline is not None and None not in cached_ref:
            cached_ref[None] = shared_ref_baseline

        checkpoint_ref = make_checkpoint_fn(all_ml_results, results_file_path, clean_title, "Light")

        res_ref = evaluate_outlier_filters(
            trained_curve,
            features_df,
            y_full,
            outlier_filters,
            clean_title,
            mode_name="Light",
            cached_results=cached_ref,
            models=["cnn", "lstm", "gru", "transformer"],
            checkpoint_fn=checkpoint_ref,
        )

        if shared_ref_baseline is None and None in res_ref:
            shared_ref_baseline = res_ref[None]

        all_ml_results[clean_title]["Light"] = res_ref

        joblib.dump(all_ml_results, results_file_path, compress=3)

    gc.collect()
    print("  -> Light training complete.\n")
    return results_file_path


def run_pipeline(task_id, exp_folder, out_subdir):
    exp_paths = get_exp_paths(exp_folder)
    if task_id >= len(exp_paths):
        print(f"Task ID {task_id} is out of bounds for {len(exp_paths)} folders. Exiting.")
        sys.exit(0)

    exp_path = exp_paths[task_id]
    out_dir = exp_path / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n\n{'#'*80}\nSTARTING LIGHT PIPELINE FOR: {exp_path.name}\n{'#'*80}")

    step1_path = run_step1(exp_path, out_dir)
    step2_path = run_step2(exp_path, out_dir, step1_path)
    run_step3(exp_path, out_dir, step2_path)


def main():
    parser = argparse.ArgumentParser(description="Light Pipeline (Steps 01-03 Combined)")
    parser.add_argument("--task_id", type=int, default=0, help="Array Job ID")
    parser.add_argument("--exp_folder", type=str, default=config.DEFAULT_EXP_FOLDER)
    parser.add_argument("--out_subdir", type=str, default="light_pipeline", help="Output subfolder under each experiment")
    args = parser.parse_args()

    run_pipeline(args.task_id, args.exp_folder, args.out_subdir)


if __name__ == "__main__":
    main()
