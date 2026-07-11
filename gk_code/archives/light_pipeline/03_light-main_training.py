import os
import sys
import gc
import argparse
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, "..")
sys.path.insert(0, "../utils/model_training")
import config
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


def load_training_data(exp_path, out_subdir):
    data_path = exp_path / out_subdir / "light_training_data.joblib"
    if not data_path.exists():
        print(f"  -> Skipping {exp_path.name}: '{data_path}' not found.")
        sys.exit(0)
    return joblib.load(data_path)


def load_or_init_results(results_file_path):
    if results_file_path.exists():
        return joblib.load(results_file_path)
    return {}


def make_checkpoint_fn(all_ml_results, results_file_path, clean_title, mode_key):
    def _checkpoint(updated_results):
        all_ml_results[clean_title][mode_key] = updated_results
        joblib.dump(all_ml_results, results_file_path, compress=3)
    return _checkpoint


def main():
    parser = argparse.ArgumentParser(description="Light Main Training Pipeline")
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

    print(f"\n\n{'#'*80}\nSTARTING LIGHT TRAINING FOR: {exp_path.name}\n{'#'*80}")

    results_file_path = out_dir / "light_training_results.joblib"

    training_data = load_training_data(exp_path, args.out_subdir)

    dataset_name = training_data["dataset_name"]
    dataset = training_data["dataset"]
    outlier_features = training_data["outlier_features"]
    y_well = training_data["Y_well"]

    encoder = LabelEncoder()
    y_full = encoder.fit_transform(y_well)

    outlier_filters = [
        None,
        f"cnn_ae_glb_ds{config.AE_DOWNSAMPLE_FACTOR}_label_elbow",
        f"cnn_ae_glb_ds{config.AE_DOWNSAMPLE_FACTOR}_label_95",
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


if __name__ == "__main__":
    main()
