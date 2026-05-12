#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime
import itertools
import math
from pathlib import Path
import json
import os
import sys
import warnings

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / ".cache"
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))

warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores.*",
    category=UserWarning,
)

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import seed
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

import python_libraries.classifiers_v2 as mlfunc
import python_libraries.plotting_func as plotfunc


NMETA = 6
FAKE_AC = False
NN = False
N_SPLITS = 10
HOLDOUT_TEST_SIZE = 0.10
HOLDOUT_RANDOM_STATE = 0
CV_RANDOM_STATE = 0
OVO_N_NEIGHS = 5
KNN_NEIGHBORS = 20
KNN_WEIGHTS = "uniform"
KNN_METRIC = "minkowski"
KNN_ALGORITHM = "auto"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate ACA models on a train_aca-compatible dataset."
    )
    parser.add_argument(
        "dataset_csv",
        nargs="?",
        default=None,
        help="ACA dataset CSV path. If omitted, the script will prompt for it.",
    )
    parser.add_argument(
        "--knn-neighbors",
        type=int,
        default=KNN_NEIGHBORS,
        help="Number of neighbors for KNN-based ACA classification.",
    )
    parser.add_argument(
        "--knn-weights",
        default=KNN_WEIGHTS,
        help="KNN weighting scheme, for example 'uniform' or 'distance'.",
    )
    parser.add_argument(
        "--knn-metric",
        default=KNN_METRIC,
        help="KNN distance metric, for example 'minkowski', 'manhattan', or 'cosine'.",
    )
    parser.add_argument(
        "--knn-algorithm",
        default=KNN_ALGORITHM,
        help="KNN backend algorithm. Use 'auto' unless you need to override it.",
    )
    return parser.parse_args()


def resolve_dataset_path(raw_path: str | None) -> Path:
    if raw_path:
        return Path(raw_path).expanduser().resolve()

    raw = input("Please enter the dataset CSV path: ").strip()
    if not raw:
        raise ValueError("No dataset path provided.")
    return Path(raw).expanduser().resolve()


def current_knn_config() -> dict:
    return {
        "n_neighbors": KNN_NEIGHBORS,
        "weights": KNN_WEIGHTS,
        "metric": KNN_METRIC,
        "algorithm": KNN_ALGORITHM,
    }


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    df = pd.read_csv(csv_path, header=0, index_col=0, low_memory=False)

    # If Target ended up as the index (first column was Target), reset it to a column
    if df.index.name == "Target":
        df = df.reset_index()

    if "Target" not in df.columns:
        raise ValueError("Dataset is missing the required column: Target")

    if df.shape[1] <= NMETA + 1:
        raise ValueError(
            "Dataset does not have enough columns to extract ACA features "
            "using the original notebook logic."
        )

    return df


META_COLS = {"Channel", "PrimerMix", "Target", "Target_cat", "Assay", "Conc", "Exp_id", "MeltPeaks"}


def prepare_training_data(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in META_COLS]
    X_AC = df[feature_cols].values

    encoder = LabelEncoder()
    encoder.fit(df["Target"])
    ytrue = encoder.transform(df["Target"])

    target_counts = df["Target"].value_counts().sort_index()
    min_class_count = int(target_counts.min())

    if len(df) < 20:
        raise ValueError(
            "The original ACA model uses KNeighborsClassifier(n_neighbors=20), "
            "so the dataset must contain at least 20 rows."
        )

    if len(encoder.classes_) < 2:
        raise ValueError("Evaluation requires at least two target classes.")

    if min_class_count < 2:
        raise ValueError(
            "Each target must have at least 2 samples for stratified train/test split."
        )

    if min_class_count < N_SPLITS:
        raise ValueError(
            f"Each target must have at least {N_SPLITS} samples for "
            f"StratifiedKFold(n_splits={N_SPLITS})."
        )

    return X_AC, ytrue, encoder, target_counts


def create_output_dir(csv_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = csv_path.parent / f"{csv_path.stem}_aca_outputs_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def save_classification_outputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    encoder: LabelEncoder,
    prefix: str,
    output_dir: Path,
) -> dict:
    classes = encoder.classes_
    labels = list(range(len(classes)))

    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=classes,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=classes,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.to_csv(output_dir / f"{prefix}_confusion_matrix.csv")

    (output_dir / f"{prefix}_classification_report.txt").write_text(report_text)
    pd.DataFrame(report_dict).transpose().to_csv(
        output_dir / f"{prefix}_classification_report.csv"
    )
    save_json(output_dir / f"{prefix}_classification_report.json", report_dict)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
    plotfunc.plot_confusion_matrix(y_true, y_pred, classes, ax, normalize=False)
    ax.set_title(f"{prefix.replace('_', ' ').title()}\n" + ax.get_title(), fontsize=14, weight="bold")
    ax.set_ylabel("True Target", fontsize=12, weight="bold")
    ax.set_xlabel("Predicted Target", fontsize=12, weight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / f"{prefix}_confusion_matrix.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{prefix}_confusion_matrix.pdf", bbox_inches="tight")
    plt.close(fig)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "classification_report_path": str(output_dir / f"{prefix}_classification_report.txt"),
        "confusion_matrix_path": str(output_dir / f"{prefix}_confusion_matrix.csv"),
    }


def save_prediction_table(
    indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    encoder: LabelEncoder,
    path: Path,
    extra_columns: dict[str, np.ndarray] | None = None,
) -> None:
    payload = {
        "row_index": indices,
        "y_true_encoded": y_true,
        "y_pred_encoded": y_pred,
        "y_true_label": encoder.inverse_transform(y_true),
        "y_pred_label": encoder.inverse_transform(y_pred),
    }

    if extra_columns:
        payload.update(extra_columns)

    pd.DataFrame(payload).to_csv(path, index=False)


def save_model_bundle(
    output_path: Path,
    csv_path: Path,
    df: pd.DataFrame,
    encoder: LabelEncoder,
    clf_AC,
    split_name: str,
) -> None:
    bundle = {
        "model_type": "ACA",
        "dataset_path": str(csv_path),
        "split_name": split_name,
        "nmeta": NMETA,
        "fake_ac": FAKE_AC,
        "nn": NN,
        "targets": list(encoder.classes_),
        "feature_columns": [c for c in df.columns if c not in META_COLS],
        "knn_config": None if NN else current_knn_config(),
        "label_encoder": encoder,
        "classifier": clf_AC,
    }
    joblib.dump(bundle, output_path)


def pair_confusion_metrics_from_notebook(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm_notebook = confusion_matrix(y_pred, y_true)
    tp = cm_notebook[0, 0]
    fp = cm_notebook[0, 1]
    fn = cm_notebook[1, 0]
    tn = cm_notebook[1, 1]

    sensitivity = 100.0 * tp / (tp + fn)
    specificity = 100.0 * tn / (tn + fp)
    accuracy = 100.0 * np.mean(y_true == y_pred)

    return {
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


def plot_pair_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    ax,
) -> np.ndarray:
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    mask = np.eye(len(labels), len(labels))

    ax.imshow(mask, interpolation="nearest", cmap="Greens")

    accuracy = 100.0 * np.mean(y_true == y_pred)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=f"Confusion Matrix\n(Acc: {accuracy:.2f}%)",
        ylabel="True label",
        xlabel="Predicted label",
    )

    fmt = "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.grid(False)
    return cm


def run_one_vs_one_cross_validation(
    X_AC: np.ndarray,
    ytrue: np.ndarray,
    encoder: LabelEncoder,
    output_dir: Path,
) -> dict:
    ovo_dir = output_dir / "one_vs_one"
    ovo_dir.mkdir(exist_ok=True)

    class_indices = list(range(len(encoder.classes_)))
    class_pairs = list(itertools.combinations(class_indices, 2))

    clf = OneVsOneClassifier(
        mlfunc.build_knn_classifier(
            n_neighbors=OVO_N_NEIGHS,
            weights=KNN_WEIGHTS,
            metric=KNN_METRIC,
            algorithm=KNN_ALGORITHM,
        )
    )
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)

    true_each_fold = []
    pred_each_fold = []

    for train_index, test_index in folds.split(X_AC, ytrue):
        X_train, X_test = X_AC[train_index], X_AC[test_index]
        y_train, y_test = ytrue[train_index], ytrue[test_index]

        clf.fit(X_train, y_train)

        trues = {}
        preds = {}

        for estimator, pair in zip(clf.estimators_, class_pairs):
            mask = (y_test == pair[0]) | (y_test == pair[1])
            pair_true = y_test[mask]
            pair_pred_binary = estimator.predict(X_test[mask])
            pair_pred = np.where(pair_pred_binary == 0, pair[0], pair[1]).astype(int)

            trues[pair] = pair_true
            preds[pair] = pair_pred

        true_each_fold.append(trues)
        pred_each_fold.append(preds)

    summary_rows = []
    ncols = min(5, max(1, len(class_pairs)))
    nrows = math.ceil(len(class_pairs) / ncols)
    grid_fig, grid_axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.5 * ncols, 3.8 * nrows),
        dpi=300,
    )
    grid_axes = np.atleast_1d(grid_axes).ravel()

    for i, pair in enumerate(class_pairs):
        pair_name = f"{encoder.classes_[pair[0]]}_vs_{encoder.classes_[pair[1]]}"
        pair_title = f"{encoder.classes_[pair[0]]} vs {encoder.classes_[pair[1]]}"
        y_true_pair = np.concatenate([fold[pair] for fold in true_each_fold])
        y_pred_pair = np.concatenate([fold[pair] for fold in pred_each_fold])

        notebook_metrics = pair_confusion_metrics_from_notebook(y_true_pair, y_pred_pair)

        pair_true_binary = np.where(y_true_pair == pair[0], 0, 1)
        pair_pred_binary = np.where(y_pred_pair == pair[0], 0, 1)

        pair_cm = confusion_matrix(pair_true_binary, pair_pred_binary, labels=[0, 1])
        pd.DataFrame(
            pair_cm,
            index=[encoder.classes_[pair[0]], encoder.classes_[pair[1]]],
            columns=[encoder.classes_[pair[0]], encoder.classes_[pair[1]]],
        ).to_csv(ovo_dir / f"{pair_name}_confusion_matrix.csv")

        pair_fig, pair_ax = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
        plot_pair_confusion_matrix(
            pair_true_binary,
            pair_pred_binary,
            [encoder.classes_[pair[0]], encoder.classes_[pair[1]]],
            pair_ax,
        )
        pair_ax.set_title(
            pair_title
            + f"\nAcc: {notebook_metrics['accuracy']:.2f}%"
            + f"\nSens: {notebook_metrics['sensitivity']:.2f}%"
            + f"\nSpec: {notebook_metrics['specificity']:.2f}%",
            fontsize=11,
            weight="bold",
        )
        pair_ax.set_xlabel("Predicted label")
        pair_ax.set_ylabel("True label")
        plt.xticks(rotation=0)
        plt.tight_layout()
        pair_fig.savefig(ovo_dir / f"{pair_name}_confusion_matrix.png", bbox_inches="tight")
        pair_fig.savefig(ovo_dir / f"{pair_name}_confusion_matrix.pdf", bbox_inches="tight")
        plt.close(pair_fig)

        plot_pair_confusion_matrix(
            pair_true_binary,
            pair_pred_binary,
            [encoder.classes_[pair[0]], encoder.classes_[pair[1]]],
            grid_axes[i],
        )
        grid_axes[i].set_title(
            pair_title
            + f"\nSens: {notebook_metrics['sensitivity']:.2f}%"
            + f"\nSpec: {notebook_metrics['specificity']:.2f}%",
            fontsize=10,
            weight="bold",
        )
        grid_axes[i].set_xlabel("")
        grid_axes[i].set_ylabel("")
        grid_axes[i].tick_params(axis="x", rotation=0)

        summary_rows.append(
            {
                "targets": pair_title,
                "class_a": encoder.classes_[pair[0]],
                "class_b": encoder.classes_[pair[1]],
                "accuracy": notebook_metrics["accuracy"],
                "sensitivity": notebook_metrics["sensitivity"],
                "specificity": notebook_metrics["specificity"],
                "n_samples": int(len(y_true_pair)),
            }
        )

    for ax in grid_axes[len(class_pairs) :]:
        ax.axis("off")

    plt.tight_layout()
    grid_fig.savefig(output_dir / "one_vs_one_confusion_matrix_grid.png", bbox_inches="tight")
    grid_fig.savefig(output_dir / "one_vs_one_confusion_matrix_grid.pdf", bbox_inches="tight")
    plt.close(grid_fig)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "one_vs_one_summary.csv", index=False)
    save_json(
        output_dir / "one_vs_one_summary.json",
        {
            "n_pairs": int(len(class_pairs)),
            "k_neighbors": OVO_N_NEIGHS,
            "weights": KNN_WEIGHTS,
            "metric": KNN_METRIC,
            "algorithm": KNN_ALGORITHM,
            "n_splits": N_SPLITS,
            "pairs": summary_rows,
        },
    )

    return {
        "n_pairs": int(len(class_pairs)),
        "k_neighbors": OVO_N_NEIGHS,
        "weights": KNN_WEIGHTS,
        "metric": KNN_METRIC,
        "summary_path": str(output_dir / "one_vs_one_summary.csv"),
        "grid_path": str(output_dir / "one_vs_one_confusion_matrix_grid.png"),
    }


def predict_labels(clf_AC, X_AC: np.ndarray) -> np.ndarray:
    if NN:
        return clf_AC.predict(X_AC).argmax(axis=1)
    return clf_AC.predict(X_AC)


def run_holdout_evaluation(
    X_AC: np.ndarray,
    ytrue: np.ndarray,
    df: pd.DataFrame,
    encoder: LabelEncoder,
    csv_path: Path,
    output_dir: Path,
) -> dict:
    X_train, X_test, y_train, y_test, train_index, test_index = train_test_split(
        X_AC,
        ytrue,
        np.arange(len(ytrue)),
        test_size=HOLDOUT_TEST_SIZE,
        random_state=HOLDOUT_RANDOM_STATE,
        shuffle=True,
        stratify=ytrue,
    )

    clf_AC, clf_AC_proba = mlfunc.train_ACA_model(
        X_train, y_train, FAKE_AC, NN, knn_kwargs=current_knn_config()
    )
    y_pred_test = predict_labels(clf_AC, X_test)
    y_pred_train = clf_AC_proba.argmax(axis=1)

    holdout_dir_metrics = save_classification_outputs(
        y_test, y_pred_test, encoder, "holdout_test", output_dir
    )
    train_metrics = save_classification_outputs(
        y_train, y_pred_train, encoder, "holdout_train", output_dir
    )

    save_prediction_table(
        train_index,
        y_train,
        y_pred_train,
        encoder,
        output_dir / "holdout_train_predictions.csv",
    )
    save_prediction_table(
        test_index,
        y_test,
        y_pred_test,
        encoder,
        output_dir / "holdout_test_predictions.csv",
    )

    save_model_bundle(
        output_dir / "aca_model_holdout.joblib",
        csv_path,
        df,
        encoder,
        clf_AC,
        "holdout_train_split",
    )

    summary = {
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "train_accuracy": train_metrics["accuracy"],
        "test_accuracy": holdout_dir_metrics["accuracy"],
    }
    save_json(output_dir / "holdout_summary.json", summary)
    return summary


def run_cross_validation(
    X_AC: np.ndarray,
    ytrue: np.ndarray,
    encoder: LabelEncoder,
    output_dir: Path,
) -> dict:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)

    fold_rows = []
    y_trues = []
    y_preds_AC = []
    prediction_indices = []
    prediction_folds = []

    for fold_id, (train_index, test_index) in enumerate(skf.split(X_AC, ytrue), start=1):
        X_train, X_test = X_AC[train_index], X_AC[test_index]
        y_train, y_test = ytrue[train_index], ytrue[test_index]

        clf_AC, _ = mlfunc.train_ACA_model(
            X_train, y_train, FAKE_AC, NN, knn_kwargs=current_knn_config()
        )
        fold_pred = predict_labels(clf_AC, X_test)

        y_trues.append(y_test)
        y_preds_AC.append(fold_pred)
        prediction_indices.append(test_index)
        prediction_folds.append(np.full(len(test_index), fold_id))

        fold_rows.append(
            {
                "fold": fold_id,
                "train_size": int(len(train_index)),
                "test_size": int(len(test_index)),
                "accuracy": float(accuracy_score(y_test, fold_pred)),
            }
        )

    y_trues_all = np.concatenate(y_trues)
    y_preds_all = np.concatenate(y_preds_AC)
    prediction_indices_all = np.concatenate(prediction_indices)
    prediction_folds_all = np.concatenate(prediction_folds)

    metrics = save_classification_outputs(
        y_trues_all, y_preds_all, encoder, "cross_validation", output_dir
    )

    save_prediction_table(
        prediction_indices_all,
        y_trues_all,
        y_preds_all,
        encoder,
        output_dir / "cross_validation_predictions.csv",
        extra_columns={"fold": prediction_folds_all},
    )

    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(output_dir / "cross_validation_fold_metrics.csv", index=False)

    summary = {
        "n_splits": N_SPLITS,
        "mean_accuracy": float(folds_df["accuracy"].mean()),
        "std_accuracy": float(folds_df["accuracy"].std(ddof=0)),
        "overall_accuracy": metrics["accuracy"],
    }
    save_json(output_dir / "cross_validation_summary.json", summary)
    return summary


def train_final_model(
    X_AC: np.ndarray,
    ytrue: np.ndarray,
    df: pd.DataFrame,
    encoder: LabelEncoder,
    csv_path: Path,
    output_dir: Path,
) -> Path:
    clf_AC, _ = mlfunc.train_ACA_model(
        X_AC, ytrue, FAKE_AC, NN, knn_kwargs=current_knn_config()
    )
    output_path = output_dir / "aca_model_full.joblib"
    save_model_bundle(output_path, csv_path, df, encoder, clf_AC, "full_dataset")
    return output_path


def main() -> int:
    global KNN_NEIGHBORS, KNN_WEIGHTS, KNN_METRIC, KNN_ALGORITHM
    seed(10)
    CACHE_DIR.mkdir(exist_ok=True)
    (CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)

    try:
        args = parse_args()
        csv_path = resolve_dataset_path(args.dataset_csv)
        KNN_NEIGHBORS = args.knn_neighbors
        KNN_WEIGHTS = args.knn_weights
        KNN_METRIC = args.knn_metric
        KNN_ALGORITHM = args.knn_algorithm
        if KNN_METRIC == "cosine" and KNN_ALGORITHM == "auto":
            KNN_ALGORITHM = "brute"

        print(f"Loading dataset: {csv_path}")

        df = load_dataset(csv_path)
        X_AC, ytrue, encoder, target_counts = prepare_training_data(df)
        output_dir = create_output_dir(csv_path)

        print(f"Output directory: {output_dir}")
        print(f"Rows: {len(df)}")
        print(f"Detected targets: {', '.join(encoder.classes_)}")
        print(f"ACA feature matrix shape: {X_AC.shape}")
        if not NN:
            print(f"KNN config: {current_knn_config()}")

        dataset_summary = {
            "dataset_path": str(csv_path),
            "rows": int(len(df)),
            "nmeta": NMETA,
            "fake_ac": FAKE_AC,
            "nn": NN,
            "aca_feature_count": int(X_AC.shape[1]),
            "targets": list(encoder.classes_),
            "target_counts": {str(k): int(v) for k, v in target_counts.items()},
            "knn_config": current_knn_config(),
            "holdout_test_size": HOLDOUT_TEST_SIZE,
            "holdout_random_state": HOLDOUT_RANDOM_STATE,
            "cross_validation_splits": N_SPLITS,
            "cross_validation_random_state": CV_RANDOM_STATE,
            "one_vs_one_k_neighbors": OVO_N_NEIGHS,
        }
        save_json(output_dir / "dataset_summary.json", dataset_summary)

        print("Running holdout train/test split evaluation...")
        holdout_summary = run_holdout_evaluation(
            X_AC, ytrue, df, encoder, csv_path, output_dir
        )
        print(
            f"Holdout accuracy: train={holdout_summary['train_accuracy']:.4f}, "
            f"test={holdout_summary['test_accuracy']:.4f}"
        )

        print("Running 10-fold cross validation...")
        cv_summary = run_cross_validation(X_AC, ytrue, encoder, output_dir)
        print(
            f"Cross-validation accuracy: overall={cv_summary['overall_accuracy']:.4f}, "
            f"mean={cv_summary['mean_accuracy']:.4f}"
        )

        print("Running one-vs-one pairwise classification...")
        ovo_summary = run_one_vs_one_cross_validation(X_AC, ytrue, encoder, output_dir)
        print(
            f"One-vs-one evaluation completed: {ovo_summary['n_pairs']} pairs, "
            f"k={ovo_summary['k_neighbors']}"
        )

        print("Training final ACA classifier on the full dataset...")
        final_model_path = train_final_model(
            X_AC, ytrue, df, encoder, csv_path, output_dir
        )

        run_summary = {
            "output_dir": str(output_dir),
            "final_model_path": str(final_model_path),
            "holdout_summary_path": str(output_dir / "holdout_summary.json"),
            "cross_validation_summary_path": str(output_dir / "cross_validation_summary.json"),
            "one_vs_one_summary_path": str(output_dir / "one_vs_one_summary.csv"),
        }
        save_json(output_dir / "run_summary.json", run_summary)

        print("Pipeline completed successfully.")
        print(f"Final ACA model: {final_model_path}")
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
