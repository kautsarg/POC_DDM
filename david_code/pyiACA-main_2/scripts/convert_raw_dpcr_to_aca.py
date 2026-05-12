#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import python_libraries.fitting_func as fitfunc


META_COLUMNS = [
    "Channel",
    "PrimerMix",
    "Target",
    "Assay",
    "Conc",
    "Exp_id",
    "MeltPeaks",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert raw dPCR well files (for example 10_Hadv_M.txt) into an "
            "ACA-compatible training CSV."
        )
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=None,
        help="Directory containing raw dPCR txt files. If omitted, the script will prompt for it.",
    )
    parser.add_argument(
        "output_csv",
        nargs="?",
        type=Path,
        default=None,
        help="Optional output CSV path. Defaults to dataframe_saved/<folder>_aca_ready.csv.",
    )
    parser.add_argument(
        "--background-cycles",
        type=int,
        default=5,
        help="Number of initial cycles used for mean background subtraction.",
    )
    parser.add_argument(
        "--drop-initial-cycles",
        type=int,
        default=5,
        help="Number of leading cycles to discard after preprocessing.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="Normalize each partition curve to 0-1 after clipping negatives to 0.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_false",
        dest="normalize",
        help="Skip per-curve normalization.",
    )
    parser.add_argument(
        "--exp-id",
        default=None,
        help="Value to store in the Exp_id column. Defaults to input directory name.",
    )
    return parser.parse_args()


def prompt_for_input_dir() -> Path:
    raw = input("Please enter the Raw Data folder path: ").strip().strip('"').strip("'")
    if not raw:
        raise ValueError("No Raw Data folder path provided.")
    return Path(raw).expanduser().resolve()


def default_output_csv(input_dir: Path) -> Path:
    safe_name = input_dir.name.strip().replace(" ", "_")
    return PROJECT_ROOT / "dataframe_saved" / f"{safe_name}_aca_ready.csv"


def resolve_input_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    input_dir = args.input_dir.expanduser().resolve() if args.input_dir else prompt_for_input_dir()
    output_csv = (
        args.output_csv.expanduser().resolve()
        if args.output_csv
        else default_output_csv(input_dir)
    )
    return input_dir, output_csv


def parse_well_filename(path: Path) -> tuple[int, str]:
    parts = path.stem.split("_", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Expected filename like '10_Hadv_M.txt', but got: {path.name}"
        )

    well_str, target = parts
    try:
        well = int(well_str)
    except ValueError as exc:
        raise ValueError(f"Well prefix is not numeric in filename: {path.name}") from exc

    return well, target


def split_target_name(target: str) -> tuple[str, str]:
    if "_" not in target:
        return target, target
    primer_mix, assay = target.rsplit("_", 1)
    return primer_mix, assay


def normalize_curves(curves: pd.DataFrame) -> pd.DataFrame:
    values = curves.to_numpy(dtype=float, copy=True)
    values[values < 0] = 0.0

    row_min = values.min(axis=1, keepdims=True)
    row_max = values.max(axis=1, keepdims=True)
    denom = row_max - row_min
    denom[denom == 0] = 1.0

    normalized = (values - row_min) / denom
    return pd.DataFrame(normalized, index=curves.index, columns=curves.columns)


def preprocess_raw_file(
    path: Path,
    background_cycles: int,
    drop_initial_cycles: int,
    normalize: bool,
) -> pd.DataFrame:
    raw_curves = fitfunc.extract_curves(path)

    if background_cycles < 0 or drop_initial_cycles < 0:
        raise ValueError("Cycle parameters must be non-negative.")

    if background_cycles > raw_curves.shape[0]:
        raise ValueError(
            f"{path.name} only has {raw_curves.shape[0]} cycles, so "
            f"--background-cycles={background_cycles} is too large."
        )

    if drop_initial_cycles >= raw_curves.shape[0]:
        raise ValueError(
            f"{path.name} only has {raw_curves.shape[0]} cycles, so "
            f"--drop-initial-cycles={drop_initial_cycles} removes everything."
        )

    if background_cycles:
        raw_curves = fitfunc.remove_background(
            raw_curves,
            order=0,
            n_ct_fit=background_cycles,
            n_ct_skip=0,
        )

    curves = raw_curves.transpose().astype(float)

    if normalize:
        curves = normalize_curves(curves)

    if drop_initial_cycles:
        curves = curves.iloc[:, drop_initial_cycles:]

    feature_count = curves.shape[1]
    curves.columns = [f"{float(i):.1f}" for i in range(1, feature_count + 1)]
    curves = curves.reset_index(drop=True)
    return curves


def build_metadata_frame(
    well: int,
    target: str,
    curve_count: int,
    exp_id: str,
) -> pd.DataFrame:
    primer_mix, assay = split_target_name(target)
    return pd.DataFrame(
        {
            "Channel": [f"well{well:02d}"] * curve_count,
            "PrimerMix": [primer_mix] * curve_count,
            "Target": [target] * curve_count,
            "Assay": [assay] * curve_count,
            "Conc": [np.nan] * curve_count,
            "Exp_id": [exp_id] * curve_count,
            "MeltPeaks": [np.nan] * curve_count,
        }
    )


def convert_directory(
    input_dir: Path,
    output_csv: Path,
    background_cycles: int,
    drop_initial_cycles: int,
    normalize: bool,
    exp_id: str | None,
) -> tuple[pd.DataFrame, list[str]]:
    input_dir = input_dir.expanduser().resolve()
    output_csv = output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.txt"), key=lambda path: parse_well_filename(path)[0])
    if not files:
        raise FileNotFoundError(f"No .txt files found in {input_dir}")

    exp_value = exp_id or input_dir.name.replace(" ", "_")
    frames: list[pd.DataFrame] = []
    skipped_files: list[str] = []

    for path in files:
        if path.stat().st_size == 0:
            skipped_files.append(f"{path.name}: empty file")
            continue

        well, target = parse_well_filename(path)
        curves = preprocess_raw_file(
            path,
            background_cycles=background_cycles,
            drop_initial_cycles=drop_initial_cycles,
            normalize=normalize,
        )
        metadata = build_metadata_frame(
            well=well,
            target=target,
            curve_count=len(curves),
            exp_id=exp_value,
        )
        frames.append(pd.concat([metadata, curves], axis=1))

    if not frames:
        raise ValueError("No usable raw data files were found after filtering empty files.")

    dataset = pd.concat(frames, ignore_index=True)
    dataset.index.name = ""
    dataset.to_csv(output_csv)
    return dataset, skipped_files


def main() -> None:
    args = parse_args()
    input_dir, output_csv = resolve_input_output_paths(args)
    dataset, skipped_files = convert_directory(
        input_dir=input_dir,
        output_csv=output_csv,
        background_cycles=args.background_cycles,
        drop_initial_cycles=args.drop_initial_cycles,
        normalize=args.normalize,
        exp_id=args.exp_id,
    )

    excel_path = output_csv.with_suffix(".xlsx")
    excel_written = False
    excel_error = None
    try:
        dataset.to_excel(excel_path)
        excel_written = True
    except Exception as exc:
        excel_error = exc

    print(f"Input folder: {input_dir}")
    print(f"Saved ACA dataset to: {output_csv}")
    print(f"Rows: {len(dataset)}")
    print(f"Feature columns: {dataset.shape[1] - len(META_COLUMNS)}")
    print(f"Detected targets: {', '.join(sorted(dataset['Target'].unique()))}")
    print("Target counts:")
    print(dataset["Target"].value_counts().sort_index().to_string())
    if skipped_files:
        print("Skipped files:")
        for item in skipped_files:
            print(f"- {item}")
    if excel_written:
        print(f"Saved Excel copy to: {excel_path}")
    elif excel_error is not None:
        print(f"Excel export skipped: {excel_error}")


if __name__ == "__main__":
    main()
