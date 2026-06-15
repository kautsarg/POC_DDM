from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import sys
import numpy as np


sys.path.insert(0, "../../titan_v6")
import load_functions as load_v6
from load_and_preprocessing import titan_load_and_preprocessing as titan_load_and_preprocessing_v6


def parse_experiment_name(exp_path: Path) -> Optional[str]:
    path_str = str(exp_path)
    if "KHz_U_" not in path_str:
        return None
    name = path_str.split("KHz_U_")[1]
    return name.rstrip("\\/")  # remove trailing separators


def normalize_n_a_type(n_a_type: str) -> str:
    return str(n_a_type).lower()


def resolve_ref_indices(exp_path: Path, n_a_type_norm: str, vref_ref_idx) -> Tuple[List[int], Optional[np.ndarray], Optional[dict]]:
    vref_vect = None
    dvref_tele = None
    ref_indices: List[int] = []

    if n_a_type_norm in ("v05", "v06"):
        want_all = (vref_ref_idx is None or (isinstance(vref_ref_idx, str) and vref_ref_idx.lower() == "all"))
        try:
            exp_path_vref = load_v6.find_most_recent_vref(exp_path)
            n_vrefs, vref_vect, dvref_tele = load_v6.load_vref_sweep(exp_path, return_telemetry=True)

            if dvref_tele is not None and dvref_tele.get("coarse") is not None:
                print(
                    f"Loaded dvref multi payload from {exp_path_vref.name}: "
                    f"n_vrefs={n_vrefs}, vrefs={dvref_tele.get('vrefs')}"
                )
                if dvref_tele.get("model") is not None:
                    m = dvref_tele["model"]
                    print(
                        "TTN model: "
                        f"pass_fail={m.get('pass_fail')} "
                        f"total_dcnt_x16={m.get('total_dcnt_x16')} "
                        f"threshold={m.get('coarse_dcnt_x16_threshold')}"
                    )
            else:
                print(f"Loaded vref sweep from {exp_path_vref.name}: n_vrefs={n_vrefs}")

            if want_all and n_vrefs:
                ref_indices = list(range(int(n_vrefs)))
            else:
                if isinstance(vref_ref_idx, (list, tuple, np.ndarray)):
                    ref_indices = [int(x) for x in vref_ref_idx]
                else:
                    ref_indices = [int(vref_ref_idx)]
        except Exception as exc:
            print(f"Could not load vref sweep for {exp_path.name}: {exc}")
            ref_indices = [-1 if want_all else int(vref_ref_idx)]
            vref_vect = None
            dvref_tele = None
    else:
        ref_indices = [vref_ref_idx]

    return ref_indices, vref_vect, dvref_tele


def make_slice_label(n_a_type_norm: str, ref_idx: int, vref_vect: Optional[np.ndarray]) -> Tuple[Optional[str], Optional[float], int]:
    ref_idx_norm = ref_idx
    vref_value = None

    if vref_vect is not None and len(vref_vect) > 0:
        if ref_idx_norm < 0:
            ref_idx_norm = len(vref_vect) + ref_idx_norm
        if 0 <= ref_idx_norm < len(vref_vect):
            try:
                vref_value = float(vref_vect[ref_idx_norm])
            except Exception:
                vref_value = None

    slice_label = None
    if n_a_type_norm in ("v05", "v06"):
        if vref_value is not None:
            slice_label = f"vref_idx={ref_idx_norm}_vref={vref_value:g}"
        else:
            slice_label = f"vref_idx={ref_idx_norm}"

    return slice_label, vref_value, ref_idx_norm

def load_and_preprocess_v6(exp_path: Path, n_wells: int, n_a_type: str, vref_ref_idx):
    experiment_name = parse_experiment_name(exp_path)
    if experiment_name is None:
        print(f"[SKIP] {exp_path.name}: invalid experiment directory (missing 'KHz_U_').")
        return []

    readout_files = list(exp_path.glob("*readout*.bin"))
    if not readout_files:
        find_active_files = list(exp_path.glob("*find_active*.bin"))
        if not find_active_files:
            print(f"[WARN] {exp_path.name}: no readout or find_active files found.")

    print(f"[INFO] {exp_path.name}: found {len(readout_files)} readout file(s). Processing...")

    n_a_type_norm = normalize_n_a_type(n_a_type)
    ref_indices, vref_vect, _dvref_tele = resolve_ref_indices(exp_path, n_a_type_norm, vref_ref_idx)
    is_multi_vref = len(ref_indices) > 1

    exps = []
    total_slices = len(ref_indices)

    for i_slice, ref_idx in enumerate(ref_indices, start=1):
        slice_label, _vref_value, ref_idx_norm = make_slice_label(n_a_type_norm, int(ref_idx), vref_vect)
        experiment_name_slice = experiment_name
        if slice_label is not None and (is_multi_vref or (vref_ref_idx not in (-1, "-1"))):
            experiment_name_slice = f"{experiment_name}__{slice_label}"

        print(f"[INFO] Slice {i_slice}/{total_slices}: {experiment_name_slice}")

        try:
            exps.append(
                titan_load_and_preprocessing_v6(
                    exp_path,
                    n_wells=n_wells,
                    start_type="temperature",
                    end_time_min=60,
                    n_a_type=n_a_type,
                    print_status=True,
                    plt_gain_calib=False,
                    save_gain_calib=False,
                    plot_gain_3d=False,
                    ref_idx=int(ref_idx) if n_a_type_norm in ("v05", "v06") else ref_idx,
                )
            )
            print(f"[OK] Completed slice {i_slice}/{total_slices} (ref_idx={ref_idx_norm}).")
        except Exception as exc:
            print(f"[ERROR] {exp_path.name} | ref_idx={ref_idx}: {exc}")
            print("[INFO] Skipping this ref slice and continuing...")

    return exps