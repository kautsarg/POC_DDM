#!/usr/bin/env python3
"""
Predict chip heating curves using TEMP_P / TEMP_I / TEMP_D from optimize_pid.

Uses the same FOPDT plant + discrete PID loop as ``optimize_pid.py``:
  - Plant identified from measured ``*_temp_log.bin`` (or readout temp_lin)
  - One simulated temperature trace per tuning model in the recommendations JSON
  - Measured heating ramp overlaid for comparison

Example:
    python test_pid_heating_prediction.py \\
        "C:\\Users\\Matthew L\\OneDrive - ProtonDx\\Data\\PID Model\\D20260519_E00_C00_F4500KHz_U_pid_measure_01"

    python test_pid_heating_prediction.py --recommendations-json "....json" --show
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Non-interactive backend avoids Tcl/Tk (_tkinter) on Windows when Tk is missing/broken.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from optimize_pid import _pick_temperature_trace, load_pid_ingest, run_models
from pid_firmware_sim import (
    DEFAULT_PWM_MAX,
    DEFAULT_PWM_OFFSET,
    FirmwareSimConfig,
    PidGains,
    build_plant_from_measured,
    simulate_firmware_pid,
)

DEFAULT_EXP = Path.home() / "OneDrive - ProtonDx" / "Data" / "PID Model" / "D20260519_E00_C00_F4500KHz_U_pid_measure_01"
DEFAULT_JSON = (
    Path.home()
    / "OneDrive - ProtonDx"
    / "Data"
    / "Saved Images"
    / "pid_measure_01_pid_recommendations.json"
)

# Models that are meaningful on the heating surrogate (skip empty back-calculate).
DEFAULT_MODEL_IDS = (
    "ziegler_nichols",
    "cohen_coon",
    "itae_fopdt",
    "differential_evolution",
    "relay_autotune",
)

MODEL_COLORS = {
    "ziegler_nichols": "#E69F00",
    "cohen_coon": "#56B4E9",
    "itae_fopdt": "#009E73",
    "differential_evolution": "#CC79A7",
    "relay_autotune": "#D55E00",
    "backcalculate": "#999999",
}


def _gains_from_rec(rec: Dict[str, Any]) -> PidGains:
    g = rec["gains"]
    return PidGains(
        kp=float(g["TEMP_P"]),
        ki=float(g["TEMP_I"]),
        kd=float(g["TEMP_D"]),
        notes=rec.get("notes", ""),
    )


def _load_recommendations(
    exp_path: Path,
    json_path: Optional[Path],
    model_ids: Optional[List[str]],
) -> Tuple[List[Dict[str, Any]], Path]:
    if json_path is not None and json_path.is_file():
        with open(json_path, encoding="utf-8") as fp:
            payload = json.load(fp)
        recs = payload["recommendations"]
        return recs, json_path

    ingest = load_pid_ingest(exp_path)
    mids = model_ids or list(DEFAULT_MODEL_IDS)
    live = run_models(ingest, model_ids=mids)
    recs = []
    for r in live:
        recs.append(
            {
                "model_id": r.model_id,
                "name": r.name,
                "score": r.score,
                "gains": r.gains.as_dict(),
                "notes": r.gains.notes,
                "details": r.details,
            }
        )
    return recs, exp_path


def _simulation_horizon(time_s: np.ndarray, duration_mult: float = 1.0) -> Tuple[float, float, int]:
    """Return (dt, duration_s, n_steps) for the closed-loop simulation."""
    t = np.asarray(time_s, dtype=float)
    if t.size < 2:
        dt = 1.0
        duration = 120.0
    else:
        dt = float(np.median(np.diff(t)))
        dt = max(dt, 1e-3)
        duration = float(t[-1] - t[0])
        if duration <= 0:
            duration = dt * max(t.size - 1, 1)
    duration = max(duration * duration_mult, 30.0)
    n = int(duration / dt) + 1
    n = min(n, 8000)
    return dt, duration, n


def build_predictions(
    ingest,
    recommendations: List[Dict[str, Any]],
    model_ids: Optional[List[str]] = None,
    duration_mult: float = 1.0,
    sim_cfg: Optional[FirmwareSimConfig] = None,
) -> Dict[str, Any]:
    """Run firmware-aligned FOPDT + PID simulations for each requested model."""
    t_meas, y_meas = _pick_temperature_trace(ingest)
    if sim_cfg is None:
        sim_cfg = FirmwareSimConfig()

    k_plant, tau, l_delay, dt = build_plant_from_measured(
        t_meas, y_meas, pwm_max=sim_cfg.pwm_max
    )
    _, _, n_steps = _simulation_horizon(t_meas, duration_mult=duration_mult)

    y0 = float(y_meas[0])
    if ingest.has_pid_tail and np.any(ingest.reg_ref > 0):
        setpoint = float(np.median(ingest.reg_ref[ingest.reg_ref > 0]))
    else:
        setpoint = float(np.percentile(y_meas, 95))

    setpoint_lin = setpoint

    allow = set(model_ids) if model_ids else None
    curves: Dict[str, Dict[str, Any]] = {}

    for rec in recommendations:
        mid = rec["model_id"]
        if allow is not None and mid not in allow:
            continue
        gains = _gains_from_rec(rec)
        if gains.kp == 0.0 and gains.ki == 0.0 and gains.kd == 0.0:
            continue

        t_sim, y_sim, pwm_sim, _pid_state = simulate_firmware_pid(
            gains,
            setpoint_lin,
            y0,
            n_steps,
            dt,
            k_plant,
            tau,
            l_delay,
            cfg=sim_cfg,
        )
        curves[mid] = {
            "name": rec["name"],
            "gains": gains,
            "time_s": t_sim,
            "temp_lin": y_sim,
            "pwm": pwm_sim,
            "score": rec.get("score"),
        }

    return {
        "measured": {"time_s": t_meas, "temp_lin": y_meas},
        "plant": {
            "K": k_plant,
            "tau": tau,
            "L": l_delay,
            "dt": dt,
            "n_steps": n_steps,
            "pwm_max": sim_cfg.pwm_max,
            "pwm_offset": sim_cfg.pwm_offset,
        },
        "setpoint_lin": setpoint_lin,
        "y0": y0,
        "curves": curves,
    }


def plot_new_pid_only(
    bundle: Dict[str, Any],
    model_id: str,
    title: str,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    """Measured ramp vs a single recommended TEMP_P / TEMP_I / TEMP_D set."""
    if model_id not in bundle["curves"]:
        raise KeyError(
            f"Model {model_id!r} not in simulation results. "
            f"Available: {list(bundle['curves'].keys())}"
        )

    item = bundle["curves"][model_id]
    g = item["gains"]
    t_meas = bundle["measured"]["time_s"]
    y_meas = bundle["measured"]["temp_lin"]
    setpoint = bundle["setpoint_lin"]
    plant = bundle["plant"]

    fig, (ax_temp, ax_pwm) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, constrained_layout=True)

    ax_temp.plot(t_meas, y_meas, color="#333333", linewidth=2.0, label="Measured (current run)")
    ax_temp.plot(
        item["time_s"],
        item["temp_lin"],
        color="#D55E00",
        linewidth=2.6,
        label=(
            f"Predicted with new PID  "
            f"(P={g.kp:.4f}, I={g.ki:.4f}, D={g.kd:.4f})"
        ),
    )
    ax_temp.axhline(setpoint, color="#0072B2", linestyle="--", linewidth=1.4, label=f"Setpoint ({setpoint:.0f})")
    ax_temp.set_ylabel("Linearized temperature")
    ax_temp.set_title(title, fontsize=12, fontweight="bold")
    ax_temp.grid(True, alpha=0.35)
    ax_temp.legend(loc="lower right", fontsize=9)

    ax_pwm.plot(item["time_s"], item["pwm"], color="#D55E00", linewidth=2.2, label="Heater PWM (new PID)")
    pwm_cap = plant.get("pwm_max", DEFAULT_PWM_MAX)
    ax_pwm.axhline(pwm_cap, color="#0072B2", linestyle=":", linewidth=1.2, label=f"PWM max ({pwm_cap:.0f}%)")
    ax_pwm.set_xlabel("Time (s)")
    ax_pwm.set_ylabel("Heater PWM (%)")
    ax_pwm.set_ylim(-2, max(pwm_cap + 5, 45))
    ax_pwm.grid(True, alpha=0.35)
    ax_pwm.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        f"Firmware-aligned sim — K={plant['K']:.2f} (cal @ {pwm_cap:.0f}% PWM), "
        f"tau={plant['tau']:.2f}s, L={plant['L']:.2f}s",
        fontsize=9,
    )

    _save_and_maybe_show(fig, save_path, show)


def _save_and_maybe_show(fig, save_path: Optional[Path], show: bool) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {save_path}")

    if show and save_path is not None:
        try:
            if sys.platform == "win32":
                os.startfile(str(save_path))
            else:
                import subprocess

                subprocess.run(["xdg-open", str(save_path)], check=False)
        except Exception as exc:
            print(f"Could not open plot viewer ({exc}). Open manually: {save_path}")
    plt.close(fig)


def plot_heating_prediction(
    bundle: Dict[str, Any],
    title: str,
    save_path: Optional[Path] = None,
    show: bool = False,
    highlight_model: str = "differential_evolution",
) -> None:
    """Two-panel figure: temperature and heater PWM."""
    t_meas = bundle["measured"]["time_s"]
    y_meas = bundle["measured"]["temp_lin"]
    setpoint = bundle["setpoint_lin"]
    plant = bundle["plant"]

    fig, (ax_temp, ax_pwm) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)

    ax_temp.plot(t_meas, y_meas, color="black", linewidth=2.2, label="Measured (temp_log)", zorder=10)
    ax_temp.axhline(setpoint, color="#666666", linestyle="--", linewidth=1.2, label=f"Setpoint ({setpoint:.1f})")

    for mid, item in bundle["curves"].items():
        g = item["gains"]
        lw = 2.4 if mid == highlight_model else 1.5
        alpha = 1.0 if mid == highlight_model else 0.85
        color = MODEL_COLORS.get(mid, None)
        label = f"{item['name']}  P={g.kp:.3g} I={g.ki:.3g} D={g.kd:.3g}"
        ax_temp.plot(item["time_s"], item["temp_lin"], color=color, linewidth=lw, alpha=alpha, label=label)
        ax_pwm.plot(item["time_s"], item["pwm"], color=color, linewidth=lw, alpha=alpha, label=label)

    ax_temp.set_ylabel("Linearized temperature")
    ax_temp.set_title(title)
    ax_temp.grid(True, alpha=0.3)
    ax_temp.legend(loc="lower right", fontsize=8)

    ax_pwm.set_xlabel("Time (s)")
    ax_pwm.set_ylabel("Heater PWM (%)")
    pwm_cap = plant.get("pwm_max", DEFAULT_PWM_MAX)
    ax_pwm.axhline(pwm_cap, color="#666666", linestyle=":", linewidth=1.0, label=f"PWM max ({pwm_cap:.0f}%)")
    ax_pwm.set_ylim(-2, max(pwm_cap + 5, 45))
    ax_pwm.grid(True, alpha=0.3)

    subtitle = (
        f"FOPDT surrogate: K={plant['K']:.2g}, tau={plant['tau']:.2f}s, L={plant['L']:.2f}s, "
        f"dt={plant['dt']:.3g}s"
    )
    fig.suptitle(subtitle, fontsize=9, y=1.02)

    _save_and_maybe_show(fig, save_path, show)


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot predicted chip heating for optimize_pid TEMP_P/I/D recommendations.",
    )
    parser.add_argument(
        "exp_path",
        type=Path,
        nargs="?",
        default=DEFAULT_EXP,
        help="Experiment folder with temp_log.bin and optional recommendations JSON",
    )
    parser.add_argument(
        "--recommendations-json",
        type=Path,
        default=None,
        help=f"Path to *_pid_recommendations.json (default: {DEFAULT_JSON.name} if present)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"Model ids to plot (default: {', '.join(DEFAULT_MODEL_IDS)})",
    )
    parser.add_argument(
        "--duration-mult",
        type=float,
        default=1.0,
        help="Simulation length as multiple of measured temp_log duration (default 1.0)",
    )
    parser.add_argument(
        "--pwm-max",
        type=float,
        default=DEFAULT_PWM_MAX,
        help="Heater PWM ceiling %% (L_ttn TEMP_PWM_MAX, stock default 35)",
    )
    parser.add_argument(
        "--firmware-exact",
        action="store_true",
        help="Match L_ttn.c P schedule (0.1/0.4) and transition gain preset; default uses your recommended P/I/D only",
    )
    parser.add_argument(
        "--pwm-offset",
        type=float,
        default=DEFAULT_PWM_OFFSET,
        help="Baseline PWM before PID terms (L_ttn TEMP_PWM_OFFSET, default 40)",
    )
    parser.add_argument(
        "--highlight",
        default="differential_evolution",
        help="Model id drawn with heavier line weight (all-models plot)",
    )
    parser.add_argument(
        "--recommended-model",
        default="differential_evolution",
        help="Model id used for the 'new PID only' plot (default: differential_evolution)",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Also save comparison plot with every tuning model",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Output PNG for new-PID plot (default: Saved Images/<exp>_new_pid_heating.png)",
    )
    parser.add_argument("--show", action="store_true", help="Open saved PNG in default image viewer")
    args = parser.parse_args()

    exp_path = Path(args.exp_path)
    json_path = args.recommendations_json
    if json_path is None and DEFAULT_JSON.is_file():
        json_path = DEFAULT_JSON

    ingest = load_pid_ingest(exp_path)
    recommendations, rec_source = _load_recommendations(exp_path, json_path, args.models)
    print(f"Experiment: {exp_path}")
    print(f"Recommendations from: {rec_source}")
    print(f"Models in file: {[r['model_id'] for r in recommendations]}")

    sim_cfg = FirmwareSimConfig(
        pwm_max=args.pwm_max,
        pwm_offset=args.pwm_offset,
        use_firmware_p_schedule=bool(args.firmware_exact),
        honor_transition_gain_preset=bool(args.firmware_exact),
    )
    bundle = build_predictions(
        ingest,
        recommendations,
        model_ids=args.models,
        duration_mult=args.duration_mult,
        sim_cfg=sim_cfg,
    )
    if not bundle["curves"]:
        raise SystemExit("No non-zero PID models to simulate (check recommendations JSON).")

    exp_slug = exp_path.name.split("KHz_U_")[-1] if "KHz_U_" in exp_path.name else exp_path.name
    save_dir = Path.home() / "OneDrive - ProtonDx" / "Data" / "Saved Images"

    rec_mid = args.recommended_model
    if rec_mid not in bundle["curves"]:
        raise SystemExit(
            f"Recommended model {rec_mid!r} has no simulation (zero gains or not in JSON). "
            f"Try: {list(bundle['curves'].keys())}"
        )

    g_new = bundle["curves"][rec_mid]["gains"]
    save_new = args.save
    if save_new is None:
        save_new = save_dir / f"{exp_slug}_new_pid_heating.png"

    title_new = f"New PID heating — {exp_slug} [{rec_mid}]"
    plot_new_pid_only(bundle, rec_mid, title=title_new, save_path=save_new, show=args.show)

    if args.all_models:
        save_all = save_dir / f"{exp_slug}_pid_heating_all_models.png"
        plot_heating_prediction(
            bundle,
            title=f"PID heating prediction (all models) — {exp_slug}",
            save_path=save_all,
            show=False,
            highlight_model=args.highlight,
        )

    print(f"\nRecommended [{rec_mid}] gains to program on chip:")
    print(f"  TEMP_P = {g_new.kp:.6g}")
    print(f"  TEMP_I = {g_new.ki:.6g}")
    print(f"  TEMP_D = {g_new.kd:.6g}")

    print("\nAll simulated models:")
    for mid, item in bundle["curves"].items():
        g = item["gains"]
        print(f"  [{mid}] TEMP_P={g.kp:.4g} TEMP_I={g.ki:.4g} TEMP_D={g.kd:.4g}")


if __name__ == "__main__":
    _main()
