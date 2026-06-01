"""
Ingest chip PID telemetry and recommend optimal TEMP_P / TEMP_I / TEMP_D gains.

Data sources (Lacewing v05+ with version=5 readout tails):
  - *_readout_time.bin: per-frame reg_err, reg_P, reg_I, reg_D, reg_ref (uint16)
  - *_temp_log.bin: heating-phase time (s) and linearised temperature (optional)

Firmware reference (Lacewing_STM32 L_ttn.c):
  temp_p_term = TEMP_P * temp_err_new
  temp_i_term = TEMP_I * temp_err_sum
  temp_d_term = TEMP_D * (temp_err_new - temp_err_old)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import differential_evolution, minimize
except ImportError:  # pragma: no cover
    differential_evolution = None
    minimize = None

import load_functions as load
from pid_firmware_sim import (
    FirmwareSimConfig,
    PidGains,
    build_plant_from_measured,
    identify_fopdt,
    simulate_firmware_pid,
)

_identify_fopdt = identify_fopdt  # backward-compatible alias for tests/imports

NROWS = 290
NCOLS = 204
P = NROWS * NCOLS
V05_SENTINEL = 11
PID_TAIL_WORDS = 5  # reg_err, reg_P, reg_I, reg_D, reg_ref


@dataclass
class PidIngest:
    """Per-frame PID telemetry extracted from a readout run."""

    exp_path: Path
    readout_path: Path
    n_refs: int
    time_raw: np.ndarray
    time_s: np.ndarray
    temp_lin: np.ndarray
    reg_err: np.ndarray
    reg_p_term: np.ndarray
    reg_i_term: np.ndarray
    reg_d_term: np.ndarray
    reg_ref: np.ndarray
    has_pid_tail: bool
    heat_time_s: Optional[np.ndarray] = None
    heat_temp_lin: Optional[np.ndarray] = None

    @property
    def n_frames(self) -> int:
        return int(self.time_raw.size)


@dataclass
class ModelRecommendation:
    model_id: str
    name: str
    gains: PidGains
    score: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecommendedModelInfo:
    """Catalog entry for a tuning approach (implemented or planned)."""

    model_id: str
    name: str
    description: str
    best_for: str
    requires: List[str]
    implemented: bool


RECOMMENDED_PID_MODELS: List[RecommendedModelInfo] = [
    RecommendedModelInfo(
        model_id="backcalculate",
        name="Back-calculate from logged terms",
        description=(
            "Estimate TEMP_P, TEMP_I, TEMP_D as median(reg_term / reg_err) using "
            "frames where |reg_err| is above a noise floor. Fast baseline from the "
            "gains the chip actually applied."
        ),
        best_for="Auditing current firmware behaviour on a completed run.",
        requires=["readout_time.bin with PID tail (version=5)"],
        implemented=True,
    ),
    RecommendedModelInfo(
        model_id="ziegler_nichols",
        name="Ziegler–Nichols (open-loop FOPDT)",
        description=(
            "Fit first-order-plus-dead-time (FOPDT) parameters from the heating ramp "
            "in temp_log.bin, then apply classic Z–N PID rules."
        ),
        best_for="Initial tuning when you have a clear S-shaped heating ramp.",
        requires=["temp_log.bin (preferred) or readout temperature trace"],
        implemented=True,
    ),
    RecommendedModelInfo(
        model_id="cohen_coon",
        name="Cohen–Coon (open-loop FOPDT)",
        description=(
            "Same FOPDT identification as Z–N, but Cohen–Coon rules (often less "
            "aggressive overshoot on slow thermal plants)."
        ),
        best_for="Thermal systems where Z–N overshoots or oscillates.",
        requires=["temp_log.bin (preferred) or readout temperature trace"],
        implemented=True,
    ),
    RecommendedModelInfo(
        model_id="itae_fopdt",
        name="ITAE minimization on FOPDT plant",
        description=(
            "Simulate a discrete PID loop against an identified FOPDT plant and "
            "minimize integrated time-weighted absolute error (ITAE) with SciPy."
        ),
        best_for="Balanced set-point tracking when scipy is available.",
        requires=["scipy", "temperature trajectory"],
        implemented=True,
    ),
    RecommendedModelInfo(
        model_id="differential_evolution",
        name="Global search (differential evolution)",
        description=(
            "Search Kp/Ki/Kd bounds with differential evolution to minimize ITAE "
            "on the same FOPDT surrogate. Robust to local minima."
        ),
        best_for="Noisy traces or when gradient-based ITAE stalls.",
        requires=["scipy", "temperature trajectory"],
        implemented=True,
    ),
    RecommendedModelInfo(
        model_id="relay_autotune",
        name="Relay / Åström–Hägglund-style",
        description=(
            "Estimate ultimate gain and period from temperature oscillations near "
            "setpoint, then apply Z–N closed-loop rules (Ku, Pu)."
        ),
        best_for="Runs that already show limit-cycle behaviour near target temp.",
        requires=["temperature trace with visible oscillation"],
        implemented=True,
    ),
    RecommendedModelInfo(
        model_id="bayesian_optimization",
        name="Bayesian optimization (Gaussian process)",
        description=(
            "Sample Kp/Ki/Kd with a GP surrogate (e.g. scikit-optimize). Fewer "
            "simulations than grid search when each evaluation is expensive."
        ),
        best_for="Multi-objective tuning (ITAE + overshoot + energy) with few trials.",
        requires=["scikit-optimize (optional extra)", "temperature trajectory"],
        implemented=False,
    ),
    RecommendedModelInfo(
        model_id="reinforcement_learning",
        name="Reinforcement learning (policy search)",
        description=(
            "Learn a mapping from error state to PWM or gain adjustments across many "
            "runs. Needs a training corpus of labelled experiments."
        ),
        best_for="Fleet-wide adaptive tuning once many chips/runs are archived.",
        requires=["Many labelled runs", "ML stack (PyTorch/JAX, etc.)"],
        implemented=False,
    ),
    RecommendedModelInfo(
        model_id="system_identification",
        name="System ID (prediction-error minimization)",
        description=(
            "Identify a discrete ARX/STATE-SPACE model (e.g. python-control, "
            "SystemIdentification.jl) then synthesize PID with pole placement or LQR."
        ),
        best_for="High-fidelity plant models when linearisation assumptions break down.",
        requires=["python-control or similar", "rich input–output logs (PWM + temp)"],
        implemented=False,
    ),
]


def list_recommended_models(include_unimplemented: bool = True) -> List[Dict[str, Any]]:
    """Return the model catalog as JSON-serialisable dicts."""
    out = []
    for m in RECOMMENDED_PID_MODELS:
        if not include_unimplemented and not m.implemented:
            continue
        out.append(asdict(m))
    return out


def print_recommended_models(include_unimplemented: bool = True) -> None:
    for i, m in enumerate(RECOMMENDED_PID_MODELS, start=1):
        if not include_unimplemented and not m.implemented:
            continue
        status = "implemented" if m.implemented else "planned"
        print(f"{i}. [{m.model_id}] {m.name} ({status})")
        print(f"   {m.description}")
        print(f"   Best for: {m.best_for}")
        print(f"   Requires: {', '.join(m.requires)}")
        print()


def _infer_v05_frame_layout(data_list: List[int]) -> Tuple[int, int]:
    """Return (n_refs, A_total) for v05 readout frames."""
    if len(data_list) < P + 2:
        raise ValueError("Readout file too short for v05 layout inference")

    for cand_n_refs in range(1, 33):
        idx_a = P * cand_n_refs
        if idx_a >= len(data_list):
            break
        cand_a_raw = int(data_list[idx_a])
        if cand_a_raw < 9 or cand_a_raw > 512:
            continue
        for a_total in (cand_a_raw, cand_a_raw + 1):
            cand_n = (P * cand_n_refs) + a_total
            if cand_n > 0 and len(data_list) % cand_n == 0:
                return cand_n_refs, a_total
        a_lo = max(9, cand_a_raw - 2)
        a_hi = min(512, cand_a_raw + 12)
        for a_total in range(a_lo, a_hi + 1):
            if a_total in (cand_a_raw, cand_a_raw + 1):
                continue
            cand_n = (P * cand_n_refs) + a_total
            if cand_n > 0 and len(data_list) % cand_n == 0:
                return cand_n_refs, a_total

    raise ValueError("Could not infer v05 frame layout (n_refs, A_total)")


def _unwrap_time_seconds(time_raw: np.ndarray) -> np.ndarray:
    """Convert uint16 timestamp stream to monotonic seconds (handles wrap)."""
    if time_raw.size == 0:
        return time_raw.astype(float)
    t = np.zeros(time_raw.size, dtype=float)
    t[0] = float(time_raw[0])
    offset = 0.0
    prev = float(time_raw[0])
    for i in range(1, time_raw.size):
        cur = float(time_raw[i])
        if cur < prev:
            offset += 65536.0
        t[i] = cur + offset
        prev = cur
    return t


def _parse_v05_pid_tail(pack: List[int], p_frame: int, a_total: int) -> Optional[Dict[str, int]]:
    """
    Parse optional PID fields after the fixed v05 tail.

    Tail layout: [sentinel=11, ts, vref*n, chem, temp, temp_lin+100, adc, tam, tch, n_wells, version, (pid×5)]
    """
    idx = p_frame
    if idx >= len(pack) or int(pack[idx]) != V05_SENTINEL:
        return None
    idx += 1  # timestamp
    n_fields_after_sentinel = a_total - 1
    # timestamp + vrefs + 8 fixed fields = 1 + n_vref + 8
    n_vref = n_fields_after_sentinel - 9 - PID_TAIL_WORDS
    if n_vref < 0:
        n_vref = n_fields_after_sentinel - 9
        if n_vref < 0:
            return None
        # No PID extension in this frame layout
        if idx + 1 + n_vref + 8 > len(pack):
            return None
        version = int(pack[idx + 1 + n_vref + 7])
        if version != 5:
            return None
        return None

    base = idx + 1 + n_vref + 8
    if base + PID_TAIL_WORDS > len(pack):
        return None
    version = int(pack[base - 1])
    if version != 5:
        return None
    reg_err, reg_p, reg_i, reg_d, reg_ref = (int(pack[base + k]) for k in range(PID_TAIL_WORDS))
    return {
        "reg_err": reg_err,
        "reg_p_term": reg_p,
        "reg_i_term": reg_i,
        "reg_d_term": reg_d,
        "reg_ref": reg_ref,
        "temp_lin": int(pack[base - 5]) - 100,
    }


def load_pid_ingest(exp_path: Path, nrows: int = NROWS, ncols: int = NCOLS) -> PidIngest:
    """
    Load PID-related fields from the most recent *_readout_time.bin in an experiment folder.
    """
    exp_path = Path(exp_path)
    readout_path = load.find_most_recent_readout(exp_path)
    data_list = load.binary_file_read(readout_path)
    n_refs, a_total = _infer_v05_frame_layout(data_list)
    p_frame = P * n_refs
    n_frame = len(data_list) // (p_frame + a_total)

    time_raw = np.zeros(n_frame, dtype=np.int64)
    temp_lin = np.zeros(n_frame, dtype=float)
    reg_err = np.zeros(n_frame, dtype=np.int64)
    reg_p = np.zeros(n_frame, dtype=np.int64)
    reg_i = np.zeros(n_frame, dtype=np.int64)
    reg_d = np.zeros(n_frame, dtype=np.int64)
    reg_ref = np.zeros(n_frame, dtype=np.int64)
    has_pid = False

    for fi in range(n_frame):
        pack = data_list[fi * (p_frame + a_total) : (fi + 1) * (p_frame + a_total)]
        idx = p_frame
        time_raw[fi] = int(pack[idx + 1])
        pid = _parse_v05_pid_tail(pack, p_frame, a_total)
        if pid is None:
            # Still try to read temp_lin from standard tail without PID
            n_fields = a_total - 1
            n_vref = n_fields - 9
            if n_vref >= 0:
                k = idx + 1 + n_vref
                if k + 2 < len(pack):
                    temp_lin[fi] = int(pack[k + 2]) - 100
            continue
        has_pid = True
        temp_lin[fi] = float(pid["temp_lin"])
        reg_err[fi] = pid["reg_err"]
        reg_p[fi] = pid["reg_p_term"]
        reg_i[fi] = pid["reg_i_term"]
        reg_d[fi] = pid["reg_d_term"]
        reg_ref[fi] = pid["reg_ref"]

    time_s = _unwrap_time_seconds(time_raw)
    heat_t, heat_temp = load.load_temp_log(exp_path)

    return PidIngest(
        exp_path=exp_path,
        readout_path=readout_path,
        n_refs=n_refs,
        time_raw=time_raw,
        time_s=time_s,
        temp_lin=temp_lin,
        reg_err=reg_err,
        reg_p_term=reg_p,
        reg_i_term=reg_i,
        reg_d_term=reg_d,
        reg_ref=reg_ref,
        has_pid_tail=has_pid,
        heat_time_s=heat_t,
        heat_temp_lin=heat_temp,
    )


def _signed_u16(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.int64)
    return np.where(x >= 32768, x - 65536, x)


def model_backcalculate(ingest: PidIngest, err_floor: float = 2.0) -> ModelRecommendation:
    err = _signed_u16(ingest.reg_err).astype(float)
    p_term = _signed_u16(ingest.reg_p_term).astype(float)
    i_term = _signed_u16(ingest.reg_i_term).astype(float)
    d_term = _signed_u16(ingest.reg_d_term).astype(float)

    mask_p = np.abs(err) >= err_floor
    kp = float(np.median(p_term[mask_p] / err[mask_p])) if np.any(mask_p) else 0.0

    # I term uses accumulated error on-chip; approximate effective Ki from term magnitude
    mask_i = np.abs(i_term) > 0
    ki = float(np.median(i_term[mask_i] / np.maximum(np.abs(err[mask_i]), err_floor))) if np.any(mask_i) else 0.0

    d_err = np.diff(err, prepend=err[0])
    mask_d = np.abs(d_err) >= 1.0
    kd = float(np.median(d_term[mask_d] / d_err[mask_d])) if np.any(mask_d) else 0.0

    ref_med = float(np.median(ingest.reg_ref[ingest.reg_ref > 0])) if ingest.has_pid_tail else float("nan")
    return ModelRecommendation(
        model_id="backcalculate",
        name="Back-calculate from logged terms",
        gains=PidGains(kp=kp, ki=ki, kd=kd, notes="Median term/error ratios from readout tail"),
        details={"reg_ref_median": ref_med, "n_frames_used_p": int(mask_p.sum())},
    )


def _pick_temperature_trace(ingest: PidIngest) -> Tuple[np.ndarray, np.ndarray]:
    if ingest.heat_time_s is not None and ingest.heat_temp_lin is not None:
        if ingest.heat_time_s.size > 10:
            return ingest.heat_time_s.astype(float), ingest.heat_temp_lin.astype(float)
    t = ingest.time_s.astype(float)
    y = ingest.temp_lin.astype(float)
    if t.size > 10 and np.any(y != 0):
        return t, y
    raise ValueError("No usable temperature trajectory (need temp_log.bin or readout temp_lin)")


def _clamp_firmware_gains(g: PidGains) -> PidGains:
    """Keep recommendations in a plausible range for Lacewing TEMP_P/I/D."""
    kp = float(np.clip(g.kp, 0.01, 2.0))
    ki = float(np.clip(g.ki, 0.0, 0.5))
    kd = float(np.clip(g.kd, 0.0, 2.0))
    note = g.notes
    if (kp, ki, kd) != (g.kp, g.ki, g.kd):
        note = (note + "; clamped to firmware-typical bounds").strip("; ")
    return PidGains(kp=kp, ki=ki, kd=kd, notes=note)


def _zn_pid(k: float, tau: float, l: float) -> PidGains:
    if l <= 0 or tau <= 0:
        raise ValueError("Invalid FOPDT parameters for Ziegler–Nichols")
    kp = 1.2 * tau / (k * l)
    ti = 2.0 * l
    td = 0.5 * l
    ki = kp / ti if ti > 0 else 0.0
    kd = kp * td
    return _clamp_firmware_gains(
        PidGains(kp=kp, ki=ki, kd=kd, notes="Ziegler–Nichols open-loop rules (normalised FOPDT)")
    )


def _cc_pid(k: float, tau: float, l: float) -> PidGains:
    if l <= 0 or tau <= 0:
        raise ValueError("Invalid FOPDT parameters for Cohen–Coon")
    kp = (1.35 + 0.25 * l / tau) * tau / (k * l)
    ti = l * (2.5 + 2.0 * l / tau) / (1.0 + 0.39 * l / tau)
    td = l * 0.37 / (1.0 + 0.19 * l / tau)
    ki = kp / ti if ti > 0 else 0.0
    kd = kp * td
    return _clamp_firmware_gains(
        PidGains(kp=kp, ki=ki, kd=kd, notes="Cohen–Coon open-loop rules (normalised FOPDT)")
    )


def model_ziegler_nichols(ingest: PidIngest) -> ModelRecommendation:
    t, y = _pick_temperature_trace(ingest)
    k, tau, l = _identify_fopdt(t, y)
    gains = _zn_pid(k, tau, l)
    return ModelRecommendation(
        model_id="ziegler_nichols",
        name="Ziegler–Nichols (open-loop FOPDT)",
        gains=gains,
        details={"K": k, "tau": tau, "L": l},
    )


def model_cohen_coon(ingest: PidIngest) -> ModelRecommendation:
    t, y = _pick_temperature_trace(ingest)
    k, tau, l = _identify_fopdt(t, y)
    gains = _cc_pid(k, tau, l)
    return ModelRecommendation(
        model_id="cohen_coon",
        name="Cohen–Coon (open-loop FOPDT)",
        gains=gains,
        details={"K": k, "tau": tau, "L": l},
    )


def _simulate_pid(
    kp: float,
    ki: float,
    kd: float,
    setpoint: float,
    y0: float,
    n: int,
    dt: float,
    k_plant: float,
    tau: float,
    l_delay_s: float,
    sim_cfg: Optional[FirmwareSimConfig] = None,
) -> np.ndarray:
    gains = PidGains(kp=kp, ki=ki, kd=kd)
    _, y, _, _ = simulate_firmware_pid(
        gains, setpoint, y0, n, dt, k_plant, tau, l_delay_s, cfg=sim_cfg
    )
    return y


def _itae(y: np.ndarray, setpoint: float, dt: float) -> float:
    t = np.arange(y.size, dtype=float) * dt
    return float(np.sum(t * np.abs(setpoint - y)) * dt)


def model_itae_fopdt(
    ingest: PidIngest,
    bounds: Tuple[Tuple[float, float], ...] = ((0.01, 2.0), (0.0, 0.5), (0.0, 2.0)),
) -> ModelRecommendation:
    if minimize is None:
        raise ImportError("scipy is required for model_itae_fopdt (pip install scipy)")
    t, y = _pick_temperature_trace(ingest)
    k_plant, tau, l, dt = build_plant_from_measured(t, y)
    setpoint = float(np.median(ingest.reg_ref[ingest.reg_ref > 0])) if ingest.has_pid_tail else float(np.percentile(y, 95))
    n = min(int(t[-1] / dt) + 1, 5000) if t.size else 500
    y0 = float(y[0])

    def objective(x: np.ndarray) -> float:
        kp, ki, kd = float(x[0]), float(x[1]), float(x[2])
        sim = _simulate_pid(kp, ki, kd, setpoint, y0, n, dt, k_plant, tau, l)
        return _itae(sim, setpoint, dt)

    x0 = np.array([0.4, 0.01, 0.3])
    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    gains = _clamp_firmware_gains(
        PidGains(
            kp=float(res.x[0]),
            ki=float(res.x[1]),
            kd=float(res.x[2]),
            notes="ITAE-optimal on FOPDT surrogate",
        )
    )
    return ModelRecommendation(
        model_id="itae_fopdt",
        name="ITAE minimization on FOPDT plant",
        gains=gains,
        score=float(res.fun),
        details={"K": k_plant, "tau": tau, "L": l, "setpoint": setpoint, "success": bool(res.success)},
    )


def model_differential_evolution(
    ingest: PidIngest,
    bounds: Tuple[Tuple[float, float], ...] = ((0.01, 2.0), (0.0, 0.5), (0.0, 2.0)),
) -> ModelRecommendation:
    if differential_evolution is None:
        raise ImportError("scipy is required for model_differential_evolution")
    t, y = _pick_temperature_trace(ingest)
    k_plant, tau, l, dt = build_plant_from_measured(t, y)
    setpoint = float(np.median(ingest.reg_ref[ingest.reg_ref > 0])) if ingest.has_pid_tail else float(np.percentile(y, 95))
    n = min(int(t[-1] / dt) + 1, 5000)
    y0 = float(y[0])

    def objective(x: np.ndarray) -> float:
        kp, ki, kd = float(x[0]), float(x[1]), float(x[2])
        sim = _simulate_pid(kp, ki, kd, setpoint, y0, n, dt, k_plant, tau, l)
        return _itae(sim, setpoint, dt)

    res = differential_evolution(objective, bounds, seed=42, maxiter=80, polish=True)
    gains = _clamp_firmware_gains(
        PidGains(
            kp=float(res.x[0]),
            ki=float(res.x[1]),
            kd=float(res.x[2]),
            notes="DE global search on FOPDT surrogate",
        )
    )
    return ModelRecommendation(
        model_id="differential_evolution",
        name="Global search (differential evolution)",
        gains=gains,
        score=float(res.fun),
        details={"K": k_plant, "tau": tau, "L": l, "setpoint": setpoint},
    )


def model_relay_autotune(ingest: PidIngest) -> ModelRecommendation:
    t, y = _pick_temperature_trace(ingest)
    setpoint = float(np.median(ingest.reg_ref[ingest.reg_ref > 0])) if ingest.has_pid_tail else float(np.percentile(y, 90))
    mask = (t > t[int(0.5 * len(t))]) & (np.abs(y - setpoint) < 0.05 * max(np.ptp(y), 1.0))
    seg = y[mask] if np.sum(mask) > 20 else y[int(0.7 * len(y)) :]
    if seg.size < 10:
        raise ValueError("Insufficient near-setpoint samples for relay autotune")
    pu = _estimate_ultimate_period(seg)
    a = float(np.std(seg))
    if a < 1e-6 or pu <= 0:
        raise ValueError("Could not estimate oscillation amplitude/period")
    ku = 4.0 * 0.5 / (np.pi * a)  # relay amplitude d=0.5 surrogate
    kp = 0.6 * ku
    ki = 2.0 * kp / pu
    kd = kp * pu / 8.0
    return ModelRecommendation(
        model_id="relay_autotune",
        name="Relay / Åström–Hägglund-style",
        gains=_clamp_firmware_gains(
            PidGains(kp=kp, ki=ki, kd=kd, notes="Z–N closed-loop from estimated Ku, Pu")
        ),
        details={"Ku": ku, "Pu": pu, "osc_amp": a},
    )


def _estimate_ultimate_period(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    y = y - np.mean(y)
    ac = np.correlate(y, y, mode="full")
    ac = ac[ac.size // 2 :]
    if ac.size < 3:
        return 0.0
    peaks = []
    for i in range(1, ac.size - 1):
        if ac[i - 1] < ac[i] > ac[i + 1]:
            peaks.append(i)
    if len(peaks) < 2:
        return 0.0
    return float(np.median(np.diff(peaks[:5])))


MODEL_RUNNERS: Dict[str, Callable[[PidIngest], ModelRecommendation]] = {
    "backcalculate": model_backcalculate,
    "ziegler_nichols": model_ziegler_nichols,
    "cohen_coon": model_cohen_coon,
    "itae_fopdt": model_itae_fopdt,
    "differential_evolution": model_differential_evolution,
    "relay_autotune": model_relay_autotune,
}


def run_models(
    ingest: PidIngest,
    model_ids: Optional[List[str]] = None,
) -> List[ModelRecommendation]:
    """Run one or more implemented tuning models."""
    if model_ids is None:
        model_ids = [m.model_id for m in RECOMMENDED_PID_MODELS if m.implemented]
    results: List[ModelRecommendation] = []
    for mid in model_ids:
        if mid not in MODEL_RUNNERS:
            raise KeyError(f"Unknown or unimplemented model_id: {mid}")
        results.append(MODEL_RUNNERS[mid](ingest))
    return results


def recommendations_to_dict(recs: List[ModelRecommendation]) -> List[Dict[str, Any]]:
    out = []
    for r in recs:
        out.append(
            {
                "model_id": r.model_id,
                "name": r.name,
                "score": r.score,
                "gains": r.gains.as_dict(),
                "notes": r.gains.notes,
                "details": r.details,
            }
        )
    return out


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest chip PID telemetry and recommend TEMP_P/I/D gains.",
    )
    parser.add_argument("exp_path", type=Path, nargs="?", help="Experiment folder with .bin files")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print recommended tuning models and exit",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"Model ids to run (default: all implemented). Choices: {', '.join(MODEL_RUNNERS)}",
    )
    parser.add_argument("--json", action="store_true", help="Emit results as JSON")
    args = parser.parse_args()

    if args.list_models:
        print_recommended_models()
        return

    if args.exp_path is None:
        parser.error("exp_path is required unless --list-models is set")

    ingest = load_pid_ingest(args.exp_path)
    if not ingest.has_pid_tail:
        print(
            "Warning: readout file has no version=5 PID tail; "
            "back-calculate will be empty — open-loop models still use temperature."
        )

    recs = run_models(ingest, model_ids=args.models)
    payload = {
        "exp_path": str(ingest.exp_path),
        "readout_path": str(ingest.readout_path),
        "n_frames": ingest.n_frames,
        "has_pid_tail": ingest.has_pid_tail,
        "recommendations": recommendations_to_dict(recs),
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Experiment: {ingest.exp_path}")
        print(f"Readout:    {ingest.readout_path}")
        print(f"Frames:     {ingest.n_frames}  PID tail: {ingest.has_pid_tail}\n")
        for r in recs:
            g = r.gains
            print(f"--- {r.name} [{r.model_id}] ---")
            if r.score is not None:
                print(f"  score (ITAE): {r.score:.6g}")
            print(f"  TEMP_P = {g.kp:.6g}")
            print(f"  TEMP_I = {g.ki:.6g}")
            print(f"  TEMP_D = {g.kd:.6g}")
            if g.notes:
                print(f"  notes: {g.notes}")
            print()


if __name__ == "__main__":
    _main()
