"""
Firmware-aligned PID heating simulation — mirrors ``TTN_Heating_Time()`` in L_ttn.c.

Controller behaviour (see Lacewing_STM32_v05/Application/Src/L_ttn.c):
  - Control variable: ``temp_avg_Vs`` (shifted Vs linearised temp); error = ``temp_ref - temp_avg_Vs``
  - PID runs only when ``n_frame_heat > 2``; frames 0–2 force PWM = 0
  - Near setpoint (|err| < 2): ``temp_err_sum_k = 0.2`` and firmware may override TEMP_P to 0.1
  - Far from setpoint: ``temp_err_sum_k = 1`` and firmware may override TEMP_P to 0.4
  - Four PWM modes via ``temp_check_flag`` and ``temp_range_init`` / ``temp_range_final``:
      * err > 60, flag==1: P-only
      * err < 60, flag==1: clear integral, flag=0 (optional transition gain preset); PWM held
      * flag==0, |err|<=2: P+I
      * else: P+D plus adaptive ``TEMP_PWM_OFFSET`` nudges
  - Slew: ±5/frame before stability, ±2 after ``frame_temp_st``
  - Output clipped in ``TTN_Set_Ipel()`` to [TEMP_PWM_MIN, TEMP_PWM_MAX] (stock max = 35 %)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

import numpy as np

# L_ttn.c defaults
DEFAULT_TEMP_REF = 440.0
DEFAULT_PWM_OFFSET = 40.0
DEFAULT_PWM_MAX = 35.0  # TEMP_PWM_MAX in stock firmware
DEFAULT_PWM_MIN = 0.0
DEFAULT_TEMP_RANGE_INIT = 60.0
DEFAULT_TEMP_RANGE_FINAL = 2.0
DEFAULT_ERR_NEAR_SETPOINT = 2.0
DEFAULT_PWM_SLEW_HEATING = 5.0
DEFAULT_PWM_SLEW_STABLE = 2.0
DEFAULT_STABILITY_DELTA = 3.0
DEFAULT_STABILITY_FRAMES = 10
DEFAULT_HEAT_START_FRAMES = 3  # PID when n_frame_heat > 2


@dataclass
class PidGains:
    """Firmware-style gains (TEMP_P, TEMP_I, TEMP_D) — UI stores P/I/D ×10 on chip."""

    kp: float
    ki: float
    kd: float
    notes: str = ""

    def as_dict(self) -> Dict[str, float]:
        return {"TEMP_P": self.kp, "TEMP_I": self.ki, "TEMP_D": self.kd}


@dataclass
class FirmwarePidState:
    """Mutable controller state carried across heating frames (L_ttn.c globals)."""

    n_frame_heat: int = 0
    temp_err_sum: float = 0.0
    temp_err_old: float = 0.0
    temp_check_flag: float = 1.0
    temp_pwm_prev: float = 0.0
    pwm_offset: float = DEFAULT_PWM_OFFSET
    temp_sum: float = 0.0
    temp_average: float = 0.0
    count: int = 1
    prev_frame_heat: int = 0
    frame_temp_ok: int = 0
    frame_temp_st: int = 0
    temp_st_count: int = 0
    temp_avg_Vs: float = 0.0
    temp_avg_Vs_prev: float = 0.0
    temp_avg_Vs_init: float = 0.0
    phase: str = "warmup"  # diagnostic: warmup | p_only | transition | pi | pd


@dataclass
class FirmwareSimConfig:
    temp_ref: float = DEFAULT_TEMP_REF
    pwm_offset: float = DEFAULT_PWM_OFFSET
    pwm_max: float = DEFAULT_PWM_MAX
    pwm_min: float = DEFAULT_PWM_MIN
    temp_range_init: float = DEFAULT_TEMP_RANGE_INIT
    temp_range_final: float = DEFAULT_TEMP_RANGE_FINAL
    err_near_setpoint: float = DEFAULT_ERR_NEAR_SETPOINT
    slew_heating: float = DEFAULT_PWM_SLEW_HEATING
    slew_stable: float = DEFAULT_PWM_SLEW_STABLE
    stability_delta: float = DEFAULT_STABILITY_DELTA
    stability_frames: int = DEFAULT_STABILITY_FRAMES
    heat_start_frames: int = DEFAULT_HEAT_START_FRAMES
    heat_en: bool = True
    # If True: near setpoint use TEMP_P=0.1/0.4 schedule (overrides gains.kp each frame).
    use_firmware_p_schedule: bool = False
    # If True: on first entry below temp_range_init, force P=0.4 I=0.01 D=0.30 like firmware.
    honor_transition_gain_preset: bool = False
    transition_kp: float = 0.4
    transition_ki: float = 0.01
    transition_kd: float = 0.30


def identify_fopdt(time_s: np.ndarray, temp: np.ndarray) -> Tuple[float, float, float]:
    """Normalised FOPDT (K=1) time constants from a heating ramp."""
    t = np.asarray(time_s, dtype=float)
    y = np.asarray(temp, dtype=float)
    if t.size < 5:
        raise ValueError("Too few samples for FOPDT identification")
    y0 = float(y[0])
    yf = float(np.percentile(y, 98))
    span = max(yf - y0, 1e-6)
    y_norm = (y - y0) / span
    i28 = int(np.argmax(y_norm >= 0.283))
    i63 = int(np.argmax(y_norm >= 0.632))
    t28 = float(t[max(i28, 0)])
    t63 = float(t[max(i63, min(i28 + 1, len(t) - 1))])
    tau = max(t63 - t28, 1e-3)
    l_delay = max(t28 - t[0], 0.0)
    dt_med = float(np.median(np.diff(t))) if t.size > 1 else 1.0
    duration = max(float(t[-1] - t[0]), dt_med)
    l_delay = max(l_delay, max(0.05 * duration, 3.0 * dt_med))
    return 1.0, tau, l_delay


def calibrate_plant_gain_from_measured(
    y_meas: np.ndarray,
    pwm_sat: float = DEFAULT_PWM_MAX,
    y0: float | None = None,
) -> float:
    y = np.asarray(y_meas, dtype=float)
    if y0 is None:
        y0 = float(y[0])
    y_ss = float(np.percentile(y, 98))
    return max((y_ss - y0) / max(float(pwm_sat), 1e-6), 1e-3)


def _effective_gains(
    gains: PidGains,
    err: float,
    cfg: FirmwareSimConfig,
    transition_active: bool,
) -> Tuple[float, float, float, float]:
    """Return (kp, ki, kd, err_sum_k) for this frame."""
    if transition_active and cfg.honor_transition_gain_preset:
        kp, ki, kd = cfg.transition_kp, cfg.transition_ki, cfg.transition_kd
    else:
        kp, ki, kd = gains.kp, gains.ki, gains.kd

    if cfg.use_firmware_p_schedule:
        if -cfg.err_near_setpoint < err < cfg.err_near_setpoint:
            return 0.1, ki, kd, 0.2
        return 0.4, ki, kd, 1.0

    err_sum_k = 0.2 if -cfg.err_near_setpoint < err < cfg.err_near_setpoint else 1.0
    return kp, ki, kd, err_sum_k


def firmware_pid_step(
    state: FirmwarePidState,
    temp_avg_Vs: float,
    gains: PidGains,
    cfg: FirmwareSimConfig,
) -> float:
    """
    One heating-frame PID update. Returns PWM sent to ``TTN_Set_Ipel`` (after clip).

    ``temp_avg_Vs`` is the control variable after Vs init/shift (same scale as ``temp_ref``).
    """
    state.n_frame_heat += 1
    state.temp_avg_Vs_prev = state.temp_avg_Vs
    state.temp_avg_Vs = float(temp_avg_Vs)

    if state.n_frame_heat == 1:
        state.temp_avg_Vs_init = state.temp_avg_Vs

    if not cfg.heat_en:
        pwm = cfg.pwm_min
        state.temp_pwm_prev = pwm
        return pwm

    temp_ref = cfg.temp_ref

    # Stability bookkeeping (L_ttn.c lines 2158–2186)
    if state.frame_temp_ok == 0 and state.temp_avg_Vs > 0.95 * temp_ref:
        state.frame_temp_ok = state.n_frame_heat

    if 0.98 * temp_ref < state.temp_avg_Vs < 1.02 * temp_ref:
        if abs(state.temp_avg_Vs - state.temp_avg_Vs_prev) < cfg.stability_delta:
            state.temp_st_count = min(state.temp_st_count + 1, 65535)
        else:
            state.temp_st_count = 0
        if state.temp_st_count > cfg.stability_frames and state.frame_temp_st == 0:
            state.frame_temp_st = state.n_frame_heat
    else:
        state.temp_st_count = 0

    if state.n_frame_heat <= cfg.heat_start_frames:
        state.phase = "warmup"
        pwm = cfg.pwm_min
        state.temp_pwm_prev = pwm
        return pwm

    err = float(temp_ref - state.temp_avg_Vs)

    kp, ki, kd, err_sum_k = _effective_gains(gains, err, cfg, transition_active=False)

    state.temp_sum += err
    state.temp_average = state.temp_sum / max(state.count, 1)
    state.temp_err_sum += err * err_sum_k
    if state.temp_err_sum < 0.0:
        state.temp_err_sum = 0.0

    p_term = kp * err
    i_term = ki * state.temp_err_sum
    d_term = kd * (err - state.temp_err_old)

    pwm = state.temp_pwm_prev

    if err > cfg.temp_range_init and state.temp_check_flag == 1.0:
        state.phase = "p_only"
        pwm = state.pwm_offset + p_term
    elif err < cfg.temp_range_init and state.temp_check_flag == 1.0:
        state.phase = "transition"
        state.temp_err_sum = 0.0
        state.temp_check_flag = 0.0
        kp, ki, kd, err_sum_k = _effective_gains(gains, err, cfg, transition_active=True)
        p_term = kp * err
        i_term = ki * state.temp_err_sum
        d_term = kd * (err - state.temp_err_old)
        # Firmware does not assign temp_pwm in this branch — hold previous command.
    elif (
        state.temp_check_flag == 0.0
        and -cfg.temp_range_final <= err <= cfg.temp_range_final
    ):
        state.phase = "pi"
        pwm = state.pwm_offset + p_term + i_term
    else:
        state.phase = "pd"
        pwm = state.pwm_offset + p_term + d_term
        if (
            state.temp_check_flag == 0.0
            and state.temp_average < -cfg.temp_range_final
            and (state.n_frame_heat - state.prev_frame_heat) > 5
        ):
            state.pwm_offset -= 0.7
            state.prev_frame_heat = state.n_frame_heat
            state.temp_sum = 0.0
            state.count = 1
        elif (
            state.temp_check_flag == 0.0
            and state.temp_average > cfg.temp_range_final
            and (state.n_frame_heat - state.prev_frame_heat) > 5
        ):
            state.pwm_offset += 1.0
            state.prev_frame_heat = state.n_frame_heat
            state.temp_sum = 0.0
            state.count = 1

    if state.count > 5:
        state.count = 1
        state.temp_sum = 0.0
    else:
        state.count += 1

    slew = cfg.slew_stable if state.frame_temp_st != 0 else cfg.slew_heating
    pwm = min(pwm, state.temp_pwm_prev + slew)
    pwm = max(pwm, state.temp_pwm_prev - slew)
    if pwm < cfg.pwm_min:
        pwm = cfg.pwm_min

    # TTN_Set_Ipel — clip to [pwm_min, pwm_max]
    pwm_delivered = float(np.clip(pwm, cfg.pwm_min, cfg.pwm_max))

    state.temp_err_old = err
    state.temp_pwm_prev = pwm_delivered
    return pwm_delivered


def plant_step(
    y: float,
    y0: float,
    pwm: float,
    dt: float,
    k_plant: float,
    tau: float,
    delay_buf: list,
) -> float:
    """First-order plant with transport delay buffer."""
    dy = (k_plant * pwm - (y - y0)) / max(tau, 1e-3)
    y_new = y + dy * dt
    delay_buf.append(y_new)
    delay_buf.pop(0)
    return y_new


def simulate_firmware_pid(
    gains: PidGains,
    setpoint: float,
    y0: float,
    n_steps: int,
    dt: float,
    k_plant: float,
    tau: float,
    l_delay_s: float,
    cfg: Optional[FirmwareSimConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, FirmwarePidState]:
    """
    Closed-loop run on the ``temp_avg_Vs`` scale (matches ``temp_ref`` / temp_log linearized).

    Returns time_s, temp_vs, pwm_pct, final controller state.
    """
    cfg = replace(cfg or FirmwareSimConfig(), temp_ref=setpoint)

    state = FirmwarePidState(pwm_offset=cfg.pwm_offset)
    y = np.zeros(n_steps, dtype=float)
    pwm = np.zeros(n_steps, dtype=float)
    y[0] = y0
    pwm[0] = cfg.pwm_min

    l_steps = max(int(l_delay_s / max(dt, 1e-6)), 1)
    delay_buf = [y0] * l_steps

    for i in range(1, n_steps):
        # Frame 0 behaviour: Vs reported as 0 after init (L_ttn.c n_frame_heat==0 path).
        if state.n_frame_heat == 0:
            y_vs = 0.0
        else:
            y_vs = y[i - 1]

        u = firmware_pid_step(state, y_vs, gains, cfg)
        pwm[i] = u
        y[i] = plant_step(y[i - 1], y0, u, dt, k_plant, tau, delay_buf)

    time_s = np.arange(n_steps, dtype=float) * dt
    return time_s, y, pwm, state


def build_plant_from_measured(
    time_s: np.ndarray,
    temp_lin: np.ndarray,
    pwm_max: float = DEFAULT_PWM_MAX,
) -> Tuple[float, float, float, float]:
    _k_unit, tau, l_delay = identify_fopdt(time_s, temp_lin)
    k_plant = calibrate_plant_gain_from_measured(temp_lin, pwm_sat=pwm_max)
    dt = float(np.median(np.diff(time_s))) if time_s.size > 1 else 1.0
    return k_plant, tau, l_delay, max(dt, 1e-3)
