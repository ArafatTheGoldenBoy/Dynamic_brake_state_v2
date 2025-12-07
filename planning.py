# planning.py
"""
AEB / longitudinal planning ECU.

This module factors the braking / AEB logic out of the monolithic app and into
a reusable planner that can be wrapped by :class:`ecu.PlanningECU`.

The goal is *behaviourally similar* to the original `_control_step` in
`dynamic_brake_state.App`, but with a slightly cleaner state container
(:class:`PlanningContext`). 
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from config import (  # type: ignore
    DT,
    A_MAX,
    V_STOP,
    V_TARGET,
    KP_THROTTLE,
    EPS,
    CLEAR_DELAY_OBS,
    CLEAR_DELAY_RED,
    STOP_WAIT_S,
    KICK_SEC,
    KICK_THR,
    TAU_MIN,
    TAU_MAX,
    K_LAT_TAU,
    K_MU_TAU,
    K_UNC_TAU,
    D_MIN,
    K_LAT_D,
    K_UNC_D,
    K_MU_D,
    BRAKE_STAGE_FULL_FACTOR,
    FALSE_STOP_MARGIN_M,
    S_ENGAGE,
    S_ENGAGE_TL,
)
from calibrations import AEBPlanningCalibration  # type: ignore


@dataclass
class PlanningContext:
    """
    Persistent state for the planner between control loop iterations.

    It is deliberately explicit so you can inspect it easily in telemetry or unit
    tests; all fields are safe to initialize with defaults.
    """

    # PI integrator on measured decel tracking
    I_err: float = 0.0

    # Previous ego speed [m/s]
    v_prev: float = 0.0

    # Stop‑sign latch and hold
    stop_armed: bool = False
    stop_latch_time: float = -1.0
    stop_release_ignore_until: float = -1.0

    # Hold / brake state
    hold_blocked: bool = False
    hold_reason: Optional[str] = None
    kick_until: float = 0.0

    # Hazard / gate memory
    last_s0: Optional[float] = None
    gate_confirm_counter: int = 0
    hazard_confirm_since: float = -1.0

    # Timers for “clear” logic
    no_trigger_elapsed: float = 0.0
    no_red_elapsed: float = 0.0
    red_green_since: float = -1.0

    # Optional lead tracking info (for TTC refinement)
    tracked_rate: Optional[float] = None

    # For diagnostics
    sim_time: float = 0.0

    # For simple timer bookkeeping
    last_trigger_active: bool = False
    last_tl_state: str = "UNKNOWN"


@dataclass
class AEBPlannerConfig:
    """
    Wraps calibration + a few environment parameters needed by the planner.
    """

    calibration: AEBPlanningCalibration
    mu: float
    extra_latency_ms: float = 0.0
    range_mode: str = "depth"
    depth_roi_shrink: float = 0.4
    stereo_roi_shrink: float = 0.35

    def __post_init__(self) -> None:
        self.mu = max(0.05, min(1.5, float(self.mu)))
        self.range_mode = (self.range_mode or "depth").lower()
        self.depth_roi_shrink = float(min(0.9, max(0.0, self.depth_roi_shrink)))
        self.stereo_roi_shrink = float(min(0.9, max(0.0, self.stereo_roi_shrink)))


def _depth_sigma_in_box(
    depth_m: Optional[np.ndarray],
    box: Optional[Tuple[int, int, int, int]],
    shrink: float,
) -> Optional[float]:
    """Robust scatter estimate (MAD→sigma) inside nearest‑object ROI."""
    if depth_m is None or box is None:
        return None
    x, y, w, h = box
    shrink = float(min(0.9, max(0.0, shrink)))
    x0 = max(0, int(x + w * shrink / 2.0))
    y0 = max(0, int(y + h * shrink / 2.0))
    x1 = min(depth_m.shape[1], int(x + w - w * shrink / 2.0))
    y1 = min(depth_m.shape[0], int(y + h - h * shrink / 2.0))
    if x1 <= x0 or y1 <= y0:
        return None
    roi = depth_m[y0:y1, x0:x1]
    flat = roi[np.isfinite(roi)]
    if flat.size < 25:
        return None
    med = np.median(flat)
    mad = np.median(np.abs(flat - med))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0.0:
        return None
    return float(sigma)


class AEBPlanner:
    """
    Longitudinal planner / AEB decision logic.

    The main entry point is :meth:`step`, which consumes the current scene and
    updates a :class:`PlanningContext` in place.
    """

    def __init__(self, cfg: AEBPlannerConfig) -> None:
        self.cfg = cfg
        # Effective μ used everywhere in the planner; starts from CLI value
        self.mu_eff = float(cfg.mu)
        self.A_MU = self.mu_eff * 9.81
        self.stage_factor_full = BRAKE_STAGE_FULL_FACTOR
        self._a_des_prev = 0.0  # for ramp‑rate limiting
        self._sigma_depth_ema = 0.40
        self._D_safety_dyn_prev = D_MIN

    def update_mu_from_abs(self, mu_meas: float, alpha: float = 0.2) -> None:
        """
        Update the planner's effective friction μ from ABS feedback.

        Parameters
        ----------
        mu_meas
            Instantaneous μ estimate from ABS (ActuationResult.abs_dbg['mu_est']).
        alpha
            Blend factor for extra low‑pass filtering on top of the ABS estimator.
        """
        try:
            mu_meas_f = float(mu_meas)
        except Exception:
            return

        # Clamp μ to a physically reasonable range
        mu_meas_f = max(0.05, min(1.5, mu_meas_f))

        # Simple EMA towards the ABS estimate
        alpha = max(0.0, min(1.0, float(alpha)))
        self.mu_eff = (1.0 - alpha) * self.mu_eff + alpha * mu_meas_f

        # Keep cfg.mu consistent (used in debug / logging)
        self.cfg.mu = self.mu_eff

        # Update friction‑limited decel cap
        self.A_MU = self.mu_eff * 9.81

    # ------------------------------------------------------------------ core helpers

    def _safety_envelope(
        self,
        ctx: PlanningContext,
        v: float,
        ema_loop_ms: float,
        nearest_box: Optional[Tuple[int, int, int, int]],
        nearest_conf: Optional[float],
        depth_m: Optional[np.ndarray],
        depth_stereo_m: Optional[np.ndarray],
    ) -> Tuple[float, float, float, float]:
        """
        Compute (tau_dyn, D_safety_dyn, sigma_depth, latency_s).

        This is a lightly simplified version of the envelope logic in the
        original app; it’s sufficient to drive the same AEB decisions and
        offline analysis. :contentReference[oaicite:4]{index=4}
        """
        extra_ms = float(self.cfg.extra_latency_ms or 0.0)
        latency_s = max(DT, (ema_loop_ms + extra_ms) / 1000.0) + 0.03

        # Depth uncertainty estimate
        sd = None
        if nearest_box is not None:
            if self.cfg.range_mode == "stereo" and depth_stereo_m is not None:
                sd = _depth_sigma_in_box(
                    depth_stereo_m, nearest_box, self.cfg.stereo_roi_shrink
                )
            if sd is None and depth_m is not None:
                sd = _depth_sigma_in_box(
                    depth_m, nearest_box, self.cfg.depth_roi_shrink
                )

        if sd is not None and np.isfinite(sd):
            target = float(max(0.05, min(3.0, sd)))
        else:
            target = self._sigma_depth_ema

        # Simple EMA with clipped step size
        delta = max(-0.50, min(0.50, target - self._sigma_depth_ema))
        self._sigma_depth_ema += 0.30 * delta
        sigma_depth = self._sigma_depth_ema

        c_near = nearest_conf if (nearest_conf is not None) else 0.5
        # Use dynamic μ here
        mu = float(self.mu_eff)
        mu_short = max(0.0, 0.90 - mu)

        tau_dyn = TAU_MIN + K_LAT_TAU * latency_s + K_MU_TAU * mu_short + K_UNC_TAU * (
            1.0 - c_near
        )
        tau_dyn = max(TAU_MIN, min(TAU_MAX, tau_dyn))

        D_safety_dyn = (
            D_MIN
            + K_LAT_D * (v * latency_s)
            + K_UNC_D * sigma_depth
            + K_MU_D * mu_short
        )
        # Gentle rate limiting to avoid big jumps frame‑to‑frame
        D_safety_dyn = self._D_safety_dyn_prev + max(
            -1.0, min(1.0, D_safety_dyn - self._D_safety_dyn_prev)
        )
        self._D_safety_dyn_prev = D_safety_dyn

        return tau_dyn, D_safety_dyn, sigma_depth, latency_s

    # ------------------------------------------------------------------ public API

    def step(
        self,
        ctx: PlanningContext,
        *,
        sim_time: float,
        trigger_name: Optional[str],
        nearest_s_active: Optional[float],
        nearest_thr: Optional[float],
        tl_state: str,
        tl_s_active: Optional[float],
        v: float,
        v_target: Optional[float],
        ema_loop_ms: float,
        tracked_distance_for_control: Optional[float],
        depth_m: Optional[np.ndarray],
        depth_stereo_m: Optional[np.ndarray],
        nearest_box: Optional[Tuple[int, int, int, int]],
        nearest_conf: Optional[float],
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        One planning step.

        Parameters
        ----------
        ctx
            PlanningContext holding integrator, stop‑latch state, etc.
        sim_time
            Simulation time [s] at this control tick.
        trigger_name
            Human‑readable label for the current braking trigger (car, TL, etc.).
        nearest_s_active
            Distance [m] to the closest gated object in front of ego.
        nearest_thr
            Engage distance threshold for that object (S_ENGAGE, etc.).
        tl_state, tl_s_active
            Traffic‑light state ('RED'/'YELLOW'/'GREEN'/'UNKNOWN') and distance.
        v
            Ego speed [m/s].
        v_target
            Desired cruise speed [m/s] when not braking. If None, uses V_TARGET.
        ema_loop_ms
            Exponential‑moving‑average of loop period [ms].
        tracked_distance_for_control
            Optional distance estimate from a tracking filter; if provided, it
            is used as the “main” stopping‑distance reference.
        depth_m, depth_stereo_m, nearest_box, nearest_conf
            Used by the safety envelope for TTC / uncertainty.

        Returns
        -------
        throttle, brake, debug
            Where debug is a flat dict with tau_dyn, D_safety_dyn, ttc, etc.
        """
        v_target = float(v_target if v_target is not None else V_TARGET)
        ctx.sim_time = sim_time

        A_MU = self.mu_eff * 9.81

        throttle = 0.0
        brake = 0.0

        # -------------------------------------- choose “reason” for braking

        reason_object = False
        reason_stop = False
        reason_red = False

        if (
            nearest_s_active is not None
            and nearest_thr is not None
            and nearest_s_active <= nearest_thr
        ):
            reason_object = True
        elif ctx.stop_armed and (not ctx.hold_blocked):
            reason_stop = True
            if nearest_s_active is None and ctx.last_s0 is not None:
                nearest_s_active = ctx.last_s0
        elif (
            tl_state == "RED"
            and tl_s_active is not None
            and tl_s_active <= S_ENGAGE_TL
        ):
            reason_red = True
            nearest_s_active = tl_s_active


        # --- Update clear timers / TL state memory --------------------------

        active_trigger = reason_object or reason_stop or reason_red
        if active_trigger:
            ctx.no_trigger_elapsed = 0.0
        else:
            ctx.no_trigger_elapsed += DT

        if tl_state == "RED":
            ctx.no_red_elapsed = 0.0
        elif tl_state == "GREEN":
            if ctx.last_tl_state != "GREEN":
                # First frame we see GREEN after non‑GREEN
                ctx.red_green_since = sim_time
            ctx.no_red_elapsed += DT
        else:
            ctx.no_red_elapsed += DT

        ctx.last_tl_state = tl_state
        ctx.last_trigger_active = active_trigger

        # -------------------------------------- dynamic envelope / TTC

        s_used: Optional[float] = None
        if tracked_distance_for_control is not None:
            s_used = tracked_distance_for_control
        elif nearest_s_active is not None:
            s_used = nearest_s_active
        elif ctx.last_s0 is not None:
            s_used = ctx.last_s0

        tau_dyn: Optional[float] = None
        D_safety_dyn: Optional[float] = None
        sigma_depth: Optional[float] = None
        latency_s: Optional[float] = None
        required_dist_physics: Optional[float] = None
        gate_hit = False
        s_eff: Optional[float] = None

        if s_used is not None:
            (
                tau_dyn,
                D_safety_dyn,
                sigma_depth,
                latency_s,
            ) = self._safety_envelope(
                ctx, v, ema_loop_ms, nearest_box, nearest_conf, depth_m, depth_stereo_m
            )
            required_dist_physics = (v * v) / (2.0 * max(1e-3, A_MU)) + v * tau_dyn + D_safety_dyn
            gate_hit = required_dist_physics >= s_used
            if tau_dyn is not None and D_safety_dyn is not None:
                s_eff = max(EPS, s_used - D_safety_dyn - v * tau_dyn)

        # TTC estimate (optionally refined by tracking rate)
        ttc: Optional[float] = None
        if s_used is not None:
            closing_speed = v
            if ctx.tracked_rate is not None and math.isfinite(ctx.tracked_rate):
                if ctx.tracked_rate < -0.05:  # distance decreasing
                    closing_speed = max(0.1, -ctx.tracked_rate)
            closing_speed = max(0.1, closing_speed)
            ttc = s_used / closing_speed

        in_brake_band = reason_object or reason_stop or reason_red

        # Map to a canonical brake reason string for telemetry
        brake_reason: Optional[str] = None
        if reason_stop:
            brake_reason = "stop_sign"
        elif reason_red:
            brake_reason = "red_light"
        elif reason_object:
            brake_reason = "obstacle"
        if brake_reason is None and trigger_name:
            tnorm = trigger_name.lower()
            if "stop" in tnorm and "sign" in tnorm:
                brake_reason = "stop_sign"
            elif "traffic" in tnorm and "light" in tnorm:
                brake_reason = "red_light"

        # Stop / TL specific tweaks
        if reason_stop:
            if trigger_name is None:
                trigger_name = "stop sign"
            nearest_thr = S_ENGAGE
        if reason_red:
            nearest_thr = S_ENGAGE_TL
            trigger_name = "traffic light (RED)"

        # -------------------------------------- AEB gating / confirmation

        gate_confirmed = gate_hit
        object_ready = reason_object

        if reason_object:
            if gate_hit:
                ctx.gate_confirm_counter = min(
                    ctx.gate_confirm_counter + 1, self.cfg.calibration.gate_confirm_frames * 2
                )
            else:
                ctx.gate_confirm_counter = max(0, ctx.gate_confirm_counter - 1)
            gate_confirmed = (
                ctx.gate_confirm_counter >= self.cfg.calibration.gate_confirm_frames
            )
            ttc_ok = (ttc is None) or (ttc <= self.cfg.calibration.ttc_confirm_s)
            speed_ok = v >= self.cfg.calibration.min_aeb_speed
            object_ready = gate_confirmed and ttc_ok and speed_ok
        else:
            ctx.gate_confirm_counter = 0
            gate_confirmed = False
            object_ready = False

        in_brake_band = object_ready or reason_stop or reason_red

        # -------------------------------------- multi‑stage AEB shaping

        brake_stage = 0
        stage_factor = 0.0
        if in_brake_band:
            if (ttc is None) or (ttc > self.cfg.calibration.ttc_stage_strong):
                brake_stage = 1
            if (ttc is not None) and (ttc <= self.cfg.calibration.ttc_stage_strong):
                brake_stage = 2
            if (ttc is not None) and (ttc <= self.cfg.calibration.ttc_stage_full):
                brake_stage = 3
            if reason_stop or reason_red:
                brake_stage = max(1, brake_stage)

            stage_map = {
                1: self.cfg.calibration.stage_factor_comfort,
                2: self.cfg.calibration.stage_factor_strong,
                3: self.stage_factor_full,
            }
            stage_factor = stage_map.get(brake_stage, 0.0)

        dbg_target = s_used if s_used is not None else nearest_s_active

        dbg: Dict[str, Any] = {
            "tau_dyn": tau_dyn,
            "D_safety_dyn": D_safety_dyn,
            "sigma_depth": sigma_depth,
            "gate_hit": gate_hit,
            "a_des": None,
            "brake": None,
            "brake_active": False,
            "brake_reason": brake_reason,
            "brake_target_dist": dbg_target,
            "latency_s": latency_s,
            "ttc": ttc,
            "gate_confirmed": gate_confirmed,
            "brake_stage": brake_stage,
            "brake_stage_factor": stage_factor,
        }

        # -------------------------------------- active braking or cruise / hold

        if (
            in_brake_band
            and s_used is not None
            and tau_dyn is not None
            and D_safety_dyn is not None
        ):
            # AEB / braking mode
            if not gate_hit and s_eff is not None:
                a_candidate = min(A_MAX, (v * v) / (2.0 * max(EPS, s_eff)))
                a_candidate = min(a_candidate, A_MU)
            else:
                a_candidate = A_MU if gate_hit else min(A_MU, A_MAX)

            # Stage‑limited max decel
            if brake_stage == 1:
                stage_limit = self.cfg.calibration.stage_factor_comfort * A_MU
            elif brake_stage == 2:
                stage_limit = self.cfg.calibration.stage_factor_strong * A_MU
            elif brake_stage >= 3:
                stage_limit = self.stage_factor_full * A_MU
            else:
                stage_limit = None

            if stage_limit is not None:
                a_target = min(a_candidate, stage_limit)
            else:
                a_target = a_candidate

            # Ramp rate limiter on a_des
            prev_a = self._a_des_prev
            ramp_up = self.cfg.calibration.aeb_ramp_up * DT
            ramp_down = self.cfg.calibration.aeb_ramp_down * DT
            if a_target > prev_a:
                a_des = min(a_target, prev_a + ramp_up)
            else:
                a_des = max(a_target, prev_a - ramp_down)
            self._a_des_prev = a_des

            # Feed‑forward brake from commanded decel
            brake_ff = max(0.0, min(1.0, a_des / A_MAX))

            a_meas = (
                0.0
                if ctx.v_prev is None
                else max(0.0, (ctx.v_prev - v) / DT)
            )
            e = max(0.0, a_des - a_meas)
            ctx.I_err = max(
                -self.cfg.calibration.metadata.get("I_max", 8.0),
                min(self.cfg.calibration.metadata.get("I_max", 8.0),
                    ctx.I_err + e * DT),
            )
            # Use the original gains from config.py
            from config import KPB, KIB  # local import to avoid cycles  # type: ignore

            brake = max(
                0.0,
                min(1.0, brake_ff + (KPB * e + KIB * ctx.I_err) / A_MAX),
            )

            # Hold latching when fully stopped
            if v < V_STOP:
                ctx.hold_blocked = True
                if brake_reason == "red_light":
                    ctx.hold_reason = "red_light"
                elif brake_reason == "stop_sign":
                    ctx.hold_reason = "stop_sign"
                    ctx.stop_latch_time = sim_time
                    ctx.stop_armed = False
                else:
                    ctx.hold_reason = "obstacle"
            else:
                ctx.hold_reason = brake_reason or ctx.hold_reason

            dbg.update(
                {
                    "a_des": a_des,
                    "brake": brake,
                    "a_des_target": a_target,
                    "brake_active": True,
                }
            )

        elif ctx.hold_blocked:
            # Hold mode: stay stopped until a clear condition is met.
            release = False
            if ctx.hold_reason == "red_light":
                if tl_state == "GREEN" and ctx.red_green_since >= 0.0:
                    if (sim_time - ctx.red_green_since) >= CLEAR_DELAY_RED:
                        release = True
            elif ctx.hold_reason == "stop_sign":
                if (sim_time - ctx.stop_latch_time) >= STOP_WAIT_S:
                    release = True
            else:  # generic obstacle hold
                if ctx.no_trigger_elapsed >= CLEAR_DELAY_OBS:
                    release = True

            if release:
                ctx.hold_blocked = False
                ctx.hold_reason = None
                ctx.last_s0 = None
                self._a_des_prev = 0.0
                throttle, brake = 0.0, 0.0
                ctx.kick_until = sim_time + KICK_SEC
                ctx.stop_release_ignore_until = sim_time + 2.0
            else:
                throttle, brake = 0.0, 1.0
        else:
            # Cruise / follow mode
            e_v = v_target - v
            throttle = max(0.0, min(1.0, KP_THROTTLE * e_v))
            brake = 0.0
            self._a_des_prev = 0.0

            # Start‑up “kick” from rest
            if sim_time < ctx.kick_until and v < 0.3:
                throttle = max(throttle, KICK_THR)
            if not ctx.hold_blocked and v < 0.25:
                throttle = max(throttle, 0.35)

        # Update context + debug
        ctx.v_prev = v
        if s_used is not None:
            ctx.last_s0 = s_used

        dbg["brake"] = brake

        return throttle, brake, dbg
