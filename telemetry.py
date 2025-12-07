# telemetry.py
"""Telemetry and scenario logging helpers for braking experiments."""

from __future__ import annotations

import csv
import os
from typing import Optional


class _TelemetryLogger:
    """
    Lightweight telemetry CSV logger for dynamic_brake_state.

    Schema is aligned with offline_braking_analysis.py / results_analysis.py
    and with the argument order used in dynamic_brake_state.App.run:

        maybe_log(
            t, v_mps,
            tau_dyn, D_safety_dyn, sigma_depth,
            a_des, brake,
            lambda_max, abs_factor, mu_est, mu_regime,
            loop_ms, loop_ms_max, detect_ms, latency_ms,
            a_meas, x_rel_m, range_est_m,
            ttc_s,
            gate_hit, gate_confirmed,
            false_stop_flag,
            brake_stage, brake_stage_factor,
            tracker_s_m, tracker_rate_mps,
            lead_track_id, active_track_count,
            sensor_ts, control_ts,
            sensor_to_control_ms,
            actuation_ts, control_to_act_ms,
            sensor_to_act_ms
        )
    """

    def __init__(self, csv_path: str, hz: float = 10.0) -> None:
        folder = os.path.dirname(os.path.abspath(csv_path))
        if folder:
            os.makedirs(folder, exist_ok=True)

        self.path = csv_path
        self.hz = max(1e-6, float(hz))
        self.dt = 1.0 / self.hz
        self.last_time: Optional[float] = None

        self._csv_f = open(csv_path, "w", newline="")
        self._csv = csv.writer(self._csv_f)

        # Column names chosen to match offline_braking_analysis / results_analysis.
        self._csv.writerow(
            [
                "t",  # simulation time [s]
                "v_mps",
                "tau_dyn",
                "D_safety_dyn",
                "sigma_depth",
                "a_des",
                "brake",
                "lambda_max",
                "abs_factor",
                "mu_est",
                "mu_regime",
                "loop_ms",
                "loop_ms_max",
                "detect_ms",
                "latency_ms",
                "a_meas",
                "x_rel_m",
                "range_est_m",
                "ttc_s",
                "gate_hit",
                "gate_confirmed",
                "false_stop_flag",
                "brake_stage",
                "brake_stage_factor",
                "tracker_s_m",
                "tracker_rate_mps",
                "lead_track_id",
                "active_track_count",
                "sensor_ts",
                "control_ts",
                "sensor_to_control_ms",
                "actuation_ts",
                "control_to_act_ms",
                "sensor_to_act_ms",
            ]
        )
        self._csv_f.flush()

    @staticmethod
    def _f(x: Optional[float]) -> str:
        """Convert optional float to a CSV-friendly string ('' -> NaN later)."""
        if x is None:
            return ""
        return f"{float(x):.9g}"

    @staticmethod
    def _b(x: Optional[bool]) -> str:
        """Convert optional bool to '0'/'1' (or '' if None)."""
        if x is None:
            return ""
        return "1" if x else "0"

    @staticmethod
    def _i(x: Optional[int]) -> str:
        if x is None:
            return ""
        return str(int(x))

    def maybe_log(
        self,
        t: float,
        v_mps: float,
        tau_dyn: Optional[float],
        D_safety_dyn: Optional[float],
        sigma_depth: Optional[float],
        a_des: Optional[float],
        brake: float,
        lambda_max: Optional[float],
        abs_factor: Optional[float],
        mu_est: Optional[float],
        mu_regime: Optional[str],
        loop_ms: Optional[float],
        loop_ms_max: Optional[float],
        detect_ms: Optional[float],
        latency_ms: Optional[float],
        a_meas: Optional[float],
        x_rel_m: Optional[float],
        range_est_m: Optional[float],
        ttc_s: Optional[float],
        gate_hit: Optional[bool],
        gate_confirmed: Optional[bool],
        false_stop_flag: Optional[bool],
        brake_stage: Optional[int],
        brake_stage_factor: Optional[float],
        tracker_s_m: Optional[float],
        tracker_rate_mps: Optional[float],
        lead_track_id: Optional[int],
        active_track_count: Optional[int],
        sensor_ts: Optional[float],
        control_ts: Optional[float],
        sensor_to_control_ms: Optional[float],
        actuation_ts: Optional[float],
        control_to_act_ms: Optional[float],
        sensor_to_act_ms: Optional[float],
    ) -> None:
        """
        Log a telemetry sample if enough simulated time has elapsed.

        The first argument `t` is simulation time in seconds. Throttle/ABS
        logic should call this once per control loop, not multiple times per frame.
        """
        # Simple fixed-rate throttling on simulation time
        if self.last_time is not None and (t - self.last_time) < self.dt:
            return
        self.last_time = t

        row = [
            self._f(t),
            self._f(v_mps),
            self._f(tau_dyn),
            self._f(D_safety_dyn),
            self._f(sigma_depth),
            self._f(a_des),
            self._f(brake),
            self._f(lambda_max),
            self._f(abs_factor),
            self._f(mu_est),
            (mu_regime or ""),
            self._f(loop_ms),
            self._f(loop_ms_max),
            self._f(detect_ms),
            self._f(latency_ms),
            self._f(a_meas),
            self._f(x_rel_m),
            self._f(range_est_m),
            self._f(ttc_s),
            self._b(gate_hit),
            self._b(gate_confirmed),
            self._b(false_stop_flag),
            self._i(brake_stage),
            self._f(brake_stage_factor),
            self._f(tracker_s_m),
            self._f(tracker_rate_mps),
            self._i(lead_track_id),
            self._i(active_track_count),
            self._f(sensor_ts),
            self._f(control_ts),
            self._f(sensor_to_control_ms),
            self._f(actuation_ts),
            self._f(control_to_act_ms),
            self._f(sensor_to_act_ms),
        ]
        self._csv.writerow(row)
        # Flush for safety; you can remove this for performance
        self._csv_f.flush()

    def close(self) -> None:
        try:
            self._csv_f.close()
        except Exception:
            pass


class _ScenarioLogger:
    """
    High-level braking episode logger.

    CSV schema is designed to be consumed by results_analysis / offline tools.
    """

    HEADER = [
        "scenario",
        "trigger_kind",
        "mu",
        "v_init_mps",
        "s_init_m",
        "s_min_m",
        "s_init_gt_m",
        "s_min_gt_m",
        "stopped",
        "t_to_stop_s",
        "collision",
        "range_margin_m",
        "tts_margin_s",
        "ttc_init_s",
        "ttc_min_s",
        "reaction_time_s",
        "max_lambda",
        "mean_abs_factor",
        "false_stop",
    ]

    def __init__(self, csv_path: str) -> None:
        folder = os.path.dirname(os.path.abspath(csv_path))
        if folder:
            os.makedirs(folder, exist_ok=True)

        self.path = csv_path
        self._csv_f = open(csv_path, "w", newline="")
        self._csv = csv.writer(self._csv_f)
        self._csv.writerow(self.HEADER)
        self._csv_f.flush()

    def log(
        self,
        scenario: str,
        trigger_kind: str,
        mu: float,
        v_init: float,
        s_init: float,
        s_min: float,
        s_init_gt: Optional[float],
        s_min_gt: Optional[float],
        stopped: bool,
        t_to_stop: float,
        collision: bool,
        range_margin: Optional[float],
        tts_margin: Optional[float],
        ttc_init: Optional[float],
        ttc_min: Optional[float],
        reaction_time: Optional[float],
        max_lambda: Optional[float],
        mean_abs_factor: Optional[float],
        false_stop: bool,
    ) -> None:
        row = [
            scenario,
            trigger_kind,
            float(mu),
            float(v_init),
            float(s_init),
            float(s_min),
            ("" if s_init_gt is None else f"{float(s_init_gt):.9g}"),
            ("" if s_min_gt is None else f"{float(s_min_gt):.9g}"),
            "1" if stopped else "0",
            f"{float(t_to_stop):.9g}",
            "1" if collision else "0",
            ("" if range_margin is None else f"{float(range_margin):.9g}"),
            ("" if tts_margin is None else f"{float(tts_margin):.9g}"),
            ("" if ttc_init is None else f"{float(ttc_init):.9g}"),
            ("" if ttc_min is None else f"{float(ttc_min):.9g}"),
            ("" if reaction_time is None else f"{float(reaction_time):.9g}"),
            ("" if max_lambda is None else f"{float(max_lambda):.9g}"),
            ("" if mean_abs_factor is None else f"{float(mean_abs_factor):.9g}"),
            "1" if false_stop else "0",
        ]
        self._csv.writerow(row)
        self._csv_f.flush()

    def close(self) -> None:
        try:
            self._csv_f.close()
        except Exception:
            pass

TelemetryLogger = _TelemetryLogger
ScenarioLogger = _ScenarioLogger