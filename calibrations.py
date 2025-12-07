"""Calibrations and validation helpers for ECU-style configuration.

This module keeps tunable parameters out of the core control loop and lets us
validate ranges before applying them to the runtime objects.  JSON/YAML-like
structures can be loaded by :func:`load_aeb_calibration`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass
class AEBPlanningCalibration:
    min_aeb_speed: float
    gate_confirm_frames: int
    ttc_confirm_s: float
    ttc_stage_strong: float
    ttc_stage_full: float
    stage_factor_comfort: float
    stage_factor_strong: float
    aeb_ramp_up: float
    aeb_ramp_down: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def clamp_and_validate(self) -> None:
        self.min_aeb_speed = max(0.0, float(self.min_aeb_speed))
        self.gate_confirm_frames = max(1, int(self.gate_confirm_frames))
        self.ttc_confirm_s = max(0.5, float(self.ttc_confirm_s))
        self.ttc_stage_strong = max(0.1, min(self.ttc_confirm_s, float(self.ttc_stage_strong)))
        self.ttc_stage_full = max(0.05, min(self.ttc_stage_strong - 0.05, float(self.ttc_stage_full)))
        self.stage_factor_comfort = min(0.95, max(0.1, float(self.stage_factor_comfort)))
        self.stage_factor_strong = min(0.99, max(self.stage_factor_comfort + 0.01, float(self.stage_factor_strong)))
        self.aeb_ramp_up = max(0.5, float(self.aeb_ramp_up))
        self.aeb_ramp_down = max(0.5, float(self.aeb_ramp_down))


@dataclass
class SafetyCalibration:
    """Safety envelopes and freshness thresholds applied across ECUs."""

    perception_freshness_s: float
    planning_freshness_s: float
    actuation_freshness_s: float
    ttc_floor_s: float
    v_min_plausible: float
    wheel_slip_max: float
    brake_fail_safe: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def clamp_and_validate(self) -> None:
        self.perception_freshness_s = max(0.05, float(self.perception_freshness_s))
        self.planning_freshness_s = max(0.05, float(self.planning_freshness_s))
        self.actuation_freshness_s = max(0.05, float(self.actuation_freshness_s))
        self.ttc_floor_s = max(0.05, float(self.ttc_floor_s))
        self.v_min_plausible = max(0.0, float(self.v_min_plausible))
        self.wheel_slip_max = min(1.0, max(0.0, float(self.wheel_slip_max)))
        self.brake_fail_safe = min(1.0, max(0.5, float(self.brake_fail_safe)))


@dataclass
class BusTopicCalibration:
    """Topic-level communication characteristics for the in-process bus."""

    drop_rate: float
    jitter_s: float
    max_age_s: float
    max_depth: int
    deadline_s: float
    priority: int = 0

    def clamp_and_validate(self) -> None:
        self.drop_rate = min(1.0, max(0.0, float(self.drop_rate)))
        self.jitter_s = max(0.0, float(self.jitter_s))
        self.max_age_s = max(0.05, float(self.max_age_s))
        self.deadline_s = max(0.01, float(self.deadline_s))
        self.max_depth = max(1, int(self.max_depth))
        self.priority = max(0, int(self.priority))


def _load_json_like(path: str | None, defaults: Dict[str, Any]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = dict(defaults)
    checksum: Tuple[str, str] | None = None
    if path:
        try:
            with open(path, "r") as fp:
                blob = fp.read()
                checksum = ("sha256", hashlib.sha256(blob.encode("utf-8")).hexdigest())
                loaded = json.loads(blob)
            if isinstance(loaded, dict):
                cfg.update({k: v for k, v in loaded.items() if k in cfg})
                cfg["metadata"] = {
                    "source": path,
                    "provided_keys": sorted(loaded.keys()),
                }
        except Exception as exc:
            cfg["metadata"] = {"source": path, "error": str(exc)}
    if checksum:
        cfg.setdefault("metadata", {})["checksum"] = {checksum[0]: checksum[1]}
    cfg.setdefault("metadata", {})["schema_version"] = "1.0"
    return cfg


def load_aeb_calibration(path: str | None, defaults: Dict[str, Any]) -> AEBPlanningCalibration:
    cfg = _load_json_like(path, defaults)
    calibration = AEBPlanningCalibration(**cfg)
    calibration.clamp_and_validate()
    return calibration


def load_safety_calibration(path: str | None, defaults: Dict[str, Any]) -> SafetyCalibration:
    cfg = _load_json_like(path, defaults)
    calibration = SafetyCalibration(**cfg)
    calibration.clamp_and_validate()
    return calibration


def load_bus_calibration(path: str | None, defaults: Dict[str, Dict[str, Any]]) -> Dict[str, BusTopicCalibration]:
    cfg = {k: dict(v) for k, v in defaults.items()}
    if path:
        try:
            with open(path, "r") as fp:
                blob = fp.read()
                loaded = json.loads(blob)
            if isinstance(loaded, dict):
                for topic, topic_cfg in loaded.items():
                    if topic in cfg and isinstance(topic_cfg, dict):
                        cfg[topic].update({k: v for k, v in topic_cfg.items() if k in cfg[topic]})
                        cfg[topic].setdefault("metadata", {})
                        cfg[topic]["metadata"]["source"] = path
                        cfg[topic]["metadata"]["provided_keys"] = sorted(topic_cfg.keys())
        except Exception:
            pass

    calibrated: Dict[str, BusTopicCalibration] = {}
    for topic, vals in cfg.items():
        cal = BusTopicCalibration(
            drop_rate=vals.get("drop_rate", 0.0),
            jitter_s=vals.get("jitter_s", 0.0),
            max_age_s=vals.get("max_age_s", 0.5),
            max_depth=vals.get("max_depth", 8),
            deadline_s=vals.get("deadline_s", vals.get("max_age_s", 0.5)),
            priority=vals.get("priority", 0),
        )
        cal.clamp_and_validate()
        calibrated[topic] = cal
    return calibrated
