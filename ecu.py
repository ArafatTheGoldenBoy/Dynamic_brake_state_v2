"""ECU-oriented adapters for the dynamic brake state app.

The existing :class:`App` class already splits perception and control into
private helpers (``_perception_step`` and ``_control_step``).  To make the
architecture more modular for an automotive-style ECU layout, these adapters
delegate the existing helpers behind explicit ECU interfaces.  This keeps
behavior unchanged while clarifying which subsystem owns which responsibility.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import queue as queue_mod
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


def _clamp(value: float, bounds: Tuple[float, float]) -> float:
    lo, hi = bounds
    return max(lo, min(hi, value))


@dataclass
class ComponentStatus:
    """Health status emitted by each ECU for safety aggregation."""

    name: str
    ok: bool
    faults: List[str] = field(default_factory=list)
    severity: str = "INFO"
    timestamp: float = field(default_factory=lambda: time.time())

    def mark_fault(self, code: str, severity: str = "WARN") -> None:
        if code not in self.faults:
            self.faults.append(code)
        self.ok = False
        self.severity = severity


@dataclass
class PerceptionSignal:
    """CAN/Ethernet friendly perception payload.

    This mirrors the detection/fusion products that a dedicated camera/radar ECU
    would broadcast.  The mapping keeps the rest of the app unchanged while
    documenting what gets exchanged between ECUs instead of direct function
    calls.
    """

    bgr: Any
    det_points: Iterable[Any]
    nearest_s_active: Optional[float]
    nearest_kind: Optional[str]
    nearest_thr: Optional[float]
    nearest_box: Optional[Any]
    nearest_conf: Optional[float]
    tl_state: str
    tl_s_active: Optional[float]
    tl_det_box: Optional[Any]
    stop_detected_current: bool
    obstacle_measurements: Iterable[Any] = field(default_factory=list)
    timestamp: float = field(default_factory=lambda: time.time())
    frame_id: Optional[int] = None
    valid: bool = True
    fault_code: Optional[str] = None
    freshness_s: float = 0.5
    units: Dict[str, str] = field(default_factory=lambda: {
        "nearest_s_active": "m",
        "nearest_conf": "[0,1]",
        "tl_s_active": "m",
    })

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def validate(self, freshness_s: Optional[float] = None) -> None:
        now = time.time()
        freshness = self.freshness_s if freshness_s is None else freshness_s
        if not math.isfinite(self.timestamp) or (now - float(self.timestamp) > freshness):
            self.valid = False
            self.fault_code = self.fault_code or "STALE_PERCEPTION"
        if self.nearest_s_active is not None:
            if not math.isfinite(self.nearest_s_active) or self.nearest_s_active < 0:
                self.valid = False
                self.fault_code = self.fault_code or "NEGATIVE_DISTANCE"
        if self.nearest_conf is not None and (self.nearest_conf < 0 or self.nearest_conf > 1.0):
            self.valid = False
            self.fault_code = self.fault_code or "CONFIDENCE_OOB"
        if self.nearest_conf is not None:
            self.nearest_conf = _clamp(float(self.nearest_conf), (0.0, 1.0))
        if self.tl_s_active is not None and self.tl_s_active < 0:
            self.valid = False
            self.fault_code = self.fault_code or "TL_DISTANCE_INVALID"
        if self.obstacle_measurements:
            for m in self.obstacle_measurements:
                try:
                    dist_val = float(m.get("distance", 0.0))
                except Exception:
                    dist_val = -1.0
                if dist_val < 0:
                    self.valid = False
                    self.fault_code = self.fault_code or "OBSTACLE_DISTANCE_INVALID"


@dataclass
class AEBRequest:
    """High-level deceleration request sent from ADAS domain ECU to brake ECU."""

    mode: str
    target_decel: float
    priority: str = "HIGH"
    units: Dict[str, str] = field(default_factory=lambda: {"target_decel": "m/s^2"})

    def validate(self) -> None:
        self.target_decel = float(self.target_decel)
        if self.target_decel < 0.0:
            raise ValueError("target_decel must be non-negative")
        self.priority = self.priority.upper()


@dataclass
class PlanningDecision:
    """Encapsulates planning output plus the high-level AEB request."""

    throttle: float
    brake: float
    ctrl: Optional[Any]
    hold_blocked: bool
    hold_reason: Optional[str]
    stop_armed: bool
    stop_latch_time: float
    stop_release_ignore_until: float
    debug: Dict[str, Any]
    integral_error: float
    aeb_request: Optional[AEBRequest]
    timestamp: float = field(default_factory=lambda: time.time())
    valid: bool = True
    fault_code: Optional[str] = None
    units: Dict[str, str] = field(default_factory=lambda: {
        "throttle": "[0,1]",
        "brake": "[0,1]",
    })

    def validate(self, freshness_s: Optional[float] = None) -> None:
        now = time.time()
        if not math.isfinite(self.timestamp):
            self.valid = False
            self.fault_code = self.fault_code or "BAD_TIMESTAMP"
        elif freshness_s is not None and (now - float(self.timestamp) > freshness_s):
            self.valid = False
            self.fault_code = self.fault_code or "PLANNING_STALE"
        self.throttle = _clamp(float(self.throttle), (0.0, 1.0))
        self.brake = _clamp(float(self.brake), (0.0, 1.0))
        if self.aeb_request is not None:
            try:
                self.aeb_request.validate()
            except Exception as exc:
                self.valid = False
                self.fault_code = self.fault_code or f"AEB_REQ_INVALID:{exc}"

    def as_tuple(self):
        return (
            self.throttle,
            self.brake,
            self.ctrl,
            self.hold_blocked,
            self.hold_reason,
            self.stop_armed,
            self.stop_latch_time,
            self.stop_release_ignore_until,
            self.debug,
            self.integral_error,
        )


@dataclass
class ActuationResult:
    """Final actuator-facing command plus optional ABS debug signals."""

    brake: float
    abs_dbg: Optional[Dict[str, Any]] = None
    fault_code: Optional[str] = None
    timestamp: float = field(default_factory=lambda: time.time())
    valid: bool = True
    freshness_s: float = 0.5

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def validate(self, freshness_s: Optional[float] = None) -> None:
        now = time.time()
        if not math.isfinite(self.timestamp):
            self.valid = False
            self.fault_code = self.fault_code or "ACT_TS_INVALID"
        elif freshness_s is not None and (now - float(self.timestamp) > freshness_s):
            self.valid = False
            self.fault_code = self.fault_code or "ACT_STALE"
        self.brake = _clamp(float(self.brake), (0.0, 1.0))


@dataclass
class PerceptionECU:
    """Wraps the application's perception helper for ECU-style separation."""

    perception_fn: Callable[..., Dict[str, Any]]

    def process(
        self,
        bgr,
        depth_m,
        depth_stereo_m,
        fx,
        fy,
        cx,
        cy,
        sim_time: float,
        sensor_timestamp: Optional[float],
        v: float,
        mu: float,
        log_both: bool,
        csv_writer,
        **kwargs,
    ) -> PerceptionSignal:
        """Run perception for the current frame and return the decoded state."""

        raw = self.perception_fn(
            bgr,
            depth_m,
            depth_stereo_m,
            fx,
            fy,
            cx,
            cy,
            sim_time,
            sensor_timestamp,
            v,
            mu,
            log_both,
            csv_writer,
            **kwargs,
        )
        return PerceptionSignal(
            bgr=raw.get("bgr"),
            det_points=raw.get("det_points", []),
            nearest_s_active=raw.get("nearest_s_active"),
            nearest_kind=raw.get("nearest_kind"),
            nearest_thr=raw.get("nearest_thr"),
            nearest_box=raw.get("nearest_box"),
            nearest_conf=raw.get("nearest_conf"),
            tl_state=raw.get("tl_state", "UNKNOWN"),
            tl_s_active=raw.get("tl_s_active"),
            tl_det_box=raw.get("tl_det_box"),
            stop_detected_current=raw.get("stop_detected_current", False),
            obstacle_measurements=raw.get("obstacle_measurements", []),
        )


@dataclass
class PlanningECU:
    """Delegates control computation to the application's control helper."""

    control_fn: Callable[..., Any]
    context: Optional[Any] = None  # e.g. planning.PlanningContext
    def plan(self, *args, **kwargs) -> PlanningDecision:
        """Compute throttle/brake commands based on the current world state."""

        # If a PlanningContext is provided, treat control_fn as AEBPlanner.step
        if self.context is not None:
            # Filter kwargs down to those actually accepted by control_fn
            import inspect

            if not hasattr(self, "_allowed_kwargs"):
                sig = inspect.signature(self.control_fn)
                allowed = []
                for name, param in sig.parameters.items():
                    if name == "ctx":
                        continue
                    if param.kind in (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    ):
                        allowed.append(name)
                self._allowed_kwargs = set(allowed)
            filtered = {k: v for k, v in kwargs.items() if k in self._allowed_kwargs}
            result = self.control_fn(self.context, **filtered)
        else:
            # Legacy path: pass through args/kwargs directly
            result = self.control_fn(*args, **kwargs)

        if not isinstance(result, tuple):
            raise ValueError("control_fn must return a tuple")

        # New planner: (throttle, brake, dbg)
        if len(result) == 3:
            throttle, brake, dbg_map = result
            ctrl = None
            ctx = self.context

            if ctx is not None:
                hold_blocked = bool(getattr(ctx, "hold_blocked", False))
                hold_reason = getattr(ctx, "hold_reason", None)
                stop_armed = bool(getattr(ctx, "stop_armed", False))
                stop_latch_time = float(
                    getattr(ctx, "stop_latch_time", -1.0) or -1.0
                )
                stop_release_ignore_until = float(
                    getattr(ctx, "stop_release_ignore_until", -1.0) or -1.0
                )
                I_err = float(getattr(ctx, "I_err", 0.0) or 0.0)
            else:
                hold_blocked = False
                hold_reason = None
                stop_armed = False
                stop_latch_time = -1.0
                stop_release_ignore_until = -1.0
                I_err = 0.0

        # Legacy control helper: 10‑tuple as in the original app
        elif len(result) == 10:
            (
                throttle,
                brake,
                ctrl,
                hold_blocked,
                hold_reason,
                stop_armed,
                stop_latch_time,
                stop_release_ignore_until,
                dbg_map,
                I_err,
            ) = result
        else:
            raise ValueError("control_fn must return a 3‑ or 10‑tuple")

        a_des = None if dbg_map is None else dbg_map.get("a_des")
        aeb_request = None
        if a_des is not None:
            aeb_request = AEBRequest(mode="AEB", target_decel=float(a_des), priority="HIGH")

        return PlanningDecision(
            throttle=throttle,
            brake=brake,
            ctrl=ctrl,
            hold_blocked=bool(hold_blocked),
            hold_reason=hold_reason,
            stop_armed=bool(stop_armed),
            stop_latch_time=float(stop_latch_time),
            stop_release_ignore_until=float(stop_release_ignore_until),
            debug=dbg_map or {},
            integral_error=float(I_err),
            aeb_request=aeb_request,
        )



@dataclass
class ActuationECU:
    """Handles actuation-level tasks such as ABS shaping."""

    abs_fn: Optional[Callable[[float, float, Any, float], float]] = None
    abs_debug_fn: Optional[Callable[[], Dict[str, Any]]] = None

    def apply_abs(
        self, brake_cmd: float, v_ego: float, wheel_speeds, a_long: float
    ) -> ActuationResult:
        """Apply ABS shaping and return both command and debug info."""

        if self.abs_fn is None:
            return ActuationResult(brake=brake_cmd, abs_dbg=None)

        brake_out = brake_cmd
        abs_dbg = None
        fault_code = None
        try:
            brake_out = self.abs_fn(brake_cmd, v_ego, wheel_speeds, a_long)
            abs_dbg = self.abs_debug_fn() if self.abs_debug_fn is not None else None
        except Exception as exc:
            fault_code = f"ABS_FAIL:{exc}"

        valid = fault_code is None
        return ActuationResult(brake=brake_out, abs_dbg=abs_dbg, fault_code=fault_code, valid=valid)


class MessageBus:
    """Bus that approximates ECU message passing with optional shared state."""

    def __init__(self, shared_manager: Optional[mp.Manager] = None):
        self._manager = shared_manager
        if shared_manager is None:
            self._queues: Dict[str, List[Tuple[float, int, int, Any]]] = {}
            self._topic_config: Dict[str, Dict[str, float]] = {}
            self.metrics: Dict[str, Dict[str, float]] = {}
        else:
            self._queues = shared_manager.dict()
            self._topic_config = shared_manager.dict()
            self.metrics = shared_manager.dict()
        self._locks: Dict[str, mp.Lock] = {}

    def configure_topic(
        self,
        topic: str,
        drop_rate: float = 0.0,
        jitter_s: float = 0.0,
        max_age_s: float = 0.5,
        max_depth: int = 8,
        deadline_s: float = 0.25,
        priority: int = 0,
    ) -> None:
        cfg = {
            "drop_rate": max(0.0, min(1.0, drop_rate)),
            "jitter_s": max(0.0, jitter_s),
            "max_age_s": max(0.05, max_age_s),
            "max_depth": max(1, max_depth),
            "deadline_s": max(0.01, deadline_s),
            "priority": max(0, priority),
        }
        self._topic_config[topic] = cfg
        metrics_payload = {
            "sent": 0,
            "dropped_tx": 0,
            "delivered": 0,
            "deadline_miss": 0,
            "expired": 0,
            "queue_depth": 0,
        }
        if self._manager is None:
            self.metrics[topic] = metrics_payload
            self._queues.setdefault(topic, [])
        else:
            self.metrics[topic] = self._manager.dict(metrics_payload)
            if topic not in self._queues:
                self._queues[topic] = self._manager.list()
        if topic not in self._locks:
            self._locks[topic] = mp.Lock()

    def _cfg(self, topic: str) -> Dict[str, float]:
        if topic not in self._topic_config:
            self.configure_topic(topic)
        return self._topic_config[topic]

    def send(
        self,
        topic: str,
        message: Any,
        now: Optional[float] = None,
        latency_s: float = 0.0,
        priority: Optional[int] = None,
    ) -> None:
        cfg = self._cfg(topic)
        if random.random() < cfg["drop_rate"]:
            self.metrics[topic]["dropped_tx"] = self.metrics[topic]["dropped_tx"] + 1
            return
        deliver_at = (now if now is not None else time.time()) + max(0.0, latency_s)
        if cfg["jitter_s"] > 0.0:
            deliver_at += random.uniform(-cfg["jitter_s"], cfg["jitter_s"])
        with self._locks[topic]:
            q = self._queues.setdefault(topic, [] if self._manager is None else self._manager.list())
            q.append((deliver_at, priority if priority is not None else cfg["priority"], len(q), message))
            if len(q) > cfg["max_depth"]:
                q.pop(0)
            self.metrics[topic]["sent"] = self.metrics[topic]["sent"] + 1
            self.metrics[topic]["queue_depth"] = len(q)

    def receive_latest(
        self, topic: str, now: Optional[float] = None, max_age_s: Optional[float] = None
    ) -> Optional[Any]:
        cfg = self._cfg(topic)
        max_age = cfg["max_age_s"] if max_age_s is None else max_age_s
        now_ts = now if now is not None else time.time()
        queue = self._queues.get(topic, [] if self._manager is None else self._manager.list())
        with self._locks[topic]:
            ready = [(ts, prio, idx, msg) for ts, prio, idx, msg in queue if ts <= now_ts]
            if not ready:
                return None
            remaining = [(ts, prio, idx, msg) for ts, prio, idx, msg in queue if ts > now_ts]
            if isinstance(queue, list):
                queue[:] = remaining
            else:
                queue[:] = remaining
        ready.sort(key=lambda t: (t[0], t[1], t[2]))
        _, _, _, latest = ready[-1]
        age = now_ts - latest.timestamp if hasattr(latest, "timestamp") else 0.0
        if age > max_age:
            self.metrics[topic]["expired"] = self.metrics[topic]["expired"] + 1
            return None
        deadline_s = cfg.get("deadline_s", max_age)
        if now_ts - ready[-1][0] > deadline_s:
            self.metrics[topic]["deadline_miss"] = self.metrics[topic]["deadline_miss"] + 1
        self.metrics[topic]["delivered"] = self.metrics[topic]["delivered"] + 1
        return latest

    def get_metrics_snapshot(self) -> Dict[str, Dict[str, float]]:
        """Return a shallow copy of per-topic bus metrics for reporting.

        This is designed for end-of-run summaries and light debugging, not
        for tight inner loops.
        """
        if self._manager is None:
            return {topic: dict(vals) for topic, vals in self.metrics.items()}
        # For manager-backed dicts, materialize plain dicts for stability.
        out: Dict[str, Dict[str, float]] = {}
        for topic, vals in self.metrics.items():
            try:
                out[topic] = {k: float(v) for k, v in vals.items()}
            except Exception:
                out[topic] = dict(vals)
        return out


@dataclass
class PerceptionJob:
    bgr: Any
    depth_m: Any
    depth_stereo_m: Any
    fx: float
    fy: float
    cx: float
    cy: float
    sim_time: float
    sensor_timestamp: Optional[float]
    v: float
    mu: float
    log_both: bool
    csv_writer: Any = None
    tele_bgr: Any = None
    tele_depth_m: Any = None


@dataclass
class PlanningJob:
    trigger_name: Optional[str]
    nearest_s_active: Optional[float]
    nearest_thr: Optional[float]
    tl_state: str
    tl_s_active: Optional[float]
    v: float
    v_target: float
    mu: float
    ema_loop_ms: float
    tracked_distance_for_control: float
    stop_armed: bool
    stop_latch_time: float
    stop_release_ignore_until: float
    red_green_since: float
    no_trigger_elapsed: float
    no_red_elapsed: float
    depth_m: Any
    depth_stereo_m: Any
    nearest_box: Any
    nearest_conf: Optional[float]
    I_err: float
    v_prev: float


@dataclass
class ActuationJob:
    brake_cmd: float
    v_ego: float
    wheel_speeds: Any
    a_long: float


class ECUProcessNode:
    """Small worker process wrapper to host an ECU in a separate process."""

    def __init__(
        self,
        name: str,
        handler: Callable[[Any], Any],
        request_queue: mp.Queue,
        response_queue: mp.Queue,
        stop_event: mp.Event,
    ):
        self.name = name
        self.handler = handler
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.stop_event = stop_event

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                payload = self.request_queue.get(timeout=0.05)
            except queue_mod.Empty:
                continue
            if payload is None:
                break
            try:
                result = self.handler(payload)
                self.response_queue.put(result)
            except Exception as exc:
                self.response_queue.put(exc)


class DistributedECUPipeline:
    """Spawns separate OS processes for perception, planning, and actuation."""

    def __init__(
        self,
        perception_handler: Callable[[PerceptionJob], PerceptionSignal],
        planning_handler: Callable[[PlanningJob], PlanningDecision],
        actuation_handler: Callable[[ActuationJob], ActuationResult],
    ):
        self.stop_event = mp.Event()
        self.perception_q: mp.Queue = mp.Queue(maxsize=2)
        self.perception_out: mp.Queue = mp.Queue(maxsize=2)
        self.planning_q: mp.Queue = mp.Queue(maxsize=2)
        self.planning_out: mp.Queue = mp.Queue(maxsize=2)
        self.actuation_q: mp.Queue = mp.Queue(maxsize=2)
        self.actuation_out: mp.Queue = mp.Queue(maxsize=2)

        self.perception_proc = mp.Process(
            target=ECUProcessNode(
                "perception",
                perception_handler,
                self.perception_q,
                self.perception_out,
                self.stop_event,
            ).run,
            daemon=True,
        )
        self.planning_proc = mp.Process(
            target=ECUProcessNode(
                "planning",
                planning_handler,
                self.planning_q,
                self.planning_out,
                self.stop_event,
            ).run,
            daemon=True,
        )
        self.actuation_proc = mp.Process(
            target=ECUProcessNode(
                "actuation",
                actuation_handler,
                self.actuation_q,
                self.actuation_out,
                self.stop_event,
            ).run,
            daemon=True,
        )

        self.perception_proc.start()
        self.planning_proc.start()
        self.actuation_proc.start()

    def shutdown(self) -> None:
        self.stop_event.set()
        for q in (self.perception_q, self.planning_q, self.actuation_q):
            try:
                q.put_nowait(None)
            except Exception:
                pass
        for proc in (self.perception_proc, self.planning_proc, self.actuation_proc):
            try:
                proc.join(timeout=1.0)
            except Exception:
                pass

    def _request(self, q_in: mp.Queue, q_out: mp.Queue, payload: Any, timeout: float) -> Any:
        q_in.put(payload)
        try:
            return q_out.get(timeout=timeout)
        except queue_mod.Empty:
            raise TimeoutError("ECU process did not respond in time")

    def run_perception(self, job: PerceptionJob, timeout: float = 0.5) -> PerceptionSignal:
        result = self._request(self.perception_q, self.perception_out, job, timeout)
        if isinstance(result, Exception):
            raise result
        return result

    def run_planning(self, job: PlanningJob, timeout: float = 0.5) -> PlanningDecision:
        result = self._request(self.planning_q, self.planning_out, job, timeout)
        if isinstance(result, Exception):
            raise result
        return result

    def run_actuation(self, job: ActuationJob, timeout: float = 0.5) -> ActuationResult:
        result = self._request(self.actuation_q, self.actuation_out, job, timeout)
        if isinstance(result, Exception):
            raise result
        return result


class SafetyMode(str, Enum):
    NOMINAL = "NOMINAL"
    DEGRADED = "DEGRADED"
    FAIL_SAFE = "FAIL_SAFE"


@dataclass
class SafetyDecision:
    throttle: float
    brake: float
    mode: SafetyMode
    faults: List[str] = field(default_factory=list)
    latched: List[str] = field(default_factory=list)


class SafetyManager:
    """Aggregates ECU faults and enforces degraded behavior."""

    def __init__(
        self,
        brake_fail_safe: float = 1.0,
        perception_freshness_s: float = 0.5,
        planning_freshness_s: float = 0.5,
        actuation_freshness_s: float = 0.5,
        ttc_floor_s: float = 0.2,
        v_min_plausible: float = 0.1,
        wheel_slip_max: float = 1.0,
    ):
        self.brake_fail_safe = _clamp(brake_fail_safe, (0.5, 1.0))
        self.perception_freshness_s = perception_freshness_s
        self.planning_freshness_s = planning_freshness_s
        self.actuation_freshness_s = actuation_freshness_s
        self.ttc_floor_s = ttc_floor_s
        self.v_min_plausible = v_min_plausible
        self.wheel_slip_max = wheel_slip_max
        self.latched_faults: List[str] = []
        # Lightweight counters for offline safety telemetry.
        self.metrics: Dict[str, float] = {
            "evaluations": 0.0,
            "failsafe_frames": 0.0,
            "degraded_frames": 0.0,
            "aeb_frames": 0.0,
            "aeb_full_frames": 0.0,
        }

    def _latch(self, code: Optional[str]) -> None:
        if code and code not in self.latched_faults:
            self.latched_faults.append(code)

    def evaluate(
        self,
        perception: Optional[PerceptionSignal],
        planning: Optional[PlanningDecision],
        actuation: Optional[ActuationResult],
        v_ego: float = 0.0,
        ttc: Optional[float] = None,
    ) -> SafetyDecision:
        now = time.time()
        faults: List[str] = []

        if perception is not None:
            if not perception.valid or perception.fault_code:
                self._latch(perception.fault_code or "PERCEPTION_INVALID")
            elif (now - perception.timestamp) > self.perception_freshness_s:
                self._latch("PERCEPTION_STALE")

        if planning is not None:
            if not planning.valid or planning.fault_code:
                self._latch(planning.fault_code or "PLANNING_INVALID")
            elif (now - planning.timestamp) > self.planning_freshness_s:
                self._latch("PLANNING_STALE")
            if ttc is None and planning.debug is not None:
                try:
                    ttc = float(planning.debug.get("ttc"))
                except Exception:
                    ttc = None
            if ttc is not None and math.isfinite(ttc) and ttc < self.ttc_floor_s and v_ego > self.v_min_plausible:
                self._latch("TTC_IMPLAUSIBLE")

        if actuation is not None:
            if not actuation.valid or actuation.fault_code:
                self._latch(actuation.fault_code or "ACTUATION_INVALID")
            elif (now - actuation.timestamp) > self.actuation_freshness_s:
                self._latch("ACT_STALE")
            if actuation.abs_dbg is not None:
                slip_val = actuation.abs_dbg.get("lambda_max")
                try:
                    if slip_val is not None and float(slip_val) > self.wheel_slip_max:
                        self._latch("SLIP_OUT_OF_RANGE")
                except Exception:
                    pass

        faults = list(self.latched_faults)
        critical = {"ACTUATION_INVALID", "ABS_FAIL", "PERCEPTION_INVALID", "PLANNING_INVALID", "SLIP_OUT_OF_RANGE"}

        mode: SafetyMode
        if any(code in critical for code in faults):
            throttle_cmd = 0.0
            brake_cmd = self.brake_fail_safe
            mode = SafetyMode.FAIL_SAFE
        elif faults:
            throttle_cmd = 0.0 if planning is None else min(planning.throttle, 0.25)
            brake_cmd = 0.0 if planning is None else max(planning.brake, 0.25)
            mode = SafetyMode.DEGRADED
        else:
            throttle_cmd = planning.throttle if planning is not None else 0.0
            brake_cmd = planning.brake if planning is not None else 0.0
            mode = SafetyMode.NOMINAL

        # Update simple safety metrics.
        self.metrics["evaluations"] = self.metrics.get("evaluations", 0.0) + 1.0
        if mode is SafetyMode.FAIL_SAFE:
            self.metrics["failsafe_frames"] = self.metrics.get("failsafe_frames", 0.0) + 1.0
        elif mode is SafetyMode.DEGRADED:
            self.metrics["degraded_frames"] = self.metrics.get("degraded_frames", 0.0) + 1.0

        return SafetyDecision(
            throttle=throttle_cmd,
            brake=brake_cmd,
            mode=mode,
            faults=faults,
            latched=list(self.latched_faults),
        )

