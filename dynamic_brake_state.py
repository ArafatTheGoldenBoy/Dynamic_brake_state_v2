# dynamic_brake_state.py
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pygame

from cli_parser import parse_args, apply_preset
from world_manager import WorldManager
from sensor_rig import SensorRig
from range_estimator import RangeEstimator
from ecu import (
    PerceptionECU,
    PlanningECU,
    ActuationECU,
    MessageBus,
    DistributedECUPipeline,
    SafetyManager,
    PerceptionJob,
    PlanningJob,
    ActuationJob,
)
from abs_system import AdaptivePISlipABSActuator, PISlipABSActuator
from telemetry import TelemetryLogger, ScenarioLogger
from perception_pipeline import PerceptionPipeline, PerceptionConfig
from planning import AEBPlanner, PlanningContext, AEBPlannerConfig
from hud import HUD
from vehicle_control import steer_to_waypoint, wheel_linear_speeds
from calibrations import (
    load_aeb_calibration,
    load_safety_calibration,
    load_bus_calibration,
)
from detectors import YOLODetector
from label_utils import (
    parse_per_class_iou_map,
)
from config import (
    DT,
    MU_DEFAULT,
    V_TARGET,
    DEPTH_ROI_SHRINK_DEFAULT,
    STEREO_ROI_SHRINK_DEFAULT,
    FALSE_STOP_MARGIN_M,
    FALSE_STOP_TTC_S,
)

from camera_utils import intrinsics_from_fov, decode_depth_meters_from_bgra
from config import IMG_W, IMG_H, FOV_X_DEG

from carla_utils import import_carla

carla = import_carla()


@dataclass
class AppState:
    """Small bag of state carried between steps to keep App readable."""

    sim_time: float = 0.0
    ema_loop_ms: float = DT * 1000.0
    v_prev: float = 0.0
    last_tracked_dist: Optional[float] = None

    last_loop_ms: float = DT * 1000.0
    last_fps: float = 1.0 / DT

@dataclass
class ScenarioState:
    """Aggregate per-run braking metrics for ScenarioLogger."""

    # Did we see any AEB braking?
    active: bool = False
    start_time: Optional[float] = None

    # Initial conditions when AEB first engages
    v_init_mps: Optional[float] = None
    s_init_m: Optional[float] = None
    s_min_m: float = float("inf")

    # Ground-truth distances from WorldManager.lead_distance_ahead()
    s_init_gt_m: Optional[float] = None
    s_min_gt_m: float = float("inf")

    # Stopping / collision outcome
    t_stop_s: Optional[float] = None
    stopped: bool = False
    collision: bool = False

    # Threat / timing
    trigger_kind: Optional[str] = None
    ttc_init_s: Optional[float] = None
    ttc_min_s: Optional[float] = None

    # Label whether this run looks like a false stop
    false_stop: bool = False

class App:
    """
    High‑level orchestrator: glue modules together.

    Heavy logic lives in:
      * PerceptionPipeline  → camera ECU SWC
      * AEBPlanner          → ADAS / AEB SWC
      * ABS actuators       → brake ECU SWC
      * ECU / Safety        → ecu.py (MessageBus, SafetyManager, etc.)
      * Telemetry / Scenario → telemetry.py
    """

    def __init__(self, args):
        self.args = apply_preset(args)

        # Normalized road friction used throughout
        mu_val = self.args.mu if getattr(self.args, "mu", None) is not None else MU_DEFAULT
                # === 8) Misc state ==================================================
        self.mu = max(0.05, min(1.2, self.args.mu or MU_DEFAULT))
        # Use ABS friction estimator feedback in the planner if enabled via CLI
        self.use_abs_mu_feedback = bool(getattr(self.args, "planner_use_abs_mu", False))
        self.state = AppState()
        self.clock = None  # pygame clock if HUD uses it


        # === 1) World + sensors ==============================================
        self.world = WorldManager(
            host=self.args.host,
            port=self.args.port,
            town=self.args.town,
            mu=self.mu,
            apply_tire_friction=self.args.apply_tire_friction,
            npc_vehicles=self.args.npc_vehicles,
            npc_walkers=self.args.npc_walkers,
            npc_seed=self.args.npc_seed,
            npc_autopilot=(not self.args.npc_disable_autopilot),
            npc_speed_diff_pct=self.args.npc_speed_diff_pct,
        )

        # Respect depth/telephoto toggles for sensor rig
        range_est_mode = self.args.range_est
        if range_est_mode in ("depth", "both") and self.args.no_depth_cam:
            range_est_mode = "pinhole"

        self.sensors = SensorRig(
            world=self.world.world,
            vehicle=self.world.ego,
            range_est=range_est_mode,
            enable_depth=not self.args.no_depth_cam,
            enable_telephoto=not self.args.no_telephoto,
        )

        self.range = RangeEstimator(
            use_cuda=self.args.stereo_cuda,
            method=self.args.stereo_method,
        )
        if range_est_mode == "stereo":
            self.range.ensure_stereo()

        self.fx, self.fy, self.cx, self.cy = intrinsics_from_fov(
            IMG_W, IMG_H, FOV_X_DEG
        )

        # === 2) Perception pipeline (+ ECU wrapper) ==========================
        per_class_iou_map = parse_per_class_iou_map(self.args.yolo_class_iou)

        self.detector = YOLODetector(
            img_size=self.args.yolo_img,
            device=self.args.yolo_device,
            use_half=self.args.yolo_half,
            agnostic=self.args.yolo_agnostic,
            classes=self.args.yolo_classes,
            max_det=self.args.yolo_max_det,
            dnn=self.args.yolo_dnn,
            augment=self.args.yolo_augment,
            per_class_iou_map=per_class_iou_map,
        )

        perc_cfg = PerceptionConfig(
            yolo_img=self.args.yolo_img,
            yolo_device=self.args.yolo_device,
            yolo_half=self.args.yolo_half,
            yolo_agnostic=self.args.yolo_agnostic,
            yolo_classes=self.args.yolo_classes,
            yolo_class_thr=self.args.yolo_class_thr,
            yolo_class_iou=self.args.yolo_class_iou,
            yolo_max_det=self.args.yolo_max_det,
            yolo_dnn=self.args.yolo_dnn,
            yolo_augment=self.args.yolo_augment,
            range_est=range_est_mode,
            depth_roi_shrink=getattr(
                self.args, "depth_roi_shrink", DEPTH_ROI_SHRINK_DEFAULT
            ),
            stereo_roi_shrink=getattr(
                self.args, "stereo_roi_shrink", STEREO_ROI_SHRINK_DEFAULT
            ),
            telephoto_enabled=not self.args.no_telephoto,
            telephoto_stride=self.args.telephoto_stride,
            telephoto_zoom=self.args.telephoto_zoom,
            tl_unknown_conservative=self.args.tl_unknown_conservative,
            engage_override=self.args.engage_override,
            min_h_override=self.args.min_h_override,
            gate_frac_override=self.args.gate_frac_override,
            gate_lateral_override=self.args.gate_lateral_override,
        )

        self.perception_pipeline = PerceptionPipeline(
            detector=self.detector,
            range_estimator=self.range,
            config=perc_cfg,
        )
        self.perception_ecu = PerceptionECU(
            perception_fn=self.perception_pipeline.run
        )

        # === 3) Planning / AEB (+ ECU wrapper) ==============================
        cal_defaults = {
            "min_aeb_speed": self.args.min_aeb_speed,
            "gate_confirm_frames": self.args.gate_confirm_frames,
            "ttc_confirm_s": self.args.ttc_confirm_s,
            "ttc_stage_strong": self.args.ttc_stage_strong,
            "ttc_stage_full": self.args.ttc_stage_full,
            "stage_factor_comfort": self.args.aeb_stage_comfort,
            "stage_factor_strong": self.args.aeb_stage_strong,
            "aeb_ramp_up": self.args.aeb_ramp_up,
            "aeb_ramp_down": self.args.aeb_ramp_down,
        }
        self.aeb_cal = load_aeb_calibration(
            getattr(self.args, "calibration_file", None), cal_defaults
        )

        self.plan_ctx = PlanningContext()
        self.aeb_cfg = AEBPlannerConfig(
            calibration=self.aeb_cal,
            mu=self.mu,
            extra_latency_ms=self.args.extra_latency_ms,
            range_mode=range_est_mode,
            depth_roi_shrink=getattr(
                self.args, "depth_roi_shrink", DEPTH_ROI_SHRINK_DEFAULT
            ),
            stereo_roi_shrink=getattr(
                self.args, "stereo_roi_shrink", STEREO_ROI_SHRINK_DEFAULT
            ),
        )

        self.planner = AEBPlanner(self.aeb_cfg)
        self.planning_ecu = PlanningECU(
            control_fn=self.planner.step,
            context=self.plan_ctx,
        )

        # === 4) ABS actuator / Actuation ECU ===============================
        if self.args.abs_mode == "off":
            abs_actuator = None
        elif self.args.abs_mode == "fixed":
            abs_actuator = PISlipABSActuator(dt=DT)
        else:
            abs_actuator = AdaptivePISlipABSActuator(dt=DT)

        self.actuation_ecu = ActuationECU(
            abs_fn=None if abs_actuator is None else abs_actuator.step,
            abs_debug_fn=None if abs_actuator is None else abs_actuator.debug_metrics,
        )

        # === 5) Message bus + safety manager ===============================
        self.bus = MessageBus()
        bus_defaults = {
            "perception": {
                "drop_rate": self.args.bus_drop_perception,
                "jitter_s": self.args.bus_jitter_perception,
                "max_age_s": 0.35,
                "max_depth": 8,
                "deadline_s": self.args.bus_deadline_perception,
                "priority": 1,
            },
            "planning": {
                "drop_rate": self.args.bus_drop_planning,
                "jitter_s": self.args.bus_jitter_planning,
                "max_age_s": 0.35,
                "max_depth": 8,
                "deadline_s": self.args.bus_deadline_planning,
                "priority": 0,
            },
        }
        bus_cal = load_bus_calibration(
            getattr(self.args, "bus_calibration_file", None), bus_defaults
        )
        for topic, cfg in bus_cal.items():
            self.bus.configure_topic(
                topic,
                drop_rate=cfg.drop_rate,
                jitter_s=cfg.jitter_s,
                max_age_s=cfg.max_age_s,
                max_depth=cfg.max_depth,
                deadline_s=cfg.deadline_s,
                priority=cfg.priority,
            )

        safety_defaults = {
            "perception_freshness_s": self.args.perception_freshness_s,
            "planning_freshness_s": self.args.planning_freshness_s,
            "actuation_freshness_s": self.args.actuation_freshness_s,
            "ttc_floor_s": self.args.safety_ttc_floor_s,
            "v_min_plausible": self.args.safety_v_min_plausible,
            "wheel_slip_max": self.args.safety_wheel_slip_max,
            "brake_fail_safe": self.args.brake_fail_safe,
        }
        safety_cal = load_safety_calibration(
            getattr(self.args, "safety_calibration_file", None), safety_defaults
        )
        self.safety_manager = SafetyManager(
            brake_fail_safe=safety_cal.brake_fail_safe,
            perception_freshness_s=safety_cal.perception_freshness_s,
            planning_freshness_s=safety_cal.planning_freshness_s,
            actuation_freshness_s=safety_cal.actuation_freshness_s,
            ttc_floor_s=safety_cal.ttc_floor_s,
            v_min_plausible=safety_cal.v_min_plausible,
            wheel_slip_max=safety_cal.wheel_slip_max,
        )

        # === 6) Optional distributed ECU pipeline ==========================
        self.multiprocess = self.args.multiprocess_ecus
        self.ecu_timeout = self.args.ecu_process_timeout
        self.ecu_pipeline: Optional[DistributedECUPipeline] = None
        if self.multiprocess:
            self.ecu_pipeline = DistributedECUPipeline(
                perception_handler=lambda job: self.perception_ecu.process(**job.__dict__),
                planning_handler=lambda job: self.planning_ecu.plan(**job.__dict__),
                actuation_handler=lambda job: self.actuation_ecu.apply_abs(
                    job.brake_cmd, job.v_ego, job.wheel_speeds, job.a_long
                ),
            )

        # === 7) HUD + telemetry / scenarios ================================
        self.hud: Optional[HUD]
        if self.args.headless:
            self.hud = None
        else:
            pygame.init()
            win_w = IMG_W * (2 if not self.args.no_telephoto else 1)
            win_h = IMG_H
            screen = pygame.display.set_mode((win_w, win_h))
            pygame.display.set_caption("Dynamic Brake State AEB Demo")
            self.hud = HUD(
                screen=screen,
                telephoto_enabled=not self.args.no_telephoto,
                headless=False,
            )

        self.telemetry = (
            TelemetryLogger(self.args.telemetry_csv, self.args.telemetry_hz)
            if self.args.telemetry_csv
            else None
        )
        self.scenario_logger = (
            ScenarioLogger(self.args.scenario_csv)
            if self.args.scenario_csv
            else None
        )

        # === 8) Misc state ==================================================
        self.state = AppState()
        self.scenario_state = ScenarioState()
    # ------------------------------------------------------------------ utils

    def _decode_frames(self, frames: Dict[str, Any]) -> Dict[str, Any]:
        """Convert CARLA image objects → numpy arrays for perception & HUD."""
        out: Dict[str, Any] = {}

        front = frames.get("front")
        if front is None:
            raise RuntimeError("Front RGB frame missing")

        front_bgra = np.frombuffer(front.raw_data, dtype=np.uint8)
        front_bgra = front_bgra.reshape((front.height, front.width, 4))
        out["front_bgr"] = front_bgra[:, :, :3].copy()

        depth_img = frames.get("depth")
        if depth_img is not None:
            depth_bgra = np.frombuffer(depth_img.raw_data, dtype=np.uint8)
            depth_bgra = depth_bgra.reshape((depth_img.height, depth_img.width, 4))
            out["depth_m"] = decode_depth_meters_from_bgra(depth_bgra)
        else:
            out["depth_m"] = None

        stereo_depth_m = None
        if self.args.range_est in ("stereo", "both"):
            left = frames.get("stereo_left")
            right = frames.get("stereo_right")
            if left is not None and right is not None:
                left_bgra = np.frombuffer(left.raw_data, dtype=np.uint8).reshape(
                    (left.height, left.width, 4)
                )
                right_bgra = np.frombuffer(right.raw_data, dtype=np.uint8).reshape(
                    (right.height, right.width, 4)
                )
                stereo_depth_m = self.range.stereo_depth(left_bgra, right_bgra)
        out["stereo_depth_m"] = stereo_depth_m

        tele_rgb = frames.get("tele_rgb")
        if tele_rgb is not None:
            tele_bgra = np.frombuffer(tele_rgb.raw_data, dtype=np.uint8).reshape(
                (tele_rgb.height, tele_rgb.width, 4)
            )
            out["tele_bgr"] = tele_bgra[:, :, :3].copy()
        else:
            out["tele_bgr"] = None

        tele_depth_img = frames.get("tele_depth")
        if tele_depth_img is not None:
            tele_depth_bgra = np.frombuffer(tele_depth_img.raw_data, dtype=np.uint8)
            tele_depth_bgra = tele_depth_bgra.reshape(
                (tele_depth_img.height, tele_depth_img.width, 4)
            )
            out["tele_depth_m"] = decode_depth_meters_from_bgra(tele_depth_bgra)
        else:
            out["tele_depth_m"] = None

        return out

    # ----------------------------------------------------------------- main step
    def _update_scenario_state(
        self,
        sim_time: float,
        v: float,
        brake_cmd: float,
        perc,
        dbg: Dict[str, Any],
    ) -> None:
        """
        Per-step hook to:
        - flag likely false stops
        - accumulate per-run scenario metrics for ScenarioLogger
        """

        ss = self.scenario_state

        # Current estimated and ground-truth distance to lead
        s_est = perc.nearest_s_active
        try:
            s_gt = self.world.lead_distance_ahead()
        except Exception:
            s_gt = None

        ttc = dbg.get("ttc")
        D_safe = dbg.get("D_safety_dyn")

        # --- Frame-level false-stop heuristic -------------------------------
        false_stop_frame = False
        gap = s_gt if s_gt is not None else s_est
        if (
            brake_cmd > 0.05
            and ttc is not None
            and D_safe is not None
            and gap is not None
        ):
            # If we are braking hard while TTC is large AND the true gap
            # is significantly above the dynamic safety distance, suspect a false stop.
            if (ttc >= FALSE_STOP_TTC_S) and (gap >= D_safe + FALSE_STOP_MARGIN_M):
                false_stop_frame = True

        # Expose this in the planner debug dict so TelemetryLogger can store it.
        dbg["false_stop_flag"] = false_stop_frame

        # If user did not request scenario CSV, don't bother aggregating episodes.
        if self.scenario_logger is None:
            return

        # --- Scenario start (first time AEB brake actually engages) ---------
        if (not ss.active) and brake_cmd > 0.05:
            ss.active = True
            ss.start_time = sim_time
            ss.v_init_mps = float(v)

            gap0 = s_est if s_est is not None else s_gt
            if gap0 is None:
                gap0 = 0.0
            ss.s_init_m = float(gap0)
            ss.s_min_m = float(gap0)

            if s_gt is not None:
                ss.s_init_gt_m = float(s_gt)
                ss.s_min_gt_m = float(s_gt)

            ss.trigger_kind = perc.nearest_kind or dbg.get("brake_reason") or "unknown"

            if ttc is not None:
                ss.ttc_init_s = float(ttc)
                ss.ttc_min_s = float(ttc)

        # If scenario never started yet, still track collisions and return
        if not ss.active:
            if self.world.collision_happened:
                ss.collision = True
            return

        # --- Scenario is active: update minima / flags ----------------------
        if s_est is not None:
            ss.s_min_m = min(ss.s_min_m, float(s_est))
        if s_gt is not None:
            ss.s_min_gt_m = min(ss.s_min_gt_m, float(s_gt))

        if ttc is not None:
            ss.ttc_min_s = float(ttc) if ss.ttc_min_s is None else min(ss.ttc_min_s, float(ttc))

        if brake_cmd > 0.05 and v < 0.3 and (not ss.stopped):
            ss.stopped = True
            if ss.start_time is not None:
                ss.t_stop_s = sim_time - ss.start_time

        if self.world.collision_happened:
            ss.collision = True

        if false_stop_frame:
            ss.false_stop = True

    def step_once(self):
        # Basic pygame event handling (for closing the window)
        if self.hud is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

        # === 1) tick world & read sensors ===================================
        frame_id = self.world.tick()
        self.state.sim_time += DT
        sim_time = self.state.sim_time

        frames = self.sensors.read(expected_frame=frame_id)
        if frames is None:
            # Fallback: safe-ish braking if sensors failed
            self.world.ego.apply_control(
                carla.VehicleControl(throttle=0.0, brake=0.3)
            )
            return

        decoded = self._decode_frames(frames)

        # ego kinematics
        vel = self.world.ego.get_velocity()
        v = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        a_long = (v - self.state.v_prev) / DT if self.state.v_prev is not None else 0.0
        self.state.v_prev = v
        wheel_speeds = wheel_linear_speeds(self.world)

        # === 2) Perception ECU =============================================
        sensor_ts = getattr(frames["front"], "timestamp", None)
        perc_job = PerceptionJob(
            bgr=decoded["front_bgr"],
            depth_m=decoded.get("depth_m"),
            depth_stereo_m=decoded.get("stereo_depth_m"),
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            sim_time=sim_time,
            sensor_timestamp=sensor_ts,
            v=v,
            mu=self.mu,
            log_both=bool(self.args.compare_csv),
            csv_writer=None,
            tele_bgr=decoded.get("tele_bgr"),
            tele_depth_m=decoded.get("tele_depth_m"),
        )

        if self.ecu_pipeline:
            perc = self.ecu_pipeline.run_perception(perc_job, timeout=self.ecu_timeout)
        else:
            perc = self.perception_ecu.process(**perc_job.__dict__)

        perc.validate(freshness_s=self.safety_manager.perception_freshness_s)

        now_rt = time.time()
        self.bus.send("perception", perc, now=now_rt, latency_s=self.args.bus_latency_perception)
        perc_bus = self.bus.receive_latest("perception", now=now_rt)
        if perc_bus is not None:
            perc = perc_bus

        # === 3) Planning ECU ===============================================
        tracked_dist = perc.nearest_s_active
        self.state.last_tracked_dist = tracked_dist

        plan_kwargs = dict(
            sim_time=sim_time,
            trigger_name=perc.nearest_kind,
            nearest_s_active=perc.nearest_s_active,
            nearest_thr=perc.nearest_thr,
            tl_state=perc.tl_state,
            tl_s_active=perc.tl_s_active,
            v=v,
            v_target=V_TARGET,
            ema_loop_ms=self.state.ema_loop_ms,
            tracked_distance_for_control=tracked_dist,
            depth_m=decoded.get("depth_m"),
            depth_stereo_m=decoded.get("stereo_depth_m"),
            nearest_box=perc.nearest_box,
            nearest_conf=perc.nearest_conf,
        )

        plan_job = PlanningJob(
            trigger_name=perc.nearest_kind,
            nearest_s_active=perc.nearest_s_active,
            nearest_thr=perc.nearest_thr,
            tl_state=perc.tl_state,
            tl_s_active=perc.tl_s_active,
            v=v,
            v_target=V_TARGET,
            mu=self.mu,
            ema_loop_ms=self.state.ema_loop_ms,
            tracked_distance_for_control=tracked_dist if tracked_dist is not None else 0.0,
            stop_armed=self.plan_ctx.stop_armed,
            stop_latch_time=self.plan_ctx.stop_latch_time,
            stop_release_ignore_until=self.plan_ctx.stop_release_ignore_until,
            red_green_since=self.plan_ctx.red_green_since,
            no_trigger_elapsed=self.plan_ctx.no_trigger_elapsed,
            no_red_elapsed=self.plan_ctx.no_red_elapsed,
            depth_m=decoded.get("depth_m"),
            depth_stereo_m=decoded.get("stereo_depth_m"),
            nearest_box=perc.nearest_box,
            nearest_conf=perc.nearest_conf,
            I_err=self.plan_ctx.I_err,
            v_prev=self.plan_ctx.v_prev,
        )

        if self.ecu_pipeline:
            planning = self.ecu_pipeline.run_planning(plan_job, timeout=self.ecu_timeout)
        else:
            planning = self.planning_ecu.plan(**plan_kwargs)

        planning.validate(freshness_s=self.safety_manager.planning_freshness_s)

        now_rt2 = time.time()
        self.bus.send("planning", planning, now=now_rt2, latency_s=self.args.bus_latency_planning)
        planning_bus = self.bus.receive_latest("planning", now=now_rt2)
        if planning_bus is not None:
            planning = planning_bus

        throttle_cmd = planning.throttle
        brake_cmd = planning.brake

        # === 4) Actuation ECU (ABS) + safety manager =======================
        act_job = ActuationJob(
            brake_cmd=brake_cmd, v_ego=v, wheel_speeds=wheel_speeds, a_long=a_long
        )
        if self.ecu_pipeline:
            act_result = self.ecu_pipeline.run_actuation(act_job, timeout=self.ecu_timeout)
        else:
            act_result = self.actuation_ecu.apply_abs(
                act_job.brake_cmd, act_job.v_ego, act_job.wheel_speeds, act_job.a_long
            )
        act_result.validate(freshness_s=self.safety_manager.actuation_freshness_s)
        mu_est = None
        if act_result.abs_dbg is not None:
            mu_est = act_result.abs_dbg.get("mu_est", None)

        # Optional flag so you can turn this on/off for experiments
        if getattr(self, "use_abs_mu_feedback", True) and mu_est is not None:
            try:
                self.planner.update_mu_from_abs(mu_est)
            except Exception:
                # fail-safe: ignore bad μ measurements, keep last value
                pass

        safety = self.safety_manager.evaluate(
            perception=perc,
            planning=planning,
            actuation=act_result,
            v_ego=v,
            ttc=planning.debug.get("ttc") if planning.debug else None,
        )
        throttle_cmd = safety.throttle
        brake_cmd = safety.brake
        # --- NEW: update scenario + false-stop flag -------------------------
        dbg = planning.debug or {}
        self._update_scenario_state(sim_time, v, brake_cmd, perc, dbg)
        # === 5) Apply controls =============================================
        steer_cmd = steer_to_waypoint(self.world, v)

        self.world.ego.apply_control(
            carla.VehicleControl(
                throttle=float(throttle_cmd),
                brake=float(brake_cmd),
                steer=float(steer_cmd),
                hand_brake=False,
            )
        )

        # === 6) HUD + telemetry + scenario ================================
        if self.hud is not None:
            tr = self.world.ego.get_transform()
            loc = tr.location
            yaw_deg = tr.rotation.yaw
            dbg = planning.debug or {}
            abs_dbg = act_result.abs_dbg or {}

            perf_ms = self.state.last_loop_ms
            perf_fps = self.state.last_fps

            self.hud.draw(
                bgr=decoded["front_bgr"],
                perf_fps=perf_fps,
                perf_ms=perf_ms,
                x=loc.x,
                y=loc.y,
                z=loc.z,
                yaw_deg=yaw_deg,
                frame_id=frame_id,
                v_mps=v,
                trigger_name=perc.nearest_kind,
                tl_state=perc.tl_state,
                throttle=throttle_cmd,
                brake=brake_cmd,
                hold_blocked=getattr(self.plan_ctx, "hold_blocked", False),
                hold_reason=getattr(self.plan_ctx, "hold_reason", None),
                no_trigger_elapsed=getattr(self.plan_ctx, "no_trigger_elapsed", 0.0),
                no_red_elapsed=getattr(self.plan_ctx, "no_red_elapsed", 0.0),
                stop_armed=getattr(self.plan_ctx, "stop_armed", False),
                stop_release_ignore_until=getattr(
                    self.plan_ctx, "stop_release_ignore_until", -1.0
                ),
                sim_time=sim_time,
                dbg_tau_dyn=dbg.get("tau_dyn"),
                dbg_D_safety_dyn=dbg.get("D_safety_dyn"),
                dbg_sigma_depth=dbg.get("sigma_depth"),
                dbg_gate_hit=dbg.get("gate_hit"),
                dbg_a_des=dbg.get("a_des"),
                dbg_brake=dbg.get("brake"),
                v_target=V_TARGET,
                collision_flag=bool(self.world.collision_happened),
                det_points=list(perc.det_points) if perc.det_points else None,
                range_mode_label=self.args.range_est,
                abs_lambda=abs_dbg.get("lambda_max"),
                abs_factor=abs_dbg.get("f_global"),
                abs_mu=abs_dbg.get("mu_est"),
                abs_regime=abs_dbg.get("regime"),
                tele_bgr=decoded.get("tele_bgr"),
            )

        if self.telemetry is not None:
            dbg = planning.debug or {}
            self.telemetry.maybe_log(
                t=sim_time,
                v_mps=v,
                tau_dyn=dbg.get("tau_dyn"),
                D_safety_dyn=dbg.get("D_safety_dyn"),
                sigma_depth=dbg.get("sigma_depth"),
                a_des=dbg.get("a_des"),
                brake=brake_cmd,
                lambda_max=(act_result.abs_dbg or {}).get("lambda_max")
                if act_result.abs_dbg
                else None,
                abs_factor=(act_result.abs_dbg or {}).get("f_global")
                if act_result.abs_dbg
                else None,
                mu_est=(act_result.abs_dbg or {}).get("mu_est")
                if act_result.abs_dbg
                else None,
                mu_regime=(act_result.abs_dbg or {}).get("regime")
                if act_result.abs_dbg
                else None,
                loop_ms=self.state.ema_loop_ms,
                loop_ms_max=None,
                detect_ms=None,
                latency_ms=None,
                a_meas=a_long,
                x_rel_m=None,
                range_est_m=perc.nearest_s_active,
                ttc_s=dbg.get("ttc"),
                gate_hit=dbg.get("gate_hit"),
                gate_confirmed=dbg.get("gate_confirmed"),
                false_stop_flag=dbg.get("false_stop_flag"),
                brake_stage=dbg.get("brake_stage"),
                brake_stage_factor=dbg.get("brake_stage_factor"),
                tracker_s_m=None,
                tracker_rate_mps=None,
                lead_track_id=None,
                active_track_count=None,
                sensor_ts=sensor_ts,
                control_ts=sim_time,
                sensor_to_control_ms=None,
                actuation_ts=None,
                control_to_act_ms=None,
                sensor_to_act_ms=None,
            )

    # ----------------------------------------------------------------- run/teardown

    def run(self):
        try:
            while True:
                t0 = time.time()
                self.step_once()
                loop_ms = (time.time() - t0) * 1000.0
                self.state.ema_loop_ms = 0.9 * self.state.ema_loop_ms + 0.1 * loop_ms
                self.state.last_loop_ms = loop_ms
                self.state.last_fps = 1000.0 / max(1e-3, loop_ms)
        finally:
            self.shutdown()
    
    def _flush_scenario(self) -> None:
        """Emit a single ScenarioLogger row for this run, if active."""
        if self.scenario_logger is None:
            return

        ss = self.scenario_state
        if not ss.active or ss.start_time is None or ss.v_init_mps is None:
            # No AEB braking in this run → nothing to log
            return

        # Robust defaults for missing minima
        s_min_m = ss.s_min_m if math.isfinite(ss.s_min_m) else (ss.s_init_m or 0.0)
        s_init_m = ss.s_init_m if ss.s_init_m is not None else s_min_m

        s_min_gt_m = ss.s_min_gt_m if math.isfinite(ss.s_min_gt_m) else ss.s_init_gt_m
        s_init_gt_m = ss.s_init_gt_m

        # If the car never came to V_STOP, treat t_to_stop as time until end
        if ss.t_stop_s is not None:
            t_stop = ss.t_stop_s
        else:
            t_stop = self.state.sim_time - ss.start_time

        scenario_name = getattr(self.args, "scenario_tag", "default")

        self.scenario_logger.log(
            scenario=scenario_name,
            trigger_kind=ss.trigger_kind or "",
            mu=self.mu,
            v_init=ss.v_init_mps,
            s_init=s_init_m,
            s_min=s_min_m,
            s_init_gt=s_init_gt_m,
            s_min_gt=s_min_gt_m,
            stopped=ss.stopped,
            t_to_stop=t_stop,
            collision=ss.collision,
            range_margin=None,
            tts_margin=None,
            ttc_init=ss.ttc_init_s,
            ttc_min=ss.ttc_min_s,
            reaction_time=None,
            max_lambda=None,
            mean_abs_factor=None,
            false_stop=ss.false_stop,
        )


    def shutdown(self):
        if self.telemetry is not None:
            self.telemetry.close()
        if self.scenario_logger is not None:
            self._flush_scenario()
            self.scenario_logger.close()
        if self.sensors is not None:
            self.sensors.destroy()
        if self.world is not None:
            self.world.destroy()
        if self.ecu_pipeline is not None:
            self.ecu_pipeline.shutdown()


if __name__ == "__main__":
    args = apply_preset(parse_args())
    App(args).run()
