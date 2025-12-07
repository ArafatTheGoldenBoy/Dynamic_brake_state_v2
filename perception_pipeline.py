# perception_pipeline.py
"""
Camera / perception ECU pipeline.

This module wraps the YOLO detector + range estimation + simple gating into a
single object that can be passed into :class:`ecu.PerceptionECU`.

Typical use:

    from detectors import YOLODetector
    from range_estimator import RangeEstimator
    from perception_pipeline import PerceptionPipeline, PerceptionConfig
    from ecu import PerceptionECU

    det = YOLODetector(...)
    rng = RangeEstimator(...)
    cfg = PerceptionConfig(range_est="depth")
    pipe = PerceptionPipeline(detector=det, range_estimator=rng, config=cfg)

    perception_ecu = PerceptionECU(pipe.run)

The `run` method intentionally mirrors the original `_perception_step` helper
in `dynamic_brake_state.App`, so it can be dropped in behind `PerceptionECU`
without changing the ECU wrapper. 
"""

from __future__ import annotations

import math
from collections import deque, Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from config import (  # type: ignore
    IMG_W,
    IMG_H,
    TELEPHOTO_IMG_W,
    TELEPHOTO_IMG_H,
    TELEPHOTO_FOV_X_DEG,
    TELEPHOTO_ZOOM_TOP_BIAS,
    TELEPHOTO_CACHE_MAX_AGE_S,
    TL_PRIMARY_CROP_FRAC,
    TL_PRIMARY_SHORT_RANGE_M,
    OBJ_HEIGHT_M,
    CENTER_BAND_FRAC,
    VEHICLE_CLASSES,
    PEDESTRIAN_CLASSES,
    LATERAL_MAX,
    H_MIN_PX,
    DEPTH_ROI_SHRINK_DEFAULT,
    STEREO_ROI_SHRINK_DEFAULT,
    S_ENGAGE,
    S_ENGAGE_PED,
    S_ENGAGE_TL,
    STEREO_FUSE_NEAR_WEIGHT,
    STEREO_FUSE_FAR_WEIGHT,
    STEREO_FUSE_DISAGREE_M,
)
from detectors import BaseDetector  # type: ignore
from range_estimator import RangeEstimator  # type: ignore
from camera_utils import intrinsics_from_fov, pixel_to_camera  # type: ignore
from label_utils import (  # type: ignore
    TRIGGER_NAMES_NORM,
    _norm_label,
    parse_per_class_conf_map,
    parse_engage_override_map,
    parse_min_h_override_map,
    parse_gate_frac_override_map,
    parse_gate_lateral_override_map,
)


@dataclass
class PerceptionConfig:
    """
    Tunable knobs for the perception pipeline.

    All dictionaries are keyed by *normalized* class name
    (see :func:`label_utils._norm_label`).
    """
    # Direct mirrors of CLI flags so dynamic_brake_state can pass everything by name.
    yolo_img: int = 640
    yolo_device: str = "cuda"
    yolo_half: bool = False
    yolo_agnostic: bool = False
    yolo_classes: Optional[str] = None
    yolo_class_thr: Optional[str] = None
    yolo_class_iou: Optional[str] = None
    yolo_max_det: int = 200
    yolo_dnn: bool = False
    yolo_augment: bool = False

    # Gating / geometric overrides
    min_h_override: Optional[str] = None
    gate_frac_override: Optional[str] = None
    gate_lateral_override: Optional[str] = None
    engage_override: Optional[str] = None

    # Range / depth options
    range_est: str = "pinhole"          # 'pinhole' | 'depth' | 'stereo' | 'both'
    depth_roi_shrink: float = DEPTH_ROI_SHRINK_DEFAULT
    stereo_roi_shrink: float = STEREO_ROI_SHRINK_DEFAULT

    # Telephoto helper camera
    telephoto_enabled: bool = True
    telephoto_stride: int = 3
    telephoto_zoom: float = 1.5

    tl_unknown_conservative: bool = False

    min_h_px_default: int = H_MIN_PX
    min_h_px: Dict[str, int] = field(default_factory=dict)

    gate_center_frac: Dict[str, float] = field(default_factory=dict)
    gate_lateral_max: Dict[str, float] = field(default_factory=dict)
    engage_distance: Dict[str, float] = field(default_factory=dict)

    class_conf_map: Dict[str, float] = field(default_factory=dict)

    tl_smoothing_frames: int = 5

    def __post_init__(self) -> None:
        self.telephoto_stride = max(1, int(self.telephoto_stride))
        self.telephoto_zoom = max(1.0, float(self.telephoto_zoom))
        self.depth_roi_shrink = float(min(0.9, max(0.0, self.depth_roi_shrink)))
        self.stereo_roi_shrink = float(min(0.9, max(0.0, self.stereo_roi_shrink)))
        self.tl_smoothing_frames = max(1, int(self.tl_smoothing_frames))
        self.range_est = (self.range_est or "pinhole").lower()

        # Parse per‑class confidence thresholds.
        if self.yolo_class_thr:
            self.class_conf_map.update(
                parse_per_class_conf_map(self.yolo_class_thr)
            )

        # Parse per‑class min box heights (px).
        if self.min_h_override:
            self.min_h_px.update(parse_min_h_override_map(self.min_h_override))

        # Parse per‑class gating overrides (center band fraction and lateral max).
        if self.gate_frac_override:
            self.gate_center_frac.update(
                parse_gate_frac_override_map(self.gate_frac_override)
            )
        if self.gate_lateral_override:
            self.gate_lateral_max.update(
                parse_gate_lateral_override_map(self.gate_lateral_override)
            )

        # Engage‑distance overrides in meters (keys are normalized class names).
        if self.engage_override:
            for k, v in parse_engage_override_map(self.engage_override).items():
                self.engage_distance[k] = float(v)

        # Reasonable engage‑distance defaults (only for keys not overridden).
        for norm in ("car", "truck", "bus", "train", "motorcycle", "motorbike", "bicycle"):
            self.engage_distance.setdefault(norm, S_ENGAGE)
        self.engage_distance.setdefault("person", S_ENGAGE_PED)
        self.engage_distance.setdefault("stopsign", S_ENGAGE)
        self.engage_distance.setdefault("trafficlight", S_ENGAGE_TL)



def _median_depth_in_box(depth_m: Optional[np.ndarray],
                         box: Tuple[int, int, int, int],
                         shrink: float) -> Optional[float]:
    """Robust median depth in a shrunken ROI, or None if invalid."""
    if depth_m is None:
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
    if flat.size < 5:
        return None
    d = float(np.median(flat))
    if not np.isfinite(d) or d <= 0.0 or d > 999.0:
        return None
    return d


def _fuse_depth_sources(
    s_depth: Optional[float],
    s_stereo: Optional[float],
    box_h_px: int,
) -> Tuple[Optional[float], str]:
    """Blend depth and stereo with a simple heuristic."""
    if s_depth is None and s_stereo is None:
        return None, "none"
    if s_depth is None:
        return s_stereo, "stereo"
    if s_stereo is None:
        return s_depth, "depth"

    prox = max(0.0, min(1.0, float(box_h_px) / max(1.0, float(IMG_H))))
    w_depth = STEREO_FUSE_FAR_WEIGHT + (STEREO_FUSE_NEAR_WEIGHT - STEREO_FUSE_FAR_WEIGHT) * prox
    fused = (w_depth * s_depth) + ((1.0 - w_depth) * s_stereo)
    if abs(s_depth - s_stereo) > STEREO_FUSE_DISAGREE_M:
        # If they violently disagree, prefer the nearer obstacle.
        return min(s_depth, s_stereo), "min"
    return fused, "fused"


def _estimate_tl_color_from_roi(bgr: np.ndarray,
                                box: Tuple[int, int, int, int]) -> str:
    """
    Very simple HSV‑based traffic light color classifier.

    Returns 'RED', 'YELLOW', 'GREEN' or 'UNKNOWN'.
    """
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return "UNKNOWN"
    h_lim, w_lim = bgr.shape[0], bgr.shape[1]
    roi = bgr[max(0, y):min(h_lim, y + h),
              max(0, x):min(w_lim, x + w)]
    if roi.size == 0:
        return "UNKNOWN"

    roi_h = roi.shape[0]
    thirds = [
        (0, int(roi_h / 3)),
        (int(roi_h / 3), int(2 * roi_h / 3)),
        (int(2 * roi_h / 3), roi_h),
    ]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    means: List[float] = []
    for (y0, y1) in thirds:
        seg = hsv[y0:y1, :, :]
        if seg.size != 0:
            vch = seg[..., 2].astype(np.float32)
            means.append(float(np.mean(vch)))
        else:
            means.append(0.0)

    idx = int(np.argmax(means))
    seg = hsv[thirds[idx][0]:thirds[idx][1], :, :]
    if seg.size == 0:
        return "UNKNOWN"

    hch = seg[..., 0].astype(np.float32)
    sch = seg[..., 1].astype(np.float32) / 255.0
    mask_sat = sch > 0.35
    if not np.any(mask_sat):
        return "UNKNOWN"
    hsel = hch[mask_sat]

    red_mask = (hsel < 12) | (hsel > 168)
    yellow_mask = (hsel >= 18) & (hsel <= 35)
    green_mask = (hsel >= 40) & (hsel <= 105)

    if idx == 0 and np.mean(red_mask) > 0.25:
        return "RED"
    if idx == 1 and np.mean(yellow_mask) > 0.25:
        return "YELLOW"
    if idx == 2 and np.mean(green_mask) > 0.25:
        return "GREEN"

    # Fallback: bottom bulb strong green
    b0, b1 = thirds[2]
    seg_bot = hsv[b0:b1, :, :]
    if seg_bot.size != 0:
        h_bot = seg_bot[..., 0].astype(np.float32)
        s_bot = seg_bot[..., 1].astype(np.float32) / 255.0
        mask_sat_bot = s_bot > 0.35
        if np.any(mask_sat_bot):
            g_frac = np.mean(((h_bot >= 40) & (h_bot <= 105))[mask_sat_bot])
            if g_frac > 0.35:
                return "GREEN"

    return "UNKNOWN"


def _apply_digital_zoom(image: np.ndarray,
                        zoom: float,
                        top_bias: float = TELEPHOTO_ZOOM_TOP_BIAS
                        ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """Crop+resize upper‑center region of an image to emulate extra zoom."""
    if image is None:
        return image, None
    zoom = max(1.0, float(zoom))
    if zoom <= 1.0 + 1e-3:
        return image, None
    h, w = image.shape[:2]
    crop_w = max(4, min(w, int(round(w / zoom))))
    crop_h = max(4, min(h, int(round(h / zoom))))
    x0 = max(0, (w - crop_w) // 2)
    bias = max(0.0, min(1.0, float(top_bias)))
    y0 = max(0, min(h - crop_h, int(round((h - crop_h) * bias))))
    crop = image[y0:y0 + crop_h, x0:x0 + crop_w]
    resized = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    meta = {
        "crop": (x0, y0, crop_w, crop_h),
        "scale_x": crop_w / float(w),
        "scale_y": crop_h / float(h),
    }
    return resized, meta


def _remap_zoom_box(
    box: Tuple[int, int, int, int],
    meta: Optional[Dict[str, Any]],
    full_w: int,
    full_h: int,
) -> Tuple[int, int, int, int]:
    """Convert a bounding box from zoomed coordinates back to original image."""
    if not meta:
        return box

    x0, y0, crop_w, crop_h = meta["crop"]
    scale_x = meta["scale_x"]
    scale_y = meta["scale_y"]

    x1, y1, w, h = box
    x1_full = int(round(x0 + x1 * scale_x))
    y1_full = int(round(y0 + y1 * scale_y))
    w_full = int(round(max(1, w * scale_x)))
    h_full = int(round(max(1, h * scale_y)))

    x1_full = max(0, min(full_w - 1, x1_full))
    y1_full = max(0, min(full_h - 1, y1_full))
    if x1_full + w_full > full_w:
        w_full = max(1, full_w - x1_full)
    if y1_full + h_full > full_h:
        h_full = max(1, full_h - y1_full)

    return x1_full, y1_full, w_full, h_full


class PerceptionPipeline:
    """
    High‑level perception stack: YOLO -> distance -> gating -> TL/stop cues.

    Designed to be wrapped by :class:`ecu.PerceptionECU`:

        pipe = PerceptionPipeline(detector, range_estimator, config)
        perc_ecu = PerceptionECU(pipe.run)
    """

    def __init__(
        self,
        detector: BaseDetector,
        range_estimator: Optional[RangeEstimator],
        config: Optional[PerceptionConfig] = None,
    ) -> None:
        self.detector = detector
        self.range_estimator = range_estimator
        self.config = config or PerceptionConfig()

        # Telephoto intrinsics (used for TL ranging).
        self.tele_fx, self.tele_fy, self.tele_cx, self.tele_cy = intrinsics_from_fov(
            TELEPHOTO_IMG_W, TELEPHOTO_IMG_H, TELEPHOTO_FOV_X_DEG
        )

        self._frame_index = 0
        self._tl_state_history: deque[Dict[str, Any]] = deque(
            maxlen=self.config.tl_smoothing_frames
        )
        self._telephoto_last_candidate: Optional[Dict[str, Any]] = None
        self._telephoto_last_time: float = -1.0

    # ------------------------------------------------------------------ public API

    def run(
        self,
        bgr: np.ndarray,
        depth_m: Optional[np.ndarray],
        depth_stereo_m: Optional[np.ndarray],
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        sim_time: float,
        sensor_timestamp: Optional[float],
        v: float,
        mu: float,
        log_both: bool,        # kept for ECU compatibility; currently unused
        csv_writer: Any,       # kept for ECU compatibility; currently unused
        tele_bgr: Optional[np.ndarray] = None,
        tele_depth_m: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Main perception step.

        The signature is intentionally identical to the original
        ``App._perception_step`` so it can be passed directly into
        :class:`ecu.PerceptionECU`. :contentReference[oaicite:2]{index=2}
        """
        del log_both, csv_writer  # unused in this refactoring

        self._frame_index += 1
        labels = self.detector.labels or {}

        classIds, confs, boxes = self.detector.predict_raw(bgr)

        nearest_s_active: Optional[float] = None
        nearest_kind: Optional[str] = None
        nearest_thr: Optional[float] = None
        nearest_box: Optional[Tuple[int, int, int, int]] = None
        nearest_conf: Optional[float] = None
        stop_detected_current = False

        tl_candidate_primary_near: Optional[Dict[str, Any]] = None
        tl_candidate_primary_any: Optional[Dict[str, Any]] = None

        det_points: List[Dict[str, Any]] = []
        obstacle_measurements: List[Dict[str, Any]] = []

        primary_crop_top = int(TL_PRIMARY_CROP_FRAC * IMG_H)
        cx0 = IMG_W / 2.0

        if len(classIds):
            for cid, conf, box in zip(
                np.array(classIds).flatten(),
                np.array(confs).flatten(),
                boxes,
            ):
                x1, y1, w, h = map(int, box)
                x2, y2 = x1 + w, y1 + h

                name = labels.get(cid, str(cid))
                norm = _norm_label(name)

                # Confidence threshold (per‑class if configured)
                conf_req = self.config.class_conf_map.get(norm, self.detector.conf_thr)
                if conf < conf_req:
                    continue

                # Minimum box height
                h_req = int(self.config.min_h_px.get(norm, self.config.min_h_px_default))
                is_tl_like = ("traffic" in norm and "light" in norm)
                if (norm not in TRIGGER_NAMES_NORM and not is_tl_like) or (h < h_req):
                    # Let TL‑like boxes through even if not in TRIGGER_NAMES_NORM.
                    if not (is_tl_like and h >= h_req):
                        continue

                xc = x1 + w / 2.0

                # Range estimation
                #  - pinhole: canonical object height
                #  - depth: CARLA depth camera
                #  - stereo: fused with depth if available
                H_real = OBJ_HEIGHT_M.get(
                    name,
                    OBJ_HEIGHT_M.get("traffic light") if is_tl_like else None,
                )
                s_pinhole = (fy * float(H_real) / float(h)) if (H_real and h > 0) else None

                box_key = (x1, y1, w, h)

                s_depth = _median_depth_in_box(
                    depth_m, box_key, self.config.depth_roi_shrink
                ) if depth_m is not None else None

                s_stereo = _median_depth_in_box(
                    depth_stereo_m, box_key, self.config.stereo_roi_shrink
                ) if depth_stereo_m is not None else None

                fusion_src = "none"
                range_mode = self.config.range_est.lower()
                if range_mode == "pinhole":
                    s_use = s_pinhole
                elif range_mode == "depth":
                    s_use = s_depth
                elif range_mode == "stereo":
                    s_use, fusion_src = _fuse_depth_sources(s_depth, s_stereo, h)
                else:  # "both" or unknown → prefer depth, fall back to pinhole
                    s_use = s_depth if s_depth is not None else s_pinhole

                if s_use is None or not math.isfinite(s_use) or s_use <= 0.0:
                    continue

                # Lateral gating for vehicles / pedestrians / generic triggers
                do_gating = (
                    norm in VEHICLE_CLASSES
                    or norm in PEDESTRIAN_CLASSES
                    or norm in ("stopsign", "stop sign")
                    or is_tl_like
                    or norm in self.config.gate_center_frac
                    or norm in self.config.gate_lateral_max
                )

                lateral_ok = True
                if do_gating:
                    frac_cls = float(
                        self.config.gate_center_frac.get(norm, CENTER_BAND_FRAC)
                    )
                    band_px_cls = frac_cls * IMG_W
                    if abs(xc - cx0) > band_px_cls:
                        lateral_ok = False
                    else:
                        lateral_range = s_use
                        lateral_max_m = float(
                            self.config.gate_lateral_max.get(norm, LATERAL_MAX)
                        )
                        # approximate horizontal angle → lateral
                        angle_x = (xc - cx) / max(1e-6, fx)
                        lateral = angle_x * max(1.0, lateral_range)
                        if abs(lateral) > lateral_max_m:
                            lateral_ok = False

                if not lateral_ok:
                    continue

                # Traffic‑light ROI color + candidates
                tl_state_roi = "UNKNOWN"
                if is_tl_like:
                    # Ignore tiny TLs in the very far field
                    if y2 <= primary_crop_top:
                        continue
                    y_tl = max(y1, primary_crop_top)
                    h_tl = max(1, y2 - y_tl)
                    tl_state_roi = _estimate_tl_color_from_roi(
                        bgr, (x1, y_tl, w, h_tl)
                    )
                    cand = {
                        "distance": float(s_use),
                        "box": (x1, y1, w, h),
                        "state": tl_state_roi,
                        "source": "primary",
                    }
                    if (
                        tl_candidate_primary_any is None
                        or cand["distance"] < tl_candidate_primary_any["distance"]
                    ):
                        tl_candidate_primary_any = cand
                    if cand["distance"] <= TL_PRIMARY_SHORT_RANGE_M:
                        if (
                            tl_candidate_primary_near is None
                            or cand["distance"] < tl_candidate_primary_near["distance"]
                        ):
                            tl_candidate_primary_near = cand

                # Nearest active obstacle
                if (nearest_s_active is None) or (s_use < nearest_s_active):
                    nearest_s_active = s_use
                    nearest_kind = name
                    nearest_box = (x1, y1, w, h)
                    nearest_conf = float(conf)

                # Engage distance by kind
                kind: Optional[str] = None
                thr_for_kind: Optional[float] = None
                if norm == "stopsign":
                    kind = "stop sign"
                    thr_for_kind = self.config.engage_distance.get(norm, S_ENGAGE)
                    if s_use <= thr_for_kind:
                        stop_detected_current = True
                elif norm in PEDESTRIAN_CLASSES:
                    kind = name
                    thr_for_kind = self.config.engage_distance.get(norm, S_ENGAGE_PED)
                elif norm in VEHICLE_CLASSES:
                    kind = name
                    thr_for_kind = self.config.engage_distance.get(norm, S_ENGAGE)

                if thr_for_kind is not None and nearest_thr is None:
                    nearest_thr = thr_for_kind

                # Obstacle measurements for lead tracking (exclude stop sign)
                if kind is not None and kind != "stop sign":
                    obstacle_measurements.append(
                        {
                            "distance": float(s_use),
                            "box": (x1, y1, w, h),
                            "kind": kind,
                            "timestamp": (
                                sensor_timestamp
                                if sensor_timestamp is not None
                                and math.isfinite(sensor_timestamp)
                                else sim_time
                            ),
                            "confidence": float(conf),
                        }
                    )

                # Annotate bgr for HUD / video
                color = (0, 255, 255)
                if norm == "stopsign":
                    color = (0, 0, 255)
                label = name
                if is_tl_like:
                    label = f"{name} {tl_state_roi}"

                cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
                ann_parts: List[str] = []
                if s_pinhole is not None:
                    ann_parts.append(f"P:{s_pinhole:.1f}m")
                if s_depth is not None:
                    ann_parts.append(f"D:{s_depth:.1f}m")
                if s_stereo is not None:
                    ann_parts.append(f"S:{s_stereo:.1f}m")
                if (
                    range_mode == "stereo"
                    and s_use is not None
                    and fusion_src in ("fused", "min")
                ):
                    ann_parts.append(f"F:{s_use:.1f}m")
                cv2.putText(
                    bgr,
                    f"{label} " + " ".join(ann_parts),
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

                # HUD 3D point for distance overlay
                u = x1 + w / 2.0
                v_px = y1 + h / 2.0
                z = s_use
                xyz = pixel_to_camera(u, v_px, z, fx, fy, cx, cy)
                if xyz is not None:
                    det_points.append(
                        {"name": name, "box": (x1, y1, w, h), "xyz": xyz, "z": z}
                    )

        # Telephoto TL helper (optional)
        tl_det_s: Optional[float]
        tl_det_box: Optional[Tuple[int, int, int, int]]
        tl_det_state: str
        tl_source: str

        tl_candidate = tl_candidate_primary_near
        if tl_candidate is None:
            tele_cand = self._maybe_run_telephoto_tl(
                tele_bgr, tele_depth_m, sim_time
            )
            if tele_cand is not None:
                tl_candidate = tele_cand
        if tl_candidate is None and tl_candidate_primary_any is not None:
            tl_candidate = tl_candidate_primary_any

        if tl_candidate is not None:
            tl_det_s = tl_candidate.get("distance")
            tl_det_box = (
                tl_candidate.get("box")
                if tl_candidate.get("source") == "primary"
                else None
            )
            tl_det_state = tl_candidate.get("state", "UNKNOWN")
            tl_source = tl_candidate.get("source", "primary")
        else:
            tl_det_s, tl_det_box, tl_det_state = None, None, "UNKNOWN"
            tl_source = "none"

        tl_det_state, tl_det_s = self._smooth_tl_state(tl_det_state, tl_det_s)

        return {
            "bgr": bgr,
            "det_points": det_points,
            "nearest_s_active": nearest_s_active,
            "nearest_kind": nearest_kind,
            "nearest_thr": nearest_thr,
            "nearest_box": nearest_box,
            "nearest_conf": nearest_conf,
            "tl_state": tl_det_state,
            "tl_s_active": tl_det_s,
            "tl_det_box": tl_det_box,
            "tl_source": tl_source,
            "stop_detected_current": stop_detected_current,
            "obstacle_measurements": obstacle_measurements,
        }

    # ------------------------------------------------------------------ helpers

    def _smooth_tl_state(
        self, state: str, distance: Optional[float]
    ) -> Tuple[str, Optional[float]]:
        """Short rolling window majority‑vote smoothing for TL state."""
        self._tl_state_history.append({"state": state, "distance": distance})
        votes = [entry for entry in self._tl_state_history if entry["state"] != "UNKNOWN"]
        if votes:
            counts = Counter(entry["state"] for entry in votes)
            best_state, _ = counts.most_common(1)[0]
            dists = [
                entry["distance"]
                for entry in votes
                if entry["state"] == best_state and entry["distance"] is not None
            ]
            best_dist: Optional[float] = None
            if dists:
                best_dist = float(sum(dists) / len(dists))
            elif distance is not None:
                best_dist = distance
            return best_state, best_dist
        return state, distance

    def _maybe_run_telephoto_tl(
        self,
        tele_bgr: Optional[np.ndarray],
        tele_depth_m: Optional[np.ndarray],
        sim_time: float,
    ) -> Optional[Dict[str, Any]]:
        """Optional telephoto TL detector to extend red‑light range."""
        if not self.config.telephoto_enabled or tele_bgr is None:
            return None

        # Simple stride+cache behaviour
        stride = max(1, int(self.config.telephoto_stride))
        self._frame_index = (self._frame_index + 1) % (10_000_000)
        run_now = (self._frame_index % stride) == 0
        if not run_now:
            cached = self._telephoto_last_candidate
            if cached is not None and (
                sim_time - float(self._telephoto_last_time)
            ) <= TELEPHOTO_CACHE_MAX_AGE_S:
                return cached
            return None

        infer_img = tele_bgr
        zoom_meta = None
        if self.config.telephoto_zoom > 1.0 + 1e-3:
            infer_img, zoom_meta = _apply_digital_zoom(
                tele_bgr, self.config.telephoto_zoom
            )

        labels = self.detector.labels or {}
        classIds, confs, boxes = self.detector.predict_raw(infer_img)

        best: Optional[Dict[str, Any]] = None
        for cid, conf, box in zip(
            np.array(classIds).flatten(),
            np.array(confs).flatten(),
            boxes,
        ):
            x1, y1, w, h = map(int, box)
            name = labels.get(cid, str(cid))
            norm = _norm_label(name)
            if "traffic" not in norm or "light" not in norm:
                continue

            conf_req = self.config.class_conf_map.get(norm, self.detector.conf_thr)
            if conf < conf_req or h <= 0 or w <= 0:
                continue

            full_box = _remap_zoom_box(
                (x1, y1, w, h), zoom_meta, TELEPHOTO_IMG_W, TELEPHOTO_IMG_H
            )
            x1_f, y1_f, w_f, h_f = full_box
            if h_f <= 0 or w_f <= 0:
                continue

            H_real = OBJ_HEIGHT_M.get(name, OBJ_HEIGHT_M.get("traffic light"))
            s_pinhole = (
                self.tele_fy * float(H_real) / float(h_f)
                if (H_real is not None and h_f > 0)
                else None
            )
            s_depth = _median_depth_in_box(
                tele_depth_m, full_box, self.config.depth_roi_shrink
            )
            s_use = s_depth if s_depth is not None else s_pinhole
            if s_use is None or not math.isfinite(s_use):
                continue

            tl_state_roi = _estimate_tl_color_from_roi(tele_bgr, full_box)
            cand = {
                "distance": float(s_use),
                "box": full_box,
                "state": tl_state_roi,
                "source": "telephoto",
            }
            if best is None or cand["distance"] < best["distance"]:
                best = cand

        self._telephoto_last_candidate = best
        self._telephoto_last_time = sim_time
        return best
