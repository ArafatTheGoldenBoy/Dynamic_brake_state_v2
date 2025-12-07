# vehicle_control.py
"""
Generic vehicle‑control helpers: lateral steering and wheel‑speed decoding.

These functions are factored out of the monolithic app so both the live demo
and any offline replay tools can reuse the same logic. 
"""

from __future__ import annotations

import math
from typing import List, Any

from carla_utils import import_carla  # type: ignore
from world_manager import WorldManager  # type: ignore

carla = import_carla()


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def steer_to_waypoint(world: WorldManager, v_mps: float) -> float:
    """
    Simple look‑ahead lane‑following controller.

    Parameters
    ----------
    world
        :class:`world_manager.WorldManager` instance; uses `world.ego` and `world.map`.
    v_mps
        Ego speed in m/s (used to scale look‑ahead distance).

    Returns
    -------
    float
        Normalized steering command in [-1, 1].
    """
    ego = world.ego
    tr = ego.get_transform()
    loc = tr.location
    yaw_r = math.radians(tr.rotation.yaw)

    wp = world.map.get_waypoint(
        loc, project_to_road=True, lane_type=carla.LaneType.Driving
    )

    lookahead = max(6.0, min(12.0, 0.8 * max(v_mps, 1.0)))
    next_wps = wp.next(lookahead) or wp.next(5.0)
    if not next_wps:
        return 0.0

    best = None
    best_diff = 1e9
    for cand in next_wps:
        yaw_t = math.radians(cand.transform.rotation.yaw)
        diff = abs(_wrap_pi(yaw_t - yaw_r))
        if diff < best_diff:
            best, best_diff = cand, diff
    assert best is not None

    tx, ty = best.transform.location.x, best.transform.location.y
    dx, dy = tx - loc.x, ty - loc.y

    angle_to_point = math.atan2(dy, dx)
    heading_error = _wrap_pi(angle_to_point - yaw_r)
    cross_track = (-math.sin(yaw_r)) * dx + (math.cos(yaw_r)) * dy

    steer_cmd = heading_error + math.atan2(0.8 * cross_track, v_mps + 1e-3)
    return max(-1.0, min(1.0, steer_cmd))


def wheel_linear_speeds(world: WorldManager) -> List[float]:
    """
    Best‑effort estimate of per‑wheel linear speeds [m/s].

    Uses CARLA's `get_wheel_angular_velocity` plus wheel radii from the
    physics control; if that fails, returns an empty list.
    """
    veh = getattr(world, "ego", None)
    if veh is None:
        return []

    radii: List[float] = []
    try:
        phys = veh.get_physics_control()
        wheels = getattr(phys, "wheels", None)
        if wheels:
            for w in wheels:
                try:
                    radii.append(max(0.05, float(getattr(w, "radius", 0.35))))
                except Exception:
                    radii.append(0.35)
    except Exception:
        pass

    def _coerce(val: Any) -> float | None:
        if val is None:
            return None
        if isinstance(val, (list, tuple)) and val:
            try:
                return float(val[0])
            except Exception:
                return None
        if hasattr(val, "x") and hasattr(val, "y") and hasattr(val, "z"):
            try:
                return math.sqrt(float(val.x) ** 2 + float(val.y) ** 2 + float(val.z) ** 2)
            except Exception:
                return None
        try:
            return float(val)
        except Exception:
            return None

    ang_vals: List[float] = []
    method = getattr(veh, "get_wheel_angular_velocity", None)
    if callable(method):
        seq_vals = None
        try:
            seq = method()
            if isinstance(seq, (list, tuple)):
                seq_vals = seq
        except TypeError:
            seq_vals = None
        except Exception:
            seq_vals = None

        if seq_vals is not None:
            for val in seq_vals:
                coerced = _coerce(val)
                if coerced is not None:
                    ang_vals.append(max(0.0, coerced))

        if not ang_vals:
            count = len(radii) if radii else 4
            for idx in range(count):
                try:
                    val = method(idx)
                except TypeError:
                    break
                except Exception:
                    continue
                coerced = _coerce(val)
                if coerced is not None:
                    ang_vals.append(max(0.0, coerced))

    if not ang_vals:
        return []

    if not radii:
        radii = [0.34] * len(ang_vals)
    if len(radii) < len(ang_vals):
        radii.extend([radii[-1]] * (len(ang_vals) - len(radii)))

    return [max(0.0, ang * max(0.05, radius)) for ang, radius in zip(ang_vals, radii)]
