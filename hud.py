# hud.py
"""
Simple pygame HUD for the braking demo.

This is a thin wrapper around the original `_draw_hud` helper in
`dynamic_brake_state.App`, turned into a proper class so the orchestration
file can stay small. 
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pygame

from camera_utils import bgr_to_pygame_surface  # type: ignore
from config import IMG_W, IMG_H  # type: ignore


def yaw_to_compass(yaw_deg: float) -> str:
    """Convert CARLA yaw [deg] into a human‑friendly compass string."""
    y = (yaw_deg + 360.0) % 360.0
    dirs = ["E", "SE", "S", "SW", "W", "NW", "N", "NE", "E"]
    idx = int((y + 22.5) // 45)
    return dirs[idx]


def _shadow_text(
    surface: pygame.Surface,
    text: str,
    pos: Tuple[int, int],
    color: Tuple[int, int, int],
    shadow_color: Tuple[int, int, int] = (0, 0, 0),
    offset: int = 1,
) -> None:
    font = pygame.font.SysFont("Arial", 20)
    s_shadow = font.render(text, True, shadow_color)
    surface.blit(s_shadow, (pos[0] + offset, pos[1] + offset))
    s = font.render(text, True, color)
    surface.blit(s, pos)


class HUD:
    """
    HUD renderer.

    Parameters
    ----------
    screen
        pygame display surface.
    telephoto_enabled
        If True, show telephoto pane on the right.
    headless
        If True, :meth:`draw` becomes a no‑op.
    """

    def __init__(
        self,
        screen: pygame.Surface,
        telephoto_enabled: bool = True,
        headless: bool = False,
    ) -> None:
        self.screen = screen
        self.telephoto_enabled = bool(telephoto_enabled)
        self.headless = bool(headless)

    def draw(
        self,
        bgr: np.ndarray,
        perf_fps: float,
        perf_ms: float,
        *,
        x: float,
        y: float,
        z: float,
        yaw_deg: float,
        frame_id: int,
        v_mps: float,
        trigger_name: Optional[str],
        tl_state: str,
        throttle: float,
        brake: float,
        hold_blocked: bool,
        hold_reason: Optional[str],
        no_trigger_elapsed: float,
        no_red_elapsed: float,
        stop_armed: bool,
        stop_release_ignore_until: float,
        sim_time: float,
        dbg_tau_dyn: Optional[float],
        dbg_D_safety_dyn: Optional[float],
        dbg_sigma_depth: Optional[float],
        dbg_gate_hit: Optional[bool],
        dbg_a_des: Optional[float],
        dbg_brake: Optional[float],
        v_target: Optional[float],
        collision_flag: bool,
        det_points: Optional[List[Dict[str, Any]]] = None,
        range_mode_label: Optional[str] = None,
        abs_lambda: Optional[float] = None,
        abs_factor: Optional[float] = None,
        abs_mu: Optional[float] = None,
        abs_regime: Optional[str] = None,
        tele_bgr: Optional[np.ndarray] = None,
    ) -> None:
        """Render the HUD for a single frame."""
        if self.headless:
            return

        screen = self.screen
        screen.fill((0, 0, 0))

        surf_front = bgr_to_pygame_surface(bgr)
        screen.blit(surf_front, (0, 0))

        if tele_bgr is not None and self.telephoto_enabled:
            if tele_bgr.shape[0] != IMG_H or tele_bgr.shape[1] != IMG_W:
                import cv2

                tele_bgr = cv2.resize(tele_bgr, (IMG_W, IMG_H))
            surf_tele = bgr_to_pygame_surface(tele_bgr)
            screen.blit(surf_tele, (IMG_W, 0))

        v_kmh = v_mps * 3.6
        compass = yaw_to_compass(yaw_deg)
        txt0 = f"ego @ x={x:8.2f}  y={y:8.2f}  z={z:6.2f}  | yaw={yaw_deg:+6.1f}° {compass}"
        txt1 = (
            f"Frame {frame_id} | v={v_kmh:5.1f} km/h | "
            f"trigger={trigger_name or 'None'} | TL={tl_state}"
        )
        txt2 = (
            f"thr={throttle:.2f}  brk={brake:.2f}  hold={hold_blocked}"
            f"({hold_reason})  clear={no_trigger_elapsed:.1f}s"
            f"  red_clear={no_red_elapsed:.1f}s  stopArmed={stop_armed}"
            f"  ignoreT={(max(0.0, stop_release_ignore_until - sim_time)):.1f}s"
        )
        txt_perf = f"FPS={perf_fps:.1f}  time={perf_ms:.1f}ms"

        _shadow_text(screen, txt_perf, (10, IMG_H - 156), (255, 200, 0))
        _shadow_text(screen, txt0, (10, IMG_H - 134), (200, 200, 255))
        _shadow_text(screen, txt1, (10, IMG_H - 90), (255, 255, 255))
        _shadow_text(screen, txt2, (10, IMG_H - 68), (0, 255, 160))

        if any(
            val is not None
            for val in (
                dbg_tau_dyn,
                dbg_D_safety_dyn,
                dbg_sigma_depth,
                dbg_gate_hit,
                dbg_a_des,
                dbg_brake,
                v_target,
            )
        ):
            tau_txt = "n/a" if dbg_tau_dyn is None else f"{dbg_tau_dyn:0.2f}"
            dsafe_txt = "n/a" if dbg_D_safety_dyn is None else f"{dbg_D_safety_dyn:0.1f}"
            sigma_txt = "n/a" if dbg_sigma_depth is None else f"{dbg_sigma_depth:0.2f}"
            gate_txt = "n/a" if dbg_gate_hit is None else f"{int(bool(dbg_gate_hit))}"
            a_des_txt = "n/a" if dbg_a_des is None else f"{dbg_a_des:0.2f}"
            brk_txt = "n/a" if dbg_brake is None else f"{dbg_brake:0.2f}"
            vtgt_txt = "n/a" if v_target is None else f"{v_target * 3.6:0.0f}"
            _shadow_text(
                screen,
                (
                    f"tau={tau_txt}  Dsafe={dsafe_txt} m  sigma={sigma_txt} m"
                    f"  gate={gate_txt}  a_des={a_des_txt}  brk={brk_txt}"
                    f"  Vtgt={vtgt_txt}km/h"
                ),
                (10, IMG_H - 24),
                (255, 255, 0),
            )

        if abs_lambda is not None or abs_factor is not None or abs_mu is not None:
            slip_txt = "n/a" if abs_lambda is None else f"{abs_lambda:.2f}"
            fac_txt = "1.00" if abs_factor is None else f"{abs_factor:.2f}"
            mu_txt = "n/a" if abs_mu is None else f"{abs_mu:.2f}"
            regime_txt = abs_regime or "n/a"
            _shadow_text(
                screen,
                f"ABS slip={slip_txt}  f={fac_txt}  mu_est={mu_txt}  regime={regime_txt}",
                (10, IMG_H - 46),
                (180, 255, 255),
            )

        # Distance overlay on the left
        log_y = 10
        log_x = 10
        mode_label = range_mode_label or "range"
        _shadow_text(
            screen,
            f"Ego→Object distances ({mode_label}, meters):",
            (log_x, log_y),
            (0, 255, 255),
        )
        log_y += 22

        det_points_sorted: List[Dict[str, Any]] = []
        if det_points:
            det_points_sorted = sorted(
                det_points, key=lambda d: d.get("z", float("inf"))
            )

        if det_points_sorted:
            for d in det_points_sorted[:22]:
                xcam, ycam, zcam = d["xyz"]
                _shadow_text(
                    screen,
                    (
                        f"{d['name']:<14}  Z={zcam:5.1f} m"
                        f"   X={xcam:+4.1f} m   Y={ycam:+4.1f} m"
                    ),
                    (log_x, log_y),
                    (180, 255, 180),
                )
                log_y += 18
        else:
            _shadow_text(
                screen, "No detections", (log_x, log_y), (180, 180, 180)
            )

        if collision_flag:
            _shadow_text(
                screen,
                "*** COLLISION DETECTED ***",
                (IMG_W // 4, IMG_H // 2),
                (255, 40, 40),
            )

        pygame.display.flip()
