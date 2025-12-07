import math
from typing import Optional

import numpy as np
import pygame
import cv2


def fov_y_from_x(width: int, height: int, fov_x_deg: float) -> float:
    fov_x = math.radians(fov_x_deg)
    return 2.0 * math.atan((height / width) * math.tan(fov_x / 2.0))


def focal_length_y_px(width: int, height: int, fov_x_deg: float) -> float:
    fovy = fov_y_from_x(width, height, fov_x_deg)
    return (height / 2.0) / math.tan(fovy / 2.0)


def intrinsics_from_fov(width: int, height: int, fov_x_deg: float):
    fov_x = math.radians(fov_x_deg)
    fx = (width / 2.0) / math.tan(fov_x / 2.0)
    fy = focal_length_y_px(width, height, fov_x_deg)
    cx, cy = width / 2.0, height / 2.0
    return fx, fy, cx, cy


def pixel_to_camera(u: float, v: float, z_m: float, fx: float, fy: float, cx: float, cy: float):
    if z_m <= 0 or not np.isfinite(z_m):
        return None
    x = (u - cx) * z_m / max(1e-6, fx)
    y = (v - cy) * z_m / max(1e-6, fy)
    return (float(x), float(y), float(z_m))


def bgr_to_pygame_surface(bgr: np.ndarray) -> pygame.Surface:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))


def carla_image_to_surface(image) -> pygame.Surface:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    rgb = arr[:, :, :3][:, :, ::-1]  # BGRA->RGB
    return pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))


def decode_depth_meters_from_bgra(depth_bgra: np.ndarray) -> np.ndarray:
    b = depth_bgra[..., 0].astype(np.uint32)
    g = depth_bgra[..., 1].astype(np.uint32)
    r = depth_bgra[..., 2].astype(np.uint32)
    normalized = (r + g * 256 + b * 256 * 256).astype(np.float32) / float(256**3 - 1)
    return 1000.0 * normalized
