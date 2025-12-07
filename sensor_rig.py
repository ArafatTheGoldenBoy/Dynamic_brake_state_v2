import queue
import time
from typing import Any, Dict, Optional

from carla_utils import import_carla
from config import (
    FOV_X_DEG,
    IMG_H,
    IMG_W,
    STEREO_BASELINE_M,
    TELEPHOTO_FOV_X_DEG,
    TELEPHOTO_IMG_H,
    TELEPHOTO_IMG_W,
)

carla = import_carla()


class SensorRig:
    def __init__(self, world: carla.World, vehicle: carla.Vehicle, range_est: str,
                 enable_depth: bool = True, enable_telephoto: bool = True):
        self.world = world
        self.vehicle = vehicle
        self.range_est = range_est
        self.enable_telephoto = enable_telephoto
        bp_lib = world.get_blueprint_library()

        cam_bp = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(IMG_W))
        cam_bp.set_attribute('image_size_y', str(IMG_H))
        cam_bp.set_attribute('fov', str(FOV_X_DEG))
        cam_bp.set_attribute('fstop', '8.0')
        cam_bp.set_attribute('focal_distance', '1000.0')
        cam_bp.set_attribute('lens_circle_multiplier', '0.0')
        cam_bp.set_attribute('lens_k', '0.0')
        try:
            if cam_bp.has_attribute('motion_blur_intensity'):
                cam_bp.set_attribute('motion_blur_intensity', '0.0')
            if cam_bp.has_attribute('motion_blur_max_distortion'):
                cam_bp.set_attribute('motion_blur_max_distortion', '0.0')
            if cam_bp.has_attribute('motion_blur_min_object_screen_size'):
                cam_bp.set_attribute('motion_blur_min_object_screen_size', '1.0')
        except RuntimeError:
            pass

        self.cam_front = world.spawn_actor(
            cam_bp, carla.Transform(carla.Location(x=1.6, z=1.5)), attach_to=vehicle)
        self.q_front: "queue.Queue[carla.Image]" = queue.Queue()
        self.cam_front.listen(self.q_front.put)

        self.cam_tele = None
        self.q_tele = None
        self.cam_tele_depth = None
        self.q_tele_depth = None
        if enable_telephoto:
            tele_bp = bp_lib.find('sensor.camera.rgb')
            tele_bp.set_attribute('image_size_x', str(TELEPHOTO_IMG_W))
            tele_bp.set_attribute('image_size_y', str(TELEPHOTO_IMG_H))
            tele_bp.set_attribute('fov', str(TELEPHOTO_FOV_X_DEG))
            tele_bp.set_attribute('fstop', '8.0')
            tele_bp.set_attribute('focal_distance', '1000.0')
            tele_bp.set_attribute('lens_circle_multiplier', '0.0')
            tele_bp.set_attribute('lens_k', '0.0')
            try:
                if tele_bp.has_attribute('motion_blur_intensity'):
                    tele_bp.set_attribute('motion_blur_intensity', '0.0')
                if tele_bp.has_attribute('motion_blur_max_distortion'):
                    tele_bp.set_attribute('motion_blur_max_distortion', '0.0')
                if tele_bp.has_attribute('motion_blur_min_object_screen_size'):
                    tele_bp.set_attribute('motion_blur_min_object_screen_size', '1.0')
            except RuntimeError:
                pass
            tele_tf = carla.Transform(carla.Location(x=1.6, z=1.5))
            self.cam_tele = world.spawn_actor(tele_bp, tele_tf, attach_to=vehicle)
            self.q_tele = queue.Queue()
            self.cam_tele.listen(self.q_tele.put)

            if enable_depth:
                tele_depth_bp = bp_lib.find('sensor.camera.depth')
                tele_depth_bp.set_attribute('image_size_x', str(TELEPHOTO_IMG_W))
                tele_depth_bp.set_attribute('image_size_y', str(TELEPHOTO_IMG_H))
                tele_depth_bp.set_attribute('fov', str(TELEPHOTO_FOV_X_DEG))
                self.cam_tele_depth = world.spawn_actor(tele_depth_bp, tele_tf, attach_to=vehicle)
                self.q_tele_depth = queue.Queue()
                self.cam_tele_depth.listen(self.q_tele_depth.put)

        depth_bp = bp_lib.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', str(IMG_W))
        depth_bp.set_attribute('image_size_y', str(IMG_H))
        depth_bp.set_attribute('fov', str(FOV_X_DEG))
        self.cam_depth = None
        self.q_depth = None
        if enable_depth:
            self.cam_depth = world.spawn_actor(
                depth_bp, carla.Transform(carla.Location(x=1.6, z=1.5)), attach_to=vehicle)
            self.q_depth = queue.Queue()
            self.cam_depth.listen(self.q_depth.put)

        self.cam_stereo_left = None
        self.cam_stereo_right = None
        self.q_stereo_left = None
        self.q_stereo_right = None
        if range_est == 'stereo':
            self.cam_stereo_left  = world.spawn_actor(
                cam_bp, carla.Transform(carla.Location(x=1.6, y=-STEREO_BASELINE_M/2.0, z=1.5)), attach_to=vehicle)
            self.cam_stereo_right = world.spawn_actor(
                cam_bp, carla.Transform(carla.Location(x=1.6, y= STEREO_BASELINE_M/2.0, z=1.5)), attach_to=vehicle)
            self.q_stereo_left, self.q_stereo_right = queue.Queue(), queue.Queue()
            self.cam_stereo_left.listen(self.q_stereo_left.put)
            self.cam_stereo_right.listen(self.q_stereo_right.put)

        self.actors = [a for a in [self.cam_front, self.cam_depth,
                                   self.cam_stereo_left, self.cam_stereo_right,
                                   self.cam_tele, self.cam_tele_depth] if a is not None]

    @staticmethod
    def _get_latest(q: "queue.Queue[Any]", timeout: float):
        try:
            item = q.get(timeout=timeout)
        except queue.Empty:
            return None
        try:
            while True:
                item = q.get_nowait()
        except queue.Empty:
            pass
        return item

    @staticmethod
    def _get_for_frame(q: "queue.Queue[Any]", expected_frame: int, timeout: float):
        """Best-effort fetch for a specific frame, falling back to latest or None."""
        deadline = time.time() + max(timeout, 0.5)
        latest = None
        while time.time() < deadline:
            try:
                item = q.get(timeout=max(0.05, min(0.2, deadline - time.time())))
            except queue.Empty:
                break
            latest = item
            try:
                f = getattr(item, 'frame')
            except Exception:
                f = None
            if f == expected_frame:
                return item
            try:
                while True:
                    nxt = q.get_nowait()
                    latest = nxt
                    f = getattr(nxt, 'frame', None)
                    if f == expected_frame:
                        return nxt
            except queue.Empty:
                pass
        return latest

    def read(self, timeout: float = 2.0, expected_frame: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Read the latest frames, returning None on timeout instead of raising.

        All configured sensors are read best-effort. If the primary front RGB
        frame is missing, we treat this as a hard failure for the tick and
        return None so callers can apply a safe fallback.
        """
        out: Dict[str, Any] = {}
        if expected_frame is None:
            front = self._get_latest(self.q_front, timeout)
            if front is None:
                return None
            out['front'] = front
            if self.q_depth is not None:
                out['depth'] = self._get_latest(self.q_depth, timeout)
            if self.q_stereo_left is not None and self.q_stereo_right is not None:
                out['stereo_left']  = self._get_latest(self.q_stereo_left, timeout)
                out['stereo_right'] = self._get_latest(self.q_stereo_right, timeout)
            if self.q_tele is not None:
                out['tele_rgb'] = self._get_latest(self.q_tele, timeout)
            if self.q_tele_depth is not None:
                out['tele_depth'] = self._get_latest(self.q_tele_depth, timeout)
        else:
            front = self._get_for_frame(self.q_front, expected_frame, timeout)
            if front is None:
                return None
            out['front'] = front
            if self.q_depth is not None:
                out['depth'] = self._get_for_frame(self.q_depth, expected_frame, timeout)
            if self.q_stereo_left is not None and self.q_stereo_right is not None:
                out['stereo_left']  = self._get_for_frame(self.q_stereo_left, expected_frame, timeout)
                out['stereo_right'] = self._get_for_frame(self.q_stereo_right, expected_frame, timeout)
            if self.q_tele is not None:
                out['tele_rgb'] = self._get_for_frame(self.q_tele, expected_frame, timeout)
            if self.q_tele_depth is not None:
                out['tele_depth'] = self._get_for_frame(self.q_tele_depth, expected_frame, timeout)
        return out

    def destroy(self):
        for a in self.actors:
            try:
                a.stop()
            except Exception:
                pass
            try:
                a.listen(lambda *_: None)
            except Exception:
                pass
            try:
                a.destroy()
            except Exception:
                pass
