import cv2
import numpy as np
from typing import Optional

from camera_utils import intrinsics_from_fov
from config import IMG_W, IMG_H, FOV_X_DEG, STEREO_BASELINE_M


class RangeEstimator:
    def __init__(self, use_cuda: bool = False, method: str = 'bm'):
        self.fx, self.fy, self.cx, self.cy = intrinsics_from_fov(IMG_W, IMG_H, FOV_X_DEG)
        self.use_cuda = bool(use_cuda)
        self.method = method.lower()
        self.stereo = None           # CPU StereoBM
        self.stereo_cuda = None      # CUDA StereoBM/SGM

    # ===================== World & Sensors
    def ensure_stereo(self, num_disp: int = 16*6, block: int = 15):
        """Create a stereo matcher. If --stereo-cuda is set and OpenCV CUDA is available,
        prefer GPU; otherwise fall back to CPU StereoBM. Always returns True/False for 'ready'."""
        if self.use_cuda and hasattr(cv2, 'cuda'):
            try:
                if self.method == 'sgm' and hasattr(cv2.cuda, 'StereoSGM_create'):
                    self.stereo_cuda = cv2.cuda.StereoSGM_create()  # type: ignore[attr-defined]
                else:
                    if hasattr(cv2.cuda, 'createStereoBM'):
                        self.stereo_cuda = cv2.cuda.createStereoBM(numDisparities=num_disp, blockSize=block)  # type: ignore[attr-defined]
                    elif hasattr(cv2, 'cuda_StereoBM_create'):
                        self.stereo_cuda = cv2.cuda_StereoBM_create(numDisparities=num_disp, blockSize=block)  # type: ignore[attr-defined]
            except Exception:
                self.stereo_cuda = None
        if self.stereo_cuda is None and self.stereo is None:
            try:
                self.stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block)
            except Exception:
                self.stereo = None
        return (self.stereo_cuda is not None) or (self.stereo is not None)

    def stereo_depth(self, left_bgra: np.ndarray, right_bgra: np.ndarray) -> Optional[np.ndarray]:
        if self.stereo_cuda is not None and hasattr(cv2, 'cuda'):
            try:
                left_gray  = cv2.cvtColor(left_bgra,  cv2.COLOR_BGRA2GRAY)
                right_gray = cv2.cvtColor(right_bgra, cv2.COLOR_BGRA2GRAY)
                gL = cv2.cuda_GpuMat(); gR = cv2.cuda_GpuMat()
                gL.upload(left_gray); gR.upload(right_gray)
                disp_gpu = self.stereo_cuda.compute(gL, gR)  # type: ignore[attr-defined]
                disp = disp_gpu.download().astype(np.float32) / 16.0
                depth = np.full_like(disp, 1e6, dtype=np.float32)
                valid = disp > 0.1
                if np.any(valid):
                    depth[valid] = (self.fx * STEREO_BASELINE_M) / disp[valid]
                return depth
            except Exception:
                pass
        if self.stereo is None:
            return None
        left_gray  = cv2.cvtColor(left_bgra,  cv2.COLOR_BGRA2GRAY)
        right_gray = cv2.cvtColor(right_bgra, cv2.COLOR_BGRA2GRAY)
        disp = self.stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        depth = np.full_like(disp, 1e6, dtype=np.float32)
        valid = disp > 0.1
        if np.any(valid):
            depth[valid] = (self.fx * STEREO_BASELINE_M) / disp[valid]
        return depth
