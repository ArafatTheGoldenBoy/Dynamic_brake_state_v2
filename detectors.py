from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from config import YOLO_MODEL_PATH
from label_utils import _norm_label

try:
    from ultralytics import YOLO as _ULTRA_YOLO  # type: ignore
except Exception:
    _ULTRA_YOLO = None


CONF_THR_DEFAULT = 0.35
NMS_THR = 0.45


class BaseDetector:
    """Minimal detector interface so we can swap YOLO, SSD, etc."""

    def predict_raw(self, bgr: np.ndarray):
        raise NotImplementedError


def _fallback_labels_91():
    labels = [""] * 90
    mapping = {
        1: "person",
        2: "bicycle",
        3: "car",
        4: "motorcycle",
        6: "bus",
        7: "train",
        8: "truck",
        10: "traffic light",
        13: "stop sign",
    }
    for key, value in mapping.items():
        idx = key - 1
        if 0 <= idx < 90:
            labels[idx] = value
    return labels


class YOLODetector(BaseDetector):
    def __init__(
        self,
        conf_thr: float = CONF_THR_DEFAULT,
        nms_thr: float = NMS_THR,
        img_size: int = 640,
        device: str = "auto",
        use_half: bool = False,
        agnostic: bool = False,
        classes: Optional[str] = None,
        max_det: int = 300,
        dnn: bool = False,
        augment: bool = False,
        per_class_iou_map: Optional[Dict[str, float]] = None,
    ):
        self.conf_thr = conf_thr
        self.nms_thr = nms_thr
        self.img_size = int(img_size)
        self.device = device
        self.use_half = use_half
        self.agnostic = agnostic
        self.classes_raw = classes
        self.max_det = int(max_det)
        self.dnn = bool(dnn)
        self.augment = bool(augment)
        self.per_class_iou_map = per_class_iou_map or {}

        self.model = None
        self.labels: Optional[Dict[int, str]] = None
        self.enabled = False

        if _ULTRA_YOLO is not None and os.path.exists(YOLO_MODEL_PATH):
            try:
                self.model = _ULTRA_YOLO(YOLO_MODEL_PATH)
                self.enabled = True
                try:
                    if hasattr(self.model, "names"):
                        self.labels = {int(k): v for k, v in self.model.names.items()}  # type: ignore
                except Exception:
                    self.labels = None
                dev = self._resolve_device(self.device)
                if (torch is not None) and (self.model is not None):
                    try:
                        self.model.to(dev)
                        if self.use_half and "cuda" in dev and torch.cuda.is_available():
                            if torch.backends and hasattr(torch.backends, "cudnn"):
                                torch.backends.cudnn.benchmark = True  # type: ignore
                    except Exception:
                        pass
            except Exception as exc:
                print(f"[WARN] YOLO load failed: {exc}")
        if self.labels is None:
            fallback = _fallback_labels_91()
            self.labels = {i: v for i, v in enumerate(fallback)}

    def _resolve_device(self, device: str) -> str:
        if device in (None, "", "auto"):
            if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return device

    def _parse_classes(self) -> Optional[List[int]]:
        if not self.classes_raw:
            return None
        raw = [s.strip() for s in str(self.classes_raw).split(",") if s.strip()]
        idxs: List[int] = []
        for token in raw:
            if token.isdigit():
                idxs.append(int(token))
            else:
                if self.labels is None:
                    continue
                lower = token.lower()
                found = [i for i, n in self.labels.items() if isinstance(n, str) and n.lower() == lower]
                if found:
                    idxs.append(found[0])
        return idxs if idxs else None

    @staticmethod
    def _iou_xyxy(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / float(area_a + area_b - inter + 1e-9)

    def _apply_per_class_nms(self, classIds, confs, boxes):
        xyxy = [(x, y, x + w, y + h) for (x, y, w, h) in boxes]
        classIds = list(classIds)
        confs = list(confs)
        keep_mask = [True] * len(xyxy)
        class_to_indices: Dict[str, List[int]] = {}
        for i, cid in enumerate(classIds):
            name = self.labels.get(cid, str(cid)) if self.labels else str(cid)
            class_to_indices.setdefault(_norm_label(name), []).append(i)
        for cname, idxs in class_to_indices.items():
            iou_thr = self.per_class_iou_map.get(cname, None)
            if iou_thr is None:
                continue
            order = sorted(idxs, key=lambda i: confs[i], reverse=True)
            kept: List[int] = []
            while order:
                i = order.pop(0)
                kept.append(i)
                rest = []
                for j in order:
                    if self._iou_xyxy(xyxy[i], xyxy[j]) <= iou_thr:
                        rest.append(j)
                    else:
                        keep_mask[j] = False
                order = rest
        classIds_f = [c for k, c in enumerate(classIds) if keep_mask[k]]
        confs_f = [s for k, s in enumerate(confs) if keep_mask[k]]
        boxes_f = [b for k, b in enumerate(boxes) if keep_mask[k]]
        return classIds_f, confs_f, boxes_f

    def predict_raw(self, bgr: np.ndarray):
        if not self.enabled or self.model is None:
            return [], [], []
        boxes_out: List[Any] = []
        confs_out: List[float] = []
        classIds_out: List[int] = []
        try:
            dev = self._resolve_device(self.device)
            classes_arg = self._parse_classes()
            half_flag = (self.use_half and ("cuda" in dev)) and (
                getattr(torch, "cuda", None) is None or torch.cuda.is_available()
            )
            try:
                results = self.model.predict(
                    bgr,
                    imgsz=self.img_size,
                    conf=self.conf_thr,
                    iou=self.nms_thr,
                    device=dev,
                    half=half_flag,
                    classes=classes_arg,
                    agnostic_nms=self.agnostic,
                    max_det=self.max_det,
                    dnn=self.dnn,
                    augment=self.augment,
                    verbose=False,
                )
            except Exception as prediction_err:
                msg = str(prediction_err)
                if ("same dtype" in msg) or ("Half" in msg) or ("mat1 and mat2" in msg):
                    print("[WARN] YOLO half-precision failed; retrying in FP32...")
                    results = self.model.predict(
                        bgr,
                        imgsz=self.img_size,
                        conf=self.conf_thr,
                        iou=self.nms_thr,
                        device=dev,
                        half=False,
                        classes=classes_arg,
                        agnostic_nms=self.agnostic,
                        max_det=self.max_det,
                        dnn=self.dnn,
                        augment=self.augment,
                        verbose=False,
                    )
                else:
                    raise
            for res in results:
                xyxy = getattr(res.boxes, "xyxy", None)
                confs_tensor = getattr(res.boxes, "conf", None)
                cls_tensor = getattr(res.boxes, "cls", None)
                if xyxy is None or confs_tensor is None or cls_tensor is None:
                    continue
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i].cpu().numpy().tolist()
                    conf_val = float(confs_tensor[i].cpu().numpy())
                    cls_val = int(cls_tensor[i].cpu().numpy())
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                    boxes_out.append((x, y, w, h))
                    classIds_out.append(cls_val)
                    confs_out.append(conf_val)
        except Exception as exc:
            print(f"[WARN] YOLO inference failed: {exc}")
            return [], [], []
        if self.per_class_iou_map:
            try:
                classIds_out, confs_out, boxes_out = self._apply_per_class_nms(classIds_out, confs_out, boxes_out)
            except Exception:
                pass
        return classIds_out, confs_out, boxes_out


__all__ = ["BaseDetector", "YOLODetector", "CONF_THR_DEFAULT", "NMS_THR"]
