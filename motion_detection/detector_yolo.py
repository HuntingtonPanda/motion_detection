"""YOLOv8 detector wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import Detection

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover - import error depends on environment
    YOLO = None


@dataclass
class YoloDetector:
    model_path: str
    device: str = "auto"
    conf_thresh: float = 0.35
    iou_thresh: float = 0.45
    person_class_id: int = 0

    def __post_init__(self) -> None:
        if YOLO is None:
            raise RuntimeError("ultralytics is required to use YoloDetector")
        self.device = self._resolve_device(self.device)
        self.model = YOLO(self.model_path)
        self.is_cuda = self.device.startswith("cuda")

    def _resolve_device(self, device: str) -> str:
        if device in {"cpu", "cuda", "cuda:0"}:
            return "cuda:0" if device == "cuda" else device
        if device != "auto":
            return device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:
            pass
        return "cpu"

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            classes=[self.person_class_id],
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        detections: list[Detection] = []
        boxes = results[0].boxes
        if boxes is None:
            return detections

        for box in boxes:
            xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
            conf = float(box.conf.item())
            class_id = int(box.cls.item())
            detections.append(
                Detection(
                    bbox_xyxy=xyxy,
                    conf=conf,
                    class_id=class_id,
                    label="person",
                )
            )
        return detections
