"""Motion region extraction using OpenCV background subtraction."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .types import MotionRegion


@dataclass
class MotionClassifier:
    min_motion_area: int = 1200
    history: int = 300
    var_threshold: float = 16.0
    detect_shadows: bool = False

    def __post_init__(self) -> None:
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows,
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def extract_regions(self, frame: np.ndarray) -> list[MotionRegion]:
        fg_mask = self._subtractor.apply(frame)
        cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self._kernel, iterations=2)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions: list[MotionRegion] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_motion_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            regions.append(
                MotionRegion(
                    bbox_xyxy=(float(x), float(y), float(x + w), float(y + h)),
                    area=area,
                )
            )
        return regions
