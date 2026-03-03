"""Core frame processing pipeline for motion + detection + tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import AppConfig
from .state_machine import TrackStateMachine
from .types import BBox, Detection, MotionRegion, Track


class DetectorLike(Protocol):
    is_cuda: bool

    def detect(self, frame: np.ndarray) -> list[Detection]:
        ...


class TrackerLike(Protocol):
    def update(self, detections: list[Detection]) -> list[Track]:
        ...


class MotionLike(Protocol):
    def extract_regions(self, frame: np.ndarray) -> list[MotionRegion]:
        ...


def bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return inter / denom


def _center_inside(box: BBox, region: BBox) -> bool:
    x1, y1, x2, y2 = box
    rx1, ry1, rx2, ry2 = region
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


def track_overlaps_motion(track_box: BBox, regions: list[MotionRegion], iou_threshold: float = 0.01) -> bool:
    for region in regions:
        region_box = region.bbox_xyxy
        if bbox_iou(track_box, region_box) >= iou_threshold:
            return True
        if _center_inside(track_box, region_box):
            return True
    return False


@dataclass(frozen=True)
class FrameResult:
    tracks: list[Track]
    motion_regions: list[MotionRegion]


@dataclass
class MotionPipeline:
    config: AppConfig
    motion: MotionLike
    detector: DetectorLike
    tracker: TrackerLike
    state_machine: TrackStateMachine

    def __post_init__(self) -> None:
        detector_is_cuda = bool(getattr(self.detector, "is_cuda", False))
        self._detect_every = (
            self.config.det_every_cuda if detector_is_cuda else self.config.det_every_cpu
        )
        self._frame_idx = 0

    def process_frame(self, frame: np.ndarray, now_ts: float) -> FrameResult:
        motion_regions = self.motion.extract_regions(frame)
        should_detect = (self._frame_idx % self._detect_every) == 0
        detections = self.detector.detect(frame) if should_detect else []
        tracks = self.tracker.update(detections)

        for track in tracks:
            if track_overlaps_motion(track.bbox_xyxy, motion_regions):
                self.state_machine.mark_motion(track.track_id, now_ts)

            track.state = self.state_machine.get_state(track.track_id, now_ts)

        self.state_machine.prune_inactive(now_ts)
        self._frame_idx += 1
        return FrameResult(tracks=tracks, motion_regions=motion_regions)
