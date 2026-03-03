from __future__ import annotations

import numpy as np

from motion_detection.config import AppConfig
from motion_detection.overlay import OverlayRenderer
from motion_detection.pipeline import MotionPipeline
from motion_detection.state_machine import TrackStateMachine
from motion_detection.types import Detection, MotionRegion, MotionState, Track


class StepMotion:
    def __init__(self) -> None:
        self._calls = 0

    def extract_regions(self, frame: np.ndarray) -> list[MotionRegion]:
        self._calls += 1
        if self._calls == 1:
            return [MotionRegion(bbox_xyxy=(0.0, 0.0, 90.0, 90.0), area=8100.0)]
        return []


class StepDetector:
    is_cuda = False

    def __init__(self) -> None:
        self._calls = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self._calls += 1
        return [
            Detection(
                bbox_xyxy=(10.0, 10.0, 60.0, 60.0),
                conf=0.9,
                class_id=0,
                label="person",
            )
        ]


class StickyTracker:
    def __init__(self) -> None:
        self._bbox = (10.0, 10.0, 60.0, 60.0)

    def update(self, detections: list[Detection]) -> list[Track]:
        if detections:
            self._bbox = detections[0].bbox_xyxy
        return [Track(track_id=1, bbox_xyxy=self._bbox, label="person")]


def test_pipeline_smoke_active_then_recent_then_inactive() -> None:
    pipeline = MotionPipeline(
        config=AppConfig(det_every_cpu=3, det_every_cuda=1, active_seconds=0.5, previous_seconds=3.0),
        motion=StepMotion(),
        detector=StepDetector(),
        tracker=StickyTracker(),
        state_machine=TrackStateMachine(active_seconds=0.5, previous_seconds=3.0),
    )
    renderer = OverlayRenderer()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    first = pipeline.process_frame(frame, now_ts=100.0)
    assert len(first.tracks) == 1
    assert first.tracks[0].state == MotionState.ACTIVE
    first_render = renderer.render(frame, first.tracks, fps=24.0)
    assert first_render.shape == frame.shape
    assert np.any(first_render != frame)

    second = pipeline.process_frame(frame, now_ts=100.8)
    assert len(second.tracks) == 1
    assert second.tracks[0].state == MotionState.RECENT
    second_render = renderer.render(frame, second.tracks, fps=24.0)
    assert second_render.shape == frame.shape
    assert np.any(second_render != frame)

    third = pipeline.process_frame(frame, now_ts=104.0)
    assert third.tracks == []
