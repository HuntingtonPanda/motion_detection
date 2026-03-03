from __future__ import annotations

import numpy as np

from motion_detection.config import AppConfig
from motion_detection.pipeline import MotionPipeline, track_overlaps_motion
from motion_detection.state_machine import TrackStateMachine
from motion_detection.types import Detection, MotionRegion, MotionState, Track


class StubMotion:
    def __init__(self, frames: list[list[MotionRegion]]) -> None:
        self._frames = frames
        self._idx = 0

    def extract_regions(self, frame: np.ndarray) -> list[MotionRegion]:
        if self._idx >= len(self._frames):
            return []
        regions = self._frames[self._idx]
        self._idx += 1
        return regions


class StubDetector:
    is_cuda = False

    def __init__(self, detections: list[Detection]) -> None:
        self._detections = detections

    def detect(self, frame: np.ndarray) -> list[Detection]:
        return list(self._detections)


class StubTracker:
    def __init__(self, initial_box: tuple[float, float, float, float]) -> None:
        self._box = initial_box
        self._track_id = 1
        self._alive = 0

    def update(self, detections: list[Detection]) -> list[Track]:
        if detections:
            self._box = detections[0].bbox_xyxy
        self._alive += 1
        if self._alive > 6:
            return []
        return [Track(track_id=self._track_id, bbox_xyxy=self._box)]


def test_track_overlaps_motion_via_iou() -> None:
    regions = [MotionRegion(bbox_xyxy=(0.0, 0.0, 40.0, 40.0), area=1600.0)]
    assert track_overlaps_motion((10.0, 10.0, 50.0, 50.0), regions)


def test_track_overlaps_motion_via_center_inside() -> None:
    regions = [MotionRegion(bbox_xyxy=(20.0, 20.0, 80.0, 80.0), area=3600.0)]
    # Tiny box mostly outside but center remains inside region.
    assert track_overlaps_motion((61.0, 61.0, 99.0, 99.0), regions)


def test_pipeline_marks_motion_from_overlap_and_transitions_states() -> None:
    config = AppConfig(det_every_cpu=3, det_every_cuda=1, active_seconds=0.5, previous_seconds=3.0)
    motion = StubMotion(
        [
            [MotionRegion(bbox_xyxy=(0.0, 0.0, 100.0, 100.0), area=10000.0)],
            [],
            [],
        ]
    )
    detector = StubDetector(
        [
            Detection(
                bbox_xyxy=(10.0, 10.0, 60.0, 60.0),
                conf=0.9,
                class_id=0,
                label="person",
            )
        ]
    )
    tracker = StubTracker(initial_box=(10.0, 10.0, 60.0, 60.0))
    state_machine = TrackStateMachine(active_seconds=0.5, previous_seconds=3.0)
    pipeline = MotionPipeline(
        config=config,
        motion=motion,
        detector=detector,
        tracker=tracker,
        state_machine=state_machine,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    first = pipeline.process_frame(frame, now_ts=1.0)
    assert len(first.tracks) == 1
    assert first.tracks[0].state == MotionState.ACTIVE

    second = pipeline.process_frame(frame, now_ts=1.7)
    assert len(second.tracks) == 1
    assert second.tracks[0].state == MotionState.RECENT

    third = pipeline.process_frame(frame, now_ts=4.2)
    assert third.tracks == []
