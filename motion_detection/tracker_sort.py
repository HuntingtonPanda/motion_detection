"""SORT-style tracker with IoU matching and Hungarian assignment."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .types import BBox, Detection, Track


def _bbox_iou(a: BBox, b: BBox) -> float:
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


@dataclass
class _TrackInternal:
    track_id: int
    bbox_xyxy: BBox
    conf: float
    class_id: int
    label: str
    hits: int = 1
    time_since_update: int = 0
    age: int = 1


@dataclass
class SortTracker:
    max_age: int = 10
    min_hits: int = 1
    iou_threshold: float = 0.3
    _tracks: list[_TrackInternal] = field(default_factory=list)
    _next_track_id: int = 1

    def update(self, detections: list[Detection]) -> list[Track]:
        for track in self._tracks:
            track.age += 1
            track.time_since_update += 1

        matches, unmatched_tracks, unmatched_dets = self._associate(detections)

        for track_idx, det_idx in matches:
            det = detections[det_idx]
            track = self._tracks[track_idx]
            track.bbox_xyxy = det.bbox_xyxy
            track.conf = det.conf
            track.class_id = det.class_id
            track.label = det.label
            track.hits += 1
            track.time_since_update = 0

        for det_idx in unmatched_dets:
            det = detections[det_idx]
            self._tracks.append(
                _TrackInternal(
                    track_id=self._next_track_id,
                    bbox_xyxy=det.bbox_xyxy,
                    conf=det.conf,
                    class_id=det.class_id,
                    label=det.label,
                )
            )
            self._next_track_id += 1

        self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]

        public_tracks: list[Track] = []
        for track in self._tracks:
            if track.hits < self.min_hits:
                continue
            public_tracks.append(
                Track(
                    track_id=track.track_id,
                    bbox_xyxy=track.bbox_xyxy,
                    conf=track.conf,
                    class_id=track.class_id,
                    label=track.label,
                )
            )
        return public_tracks

    def _associate(
        self, detections: list[Detection]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not self._tracks or not detections:
            return [], list(range(len(self._tracks))), list(range(len(detections)))

        iou_matrix = np.zeros((len(self._tracks), len(detections)), dtype=np.float32)
        for i, track in enumerate(self._tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = _bbox_iou(track.bbox_xyxy, det.bbox_xyxy)

        row_idx, col_idx = linear_sum_assignment(1.0 - iou_matrix)
        matches: list[tuple[int, int]] = []
        unmatched_tracks = set(range(len(self._tracks)))
        unmatched_dets = set(range(len(detections)))

        for r, c in zip(row_idx.tolist(), col_idx.tolist()):
            if iou_matrix[r, c] < self.iou_threshold:
                continue
            matches.append((r, c))
            unmatched_tracks.discard(r)
            unmatched_dets.discard(c)

        return matches, sorted(unmatched_tracks), sorted(unmatched_dets)
