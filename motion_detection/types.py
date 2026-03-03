"""Shared data types for the motion pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


BBox = tuple[float, float, float, float]


class MotionState(str, Enum):
    ACTIVE = "active"
    RECENT = "recent"
    INACTIVE = "inactive"


@dataclass(frozen=True)
class Detection:
    bbox_xyxy: BBox
    conf: float
    class_id: int
    label: str


@dataclass(frozen=True)
class MotionRegion:
    bbox_xyxy: BBox
    area: float


@dataclass
class Track:
    track_id: int
    bbox_xyxy: BBox
    conf: float = 1.0
    label: str = "person"
    class_id: int = 0
    last_motion_ts: float | None = None
    state: MotionState = MotionState.INACTIVE
