"""Track-level motion state transitions based on timestamps."""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import MotionState


@dataclass
class TrackStateMachine:
    active_seconds: float = 0.5
    previous_seconds: float = 3.0
    _last_motion_ts: dict[int, float] = field(default_factory=dict)

    def mark_motion(self, track_id: int, ts: float) -> None:
        self._last_motion_ts[track_id] = ts

    def get_state(self, track_id: int, now_ts: float) -> MotionState:
        last_motion_ts = self._last_motion_ts.get(track_id)
        if last_motion_ts is None:
            return MotionState.INACTIVE

        delta = now_ts - last_motion_ts
        if delta <= self.active_seconds:
            return MotionState.ACTIVE
        if delta <= self.previous_seconds:
            return MotionState.RECENT
        return MotionState.INACTIVE

    def prune_inactive(self, now_ts: float) -> None:
        stale = [
            track_id
            for track_id, ts in self._last_motion_ts.items()
            if (now_ts - ts) > self.previous_seconds
        ]
        for track_id in stale:
            del self._last_motion_ts[track_id]
