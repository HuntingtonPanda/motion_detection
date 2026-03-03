from motion_detection.state_machine import TrackStateMachine
from motion_detection.types import MotionState


def test_state_transitions() -> None:
    sm = TrackStateMachine(active_seconds=0.5, previous_seconds=3.0)
    sm.mark_motion(track_id=7, ts=10.0)

    assert sm.get_state(track_id=7, now_ts=10.0) == MotionState.ACTIVE
    assert sm.get_state(track_id=7, now_ts=10.5) == MotionState.ACTIVE
    assert sm.get_state(track_id=7, now_ts=10.50001) == MotionState.RECENT
    assert sm.get_state(track_id=7, now_ts=13.0) == MotionState.RECENT
    assert sm.get_state(track_id=7, now_ts=13.00001) == MotionState.INACTIVE


def test_unknown_track_is_inactive() -> None:
    sm = TrackStateMachine()
    assert sm.get_state(track_id=404, now_ts=1.0) == MotionState.INACTIVE


def test_prune_inactive_tracks() -> None:
    sm = TrackStateMachine(active_seconds=0.5, previous_seconds=3.0)
    sm.mark_motion(track_id=1, ts=1.0)
    sm.mark_motion(track_id=2, ts=3.5)

    sm.prune_inactive(now_ts=5.0)

    assert sm.get_state(track_id=1, now_ts=5.0) == MotionState.INACTIVE
    assert sm.get_state(track_id=2, now_ts=5.0) == MotionState.RECENT
