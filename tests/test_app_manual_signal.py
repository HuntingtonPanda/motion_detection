from __future__ import annotations

from motion_detection.app import _handle_manual_signal_key


class _FakeSender:
    def __init__(self) -> None:
        self.sent_ts: list[float] = []

    def send_manual_greet(self, now_ts: float) -> bool:
        self.sent_ts.append(now_ts)
        return True


def test_handle_manual_signal_key_sends_on_space() -> None:
    sender = _FakeSender()

    last_sent = _handle_manual_signal_key(
        key=ord(" "),
        now_ts=5.0,
        ev3_sender=sender,
        last_manual_signal_ts=float("-inf"),
        manual_signal_cooldown_s=0.35,
    )

    assert sender.sent_ts == [5.0]
    assert last_sent == 5.0


def test_handle_manual_signal_key_ignores_non_space() -> None:
    sender = _FakeSender()

    last_sent = _handle_manual_signal_key(
        key=ord("a"),
        now_ts=5.0,
        ev3_sender=sender,
        last_manual_signal_ts=4.0,
        manual_signal_cooldown_s=0.35,
    )

    assert sender.sent_ts == []
    assert last_sent == 4.0


def test_handle_manual_signal_key_debounce_blocks_rapid_repeat() -> None:
    sender = _FakeSender()
    last_sent = float("-inf")

    last_sent = _handle_manual_signal_key(
        key=ord(" "),
        now_ts=1.0,
        ev3_sender=sender,
        last_manual_signal_ts=last_sent,
        manual_signal_cooldown_s=0.35,
    )
    last_sent = _handle_manual_signal_key(
        key=ord(" "),
        now_ts=1.2,
        ev3_sender=sender,
        last_manual_signal_ts=last_sent,
        manual_signal_cooldown_s=0.35,
    )
    last_sent = _handle_manual_signal_key(
        key=ord(" "),
        now_ts=1.36,
        ev3_sender=sender,
        last_manual_signal_ts=last_sent,
        manual_signal_cooldown_s=0.35,
    )

    assert sender.sent_ts == [1.0, 1.36]
    assert last_sent == 1.36
