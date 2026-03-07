from __future__ import annotations

import json

from motion_detection.ev3_sender import Ev3UdpSender


class _FakeSocket:
    def __init__(self) -> None:
        self.sent: list[tuple[bytes, tuple[str, int]]] = []
        self.closed = False

    def sendto(self, data: bytes, addr: tuple[str, int]) -> int:
        self.sent.append((data, addr))
        return len(data)

    def close(self) -> None:
        self.closed = True


def test_ev3_sender_manual_greet_payload() -> None:
    fake_socket = _FakeSocket()
    sender = Ev3UdpSender(
        host="192.168.0.20",
        port=5005,
        socket_factory=lambda: fake_socket,
    )
    sent = sender.send_manual_greet(now_ts=1.23456)

    assert sent
    assert len(fake_socket.sent) == 1
    payload_bytes, addr = fake_socket.sent[0]
    assert addr == ("192.168.0.20", 5005)

    payload = json.loads(payload_bytes.decode("utf-8"))
    assert payload == {"kind": "manual_greet", "timestamp_s": 1.235}


def test_ev3_sender_manual_greet_sends_every_call() -> None:
    fake_socket = _FakeSocket()
    sender = Ev3UdpSender(
        host="127.0.0.1",
        port=5005,
        socket_factory=lambda: fake_socket,
    )

    assert sender.send_manual_greet(now_ts=0.0)
    assert sender.send_manual_greet(now_ts=0.05)
    assert sender.send_manual_greet(now_ts=0.1)
    assert len(fake_socket.sent) == 3


def test_ev3_sender_close_closes_socket() -> None:
    fake_socket = _FakeSocket()
    sender = Ev3UdpSender(
        host="127.0.0.1",
        port=5005,
        socket_factory=lambda: fake_socket,
    )

    sender.close()

    assert fake_socket.closed
