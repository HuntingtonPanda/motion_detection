"""UDP sender for manual greet signals to an EV3 receiver."""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass, field
from typing import Callable

SocketFactory = Callable[[], socket.socket]


def _default_socket_factory() -> socket.socket:
    return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


@dataclass
class Ev3UdpSender:
    host: str
    port: int
    socket_factory: SocketFactory = _default_socket_factory
    _sock: socket.socket = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._sock = self.socket_factory()

    def close(self) -> None:
        self._sock.close()

    def send_manual_greet(self, now_ts: float) -> bool:
        """Send one manual greet signal."""
        payload = {
            "kind": "manual_greet",
            "timestamp_s": round(float(now_ts), 3),
        }
        data = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        self._sock.sendto(data, (self.host, self.port))
        return True
