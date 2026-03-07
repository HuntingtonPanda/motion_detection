"""CLI entrypoint for the motion detection app."""

from __future__ import annotations

import sys
import time

import cv2

from .config import parse_args
from .detector_yolo import YoloDetector
from .ev3_sender import Ev3UdpSender
from .motion import MotionClassifier
from .overlay import OverlayRenderer
from .pipeline import MotionPipeline
from .state_machine import TrackStateMachine
from .tracker_sort import SortTracker


def _open_capture(camera_index: int) -> cv2.VideoCapture:
    """Open camera capture with a backend that avoids Windows startup hangs."""
    if sys.platform.startswith("win"):
        return cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    return cv2.VideoCapture(camera_index)


def _handle_manual_signal_key(
    *,
    key: int,
    now_ts: float,
    ev3_sender: Ev3UdpSender | None,
    last_manual_signal_ts: float,
    manual_signal_cooldown_s: float,
) -> float:
    if key != ord(" ") or ev3_sender is None:
        return last_manual_signal_ts
    if now_ts - last_manual_signal_ts < manual_signal_cooldown_s:
        return last_manual_signal_ts
    ev3_sender.send_manual_greet(now_ts=now_ts)
    return now_ts


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)

    detector = YoloDetector(
        model_path=config.model_path,
        device=config.device,
        conf_thresh=config.conf_thresh,
        iou_thresh=config.iou_thresh,
    )
    motion = MotionClassifier(
        min_motion_area=config.min_motion_area,
        var_threshold=config.motion_var_threshold,
    )
    track_max_age = max(1, int(config.previous_seconds * max(1, config.fps_hint)))
    tracker = SortTracker(max_age=track_max_age, min_hits=1, iou_threshold=0.3)
    state_machine = TrackStateMachine(
        active_seconds=config.active_seconds,
        previous_seconds=config.previous_seconds,
    )
    pipeline = MotionPipeline(
        config=config,
        motion=motion,
        detector=detector,
        tracker=tracker,
        state_machine=state_machine,
    )
    renderer = OverlayRenderer()
    ev3_sender: Ev3UdpSender | None = None
    if config.ev3_host:
        ev3_sender = Ev3UdpSender(
            host=config.ev3_host,
            port=config.ev3_port,
        )

    cap = _open_capture(config.camera_index)
    if not cap.isOpened():
        print(f"Failed to open webcam at index {config.camera_index}.")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    cap.set(cv2.CAP_PROP_FPS, config.fps_hint)

    prev_ts = time.perf_counter()
    manual_signal_cooldown_s = 0.35
    last_manual_signal_ts = float("-inf")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from webcam.")
                return 1

            now_ts = time.perf_counter()
            frame_result = pipeline.process_frame(frame, now_ts=now_ts)

            delta = max(1e-6, now_ts - prev_ts)
            fps = 1.0 / delta
            prev_ts = now_ts

            output = renderer.render(
                frame=frame,
                tracks=frame_result.tracks,
                fps=fps if config.show_fps else None,
            )
            cv2.imshow("Motion Detection", output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            last_manual_signal_ts = _handle_manual_signal_key(
                key=key,
                now_ts=now_ts,
                ev3_sender=ev3_sender,
                last_manual_signal_ts=last_manual_signal_ts,
                manual_signal_cooldown_s=manual_signal_cooldown_s,
            )
    finally:
        if ev3_sender is not None:
            ev3_sender.close()
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
