"""CLI entrypoint for the motion detection app."""

from __future__ import annotations

import time

import cv2

from .config import parse_args
from .detector_yolo import YoloDetector
from .motion import MotionClassifier
from .overlay import OverlayRenderer
from .pipeline import MotionPipeline
from .state_machine import TrackStateMachine
from .tracker_sort import SortTracker


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)

    detector = YoloDetector(
        model_path=config.model_path,
        device=config.device,
        conf_thresh=config.conf_thresh,
        iou_thresh=config.iou_thresh,
    )
    motion = MotionClassifier(min_motion_area=config.min_motion_area)
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

    cap = cv2.VideoCapture(config.camera_index)
    if not cap.isOpened():
        print(f"Failed to open webcam at index {config.camera_index}.")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    cap.set(cv2.CAP_PROP_FPS, config.fps_hint)

    prev_ts = time.perf_counter()
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
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
