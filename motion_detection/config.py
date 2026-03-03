"""Runtime configuration and CLI parsing."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    camera_index: int = 0
    width: int = 640
    height: int = 480
    fps_hint: int = 30
    model_path: str = "yolov8n.pt"
    device: str = "auto"
    conf_thresh: float = 0.35
    iou_thresh: float = 0.45
    motion_iou_thresh: float = 0.03
    min_motion_area: int = 1200
    active_seconds: float = 0.5
    previous_seconds: float = 3.0
    det_every_cpu: int = 3
    det_every_cuda: int = 1
    show_fps: bool = False
    startup_profile: bool = False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="motion-detection",
        description="Real-time webcam motion detection with YOLOv8 labels.",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps-hint", type=int, default=30)
    parser.add_argument("--model-path", type=str, default="yolov8n.pt")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--conf-thresh", type=float, default=0.35)
    parser.add_argument("--iou-thresh", type=float, default=0.45)
    parser.add_argument("--motion-iou-thresh", type=float, default=0.03)
    parser.add_argument("--min-motion-area", type=int, default=1200)
    parser.add_argument("--active-seconds", type=float, default=0.5)
    parser.add_argument("--previous-seconds", type=float, default=3.0)
    parser.add_argument("--det-every-cpu", type=int, default=3)
    parser.add_argument("--det-every-cuda", type=int, default=1)
    parser.add_argument("--show-fps", action="store_true")
    parser.add_argument("--startup-profile", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> AppConfig:
    args = _build_parser().parse_args(argv)
    if args.active_seconds <= 0:
        raise ValueError("--active-seconds must be > 0")
    if args.previous_seconds <= args.active_seconds:
        raise ValueError("--previous-seconds must be greater than --active-seconds")
    if args.det_every_cpu <= 0 or args.det_every_cuda <= 0:
        raise ValueError("--det-every-* values must be > 0")
    if args.min_motion_area <= 0:
        raise ValueError("--min-motion-area must be > 0")
    if args.motion_iou_thresh <= 0 or args.motion_iou_thresh > 1:
        raise ValueError("--motion-iou-thresh must be > 0 and <= 1")

    return AppConfig(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        fps_hint=args.fps_hint,
        model_path=args.model_path,
        device=args.device,
        conf_thresh=args.conf_thresh,
        iou_thresh=args.iou_thresh,
        motion_iou_thresh=args.motion_iou_thresh,
        min_motion_area=args.min_motion_area,
        active_seconds=args.active_seconds,
        previous_seconds=args.previous_seconds,
        det_every_cpu=args.det_every_cpu,
        det_every_cuda=args.det_every_cuda,
        show_fps=args.show_fps,
        startup_profile=args.startup_profile,
    )
