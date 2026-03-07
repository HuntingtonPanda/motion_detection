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
    min_motion_area: int = 3600
    motion_var_threshold: float = 16.0
    active_seconds: float = 0.5
    previous_seconds: float = 3.0
    det_every_cpu: int = 3
    det_every_cuda: int = 1
    show_fps: bool = False
    ev3_host: str = ""
    ev3_port: int = 5005


def _build_parser() -> argparse.ArgumentParser:
    defaults = AppConfig()
    parser = argparse.ArgumentParser(
        prog="motion-detection",
        description="Real-time webcam motion detection with YOLOv8 labels.",
    )
    parser.add_argument("--camera-index", type=int, default=defaults.camera_index)
    parser.add_argument("--width", type=int, default=defaults.width)
    parser.add_argument("--height", type=int, default=defaults.height)
    parser.add_argument("--fps-hint", type=int, default=defaults.fps_hint)
    parser.add_argument("--model-path", type=str, default=defaults.model_path)
    parser.add_argument("--device", type=str, default=defaults.device, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--conf-thresh", type=float, default=defaults.conf_thresh)
    parser.add_argument("--iou-thresh", type=float, default=defaults.iou_thresh)
    parser.add_argument("--motion-iou-thresh", type=float, default=defaults.motion_iou_thresh)
    parser.add_argument("--min-motion-area", type=int, default=defaults.min_motion_area)
    parser.add_argument("--motion-var-threshold", type=float, default=defaults.motion_var_threshold)
    parser.add_argument("--active-seconds", type=float, default=defaults.active_seconds)
    parser.add_argument("--previous-seconds", type=float, default=defaults.previous_seconds)
    parser.add_argument("--det-every-cpu", type=int, default=defaults.det_every_cpu)
    parser.add_argument("--det-every-cuda", type=int, default=defaults.det_every_cuda)
    parser.add_argument("--show-fps", action="store_true")
    parser.add_argument("--ev3-host", type=str, default=defaults.ev3_host)
    parser.add_argument("--ev3-port", type=int, default=defaults.ev3_port)
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
    if args.motion_var_threshold <= 0:
        raise ValueError("--motion-var-threshold must be > 0")
    if args.motion_iou_thresh <= 0 or args.motion_iou_thresh > 1:
        raise ValueError("--motion-iou-thresh must be > 0 and <= 1")
    if args.ev3_port <= 0 or args.ev3_port > 65535:
        raise ValueError("--ev3-port must be in 1..65535")

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
        motion_var_threshold=args.motion_var_threshold,
        active_seconds=args.active_seconds,
        previous_seconds=args.previous_seconds,
        det_every_cpu=args.det_every_cpu,
        det_every_cuda=args.det_every_cuda,
        show_fps=args.show_fps,
        ev3_host=args.ev3_host.strip(),
        ev3_port=args.ev3_port,
    )
