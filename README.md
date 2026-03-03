# motion_detection

Real-time webcam motion detection with OpenCV MOG2, YOLOv8 person detection, SORT-style tracking, and per-box status indicators.

## Features

- Webcam-only real-time processing at a default `640x480`.
- Person-only YOLOv8 detection (`yolov8n` default).
- Track-level motion states with visual markers:
  - `💀` means active motion in the last `0.5s`.
  - `✅` means recently moving in the last `3.0s` but not currently active.
- Unicode icon rendering with Pillow and graceful ASCII fallback if emoji fonts are unavailable.
- Auto device selection (`CUDA` if available, otherwise `CPU`).

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python -m motion_detection.app
```

Press `q` to quit.

## Common Options

```powershell
python -m motion_detection.app `
  --camera-index 0 `
  --width 640 --height 480 `
  --model-path yolov8n.pt `
  --device auto `
  --conf-thresh 0.35 `
  --iou-thresh 0.45 `
  --min-motion-area 1200 `
  --active-seconds 0.5 `
  --previous-seconds 3.0 `
  --det-every-cpu 3 `
  --det-every-cuda 1 `
  --show-fps
```

## Tuning Notes

- Increase `--min-motion-area` if flicker/noise causes false motion.
- Lower `--det-every-cpu` only if your machine can keep up with real-time inference.
- If CPU performance is low, keep `--det-every-cpu` at `3` or higher to preserve FPS.
- Increase `--previous-seconds` for longer recent-motion persistence.

## Behavior Summary

- A tracked person box is shown when state is `ACTIVE` or `RECENT`.
- `ACTIVE`: motion overlap detected within `active_seconds`.
- `RECENT`: no current overlap, but prior overlap is within `previous_seconds`.
- `INACTIVE`: older than `previous_seconds`; the box is not rendered.

## Testing

If dependencies are installed:

```powershell
python -m pytest -q
```
