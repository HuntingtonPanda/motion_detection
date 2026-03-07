# motion_detection

Real-time webcam motion detection with OpenCV MOG2, YOLOv8 person detection, SORT-style tracking, and per-box status indicators.

## Features

- Webcam-only real-time processing at a default `640x480`.
- Person-only YOLOv8 detection (`yolov8n` default).
- Always-visible tracked person boxes while the tracker keeps the track alive.
- Track-level motion states with visual markers:
  - `⚠` means active motion in the last `0.5s` (red box).
  - `✅` means not currently moving (`RECENT` or `INACTIVE`, green box).
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
  --motion-iou-thresh 0.03 `
  --min-motion-area 1200 `
  --active-seconds 0.5 `
  --previous-seconds 3.0 `
  --det-every-cpu 3 `
  --det-every-cuda 1 `
  --ev3-host 192.168.0.20 `
  --ev3-port 5005 `
  --show-fps
```

## Send Manual Greet Signal to EV3 (UDP)

If `--ev3-host` is set, press `Space` to send one manual UDP greet signal.
This avoids per-frame network clutter and lets an operator trigger the robot greeting on demand.

Example packet:

```json
{
  "kind": "manual_greet",
  "timestamp_s": 12.345
}
```

Minimal EV3 MicroPython receiver sketch:

```python
import json
import socket

PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", PORT))

while True:
    data, _ = sock.recvfrom(4096)
    packet = json.loads(data.decode("utf-8"))
    if packet.get("kind") == "manual_greet":
        print("Greet signal received")
        # Trigger your turn-and-greet behavior here.
```

## Tuning Notes

- Increase `--min-motion-area` if flicker/noise causes false motion.
- Increase `--motion-iou-thresh` to require more overlap before a track is considered moving.
- Lower `--det-every-cpu` only if your machine can keep up with real-time inference.
- If CPU performance is low, keep `--det-every-cpu` at `3` or higher to preserve FPS.
- Increase `--previous-seconds` for longer recent-motion persistence.

## Behavior Summary

- A tracked person box is always shown while the SORT tracker still has that track.
- `ACTIVE`: motion overlap above `motion_iou_thresh` was detected within `active_seconds` (red box, moving marker).
- `RECENT`: no current overlap, but prior overlap is within `previous_seconds` (green box, checkmark).
- `INACTIVE`: no overlap for longer than `previous_seconds` or no overlap seen yet (green box, checkmark).

## Testing

If dependencies are installed:

```powershell
python -m pytest -q
```
