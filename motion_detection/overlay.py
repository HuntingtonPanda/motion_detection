"""Frame rendering helpers for track boxes, labels, and status indicators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .types import MotionState, Track


def _try_load_font(size: int) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, bool]:
    candidates = [
        Path("C:/Windows/Fonts/seguiemj.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("seguiemj.ttf"),
        Path("DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(str(candidate), size=size), False
        except OSError:
            continue
    return ImageFont.load_default(), True


@dataclass
class OverlayRenderer:
    font_size: int = 18

    def __post_init__(self) -> None:
        self._font, self._force_ascii = _try_load_font(self.font_size)

    def render(self, frame: np.ndarray, tracks: list[Track], fps: float | None = None) -> np.ndarray:
        output = frame.copy()
        for track in tracks:
            color = (30, 40, 240) if track.state == MotionState.ACTIVE else (20, 180, 80)
            x1, y1, x2, y2 = (int(v) for v in track.bbox_xyxy)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        pil_img = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        for track in tracks:
            x1, y1, _, _ = (int(v) for v in track.bbox_xyxy)
            label_text = f"{track.label} #{track.track_id}"
            draw.text((x1 + 4, max(0, y1 - 22)), label_text, font=self._font, fill=(255, 255, 255))
            marker = self._marker_for(track.state)
            draw.text((x1 + 5, y1 + 4), marker, font=self._font, fill=(255, 255, 255))

        if fps is not None:
            draw.text((8, 8), f"FPS: {fps:.1f}", font=self._font, fill=(255, 220, 90))

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _marker_for(self, state: MotionState) -> str:
        if self._force_ascii:
            return "[ACTIVE]" if state == MotionState.ACTIVE else "[RECENT]"
        if state == MotionState.ACTIVE:
            return "💀"
        return "✅"
