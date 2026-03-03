from __future__ import annotations

import pytest

from motion_detection.config import parse_args


def test_parse_args_motion_iou_thresh_default() -> None:
    config = parse_args([])
    assert config.motion_iou_thresh == 0.03


def test_parse_args_motion_iou_thresh_override() -> None:
    config = parse_args(["--motion-iou-thresh", "0.2"])
    assert config.motion_iou_thresh == 0.2


@pytest.mark.parametrize("value", ["0", "-0.1", "1.1"])
def test_parse_args_motion_iou_thresh_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="--motion-iou-thresh must be > 0 and <= 1"):
        parse_args(["--motion-iou-thresh", value])
