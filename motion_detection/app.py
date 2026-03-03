"""CLI entrypoint for the motion detection app."""

from __future__ import annotations

from .config import parse_args


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    print(
        "Pipeline scaffold initialized. "
        f"Configured webcam index {config.camera_index} at {config.width}x{config.height}."
    )
    print("Run will be fully wired after detector, tracker, and renderer modules are added.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
