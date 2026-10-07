"""Validated OpenCV video input."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Properties required by the processing pipeline."""

    width: int
    height: int
    fps: float
    frame_count: int


class FrameCapture:
    """Context-managed and validated video capture."""

    def __init__(self, path: Path, logger: logging.Logger) -> None:
        self._path = path
        self._logger = logger
        self._capture: cv2.VideoCapture | None = None
        self._metadata: VideoMetadata | None = None

    def open(self) -> VideoMetadata:
        """Open and validate the configured video.

        Returns:
            Source video metadata.

        Raises:
            FileNotFoundError: If the input path does not exist.
            RuntimeError: If OpenCV cannot open or inspect the video.
        """
        if not self._path.is_file():
            raise FileNotFoundError(f"Input video not found: {self._path}")
        capture = cv2.VideoCapture(str(self._path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Unable to open input video: {self._path}")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0:
            capture.release()
            raise RuntimeError("Input video reports invalid dimensions.")
        if not np.isfinite(fps) or fps <= 0.0:
            fps = 30.0
            self._logger.warning(
                "invalid_source_fps_using_fallback",
                extra={"event": "fps_fallback", "fallback_fps": fps},
            )

        self._capture = capture
        self._metadata = VideoMetadata(width, height, fps, frame_count)
        self._logger.info(
            "video_opened",
            extra={
                "event": "video_opened",
                "path": str(self._path),
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frame_count,
            },
        )
        return self._metadata

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]:
        """Read the next valid BGR frame."""
        if self._capture is None:
            raise RuntimeError("FrameCapture must be opened before reading.")
        success, frame = self._capture.read()
        if not success or frame is None or frame.size == 0:
            return False, None
        return True, frame

    @property
    def at_end(self) -> bool:
        """Estimate whether capture reached the declared final frame."""
        if self._capture is None:
            return True
        if self._metadata is None or self._metadata.frame_count <= 0:
            return False
        position = int(self._capture.get(cv2.CAP_PROP_POS_FRAMES))
        return position >= self._metadata.frame_count

    def skip_corrupted_frame(self) -> None:
        """Advance one frame after an isolated decode failure."""
        if self._capture is None:
            return
        position = int(self._capture.get(cv2.CAP_PROP_POS_FRAMES))
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, position + 1)

    def release(self) -> None:
        """Release the native capture handle."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> FrameCapture:
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.release()
