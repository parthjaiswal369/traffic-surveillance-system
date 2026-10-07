"""Validated and context-managed annotated video output."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


class VideoWriter:
    """Write fixed-size BGR frames to disk."""

    def __init__(
        self,
        path: Path,
        codec: str,
        fps: float,
        frame_size: tuple[int, int],
    ) -> None:
        self._path = path
        self._frame_size = frame_size
        self._writer: cv2.VideoWriter | None = None
        path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(path), fourcc, fps, frame_size)
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"Unable to open output video: {path}")
        self._writer = writer

    def write(self, frame: NDArray[np.uint8]) -> None:
        """Append one annotated frame.

        Args:
            frame: BGR image matching the configured dimensions.

        Raises:
            ValueError: If dimensions do not match the output stream.
        """
        if self._writer is None:
            raise RuntimeError("VideoWriter is closed.")
        height, width = frame.shape[:2]
        if (width, height) != self._frame_size:
            raise ValueError(
                f"Frame size {(width, height)} does not match "
                f"{self._frame_size}."
            )
        self._writer.write(frame)

    def release(self) -> None:
        """Finalize and release the output file."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *args: object) -> None:
        self.release()
