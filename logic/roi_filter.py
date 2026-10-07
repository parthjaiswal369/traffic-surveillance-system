"""Polygon-based active-side masking and point filtering."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

Point = tuple[float, float]


def resolve_points(
    points: Sequence[Point],
    width: int,
    height: int,
) -> tuple[tuple[int, int], ...]:
    """Convert normalized or absolute points to bounded pixels.

    Args:
        points: Normalized points when both coordinates are in [0, 1],
            otherwise absolute pixel points.
        width: Frame width.
        height: Frame height.

    Returns:
        Pixel coordinates clipped to the frame.
    """
    resolved: list[tuple[int, int]] = []
    for x_value, y_value in points:
        normalized = 0.0 <= x_value <= 1.0 and 0.0 <= y_value <= 1.0
        x_pixel = round(x_value * (width - 1)) if normalized else round(x_value)
        y_pixel = (
            round(y_value * (height - 1)) if normalized else round(y_value)
        )
        resolved.append(
            (
                min(max(int(x_pixel), 0), width - 1),
                min(max(int(y_pixel), 0), height - 1),
            )
        )
    return tuple(resolved)


class RoiFilter:
    """Mask and filter detections against an active-side polygon."""

    def __init__(self, polygon_points: Sequence[tuple[int, int]]) -> None:
        if len(polygon_points) < 3:
            raise ValueError("ROI polygon requires at least three points.")
        self._polygon = np.asarray(polygon_points, dtype=np.int32)
        self._mask_cache: dict[tuple[int, int], NDArray[np.uint8]] = {}

    @property
    def polygon(self) -> NDArray[np.int32]:
        """Return a copy of the drawable ROI polygon."""
        return self._polygon.copy()

    def contains(self, point: tuple[float, float]) -> bool:
        """Return whether a point is inside or on the polygon."""
        return cv2.pointPolygonTest(self._polygon, point, False) >= 0.0

    def apply_mask(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Black out pixels outside the active-side polygon.

        Args:
            frame: Source BGR image.

        Returns:
            ROI-masked image.
        """
        height, width = frame.shape[:2]
        key = (height, width)
        mask = self._mask_cache.get(key)
        if mask is None:
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [self._polygon], 255)
            self._mask_cache[key] = mask
        return cv2.bitwise_and(frame, frame, mask=mask)
