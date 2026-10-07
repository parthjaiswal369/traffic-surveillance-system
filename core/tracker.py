"""ByteTrack wrapper and normalized tracking output."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from core.detector import Detector


@dataclass(frozen=True, slots=True)
class TrackedVehicle:
    """Normalized vehicle track emitted by the tracker."""

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]

    @property
    def anchor(self) -> tuple[float, float]:
        """Return the bottom-center point used for ROI and line logic."""
        x1, _, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, float(y2))


class VehicleTracker:
    """Apply ByteTrack and convert results into typed vehicle tracks."""

    _CONFIGS = {"bytetrack": "bytetrack.yaml"}

    def __init__(self, detector: Detector, tracker_type: str) -> None:
        normalized_type = tracker_type.casefold()
        if normalized_type not in self._CONFIGS:
            raise ValueError(f"Unsupported tracker type: {tracker_type}")
        self._detector = detector
        self._tracker_config = self._CONFIGS[normalized_type]

    def update(self, frame: NDArray[np.uint8]) -> list[TrackedVehicle]:
        """Update tracks with the next consecutive frame.

        Args:
            frame: ROI-masked BGR frame.

        Returns:
            Confirmed tracks that contain persistent IDs.
        """
        result = self._detector.track_frame(frame, self._tracker_config)
        boxes = result.boxes
        if boxes is None or len(boxes) == 0 or boxes.id is None:
            return []

        coordinates = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)
        track_ids = boxes.id.detach().cpu().numpy().astype(int)
        class_names = self._detector.class_names

        tracks: list[TrackedVehicle] = []
        for bbox, confidence, class_id, track_id in zip(
            coordinates,
            confidences,
            class_ids,
            track_ids,
            strict=True,
        ):
            x1, y1, x2, y2 = (int(round(value)) for value in bbox.tolist())
            tracks.append(
                TrackedVehicle(
                    track_id=int(track_id),
                    class_id=int(class_id),
                    class_name=class_names[int(class_id)],
                    confidence=float(confidence),
                    bbox_xyxy=(x1, y1, x2, y2),
                )
            )
        return tracks
