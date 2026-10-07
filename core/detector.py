"""Ultralytics YOLO frame and batch inference wrapper."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO


class Detector:
    """Run detection or stateful tracking inference with YOLO."""

    def __init__(
        self,
        model: YOLO,
        device: str,
        confidence_threshold: float,
        iou_threshold: float,
        class_ids: Sequence[int],
    ) -> None:
        self._model = model
        self._device = device
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        self._class_ids = tuple(class_ids)

    @property
    def class_names(self) -> dict[int, str]:
        """Return model class names as an integer-keyed dictionary."""
        names = self._model.names
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}
        return {index: str(value) for index, value in enumerate(names)}

    def infer_frame(self, frame: NDArray[np.uint8]) -> Any:
        """Run stateless detection on one frame.

        Args:
            frame: BGR image.

        Returns:
            Ultralytics Results object.
        """
        return self._model.predict(
            source=frame,
            conf=self._confidence_threshold,
            iou=self._iou_threshold,
            classes=list(self._class_ids),
            device=self._device,
            verbose=False,
        )[0]

    def infer_batch(
        self,
        frames: Sequence[NDArray[np.uint8]],
    ) -> list[Any]:
        """Run stateless batched detection.

        Args:
            frames: BGR images with compatible dimensions.

        Returns:
            One Ultralytics Results object per frame.
        """
        if not frames:
            return []
        return list(
            self._model.predict(
                source=list(frames),
                conf=self._confidence_threshold,
                iou=self._iou_threshold,
                classes=list(self._class_ids),
                device=self._device,
                verbose=False,
            )
        )

    def track_frame(
        self,
        frame: NDArray[np.uint8],
        tracker_config: str,
    ) -> Any:
        """Run stateful detection and tracking on a consecutive frame.

        Args:
            frame: BGR image from the active video stream.
            tracker_config: Ultralytics tracker YAML name or path.

        Returns:
            Ultralytics Results object with persistent track IDs.
        """
        return self._model.track(
            source=frame,
            persist=True,
            tracker=tracker_config,
            conf=self._confidence_threshold,
            iou=self._iou_threshold,
            classes=list(self._class_ids),
            device=self._device,
            verbose=False,
        )[0]
