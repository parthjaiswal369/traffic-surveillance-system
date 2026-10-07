"""YOLO model loading, device selection, and validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """A validated model and its selected inference device."""

    model: YOLO
    device: str


class ModelLoader:
    """Load and validate an Ultralytics detection model."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def load(self, model_path: str, requested_device: str | None) -> LoadedModel:
        """Load a YOLO checkpoint with GPU-to-CPU fallback.

        Args:
            model_path: Local checkpoint path or Ultralytics model name.
            requested_device: Explicit device or None for auto-selection.

        Returns:
            Loaded and validated model.

        Raises:
            RuntimeError: If model loading or validation fails on all devices.
        """
        device = requested_device or ("0" if torch.cuda.is_available() else "cpu")
        self._logger.info(
            "model_loading",
            extra={
                "event": "model_loading",
                "model_path": model_path,
                "device": device,
            },
        )
        try:
            model = YOLO(model_path, task="detect")
        except Exception as exc:
            raise RuntimeError(f"Unable to load YOLO model: {model_path}") from exc

        try:
            self._validate(model, device)
        except Exception as exc:
            if device == "cpu":
                raise RuntimeError("YOLO model validation failed on CPU.") from exc
            self._logger.error(
                "gpu_validation_failed_falling_back_to_cpu",
                extra={
                    "event": "device_fallback",
                    "failed_device": device,
                    "error": str(exc),
                },
            )
            device = "cpu"
            try:
                self._validate(model, device)
            except Exception as cpu_exc:
                raise RuntimeError(
                    "YOLO model validation failed on GPU and CPU."
                ) from cpu_exc

        self._logger.info(
            "model_ready",
            extra={
                "event": "model_ready",
                "model_path": model_path,
                "device": device,
                "class_count": len(model.names),
            },
        )
        return LoadedModel(model=model, device=device)

    @staticmethod
    def _validate(model: YOLO, device: str) -> None:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        results: list[Any] = model.predict(
            source=image,
            device=device,
            imgsz=64,
            verbose=False,
        )
        if not results or not hasattr(results[0], "boxes"):
            raise RuntimeError("The checkpoint did not return detection results.")
        if not model.names:
            raise RuntimeError("The checkpoint has no class-name metadata.")
