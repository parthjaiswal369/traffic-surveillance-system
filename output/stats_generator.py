"""Count visualization and final JSON/CSV persistence."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


class StatsGenerator:
    """Render and save category-wise IN/OUT statistics."""

    _LABELS = {
        "2_wheeler": "2 Wheeler",
        "4_wheeler": "4 Wheeler",
        "heavy_vehicle": "Heavy Vehicle",
    }

    def __init__(
        self,
        json_path: Path,
        csv_path: Path,
        overlay_alpha: float,
    ) -> None:
        self._json_path = json_path
        self._csv_path = csv_path
        self._overlay_alpha = overlay_alpha

    def draw_overlay(
        self,
        frame: NDArray[np.uint8],
        counts: dict[str, dict[str, int]],
    ) -> None:
        """Draw current category counts in place.

        Args:
            frame: Annotated BGR frame.
            counts: Category and direction totals.
        """
        row_height = 28
        panel_width = 350
        panel_height = 48 + row_height * len(counts)
        overlay = frame.copy()
        cv2.rectangle(overlay, (12, 12), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(
            overlay,
            self._overlay_alpha,
            frame,
            1.0 - self._overlay_alpha,
            0.0,
            dst=frame,
        )
        cv2.putText(
            frame,
            "VEHICLE COUNTS",
            (24, 39),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        for index, (category, values) in enumerate(counts.items()):
            label = self._LABELS.get(category, category.replace("_", " ").title())
            text = f"{label:<15} IN {values['in']:>4}   OUT {values['out']:>4}"
            cv2.putText(
                frame,
                text,
                (24, 69 + index * row_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )

    def save(self, counts: dict[str, dict[str, int]]) -> None:
        """Atomically save final JSON and CSV count files.

        Args:
            counts: Final category and direction totals.
        """
        self._json_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)

        json_temp = self._json_path.with_suffix(self._json_path.suffix + ".tmp")
        json_temp.write_text(
            json.dumps(counts, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        json_temp.replace(self._json_path)

        csv_temp = self._csv_path.with_suffix(self._csv_path.suffix + ".tmp")
        with csv_temp.open("w", newline="", encoding="utf-8") as file_handle:
            writer = csv.DictWriter(
                file_handle,
                fieldnames=("category", "in", "out"),
            )
            writer.writeheader()
            for category, values in counts.items():
                writer.writerow(
                    {
                        "category": category,
                        "in": values["in"],
                        "out": values["out"],
                    }
                )
        csv_temp.replace(self._csv_path)
