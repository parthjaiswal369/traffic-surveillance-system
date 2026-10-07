"""Per-frame vehicle detection, tracking, counting, and rendering."""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np
from numpy.typing import NDArray

from core.tracker import TrackedVehicle, VehicleTracker
from logic.line_counter import LineCounter
from logic.roi_filter import RoiFilter
from logic.vehicle_mapper import VehicleMapper
from output.stats_generator import StatsGenerator


class FrameProcessor:
    """Coordinate the complete single-frame processing pipeline."""

    _CATEGORY_COLORS = {
        "2_wheeler": (0, 220, 255),
        "4_wheeler": (60, 220, 60),
        "heavy_vehicle": (30, 80, 255),
    }

    def __init__(
        self,
        tracker: VehicleTracker,
        vehicle_mapper: VehicleMapper,
        roi_filter: RoiFilter,
        line_counter: LineCounter,
        stats_generator: StatsGenerator,
        logger: logging.Logger,
        box_line_thickness: int,
    ) -> None:
        self._tracker = tracker
        self._vehicle_mapper = vehicle_mapper
        self._roi_filter = roi_filter
        self._line_counter = line_counter
        self._stats_generator = stats_generator
        self._logger = logger
        self._box_line_thickness = box_line_thickness

    @property
    def counts(self) -> dict[str, dict[str, int]]:
        """Return current category counts."""
        return self._line_counter.counts

    def process(
        self,
        frame: NDArray[np.uint8],
        frame_index: int,
    ) -> NDArray[np.uint8]:
        """Process and annotate one source frame.

        Args:
            frame: Original BGR frame.
            frame_index: Zero-based frame index.

        Returns:
            Annotated BGR frame.
        """
        started_at = time.perf_counter()
        masked_frame = self._roi_filter.apply_mask(frame)
        tracks = self._tracker.update(masked_frame)
        active_tracks = [
            track for track in tracks if self._roi_filter.contains(track.anchor)
        ]

        annotated = frame.copy()
        self._draw_static_geometry(annotated)
        rendered_tracks = 0
        for track in active_tracks:
            category = self._vehicle_mapper.category_for(track.class_name)
            if category is None:
                continue
            rendered_tracks += 1
            events = self._line_counter.update(
                track_id=track.track_id,
                point=track.anchor,
                category=category,
                frame_index=frame_index,
            )
            for event in events:
                self._logger.info(
                    "vehicle_counted",
                    extra={
                        "event": "vehicle_counted",
                        "frame_index": event.frame_index,
                        "track_id": event.track_id,
                        "category": event.category,
                        "direction": event.direction,
                    },
                )
            self._draw_track(annotated, track, category)

        self._line_counter.prune(frame_index)
        self._stats_generator.draw_overlay(annotated, self.counts)
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        self._logger.info(
            "frame_processed",
            extra={
                "event": "frame_processed",
                "frame_index": frame_index,
                "tracked_objects": len(tracks),
                "active_vehicle_tracks": rendered_tracks,
                "processing_ms": round(elapsed_ms, 3),
            },
        )
        return annotated

    def _draw_static_geometry(self, frame: NDArray[np.uint8]) -> None:
        cv2.polylines(
            frame,
            [self._roi_filter.polygon],
            isClosed=True,
            color=(255, 180, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        self._draw_line(frame, self._line_counter.entry_line, "ENTRY", (0, 255, 0))
        self._draw_line(frame, self._line_counter.exit_line, "EXIT", (0, 80, 255))

    def _draw_track(
        self,
        frame: NDArray[np.uint8],
        track: TrackedVehicle,
        category: str,
    ) -> None:
        color = self._CATEGORY_COLORS.get(category, (255, 255, 255))
        x1, y1, x2, y2 = track.bbox_xyxy
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            self._box_line_thickness,
        )
        label = (
            f"ID {track.track_id} | {category.replace('_', ' ').title()} "
            f"{track.confidence:.2f}"
        )
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1,
        )
        label_top = max(y1 - text_height - baseline - 6, 0)
        cv2.rectangle(
            frame,
            (x1, label_top),
            (x1 + text_width + 8, label_top + text_height + baseline + 6),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 4, label_top + text_height + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        anchor = (int(round(track.anchor[0])), int(round(track.anchor[1])))
        cv2.circle(frame, anchor, 4, color, -1, cv2.LINE_AA)

    @staticmethod
    def _draw_line(
        frame: NDArray[np.uint8],
        line: tuple[tuple[float, float], tuple[float, float]],
        label: str,
        color: tuple[int, int, int],
    ) -> None:
        start = (int(line[0][0]), int(line[0][1]))
        end = (int(line[1][0]), int(line[1][1]))
        cv2.line(frame, start, end, color, 3, cv2.LINE_AA)
        cv2.putText(
            frame,
            label,
            (start[0] + 8, max(start[1] - 9, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
