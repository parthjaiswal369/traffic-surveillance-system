"""Application-wide immutable configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Mapping, TypeAlias

Point: TypeAlias = tuple[float, float]
Line: TypeAlias = tuple[Point, Point]

MODEL_PATH: Final[str] = os.getenv("TS_MODEL_PATH", "yolo11n.pt")
INPUT_VIDEO_PATH: Final[Path] = Path(
    os.getenv("TS_INPUT_VIDEO", "/content/input.mp4")
)
OUTPUT_VIDEO_PATH: Final[Path] = Path(
    os.getenv("TS_OUTPUT_VIDEO", "/content/output/traffic_annotated.mp4")
)
CONFIDENCE_THRESHOLD: Final[float] = float(
    os.getenv("TS_CONFIDENCE_THRESHOLD", "0.25")
)
IOU_THRESHOLD: Final[float] = float(os.getenv("TS_IOU_THRESHOLD", "0.70"))
FPS: Final[float | None] = (
    float(os.environ["TS_FPS"]) if os.getenv("TS_FPS") else None
)
TRACKER_TYPE: Final[str] = os.getenv("TS_TRACKER_TYPE", "bytetrack")

# Coordinates in [0.0, 1.0] are normalized and scale to the video size.
# Values greater than 1.0 are interpreted as absolute pixel coordinates.
ROI_POLYGON_POINTS: Final[tuple[Point, ...]] = (
    (0.02, 0.48),
    (0.98, 0.48),
    (0.98, 0.98),
    (0.02, 0.98),
)
ENTRY_LINE_COORDS: Final[Line] = ((0.04, 0.62), (0.96, 0.62))
EXIT_LINE_COORDS: Final[Line] = ((0.04, 0.82), (0.96, 0.82))

VEHICLE_CLASS_MAPPING: Final[Mapping[str, str]] = {
    "bicycle": "2_wheeler",
    "bike": "2_wheeler",
    "motorbike": "2_wheeler",
    "motorcycle": "2_wheeler",
    "scooter": "2_wheeler",
    "car": "4_wheeler",
    "auto": "4_wheeler",
    "autorickshaw": "4_wheeler",
    "auto rickshaw": "4_wheeler",
    "rickshaw": "4_wheeler",
    "mini tempo": "4_wheeler",
    "tempo": "4_wheeler",
    "truck": "heavy_vehicle",
    "bus": "heavy_vehicle",
}
LOG_FILE_PATH: Final[Path] = Path(
    os.getenv("TS_LOG_FILE", "/content/output/traffic_surveillance.log")
)

STATS_JSON_PATH: Final[Path] = Path(
    os.getenv("TS_STATS_JSON", "/content/output/vehicle_counts.json")
)
STATS_CSV_PATH: Final[Path] = Path(
    os.getenv("TS_STATS_CSV", "/content/output/vehicle_counts.csv")
)
DEVICE: Final[str | None] = os.getenv("TS_DEVICE") or None
VIDEO_CODEC: Final[str] = os.getenv("TS_VIDEO_CODEC", "mp4v")
TRACK_HISTORY_TTL_FRAMES: Final[int] = int(
    os.getenv("TS_TRACK_HISTORY_TTL_FRAMES", "300")
)
MAX_CORRUPTED_FRAME_RETRIES: Final[int] = int(
    os.getenv("TS_MAX_CORRUPTED_FRAME_RETRIES", "5")
)
LOG_LEVEL: Final[str] = os.getenv("TS_LOG_LEVEL", "INFO")
LOG_MAX_BYTES: Final[int] = int(os.getenv("TS_LOG_MAX_BYTES", "10485760"))
LOG_BACKUP_COUNT: Final[int] = int(os.getenv("TS_LOG_BACKUP_COUNT", "3"))
BOX_LINE_THICKNESS: Final[int] = 2
OVERLAY_ALPHA: Final[float] = 0.72


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Immutable runtime settings."""

    model_path: str = MODEL_PATH
    input_video_path: Path = INPUT_VIDEO_PATH
    output_video_path: Path = OUTPUT_VIDEO_PATH
    confidence_threshold: float = CONFIDENCE_THRESHOLD
    iou_threshold: float = IOU_THRESHOLD
    fps: float | None = FPS
    tracker_type: str = TRACKER_TYPE
    roi_polygon_points: tuple[Point, ...] = ROI_POLYGON_POINTS
    entry_line_coords: Line = ENTRY_LINE_COORDS
    exit_line_coords: Line = EXIT_LINE_COORDS
    vehicle_class_mapping: Mapping[str, str] = field(
        default_factory=lambda: dict(VEHICLE_CLASS_MAPPING)
    )
    log_file_path: Path = LOG_FILE_PATH
    stats_json_path: Path = STATS_JSON_PATH
    stats_csv_path: Path = STATS_CSV_PATH
    device: str | None = DEVICE
    video_codec: str = VIDEO_CODEC
    track_history_ttl_frames: int = TRACK_HISTORY_TTL_FRAMES
    max_corrupted_frame_retries: int = MAX_CORRUPTED_FRAME_RETRIES
    log_level: str = LOG_LEVEL
    log_max_bytes: int = LOG_MAX_BYTES
    log_backup_count: int = LOG_BACKUP_COUNT
    box_line_thickness: int = BOX_LINE_THICKNESS
    overlay_alpha: float = OVERLAY_ALPHA

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If a setting is invalid.
        """
        if not 0.0 < self.confidence_threshold <= 1.0:
            raise ValueError("CONFIDENCE_THRESHOLD must be in (0, 1].")
        if not 0.0 < self.iou_threshold <= 1.0:
            raise ValueError("IOU_THRESHOLD must be in (0, 1].")
        if self.fps is not None and self.fps <= 0.0:
            raise ValueError("FPS must be positive when specified.")
        if self.tracker_type.casefold() != "bytetrack":
            raise ValueError("TRACKER_TYPE must be 'bytetrack'.")
        if len(self.roi_polygon_points) < 3:
            raise ValueError("ROI_POLYGON_POINTS needs at least three points.")
        if len(self.video_codec) != 4:
            raise ValueError("VIDEO_CODEC must contain exactly four characters.")
        if self.track_history_ttl_frames < 1:
            raise ValueError("TRACK_HISTORY_TTL_FRAMES must be positive.")


SETTINGS: Final[AppConfig] = AppConfig()
