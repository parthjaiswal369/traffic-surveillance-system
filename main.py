"""Traffic surveillance application entry point."""

from __future__ import annotations

import logging
import sys

from config.global_config import SETTINGS
from core.detector import Detector
from core.model_loader import ModelLoader
from core.tracker import VehicleTracker
from logic.line_counter import LineCounter
from logic.roi_filter import RoiFilter, resolve_points
from logic.vehicle_mapper import VehicleMapper
from output.stats_generator import StatsGenerator
from output.video_writer import VideoWriter
from pipeline.frame_capture import FrameCapture
from pipeline.frame_processor import FrameProcessor
from utils.kill_switch import KillSwitch
from utils.logger import configure_logger, shutdown_logging


def run(logger: logging.Logger, kill_switch: KillSwitch) -> int:
    """Build and run the traffic-surveillance pipeline.

    Args:
        logger: Central application logger.
        kill_switch: Process shutdown controller.

    Returns:
        Process exit code.
    """
    SETTINGS.validate()
    loaded = ModelLoader(logger).load(SETTINGS.model_path, SETTINGS.device)
    mapper = VehicleMapper(SETTINGS.vehicle_class_mapping)
    class_ids = mapper.model_class_ids(loaded.model.names)
    detector = Detector(
        model=loaded.model,
        device=loaded.device,
        confidence_threshold=SETTINGS.confidence_threshold,
        iou_threshold=SETTINGS.iou_threshold,
        class_ids=class_ids,
    )
    tracker = VehicleTracker(detector, SETTINGS.tracker_type)
    stats = StatsGenerator(
        json_path=SETTINGS.stats_json_path,
        csv_path=SETTINGS.stats_csv_path,
        overlay_alpha=SETTINGS.overlay_alpha,
    )

    capture = FrameCapture(SETTINGS.input_video_path, logger)
    writer: VideoWriter | None = None
    processor: FrameProcessor | None = None
    frame_index = 0
    decode_failures = 0

    try:
        metadata = capture.open()
        roi_points = resolve_points(
            SETTINGS.roi_polygon_points,
            metadata.width,
            metadata.height,
        )
        entry_points = resolve_points(
            SETTINGS.entry_line_coords,
            metadata.width,
            metadata.height,
        )
        exit_points = resolve_points(
            SETTINGS.exit_line_coords,
            metadata.width,
            metadata.height,
        )
        roi_filter = RoiFilter(roi_points)
        line_counter = LineCounter(
            entry_line=(entry_points[0], entry_points[1]),
            exit_line=(exit_points[0], exit_points[1]),
            categories=mapper.categories,
            history_ttl_frames=SETTINGS.track_history_ttl_frames,
        )
        processor = FrameProcessor(
            tracker=tracker,
            vehicle_mapper=mapper,
            roi_filter=roi_filter,
            line_counter=line_counter,
            stats_generator=stats,
            logger=logger,
            box_line_thickness=SETTINGS.box_line_thickness,
        )
        output_fps = SETTINGS.fps or metadata.fps
        writer = VideoWriter(
            path=SETTINGS.output_video_path,
            codec=SETTINGS.video_codec,
            fps=output_fps,
            frame_size=(metadata.width, metadata.height),
        )

        while not kill_switch.requested:
            success, frame = capture.read()
            if not success or frame is None:
                if capture.at_end:
                    break
                decode_failures += 1
                logger.error(
                    "corrupted_frame",
                    extra={
                        "event": "corrupted_frame",
                        "frame_index": frame_index,
                        "consecutive_failures": decode_failures,
                    },
                )
                if decode_failures >= SETTINGS.max_corrupted_frame_retries:
                    logger.error(
                        "maximum_decode_failures_reached",
                        extra={
                            "event": "video_decode_aborted",
                            "frame_index": frame_index,
                        },
                    )
                    return 2
                capture.skip_corrupted_frame()
                frame_index += 1
                continue

            decode_failures = 0
            annotated_frame = processor.process(frame, frame_index)
            writer.write(annotated_frame)
            frame_index += 1

        return 0
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        counts = processor.counts if processor is not None else {
            category: {"in": 0, "out": 0} for category in mapper.categories
        }
        stats.save(counts)
        logger.info(
            "pipeline_finished",
            extra={
                "event": "pipeline_finished",
                "processed_frames": frame_index,
                "counts": counts,
                "output_video": str(SETTINGS.output_video_path),
                "stats_json": str(SETTINGS.stats_json_path),
                "stats_csv": str(SETTINGS.stats_csv_path),
                "interrupted": kill_switch.requested,
            },
        )


def main() -> int:
    """Configure process services and execute the application."""
    logger = configure_logger(
        log_path=SETTINGS.log_file_path,
        level=SETTINGS.log_level,
        max_bytes=SETTINGS.log_max_bytes,
        backup_count=SETTINGS.log_backup_count,
    )
    kill_switch = KillSwitch(logger)
    kill_switch.install()
    try:
        return run(logger, kill_switch)
    except KeyboardInterrupt:
        kill_switch.request_shutdown("KeyboardInterrupt")
        return 130
    except Exception:
        logger.exception(
            "fatal_pipeline_error",
            extra={"event": "fatal_pipeline_error"},
        )
        return 1
    finally:
        shutdown_logging()


if __name__ == "__main__":
    sys.exit(main())
