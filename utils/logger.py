"""Central structured logging configuration."""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_RESERVED = frozenset(
    logging.LogRecord(
        name="", level=0, pathname="", lineno=0, msg="", args=(),
        exc_info=None
    ).__dict__
)


class JsonFormatter(logging.Formatter):
    """Format log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a log record.

        Args:
            record: Python logging record.

        Returns:
            JSON-formatted log line.
        """
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logger(
    log_path: Path,
    level: str,
    max_bytes: int,
    backup_count: int,
) -> logging.Logger:
    """Create the application logger.

    Args:
        log_path: Rotating JSON log destination.
        level: Logging severity name.
        max_bytes: Maximum size of each log file.
        backup_count: Number of rotated logs to retain.

    Returns:
        Configured application logger.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("traffic_surveillance")
    logger.setLevel(level.upper())
    logger.propagate = False
    logger.handlers.clear()

    formatter = JsonFormatter()
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def shutdown_logging() -> None:
    """Flush and close every logging handler."""
    logging.shutdown()
