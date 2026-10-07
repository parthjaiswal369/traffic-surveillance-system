"""Signal-aware graceful shutdown support."""

from __future__ import annotations

import logging
import signal
import threading
from types import FrameType


class KillSwitch:
    """Thread-safe shutdown flag controlled by process signals."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger
        self._event = threading.Event()
        self._installed = False

    def install(self) -> None:
        """Install SIGINT and SIGTERM handlers."""
        if self._installed:
            return
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        self._installed = True

    def request_shutdown(self, reason: str) -> None:
        """Set the shutdown flag.

        Args:
            reason: Human-readable shutdown reason.
        """
        if not self._event.is_set():
            self._logger.info(
                "shutdown_requested",
                extra={"event": "shutdown_requested", "reason": reason},
            )
        self._event.set()

    @property
    def requested(self) -> bool:
        """Return whether shutdown was requested."""
        return self._event.is_set()

    def _handle_signal(
        self,
        signum: int,
        frame: FrameType | None,
    ) -> None:
        del frame
        try:
            signal_name = signal.Signals(signum).name
        except ValueError:
            signal_name = str(signum)
        self.request_shutdown(signal_name)
