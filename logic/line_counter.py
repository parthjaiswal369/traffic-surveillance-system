"""Two-line directional vehicle counting with track history."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

Point = tuple[float, float]
Line = tuple[Point, Point]
Direction = Literal["in", "out"]


@dataclass(frozen=True, slots=True)
class CountEvent:
    """A vehicle that completed a directional two-line crossing."""

    track_id: int
    category: str
    direction: Direction
    frame_index: int


@dataclass(slots=True)
class _TrackState:
    previous_point: Point
    last_seen_frame: int
    line_sides: dict[str, int]
    category_votes: Counter[str] = field(default_factory=Counter)
    last_crossed_line: str | None = None


class LineCounter:
    """Count tracks crossing entry and exit segments in sequence."""

    def __init__(
        self,
        entry_line: Line,
        exit_line: Line,
        categories: tuple[str, ...],
        history_ttl_frames: int,
    ) -> None:
        self.entry_line = entry_line
        self.exit_line = exit_line
        self._lines: dict[str, Line] = {
            "entry": entry_line,
            "exit": exit_line,
        }
        self._counts = {
            category: {"in": 0, "out": 0} for category in categories
        }
        self._history_ttl_frames = history_ttl_frames
        self._states: dict[int, _TrackState] = {}
        self._counted_track_ids: set[int] = set()

    @property
    def counts(self) -> dict[str, dict[str, int]]:
        """Return an isolated copy of current counts."""
        return {
            category: dict(directions)
            for category, directions in self._counts.items()
        }

    def update(
        self,
        track_id: int,
        point: Point,
        category: str,
        frame_index: int,
    ) -> list[CountEvent]:
        """Process one tracked vehicle position.

        Args:
            track_id: Persistent ByteTrack identifier.
            point: Current vehicle bottom-center point.
            category: Mapped vehicle category.
            frame_index: Current frame number.

        Returns:
            Zero or one count event.
        """
        state = self._states.get(track_id)
        current_sides = {
            name: self._side(line, point) for name, line in self._lines.items()
        }
        if state is None:
            state = _TrackState(
                previous_point=point,
                last_seen_frame=frame_index,
                line_sides=current_sides,
            )
            state.category_votes[category] += 1
            self._states[track_id] = state
            return []

        state.last_seen_frame = frame_index
        state.category_votes[category] += 1
        if track_id in self._counted_track_ids:
            state.previous_point = point
            self._refresh_sides(state, current_sides)
            return []

        crossings: list[tuple[float, str]] = []
        for line_name, line in self._lines.items():
            previous_side = state.line_sides.get(line_name, 0)
            current_side = current_sides[line_name]
            if (
                previous_side != 0
                and current_side != 0
                and previous_side != current_side
            ):
                parameter = self._intersection_parameter(
                    state.previous_point,
                    point,
                    line[0],
                    line[1],
                )
                if parameter is not None:
                    crossings.append((parameter, line_name))

        events: list[CountEvent] = []
        for _, line_name in sorted(crossings):
            event = self._register_crossing(
                state=state,
                track_id=track_id,
                line_name=line_name,
                frame_index=frame_index,
            )
            if event is not None:
                events.append(event)
                break

        state.previous_point = point
        self._refresh_sides(state, current_sides)
        return events

    def prune(self, frame_index: int) -> None:
        """Remove stale uncounted track state to cap memory use."""
        stale_ids = [
            track_id
            for track_id, state in self._states.items()
            if track_id not in self._counted_track_ids
            and frame_index - state.last_seen_frame > self._history_ttl_frames
        ]
        for track_id in stale_ids:
            del self._states[track_id]

    def _register_crossing(
        self,
        state: _TrackState,
        track_id: int,
        line_name: str,
        frame_index: int,
    ) -> CountEvent | None:
        if state.last_crossed_line == line_name:
            return None

        direction: Direction | None = None
        if state.last_crossed_line == "entry" and line_name == "exit":
            direction = "in"
        elif state.last_crossed_line == "exit" and line_name == "entry":
            direction = "out"

        state.last_crossed_line = line_name
        if direction is None:
            return None

        category = state.category_votes.most_common(1)[0][0]
        self._counts[category][direction] += 1
        self._counted_track_ids.add(track_id)
        return CountEvent(
            track_id=track_id,
            category=category,
            direction=direction,
            frame_index=frame_index,
        )

    @staticmethod
    def _refresh_sides(
        state: _TrackState,
        current_sides: dict[str, int],
    ) -> None:
        for name, side in current_sides.items():
            if side != 0:
                state.line_sides[name] = side

    @staticmethod
    def _side(line: Line, point: Point, epsilon: float = 1e-6) -> int:
        (x1, y1), (x2, y2) = line
        cross_product = (
            (x2 - x1) * (point[1] - y1)
            - (y2 - y1) * (point[0] - x1)
        )
        if abs(cross_product) <= epsilon:
            return 0
        return 1 if cross_product > 0.0 else -1

    @staticmethod
    def _intersection_parameter(
        movement_start: Point,
        movement_end: Point,
        line_start: Point,
        line_end: Point,
        epsilon: float = 1e-9,
    ) -> float | None:
        px, py = movement_start
        rx = movement_end[0] - px
        ry = movement_end[1] - py
        qx, qy = line_start
        sx = line_end[0] - qx
        sy = line_end[1] - qy
        denominator = rx * sy - ry * sx
        if abs(denominator) <= epsilon:
            return None
        q_minus_px = qx - px
        q_minus_py = qy - py
        movement_parameter = (
            q_minus_px * sy - q_minus_py * sx
        ) / denominator
        line_parameter = (
            q_minus_px * ry - q_minus_py * rx
        ) / denominator
        if (
            -epsilon <= movement_parameter <= 1.0 + epsilon
            and -epsilon <= line_parameter <= 1.0 + epsilon
        ):
            return movement_parameter
        return None
