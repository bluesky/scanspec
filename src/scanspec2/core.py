"""Core data structures for scanspec 2.0."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np

AxisT = TypeVar("AxisT")
DetectorT = TypeVar("DetectorT")
MonitorT = TypeVar("MonitorT")

# Type alias for position functions.  Within generic classes the concrete AxisT
# replaces Any; at module level Any is used for list[PositionFn] etc.
PositionFn = Callable[[np.ndarray], dict[Any, np.ndarray]]


@dataclass
class TriggerPattern:
    """One entry in a TriggerGroup's trigger sequence.

    repeats:  number of times this pattern repeats within the window.
    livetime: detector exposure time in seconds.
    deadtime: detector readout/gap time in seconds.
    """

    repeats: int
    livetime: float
    deadtime: float


@dataclass
class TriggerGroup(Generic[DetectorT]):
    """Detector triggering description for one group within a collection window.

    A group is a set of detectors sharing an identical trigger sequence.
    A window may contain groups from different streams; consumers identify
    their group by matching against their known detector names.  The set of
    detectors is unique across all groups within a window (enforced at
    Path construction time).

    trigger_patterns uniformly expresses:
      single rate, fixed timing:    [TriggerPattern(500, 0.003, 0.001)]
      multi-rate (10x encoders):    [TriggerPattern(5000, 0.0003, 8e-9)]
      ptychography variable gaps:   [TriggerPattern(1, 0.1, 0.01),
                                     TriggerPattern(1, 0.1, 0.3), ...]

    Baked in from DetectorGroup.livetime/deadtime at Path construction time.
    """

    detectors: list[DetectorT]
    trigger_patterns: list[TriggerPattern]


@dataclass
class AxisMotion:
    """Boundary kinematics for one moving axis within a Window.

    All four values are always present together — it is impossible for an axis
    to have a start_position without a start_velocity (etc.) because the struct
    is the unit of storage.  Axis-set consistency across all motion fields is
    therefore structural: the keys of moving_axes are the sole source of truth.
    """

    start_position: float
    start_velocity: float
    end_position: float
    end_velocity: float


@dataclass
class LinearPositions(Generic[AxisT]):
    """Linear position function: interpolates each axis from start to stop.

    ``axis_ranges[ax] = (start, stop)`` where *start* is the midpoint at
    index 0 and *stop* is the midpoint at index *length* - 1.
    """

    axis_ranges: dict[AxisT, tuple[float, float]]
    length: int

    def __call__(self, indexes: np.ndarray) -> dict[AxisT, np.ndarray]:
        """Evaluate positions at *indexes*."""
        result: dict[AxisT, np.ndarray] = {}
        for ax, (start, stop) in self.axis_ranges.items():
            if self.length <= 1 or start == stop:
                result[ax] = np.full_like(indexes, start, dtype=float)
            else:
                result[ax] = start + indexes * (stop - start) / (self.length - 1)
        return result


class Dimension(Generic[AxisT]):
    """One dimension of the compiled scan geometry.

    Produced by Spec.compile(). One entry is created per motion primitive
    (Linspace, Spiral, etc.) in the spec tree; Zip merges two primitives
    into one entry with multiple axes.

    Whether the innermost dimension is flown or stepped is recorded on
    Scan.fly — not here, since only the innermost dimension can
    ever be a flyscan axis.

    Construct with either *start_positions* + *end_positions* (linear motion,
    ``non_linear=False``) or *position_fn* (arbitrary trajectory,
    ``non_linear=True``).  ``Window.non_linear_move`` is derived from
    ``non_linear`` of the fly dimension.
    """

    def __init__(
        self,
        axes: list[AxisT],
        length: int,
        snake: bool,
        position_fn: (
            Callable[[np.ndarray], dict[AxisT, np.ndarray]] | LinearPositions[AxisT]
        ),
    ) -> None:
        self.axes = axes
        self.length = length
        self.snake = snake
        self.position_fn = position_fn
        self.non_linear = not isinstance(position_fn, LinearPositions)

    def setpoints(
        self,
        axis: AxisT,
        chunk_size: int | None = None,
    ) -> Iterator[np.ndarray]:
        """Yield nominal collection positions in the forward direction.

        Only computes the indexes for the current chunk — never allocates
        the full position array and slices it.

        chunk_size=None yields a single chunk covering all points.
        Snaking is NOT applied — dim.snake is provided for the caller.

        Full materialisation: next(dim.setpoints(axis))
        """
        step = chunk_size if chunk_size is not None else self.length
        for start in range(0, self.length, step):
            end = min(start + step, self.length)
            indexes = np.arange(start, end, dtype=float)
            yield self.position_fn(indexes)[axis]

    def __call__(self, indexes: np.ndarray) -> dict[AxisT, np.ndarray]:
        """Evaluate the position function at the given indexes."""
        return self.position_fn(indexes)


class Window(Generic[AxisT, DetectorT]):
    """A contiguous stretch of motion during which detectors are triggered.

    Windows are separated by turnarounds. scanspec provides boundary kinematics
    so the caller can compute the turnaround trajectory externally via
    calculate_turnaround.

    trigger_groups may contain groups for multiple streams. In a multi-stream
    scan a window may contain groups for only a subset of streams (e.g. some
    motion phases trigger only diffraction detectors, others only spectroscopy).

    Window is a pure data object — all fields are set at creation and never
    mutated.
    """

    def __init__(
        self,
        static_axes: dict[AxisT, float],
        moving_axes: dict[AxisT, AxisMotion],
        non_linear_move: bool,
        duration: float,
        trigger_groups: list[TriggerGroup[DetectorT]],
        previous: Window[AxisT, DetectorT] | None,
        positions_fn: Callable[[np.ndarray], dict[AxisT, np.ndarray]] | None = None,
    ) -> None:
        # Axes that do not move during this window.
        # Move to these positions before starting the window.
        self.static_axes = static_axes

        # Axes that move continuously during this window, with their boundary
        # kinematics.  Empty for step scan windows.
        self.moving_axes = moving_axes

        # True when the trajectory is nonlinear (velocity varies during window).
        self.non_linear_move = non_linear_move

        # Total time for this collection window, in index units.
        self.duration = duration

        # Detector groups active during this window.
        self.trigger_groups = trigger_groups

        # The immediately preceding window, or None for the first window.
        self.previous = previous

        # Private: callable(indexes) -> dict[axis, positions] for moving axes.
        self._positions_fn = positions_fn

    def positions(
        self, dt: float, max_duration: float | None = None
    ) -> Iterator[dict[AxisT, np.ndarray]]:
        """Yield chunks of servo-rate positions for the moving axes.

        ``dt`` is the index step (e.g. 1 = one point per collection frame;
        use a smaller value for servo-rate interpolation in future).
        ``max_duration`` limits how many index steps are yielded per chunk
        (None = yield all at once).

        Raises RuntimeError for step-scan windows (no motion).
        """
        if self._positions_fn is None:
            raise RuntimeError(
                "positions() called on a step-scan window with no motion"
            )
        n_total = int(self.duration / dt) if dt > 0 else 1
        chunk = (
            int(max_duration / dt) if max_duration is not None and dt > 0 else n_total
        )
        start = 0
        while start < n_total:
            end = min(start + chunk, n_total)
            indexes = np.linspace(start * dt, end * dt, end - start)
            yield self._positions_fn(indexes)
            start = end


@dataclass
class DetectorGroup(Generic[DetectorT]):
    """Upfront description of a set of detectors sharing trigger parameters.

    Lives on Acquire.detectors. Used to configure detectors before the scan
    starts. Static livetime/deadtime are baked into the trigger_patterns of
    each TriggerGroup at Path construction time.

    exposures_per_collection: exposures the detector accumulates per collection.
    collections_per_event:    collections that form one event in the stream.
      exposures_per_event = exposures_per_collection * collections_per_event.
    """

    exposures_per_collection: int
    collections_per_event: int
    livetime: float | None  # None means ophyd-async sets it
    deadtime: float | None  # None means ophyd-async sets it
    detectors: list[DetectorT]


@dataclass
class WindowedStream(Generic[AxisT, DetectorT]):
    """One named detector stream within a Scan, aligned to collection windows.

    A stream groups detectors whose trigger rates are integer multiples of
    each other.  Each stream has its own scan dimensions (which may differ
    from other streams' dimensions).  Detectors in different streams have no
    phase lock — only timestamps tie their data together.

    dimensions: ordered outer -> inner scan geometry for this stream.
    detector_groups: all groups within this stream; trigger rates must be
        integer multiples of each other within a stream.
    """

    name: str
    dimensions: list[Dimension[AxisT]]
    detector_groups: list[DetectorGroup[DetectorT]]


@dataclass
class ContinuousStream(Generic[DetectorT]):
    """A continuously-acquired detector stream with no scan dimensions.

    Groups detectors that run at a fixed rate for the whole scan duration,
    not frame-coupled to the motion.  Use for cameras or other triggered
    detectors that share timing but are not indexed against scan positions
    (e.g. front_cam and side_cam both at 20 Hz form one ContinuousStream).

    detector_groups: groups within this continuous stream trigger
        at integer-multiple rates of each other.
    """

    name: str
    detector_groups: list[DetectorGroup[DetectorT]]


@dataclass
class MonitorStream(Generic[MonitorT]):
    """A free-running PV sampled continuously for the scan duration.

    Not frame-coupled to the scan.  Associated with scan data by timestamp
    only.  No timing parameters -- the PV runs at its own rate.
    """

    name: str
    detector: MonitorT


class Scan(Generic[AxisT, DetectorT, MonitorT]):
    """Compiled output of Spec.compile().

    O(spec complexity) to construct -- no position arrays allocated until
    setpoints() or __iter__ is called.

    Serves as the sole entry point for both window iteration and analysis.
    fly applies to the underlying motion trajectory as a whole -- all streams
    share the same motion; fly=True means the innermost motion dimension
    sweeps continuously.

    ``motion_dims`` is the ordered outer -> inner list of Dimension objects;
    each Dimension carries its own position function.
    ``__iter__`` uses only these private Dimensions -- windowed_streams are for
    detector analysis, not iteration.
    """

    def __init__(
        self,
        motion_dims: Sequence[Dimension[AxisT]],
        windowed_streams: Sequence[WindowedStream[AxisT, DetectorT]] = (),
        continuous_streams: Sequence[ContinuousStream[DetectorT]] = (),
        monitors: Sequence[MonitorStream[MonitorT]] = (),
        fly: bool = False,
        start_window: int = 0,
        start_time: float = 0.0,
    ) -> None:
        self._motion_dims: list[Dimension[AxisT]] = list(motion_dims)
        self.windowed_streams = list(windowed_streams)
        self.continuous_streams = list(continuous_streams)
        self.monitors = list(monitors)
        self.fly = fly
        self._start_window = start_window
        self._start_time = start_time

    @property
    def dimensions(self) -> list[Dimension[AxisT]]:
        """Ordered outer -> inner scan geometry."""
        return self._motion_dims

    def with_start(
        self, window: int, time: float = 0.0
    ) -> Scan[AxisT, DetectorT, MonitorT]:
        """Return a new Scan that starts iteration at the given window and time.

        Used for pause/resume -- construct a new Scan from a known progress point
        rather than rewinding an existing iterator.
        """
        return Scan(
            motion_dims=self._motion_dims,
            windowed_streams=self.windowed_streams,
            continuous_streams=self.continuous_streams,
            monitors=self.monitors,
            fly=self.fly,
            start_window=window,
            start_time=time,
        )

    def __iter__(self) -> Iterator[Window[AxisT, DetectorT]]:
        """Iterate collection windows, yielding one Window per collection point.

        For fly=False (step scan): one Window per setpoint combination.
        For fly=True: one Window per outer-dimension combination (the innermost
        dimension sweeps continuously within each Window).

        Uses _dims and _motion_fns directly -- does not read windowed_streams.
        static_axes contains only axes whose positions changed from the previous
        window.
        """
        dims = self._motion_dims

        if not dims:
            return

        if self.fly:
            outer_dims = dims[:-1]
            inner_dim: Dimension[AxisT] | None = dims[-1]
        else:
            outer_dims = dims
            inner_dim = None

        outer_lengths = [d.length for d in outer_dims]
        total_outer = 1
        for ln in outer_lengths:
            total_outer *= ln

        prev_window: Window[AxisT, DetectorT] | None = None
        prev_all_positions: dict[AxisT, float] = {}
        window_idx = 0

        for outer_idx in range(total_outer):
            # Decode outer_idx -> per-dimension indexes (last dim = fastest).
            outer_indexes: list[int] = []
            remainder = outer_idx
            for ln in reversed(outer_lengths):
                outer_indexes.insert(0, remainder % ln)
                remainder //= ln

            # Apply snake direction per outer dimension.
            outer_index_values: list[float] = []
            for dim_i, (dim, idx) in enumerate(
                zip(outer_dims, outer_indexes, strict=False)
            ):
                if dim.snake:
                    right_product = 1
                    for j in range(dim_i + 1, len(outer_dims)):
                        right_product *= outer_dims[j].length
                    if self.fly and inner_dim is not None:
                        right_product *= inner_dim.length
                    reversed_pass = (outer_idx // right_product) % 2 == 1
                    effective_idx = (dim.length - 1 - idx) if reversed_pass else idx
                else:
                    effective_idx = idx
                outer_index_values.append(float(effective_idx))

            # Evaluate outer axis positions at this combination.
            current_positions: dict[AxisT, float] = {}
            for dim, idx_val in zip(outer_dims, outer_index_values, strict=False):
                result = dim(np.array([idx_val]))
                for axis, arr in result.items():
                    current_positions[axis] = float(arr[0])

            if self.fly and inner_dim is not None:
                window = self._make_fly_window(
                    outer_idx,
                    current_positions,
                    prev_all_positions,
                    prev_window,
                    inner_dim,
                )

                if window_idx >= self._start_window:
                    yield window

                prev_all_positions = dict(current_positions)
                for axis, am in window.moving_axes.items():
                    prev_all_positions[axis] = am.end_position
                prev_window = window
                window_idx += 1

            else:
                window = self._make_step_window(
                    current_positions, prev_all_positions, prev_window
                )

                if window_idx >= self._start_window:
                    yield window

                prev_all_positions = dict(current_positions)
                prev_window = window
                window_idx += 1

    def _make_step_window(
        self,
        current_positions: dict[AxisT, float],
        prev_all_positions: dict[AxisT, float],
        prev_window: Window[AxisT, DetectorT] | None,
    ) -> Window[AxisT, DetectorT]:
        """Build a step-scan Window for the given outer setpoint."""
        static_axes: dict[AxisT, float] = {}
        for axis, pos in current_positions.items():
            if axis not in prev_all_positions or prev_all_positions[axis] != pos:
                static_axes[axis] = pos
        return Window(
            static_axes=static_axes,
            moving_axes={},
            non_linear_move=False,
            duration=1.0,
            trigger_groups=[],
            previous=prev_window,
            positions_fn=None,
        )

    def _make_fly_window(
        self,
        outer_idx: int,
        current_positions: dict[AxisT, float],
        prev_all_positions: dict[AxisT, float],
        prev_window: Window[AxisT, DetectorT] | None,
        inner_dim: Dimension[AxisT],
    ) -> Window[AxisT, DetectorT]:
        """Build a fly-scan Window for the given outer combination."""
        reversed_inner = inner_dim.snake and outer_idx % 2 == 1

        inner_length = inner_dim.length
        if reversed_inner:
            start_i = float(inner_length) - 0.5
            end_i = -0.5
        else:
            start_i = -0.5
            end_i = float(inner_length) - 0.5

        eps = 1e-6
        moving_axes: dict[AxisT, AxisMotion] = {}
        for axis in inner_dim(np.array([0.5])):

            def _make_eval(ax: AxisT, fn: PositionFn) -> Callable[[float], float]:
                def _ev(i: float) -> float:
                    return float(fn(np.array([i]))[ax][0])

                return _ev

            ev = _make_eval(axis, inner_dim)
            moving_axes[axis] = AxisMotion(
                start_position=ev(start_i),
                start_velocity=(ev(start_i + eps) - ev(start_i - eps)) / (2 * eps),
                end_position=ev(end_i),
                end_velocity=(ev(end_i + eps) - ev(end_i - eps)) / (2 * eps),
            )

        duration = abs(end_i - start_i)

        static_axes: dict[AxisT, float] = {}
        for axis, pos in current_positions.items():
            if axis not in prev_all_positions or prev_all_positions[axis] != pos:
                static_axes[axis] = pos

        def _make_pos_fn(
            s: float, e: float, fn: Callable[[np.ndarray], dict[AxisT, np.ndarray]]
        ) -> Callable[[np.ndarray], dict[AxisT, np.ndarray]]:
            span = e - s

            def _pfn(indexes: np.ndarray) -> dict[AxisT, np.ndarray]:
                dur = abs(span)
                mapped = (
                    s + indexes * span / dur if dur != 0 else np.full_like(indexes, s)
                )
                return fn(mapped)

            return _pfn

        return Window(
            static_axes=static_axes,
            moving_axes=moving_axes,
            non_linear_move=inner_dim.non_linear,
            duration=duration,
            trigger_groups=[],
            previous=prev_window,
            positions_fn=_make_pos_fn(start_i, end_i, inner_dim),
        )
