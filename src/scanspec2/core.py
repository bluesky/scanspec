"""Core data structures for scanspec 2.0."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np

AxisT = TypeVar("AxisT")
DetectorT = TypeVar("DetectorT")
MonitorT = TypeVar("MonitorT")


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


class LinearSource(Generic[AxisT]):
    """Linear position interpolation from axis ranges.

    Each axis maps to ``(start, stop)``.  Setpoints at half-integer indexes
    follow the 1.x fence/post convention.
    """

    def __init__(
        self, axis_ranges: dict[AxisT, tuple[float, float]], length: int
    ) -> None:
        self.axis_ranges = axis_ranges
        self._length = length

    def setpoints(self, indexes: np.ndarray) -> dict[AxisT, np.ndarray]:
        """Evaluate linear positions at *indexes*."""
        result: dict[AxisT, np.ndarray] = {}
        for ax, (start, stop) in self.axis_ranges.items():
            if self._length <= 1:
                step = stop - start
            else:
                step = (stop - start) / (self._length - 1)
            first = start - step / 2
            result[ax] = first + indexes * step
        return result


class FunctionSource(Generic[AxisT]):
    """Arbitrary position function."""

    def __init__(self, fn: Callable[[np.ndarray], dict[AxisT, np.ndarray]]) -> None:
        self._fn = fn

    def setpoints(self, indexes: np.ndarray) -> dict[AxisT, np.ndarray]:
        """Evaluate positions via the wrapped callable."""
        return self._fn(indexes)


class ConcatSource(Generic[AxisT]):
    """Sequential concatenation of child WindowGenerators.

    Delegates setpoint computation to children based on cumulative
    length ranges.
    """

    def __init__(self, children: list[WindowGenerator[AxisT]]) -> None:
        self.children = children

    def setpoints(self, indexes: np.ndarray) -> dict[AxisT, np.ndarray]:
        """Delegate to children based on cumulative length ranges."""
        result: dict[AxisT, np.ndarray] = {}
        cumulative = 0
        for child in self.children:
            child_start = cumulative
            child_end = cumulative + child.length
            mask = (indexes >= child_start) & (indexes < child_end)
            if np.any(mask):
                child_idx = indexes[mask] - child_start
                child_pts = child.setpoints(child_idx)
                for ax, arr in child_pts.items():
                    if ax not in result:
                        result[ax] = np.empty(len(indexes), dtype=np.float64)
                    result[ax][mask] = arr
            cumulative = child_end
        return result


class WindowGenerator(Generic[AxisT]):
    """One dimension of window generation.

    Position computation is delegated to the ``source``:

    - ``LinearSource``: axis_ranges → linear interpolation (fence/post).
    - ``FunctionSource``: arbitrary callable.
    - ``ConcatSource``: delegates to child WindowGenerators.
    """

    def __init__(
        self,
        axes: list[AxisT],
        length: int,
        source: LinearSource[AxisT] | FunctionSource[AxisT] | ConcatSource[AxisT],
        snake: bool = False,
        fly: bool = False,
        trigger_groups: list[TriggerGroup[Any]] | None = None,
        duration: float | None = None,
    ) -> None:
        self.axes = axes
        self.length = length
        self.source = source
        self.snake = snake
        self.fly = fly
        self.trigger_groups: list[TriggerGroup[Any]] = (
            trigger_groups if trigger_groups is not None else []
        )
        self.duration = duration

    @property
    def non_linear(self) -> bool:
        """True when positions use a non-linear function."""
        return not isinstance(self.source, LinearSource)

    def setpoints(self, indexes: np.ndarray) -> dict[AxisT, np.ndarray]:
        """Evaluate positions at *indexes*."""
        return self.source.setpoints(indexes)

    def windows(self, reverse: bool = False) -> Iterator[Window[AxisT, Any]]:
        """Yield collection windows for this generator.

        Does not set ``previous`` — the caller (``Scan.__iter__``) chains
        windows together and computes delta ``static_axes``.

        For step scans: one Window per setpoint.
        For fly scans: one Window covering the whole sweep.
        For concat generators (children): iterates each child sequentially.
        """
        if isinstance(self.source, ConcatSource):
            children = (
                list(reversed(self.source.children))
                if reverse
                else self.source.children
            )
            for child in children:
                yield from child.windows(reverse)
            return

        if self.fly:
            yield from self._fly_window(reverse)
        else:
            yield from self._step_windows(reverse)

    def _step_windows(self, reverse: bool) -> Iterator[Window[AxisT, Any]]:
        """Yield one Window per setpoint."""
        for raw_idx in range(self.length):
            step_idx = (self.length - 1 - raw_idx) if reverse else raw_idx
            result = self.setpoints(np.array([step_idx + 0.5]))
            step_pos = {axis: float(arr[0]) for axis, arr in result.items()}
            yield Window(
                static_axes=step_pos,
                moving_axes={},
                non_linear=False,
                duration=self.duration if self.duration is not None else 0.0,
                trigger_groups=list(self.trigger_groups),
                previous=None,
            )

    def _fly_window(self, reverse: bool) -> Iterator[Window[AxisT, Any]]:
        """Yield a single fly-scan Window."""
        length = self.length
        if reverse:
            start_i, end_i = float(length), 0.0
        else:
            start_i, end_i = 0.0, float(length)

        eps = 1e-6
        moving_axes: dict[AxisT, AxisMotion] = {}

        def _eval(i: float, ax: AxisT) -> float:
            return float(self.setpoints(np.array([i]))[ax][0])

        for axis in self.setpoints(np.array([0.5])):
            s_vel = (_eval(start_i + eps, axis) - _eval(start_i - eps, axis)) / (
                2 * eps
            )
            e_vel = (_eval(end_i + eps, axis) - _eval(end_i - eps, axis)) / (2 * eps)
            moving_axes[axis] = AxisMotion(
                start_position=_eval(start_i, axis),
                start_velocity=s_vel,
                end_position=_eval(end_i, axis),
                end_velocity=e_vel,
            )

        setpoints_fn = self.setpoints
        sign = 1.0 if not reverse else -1.0

        def positions_fn(indexes: np.ndarray) -> dict[AxisT, np.ndarray]:
            return setpoints_fn(start_i + indexes * sign)

        duration = (
            length * self.duration if self.duration is not None else float(length)
        )
        yield Window(
            static_axes={},
            moving_axes=moving_axes,
            non_linear=self.non_linear,
            duration=duration,
            trigger_groups=list(self.trigger_groups),
            previous=None,
            positions_fn=positions_fn,
        )


class Dimension(Generic[AxisT]):
    """One dimension of the compiled scan geometry.

    Produced by Spec.compile(). One entry is created per motion primitive
    (Linspace, Spiral, etc.) in the spec tree; Zip merges two primitives
    into one entry with multiple axes.

    Whether the innermost dimension is flown or stepped is not recorded
    here, since only the innermost dimension can ever be a flyscan axis.

    """

    def __init__(
        self,
        axes: list[AxisT],
        length: int,
        snake: bool,
        position_fn: Callable[[np.ndarray], dict[AxisT, np.ndarray]],
    ) -> None:
        self.axes = axes
        self.length = length
        self.snake = snake
        self._position_fn = position_fn

    def setpoints(
        self,
        axis: AxisT,
        chunk_size: int | None = None,
    ) -> Iterator[np.ndarray]:
        """Yield nominal collection positions in the forward direction.

        Midpoints are at half-integer indexes 0.5, 1.5, ..., length - 0.5
        (the 1.x fence/post convention).

        Only computes the indexes for the current chunk — never allocates
        the full position array and slices it.

        chunk_size=None yields a single chunk covering all points.
        Snaking is NOT applied — dim.snake is provided for the caller.

        Full materialisation: next(dim.setpoints(axis))
        """
        step = chunk_size if chunk_size is not None else self.length
        for start in range(0, self.length, step):
            end = min(start + step, self.length)
            # Half-integer indexes: 0.5, 1.5, ...
            indexes = np.arange(start, end, dtype=float) + 0.5
            yield self._position_fn(indexes)[axis]


class Window(Generic[AxisT, DetectorT]):
    """A contiguous stretch of motion during which detectors are triggered.

    Windows are separated by turnarounds. scanspec provides boundary kinematics
    so the caller can compute the turnaround trajectory externally via
    calculate_turnaround.

    trigger_groups may contain groups for multiple streams. In a multi-stream
    scan a window may contain groups for only a subset of streams (e.g. some
    motion phases trigger only diffraction detectors, others only spectroscopy).
    """

    def __init__(
        self,
        static_axes: dict[AxisT, float],
        moving_axes: dict[AxisT, AxisMotion],
        non_linear: bool,
        duration: float,
        trigger_groups: list[TriggerGroup[DetectorT]],
        previous: Window[AxisT, DetectorT] | None,
        positions_fn: Callable[[np.ndarray], dict[AxisT, np.ndarray]] | None = None,
    ) -> None:
        self.static_axes = static_axes
        self.moving_axes = moving_axes
        self.non_linear = non_linear
        self.duration = duration
        self.trigger_groups = trigger_groups
        self.previous = previous
        self._positions_fn = positions_fn

    def positions(
        self, dt: float, max_duration: float | None = None
    ) -> Iterator[dict[AxisT, np.ndarray]]:
        """Yield chunks of servo-rate positions for the moving axes.

        ``dt`` is the index step (e.g. 1 = one point per collection frame;
        use a smaller value for servo-rate interpolation in future).
        ``max_duration`` limits how many index steps are yielded per chunk
        (None = yield all at once).

        Raises RuntimeError if no position function was provided (step windows).
        """
        if self._positions_fn is None:
            raise RuntimeError(
                "No position function on this window "
                "(step windows have no continuous trajectory)"
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


def _iter_with_outer(
    gens: list[WindowGenerator[AxisT]],
    depth: int,
    parent_flat_idx: int,
) -> Iterator[tuple[Window[AxisT, Any], dict[AxisT, float]]]:
    """Recursively iterate generators, yielding (inner_window, outer_positions).

    Each generator's ``windows(reverse)`` is called with a reversal flag
    derived from its own ``snake`` attribute and the flat iteration-count
    index of all ancestor generators.

    ``parent_flat_idx`` is the iteration counter (not effective position
    index) so that parity is preserved regardless of reversed dimension
    lengths.
    """
    gen = gens[depth]
    reverse = gen.snake and (parent_flat_idx % 2 == 1)

    if depth == len(gens) - 1:
        for window in gen.windows(reverse):
            yield window, {}
    else:
        for i, outer_window in enumerate(gen.windows(reverse)):
            child_flat_idx = parent_flat_idx * gen.length + i
            for inner_window, deeper_outer in _iter_with_outer(
                gens, depth + 1, child_flat_idx
            ):
                merged: dict[AxisT, float] = dict(outer_window.static_axes)
                merged.update(deeper_outer)
                yield inner_window, merged


class Scan(Generic[AxisT, DetectorT, MonitorT]):
    """Compiled output of Spec.compile().

    O(spec complexity) to construct -- no position arrays allocated until
    setpoints() or __iter__ is called.

    Serves as the sole entry point for both window iteration and analysis.

    ``generators`` is the ordered outer -> inner list of WindowGenerator
    objects.  Each generator owns ``windows(reverse)`` which yields
    ``Window`` objects (without ``previous``).  ``Scan.__iter__`` calls
    ``windows()`` recursively, merging outer positions into the innermost
    windows' ``static_axes`` and setting ``previous``.
    """

    def __init__(
        self,
        generators: Sequence[WindowGenerator[AxisT]],
        windowed_streams: Sequence[WindowedStream[AxisT, DetectorT]] = (),
        continuous_streams: Sequence[ContinuousStream[DetectorT]] = (),
        monitors: Sequence[MonitorStream[MonitorT]] = (),
        start_window: int = 0,
        start_time: float = 0.0,
    ) -> None:
        self.generators: list[WindowGenerator[AxisT]] = list(generators)
        self.windowed_streams = list(windowed_streams)
        self.continuous_streams = list(continuous_streams)
        self.monitors = list(monitors)
        self._start_window = start_window
        self._start_time = start_time

    def with_start(
        self, window: int, time: float = 0.0
    ) -> Scan[AxisT, DetectorT, MonitorT]:
        """Return a new Scan that starts iteration at the given window and time.

        Used for pause/resume -- construct a new Scan from a known progress point
        rather than rewinding an existing iterator.
        """
        return Scan(
            generators=self.generators,
            windowed_streams=self.windowed_streams,
            continuous_streams=self.continuous_streams,
            monitors=self.monitors,
            start_window=window,
            start_time=time,
        )

    @property
    def has_moving_axes(self) -> bool:
        """True if any window will have moving_axes (fly generators present)."""
        return any(g.fly for g in self.generators)

    @property
    def non_linear(self) -> bool:
        """True if any fly generator uses a non-linear position function."""
        return any(g.fly and g.non_linear for g in self.generators)

    @staticmethod
    def _changed_axes(
        current: dict[AxisT, float],
        previous: dict[AxisT, float],
    ) -> dict[AxisT, float]:
        """Return only the axes whose positions differ from *previous*."""
        return {
            ax: pos
            for ax, pos in current.items()
            if ax not in previous or previous[ax] != pos
        }

    def __iter__(self) -> Iterator[Window[AxisT, DetectorT]]:
        """Iterate collection windows.

        Outer generators are iterated via ``windows()``, yielding step-scan
        positions that are merged into the innermost windows' ``static_axes``.
        The innermost generator (which may use a ``ConcatSource`` for concat)
        yields the actual collection windows.

        ``previous`` and ``static_axes`` are set by mutating the inner Window
        before yielding it.
        """
        gens = self.generators
        if not gens:
            return

        prev_window: Window[AxisT, DetectorT] | None = None
        prev_all: dict[AxisT, float] = {}
        window_idx = 0

        for inner_window, outer_pos in _iter_with_outer(gens, 0, 0):
            all_pos: dict[AxisT, float] = dict(outer_pos)
            all_pos.update(inner_window.static_axes)

            inner_window.static_axes = self._changed_axes(all_pos, prev_all)
            inner_window.previous = prev_window

            if window_idx >= self._start_window:
                yield inner_window

            prev_all = dict(all_pos)
            for axis, am in inner_window.moving_axes.items():
                prev_all[axis] = am.end_position
            prev_window = inner_window
            window_idx += 1
