"""Core data structures for scanspec 2.0."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Generic, TypeVar

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


@dataclass
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

    # Axes that do not move during this window.
    # Move to these positions before starting the window.
    static_axes: dict[AxisT, float]

    # Axes that move continuously during this window, with their boundary
    # kinematics.  Empty for step scan windows (asserted at Path construction
    # time).  Keys are disjoint from static_axes — enforced at construction.
    moving_axes: dict[AxisT, AxisMotion]

    # True when the trajectory is nonlinear (velocity varies during the window).
    non_linear_move: bool

    # Total time for this collection window, in seconds.
    duration: float

    # Detector groups active during this window.
    trigger_groups: list[TriggerGroup[DetectorT]]

    # The immediately preceding window, or None for the first window.
    previous: Window[AxisT, DetectorT] | None


@dataclass
class ScanDimension(Generic[AxisT]):
    """One dimension of the compiled scan geometry.

    Produced by Spec.compile(). One entry is created per motion primitive
    (Linspace, Spiral, etc.) in the spec tree; Zip merges two primitives
    into one entry with multiple axes.

    Whether the innermost dimension is flown or stepped is recorded on
    Scan.fly — not here, since only the innermost dimension can
    ever be a flyscan axis.
    """

    axes: list[AxisT]
    length: int
    snake: bool

    def setpoints(
        self,
        axis: AxisT,
        chunk_size: int | None = None,
    ) -> Iterator[np.ndarray]:
        """Yield nominal collection positions in the forward direction.

        For fly=True returns midpoints of the continuous sweep.
        Snaking is NOT applied — dim.snake is provided for the caller.
        chunk_size=None yields one array; chunk_size=N yields chunks.

        Full materialisation: next(dim.setpoints(axis))
        """
        raise NotImplementedError


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

    dimensions: ordered outer → inner scan geometry for this stream.
    detector_groups: all groups within this stream; trigger rates must be
        integer multiples of each other within a stream.
    """

    name: str
    dimensions: list[ScanDimension[AxisT]]
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
    only.  No timing parameters — the PV runs at its own rate.
    """

    name: str
    detector: MonitorT


@dataclass
class Scan(Generic[AxisT, DetectorT, MonitorT]):
    """Compiled output of Spec.compile().

    O(spec complexity) to construct — no position arrays allocated until
    setpoints() or path.positions() is called.

    Serves as the sole input to Path and the sole entry point for analysis.
    fly applies to the underlying motion trajectory as a whole — all streams
    share the same motion; fly=True means the innermost motion dimension
    sweeps continuously.
    """

    # One or more named window-aligned detector streams, each with its own dimensions.
    windowed_streams: list[WindowedStream[AxisT, DetectorT]]

    # Groups of continuously-acquired detectors sharing timing (no scan dims).
    continuous_streams: list[ContinuousStream[DetectorT]]

    # Free-running PV monitors — no timing parameters.
    monitors: list[MonitorStream[MonitorT]]

    # True when the innermost motion dimension sweeps continuously (flyscan).
    fly: bool
