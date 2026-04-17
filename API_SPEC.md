# Scanspec 2.0 API Specification

This document specifies the scanspec 2.0 public API from the perspective of
application-level code (ophyd-async) that consumes a `Spec` instance. It is
written as annotated code examples. Construction of `Spec` objects is covered
in a placeholder section at the end.

---

## Type parameters

Three type parameters are used throughout:

- `AxisT` — the type used to identify axes (typically `str`; could be an enum
  or a device object). Must be hashable (used as dict key in `static_axes` and
  `moving_axes`).
- `DetectorT` — the type used to identify detectors (typically `str`; will be
  genericised to a device object later).
- `MonitorT` — the type used to identify continuously-monitored detectors.
  Only appears on `Spec` — never on `Path` or `Window`.

`Spec[AxisT, DetectorT, MonitorT]` — base class for all scan specs.
`Acquire[AxisT, DetectorT, MonitorT]` — concrete `Spec` subclass: wraps a motion spec + produces a single stream.
`Scan[AxisT, DetectorT, MonitorT]` — compiled output of `spec.compile()`; feeds both `Path` and analysis.
`WindowedStream[AxisT, DetectorT]` — one named detector stream within a `Scan`: dimensions + detector groups.
`ContinuousStream[DetectorT]` — constant-rate detector stream with no scan dimensions (e.g. cameras at 20 Hz).
`MonitorStream[MonitorT]` — on-change PV monitor; no timing parameters.
`Path[AxisT, DetectorT]` — motion-centric stateful iterator; constructed from `Scan`.
`Window[AxisT, DetectorT]` — pure data object yielded by `Path`; trigger groups may span multiple streams.
`ScanDimension[AxisT]` — one dimension of the compiled scan geometry.

---

## Data structures

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Generic, Iterator, TypeVar
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
    end_position:   float
    end_velocity:   float


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

    # True when the trajectory is nonlinear (velocity varies during the
    # window).  False for step scan windows (moving_axes is empty) and for
    # constant-velocity windows (start_velocity == end_velocity for every
    # axis).  Computed analytically at Path construction time from the Spec's
    # position functions — no floating-point comparison involved.
    non_linear_move: bool

    # Total time for this collection window, in seconds.
    # Equals sum(p.repeats * (p.livetime + p.deadtime) for all trigger patterns).
    # Stored directly rather than recomputed from trigger_groups each time it
    # is needed (e.g. motor velocity derivation in a single-axis flyscan).
    duration: float

    # Detector groups active during this window.
    # frozenset(group.detectors) is the unique lookup key per group.
    trigger_groups: list[TriggerGroup[DetectorT]]

    # The immediately preceding window, or None for the first window.
    # Never more than one step back.
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
        ...


@dataclass
class DetectorGroup(Generic[DetectorT]):
    """Upfront description of a set of detectors sharing trigger parameters.

    Lives on Acquire.detectors. Used to configure detectors before the scan
    starts. Static livetime/deadtime are baked into the trigger_patterns of
    each TriggerGroup at Path construction time.

    exposures_per_collection: exposures the detector accumulates per collection.
    collections_per_event: collections that form one event in the stream.
      exposures_per_event = exposures_per_collection * collections_per_event.
    """
    exposures_per_collection: int
    collections_per_event: int
    livetime: float | None    # None means ophyd-async sets it
    deadtime: float | None    # None means ophyd-async sets it
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
    # Acquire always produces exactly one stream.
    windowed_streams: list[WindowedStream[AxisT, DetectorT]]

    # Groups of continuously-acquired detectors sharing timing (no scan dims).
    # E.g. front_cam + side_cam at 20 Hz as one ContinuousStream.
    continuous_streams: list[ContinuousStream[DetectorT]]

    # Free-running PV monitors — no timing parameters.
    # E.g. temperature readbacks, beam current.
    monitors: list[MonitorStream[MonitorT]]

    # True when the innermost motion dimension sweeps continuously (flyscan).
    # False for software step scans.  Applies to the motion as a whole.
    fly: bool
```

---

## `Spec[AxisT, DetectorT, MonitorT]` — scan spec base class

`Spec` is the base class for all scan specs. Calling `spec.compile()`
compiles it into a `Scan` used by both `Path` and analysis.
`Acquire` is the concrete subclass for single-stream scans (see Construction).

```python
spec: Spec[str, str, str]  # provided by orchestrator — typically an Acquire

# Compile once — O(spec complexity), no position arrays allocated.
scan: Scan[str, str, str] = spec.compile()

# Configure triggered detectors before the scan starts.
for stream in scan.windowed_streams:       # list[WindowedStream[str, str]]
    for group in stream.detector_groups:   # list[DetectorGroup[str]]
        setup_detectors(
            group.detectors,               # list[str]
            group.livetime,                # float | None
            group.deadtime,                # float | None
            group.exposures_per_collection,
            group.collections_per_event,
        )

# Configure grouped continuously-acquired detectors (cameras etc.) before scan.
for cs in scan.continuous_streams:         # list[ContinuousStream[str]]
    for group in cs.detector_groups:       # list[DetectorGroup[str]]
        start_continuous_stream(
            cs.name,
            group.detectors,               # list[str]
            group.livetime,                # float | None
            group.deadtime,                # float | None
        )

# Configure free-running PV monitors — no timing parameters.
for m in scan.monitors:                    # list[MonitorStream[str]]
    start_monitor(
        m.name,
        m.detector,                        # str
    )

# scan.windowed_streams: list[WindowedStream[str, str]] — each stream has its own dimensions.
# scan.fly: bool — True if innermost motion dimension is a flyscan axis.
# Used by both Path and analysis — see sections below.
```
---

## `Path[AxisT, DetectorT]` — stateful scan iterator

`Path` is motion-centric: it iterates the underlying trajectory regardless of
how many streams the spec describes. It owns all iteration state: current
window index and time cursor within that window. `MonitorT` does not appear —
monitors are not consumed via `Path`.

Windows yielded by `Path` carry `trigger_groups` for all streams active in
that window.  Consumers identify their group by matching `group.detectors`
against their own known detector list.

`dt` and other consumption parameters are supplied to method calls, not to the
constructor, because the creator of `Path` does not know what will consume it.

```python
scan: Scan[str, str, str] = spec.compile()

# Normal start
path: Path[str, str] = Path(scan)

# Resume after pause — constructs from a known (window_index, time) point
path = Path(scan, start_window=3, start_time=1.4)
```

`path.positions(dt, max_duration)` is only valid inside a `for window in path`
loop — raises `RuntimeError` if called outside one.

---

## Consumption use cases

### 1. Software step scan

Each window is a single point. `moving_axes` is always empty since `scan.fly`
is `False` (asserted at Path construction time). Detector triggering info comes
from `window.trigger_groups`, with `trigger_patterns` baked in from each
stream's `detector_groups` at Path construction time giving
`[TriggerPattern(1, livetime, deadtime)]` per group.

```python
async def run_step_scan(spec: Spec[str, str, str]) -> None:
    scan = spec.compile()
    path: Path[str, str] = Path(scan)

    for window in path:
        # window: Window[str, str]
        assert not window.moving_axes, (
            "step scan windows must have no motion (scan.fly is False)"
        )
        await move(window.static_axes)   # dict[str, float]

        async with asyncio.TaskGroup() as tg:
            for group in window.trigger_groups:
                for pattern in group.trigger_patterns:
                    tg.create_task(
                        trigger_detectors(group.detectors, pattern.livetime, pattern.deadtime)
                    )
```

### 2. Flyscan — PandA sequence table

One sequence table per collection window. The orchestrator supplies `path`,
the exact set of `detector_names` this PandA sequence handles, the trigger
type, and motor position outputs — all hardware configuration, not from spec.

```python
def find_detector_group(
    window: Window[AxisT, DetectorT],
    detector_names: list[DetectorT],
) -> TriggerGroup[DetectorT]:
    """Return the TriggerGroup whose detectors exactly match detector_names.

    Uses frozenset equality — DetectorT must be hashable.
    Raises ValueError if no exact match found.
    """
    key = frozenset(detector_names)
    for group in window.trigger_groups:
        if frozenset(group.detectors) == key:
            return group
    raise ValueError(f"No detector group found for {detector_names}")


async def run_panda_flyscan(
    panda,
    path: Path[str, str],
    detector_names: list[str],
    trigger: SeqTrigger,
    motor_pos_outs: dict[str, PosOut],
) -> None:
    for window in path:
        # window: Window[str, str]
        group = find_detector_group(window, detector_names)

        rows = SeqTable.empty()

        # GPIO gate: low then high at window start
        rows += SeqTable.row(trigger=SeqTrigger.BITA_0)
        rows += SeqTable.row(trigger=SeqTrigger.BITA_1)

        # Optional position-compare row. Axis selected by fastest velocity
        # in encoder counts: window.moving_axes[axis].start_velocity / scale.
        if trigger == SeqTrigger.POSA_GT:
            axis, lower = pick_compare_axis(
                {a: m.start_velocity for a, m in window.moving_axes.items()},
                motor_pos_outs,
            )
            rows += SeqTable.row(trigger=trigger, position=int(lower))

        # One TriggerPattern per entry — handles single-rate,
        # multi-rate, and ptychography variable-gap cases uniformly.
        for pattern in group.trigger_patterns:
            rows += SeqTable.row(
                repeats=pattern.repeats,
                trigger=SeqTrigger.IMMEDIATE,
                time1=int(pattern.livetime * 1e6),
                time2=int(pattern.deadtime * 1e6),
                outa1=True,
                outa2=False,
            )

        await panda.seq.table.set(rows)
        await panda.wait_for_completion()
```

### 3. Flyscan — Motor record

For a single-axis constant-velocity scan driven by an EPICS motor record.
`window.non_linear_move` must be `False` and exactly one axis may be in
`window.moving_axes`.

```python
async def run_motor_record_window(
    motor: Motor,
    window: Window[str, str],
) -> None:
    """Execute one linear collection window on a single motor record.

    Raises ValueError if the window is nonlinear or has more than one moving
    axis.  Raises MotorLimitsError if the required velocity exceeds the motor's
    maximum speed, or if the ramp trajectory falls outside its travel limits.
    """
    if window.non_linear_move:
        raise ValueError(
            "Motor record flyscan requires a linear window "
            "(non_linear_move must be False)"
        )
    if len(window.moving_axes) != 1:
        raise ValueError(
            f"Motor record flyscan requires exactly one moving axis, "
            f"got {list(window.moving_axes)}"
        )

    axis, motion = next(iter(window.moving_axes.items()))

    # velocity = distance / time; window.duration is the collection time
    # excluding motor ramps (start and end position are where collection begins
    # and ends, not where the motor starts and stops moving).
    velocity = (motion.end_position - motion.start_position) / window.duration

    max_speed = await motor.max_velocity.get_value()
    if abs(velocity) > max_speed:
        egu = await motor.motor_egu.get_value()
        raise MotorLimitsError(
            f"Required scan velocity {abs(velocity):.4g} {egu}/s exceeds "
            f"motor max speed {max_speed} {egu}/s"
        )

    acceleration_time = await motor.acceleration_time.get_value()

    # Extend trajectory by half-ramp on each side so the motor is already
    # at full velocity when it reaches start_position.
    ramp_up_start   = motion.start_position - acceleration_time * velocity / 2
    ramp_down_end   = motion.end_position   + acceleration_time * velocity / 2

    # Check both ramp positions against motor soft limits.
    await motor.check_motor_limit(ramp_up_start, ramp_down_end)

    # Move to the run-up start at maximum speed, then set the scan velocity.
    await motor.velocity.set(abs(max_speed))
    await motor.set(ramp_up_start)
    await motor.velocity.set(abs(velocity))

    # Execute the constant-velocity sweep (including run-down ramp).
    # Timeout = collection time + both ramps + 10 s margin.
    timeout = window.duration + acceleration_time + 10
    await motor.set(ramp_down_end, timeout=timeout)
```

### 4. Flyscan — PMAC trajectory

Positions at servo cycle rate (e.g. 0.2ms) consumed in chunks of up to 10s.
`path.positions(dt, max_duration)` yields `dict[AxisT, np.ndarray]`, one array
per moving axis, implicitly for the current iteration window. Between windows
the caller drives the turnaround using boundary kinematics from adjacent windows.

```python
async def run_pmac_flyscan(
    pmac,
    path: Path[str, str],
) -> None:
    for window in path:
        # window: Window[str, str]

        # Turnaround from previous window into this one.
        if window.previous is not None:
            prev = window.previous.moving_axes
            curr = window.moving_axes
            bridge = calculate_turnaround(
                {a: m.end_position   for a, m in prev.items()},
                {a: m.end_velocity   for a, m in prev.items()},
                {a: m.start_position for a, m in curr.items()},
                {a: m.start_velocity for a, m in curr.items()},
            )
            await pmac.send_positions(bridge)

        # Consume this window in time-sliced chunks.
        # Yields early if the window ends before max_duration is reached.
        # Raises RuntimeError if called outside a for-window loop.
        for arrays in path.positions(dt=0.0002, max_duration=10.0):
            # arrays: dict[str, np.ndarray] — one entry per moving axis
            await pmac.send_positions(arrays)
```

### 5. Pause and resume

On pause the PandA completes the current window's triggers and reports progress
as `(window_index, time_within_window)`. Resume constructs a new `Path` from
that point. No rewind method exists on `Path`.

Both PandA and PMAC resume from the same `(window_index, time_within_window)`
point.  Each requires its own `Path` object because `Path` is a stateful
iterator.

```python
async def resume_after_pause(
    panda,
    scan: Scan[str, str, str],
) -> tuple[Path[str, str], Path[str, str]]:
    window_index       = await panda.current_window_index()   # int
    time_within_window = await panda.time_within_window()     # float

    panda_path = Path(scan, start_window=window_index, start_time=time_within_window)
    pmac_path  = Path(scan, start_window=window_index, start_time=time_within_window)
    return panda_path, pmac_path
    # Pass each to the respective run_*_flyscan as normal.
```

---

## Analysis — reshaping detector data

`spec.compile()` is the sole entry point for analysis. `scan.windowed_streams` gives
all window-aligned detector streams; each stream has its own `dimensions` and `detector_groups`.

```python
scan: Scan[str, str, str] = spec.compile()

# Analysis is per stream — each stream has its own dimensions.
for stream in scan.windowed_streams:
    # Base scan shape for this stream — ordered outer → inner.
    base_shape = [dim.length for dim in stream.dimensions]

    # Reshape each detector's frame stack into the scan grid.
    for group in stream.detector_groups:
        # Groups running faster than the base rate have an extra inner dimension.
        if group.collections_per_event > 1:
            shape = base_shape + [group.collections_per_event]
        else:
            shape = base_shape

        for detector in group.detectors:
            data = detector_frames[detector].reshape(shape)

            # De-snake: alternate rows were collected in reverse; flip them back.
            for i, dim in enumerate(stream.dimensions):
                if dim.snake:
                    slices = [slice(None)] * len(shape)
                    slices[i] = slice(1, None, 2)
                    data[tuple(slices)] = np.flip(data[tuple(slices)], axis=i)

    # Axis setpoint coordinates — full materialisation.
    for dim in stream.dimensions:
        for axis in dim.axes:                          # axis: str
            coords[axis] = next(dim.setpoints(axis))   # np.ndarray

    # Axis setpoint coordinates — chunked (e.g. for incremental file writing).
    for dim in stream.dimensions:
        for axis in dim.axes:
            for chunk in dim.setpoints(axis, chunk_size=1000):
                writer.append(axis, chunk)

# Example: 2D grid flyscan (Acquire with single stream "primary")
# scan.fly == True
# scan.windowed_streams[0].name == "primary"
# scan.windowed_streams[0].dimensions == [
#     ScanDimension(axes=["y"], length=50,  snake=False),
#     ScanDimension(axes=["x"], length=100, snake=True),
# ]
# DetectorGroup(["saxs", "waxs"]):                 collections_per_event=1  -> shape (50, 100)
# DetectorGroup(["timestamp", "x_enc", "y_enc"]):  collections_per_event=10 -> shape (50, 100, 10)

# Example: spiral scan — x and y share one dimension
# scan.windowed_streams[0].dimensions == [ScanDimension(axes=["x", "y"], length=5000, snake=False)]
x_coords = next(scan.windowed_streams[0].dimensions[0].setpoints("x"))   # shape (5000,)
y_coords = next(scan.windowed_streams[0].dimensions[0].setpoints("y"))   # shape (5000,)
```

---

## Memory model

Deserialising a `Spec` from JSON is O(spec complexity), not O(scan size).
No position arrays are allocated until `path.positions()` is called.

`Path(scan)` is O(spec complexity) — stores references to position functions,
not evaluated arrays.

`Window` objects are allocated one at a time during iteration. Each window
holds only scalar boundary values and a back-reference to the previous window.

`path.positions(dt, max_duration)` allocates at most `ceil(max_duration / dt)`
floats per axis per chunk — e.g. 10s at 0.2ms with 2 axes:
`2 × 50_000 × 8 bytes = 800 kB`.

---

## Invariants (asserted at Spec or Path construction time)

- `AxisT` must be hashable (dict key).  `DetectorT` and `MonitorT` are not
  required to be hashable by the library; the library stores them in lists only.
  Consumer helpers such as `find_detector_group` that use `frozenset` lookup do
  require `DetectorT` to be hashable.
- All `DetectorGroup`s within a single `WindowedStream` must have trigger ratios
  that are integer multiples of each other.
- The set of detectors is unique across all detector groups within a `WindowedStream`
  (and likewise within a `ContinuousStream`).
- Detector names are disjoint from continuous stream detector names and monitor
  names within the `Acquire`.
- Position functions on all motion spec nodes must be differentiable at
  collection window boundaries (required for velocity computation at
  turnarounds).
- When `scan.fly == False` (step scan), windows always have empty `moving_axes`.

---

## Construction

### Motion spec composition

The composable motion nodes — `Linspace`, `Static`, `Product`, `Zip`, `Concat`,
`Squash`, `Repeat`, `Snake` — use only `AxisT` and have no knowledge of
`DetectorT` or `MonitorT`. Assemble the full motion tree before wrapping it
in `Acquire`.

```python
# Primitive specs
x = Linspace("x", 0, 10, 100)   # 100 points from 0 to 10
y = Linspace("y", 0, 5, 50)     # 50 points from 0 to 5

# Composition operators
grid   = y * x    # Product: 50×100 = 5000 points
snaked = y * ~x   # Snake inner axis (x reverses on odd rows)

# Motion specs are freely composable.
# A Spec with detectors (Acquire, etc.) inside a composition operator is
# not a standard pattern and results are undefined; assemble the full motion
# tree before wrapping in Acquire.
# Acquire(motion) * other  →  not recommended
```

Operators available on any motion spec node:

| Expression | Result | Meaning |
|------------|--------|---------|
| `a * b`    | `Product(b, a)` — outer × inner | b is fast axis, a is slow |
| `~a`       | `Snake(a)` | reverse alternate repeats of a |
| `a.zip(b)` | `Zip(a, b)` | interleave axes of a and b |
| `a.concat(b)` | `Concat(a, b)` | concatenate a then b |

### Attaching triggering to motion — `Acquire`

`Acquire` is a `Spec` subclass that is always the outermost construction node.
It takes a pure motion spec (`Spec[AxisT, Never, Never]`) and binds detector
triggering, monitor configuration, and fly/step mode, producing a
`Spec[AxisT, DetectorT, MonitorT]` with exactly one detector stream.
`fly=True` means the innermost motion dimension sweeps continuously (flyscan);
all outer dimensions are stepped. `fly=False` (default) is a software step scan.

```python
# Step scan
spec: Acquire[str, str, Never] = Acquire(
    Product(Linspace("y", 0, 5, 50), Linspace("x", 0, 10, 100)),
    fly=False,              # default
    stream_name="primary",  # default
    detectors=[
        DetectorGroup(
            exposures_per_collection=1,
            collections_per_event=1,
            livetime=0.1,
            deadtime=0.01,
            detectors=["det1"],
        ),
    ],
)

# Flyscan — inner axis sweeps continuously; cameras are a ContinuousStream
spec: Acquire[str, str, str] = Acquire(
    Product(Linspace("y", 0, 5, 50), ~Linspace("x", 0, 10, 100)),
    fly=True,
    detectors=[
        DetectorGroup(1, 1, 0.003, 0.001, ["saxs", "waxs"]),
        DetectorGroup(10, 1, 0.0003, 8e-9, ["timestamp", "x_enc", "y_enc"]),
    ],
    continuous_streams=[
        ContinuousStream("cameras", [
            DetectorGroup(1, 1, 0.048, 0.001, ["front_cam", "side_cam"]),
        ]),
    ],
    monitors=[
        MonitorStream("temperature", "tc1"),
    ],
)
```

`Acquire.compile()` always produces a `Scan` with exactly one
stream, named `stream_name` (default `"primary"`). All detector groups within
that stream must trigger at integer-multiple rates of each other. Multi-stream
support requires a different `Spec` subclass — see Open Questions.

### Generics and type inference

Pyright infers `DetectorT` and `MonitorT` from the constructor arguments when
both are constrained. The type parameters exist for static analysis only — no
runtime generic parameterization is required by Pydantic.

```python
# Pyright infers Acquire[str, str, str] — no annotation needed.
spec = Acquire(
    motion,
    detectors=[DetectorGroup(
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=0.003,
        deadtime=0.001,
        detectors=["saxs"],
    )],
    monitors=[MonitorStream("temp", "tc1")],
)

# Without monitors, MonitorT is unconstrained; an explicit annotation
# is required to pin it to Never.
spec_no_mon: Acquire[str, str, Never] = Acquire(
    motion,
    detectors=[DetectorGroup(
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=0.003,
        deadtime=0.001,
        detectors=["saxs"],
    )],
)
```

See `tests/scanspec2/test_type_inference.py` for pyright assertions that
verify this inference.

### `spec.compile()` — producing `Scan`

`scan = spec.compile()` (or `acquire.compile()` for `Acquire`) compiles
the spec into a `Scan`. This is O(spec complexity) — no position
arrays are allocated.

`Scan` is the input to `Path` and the sole entry point for analysis:

```python
scan: Scan[str, str, str] = acquire.compile()

# For Acquire, exactly one windowed stream is produced.
assert len(scan.windowed_streams) == 1
assert scan.windowed_streams[0].name == "primary"
assert scan.fly == True                              # flyscan — innermost sweeps
assert len(scan.windowed_streams[0].detector_groups) == 2

path  = Path(scan)                                   # for consumption
shape = [d.length for d in scan.windowed_streams[0].dimensions]  # for analysis
```

`spec.compile()` is idempotent — calling it multiple times produces fresh
equivalent objects without mutating `spec` or any motion node.

### Maximal example — full construction

```python
# DCM energy outer axis × snaked XY fly scan inner.
# Optical cameras are monitors — not on the primary event grid.

energy_axis = Linspace("energy", 7.0, 7.1, 20)
xy_motion   = Product(Linspace("y", 0, 5, 50), ~Linspace("x", 0, 10, 100))
full_motion = energy_axis * xy_motion   # 20 energy steps × 50 rows = 1000 windows

spec: Acquire[str, str, str] = Acquire(
    full_motion,
    fly=True,           # innermost dimension (x) sweeps continuously
    stream_name="primary",
    detectors=[
        # SAXS and WAXS Pilatus: 1 frame per event, 3ms live, 1ms dead
        DetectorGroup(
            exposures_per_collection=1,
            collections_per_event=1,
            livetime=0.003,
            deadtime=0.001,
            detectors=["saxs", "waxs"],
        ),
        # PandA encoders: 10× faster than Pilatus, 0.3ms live, 8ns dead
        DetectorGroup(
            exposures_per_collection=10,
            collections_per_event=1,
            livetime=0.0003,
            deadtime=8e-9,
            detectors=["timestamp", "x_enc", "y_enc"],
        ),
    ],
    continuous_streams=[
        # Optical cameras: self-timed at ~20 Hz — grouped into one ContinuousStream
        ContinuousStream("cameras", [
            DetectorGroup(
                exposures_per_collection=1,
                collections_per_event=1,
                livetime=0.048,
                deadtime=0.001,
                detectors=["front_cam", "side_cam"],
            ),
        ]),
    ],
    monitors=[
        # Free-running temperature PV — no timing parameters
        MonitorStream("dcm_temp", "dcm_temperature"),
    ],
)

# spec.compile() produces:
# scan.fly == True
# scan.windowed_streams == [
#     WindowedStream(
#         name="primary",
#         dimensions=[
#             ScanDimension(axes=["energy"], length=20,  snake=False),
#             ScanDimension(axes=["y"],      length=50,  snake=False),
#             ScanDimension(axes=["x"],      length=100, snake=True),
#         ],
#         detector_groups=[
#             DetectorGroup(..., ["saxs", "waxs"]),
#             DetectorGroup(..., ["timestamp", "x_enc", "y_enc"]),
#         ],
#     )
# ]
# scan.continuous_streams == [
#     ContinuousStream("cameras",
#         [DetectorGroup(..., ["front_cam", "side_cam"])]),
# ]
# scan.monitors == [MonitorStream("dcm_temp", "dcm_temperature")]
```
```

### Validation at construction time

The following are checked when `Acquire` is constructed and raise `ValueError`
on failure:

- All `collections_per_event` values within a single `WindowedStream` must be
  integer multiples of each other (validated pairwise at `Acquire` construction).
- `frozenset(group.detectors)` must be unique across all detector groups in
  the same `WindowedStream`.
- Detector names must be disjoint from continuous stream detector names and
  monitor names within the `Acquire`.
- Any `Spec` subclass with detectors is always the outermost node and cannot
  be nested inside a composition operator (`Product`, `Zip`, `Concat`, `Snake`).

### Serialization

A spec serializes to JSON using pydantic's discriminated union on the motion
tree (each node has a `type` literal field: `"Linspace"`, `"Product"`, etc.).
`Acquire` wraps the motion tree and serializes its own fields inline.

```json
{
  "type": "Acquire",
  "spec": {
    "type": "Product",
    "outer": {"type": "Linspace", "axis": "y", "start": 0, "stop": 5, "num": 50},
    "inner": {"type": "Snake", "spec": {"type": "Linspace", "axis": "x", "start": 0, "stop": 10, "num": 100}}
  },
  "fly": true,
  "stream_name": "primary",
  "detectors": [
    {"exposures_per_collection": 1, "collections_per_event": 1,
     "livetime": 0.003, "deadtime": 0.001, "detectors": ["saxs", "waxs"]}
  ],
  "continuous_streams": [],
  "monitors": []
}
```

The `type` field is **only** used by the pydantic discriminated-union
deserializer — never in Python-side `isinstance` checks or dispatch logic.

### Open questions

1. **Multi-stream `Spec` subclass**: The use case for two streams with
   different dimensionality (e.g. diffraction `[N]` and spectroscopy
   `[N, 2, 1000]`) is addressed by a second `Spec` subclass distinct from
   `Acquire`. The name and construction API for this subclass are not yet
   specified.

2. **`path.positions(dt, max_duration)` return type**: yields
   `dict[AxisT, np.ndarray]` only for flying axes (those in `moving_axes`).
   Static axes are omitted. The PMAC consumer must union these with
   `window.static_axes` if it needs all axes.
