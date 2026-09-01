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
  Only appears on `Spec` and `Scan` — never on `Window`.

`Spec[AxisT, DetectorT, MonitorT]` — base class for all scan specs.
`Acquire[AxisT, DetectorT, MonitorT]` — concrete `Spec` subclass: wraps a motion spec + produces a single stream.
`Scan[AxisT, DetectorT, MonitorT]` — compiled output of `spec.compile()`; iterable, yielding `Window` objects.
`WindowedStream[AxisT, DetectorT]` — one named detector stream within a `Scan`: dimensions + detector groups.
`ContinuousStream[DetectorT]` — constant-rate detector stream with no scan dimensions (e.g. cameras at 20 Hz).
`MonitorStream[MonitorT]` — on-change PV monitor; no timing parameters.
`Window[AxisT, DetectorT]` — pure data object yielded by iterating a `Scan`; trigger sequences may span multiple streams.
`Dimension[AxisT]` — one dimension of the compiled scan geometry.
`TriggerSequence[DetectorT]` / `TriggerChild[DetectorT]` — detector triggering description (see Trigger structures below).

---

## Data structures

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Generic, Iterator, TypeVar
import numpy as np
from pydantic import BaseModel, ConfigDict

AxisT = TypeVar("AxisT")
DetectorT = TypeVar("DetectorT")
MonitorT = TypeVar("MonitorT")


class TriggerRepeat(BaseModel):
    """Timing parameters for one repeating trigger block.

    num:      number of times this block repeats.
    livetime: detector exposure time in seconds. None means not yet
        resolved -- a downstream process (e.g. ophyd-async) fills it in
        before compile(); must be resolved by then.
    deadtime: detector readout/spacing time in seconds. Same None
        semantics as livetime.

    Centred-livetime semantics apply: execution order per repeat is
    ½·deadtime -> livetime -> ½·deadtime.
    """
    model_config = ConfigDict(frozen=True)

    num: int
    livetime: float | None
    deadtime: float | None


class TriggerChild(BaseModel, Generic[DetectorT]):
    """One parallel child within a TriggerSequence.

    Fires during every parent repeat; its own repeats list executes
    sequentially within each parent livetime. detectors names the child's
    own detector set explicitly (rather than keying a dict by it) so the
    whole structure serializes to JSON cleanly -- frozenset is not a valid
    JSON object key, but is a perfectly ordinary field value.
    """
    model_config = ConfigDict(frozen=True)

    detectors: frozenset[DetectorT]
    repeats: list[TriggerRepeat]


class TriggerSequence(BaseModel, Generic[DetectorT]):
    """Detector triggering description for one sequential entry in a window.

    detectors is the set of detectors triggered by trigger_repeat. children
    is a list of parallel children, each firing during every parent repeat;
    each child's own repeats list executes sequentially. All child detector
    sets must be disjoint from each other and from detectors (checked at
    compile time).

    Window.trigger_sequences is an ordered list; entries execute one after
    another within the window. Compiled specs always produce a single-entry
    list -- multi-entry lists (the variable-spacing spacer pattern) only
    arise via manually-constructed Window objects.

    TriggerRepeat/TriggerChild/TriggerSequence are pydantic BaseModels, not
    plain dataclasses like most compiled output (ADR 0003 Decision 6
    carve-out): TriggerSequence doubles as caller-authored input to
    Acquire.trigger_sequence and must survive a JSON round trip, including
    partially-unresolved timing.
    """
    model_config = ConfigDict(frozen=True)

    detectors: frozenset[DetectorT]
    trigger_repeat: TriggerRepeat
    children: list[TriggerChild[DetectorT]]


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


class Window(Generic[AxisT, DetectorT]):
    """A contiguous stretch of motion during which detectors are triggered.

    Windows are separated by turnarounds. scanspec provides boundary kinematics
    so the caller can compute the turnaround trajectory externally via
    calculate_turnaround.

    trigger_sequences is an ordered sequential list; entries execute one
    after another within the window. In a multi-stream scan a window may
    contain sequences for only a subset of streams (e.g. some motion phases
    trigger only diffraction detectors, others only spectroscopy).

    Window is a pure data object — all fields are set at creation and never
    mutated.
    """
    # Axes that do not move during this window.
    # Move to these positions before starting the window.
    static_axes: dict[AxisT, float]

    # Axes that move continuously during this window, with their boundary
    # kinematics.  Empty for step scan windows.  Keys are disjoint from
    # static_axes — enforced structurally.
    moving_axes: dict[AxisT, AxisMotion]

    # True when the trajectory is nonlinear (velocity varies during the
    # window).  False for step scan windows (moving_axes is empty) and for
    # constant-velocity windows (start_velocity == end_velocity for every
    # axis).  Computed analytically from the Spec's position functions — no
    # floating-point comparison involved.
    non_linear: bool

    # Total time for this collection window, in seconds.
    # Equals sum(seq.trigger_repeat.num * (livetime + deadtime) for all
    # trigger_sequences) -- children run inside the parent livetime and do
    # not extend it.
    duration: float

    # Detector triggering for this window, in execution order.
    trigger_sequences: list[TriggerSequence[DetectorT]]

    # One step back only -- enough to compute the gap into this window.
    previous: Window[AxisT, DetectorT] | None

    def positions(self, times: np.ndarray) -> dict[AxisT, np.ndarray]:
        """Positions for the moving axes at each of the given real-second times.

        A pure function of the times array supplied -- no generator, no
        built-in chunking; the caller owns any chunking it needs. Raises
        RuntimeError if this is a step window (no continuous trajectory).
        """
        ...


class Dimension(Generic[AxisT]):
    """One dimension of the compiled scan geometry.

    Produced by Spec.compile(). One entry is created per motion primitive
    (Linspace, Spiral, etc.) in the spec tree; Zip merges two primitives
    into one entry with multiple axes.

    Uses the 1.x fence/post index convention:
    - Midpoints (detector setpoints) are at half-integer indexes
      0.5, 1.5, ..., length - 0.5.
    - Fly boundaries (posts) are at integer indexes 0, 1, ..., length.
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

        Midpoints are at half-integer indexes 0.5, 1.5, ..., length - 0.5.
        Snaking is NOT applied — dim.snake is provided for the caller.
        chunk_size=None yields one array; chunk_size=N yields chunks.

        Full materialisation: next(dim.setpoints(axis))
        """
        ...


@dataclass
class DetectorGroup(Generic[DetectorT]):
    """Upfront description of a set of detectors sharing trigger parameters.

    Lives on Acquire.detectors. Used to configure detectors before the scan
    starts. Static livetime/deadtime are resolved into TriggerRepeat
    instances when Acquire.compile() is called.

    exposures_per_collection: exposures the detector accumulates per collection.
    collections_per_event: collections that form one event in the stream.
      exposures_per_event = exposures_per_collection * collections_per_event.
    """
    exposures_per_collection: int
    collections_per_event: int
    livetime: float | None    # None means ophyd-async sets it
    deadtime: float | None    # None means ophyd-async sets it
    detectors: list[DetectorT]

    @property
    def exposures_per_event(self) -> int:
        """Total triggers per event: each collection needs its own trigger."""
        return self.exposures_per_collection * self.collections_per_event


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
    only.  No timing parameters — the PV runs at its own rate.
    """
    name: str
    detector: MonitorT


class Scan(Generic[AxisT, DetectorT, MonitorT]):
    """Compiled output of Spec.compile().

    O(spec complexity) to construct — no position arrays allocated until
    setpoints() or iteration is called.

    Iterable: ``for window in scan`` yields one ``Window`` per collection
    point (step scan) or per sweep (fly scan).  Also the sole entry point
    for analysis via ``scan.windowed_streams``.
    """
    # One or more named window-aligned detector streams, each with its own
    # dimensions.  A single Acquire always produces exactly one stream;
    # Concat of Acquires with different stream_names produces one per name.
    windowed_streams: list[WindowedStream[AxisT, DetectorT]]

    # Groups of continuously-acquired detectors sharing timing (no scan dims).
    # E.g. front_cam + side_cam at 20 Hz as one ContinuousStream.
    continuous_streams: list[ContinuousStream[DetectorT]]

    # Free-running PV monitors — no timing parameters.
    # E.g. temperature readbacks, beam current.
    monitors: list[MonitorStream[MonitorT]]

    # Every combination of stream names simultaneously active in some
    # window, so a consumer can validate sequencer-table capacity up front,
    # without iterating.  A detector-bearing Acquire contributes its own
    # singleton; Concat/Product/Zip union and deduplicate their children's
    # lists; Repeat/Snake pass their single inner spec's value through.
    active_stream_sets: list[frozenset[str]]

    @property
    def has_moving_axes(self) -> bool:
        """True if any window will have moving_axes (a fly dimension is present)."""
        ...

    @property
    def non_linear(self) -> bool:
        """True if any fly dimension uses a non-linear position function."""
        ...

    def with_start(
        self, window: int, trigger_index: int = 0
    ) -> Scan[AxisT, DetectorT, MonitorT]:
        """Return a new Scan that starts iteration at the given window.

        The first yielded window has its first `trigger_index` root-level
        trigger repeats truncated off its trigger_sequences, and its
        duration reduced to match. Used for pause/resume — constructs a new
        Scan from a known progress point rather than rewinding an existing
        iterator.
        """
        ...

    def __iter__(self) -> Iterator[Window[AxisT, DetectorT]]:
        """Yield one Window per collection point (step) or sweep (fly)."""
        ...
```

---

## `Spec[AxisT, DetectorT, MonitorT]` — scan spec base class

`Spec` is the base class for all scan specs. Calling `spec.compile()`
compiles it into a `Scan`.
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
# scan.has_moving_axes: bool — True if any window has moving_axes (a fly dimension is present).
# scan.non_linear: bool — True if any fly dimension is a non-linear trajectory.
# scan.windowed_streams[i].dimensions: list[Dimension[str]] — ordered outer → inner motion geometry for stream i.
```
---

## `Scan` iteration

`Scan` is iterable: `for window in scan` yields one `Window` per collection
point (step scan) or per continuous sweep (fly scan). Scan owns no mutable
iteration state — it creates a fresh iterator each time.

```python
scan: Scan[str, str, str] = spec.compile()

# Normal iteration
for window in scan:
    ...

# Resume after pause — scan.with_start returns a new Scan
scan2 = scan.with_start(window=3, trigger_index=12)
for window in scan2:
    ...
```

`window.positions(times: np.ndarray)` returns positions for the moving axes
at each of the given real-second times, computed directly — no chunking, no
generator; the caller supplies exactly the times it wants and owns any
chunking itself. Only valid for fly-scan windows (raises `RuntimeError` for
step-scan windows).

---

## Consumption use cases

### 1. Software step scan

Each window is a single point; `moving_axes` is always empty (`scan.has_moving_axes`
is `False`). Detector triggering comes from `window.trigger_sequences`, one
entry per top-level detector group.

```python
async def run_step_scan(spec: Spec[str, str, str]) -> None:
    scan = spec.compile()
    assert not scan.has_moving_axes

    for window in scan:
        await move(window.static_axes)   # dict[str, float]

        # Single-rate case: one TriggerSequence per detector group, no children.
        # Children require a real triggering system to fan out; this naive
        # asyncio.gather consumer can't support them.
        assert all(not seq.children for seq in window.trigger_sequences)
        await asyncio.gather(*(
            trigger_detectors(
                seq.detectors, seq.trigger_repeat.livetime, seq.trigger_repeat.deadtime
            )
            for seq in window.trigger_sequences
        ))
```

### 2. Flyscan — PandA sequence table

One sequence table per collection window. The orchestrator supplies `scan`,
the exact set of `detector_names` this PandA sequence handles, the trigger
type, and motor position outputs — all hardware configuration, not from spec.
One SEQ block per `TriggerSequence`: the root config occupies the block; each
parallel child (a multi-rate sub-group, ADR 0007 Decision 2) needs an
additional SEQ block, wired by the caller — worked example below.

```python
async def run_panda_flyscan(
    panda,
    scan: Scan[str, str, str],
    detector_names: list[str],
    trigger: SeqTrigger,
    motor_pos_outs: dict[str, PosOut],
) -> None:
    det_key = frozenset(detector_names)
    for window in scan:
        seq = next(s for s in window.trigger_sequences if s.detectors == det_key)

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

        # Root trigger_repeat. Generating the full N+1-row/2N-edge
        # position-compare gate for N live exposures, and composing it with
        # a BITB pause-checkpoint gate, is a consumer-side (ophyd-async
        # PandA driver) concern -- ADR 0007 Assumptions A3/A4. scanspec's
        # model stops at "trigger N times, this long, this often."
        tr = seq.trigger_repeat
        rows += SeqTable.row(
            repeats=tr.num,
            trigger=SeqTrigger.IMMEDIATE,
            time1=int(tr.livetime * 1e6),
            time2=int(tr.deadtime * 1e6),
            outa1=True,
            outa2=False,
        )

        await panda.seq.table.set(rows)
        await panda.wait_for_completion()
```

#### Worked example: a chained pair of SEQ blocks (two children)

Concrete instance of "each parallel child needs an additional SEQ block,
wired by the caller": a SAXS+WAXS parent with two children, PandA and
Tetramm (ADR 0007 Decision 2). The first child shares the parent's SEQ
block (a block's two output channels fit a parent+one-child pair); each
further child gets its own block, chained by wiring the previous block's
own output line to also act as the next block's trigger input. SEQ1 and
SEQ2 below are two SEQ blocks on the *same* physical PandA unit, not two
separate PandA devices — how many concurrent chains a given PandA
generation's block count can support is a hardware/driver-config detail
out of scope here.

Row-level field values below are illustrative only — generating the real
N+1-row/2N-edge position-compare gate and composing it with a pause-
checkpoint gate is a consumer-side (ophyd-async PandA driver) concern,
ADR 0007 Assumptions A3/A4. What's pinned down here is the row
*structure* each block needs: a 5-row centred-livetime cycle per parent
repeat for the first block, and a 3-row mirror for the second.

```python
async def run_panda_flyscan_chained(
    panda,  # one PandA device -- SEQ1 and SEQ2 are two of its SEQ blocks
    window: Window[str, str],
) -> None:
    seq = next(
        s for s in window.trigger_sequences
        if s.detectors == frozenset({"saxs", "waxs"})
    )
    tr = seq.trigger_repeat
    panda_child = next(c for c in seq.children if c.detectors == frozenset({"panda"}))
    tetramm_child = next(c for c in seq.children if c.detectors == frozenset({"tetramm"}))
    panda_rep = panda_child.repeats[0]
    tetramm_rep = tetramm_child.repeats[0]

    # SEQ1: parent (SAXS+WAXS) exposed on OA, first child (PandA) on OB.
    # Pause can only be honoured while the parent is unexposed -- that's
    # the only time everything downstream is also guaranteed unexposed,
    # and parent-exposure-complete is what constitutes one logical scan
    # step. Rows 1/2/5 have nothing to do in a time1 phase (valid at the
    # hardware level when the corresponding time is 0), so time1/outa1/
    # outb1 are left at their defaults in those rows.
    seq1_rows = SeqTable.empty()
    seq1_rows += SeqTable.row(  # 1: leading half of parent's deadtime
        trigger=SeqTrigger.BITB_1, repeats=1,
        time2=int(tr.deadtime / 2 * 1e6), outa2=False, outb2=False,
    )
    seq1_rows += SeqTable.row(  # 2: parent exposed; child's leading half-deadtime
        trigger=SeqTrigger.IMMEDIATE, repeats=1,
        time2=int(panda_rep.deadtime / 2 * 1e6), outa2=True, outb2=False,
    )
    seq1_rows += SeqTable.row(  # 3: collapsible middle -- child's full livetime+deadtime
        trigger=SeqTrigger.IMMEDIATE, repeats=panda_rep.num - 1,
        time1=int(panda_rep.livetime * 1e6), outa1=True, outb1=True,
        time2=int(panda_rep.deadtime * 1e6), outa2=True, outb2=False,
    )
    seq1_rows += SeqTable.row(  # 4: child's last repeat -- half-deadtime trailing gap
        trigger=SeqTrigger.IMMEDIATE, repeats=1,
        time1=int(panda_rep.livetime * 1e6), outa1=True, outb1=True,
        time2=int(panda_rep.deadtime / 2 * 1e6), outa2=True, outb2=False,
    )
    seq1_rows += SeqTable.row(  # 5: trailing half of parent's deadtime
        trigger=SeqTrigger.IMMEDIATE, repeats=1,
        time2=int(tr.deadtime / 2 * 1e6), outa2=False, outb2=False,
    )
    await panda.seq[1].table.set(seq1_rows)

    # SEQ2: Tetramm alone -- nothing driven on OB, this block handles one
    # detector group. BITA is wired to SEQ1.OA on the physical PandA: the
    # same line that fires SAXS+WAXS also re-triggers this block every
    # parent repeat, the instant the parent becomes exposed. SEQ2 never
    # needs its own pause/checkpoint logic -- it physically can't run
    # until the parent's already exposed, so the BITB gate above is
    # entirely SEQ1's concern, not duplicated downstream.
    seq2_rows = SeqTable.empty()
    seq2_rows += SeqTable.row(  # A: leading half of Tetramm's own deadtime
        trigger=SeqTrigger.BITA_1, repeats=1,
        time2=int(tetramm_rep.deadtime / 2 * 1e6), outa2=False,
    )
    seq2_rows += SeqTable.row(  # B: collapsible middle
        trigger=SeqTrigger.IMMEDIATE, repeats=tetramm_rep.num - 1,
        time1=int(tetramm_rep.livetime * 1e6), outa1=True,
        time2=int(tetramm_rep.deadtime * 1e6), outa2=False,
    )
    seq2_rows += SeqTable.row(  # C: last repeat -- half-deadtime trailing gap
        trigger=SeqTrigger.IMMEDIATE, repeats=1,
        time1=int(tetramm_rep.livetime * 1e6), outa1=True,
        time2=int(tetramm_rep.deadtime / 2 * 1e6), outa2=False,
    )
    await panda.seq[2].table.set(seq2_rows)

    await asyncio.gather(panda.seq[1].wait_for_completion(), panda.seq[2].wait_for_completion())
```

**SEQ1** (`tr = seq.trigger_repeat`, `panda_rep = panda_child.repeats[0]`):

| Row | TRIG | REP | T1 | OA1 | OB1 | T2 | OA2 | OB2 |
|---|---|---|---|---|---|---|---|---|
| 1 | BITB | 1 | — | — | — | ½·`tr.deadtime` | 0 | 0 |
| 2 | — | 1 | — | — | — | ½·`panda_rep.deadtime` | 1 | 0 |
| 3 | — | `panda_rep.num − 1` | `panda_rep.livetime` | 1 | 1 | `panda_rep.deadtime` | 1 | 0 |
| 4 | — | 1 | `panda_rep.livetime` | 1 | 1 | ½·`panda_rep.deadtime` | 1 | 0 |
| 5 | — | 1 | — | — | — | ½·`tr.deadtime` | 0 | 0 |

Loops to row 1 on the next `BITB` pulse for the next parent repeat.

**SEQ2** (`tetramm_rep = tetramm_child.repeats[0]`), gated on `BITA` ←
`SEQ1.OA`:

| Row | TRIG | REP | T1 | OA1 | T2 | OA2 |
|---|---|---|---|---|---|---|
| A | BITA | 1 | — | — | ½·`tetramm_rep.deadtime` | 0 |
| B | — | `tetramm_rep.num − 1` | `tetramm_rep.livetime` | 1 | `tetramm_rep.deadtime` | 0 |
| C | — | 1 | `tetramm_rep.livetime` | 1 | ½·`tetramm_rep.deadtime` | 0 |

### 3. Flyscan — Motor record

For a single-axis constant-velocity scan driven by an EPICS motor record.
`window.non_linear` must be `False` and exactly one axis may be in
`window.moving_axes`.

```python
async def run_motor_record_window(
    motor: Motor,
    window: Window[str, str],
) -> None:
    """Execute one linear collection window on a single motor record."""
    assert not window.non_linear
    axis, motion = next(iter(window.moving_axes.items()))
    velocity = motion.start_velocity

    acceleration_time = await motor.acceleration_time.get_value()
    ramp_up_start   = motion.start_position - acceleration_time * velocity / 2
    ramp_down_end   = motion.end_position   + acceleration_time * velocity / 2

    await motor.check_motor_limit(ramp_up_start, ramp_down_end)

    await motor.velocity.set(await motor.max_velocity.get_value())
    await motor.set(ramp_up_start)
    await motor.velocity.set(abs(velocity))
    await motor.set(ramp_down_end, timeout=window.duration + acceleration_time + 10)
```

### 4. Flyscan — PMAC trajectory

Positions at servo cycle rate (e.g. 0.2ms), consumed in chunks the caller
chooses. `window.positions(times)` is a pure function of the `times` array
given — no generator, no chunking built in; scanspec never materializes more
than the array it is handed. Between windows the caller drives the
turnaround using boundary kinematics from adjacent windows.

```python
async def run_pmac_flyscan(
    pmac,
    scan: Scan[str, str, str],
) -> None:
    dt = 0.0002
    chunk_size = 50_000  # caller's own chunking choice
    prev_window: Window[str, str] | None = None
    for window in scan:
        # Turnaround from previous window into this one.
        if prev_window is not None:
            prev = prev_window.moving_axes
            curr = window.moving_axes
            bridge = calculate_turnaround(
                {a: m.end_position   for a, m in prev.items()},
                {a: m.end_velocity   for a, m in prev.items()},
                {a: m.start_position for a, m in curr.items()},
                {a: m.start_velocity for a, m in curr.items()},
            )
            await pmac.send_positions(bridge)

        # Consume this window's continuous trajectory in caller-sized chunks.
        n_total = int(window.duration / dt)
        start = 0
        while start < n_total:
            end = min(start + chunk_size, n_total)
            times = np.arange(start, end) * dt
            arrays = window.positions(times)   # dict[str, np.ndarray]
            await pmac.send_positions(arrays)
            start = end

        prev_window = window
```

### 5. Pause and resume

On pause the PandA completes the current checkpoint and reports progress as
`(window_index, trigger_index)` — a completed-repeat count, not a time (see
Pause and resume principles in `PRD.md` §6). Resume constructs a new `Scan`
from that point via `scan.with_start()`.

```python
async def resume_after_pause(
    panda,
    scan: Scan[str, str, str],
) -> Scan[str, str, str]:
    window_index  = await panda.current_window_index()        # int
    trigger_index = await panda.completed_trigger_repeats()   # int

    return scan.with_start(window=window_index, trigger_index=trigger_index)
    # Pass to run_panda_flyscan / run_pmac_flyscan as normal -- the first
    # yielded window already has its first `trigger_index` root-level
    # repeats truncated off trigger_sequences, and duration reduced to match.
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
        for axis in dim.axes:
            coords[axis] = next(dim.setpoints(axis))   # np.ndarray

# Example: 2D grid flyscan (Acquire with single stream "primary")
# scan.has_moving_axes == True
# scan.windowed_streams[0].name == "primary"
# scan.windowed_streams[0].dimensions == [
#     Dimension(axes=["y"], length=50,  snake=False),
#     Dimension(axes=["x"], length=100, snake=True),
# ]
# DetectorGroup(["saxs", "waxs"]):                 collections_per_event=1  -> shape (50, 100)
# DetectorGroup(["timestamp", "x_enc", "y_enc"]):  collections_per_event=10 -> shape (50, 100, 10)

# Example: spiral scan — x and y share one dimension
# scan.windowed_streams[0].dimensions == [Dimension(axes=["x", "y"], length=5000, snake=False)]
x_coords = next(scan.windowed_streams[0].dimensions[0].setpoints("x"))   # shape (5000,)
y_coords = next(scan.windowed_streams[0].dimensions[0].setpoints("y"))   # shape (5000,)
```

---

## Invariants (asserted at Spec or Scan construction/compile time)

- `AxisT` must be hashable (dict key).  `DetectorT` and `MonitorT` are not
  required to be hashable by the library, except where used as `frozenset`
  members (`TriggerSequence`/`TriggerChild.detectors`).
- All `DetectorGroup`s within a single `WindowedStream` must have trigger ratios
  that are integer multiples of each other.
- Detector names must be disjoint across `detectors`, `continuous_streams`,
  and `monitors` within an `Acquire` (checked at construction time).
- If `trigger_sequence` is supplied on `Acquire`, its total detector set
  (root `detectors` union every child's `detectors`) must exactly match
  `Acquire.detectors`' detector set (checked at construction time).
- `TriggerSequence` child detector sets must be disjoint from each other and
  from the parent's `detectors`; each child must trigger at an integer ratio
  of the parent rate; each child's total duration must not exceed the
  parent's livetime (checked at `compile()` time, for both caller-supplied
  and auto-derived sequences).
- When `scan.has_moving_axes == False` (step scan), windows always have empty
  `moving_axes`.

---

## Construction

### Motion spec composition

The composable motion nodes — `Linspace`, `Static`, `Range`, `Spiral`,
`Ellipse`, `Polygon`, `Product`, `Zip`, `Concat`,
`Repeat`, `Snake` — use only `AxisT` and have no knowledge of
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
```

Operators available on any spec node:

| Expression | Result | Meaning |
|------------|--------|---------|
| `a * b`    | `Product(b, a)` — outer × inner | b is fast axis, a is slow |
| `~a`       | `Snake(a)` | reverse alternate repeats of a |
| `a.zip(b)` | `Zip(a, b)` | interleave axes of a and b |
| `a.concat(b)` | `Concat(a, b)` | concatenate a then b |

`Concat` is also how detector-bearing `Acquire`s combine into a multi-stream
scan (see "Multi-stream scans" below) — `left`/`right` may each carry their
own detector configuration. `Product` and `Zip`, by contrast, only merge
motion generators: they reject nested specs carrying `continuous_streams` or
`monitors` outright, and silently drop a nested `Acquire`'s
`windowed_streams` rather than merging them — nest detector-bearing specs
only inside `Concat`/`Repeat`, never `Product`/`Zip`.

### Attaching triggering to motion — `Acquire`

`Acquire` is a `Spec` subclass that is always the outermost construction node
for a given windowed stream. It takes a pure motion spec
(`Spec[AxisT, Never, Never]`) and binds detector triggering, monitor
configuration, and fly/step mode, producing a `Spec[AxisT, DetectorT,
MonitorT]` with exactly one windowed stream, named `stream_name` (default
`"primary"`). `fly=True` means the innermost motion dimension sweeps
continuously (flyscan); all outer dimensions are stepped. `fly=False`
(default) is a software step scan.

`duration` is per-point time in seconds. When detectors are present, duration
is derived from trigger timing. For detector-less scans: step scans default to
`duration=0`, fly scans use `duration` to compute `window.duration = num_points * duration`.
When `duration` is `None` (default), fly windows fall back to index-unit duration.

`trigger_sequence` is an optional, caller-supplied `TriggerSequence` used as
the windowed stream's trigger structure as-is, instead of deriving one from
`detectors`. It is **required** once `detectors` has more than one
`DetectorGroup` — which group becomes the parent of a multi-rate structure is
otherwise ambiguous, so `Acquire.compile()` raises `ValueError` rather than
guessing. `detectors` is still required alongside it (it is the arming-time
description consumed by `WindowedStream.detector_groups`); the two are
cross-checked for describing the same detector set, not derived from each
other.

```python
# Step scan — single DetectorGroup, trigger_sequence derived automatically.
# (No explicit Acquire[...] annotation needed -- MonitorT infers to Never.)
spec = Acquire(
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

# Flyscan — inner axis sweeps continuously; cameras are a ContinuousStream,
# temperature a free-running MonitorStream. Still a single DetectorGroup.
# (No explicit Acquire[...] annotation needed -- monitors= pins MonitorT.)
spec = Acquire(
    Product(Linspace("y", 0, 5, 50), ~Linspace("x", 0, 10, 100)),
    fly=True,
    detectors=[
        DetectorGroup(1, 1, 0.003, 0.001, ["saxs", "waxs"]),
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

A single `Acquire.compile()` always produces a `Scan` with exactly one
windowed stream. All detector groups within that stream must trigger at
integer-multiple rates of each other (see multi-rate example below and the
maximal example).

### Multi-stream scans — `Concat` of `Acquire`s

Two streams with different dimensionality (e.g. diffraction `[N]` and
spectroscopy `[N, 2, 1000]`) are expressed as a `Concat` of `Acquire`s with
different `stream_name`s — not a separate `Spec` subclass. `Concat.compile()`
merges `windowed_streams` by name (summing the innermost dimension's length
for repeated names) rather than requiring a single stream. Wrap in `Repeat`
to interleave the pattern, and an outer `Acquire` to carry scan-wide
monitors:

```python
diff_det = DetectorGroup(1, 1, 0.01, 0.001, ["diffraction"])
spec_det = DetectorGroup(1, 1, 0.003, 0.001, ["spectroscopy"])

diff_acq = Acquire(
    Static("e", 7.0), detectors=[diff_det], stream_name="diff",
)
spec_fwd = Acquire(
    Linspace("e", 7.0, 7.1, 1000), fly=True, detectors=[spec_det], stream_name="spec",
)
spec_rev = Acquire(
    Linspace("e", 7.1, 7.0, 1000), fly=True, detectors=[spec_det], stream_name="spec",
)

# 200 iterations of: step to e=7.0 (1 diffraction frame), fly e 7.0->7.1
# (1000 spectroscopy frames), fly e 7.1->7.0 (1000 spectroscopy frames).
# Acquire[...] annotation IS needed here -- not for MonitorT (monitors=
# pins that), but because DetectorT can't be inferred through the
# Repeat(...concat...concat...) combinator chain feeding `spec=`.
spec: Acquire[str, str, str] = Acquire(
    Repeat(diff_acq.concat(spec_fwd).concat(spec_rev), num=200),
    monitors=[MonitorStream("temperature", "tc1")],
)
scan = spec.compile()

# scan.windowed_streams has two entries, "diff" and "spec":
# streams_by_name["diff"].dimensions == [Dimension(["e"...], 200, ...), Dimension(["e"], 1, ...)]
# streams_by_name["spec"].dimensions == [Dimension(["e"...], 200, ...), Dimension(["e"], 2000, ...)]  # 1000 + 1000
# 600 windows total: 200 x (1 step + 1 fly + 1 fly). Each window's
# trigger_sequences carries only the detectors active in that phase --
# diffraction fires only in step windows, spectroscopy only in fly windows.
```

### Generics and type inference

Pyright infers `DetectorT` from `detectors=`. `MonitorT` infers from
`monitors=` when given; when omitted, it defaults to `Never` (a PEP 696
`TypeVar` default) rather than needing an explicit annotation. The type
parameters exist for static analysis only — no runtime generic
parameterization is required by Pydantic.

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

# Without monitors, MonitorT infers to Never — still no annotation needed.
spec_no_mon = Acquire(
    motion,
    detectors=[DetectorGroup(1, 1, 0.003, 0.001, ["saxs"])],
)
```

Explicit `Acquire[...]` annotation is still needed on the rare construction
pyright can't see through at all — e.g. `spec=` built from a `Repeat`-of-
`Concat`-of-`Acquire`s chain, where `DetectorT` (unrelated to `MonitorT`)
can't be tracked through the combinators (see the multi-stream example
above).

See `tests/scanspec/v2/test_type_inference.py` for pyright assertions.

### `spec.compile()` — producing `Scan`

`scan = spec.compile()` (or `acquire.compile()` for `Acquire`) compiles
the spec into a `Scan`. This is O(spec complexity) — no position
arrays are allocated.

`Scan` is iterable and the sole entry point for analysis:

```python
scan: Scan[str, str, str] = acquire.compile()

# For a single Acquire, exactly one windowed stream is produced.
assert len(scan.windowed_streams) == 1
assert scan.windowed_streams[0].name == "primary"
assert scan.has_moving_axes == True                  # flyscan — innermost sweeps
assert len(scan.windowed_streams[0].detector_groups) == 2

for window in scan:                                  # iterate windows
    ...
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

# No explicit Acquire[...] annotation needed -- detectors=/monitors= pin
# DetectorT/MonitorT directly (unlike the Repeat/Concat chain above).
spec = Acquire(
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
        # PandA encoders: 10× faster than Pilatus. A child's livetime
        # excludes its own deadtime when sized against the parent's
        # livetime slot: livetime = parent_livetime/ratio - deadtime, so
        # 10 child repeats fit exactly inside the 3ms parent livetime.
        DetectorGroup(
            exposures_per_collection=10,
            collections_per_event=1,
            livetime=0.000299992,
            deadtime=8e-9,
            detectors=["timestamp", "x_enc", "y_enc"],
        ),
    ],
    # Which DetectorGroup becomes the trigger-sequence parent is not
    # auto-derived once there is more than one -- supplied explicitly.
    trigger_sequence=TriggerSequence(
        detectors=frozenset({"saxs", "waxs"}),
        trigger_repeat=TriggerRepeat(num=100, livetime=0.003, deadtime=0.001),
        children=[
            TriggerChild(
                detectors=frozenset({"timestamp", "x_enc", "y_enc"}),
                repeats=[
                    TriggerRepeat(num=10, livetime=0.000299992, deadtime=8e-9),
                ],
            ),
        ],
    ),
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
# scan.has_moving_axes == True
# scan.windowed_streams == [
#     WindowedStream(
#         name="primary",
#         dimensions=[
#             Dimension(axes=["energy"], length=20,  snake=False),
#             Dimension(axes=["y"],      length=50,  snake=False),
#             Dimension(axes=["x"],      length=100, snake=True),
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
# Every window's trigger_sequences == [TriggerSequence(
#     detectors=frozenset({"saxs", "waxs"}),
#     trigger_repeat=TriggerRepeat(num=100, livetime=0.003, deadtime=0.001),
#     children=[TriggerChild(
#         detectors=frozenset({"timestamp", "x_enc", "y_enc"}),
#         repeats=[TriggerRepeat(num=10, livetime=0.000299992, deadtime=8e-9)],
#     )],
# )]
```

### Validation

**At `Acquire` construction time** (raises `ValueError` immediately):

- If `trigger_sequence` is given, its total detector set (root `detectors`
  union every child's `detectors`) must exactly match `Acquire.detectors`'
  detector set.
- Detector names must be globally unique across `detectors`,
  `continuous_streams`, and `monitors`.

**At `compile()` time**:

- If `trigger_sequence` was given: `validate_trigger_sequence` checks that
  `trigger_repeat.livetime`/`deadtime` are resolved (not `None`), that every
  child detector set is disjoint from the parent and from every other child,
  that each child's own `livetime`/`deadtime` are resolved, that each child
  triggers at an integer ratio of the parent rate, and that each child's
  total duration does not exceed the parent's livetime.
- If `trigger_sequence` was **not** given and `detectors` has more than one
  `DetectorGroup`: raises `ValueError` — the parent is ambiguous, supply
  `trigger_sequence` explicitly instead.
- If `duration` is given explicitly and is less than the detector-derived
  per-point duration: raises `ValueError`.
- Any `Spec` subclass with detectors is always the outermost node for its
  stream and cannot be nested inside `Product` or `Zip` without losing its
  `windowed_streams` (see Motion spec composition above) — use `Concat`
  instead.

### Serialization

A spec serializes to JSON using pydantic's discriminated union on the motion
tree (each node has a `type` literal field: `"Linspace"`, `"Product"`, etc.).
`Acquire` wraps the motion tree and serializes its own fields inline,
including `trigger_sequence` — `TriggerRepeat`/`TriggerChild`/`TriggerSequence`
are pydantic `BaseModel`s and round-trip natively: `frozenset` fields become
plain JSON arrays, and `children` is an ordinary list rather than a dict
keyed by `frozenset` (which JSON cannot represent). A full round trip via
`model_dump_json()`/`model_validate_json()` (or the `AnySpec` `TypeAdapter`
as part of a full `Acquire`) is supported end to end, including
partially-unresolved timing (`livetime`/`deadtime` still `None`).

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
     "livetime": 0.003, "deadtime": 0.001, "detectors": ["saxs", "waxs"]},
    {"exposures_per_collection": 10, "collections_per_event": 1,
     "livetime": 0.000299992, "deadtime": 8e-9, "detectors": ["timestamp", "x_enc", "y_enc"]}
  ],
  "continuous_streams": [],
  "monitors": [],
  "duration": null,
  "trigger_sequence": {
    "detectors": ["saxs", "waxs"],
    "trigger_repeat": {"num": 100, "livetime": 0.003, "deadtime": 0.001},
    "children": [
      {"detectors": ["timestamp", "x_enc", "y_enc"],
       "repeats": [{"num": 10, "livetime": 0.000299992, "deadtime": 8e-9}]}
    ]
  }
}
```

The `type` field is **only** used by the pydantic discriminated-union
deserializer — never in Python-side `isinstance` checks or dispatch logic.

### Open questions

1. **Pause/resume end point**: does pause/resume ever need an *end* point as
   well as a start point? Raised during design, currently assumed not.
2. **`window.positions(times)` return type**: yields `dict[AxisT,
   np.ndarray]` only for flying axes (those in `moving_axes`). Static axes
   are omitted. The PMAC consumer must union these with `window.static_axes`
   if it needs all axes.
