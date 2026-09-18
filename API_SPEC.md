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
`Sync[AxisT, DetectorT, MonitorT]` — concrete `Spec` subclass: wraps a motion spec + binds detector triggering, producing a single windowed stream.
`Monitors[AxisT, DetectorT, MonitorT]` / `ContinuousStreams[AxisT, DetectorT, MonitorT]` — outermost wrapper `Spec` subclasses (ADR 0009) that attach scan-wide monitors / continuous streams to the whole compiled tree.
`Scan[AxisT, DetectorT, MonitorT]` — compiled output of `spec.compile()`; iterable, yielding `Window` objects.
`WindowedStream[AxisT, DetectorT]` — one named detector stream within a `Scan`: dimensions + detector groups.
`ContinuousStream[DetectorT]` — constant-rate detector stream with no scan dimensions (e.g. cameras at 20 Hz).
`MonitorStream[MonitorT]` — on-change PV monitor; no timing parameters.
`Window[AxisT, DetectorT]` — pure data object yielded by iterating a `Scan`; trigger sequences may span multiple streams.
`Dimension[AxisT]` — one dimension of the compiled scan geometry.
`TriggerRepeat[DetectorT]` / `TriggerSequence[DetectorT]` — compiled detector triggering description (see Trigger structures below).
`TriggerGroup[DetectorT]` / `TriggerPlan[DetectorT]` — caller-authored trigger hierarchy (ADR 0008), passed to `Sync.trigger_plan`; `compile()` derives `TriggerRepeat`/`TriggerSequence` from it.

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


@dataclass(frozen=True)
class TriggerRepeat(Generic[DetectorT]):
    """One resolved, repeating trigger block within a TriggerSequence.

    detectors: the set of detectors this block fires.
    num:       number of times this block repeats.
    livetime:  detector exposure time in seconds.
    deadtime:  detector readout/spacing time in seconds.

    Centred-livetime semantics apply: execution order per repeat is
    ½·deadtime -> livetime -> ½·deadtime.

    Pure compiled output -- always concrete (no unresolved timing). The
    round trip for unresolved (None) timing happens entirely on
    TriggerGroup/TriggerPlan, before compile() (ADR 0008); by the time a
    TriggerSequence exists, livetime/deadtime are guaranteed resolved.
    """

    detectors: frozenset[DetectorT]
    num: int
    livetime: float
    deadtime: float


@dataclass(frozen=True)
class TriggerSequence(Generic[DetectorT]):
    """Detector triggering description for one sequential entry in a window.

    root fires first; children fire in parallel with each other during
    every root repeat. All child detector sets must be disjoint from each
    other and from root.detectors (checked at compile time).

    Window.trigger_sequences is an ordered list; entries execute one after
    another within the window. Compiled specs always produce a single-entry
    list -- multi-entry lists (the variable-spacing spacer pattern) only
    arise via manually-constructed Window objects.
    """

    root: TriggerRepeat[DetectorT]
    children: list[TriggerRepeat[DetectorT]]


class TriggerGroup(BaseModel, Generic[DetectorT]):
    """Authoring-time detector group: identity plus trigger-timing intent.

    Lives on TriggerPlan.root / TriggerPlan.children. livetime/deadtime may
    be left unresolved (None) at authoring time -- a downstream process
    (e.g. ophyd-async) fills them in before compile(), which requires both
    concrete.
    """
    model_config = ConfigDict(frozen=True)

    detectors: frozenset[DetectorT]
    exposures_per_collection: int
    collections_per_event: int
    livetime: float | None
    deadtime: float | None

    @property
    def exposures_per_event(self) -> int:
        """Total triggers per event: each collection needs its own trigger."""
        return self.exposures_per_collection * self.collections_per_event


class TriggerPlan(BaseModel, Generic[DetectorT]):
    """Caller-authored trigger hierarchy for one Sync's windowed stream.

    root/children mirror TriggerSequence's own root/children shape one
    level up: children fire during every root repeat, in parallel with
    each other, each at its own integer-multiple rate. Detector sets
    across root and all children must be disjoint -- checked here
    structurally at construction time, since timing may still be
    unresolved. Physical checks (integer-ratio rates, child duration
    fitting inside the root's livetime, concrete timing) happen at
    compile() time against the derived TriggerSequence.

    A bare TriggerGroup (no children) is also accepted anywhere a
    TriggerPlan is -- Sync.trigger_plan is typed
    TriggerPlan[DetectorT] | TriggerGroup[DetectorT] | None, and is
    normalized to a trivial single-node TriggerPlan only where actually
    needed (compile(), uniqueness validation), not eagerly.
    """
    model_config = ConfigDict(frozen=True)

    root: TriggerGroup[DetectorT]
    children: list[TriggerGroup[DetectorT]] = []

    # Raises ValueError at construction time if any two of {root, *children}
    # share a detector.


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
    # Equals sum(seq.root.num * (seq.root.livetime + seq.root.deadtime) for
    # all trigger_sequences) -- children run inside the parent livetime and
    # do not extend it.
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

    For a windowed stream this is pure compiled output, derived by
    Sync.compile() from a caller-authored TriggerGroup (ADR 0008) -- it is
    no longer directly caller-authored there. For a continuous stream it
    is still directly caller-authored, on
    ContinuousStreams.continuous_streams (ADR 0009); static
    livetime/deadtime describe the stream's own fixed rate.

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
    # dimensions.  A single Sync always produces exactly one stream;
    # Concat of Syncs with different stream_names produces one per name.
    windowed_streams: list[WindowedStream[AxisT, DetectorT]]

    # Groups of continuously-acquired detectors sharing timing (no scan dims).
    # E.g. front_cam + side_cam at 20 Hz as one ContinuousStream.
    continuous_streams: list[ContinuousStream[DetectorT]]

    # Free-running PV monitors — no timing parameters.
    # E.g. temperature readbacks, beam current.
    monitors: list[MonitorStream[MonitorT]]

    # Every combination of stream names simultaneously active in some
    # window, so a consumer can validate sequencer-table capacity up front,
    # without iterating.  A detector-bearing Sync contributes its own
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

    @property
    def number_of_events(self) -> int:
        """Total windows this Scan will yield, without iterating.

        O(generator-tree size): product of each internal generator's own
        window count, outer -> inner. Zero generators means zero windows,
        not the empty-product identity of one.
        """
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
`Sync` is the concrete subclass for single-stream scans (see Construction).

```python
spec: Spec[str, str, str]  # provided by orchestrator — typically an Sync

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
                seq.root.detectors, seq.root.livetime, seq.root.deadtime
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
        seq = next(s for s in window.trigger_sequences if s.root.detectors == det_key)

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

        # Root TriggerRepeat. Generating the full N+1-row/2N-edge
        # position-compare gate for N live exposures, and composing it with
        # a BITB pause-checkpoint gate, is a consumer-side (ophyd-async
        # PandA driver) concern -- ADR 0007 Assumptions A3/A4. scanspec's
        # model stops at "trigger N times, this long, this often."
        tr = seq.root
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
        if s.root.detectors == frozenset({"saxs", "waxs"})
    )
    tr = seq.root
    # children are flat TriggerRepeats directly -- no nested .repeats list.
    panda_rep = next(c for c in seq.children if c.detectors == frozenset({"panda"}))
    tetramm_rep = next(c for c in seq.children if c.detectors == frozenset({"tetramm"}))

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

**SEQ1** (`tr = seq.root`, `panda_rep` = the matching child `TriggerRepeat`):

| Row | TRIG | REP | T1 | OA1 | OB1 | T2 | OA2 | OB2 |
|---|---|---|---|---|---|---|---|---|
| 1 | BITB | 1 | — | — | — | ½·`tr.deadtime` | 0 | 0 |
| 2 | — | 1 | — | — | — | ½·`panda_rep.deadtime` | 1 | 0 |
| 3 | — | `panda_rep.num − 1` | `panda_rep.livetime` | 1 | 1 | `panda_rep.deadtime` | 1 | 0 |
| 4 | — | 1 | `panda_rep.livetime` | 1 | 1 | ½·`panda_rep.deadtime` | 1 | 0 |
| 5 | — | 1 | — | — | — | ½·`tr.deadtime` | 0 | 0 |

Loops to row 1 on the next `BITB` pulse for the next parent repeat.

**SEQ2** (`tetramm_rep` = the matching child `TriggerRepeat`), gated on
`BITA` ← `SEQ1.OA`:

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

# Example: 2D grid flyscan (Sync with single stream "primary")
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
  members (`TriggerRepeat`/`TriggerGroup`/`TriggerPlan.detectors`).
- All `DetectorGroup`s within a single `WindowedStream` must have trigger ratios
  that are integer multiples of each other.
- Within one `TriggerPlan`, `root` and every `children` `TriggerGroup`'s
  detector sets must be pairwise disjoint (checked at `TriggerPlan`
  construction time, ADR 0008).
- Detector names must be globally unique across windowed streams,
  `continuous_streams`, and `monitors` for a given compiled `Scan` (ADR
  0009). Since `continuous_streams`/`monitors` are attached by the
  outermost `ContinuousStreams`/`Monitors` wrapper, not `Sync` itself, this
  is checked at `compile()` time against the full `Scan` state, not at
  `Sync` construction time.
- `TriggerSequence` child detector sets must be disjoint from each other and
  from `root`'s; each child must trigger at an integer ratio of the parent
  rate; each child's total duration must not exceed the parent's livetime
  (checked at `compile()` time, via `validate_trigger_sequence`).
- When `scan.has_moving_axes == False` (step scan), windows always have empty
  `moving_axes`.

---

## Construction

### Motion spec composition

The composable motion nodes — `Linspace`, `Static`, `Range`, `Spiral`,
`Ellipse`, `Polygon`, `Product`, `Zip`, `Concat`,
`Repeat`, `Snake` — use only `AxisT` and have no knowledge of
`DetectorT` or `MonitorT`. Assemble the full motion tree before wrapping it
in `Sync`.

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

`Concat` is also how detector-bearing `Sync`s combine into a multi-stream
scan (see "Multi-stream scans" below) — `left`/`right` may each carry their
own detector configuration. `Product` and `Zip`, by contrast, only merge
motion generators: they reject nested specs carrying `continuous_streams` or
`monitors` outright, and silently drop a nested `Sync`'s
`windowed_streams` rather than merging them — nest detector-bearing specs
only inside `Concat`/`Repeat`, never `Product`/`Zip`.

### Attaching triggering to motion — `Sync`

`Sync` is a `Spec` subclass that is always the outermost construction node
for a given windowed stream. It takes a pure motion spec
(`Spec[AxisT, Never, Never]`) and binds detector triggering and fly/step
mode via `trigger_plan`, producing a `Spec[AxisT, DetectorT, MonitorT]`
with exactly one windowed stream, named `stream_name` (default
`"primary"`). `fly=True` means the innermost motion dimension sweeps
continuously (flyscan); all outer dimensions are stepped. `fly=False`
(default) is a software step scan.

`duration` is per-point time in seconds. When detectors are present, duration
is derived from trigger timing. For detector-less scans: step scans default to
`duration=0`, fly scans use `duration` to compute `window.duration = num_points * duration`.
When `duration` is `None` (default), fly windows fall back to index-unit duration.

`trigger_plan` (ADR 0008) is the caller-authored detector/timing hierarchy:
a `TriggerGroup` (the trivial no-children case) or a `TriggerPlan` (a
`root` `TriggerGroup` plus parallel `children` `TriggerGroup`s, each at
its own integer-multiple rate). `compile()` derives both the compiled
`TriggerSequence` and the `list[DetectorGroup]` the windowed stream needs
from it — there's no separate `detectors` field to keep in sync by hand,
and no ambiguity about which group is the "parent": the caller states it
directly via `root`/`children`.

`continuous_streams`/`monitors` are **not** `Sync` fields — see "Whole-scan
acquisition" below.

```python
# Step scan — single TriggerGroup, no children.
# (No explicit Sync[...] annotation needed -- MonitorT infers to Never,
# since MonitorT no longer has anything on Sync to flow from at all.)
spec = Sync(
    Product(Linspace("y", 0, 5, 50), Linspace("x", 0, 10, 100)),
    fly=False,              # default
    stream_name="primary",  # default
    trigger_plan=TriggerGroup(
        detectors=frozenset({"det1"}),
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=0.1,
        deadtime=0.01,
    ),
)

# Flyscan, multi-rate — SAXS/WAXS as root, encoders as a 10x-faster child.
spec: Sync[str, str, Never] = Sync(
    Product(Linspace("y", 0, 5, 50), ~Linspace("x", 0, 10, 100)),
    fly=True,
    trigger_plan=TriggerPlan(
        root=TriggerGroup(
            detectors=frozenset({"saxs", "waxs"}),
            exposures_per_collection=1, collections_per_event=1,
            livetime=0.003, deadtime=0.001,
        ),
        children=[
            TriggerGroup(
                detectors=frozenset({"timestamp", "x_enc", "y_enc"}),
                exposures_per_collection=10, collections_per_event=1,
                livetime=0.000299992, deadtime=8e-9,
            ),
        ],
    ),
)
```

A single `Sync.compile()` always produces a `Scan` with exactly one
windowed stream. All detector groups within that stream must trigger at
integer-multiple rates of each other (see the maximal example below).

### Whole-scan acquisition — `Monitors` and `ContinuousStreams`

`continuous_streams` and `monitors` are not-frame-coupled: they run for
the whole scan regardless of windows, not local to wherever a `Sync`
happens to sit in the combinator tree. ADR 0009 expresses this
structurally with two outermost wrapper `Spec` subclasses instead of
fields on `Sync`:

```python
class Monitors(Spec[AxisT, DetectorT, MonitorT]):
    spec: AnySpec[AxisT, DetectorT, MonitorT]
    monitors: Sequence[MonitorStream[MonitorT]] = ()

class ContinuousStreams(Spec[AxisT, DetectorT, MonitorT]):
    spec: AnySpec[AxisT, DetectorT, MonitorT]
    continuous_streams: Sequence[ContinuousStream[DetectorT]] = ()
```

Each wrapper compiles its inner spec, then attaches its own field to the
resulting `Scan`. Both must sit outside the entire spec tree — `Concat`/
`Product`/`Zip` reject a nested spec carrying either field (see
Invariants); each wrapper also rejects double-wrapping on its own field
(`Monitors(Monitors(...))`). The two wrappers commute — nesting order
between them doesn't matter, since each only ever touches its own field —
and compose freely:

```python
spec = ContinuousStreams(
    Monitors(
        Sync(motion, trigger_plan=trigger_plan),
        monitors=[MonitorStream("temperature", "tc1")],
    ),
    continuous_streams=[ContinuousStream("cameras", [...])],
)
```

**Type-inference caveat**: unlike `DetectorT`, which still flows
automatically from `trigger_plan=` on `Sync`, `MonitorT` can no longer be
inferred purely from usage once `Monitors`
wraps `Sync` — `Sync`'s own `MonitorT` is fixed to `Never` the instant a
bare `Sync(...)` call returns, since nothing on `Sync` mentions `MonitorT`
any more, even when that call sits directly inside `Monitors(...)` in the
same expression (pyright resolves nested calls argument-first, not
bidirectionally). An explicit annotation on the assignment target
(`spec: Monitors[str, str, str] = Monitors(...)`) is required to recover
the real type.

### Multi-stream scans — `Concat` of `Sync`s

Two streams with different dimensionality (e.g. diffraction `[N]` and
spectroscopy `[N, 2, 1000]`) are expressed as a `Concat` of `Sync`s with
different `stream_name`s — not a separate `Spec` subclass. `Concat.compile()`
merges `windowed_streams` by name (summing the innermost dimension's length
for repeated names) rather than requiring a single stream. Wrap in `Repeat`
to interleave the pattern, and an outer `Monitors` (ADR 0009) to carry
scan-wide monitors:

```python
diff_group = TriggerGroup(
    detectors=frozenset({"diffraction"}),
    exposures_per_collection=1, collections_per_event=1,
    livetime=0.01, deadtime=0.001,
)
spec_group = TriggerGroup(
    detectors=frozenset({"spectroscopy"}),
    exposures_per_collection=1, collections_per_event=1,
    livetime=0.003, deadtime=0.001,
)

diff_acq: Sync[str, str, Never] = Sync(
    Static("e", 7.0), trigger_plan=diff_group, stream_name="diff",
)
spec_fwd: Sync[str, str, Never] = Sync(
    Linspace("e", 7.0, 7.1, 1000), fly=True, trigger_plan=spec_group, stream_name="spec",
)
spec_rev: Sync[str, str, Never] = Sync(
    Linspace("e", 7.1, 7.0, 1000), fly=True, trigger_plan=spec_group, stream_name="spec",
)

# 200 iterations of: step to e=7.0 (1 diffraction frame), fly e 7.0->7.1
# (1000 spectroscopy frames), fly e 7.1->7.0 (1000 spectroscopy frames).
inner: Repeat[str, str, Never] = Repeat(
    diff_acq.concat(spec_fwd).concat(spec_rev), num=200,
)
spec = Monitors(inner, monitors=[MonitorStream("temperature", "tc1")])
scan = spec.compile()

# scan.windowed_streams has two entries, "diff" and "spec":
# streams_by_name["diff"].dimensions == [Dimension(["e"...], 200, ...), Dimension(["e"], 1, ...)]
# streams_by_name["spec"].dimensions == [Dimension(["e"...], 200, ...), Dimension(["e"], 2000, ...)]  # 1000 + 1000
# 600 windows total: 200 x (1 step + 1 fly + 1 fly). Each window's
# trigger_sequences carries only the detectors active in that phase --
# diffraction fires only in step windows, spectroscopy only in fly windows.
```

### Generics and type inference

Pyright infers `DetectorT` from `Sync.trigger_plan=`. `MonitorT` no longer
has anything on `Sync` to flow from at all (ADR 0009 moved `monitors` to
the separate `Monitors` wrapper) — a bare `Sync(...)` call always infers
`MonitorT=Never` (a PEP 696 `TypeVar` default), and an explicit annotation
on `Monitors(...)`'s own assignment target is required to recover the
real `MonitorT` (see "Whole-scan acquisition" above). The type parameters
exist for static analysis only — no runtime generic parameterization is
required by Pydantic.

```python
# Pyright infers Sync[str, str, Never] — no annotation needed.
spec = Sync(
    motion,
    trigger_plan=TriggerGroup(
        detectors=frozenset({"saxs"}),
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=0.003,
        deadtime=0.001,
    ),
)

# MonitorT explicit annotation required to get it right -- see caveat above.
wrapped: Monitors[str, str, str] = Monitors(
    spec, monitors=[MonitorStream("temp", "tc1")],
)
```

Explicit `Sync[...]` annotation is still needed on the rare construction
pyright can't see through at all — e.g. `spec=` built from a `Repeat`-of-
`Concat`-of-`Sync`s chain, where `DetectorT` can't be tracked through the
combinators (see the multi-stream example above).

See `tests/scanspec/v2/test_type_inference.py` for pyright assertions.

### `spec.compile()` — producing `Scan`

`scan = spec.compile()` compiles the spec into a `Scan`. This is
O(spec complexity) — no position arrays are allocated.

`Scan` is iterable and the sole entry point for analysis:

```python
scan: Scan[str, str, str] = spec.compile()

# For a single Sync, exactly one windowed stream is produced.
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

# No explicit Sync[...] annotation needed -- trigger_plan= pins DetectorT
# directly (unlike the Repeat/Concat chain above). MonitorT still needs
# the explicit Monitors[...] annotation below (see type-inference caveat).
sync: Sync[str, str, str] = Sync(
    full_motion,
    fly=True,           # innermost dimension (x) sweeps continuously
    stream_name="primary",
    # Which TriggerGroup becomes root vs child is caller-decided; num is
    # auto-derived by compile() from scan geometry (root) and timing
    # (children), not hand-computed.
    trigger_plan=TriggerPlan(
        root=TriggerGroup(
            # SAXS and WAXS Pilatus: 1 frame per event, 3ms live, 1ms dead
            detectors=frozenset({"saxs", "waxs"}),
            exposures_per_collection=1,
            collections_per_event=1,
            livetime=0.003,
            deadtime=0.001,
        ),
        children=[
            # PandA encoders: 10x faster than Pilatus. A child's livetime
            # excludes its own deadtime when sized against the parent's
            # livetime slot: livetime = parent_livetime/ratio - deadtime,
            # so 10 child repeats fit exactly inside the 3ms parent
            # livetime.
            TriggerGroup(
                detectors=frozenset({"timestamp", "x_enc", "y_enc"}),
                exposures_per_collection=10,
                collections_per_event=1,
                livetime=0.000299992,
                deadtime=8e-9,
            ),
        ],
    ),
)

# continuous_streams/monitors attach via the outermost wrappers (ADR 0009),
# not on Sync itself. Nesting order between the two doesn't matter.
spec: ContinuousStreams[str, str, str] = ContinuousStreams(
    Monitors(
        sync,
        monitors=[
            # Free-running temperature PV — no timing parameters
            MonitorStream("dcm_temp", "dcm_temperature"),
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
#     root=TriggerRepeat(
#         detectors=frozenset({"saxs", "waxs"}),
#         num=100, livetime=0.003, deadtime=0.001,
#     ),
#     children=[TriggerRepeat(
#         detectors=frozenset({"timestamp", "x_enc", "y_enc"}),
#         num=10, livetime=0.000299992, deadtime=8e-9,
#     )],
# )]
```

### Validation

**At `TriggerPlan` construction time** (raises `ValueError` immediately,
ADR 0008):

- `root` and every `children` `TriggerGroup`'s detector sets must be
  pairwise disjoint.

**At `Sync.compile()` time**:

- Every `TriggerGroup` in `trigger_plan` (`root` and each child) must have
  concrete `livetime`/`deadtime` (not `None`) — raises `ValueError`
  otherwise. Unresolved timing is only ever valid at authoring time, before
  `compile()`.
- Each child must trigger at an integer ratio of the parent rate; each
  child's total duration must not exceed the parent's livetime
  (`validate_trigger_sequence`, re-checked against the derived
  `TriggerSequence`).
- If `duration` is given explicitly and is less than the detector-derived
  per-point duration: raises `ValueError`.

**At `Monitors.compile()` / `ContinuousStreams.compile()` time** (ADR
0009):

- Detector names must be globally unique across windowed streams,
  `continuous_streams`, and `monitors` for the whole compiled `Scan` —
  checked against the full `Scan` state after attaching, not just the
  wrapper's own field, since nesting order between the two wrappers can
  put either one first.
- Double-wrapping the same field is rejected (`Monitors(Monitors(...))`,
  `ContinuousStreams(ContinuousStreams(...))`) — checking only that
  wrapper's own field, not the other one, so
  `ContinuousStreams(Monitors(...), ...)` composes normally.

**At `Product`/`Zip`/`Concat.compile()` time**:

- A nested spec carrying `continuous_streams` or `monitors` (i.e. a
  `Monitors`/`ContinuousStreams` wrapper, or a windowed stream nested
  inside one) is rejected — those must be attached via the outermost
  `Monitors`/`ContinuousStreams` wrapper, never inside a combinator.
- Any `Spec` subclass with detectors is always the outermost node for its
  stream and cannot be nested inside `Product` or `Zip` without losing its
  `windowed_streams` (see Motion spec composition above) — use `Concat`
  instead.

### Serialization

A spec serializes to JSON using pydantic's discriminated union on the
*whole* spec tree (each node has a `type` literal field: `"Linspace"`,
`"Product"`, `"Sync"`, `"Monitors"`, `"ContinuousStreams"`, etc.) — not
just the motion nodes. `Sync` serializes its own fields inline, including
`trigger_plan`: `TriggerGroup`/`TriggerPlan` are pydantic `BaseModel`s and
round-trip natively, including partially-unresolved timing
(`livetime`/`deadtime` still `None`) — `frozenset` fields become plain
JSON arrays. A bare `TriggerGroup` (no children) is stored and serialized
as-is where given, not eagerly wrapped in a trivial `TriggerPlan`.

`TriggerRepeat`/`TriggerSequence` are **not** part of a spec's own
serialization at all — they're plain (non-pydantic) dataclasses, pure
compiled output that only exists on `Window.trigger_sequences` after
`compile()` runs. `continuous_streams`/`monitors` are not `Sync` fields
either (ADR 0009) — they appear only inside a `Monitors`/`ContinuousStreams`
wrapper node, nested around the `Sync` subtree they scope.

A full round trip via `model_dump_json()`/`model_validate_json()` (or the
`AnySpec` `TypeAdapter`) is supported end to end for the whole tree,
`Monitors`/`ContinuousStreams` wrappers included:

```json
{
  "type": "ContinuousStreams",
  "spec": {
    "type": "Monitors",
    "spec": {
      "type": "Sync",
      "spec": {
        "type": "Product",
        "outer": {"type": "Linspace", "axis": "y", "start": 0, "stop": 5, "num": 50},
        "inner": {"type": "Snake", "spec": {"type": "Linspace", "axis": "x", "start": 0, "stop": 10, "num": 100}}
      },
      "fly": true,
      "stream_name": "primary",
      "trigger_plan": {
        "root": {
          "detectors": ["saxs", "waxs"],
          "exposures_per_collection": 1, "collections_per_event": 1,
          "livetime": 0.003, "deadtime": 0.001
        },
        "children": [
          {"detectors": ["timestamp", "x_enc", "y_enc"],
           "exposures_per_collection": 10, "collections_per_event": 1,
           "livetime": 0.000299992, "deadtime": 8e-9}
        ]
      },
      "duration": null
    },
    "monitors": [
      {"name": "dcm_temp", "detector": "dcm_temperature"}
    ]
  },
  "continuous_streams": [
    {"name": "cameras", "detector_groups": [
      {"exposures_per_collection": 1, "collections_per_event": 1,
       "livetime": 0.048, "deadtime": 0.001, "detectors": ["front_cam", "side_cam"]}
    ]}
  ]
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
