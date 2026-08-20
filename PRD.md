# scanspec 2.0 — Product Requirements Document

This document is the single, current statement of design intent for scanspec
2.0. It is written to be read top-to-bottom by a developer joining the
project: it contains the requirements, the design that satisfies them, the
implementation status, and the open questions — without the design history.
Historical rationale and rejected alternatives are recorded in the ADRs under
[docs/explanations/decisions/](docs/explanations/decisions/).

Relationship to other documents:

- **PRD.md** (this file) — requirements and current design intent. Authoritative.
- **API_SPEC.md** — annotated code examples of the consumption API. Where the
  two disagree, the *code* in `src/scanspec/v2/` plus this PRD reflect the
  most recent decisions; API_SPEC.md is updated to match (see §10).
- **ADRs 0001–0007** — all accepted; see `docs/explanations/decisions/` for
  supersession detail between them (several are partially or fully
  superseded by later ADRs in the set).
- Working documents (PRD.md, API_SPEC.md) are deleted or folded into `docs/`
  before the final 2.0 release; they exist for the development period.

---

## 1. Background

Scanspec is a Python library for composably describing scan trajectories at
synchrotron beamlines. A scientist (or a UI) builds a serializable `Spec`
tree describing *what* the scan should do; the library compiles it into a
structure that instrument-control software (ophyd-async) consumes to move
motors and trigger detectors.

scanspec 1.x (`src/scanspec/`) compiles specs into flat arrays of frame
positions (`Path`, `Midpoints`). Every consumer — PandA sequence-table
builders, motor-record flyscan drivers, PMAC trajectory loaders, step-scan
orchestrators — had to walk those internals to re-derive velocities, compute
trigger timing, and implement its own pause/resume. Three required use cases
could not be expressed at all (multi-rate triggering, per-phase detector
streams, servo-rate positions without materialising the whole scan).

scanspec 2.0 is a **breaking release**: no backwards compatibility for JSON
specs or Python APIs. ophyd-async will be updated to the new API. JSON is the
only serialization format (GraphQL deferred). 2.0 is developed as a nested
submodule, `src/scanspec/v2/` (`import scanspec.v2`), alongside the
unmodified 1.x package; it takes over the top-level `scanspec` name only at
final 2.0 release (see §12 for the two-phase migration).

---

## 2. Required user stories

All are in scope for 2.0; they may be delivered in stages.

1. **Servo-cycle-rate motor positions.** Kinematics move up from the motion
   controller into ophyd. Consumers must be able to ask for motor positions
   every servo cycle (e.g. 0.2 ms) instead of per detector frame, in chunks
   (~10 s of data), without scanspec ever materialising the full array.
   scanspec does *not* do kinematic smoothing: it provides a continuous,
   differentiable position function per contiguous stretch of motion;
   discontinuities between stretches are bridged externally by ophyd-async.

2. **Specified livetime and deadtime for detectors.** `duration = livetime +
   deadtime`. Usually a per-spec scalar pair, but per-point patterns must be
   supported (ptychography). When `deadtime` is omitted (`None`),
   ophyd-async fills it in; scanspec does not mandate a value.

3. **Continuously monitored detectors.** E.g. timestamped temperature from a
   PV during a grid scan, or a camera at 10 Hz for the whole scan. These are
   a separate top-level concept, associated with scan data by timestamp only
   — never frame-indexed, never a node in the motion tree.

4. **Multiple detectors at multiple rates.** E.g. SAXS/WAXS detectors taking
   1 frame per point while PandA encoders capture 10× faster. Detectors in
   the same stream must trigger at integer ratios of each other (validated
   at compile time). Detectors in different streams have no phase lock.

5. **Variable spacing in the detector trigger pattern.** Ptychography wants
   variable spacing between exposures while motion stays steady. Under
   centred-livetime semantics (§4.2), a 0.3 s spacing between two 0.1 s
   exposures sharing a 0.01 s minimum deadtime is expressed as three
   `TriggerRepeat`s:
   `(1, 0.1, 0.01)` → `(1, 0.0, 0.29)` → `(1, 0.1, 0.01)` (total 0.51 s),
   placing frames near opposite ends of the period. The motion trajectory
   must cover an integer number of repetitions.

   **Scope**: this is an execution/representation requirement only — the
   compiled `TriggerSequence`/`TriggerRepeat` model (§4.2/§4.3) must be
   *able* to express variable spacing, including via manual `Window`
   construction. A first-class `DetectorGroup`/`Acquire` authoring surface
   for spec authors is not required for 2.0; the design must simply not
   preclude adding one later.

6. **Different detector streams at different phases** (the flagship
   multi-stream pattern). N iterations of: take a diffraction image at
   static energy; flyscan energy up taking 1000 spectroscopy frames; flyscan
   energy back down taking 1000 more. Diffraction data has shape `[N]`,
   spectroscopy `[N, 2, 1000]` — different dimensionality, interleaved in
   time, on the same motion axis.

7. **Motor-controller dispatch by motion type.** A simple servo drive can do
   static positioning and constant-velocity moves; a trajectory controller
   can execute arbitrary curves. The orchestration layer must be able to
   tell, per collection window and per whole scan, which kind of motion it
   is — without re-deriving it from position arrays (§7).

### Optional user stories (design must not preclude; not required for 2.0)

- **Fast shutter in gaps** — acceptable to model the shutter as
  another "axis" restricted to 0/1 if ever needed.
- **Waiting for a sample environment** mid-sequence (else: custom plan).
- **Relative positions** that survive JSON round-trip; resolved at execution
  time in ophyd-async.
- **Ending a segment early** ("until detector value …"). Explicitly a
  *future extension point*: scan shape becomes a maximum, and a completion-
  condition mini-language would be needed. Not implemented in 2.0.

### Non-functional requirements

- **Memory**: deserialization and `compile()` are O(spec complexity), never
  O(scan size). Position arrays are generated on demand, in chunks, from
  numpy-enabled functions.
- **Step scans remain first-class**: the same compiled object drives
  software step scans, not just flyscans.
- **Spec nodes are pydantic `BaseModel`s** (frozen) with positional-argument
  support; compiled output is plain classes/dataclasses with no
  serialization requirement (ADR 0003).
- **Streams have names** (default `"primary"`), naming the Bluesky event
  stream each detector group writes to.

---

## 3. The core model: `Spec` → `compile()` → `Scan` → `Window`

```
Spec (serializable description) → .compile() → Scan (compiled) → for window in scan → Window (pure data)
```

The fundamental unit of iteration is the **collection window**: one
contiguous stretch of motion during which detectors are triggered. Windows
are separated by gaps. A step scan
yields one window per point (e.g. 5000 windows); a flyscan yields one window
per sweep (e.g. 50 row windows for a 50×100 grid).

### 3.1 Construction (`src/scanspec/v2/specs.py`)

Motion is composed first, then `Acquire` attaches acquisition:

```python
motion = Linspace("y", 0, 5, 50) * ~Linspace("x", 0, 10, 100)   # snaked grid
spec = Acquire(motion, fly=True, detectors=[
    DetectorGroup(1, 1, 0.003, 0.001, ["saxs", "waxs"]),
    DetectorGroup(10, 1, 0.000299992, 8e-9, ["timestamp", "x_enc", "y_enc"]),
])
scan = spec.compile()
```

- Motion primitives: `Linspace` (alias `Line`), `Static`, `Range`, `Spiral`,
  `Ellipse`, `Polygon`. `Linspace.bounded` / `Range.bounded` construct from
  extreme bounds.
- Combinators: `Product` (`a * b`, b fast), `Snake` (`~a`), `Zip`
  (`a.zip(b)`), `Concat` (`a.concat(b)`), `Repeat(a, n)`.
- **`Acquire` is the only place `fly=True/False` appears.** It binds a
  motion spec to one named windowed stream (`stream_name="primary"` by
  default), plus `continuous_streams` and `monitors`. `fly=True` means the
  innermost motion dimension sweeps continuously; all outer dimensions step.
- **Multi-stream scans are `Concat`s of `Acquire`s** with different
  `stream_name`s, optionally wrapped in `Repeat` and an outer `Acquire`
  carrying scan-wide monitors. This is how the flagship pattern (§2.6) is
  expressed:

  ```python
  diff = Acquire(Static("e", 7.0), detectors=[diff_det], stream_name="diff")
  up   = Acquire(Linspace("e", 7.0, 7.1, 1000), fly=True, detectors=[spec_det], stream_name="spec")
  down = Acquire(Linspace("e", 7.1, 7.0, 1000), fly=True, detectors=[spec_det], stream_name="spec")
  spec = Acquire(Repeat(diff.concat(up).concat(down), num=200),
                 monitors=[MonitorStream("temperature", "tc1")])
  ```

- `Concat`/`Product`/`Zip` **reject** specs carrying continuous streams or
  monitors: those run in parallel to the *entire* scan, so they may only
  appear on a top-level `Acquire`.
- `Snake` operates on a single-dimension spec (deviation from 1.x). `Zip`
  supports exactly the cases 1.x supported. `Squash` is not needed and was
  dropped — dimensions are never merged.
- Out-of-package `Spec` subclasses are supported: the serialization union is
  rebuilt automatically whenever a `Spec` subclass is defined.

### 3.2 The compiled `Scan` (`src/scanspec/v2/core.py`)

`Scan` is the sole entry point for execution *and* analysis. Construction is
O(spec complexity). It holds:

- `generators: list[WindowGenerator]` — internal motion engines, outer →
  inner (ADR 0004). Only the innermost generator can fly. Sources are
  `LinearSource` (uniform spacing, 1.x fence/post convention),
  `FunctionSource` (arbitrary `fn(indexes) → dict[axis, array]`; spirals and
  masked grids), or `ConcatSource` (sequential children; how concat-of-
  acquires alternates trigger sequences per window).
- `windowed_streams: list[WindowedStream]` — per stream: `name`,
  `dimensions` (its own shape — streams can differ), `detector_groups`.
  Used to **arm** detectors before the scan and to **reshape** data after.
- `continuous_streams: list[ContinuousStream]` and
  `monitors: list[MonitorStream]` — whole-scan acquisition (§2.3).
- `has_moving_axes: bool` and `non_linear: bool` properties — scan-wide
  capability dispatch: a step-only consumer asserts `not has_moving_axes`, a
  linear-flyscan consumer asserts `not non_linear`, a trajectory consumer
  takes anything. (There is deliberately *no* `Scan.fly` flag.)
- `with_start(window, trigger_index)` — pause/resume (§6).

The result of `compile()` is **owned by the caller**, who may freely mutate
it; `compile()` is repeatable and never mutates the spec.

### 3.3 `Window`

Each iteration yields a `Window` — a pure data object with everything needed
to execute one collection phase:

| Field | Meaning |
|---|---|
| `static_axes: dict[AxisT, float]` | Axes to position before the window starts. Contains **only axes that changed** since the previous window; the first window carries all axes. |
| `moving_axes: dict[AxisT, AxisMotion]` | Axes sweeping continuously, each with `start_position`, `start_velocity`, `end_position`, `end_velocity`. Empty for step windows. Disjoint from `static_axes`. |
| `non_linear: bool` | `True` → trajectory controller required (servo-rate positions). `False` → constant velocity per axis (or no motion). Computed analytically from the position functions, not by finite-differencing. |
| `duration: float` | Total window time in seconds, derived from trigger timing (§4). |
| `trigger_sequences` | Detector triggering for this window (§4). |
| `previous: Window \| None` | One step back only — enough to compute the gap into this window. |

`window.positions(times: np.ndarray) -> dict[axis, np.ndarray]` returns
positions for the moving axes at each given real-second time, computed
directly (no chunking, no generator — the caller supplies exactly the times
it wants and owns any iteration/chunking itself). Raises `RuntimeError` on
step windows (no continuous trajectory). Generating those instants
(including any hardware-specific row/edge structure, e.g. PandA
position-compare rows, ADR 0007 Assumption A4) is a consumer-layer concern,
not something the hardware-agnostic `Window` model encodes.

Gaps are out of scope for scanspec: consumers call an external
`calculate_gap(from_pos, from_vel, to_pos, to_vel)` using the
boundary kinematics of adjacent windows. Position functions are therefore
required to be differentiable at window boundaries — a design constraint on
every motion node.

### 3.4 Index convention

The 1.x fence/post convention is kept: integer indexes `0 … N` are window
boundaries (posts); half-integer indexes `0.5 … N−0.5` are collection
midpoints (detector setpoints). `Spiral` starts half a point out from the
centre to avoid the velocity singularity at r=0.

---

## 4. Trigger model

### 4.1 Upfront description vs runtime instruction

Two deliberately separate concepts:

- **`DetectorGroup`** (on `Acquire.detectors`, surfaced via
  `WindowedStream.detector_groups`) — *arming-time* description:
  `exposures_per_collection`, `collections_per_event`, `livetime`,
  `deadtime`, `detectors`. `exposures_per_event = exposures_per_collection ×
  collections_per_event`. Used with the stream's `dimensions` to call
  `StandardDetector.prepare()` before any window is iterated.
- **`TriggerSequence`** (on `Window.trigger_sequences`) — *runtime*
  instruction: `detectors` + a `trigger_repeat: TriggerRepeat(num, livetime,
  deadtime)` + a parallel `children: list[TriggerChild]` of
  integer-multiple-rate sub-groups (§4.3). Baked from `DetectorGroup`s at
  compile time (ADR 0007): every
  recorded collection needs its own trigger, so `num` is computed from
  `exposures_per_event` (`= exposures_per_collection × collections_per_event`),
  not `exposures_per_collection` alone (known gap in §8 for the pending code
  fix). The **parent** `trigger_repeat.num` is `inner_length ×
  exposures_per_event` for flyscan windows, or `exposures_per_event` for step
  windows. Each **child**'s `num` follows the same rule scaled to the
  parent's livetime rather than the whole window — conventionally
  `child_rate_Hz × parent_livetime` (e.g. `8000 Hz × 0.009 s = 72`, §4.3) —
  independent of `inner_length`, since children fire once per *parent
  repeat*, not once per window. `livetime`/`deadtime` must be resolved
  (not `None`) before `compile()`. Consumers find their sequence by matching
  `frozenset(sequence.detectors)` (unique per window, enforced); they read,
  never compute.

A stream's detector group may appear in the trigger sequences of only a subset
of windows (flagship pattern: diffraction fires only in hold windows).

Window `duration` is derived from the root `TriggerSequence`s — the sum of
`num × (livetime + deadtime)` across them, since faster children run inside
the parent livetime and do not extend it; an explicit `Acquire(duration=...)`
must be ≥ the derived value. Detector-less step scans have `duration = 0`;
detector-less fly scans require a supplied duration.

### 4.2 Centred livetime (ADR 0006 — accepted)

Execution order of each repeat is **`½·deadtime → livetime → ½·deadtime`**,
not `livetime → deadtime`. This centres the detector's active window on the
nominal scan position, which position-compare triggering requires;
leading-edge alignment would bias every position by `½·deadtime`. The struct
is unchanged — only the interpretation.

`livetime = 0.0` is explicitly valid: a pure spacing-only repeat. Because
centred semantics already give symmetric spacing, spacers are only needed
when two bursts must be separated by a spacing different from the
intra-burst deadtime — the ptychography pattern, expressed as a list of
`TriggerSequence`s in the window (§4.3):

```python
[TriggerSequence(detectors=dets,        trigger_repeat=TriggerRepeat(num=N1, livetime=livetime1, deadtime=deadtime), children=[]),  # first burst
 TriggerSequence(detectors=frozenset(), trigger_repeat=TriggerRepeat(num=1,  livetime=0.0,        deadtime=spacing),  children=[]),  # spacer
 TriggerSequence(detectors=dets,        trigger_repeat=TriggerRepeat(num=N2, livetime=livetime2, deadtime=deadtime), children=[])]  # second burst
```

### 4.3 The two-level trigger structure (ADR 0007 — accepted)

Sibling `TriggerGroup`s at integer-multiple rates have no structural link —
nothing ties "100 SAXS frames" to "7200 Tetramm samples" during the scan.
ADR 0007 replaces `TriggerPattern` + `TriggerGroup` with two types: a
**`TriggerRepeat`** (`num`, `livetime`, `deadtime`) carrying timing only, and
a **`TriggerSequence`** binding `detectors` to one parent `trigger_repeat`
plus a parallel `children: list[TriggerChild]`. Integer-multiple-rate
sub-groups become parallel children that fire *during each parent
livetime*, so progress through the parent structurally implies progress
through the children:

```python
TriggerSequence(
    detectors=frozenset({"saxs", "waxs"}),
    trigger_repeat=TriggerRepeat(num=100, livetime=0.009, deadtime=0.001),
    children=[
        TriggerChild(detectors=frozenset({"tetramm"}), repeats=[TriggerRepeat(num=72, livetime=0.000124, deadtime=0.000001)]),
        TriggerChild(detectors=frozenset({"panda"}),   repeats=[TriggerRepeat(num=45, livetime=0.000190, deadtime=0.000010)]),
    ],
)
```

`TriggerRepeat`/`TriggerChild`/`TriggerSequence` are pydantic `BaseModel`s,
not plain dataclasses like most compiled output (ADR 0003 Decision 6
carve-out) — `TriggerSequence` is also caller-authored input to
`Acquire.trigger_sequence` and must survive a JSON round trip (e.g. sent to
ophyd-async to have unresolved `livetime`/`deadtime` filled in). `children`
is a list of named entries rather than a dict keyed by the child's
`frozenset` of detectors specifically so this round-trips: a `frozenset`
is not a valid JSON object key.

The `children` list is **parallel** (every entry fires simultaneously
during each parent repeat); each entry's `repeats` is a **sequential**
`list[TriggerRepeat]`. Each child's total duration must be ≤ the parent
livetime, and child detector sets must be disjoint from each other and the
parent — all validated at compile
time. The structure is fixed at **two levels** (parent + one child layer).
One parent plus one child fits in a single PandA SEQ block; each additional
child requires an additional SEQ block — the SAXS/WAXS + Tetramm + PandA
example above has two children, so it costs two SEQ blocks, not one.

`Window.trigger_groups` becomes `Window.trigger_sequences` — an **ordered
sequential list**; there are no parallel sibling streams within a window
(no zipping of unrelated trigger streams with no common checkpoint base).
Compiled specs always produce a single-entry list — a spec-facing authoring
surface for the variable-spacing spacer pattern (§4.2) is not required for
2.0 (§2.5, §11) — so multi-entry lists, including that pattern, only arise
via manual `Window` construction.

ADR 0007 also adds `Scan.active_stream_sets: list[frozenset[str]]` — every
combination of stream names simultaneously active in some window — so a
consumer can validate sequencer-table capacity **up front, without
iterating**: `Acquire` contributes its own singleton (or nothing, if it has
no detectors); `Concat`, `Product`, and `Zip` all union and deduplicate their
children's lists; `Repeat` and `Snake` pass their single inner spec's value
through unchanged. `Concat` of two same-named `Acquire`s dedupes to one
singleton — confirmed by
`test_active_stream_sets_concat_same_name_deduplicates`. A SEQ block has 6
outputs, so independent streams can reuse different outputs of the same
block rather than each needing a separate one — up to 6 streams per block.

This ADR supersedes ADR 0005 and the `positions(TriggerPattern)` signature
of ADR 0006 (§3.3, ADR 0007 Assumption A4).

---

## 5. Analysis — reshaping detector data

Analysis is per stream, from static compiled geometry (never from windows):

- `stream.dimensions: list[Dimension]`, ordered outer → inner; each
  `Dimension` has `axes`, `length`, `snake`, and lazy
  `setpoints(axis, chunk_size=None)` yielding forward-direction coordinate
  arrays (chunked; never allocates more than a chunk).
- Base shape is `[dim.length for dim in stream.dimensions]`; groups with
  `collections_per_event > 1` get an extra inner dimension.
- De-snaking is the caller's job (`dim.snake` tells it where); multiple axes
  can share one dimension (a spiral is one `Dimension` with two axes).
- A `number_of_events` convenience (product of dimension lengths, for
  `StandardDetector.prepare()`) is agreed but not yet implemented (§8).

---

## 6. Pause and resume

Principles (agreed, ADR 0007 context):

- **Emitted data is final.** Re-emitting captured frames was judged too
  disruptive to downstream pipelines. If data is bad, cancel and restart the
  scan; resume is always forward-only — it truncates remaining work, never
  completed work.
- **Progress is a count, not a time.** Hardware tracks completed trigger
  repeats; elapsed time cannot reliably be mapped back to a repeat index
  (variable-spacing patterns). Hence `Scan.with_start(window, trigger_index)`
  returns a new `Scan` whose first yielded window has the first
  `trigger_index` repeats truncated off its trigger sequences and its
  `duration` reduced accordingly. There is no rewind method and no mutable
  iterator state.
- **Blank/spacer repeats (`livetime == 0.0`) never count toward
  `trigger_index`, and are never truncated** — a pause landing anywhere in a
  blank always replays it in full on resume. Gaps are minimum requirements
  (detector readout/recovery time, or a ptychography spacing minimum), so
  overshooting one (by whatever elapsed before the pause plus the pause
  itself) is harmless, while undershooting it — which counting blanks would
  risk, since resume could then skip the blank's unexecuted remainder — is
  not. This assumes at most one blank between live bursts per window (§4.3
  Assumption A1); a window with several blanks before the true resume point
  would replay all of them, not just the one the pause landed in —
  currently unreachable via `compile()`, worth revisiting if a
  variable-spacing authoring surface is ever built (§11).
- Intra-window resume works regardless of how many entries a window's
  `trigger_sequences` list contains — `_truncate_trigger_sequence` walks the
  flat list, counting completed root-level repeats across all of them. (This
  was a real restriction under the old `TriggerGroup` model, which required
  exactly one group per window; that type no longer exists.)
- Within a window, safe pause points are **checkpoints** at root-level
  repeat boundaries. ADR 0007 proposes gating each root repeat on a
  Bluesky-held bit (BITB) in the PandA sequencer table. When the bit
  drops, the consumer polls briefly (~0.2 s) for the sequencer to stall;
  if it has not stalled, the consumer checks whether the current
  `table_line` is a blank row (`livetime = 0`): if so, it reads the
  position immediately without waiting for the next repeat boundary
  (no data is in flight, so any mid-blank position is a safe resume
  point); if not, it waits up to `max_time_between_checkpoints` for the
  hardware to reach the next live-section checkpoint. Either way,
  `lookup_checkpoint_index(table_line)` gives the `trigger_index` passed
  to `with_start`. Pause latency for blank sections is therefore bounded
  by the poll interval, not the repeat duration; long blank spacers do
  not need to be broken into short repeats. After the stall, the
  consumer aborts the sequence and reloads it from the checkpoint via
  `with_start`, rather than rewriting the remaining table in place while
  PandA holds the gate.

---

## 7. Consumer capability dispatch

Three consumer classes, dispatchable from the `Scan` without iterating:

1. **Step-scan capable** — asserts `not scan.has_moving_axes`; moves
   `window.static_axes`, fires `trigger_sequences` per window.
2. **Linear-flyscan capable** (motor record) — asserts `not scan.non_linear`;
   uses `AxisMotion` boundary kinematics to compute ramp distances; never
   needs position arrays.
3. **Trajectory capable** (PMAC etc.) — consumes anything; generates its own
   servo-rate time array (e.g. 0.0002s spacing) and consumes it in
   self-chosen chunks via `window.positions(times)`, bridging windows with
   `calculate_gap`. Chunking is entirely the consumer's responsibility —
   scanspec never materializes more than the times array it's given.

Worked examples of all of these plus the PandA sequence-table builder and
pause/resume are in [API_SPEC.md](API_SPEC.md) §Consumption use cases, and
exercised in `tests/scanspec/v2/test_use_cases.py`.

---

## 8. Implementation status

All 2.0 code is in `src/scanspec/v2/` (tests in `tests/scanspec/v2/`); 1.x
in `src/scanspec/` (everything else) is frozen as a reference. Integration
branch: `v2-dev`.

**Implemented and passing**: all core data structures; all motion primitives
(`Linspace`+`bounded`, `Static`, `Range`+`bounded`, `Spiral`, `Ellipse`,
`Polygon`, `Line`); all combinators; `Acquire`; serialization via dynamic
discriminated union with out-of-package subclass support; centred-livetime
semantics; the ADR 0007 trigger model (`TriggerRepeat`/`TriggerSequence`,
`positions(times: np.ndarray)`, `Scan.active_stream_sets`) — this has fully
replaced ADR 0005/0006's `TriggerPattern`/`TriggerGroup`, which no longer
exist anywhere in the codebase; `with_start(window, trigger_index)`
checkpoint truncation resume via `_truncate_trigger_sequence`; compile-time
validation that same-stream detector groups trigger at integer ratios of
each other (story 4) — `_bake_trigger_sequence` checks that
`parent_livetime / child_period` is (within floating tolerance) a whole
number, in addition to the pre-existing checks that a child group's total
duration fits within the parent livetime and that detector sets are
disjoint; `num` is computed from `exposures_per_event`
(`exposures_per_collection × collections_per_event`) throughout, not
`exposures_per_collection` alone. ADR 0007 is Accepted (§9); the code and
tests it describes are in place on this branch.

**Known gaps and defects**:

1. `Scan.number_of_events` (or per-stream) property.
2. A use-case test mapping `DetectorGroup` + dimensions to an ophyd-async
   `TriggerInfo` for `StandardDetector.prepare()`.
3. `scanspec/v2/__init__.py` exports `TriggerRepeat`/`TriggerSequence` only
   — not yet the full `from scanspec.v2 import core, specs` surface.
4. Serialization test coverage is thin (smoke-test level).
5. Auxiliary modules not ported (nice-to-have, in priority order):
   `plot.py`, `cli.py` + `__main__.py`, `service.py`, `sphinxext.py`.

**Intentionally dropped from 1.x** (rationale in ADR 0003): `Path`,
`Midpoints`, `Slice`, `Squash`, `Mask`/regions, `Fly`, `ConstantDuration`,
`step()`, `fly()`, `get_constant_duration()`, `VARIABLE_DURATION`.

---

## 9. ADR review status

ADR 0006 and ADR 0007 are both **Accepted**, with no open review items.
ADR 0005 is superseded by ADR 0007; ADR 0006 Decision 3 (`positions()`
argument type) is superseded by ADR 0007 in the same pass; ADR 0003
Decisions #1/#2/#5 are likewise superseded by ADR 0007, and Decision 6 has
a carve-out for `TriggerRepeat`/`TriggerSequence`/`TriggerChild` (now
pydantic `BaseModel`s, not plain dataclasses). See
`docs/explanations/decisions/` for the full supersession detail. ADR 0007
Assumption A5 (`TriggerRepeat` needing to regain the unresolved-value
support its ADR 0005 predecessor had) is implemented: `livetime`/`deadtime`
are `float | None`, and `TriggerSequence` round-trips through JSON
(including with unresolved values) now that `children` is a
`list[TriggerChild]` rather than a `frozenset`-keyed dict.

---

## 10. Documentation debt

- **API_SPEC.md predates ADR 0006/0007 entirely** and needs a full rewrite,
  not a targeted patch: its trigger vocabulary throughout (data structures
  and 2 of its 5 worked use cases) is `TriggerPattern`/`TriggerGroup`,
  superseded by `TriggerRepeat`/`TriggerSequence` (already landed — §8); it
  also still names `Window.non_linear_move` (code: `non_linear`),
  `with_start(window, time)` (code: `trigger_index`), and a
  `Scan.fly`/`Scan.motion_dims` field (removed in favour of
  `has_moving_axes`/`non_linear` and `windowed_streams`). Its "Open
  questions §1" (multi-stream `Spec` subclass) is also already resolved —
  by `Concat` of `Acquire`s with different `stream_name`s (§3.1).
- `docs/` (user-facing Sphinx docs) still document 1.x only; they are
  rewritten as part of the final migration, not before.

---

## 11. Open questions

1. Does pause/resume ever need an *end* point as well as a start point?
   (Raised during design; unresolved, currently assumed not.)
2. ~~User-facing surface for variable-spacing trigger patterns~~ **Resolved:
   not needed.** The compiled `TriggerSequence`/`TriggerRepeat` structure
   must be able to express it (already true via manual `Window`
   construction, §4.3), but no dedicated `DetectorGroup`/`Acquire` authoring
   surface is required — hand-built `Window`s remain the accepted way to
   construct this pattern.
3. ~~Does a child `DetectorGroup`'s `livetime` already exclude its own
   `deadtime` when sized against the parent's livetime slot?~~ **Resolved:
   yes.** A child's `livetime` is sized as `parent_livetime/ratio −
   deadtime` (e.g. `0.000299992 = 0.003/10 − 8e-9`), so `ratio` child
   repeats fit exactly inside the parent's livetime — confirmed as the
   intended design, matching `validate_trigger_sequence`'s "child duration ≤
   parent livetime" rule and the tests/examples throughout this document and
   `API_SPEC.md`.

---

## 12. End state (migration)

Two phases, not one shot — this is deliberate, so ophyd-async can start
integrating against the 2.0 API before 2.0 is ready to actually replace 1.x
in production.

**Phase 1** (done): `src/scanspec2/` moved to `src/scanspec/v2/` — a
submodule nested inside the existing, unmodified 1.x package
(`import scanspec.v2`). 1.x is completely unaffected; nothing takes over
the top-level `scanspec` name yet. Stays on `v2-dev`; merging to `main` is
a separate, later decision, not part of this phase.

**Phase 2** (later, once 2.0 is feature-complete on `v2-dev` and all tests
pass):

1. Move `src/scanspec/` (1.x, everything except `v2/`) to `src/scanspec/v1/`
   (kept, not deleted — for consumers not yet migrated) and its tests
   likewise.
2. Promote `src/scanspec/v2/` to the top-level `src/scanspec/` (and
   `tests/scanspec/v2/` to `tests/scanspec/`), updating `pyproject.toml` and
   imports.
3. Rewrite `docs/` for the 2.0 API; verify it reads well end-to-end.
4. Delete the working documents (this PRD, API_SPEC.md) or fold their
   remaining content into `docs/`.
5. Merge `v2-dev` to `main` and release scanspec 2.0; ophyd-async is then
   pointed at it.
