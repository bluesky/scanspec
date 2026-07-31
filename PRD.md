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
  two disagree, the *code* in `src/scanspec2/` plus this PRD reflect the most
  recent decisions; API_SPEC.md is updated to match (see §10).
- **ADRs 0001–0005** — accepted decisions. **ADR 0006** is tentative,
  **ADR 0007** is proposed; both are incorporated here with their open points
  flagged (§9, §11).
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
only serialization format (GraphQL deferred). 2.0 is developed in parallel in
`src/scanspec2/`; on completion it replaces `src/scanspec/` (see §12).

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

### 3.1 Construction (`src/scanspec2/specs.py`)

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

### 3.2 The compiled `Scan` (`src/scanspec2/core.py`)

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

`window.positions(dt, max_duration)` yields chunked
`dict[axis, np.ndarray]` for the moving axes:

- `dt: float` — positions at a fixed interval (servo-cycle rate).
- `dt: TriggerRepeat` — one position per trigger instant, centred on each
  active window (the 1.x-equivalent "positions at my detector frames").
- Raises `RuntimeError` on step windows (no continuous trajectory).

**Not yet implemented** (ADR 0007 Assumption A4): this signature is replaced
with a plain `positions(times: np.ndarray) -> dict[axis, np.ndarray]`. All
chunking, `max_duration`, and the `TriggerRepeat` overload are removed — the
caller supplies explicit time instants and owns iteration entirely.
Rationale: generating those instants (including any hardware-specific
row/edge structure, e.g. PandA position-compare rows) is a consumer-layer
concern, not something the hardware-agnostic `Window` model should encode.

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
  deadtime)` + a parallel `children` dict of integer-multiple-rate sub-groups
  (§4.3). Baked from `DetectorGroup`s at compile time (ADR 0007): every
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

### 4.2 Centred livetime (ADR 0006 — tentative, agreed in substance)

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
[TriggerSequence(dets,         TriggerRepeat(N1, livetime1, deadtime), {}),  # first burst
 TriggerSequence(frozenset(),  TriggerRepeat(1,  0.0,       spacing),  {}),  # spacer
 TriggerSequence(dets,         TriggerRepeat(N2, livetime2, deadtime), {})]  # second burst
```

### 4.3 The two-level trigger structure (ADR 0007 — proposed)

Sibling `TriggerGroup`s at integer-multiple rates have no structural link —
nothing ties "100 SAXS frames" to "7200 Tetramm samples" during the scan.
ADR 0007 replaces `TriggerPattern` + `TriggerGroup` with two types: a
**`TriggerRepeat`** (`num`, `livetime`, `deadtime`) carrying timing only, and
a **`TriggerSequence`** binding `detectors` to one parent `trigger_repeat`
plus a parallel `children` dict. Integer-multiple-rate sub-groups become
parallel children that fire *during each parent livetime*, so progress
through the parent structurally implies progress through the children:

```python
TriggerSequence(
    detectors=frozenset({"saxs", "waxs"}),
    trigger_repeat=TriggerRepeat(num=100, livetime=0.009, deadtime=0.001),
    children={
        frozenset({"tetramm"}): [TriggerRepeat(num=72, livetime=0.000124, deadtime=0.000001)],
        frozenset({"panda"}):   [TriggerRepeat(num=45, livetime=0.000190, deadtime=0.000010)],
    },
)
```

The `children` dict is **parallel** (keys fire simultaneously during each
parent repeat); each value is a **sequential** `list[TriggerRepeat]`. Each
child's total duration must be ≤ the parent livetime, and child detector sets
must be disjoint from each other and the parent — all validated at compile
time. The structure is fixed at **two levels** (parent + one child layer),
which fits in a single PandA SEQ block.

`Window.trigger_groups` becomes `Window.trigger_sequences` — an **ordered
sequential list**; there are no parallel sibling streams within a window
(no zipping of unrelated trigger streams with no common checkpoint base).
Compiled specs produce a single-entry list, except for the variable-spacing
spacer pattern (§4.2); longer multi-entry lists can also arise from manual
`Window` construction.

ADR 0007 also adds `Scan.active_stream_sets: list[frozenset[str]]` — every
combination of stream names simultaneously active in some window — so a
consumer can validate sequencer-table capacity **up front, without
iterating**: `Acquire` contributes its own singleton (or nothing, if it has
no detectors); `Concat`, `Product`, and `Zip` all union and deduplicate their
children's lists; `Repeat` and `Snake` pass their single inner spec's value
through unchanged. `Concat` of two same-named `Acquire`s dedupes to one
singleton — confirmed by
`test_active_stream_sets_concat_same_name_deduplicates`.

This ADR is **not yet accepted**; see §9 and §11 for its review status. When
accepted it supersedes ADR 0005 and the `positions(TriggerPattern)` signature
of ADR 0006 (which becomes `positions(TriggerRepeat)`).

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
3. **Trajectory capable** (PMAC etc.) — consumes anything; streams
   `window.positions(dt=0.0002, max_duration=10.0)` chunks and bridges
   windows with `calculate_gap`. **Not yet implemented** (§8; ADR 0007
   Assumption A4): this call site moves to `window.positions(times)`, where
   the PMAC consumer itself generates the dense time array (still spaced at
   the servo cycle, still bridged across windows via `calculate_gap`) —
   `max_duration`-based chunking disappears along with the chunking logic
   inside `positions()`.

Worked examples of all of these plus the PandA sequence-table builder and
pause/resume are in [API_SPEC.md](API_SPEC.md) §Consumption use cases, and
exercised in `tests/scanspec2/test_use_cases.py`.

---

## 8. Implementation status

All 2.0 code is in `src/scanspec2/` (tests in `tests/scanspec2/`); 1.x in
`src/scanspec/` is frozen as a reference. Integration branch: `v2-dev`.

**Implemented and passing**: all core data structures; all motion primitives
(`Linspace`+`bounded`, `Static`, `Range`+`bounded`, `Spiral`, `Ellipse`,
`Polygon`, `Line`); all combinators; `Acquire`; serialization via dynamic
discriminated union with out-of-package subclass support; centred-livetime
semantics; the ADR 0007 trigger model (`TriggerRepeat`/`TriggerSequence`,
`positions(float | TriggerRepeat)`, `Scan.active_stream_sets`) — this has
fully replaced ADR 0005/0006's `TriggerPattern`/`TriggerGroup`, which no
longer exist anywhere in the codebase; `with_start(window, trigger_index)`
checkpoint truncation resume via `_truncate_trigger_sequence`; compile-time
validation that same-stream detector groups trigger at integer ratios of
each other (story 4) — `_bake_trigger_sequence` checks that
`parent_livetime / child_period` is (within floating tolerance) a whole
number, in addition to the pre-existing checks that a child group's total
duration fits within the parent livetime and that detector sets are
disjoint. (ADR 0007's formal maintainer sign-off is still pending — see
§9 — but the code and tests it describes are already in place on this
branch.)

**Known gaps and defects**:

1. `window.positions(float dt)`: `max_duration < dt` yields a zero-size
   chunk and loops forever (review finding). **Resolution path changed**:
   rather than adding a guard, this is obsoleted entirely once
   `positions()`'s signature changes to `positions(times: np.ndarray)`
   (ADR 0007 Assumption A4) — not yet implemented. `max_duration` and all
   chunking logic are removed, so the zero-size-chunk failure mode has no
   code path left to occur in.
2. `Scan.number_of_events` (or per-stream) property.
3. `_bake_trigger_sequence` computes `num` from `exposures_per_collection`
   alone, omitting `collections_per_event` — undercounts triggers whenever
   `collections_per_event > 1` (currently untested: every `DetectorGroup` in
   the test suite uses `collections_per_event=1`, masking the omission). Fix
   is to use `exposures_per_event` throughout (§4.1); code and test fixes
   pending.
4. A use-case test mapping `DetectorGroup` + dimensions to an ophyd-async
   `TriggerInfo` for `StandardDetector.prepare()`.
5. `scanspec2/__init__.py` exports `TriggerRepeat`/`TriggerSequence` only —
   not yet the full `from scanspec2 import core, specs` surface.
6. Serialization test coverage is thin (smoke-test level).
7. Auxiliary modules not ported (nice-to-have, in priority order):
   `plot.py`, `cli.py` + `__main__.py`, `service.py`, `sphinxext.py`.

**Intentionally dropped from 1.x** (rationale in ADR 0003): `Path`,
`Midpoints`, `Slice`, `Squash`, `Mask`/regions, `Fly`, `ConstantDuration`,
`step()`, `fly()`, `get_constant_duration()`, `VARIABLE_DURATION`.

---

## 9. In-flight design changes

- **ADR 0006** (centred livetime, originally `positions(TriggerPattern)`):
  status *Tentative*. The centred-livetime substance is agreed and
  implemented; review wording corrections have been applied. In the code,
  ADR 0007's replacement has already landed — `TriggerPattern` no longer
  exists, and `positions()` takes `float | TriggerRepeat` — even though
  neither ADR's formal status reflects this yet. To be marked Accepted
  (with the `positions()` argument-type description updated to
  `TriggerRepeat`) after maintainer sign-off.
- **ADR 0007** (two-level trigger structure + checkpoint pause/resume):
  status *Proposed*. The structural decisions (`TriggerSequence` /
  `TriggerRepeat`, parallel children dict, two-level depth, sequential root
  list, forward-only resume, structural `active_stream_sets`) reflect
  maintainer direction. The draft incorporates the maintainer's hardware
  corrections: the checkpoint gate bit is **BITB** (BITA is already used for
  motion-controller sync at window boundaries); stall detection polls the SEQ
  block **STATE** field, not `TABLE_LINE`/`LINE_REPEAT`; **two trigger levels
  fit in a single SEQ block** (no chained tables); and the SEQ encoding is
  given as a concrete worked gate-row sub-table rather than the discarded
  "collapse N repeats into one row" scenario. The pause hardware sequence
  (abort-and-reload from checkpoint, §6) is now resolved. When 0007 is
  accepted: mark **ADR 0005 as superseded by 0007**, and annotate ADR 0006
  accordingly.

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
2. **User-facing surface for variable-spacing trigger patterns**: not
   required for 2.0 (§2.5) — only the compiled `TriggerSequence`/
   `TriggerRepeat` structure needs to be able to express it. If a
   `DetectorGroup`/`Acquire` authoring surface is added later, candidate
   shapes include a short, fixed, tileable pattern (repeats identically
   across the window), a fully arbitrary explicit per-exposure list, or a
   third shape from an earlier design (leading/trailing half-gap spacers
   around uniform middle frames, see `CONTEXT.adr.260513.md`) that differs
   from the burst-spacer-burst example in §4.2. Left open for whenever that
   surface is actually built.
3. **Does a child `DetectorGroup`'s `livetime` already exclude its own
   `deadtime` when sized against the parent's livetime slot?** The §3.1
   worked example (`DetectorGroup(10, 1, 0.0003, 8e-9, ...)` against a
   parent `livetime=0.003`) predates the ADR 0007 parent/child
   restructuring and was never rechecked against `_bake_trigger_sequence`'s
   "child duration ≤ parent livetime" rule: `10 × (0.0003+8e-9) =
   0.00300008` exceeds `0.003`. Tests currently assume the former (child
   livetime is sized as `parent_livetime/ratio − deadtime`, e.g.
   `0.000299992`); the §3.1 example itself has not been corrected pending
   maintainer confirmation.

---

## 12. End state (migration)

When `src/scanspec2/` is feature-complete on `v2-dev` and all tests pass:

1. Delete `src/scanspec/` (1.x) and its tests.
2. Rename `src/scanspec2/` → `src/scanspec/` (and `tests/scanspec2/`),
   updating `pyproject.toml` and imports.
3. Rewrite `docs/` for the 2.0 API; verify it reads well end-to-end.
4. Delete the working documents (this PRD, API_SPEC.md) or fold their
   remaining content into `docs/`.
5. Merge `v2-dev` to `main` and release scanspec 2.0; ophyd-async is then
   pointed at it.
