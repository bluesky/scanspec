# Scanspec 2.0 design requirements

> **Release scope**: scanspec 2.0 is a major breaking release — aligned in spirit with 1.x but with no backwards compatibility for JSON specs or Python APIs. ophyd-async will be updated to work with the new API; no back-compat is required there either.

## Currently
Scanspec as it is now supports:
- **Specs**: Composable, serializable scan description (Linspace, Product, Snake, Ellipse, etc.) serialized as tagged-union JSON (`"type": "Product"`, …). Pydantic field descriptions drive UI form generation; a FastAPI service lets a UI POST JSON and get back validation and scan previews.
- **Memory efficiency**: `Product` stores a *stack* of compact per-axis `Dimension` arrays (e.g. 100+2000+2000 floats = ~32 kB) rather than expanding the full Cartesian product (3.2 GB). Expansion and snaking are lazy, computed via index arithmetic on demand.
- **Flyscan consumption**: `Path(stack).consume(N)` materialises N-point chunks of `lower`/`midpoint`/`upper`/`gap`/`duration` arrays, feeding motion controllers and detectors batch-by-batch. `Fly(spec)` enables bounds calculation so the motor moves continuously through each frame; `gap` flags turnarounds.

It is currently used in ophyd-async to supply:
- The number of frames for a detector (all the same): `int(np.prod([len(dim) for dim in stack]))`
- The batched points to pass to a motion controller for PVT interpolation: `slice = path.consume(N)` passed to https://raw.githubusercontent.com/bluesky/ophyd-async/refs/heads/main/src/ophyd_async/epics/pmac/_pmac_trajectory_generation.py
- The triggers inserted at each gap in a PandA sequence table: https://github.com/bluesky/ophyd-async/blob/b36695e9146a94a73b3f9689b1d35fd9e4697e6c/src/ophyd_async/fastcs/panda/_fly_logic.py#L102-L152

## Required user stories

There are some user stories we must support with 2.0. All are in scope for the initial 2.0 release; they may be delivered in stages to produce MVPs along the way.

### Servo cycle rate motor positions

At the moment kinematics are in the motion controller, so PVT interpolation will produce compound motor positions every servo cycle which are translated into raw motor positions by the controller. We need to lift these kinematics up into ophyd so they can be used in analysis and so we can do path planning up front for collision avoidance. This means we need a position at each motion servo loop cycle rather than at nominal detector position. This would require us to slice the path in time rather than index by detector frame, saying "I want motor positions every 0.2ms" that require no PVT interpolation

> **Clarification**: scanspec does not need kinematic calculations or motion smoothing. The `Spec` provides a continuous function for axis motion within each contiguous section; discontinuities are exposed as `gap` flags (as today) and ophyd-async handles the trapezoidal bridging. `Spec.calculate()` returns a compact **intermediate structure** (a stack of `Dimension`-like objects) where each dimension holds a reference to a numpy-enabled function; actual position arrays are generated on demand by calling those functions with a time or index array. This keeps deserialization memory cost O(spec complexity) not O(scan size), while allowing arbitrarily dense arrays to be produced in batches at consumption time. A separate `Spec.midpoints()` iterator path must also be maintained for software step-scanning.

### Specified livetime and deadtime for detectors

We also need to be able to specify the livetime (and optionally the deadtime) for detectors rather than the duration. This should be expanded when sent to ophyd-async so that the motion will then get the duration. We still need the ability to just specify duration and ophyd will fill in livetime and deadtime from it.

> **Clarification**: `duration = livetime + deadtime`. Livetime/deadtime is commonly a per-spec scalar pair but must also support per-point arrays (for ptychography variable-gap patterns). The API must expose which mode is in use, as only some detectors support per-point timing. When `deadtime` is omitted, ophyd-async fills it in; scanspec does not mandate a value.

### Supporting continuously monitor detectors

We need the ability to continuously monitor a detector for the entire duration. E.g. capture timestamped temperatures from a PV while doing a grid scan or acquire camera at 10Hz while stretching a sample

> **Clarification**: Continuously monitored detectors are a **separate top-level concept** (not a node in the `Spec` tree). Their output is associated with the scan by timestamp correlation only; no frame-index coupling.

### Multiple detectors at multiple rates

At the moment scanspec has the concept of "scan index" and "duration" and all detectors are expected to be triggered at that rate. We would like to support different detectors at different rates, but still specify the relationship between them (e.g. this one goes 10 times faster than the other, but this one is free-running)

A maximal example to demonstrate would be:
- At each DCM energy in a given range
  - Do an X,Y flyscan scan over a given grid
    - With saxs and waxs pilatus detectors taking 1x frame with 3ms livetime 1ms deadtime
    - With timestamp, X and Y encoders from the PandA taking 10x frames with 0.3ms livetime and 8e-9 deadtime
- Do time triggered detector acquisitions
  - With front and side optical cameras taking 1x frame with 48ms livetime 1ms deadtime
  - With timestamp from PandA

> **Clarification**: Detectors in the **same stream** must trigger at integer ratios of each other (validated at spec creation). Detectors in **different streams** have no phase lock — only timestamps tie them together. Data model is leaning towards `Spec[AxisT, DetectorT]` embedding detector descriptions in the spec tree, but a separate `DetectorSpec` parallel tree is also a candidate; prototype examples needed to decide.

### Gaps in detector trigger pattern

For ptychography we may want to leave variable gaps between detector exposures, so the trigger pattern would look something like "0.1s livetime, 0.01s deadtime, 0.1s livetime, 0.3s deadtime" repeated for each line of a gridded motion pattern. Care needed to make sure the motion trajectory is an integer number of these repetitions.

### Different detector streams

For flagships we have a use case where the diffraction and spectroscopy detectors are both synchronized to the motion, but at different phases with different dimensionality. This looks like:

- Do N iterations of
  - Take a diffraction detector image at static energy e0
  - Flyscan energy from e1 to e2, taking 1000 specstroscopy detector images
  - Flyscan energy from e2 to e1, taking 1000 specstroscopy detector images

The diffraction dimensions are [N], the spectroscopy dimensions are [N, 2, 1000]

### Motor controller dispatch by motion type

Motor controllers have different capabilities: a simple servo drive can only do
static positioning or constant-velocity linear moves, whereas a full trajectory
controller (PMAC trajectory mode, Aerotech, etc.) can execute arbitrary curves.
The orchestration layer needs to know which kind of motion each collection
window represents so it can choose the correct execution path — without having
to rediscover this from the raw boundary kinematics every time.

> **Clarification**: `Window` carries a `non_linear_move: bool` field and
> replaces the four parallel position/velocity dicts with a single
> `moving_axes: dict[AxisT, AxisMotion]` struct (see `AxisMotion` in
> `API_SPEC.md`), so axis-set consistency is structural. Static axes are in
> `static_axes: dict[AxisT, float]`, disjoint from `moving_axes` by
> construction. Three dispatch cases:
>
> - Step scan (no motion): `moving_axes` is empty; `non_linear_move` is `False`.
>   Any controller handles this.
> - Linear (constant velocity): `moving_axes` is non-empty, `non_linear_move`
>   is `False`; `start_velocity == end_velocity` for every axis in `AxisMotion`.
>   Controllers with a linear-interpolation mode can execute this without
>   servo-rate position arrays.
> - Nonlinear: `moving_axes` is non-empty, `non_linear_move` is `True`.
>   Requires servo-cycle-rate position arrays from
>   `path.positions(dt, max_duration)`.
>
> The consumer dispatches with `if/elif` on `window.moving_axes` and
> `window.non_linear_move` (see use case 3 in `API_SPEC.md`). scanspec sets
> `non_linear_move` analytically at path-construction time — only motion nodes
> with a provably constant velocity function (such as `Linspace` and `Static`)
> in the innermost `ScanDimension` produce `non_linear_move=False` windows.

## Optional user stories

These are user stories that we could support, we should consider them in the design, but not if it makes the implementation too messy

### Fast shutter

Most beamlines will be happy with "open the fast shutter at the beginning of the scan and close it at the end", which doesn't need to be expressed in scanspec. However for some beamlines they would like to open and close their shutters during the turnaround gaps. It is acceptable to then specify the motion for the turnaround in scanspec, with shutters being just another "axis" (although one that can only be 0 or 1).

### Waiting for a sample environment

For a purely time-based scan, we would like to express "take 10 frames with 1s livetime, then trigger a sample environment, then wait for an input to say it's done, then take 100 frames with 0.1s livetime".

If we cannot do this we can do it as a custom plan.

### Allowing positions to be specified as relative

Allow positions for axes to be specified as relative instead of absolute. This allows fully relative scans, or repetition of relative axes in inner dimensions to an absolute outer axis.

> **Clarification**: Relative positions must survive JSON round-trip serialization (e.g. `{"type": "RelativeLine", ...}`). "Relative to what" is resolved at execution time in ophyd-async, not at spec-creation time.

### Ending a segment early

Allowing ending one segment of the scan early. This would be useful for something like:

- At each DCM energy in a given range
  - For each X position of sample, until value from detector is 
    - Acquire 1x frame from detector with 0.1s livetime

The complications are that this makes the specified scan shape only a *maximum* shape. It also means we need some kind of "completion condition" mini language. This is probably a requirement to discard, but it would be good to work out where it would fit if we put this in the future.

> **Decision**: Treat as a **future extension point**. The 2.0 design must not preclude it, but it will not be implemented in 2.0. When added, the scan shape becomes a *maximum* shape and a "completion condition" mini-language would be required.

## Non-functional requirements

From developing the original scanspec, here are some requirements from the technical side

### Keeping the memory efficiency

At the moment deserialization does not take any more memory for large scans than small scans. We should keep it that way. We also need to know the size of each dimension taking memory for the whole scan. When we calculate a dimension, instead of creating upper/lower/midpoints/duration, then we should keep a reference to a numpy enabled function that can calculate these on demand from a time array.

### Ensuring we can use these easily for software step scans

Although the driving requirements for this rewrite is flyscans, we also need to be able to use this in step scans. So the `for midpoint in spec.midpoints()` functionality should be maintained, and thought given to how the detectors to be triggered at each point could fit into this API.

### Pydantic BaseModels

Pydantic dataclasses are not well supported, we only use them so we get positional args. If we are doing a rewrite we should switch to BaseModels. We should be able to support positional args by Spec have a custom `__init__(self, *args, **kwargs)` method that binds args to kwargs then calls the superclass. For typing we should be able to use `dataclass_transform` to support this statically. This needs prototyping.

### Bluesky streams

Downstream we will be making a stream for each set of detectors that are triggering at multiples of each other. We should include that stream name in the Spec, defaulting to `primary`.

> **Clarification**: A stream name identifies a group of detectors related by integer trigger ratios. The stream name lives on the detector grouping node, not on individual axes.

## Requirements on the API

We have 2 users of the API:
1. Constructing `Spec` objects that describe the scan, can be serialized and passed between UI and ophyd-async
2. Methods on the `Spec` object and other objects that use `Spec` to allow a scan to be performed

We will focus on 2 first so we can get the requirements for 1.

> **Design approach**: Work from the ophyd-async consumption end of the API first (assuming a `spec` instance exists), then derive what construction must look like. Detector grouping / embedding decision deferred until consumption API is clearer.

### Detector description

The first thing we need to do in a scan is describe what detector data will be produced. This takes the form of:
- Per stream
  - How many events in the stream, or 0 for constantly monitored
  - Per detector in that stream
    - Optionally what is the number of exposures that a detector will process into a single data collection
    - Optionally what is the number of collections that will form a single event in the stream
    - Optionally the static livetime and/or deadtime to be set on the detector

### Software step scan

Then we need to provide enough information to conduct a software step scan. The detector description above allows us to setup the detectors, then we need to call a method on the `Spec` that will provide a python iterator of collection windows, each with a dictionary of axes to move and positions to move to. 

### Flyscan: PMAC

For a flyscan we need to pass positions down to the PMAC at 0.2ms intervals during a collection window, then calculate the turnaround between collection windows using a function that is out of scope of this API. We need to operate on chunks of about 10s of data at a time rather than the entire collection window, and we need these positions as numpy arrays.

> **Clarification**: Ignore the current PVT-based implementation. The new PMAC consumption model is:
> - The term "gap" is retired. Contiguous stretches of motion are called **collection windows**; gaps between them are implied.
> - Consume up to 10s of positions at 0.2ms intervals, finishing early if the end of a collection window is reached
> - Send that batch straight to the motion controller (no PVT interpolation needed — positions are at servo cycle rate)
> - If the end of a collection window is reached, call an external function `calculate_turnaround(from_positions, from_velocities, to_positions, to_velocities)` which returns 0.2ms-interval points bridging the last point/velocity of the outgoing window to the first point/velocity of the incoming window
> - scanspec computes end positions and velocities of the current collection window, and start positions and velocities of the next collection window, as analytical derivatives of the position function at the boundary points
> - Position functions must therefore be differentiable at collection window boundaries — this is a design constraint on Spec nodes

### Flyscan: PandA

For the PandA we need to consume collection windows one by one, and for each produce sequence table rows for detector triggering. A sequence table row consists of:
- The starting trigger (position compare on an axis position, or setpoint trigger from PMAC)
- The number of iterations of this row
- For each iteration the livetime and deadtime
If all detectors in the stream operate at the same rate then this will be a single row per collection window. If there are different detectors at different rates then there will be at least 2 rows per event.

### Path API design

> **Design thoughts**:
> - The existing `path.consume()` was all that was needed to track state; now we have two levels: advancing through collection windows, and tracking time within a window
> - Both PandA and software step scans want to consume window-by-window; PMAC also wants this but needs a separate inner loop within each window for time-sliced position batches
> - `Spec` should only have internal usage methods; `Path` should be the "consume level" API
> - `dt` should be supplied to Path method calls, not to the Path constructor — the creator of Path doesn't know whether a PMAC or PandA will consume it
> - No rewind method on Path. Instead, construct a new Path from a `(window_index, time_within_window)` start point — this is the resume-after-pause primitive
> - `Path` iterates window-by-window; within each window the PMAC does a further time-sliced consumption loop
> - A `Window` is a **pure data object** containing: static positions of axes that don't move during the window; start/end positions and velocities of axes that do move; and detector triggering information
> - `window.previous` is always at most one step back — no need for deeper traversal
> - For step scans, detector triggering information comes from the window just as in the PandA case — detectors are part of the Path/Window, not passed separately to the scan runner

### Rewinding support

We need to support the ability to pause a scan and resume it later. If the pause happens during a flyscan, a signal is raised in the PandA so it completes the current window's triggers; progress (window index and time within window) is read back from the PandA. Resume is achieved by constructing a new `Path` with `start_window` and `start_time` — there is no rewind method on an existing Path.

### Detector group structure

> **Clarifications**:
> - `exposures_per_collection` and `collections_per_event` belong on `TriggerGroup`, not `DetectorInfo`
> - `ratio == exposures_per_event == exposures_per_collection * collections_per_event`
> - `TriggerGroup` should probably be renamed `DetectorGroup`
> - Trigger type (position compare vs time/setpoint) is not part of `window.detectors` — it is passed into `run_panda_flyscan` from an external source (hardware configuration, not scan spec)
> - **Key insight**: split into two separate concepts:
>   1. **Upfront detector description** (from `spec` before any windows are consumed): streams containing grouped detectors, with static livetime/deadtime per group
>   2. **Window detector information** (from each `window`): no streams, just groups, each with a `trigger_pattern` instead of `exposures_per_collection`/`collections_per_event`/`ratio`/`livetime`/`deadtime`
>   - `WindowDetectorGroup.detectors: list[str]` matches `DetectorGroup.detectors: list[str]` — this is the link between window groups and spec-level groups, and hence to the correct sequence table
>   - `DetectorInfo` is redundant — `detectors` is `list[str]` in both `DetectorGroup` and `WindowDetectorGroup`; will be genericised later
>   - `frozenset(detector_names)` uniquely identifies a `WindowDetectorGroup` — exact match, not subset

We need to reform the stack of frames written by each detector into a multi-dimensional grid. This means we need:
- For each dimension in the scan
  - Its length
  - The axes that move in this dimension
  - A method that produces the axis setpoints for axes that move in this dimension
  - Whether it snakes

> **Clarifications**:
> - Analysis comes from `spec.dimensions` directly — static scan geometry, not from `path` or `window`
> - `midpoints` renamed to `setpoints`
> - `dim.setpoints(axis)` always returns coordinates in the forward direction — snaking is indicated by `dim.snakes` and left for the caller to handle
> - `dim.setpoints(axis)` always returns an iterator of arrays; `chunk_size=None` yields a single array — full materialisation is just `next(dim.setpoints(axis))`
> - Multiple axes can share a dimension — e.g. a grid is `[Dim(y), Dim(x)]` but a spiral is `[Dim(x, y)]`
> - Snaking is left to the caller; example:
>   ```python
>   shape = [dim.length for dim in spec.dimensions]
>   data = frames.reshape(shape)
>   for i, dim in enumerate(spec.dimensions):
>       if dim.snakes:
>           # reverse every other slice along this dimension
>           slices = [slice(None)] * len(shape)
>           slices[i] = slice(1, None, 2)
>           data[tuple(slices)] = np.flip(data[tuple(slices)], axis=i)
>   ```

## API design revisions (round 2)

> **Design thoughts**:
> - A monitor stream is just a stream without dimensions — no special `Monitor` type needed, just `Stream` with `dimensions=[]`
> - `WindowDetectorGroup` renamed to `DetectorGroup`; `detectors: list[str]` renamed to `detector_names: list[str]`
> - `PositionChunk` is superfluous — `path.positions()` yields `dict[str, np.ndarray]` directly
> - `collections_per_event` is modelled as an additional detector-specific inner dimension, so `stream.dimensions` are shared between all detectors in that stream; the per-detector dimension from `collections_per_event` is appended per-detector on top of the shared stream dimensions





- **GAP 1 — `window.num_frames`**: RESOLVED — `trigger_pattern` maps directly to sequence table rows, `repeats` comes from each tuple, no `num_frames` needed.
- **GAP 2 — `path.positions(...)` signature**: RESOLVED — `window` is not passed; `path.positions(dt, max_duration)` implicitly uses the current iteration window.
- **GAP 3 — `path.positions` outside loop**: RESOLVED — raises an error; no meaningful current window exists before iteration starts.
- **GAP 4 — `pick_compare_axis` needs velocities**: RESOLVED — only start velocities needed, accessed as `window.moving_axes[axis].start_velocity`, divided by `scale` to get encoder counts/s; positions not required.
- **GAP 5 — step scan livetime/deadtime source**: RESOLVED — `Path` bakes static livetime/deadtime from each stream's `detector_groups` into `trigger_pattern` at path-creation time, giving `[(1, livetime, deadtime)]`; step scan runner uses it identically to PandA.
- **GAP 6 — continuously monitored detectors**: RESOLVED — three stream types named under Option A: (a) `WindowedStream[AxisT, DetectorT]` (formerly `Stream`) — detector stream aligned to collection windows, with `dimensions`; (b) `ContinuousStream[DetectorT]` (formerly `MonitorStream`) — groups detectors that run at a fixed rate for the whole scan (e.g. front_cam + side_cam at 20 Hz) — no dimensions; (c) `MonitorStream[MonitorT]` (formerly `MonitorPV`) — free-running PV sampled on-change, no timing parameters. `Scan.windowed_streams: list[WindowedStream[AxisT, DetectorT]]`; `Scan.continuous_streams: list[ContinuousStream[DetectorT]]`; `Scan.monitors: list[MonitorStream[MonitorT]]`.
- **GAP 7 — stream name on `TriggerGroup`**: RESOLVED (reversed) — `stream_name` was added then removed.  Consumers identify their group by matching `group.detectors` against their own known detector list; since detector sets are unique across groups in a window (invariant), no stream name is needed.  Consumers that need stream membership can derive it from the top-level `Scan.windowed_streams`.
- **GAP 8 — snaking helper**: RESOLVED — left to caller; example added to analysis section.
- **GAP 9 — `window.non_linear_move`**: RESOLVED — `non_linear_move: bool` on `Window` replaces the earlier `WindowMotionType` enum. The four flat boundary dicts are replaced by `moving_axes: dict[AxisT, AxisMotion]` (structural axis-set consistency) and `static_axes: dict[AxisT, float]`. `non_linear_move=False` covers both step scan (empty `moving_axes`) and linear constant-velocity windows; `True` means nonlinear trajectory requiring `path.positions()`. Set analytically at `Path` construction time.

## Current state (April 2026 — final design)

`API_SPEC.md` is frozen as the authoritative specification.  `thoughts.md` is
the rationale and design history; it is finalized and will not be updated
further.  The implementation plan is in the section below.

### What is fully specified (in `API_SPEC.md`)

- All data structures: `TriggerPattern`, `TriggerGroup` (no `stream_name`),
  `AxisMotion`, `Window`, `ScanDimension`, `WindowedStream`, `DetectorGroup`,
  `ContinuousStream`, `MonitorStream`, `Scan`.
- `Spec[AxisT, DetectorT, MonitorT]` as the base class for all scan specs.
- `Acquire[AxisT, DetectorT, MonitorT]` as the concrete `Spec` subclass for
  single-stream scans: wraps a motion spec with `fly: bool`, `detectors`,
  `continuous_streams` (of type `ContinuousStream`), `monitors` (of type `MonitorStream`),
  and `stream_name`.
- `spec.compile()` → `Scan[AxisT, DetectorT, MonitorT]` as the
  compiled intermediate form: sole input to `Path`, sole entry point for
  analysis. `Scan.fly: bool` (not on `ScanDimension`);
  `Scan.windowed_streams: list[WindowedStream[AxisT, DetectorT]]`;
  `Scan.continuous_streams: list[ContinuousStream[DetectorT]]`;
  `Scan.monitors: list[MonitorStream[MonitorT]]`.
- `WindowedStream[AxisT, DetectorT]`: window-aligned detector stream with its own
  `dimensions` and `detector_groups`.
- `ContinuousStream[DetectorT]`: a `WindowedStream`-like container with `name` and
  `detector_groups` but no `dimensions` — for continuously-acquired multi-
  detector groups (e.g. front_cam + side_cam at 20 Hz).
- `MonitorStream[MonitorT]`: `name` + single `detector` — for scalar PV monitors
  (e.g. `dcm_temperature`); no livetime/deadtime.
- `AxisT` must be hashable (dict key).  `DetectorT` and `MonitorT` do not need
  to be hashable; stored in lists throughout the library.
- `TriggerGroup`: `detectors` list + `trigger_patterns`; detector sets unique
  per window (enforced invariant); no `stream_name` field.
- `Path[AxisT, DetectorT]` consumption API: motion-centric;
  `for window in path`, `path.positions(dt, max_duration)`,
  `Path(scan, start_window, start_time)`.
- All five consumption use cases: step scan, PandA fly scan, motor record fly
  scan, PMAC fly scan, pause/resume (both consumers resume from the same
  `start_window` + `start_time`).
- `AxisMotion` dataclass and `moving_axes: dict[AxisT, AxisMotion]` /
  `static_axes` / `non_linear_move: bool` / `duration: float` on `Window`.
- `non_linear_move` and `duration` enable the motor record flyscan use case.
- Analysis uses `stream.dimensions` per stream — each stream has its own shape.
- Memory model.
- Invariants.
- Construction API: motion composition operators, `Acquire` wrapper,
  `acquire.compile()`, maximal example with cameras as `ContinuousStream`s and
  a temperature as `MonitorStream`, serialization format, validation rules.
- Pyright type-inference test in `tests/scanspec2/test_type_inference.py`:
  verifies that `DetectorT` and `MonitorT` are both inferred from constructor
  arguments when `monitors` is supplied; explicit annotation is needed only
  when `monitors` is omitted (to pin `MonitorT=Never`).

### Known gaps (to be resolved during implementation)

1. **Multi-stream `Spec` subclass**: The use case for two streams with
   different dimensionality is addressed by a second `Spec` subclass.  Name and
   construction API TBD.
2. **`Scan` ↔ `Path` interface**: `Scan` as specified does not yet embed the
   per-window position functions needed by `Path`.  Resolution will be found
   during Phase 3–4 implementation; `ScanDimension` will carry an internal
   callable for lazy position generation.

### Technical notes for Phase 3+

- **Pydantic plugin + generic fields**: `list[GenericModel[TypeVar]]` fields
  produce `reportUnknownVariableType` from pyright's pydantic plugin. Using
  `Sequence[GenericModel[TypeVar]]` (from `collections.abc`) resolves this
  without `# type: ignore`. `tuple[T, ...]` also works but is avoided because
  default `()` doesn't equal `[]` — `Sequence` + `default_factory=tuple` is
  the correct pattern.
- **`Union` must be a direct import**: Used inside `if not TYPE_CHECKING:` at
  module import time where PEP 563 deferred evaluation does not apply. It must
  remain in `from typing import (...)` even if not used in annotated positions.

---

## Implementation plan

All new code lives in `src/scanspec2/`; all new tests in `tests/scanspec2/`.
JSON is the only serialization format for 2.0 (GraphQL deferred).

### Phase 1 — Core data structures ✅ DONE

**Files**: `src/scanspec2/core.py`, `tests/scanspec2/test_core.py`  
**What was implemented**: All data classes as plain `dataclasses.dataclass`:
`TriggerPattern`, `TriggerGroup`, `AxisMotion`, `Window`, `ScanDimension`,
`DetectorGroup`, `WindowedStream`, `ContinuousStream`, `MonitorStream`, `Scan`.
`ScanDimension.setpoints()` is a stub raising `NotImplementedError`.  
**Tests**: 14 pytest-style functions — instantiation and field-access only.
No serialisation tests (plain dataclasses carry no serialisation logic).  
**Note**: The original plan called for pydantic `BaseModel`s and JSON
round-trip tests. Decision reversed: data structures are plain dataclasses;
serialisation belongs on `Spec` nodes only (see Phase 2).

### Phase 2 — Motion spec nodes + Acquire ✅ DONE

**Files**: `src/scanspec2/specs.py`, `tests/scanspec2/test_specs.py`,
`tests/scanspec2/test_type_inference.py`  
**What was implemented**:

- `PosargsMeta` — `@dataclass_transform` metaclass that wraps each subclass
  `__init__` at class-creation time to accept positional args; pyright infers
  types from positional constructor calls without per-class stubs.
- `Spec[AxisT, DetectorT, MonitorT]` base pydantic `BaseModel` (frozen): `__mul__`,
  `__invert__`, `.zip()`, `.concat()`, `.compile()` stub.
- `type` is a `@computed_field` on `Spec` returning `type(self).__name__` —
  no per-subclass literal field needed.
- Motion primitives: `Linspace`, `Static` — both `Spec[AxisT, Never, Never]`.
- Combinators: `Repeat`, `Snake`, `Product`, `Zip`, `Concat` — all accept
  `AnySpec[AxisT, DetectorT, MonitorT]` fields so `Acquire` can appear anywhere
  in the tree (e.g. `Concat(acq1, acq2)` and `acq1 + acq2`).
- `Acquire[AxisT, DetectorT, MonitorT]`: `spec: AnySpec[AxisT, Any, Any]`; `fly`;
  `stream_name`; `detectors: Sequence[DetectorGroup[DetectorT]]`;
  `continuous_streams: Sequence[ContinuousStream[DetectorT]]`;
  `monitors: Sequence[MonitorStream[MonitorT]]` — all default to `()`.
  `_validate_unique_detectors` model validator. `__add__` → `Concat`.
  `compile()` stub raises `NotImplementedError`.
- `AnySpec` dual personality:
  - At `TYPE_CHECKING`: `TypeAlias = Spec[AxisT, DetectorT, MonitorT]` with
    free module-level TypeVars — making it a re-subscriptable 3-param generic
    alias so `AnySpec[str, int, float]` → `Spec[str, int, float]` for pyright.
  - At runtime: a class whose `__class_getitem__` ignores params and returns
    `_ANYSPEC_UNION` (a pydantic `Annotated[Union[...], Discriminator(...)]`
    built from `_recursive_subclasses(Spec)`); `__get_pydantic_core_schema__`
    makes `TypeAdapter(AnySpec)` work.
  - `model_rebuild(_types_namespace={"AnySpec": AnySpec})` called on all
    subclasses after the runtime class is defined; required because
    `from __future__ import annotations` (PEP 563) defers all field
    annotations as strings.

**Deviations from original plan**:
- No `MotionSpec` separate type — `Acquire` can appear inside combinators.
  `Acquire.spec: AnySpec[AxisT, Any, Any]` erases detector/monitor type for
  the inner spec so pure-motion nodes (with `Never`) are accepted without error.
- Positional args via `PosargsMeta` metaclass, not a `_positional_init` helper.
- `detectors`/`continuous_streams`/`monitors` use `Sequence[...]` with
  `default_factory=tuple`; defaults are `()` not `[]`.
- `Union` must remain in `from typing import (...)` — it is used inside
  `if not TYPE_CHECKING:` at import time where PEP 563 does not apply.

**Tests**: 27 pytest functions in `test_specs.py` + 2 in `test_type_inference.py`
(41 total across both phases). All pass; `pyright src/scanspec2/ tests/scanspec2/`
reports 0 errors.

### Phase 3 — `spec.compile()` and `ScanDimension.setpoints()`

**Files to modify**: `src/scanspec2/specs.py` (add `compile()` to each node);
`src/scanspec2/core.py` (implement `ScanDimension.setpoints()`; add internal
callable for lazy position generation).  
**New test file**: `tests/scanspec2/test_compile.py`.

**Scope**:

Each motion node's `compile()` method must return a `Scan` with correct
`windowed_streams[0].dimensions` (`axes`, `length`, `snake`). Rules:

- `Linspace("x", 0, 10, 100).compile()` → `Scan(windowed_streams=[WindowedStream(
  name="primary", dimensions=[ScanDimension(axes=["x"], length=100, snake=False)],
  detector_groups=[])], continuous_streams=[], monitors=[], fly=False)`
- `Static("x", 5.0).compile()` → one dimension of length 1 (or `num` if set).
- `Product(outer, inner).compile()` → `inner.dimensions + outer.dimensions`
  (inner is fast/rightmost).
- `Snake(spec).compile()` → same as `spec.compile()` but innermost dimension
  has `snake=True`.
- `Repeat(spec, n).compile()` → prepend a new outer dimension of length `n`
  to `spec.dimensions`.
- `Zip(left, right).compile()` → merge the two innermost dimensions into one
  `ScanDimension` with both axes lists combined (both must have equal length).
- `Concat(left, right).compile()` → same dimensions as left/right (must match),
  with `length` summed for the innermost dimension.
- `Acquire(spec, fly=..., stream_name=..., detectors=[...], ...).compile()` →
  `Scan(windowed_streams=[WindowedStream(name=stream_name,
  dimensions=spec.compile().windowed_streams[0].dimensions,
  detector_groups=list(self.detectors))], continuous_streams=list(self.continuous_streams),
  monitors=list(self.monitors), fly=self.fly)`

**ScanDimension.setpoints()**: must be implemented in `core.py`. A
`ScanDimension` needs an internal callable to produce positions on demand.
Design decision to make during this phase: store a
`_position_fn: dict[AxisT, Callable[[int], np.ndarray]]` dict on the dimension
or a similar lazy mechanism. `setpoints(axis, chunk_size=None)` yields
`np.ndarray`(s); full materialisation is `next(dim.setpoints(axis))`.

**The `Scan ↔ Path` gap**: `Path` (Phase 4) needs boundary kinematics
(position and velocity at start/end of each collection window). This requires
`ScanDimension` to carry not just `length` but a differentiable position
function. Example: `Linspace` → `f(i) = start + i * (stop - start) / (num - 1)`,
velocity `= (stop - start) / (num - 1) / frame_duration`. Decide during this
phase whether to store the callable on `ScanDimension` directly (as a
protocol field) or keep `ScanDimension` a pure dataclass and store it
separately. The former is simpler for `Path`.

**Tests in `test_compile.py`**: dimensions and lengths for all primitives;
operator algebra (`*`, `~`, `.zip()`, `.concat()`); `setpoints()` values
against numpy reference; maximal-example `Scan` structure matches the comment
block in `API_SPEC.md §Maximal example`.

**Context to load for Phase 3**:
- `API_SPEC.md` sections: §Data structures, §`spec.compile()`, §Analysis,
  §Maximal example.
- `src/scanspec2/core.py` (all of it — ScanDimension is defined there).
- `src/scanspec2/specs.py` (all of it — compile() stubs are at the bottom of
  each class).
- `tests/scanspec2/test_core.py` and `tests/scanspec2/test_specs.py` (for
  patterns and fixtures already established).

### Phase 4 — `Path` iteration

**Files**: `src/scanspec2/path.py` (new).  
**Scope**: `Path.__iter__()` yields `Window` objects with correct
`static_axes`, `moving_axes` (`AxisMotion` boundary kinematics),
`non_linear_move`, `duration`, `trigger_groups`, `previous`.
`path.positions(dt, max_duration)` yields `dict[AxisT, np.ndarray]`.
Pause/resume: `Path(scan, start_window, start_time)`.  
**Tests**: `tests/scanspec2/test_path.py` — step scan windows (empty
`moving_axes`), flyscan windows (correct velocities), positions arrays for
`Linspace`, pause/resume constructs at correct offset.  
**Context to load**: API_SPEC.md §Path + §Consumption use cases + Phase 3
output.

### Context-window discipline

- Start a **new conversation** for each phase.  Provide: the relevant
  API_SPEC.md section(s) + the files modified in all prior phases.
- Do **not** load `src/scanspec/` (1.x) unless a specific algorithm is needed.
- If a phase exceeds ~300 lines of new code, split: write implementation first,
  then tests in a follow-up conversation.
- After Phase 4, run the full test suite with `tox -p` to verify integration.
