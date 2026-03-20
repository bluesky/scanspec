# PRD: scanspec 2.0

**Version**: 0.1 draft | **Date**: 19 March 2026 | **Status**: Draft

## Executive Summary

scanspec 2.0 is a breaking-change rewrite that extends the scan-path description model to cover multi-rate detector triggering, per-point livetime/deadtime, continuously monitored detectors, and relative-position axes — while preserving the core design: compact serializable specs, lazy numpy-based path generation, and a clean API for both flyscans and software step scans.

## Hardware Context

The primary hardware targets that inform the data model:

- **PandA** (PandABox): FPGA-based position capture and trigger device. Runs a *sequence table* of position/time-gated pulses loaded upfront; captures encoder positions at servo-cycle rates (~0.1–1 ms). In 1.x ophyd-async fills this table from `Path.consume()` chunks, inserting extra rows at `gap` points for turnaround. In 2.0 it must also handle multi-rate trigger rows within a single table (e.g. 10× PandA encoder rows per 1× Pilatus row).
- **Pilatus** (DECTRIS): Photon-counting pixel X-ray detector for SAXS/WAXS. Needs a hardware gate signal per frame; `livetime` = gate-open duration, `deadtime` = readout period. Does not support per-point variable deadtime (fixed readout hardware).
- **PMAC**: Motion controller that accepts PVT (position-velocity-time) trajectory tables from ophyd-async. In 2.0 it will instead receive dense per-servo-cycle position arrays computed from the `Dimension` functions.
- **Optical cameras**: Time-triggered (not position-triggered); free-running relative to the X-ray detectors; correlated via timestamps only.

## Maximal Example

This example must be expressible as a single serializable spec. It is the primary integration test target.

```
At each DCM energy in a given range:          # outer Product axis: "energy"
  In parallel:
    Stream "primary" — X,Y grid flyscan:
      Motion: X,Y axes fly a snaked grid
      Detectors (integer-ratio group):
        SAXS Pilatus:  1× frame, 3 ms livetime, 1 ms deadtime
        WAXS Pilatus:  1× frame, 3 ms livetime, 1 ms deadtime
        PandA (XY enc + timestamp): 10× frames, 0.3 ms livetime, 8 ns deadtime
          (10× rate validated at spec creation; same motion, different gate)

    Stream "cameras" — time-triggered, no motion:
      Detectors:
        Front optical camera: 1× frame, 48 ms livetime, 1 ms deadtime
        Side optical camera:  1× frame, 48 ms livetime, 1 ms deadtime
        PandA timestamp:      1× frame (phase-locked to cameras)
```

Key constraints this example exercises:
- `Product` of an outer energy axis over a parallel inner spec.
- `Fly` with snaking on inner axes.
- Two independent streams with no phase lock between them.
- Integer-ratio detector grouping (10× PandA vs 1× Pilatus) validated at spec creation.
- Per-spec scalar livetime/deadtime on all detectors.
- `gap` flags at row turnarounds in the grid.
- The full spec serializes to JSON and round-trips without loss.
## Goals

**Carried forward from 1.x (must not regress)**
1. Composable, Pythonic spec construction via operators (`*` Product, `~` Snake, `@` duration, `.zip()`, `.concat()`) and named classes.
2. All specs fully serializable as tagged-union JSON and round-trippable via `spec.serialize()` / `Spec.deserialize()`.
3. CLI tool (`scanspec plot`) for quick spec inspection.
4. `spec.shape()` returns the per-dimension frame counts without materialising any arrays.
5. Snaking (`~`) and `Squash` with path-integrity checking (`check_path_changes`) preserved.
6. `gap` flags continue to be produced at every discontinuity in the scan path.

**New in 2.0**
7. Express multi-rate, multi-stream detector triggering within a single `Spec`.
8. Support per-spec and per-point livetime/deadtime for detectors.
9. Support relative-position axes that round-trip through JSON; resolved at execution time.
10. Provide a separate `Monitor` concept for continuously acquired detectors (timestamp-correlated).
11. Memory-efficient lazy expansion: `calculate()` returns an intermediate structure of `Dimension`-like objects holding numpy-enabled functions; arrays are generated on demand by callers in batches.
12. Maintain two execution paths: `calculate()` (batched numpy array generation) and `midpoints()` (point-by-point iterator).
13. Switch from pydantic dataclasses to pydantic `BaseModel` with positional-arg ergonomics.
14. Auto-generate a JSON schema covering all new fields.

**Out of scope for 2.0 — future service layer**
A query service (REST or GraphQL) will be added after the core library is stable. See NF-4 for serialization design constraints that keep both options open.

## Non-Goals

- Kinematic calculations or motion smoothing (owned by ophyd-async).
- Backwards compatibility with scanspec 1.x JSON or Python APIs.
- Backwards compatibility with existing ophyd-async consumers (they will be updated).
- "End a segment early" (future extension point; design must not preclude it).
- Physics / collision-avoidance path planning (ophyd-async or higher layer).

## Required User Stories

### US-1: Servo-cycle-rate motor positions
**As** a motion controller interface, **I want** dense numpy arrays of axis positions at arbitrary time resolution, **so that** I can program trajectories without PVT interpolation inside the controller.

*Acceptance criteria*: `calculate()` returns an intermediate `Dimension` stack; calling the embedded numpy function with a time/index array produces dense position arrays; `gap` flags are present; no kinematic logic in scanspec.

### US-2: Livetime and deadtime for detectors
**As** a detector interface, **I want** livetime (and optionally deadtime) per spec or per point, **so that** gate widths can be configured precisely.

*Acceptance criteria*: `duration = livetime + deadtime`; per-spec scalar and per-point array modes both supported; API exposes which mode is in use; omitting `deadtime` leaves it to ophyd-async; specifying only `duration` remains valid.

### US-3: Continuously monitored detectors
**As** a beamline operator, **I want** to designate detectors as continuously active for the whole scan, **so that** their data is captured independently of the main trigger pattern.

*Acceptance criteria*: A `Monitor` top-level concept (separate from `Spec`) takes a detector and optional rate; output is timestamp-correlated with the scan only.

### US-4: Multiple detectors at multiple rates in named streams
**As** a multi-detector experiment, **I want** to group detectors into named Bluesky streams where rates are integer multiples of one another, **so that** e.g. a PandA encoder fires 10× per Pilatus frame while cameras free-run in a separate stream.

*Acceptance criteria*: Each detector group carries a `stream` name (default `"primary"`); integer-ratio constraint validated at spec creation; the maximal DCM + grid flyscan + Pilatus/PandA/cameras example is expressible in a single spec.

### US-5: Variable-gap trigger patterns (ptychography)
**As** a ptychography user, **I want** per-point livetime/deadtime arrays, **so that** the trigger pattern varies within a scan line while motion stays continuous.

*Acceptance criteria*: Per-point duration arrays supported (see US-2); motion trajectory length validated as an integer multiple of the trigger pattern period.

### US-6: Relative position axes
**As** a scan author, **I want** axis positions as relative offsets, **so that** the same spec is reusable at different absolute positions set at execution time.

*Acceptance criteria*: `RelativeLine` (or equivalent) round-trips through JSON with its relative flag intact; absolute and relative axes can be mixed in the same spec tree; resolution to absolute happens in ophyd-async.

## Optional / Future User Stories

| ID | Story | Status |
|----|-------|--------|
| US-F1 | Fast shutter as binary axis | No scanspec changes needed; use existing `gap` support |
| US-F2 | Wait for sample environment input | Implement as custom Bluesky plan |
| US-F3 | End a segment early on detector condition | Future extension point; 2.0 design must not preclude it |

## Non-Functional Requirements

### NF-1: Memory efficiency
`calculate()` must not pre-expand products. A 100×2000×2000 scan stores ~32 kB (compact stack), not 3.2 GB. Each `Dimension` in the stack holds a **reference to a numpy-enabled function** that computes positions on demand from an index or time array; no arrays are allocated at `calculate()` time. Deserialization cost is O(spec complexity), not O(scan size).

### NF-2: Two execution code paths
- **`calculate()`** — returns a compact intermediate `Dimension` stack; consumers call the embedded functions with a batch index/time array to get numpy position arrays for that batch. Used for hardware-synchronized flyscans.
- **`midpoints()`** — lazy `{axis: value}` iterator; used for software step scans.

Both must exist for any `Spec`.

### NF-3: Pydantic BaseModel with positional args
Switch from pydantic dataclasses to `BaseModel`. Custom `__init__(*args, **kwargs)` binds positional args to fields; `dataclass_transform` used for static type-checker support. **Requires a prototype before committing.**

### NF-4: Serialization and future service compatibility
All nodes serialize as tagged-union JSON (`"type": "..."`). Relative-position nodes preserve their relative flag. All new fields (detector groupings, livetime/deadtime, stream names) are included in the auto-generated schema.

The `type` discriminator field must be treated as a **serialization-layer concern only** — injected by pydantic's schema machinery, not embedded in core domain logic. Python `isinstance` checks and the class hierarchy are the ground truth for type dispatch.

This matters for the future service layer:
- A **REST/JSON** service uses the `type` field naturally as the JSON discriminator.
- A **GraphQL** service (e.g. via Strawberry) exposes the Python class hierarchy as proper GraphQL union types, using GraphQL's built-in `__typename` as the discriminator. An explicit `type: String` field on the GraphQL schema would be redundant and confusing alongside `__typename`, and should be omitted from the schema even if present on the pydantic model. Strawberry's `strawberry.union` with `isinstance`-based `__resolveType` handles this without any changes to the core model — provided `type` is not load-bearing in Python-side logic.

Conclusion: keep `type` out of any dispatch logic in the core; let pydantic own it purely for JSON round-trips.

### NF-5: Stream names on detector groupings
Detector groupings carry a `stream` field (default `"primary"`). A stream = a set of detectors at integer trigger ratios of each other.


## MVP Milestones

Stories may be delivered in this order to unlock early integration testing:

| Milestone | Includes | Unlocks |
|-----------|----------|---------|
| **M1 — Core rewrite** | NF-3 prototype (BaseModel+positional args), NF-1/NF-2 (intermediate `Dimension` structure), all 1.x spec types ported, NF-4 serialization, `spec.shape()`, CLI | Foundation for all subsequent work |
| **M2 — Flyscan basics** | US-1 (servo-cycle arrays from `Dimension` functions), US-2 scalar livetime/deadtime, `gap` flags, single-stream single-rate | ophyd-async PMAC + PandA integration |
| **M3 — Multi-rate streams** | US-4 (integer-ratio detector groups, stream names), resolve Open Question 1 (data model) | Maximal example partially expressible |
| **M4 — Maximal example** | US-3 (`Monitor`), parallel streams with no phase lock, full maximal example expressible and serializable | End-to-end integration test |
| **M5 — Advanced** | US-5 (per-point livetime/deadtime), US-6 (relative axes) | Ptychography and portable specs |

## Story Dependencies

```
NF-3 (BaseModel prototype)
  └─ M1 (core rewrite + 1.x port)
       ├─ US-1 (servo arrays)  ──────────────── M2
       ├─ US-2 scalar (livetime/deadtime) ───── M2
       │    └─ US-2 per-point ───────────────── M5 (US-5)
       ├─ Open Q1 resolved (data model)
       │    └─ US-4 (multi-rate streams) ────── M3
       │         └─ US-3 (Monitor) ──────────── M4
       └─ US-6 (relative axes) ─────────────── M5
```

## Prototypes Required Before M1

1. **BaseModel + positional args** (Open Q3): Prove that a `Spec` subclass can accept positional constructor args while being a `pydantic.BaseModel`, with `dataclass_transform` giving correct static types. Timebox: 1 day.
2. **Intermediate `Dimension` structure** (Open Q1 dependency): Define what `calculate()` returns — specifically what the callable interface on each `Dimension` looks like (signature, return type, batch semantics). Sketch for both the 1D and Product cases. Timebox: 1 day.
3. **Data model sketch** (Open Q1): Write `Spec[AxisT, DetectorT]` and a separate `DetectorSpec` version of the maximal example in pseudo-code/pseudoJSON. Choose one. Timebox: half day.


|---|----------|--------|
| 1 | `Spec[AxisT, DetectorT]` vs separate `DetectorSpec` tree — needs prototype examples | Core data model |
| 2 | How are per-point durations validated against motion trajectory length for ptychography? | US-5 complexity |
| 3 | `dataclass_transform` for positional-arg `BaseModel` — needs prototype | NF-3 risk |
| 4 | How does `Monitor` integrate with the Bluesky `RunEngine` message protocol? | Out of scope but affects API shape |

## Success Criteria

- The maximal multi-rate example (DCM + grid flyscan + Pilatus/PandA/cameras) is expressible and serializable as a single spec.
- A 100×2000×2000 scan occupies ~32 kB after `calculate()`; dense arrays are only allocated when a consumer requests a batch.
- `calculate()` returns a `Dimension` stack with callable functions; `midpoints()` returns an iterator — both work for any spec.
- All original 1.x composability operators (`*`, `~`, `@`, `.zip()`, `.concat()`) work on 2.0 specs.
- `spec.serialize()` / `Spec.deserialize()` round-trips correctly for all spec types including relative-position nodes.
- CLI plotting works for all new spec types.
- All required user stories have passing integration tests against a mock ophyd-async consumer.
- JSON schema auto-generated from Python types, covering all new fields.

