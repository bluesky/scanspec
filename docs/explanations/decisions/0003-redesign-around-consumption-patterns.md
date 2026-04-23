# 3. Redesign scanspec around consumption patterns (scanspec 2.0)

Date: 2026-04-23

## Status

Accepted

## Context

scanspec 1.x defines a composable spec tree (`Linspace`, `Product`, `Snake`, etc.)
that compiles through `Path` and `Midpoints` into position arrays. Consumers in
ophyd-async (PandA sequence table builders, motor record fly-scan drivers, PMAC
trajectory loaders, and step-scan orchestrators) all needed to:

- Walk the spec tree or `Path` internals to derive trigger timing.
- Compute motor velocities from finite-differencing position arrays.
- Implement their own pause/resume by tracking iteration state externally.
- Derive scan dimensions for data reshaping from the spec tree rather than a
  stable compiled output.

This led to duplicated, fragile logic across consumers. Each new consumer
reimplemented velocity derivation, trigger counting, and dimension extraction.
The `Squash` combinator added complexity that was never needed in practice.
`Fly` and `ConstantDuration` were bolted on as spec-level wrappers rather than
being integral to the compilation pipeline.

Meanwhile, the 1.x `Spec` base was a pydantic `BaseModel` with discriminated
union serialisation. The spec tree (construction side) benefits from pydantic's
validation and JSON round-trip capabilities. The compiled output (consumption
side) has no serialisation requirement — it is constructed in-process and
consumed immediately.

## Decision

Redesign scanspec around the consumption API:

1. **`Spec.compile()` → `Scan` → `Window`**: A spec tree compiles into a
   `Scan` (iterable), which yields `Window` objects. Each `Window` is a pure
   data object carrying everything the consumer needs: `static_axes`,
   `moving_axes` (with `AxisMotion` boundary kinematics), `trigger_groups`,
   `duration`, `non_linear_move`, and a `previous` link.

2. **`Acquire` as the detector-binding boundary**: `Acquire` wraps a pure
   motion spec and attaches `DetectorGroup`s, `ContinuousStream`s,
   `MonitorStream`s, and `fly: bool`. It is always the outermost spec node.
   `compile()` bakes `DetectorGroup` timing into `TriggerPattern`s on each
   `Window`.

3. **Boundary kinematics, not position arrays**: `AxisMotion` provides
   `start_position`, `start_velocity`, `end_position`, `end_velocity` for
   each moving axis. Consumers compute acceleration ramps or position-compare
   thresholds directly — no finite-differencing of arrays.

4. **Per-stream dimensions for analysis**: `scan.windowed_streams` gives each
   named detector stream its own `dimensions` list. `Dimension.setpoints()`
   yields coordinate arrays. Analysis code reshapes frame stacks using
   `[dim.length for dim in stream.dimensions]`.

5. **First-class pause/resume**: `scan.with_start(window, time)` returns a new
   `Scan` starting from a known progress point.

6. **Spec nodes remain pydantic `BaseModel`s** for serialisation. Compiled
   output (`Scan`, `Window`, `Dimension`, etc.) uses plain classes and
   `dataclass`es — no pydantic overhead, no serialisation requirement.

7. **Drop `Squash`, `Mask`, `Fly`, `ConstantDuration`**. `Squash` was never
   needed (dimensions are not merged). `Mask` and regions were already removed
   in late 1.x; `Ellipse` and `Polygon` became `Spec` subclasses. `Fly` and
   `ConstantDuration` are replaced by `Acquire(fly=True)` and detector-derived
   timing.

## Consequences

- **Simpler consumers**: PandA, motor record, PMAC, and step-scan code all
  iterate `Window` objects with pre-computed fields. No spec tree walking.
- **Single source of truth**: `Scan` is the sole entry point for both
  iteration and analysis.
- **Migration path**: `src/scanspec2/` is developed in parallel with
  `src/scanspec/`. When feature-complete, scanspec2 replaces scanspec and
  the package version bumps to 2.0.
- **Breaking change**: All consumers must update to the new `Window`-based
  iteration API. The spec construction API (`Linspace`, `Product`, `~`,
  `*`, etc.) is largely preserved.
