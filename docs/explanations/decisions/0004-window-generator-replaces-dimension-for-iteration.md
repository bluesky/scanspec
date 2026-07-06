# 4. WindowGenerator replaces Dimension for iteration

Date: 2026-04-23

## Status

Accepted

## Context

The initial 2.0 design used `Dimension` (with a position function) as both
the iteration primitive and the analysis descriptor. `Scan` held a list of
`Dimension`s, and `__iter__` derived `Window` objects by evaluating position
functions on each dimension.

This broke down when fly and step scans needed to coexist in the same spec
tree (e.g. `Acquire(Static("e", 7.0), fly=False).concat(Acquire(Linspace("e",
7.0, 7.1, 1000), fly=True))`). A `Dimension` has no concept of fly vs step —
it only knows axes, length, and snake. The `fly` flag lived on `Scan`, but
`Scan` only had one flag for the whole scan. Concat of fly and step acquires
needed per-segment fly/step decisions.

Additionally, `Dimension` conflated two roles: (1) generating windows during
iteration, and (2) describing the scan shape for analysis. These have
different requirements — iteration needs boundary kinematics and trigger
groups; analysis needs axes, length, and setpoints.

## Decision

Split the two roles:

- **`WindowGenerator`** owns iteration. It holds `axes`, `length`, `snake`,
  `fly`, `axis_ranges` (linear) or `position_fn` (nonlinear), `trigger_groups`,
  `duration`, and optional `children` (for concat). Its `windows(reverse)`
  method yields `Window` objects with correct `static_axes` or `moving_axes`
  depending on `fly`. `Scan.generators` is the ordered outer→inner list.

- **`Dimension`** is the analysis descriptor. It holds `axes`, `length`,
  `snake`, and a private `_position_fn` for `setpoints()`. It lives on
  `WindowedStream.dimensions` and is created by `Acquire.compile()` from the
  generator stack.

`Scan.__iter__` iterates the generator stack recursively. Outer generators
always produce step windows (their `static_axes` are merged into the inner
window). Only the innermost generator may be fly. `_iter_with_outer` handles
the recursion.

## Consequences

- **Fly/step coexistence**: Each `WindowGenerator` carries its own `fly` flag.
  Concat of fly and step acquires works by concatenating their inner window
  generators (via `children`).
- **Clean separation**: Iteration machinery (`WindowGenerator`, trigger groups,
  duration) is decoupled from analysis descriptors (`Dimension`, setpoints).
- **`Acquire` creates both**: `Acquire.compile()` reads the generator stack
  to build `Dimension`s for its `WindowedStream`, and configures the
  innermost generator's `fly`, `trigger_groups`, and `duration`.
