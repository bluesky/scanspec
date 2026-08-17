# 6. TriggerPattern centred-livetime semantics and Window.positions() signature

Date: 2026-05-13

## Status

Accepted (partially superseded by ADR 0007)

- Decision 3 (`Window.positions()` accepts `float | TriggerPattern`) — superseded by ADR 0007.

## Context

**Livetime ordering**: ADR 0005 defined `TriggerPattern(repeats, livetime,
deadtime)` but did not specify whether execution was `livetime → deadtime`
or some centred arrangement. Hardware implementations (specifically the PandA
sequencer table and position-compare triggering) require the detector's active
window to be centred on the nominal scan position. With `livetime → deadtime`
semantics the active window is leading-edge-aligned, which produces a
systematic position error proportional to `½·deadtime`.

**Zero livetime**: The variable-gap ptychography use case requires
a pure dead-gap spacer `TriggerPattern` with `livetime=0.0` to encode
non-uniform inter-trigger spacing. The original design did not explicitly
address this case.

**`Window.positions()` argument type**: The original signature accepted only a
`float dt` (servo-cycle interval). Consumers that want positions at each
trigger instant — equivalent to what scanspec 1.x returned — had no
first-class way to express this. A `TriggerPattern` carries all the
information needed to derive those instants, so it is a natural argument type.

## Decision

1. **`TriggerPattern` execution is centred**: the execution order for each
   repeat is `½·deadtime → livetime → ½·deadtime`. The struct fields are
   unchanged; only the interpretation changes. The total period per repeat
   remains `livetime + deadtime`.

2. **`livetime=0.0` is explicitly valid**: a `TriggerPattern(repeats, 0.0,
   deadtime)` is a pure dead-gap spacer. Because centred-livetime semantics
   already produce symmetric `½·deadtime` gaps around each active window,
   there is no need for explicit leading/trailing spacers in uniform sequences.
   The spacer pattern is required only when two groups of frames must be
   separated by a gap that differs from the intra-group deadtime, for example
   in variable-gap ptychography:

   ```python
   [
       TriggerPattern(N1, livetime1, deadtime),   # first burst
       TriggerPattern(1,  0.0,       gap),        # inter-burst spacer
       TriggerPattern(N2, livetime2, deadtime),   # second burst
   ]
   ```

3. **`Window.positions()` accepts `float | TriggerPattern`**:

   ```python
   def positions(
       self,
       dt_or_pattern: float | TriggerPattern,
       max_duration: float | None = None,
   ) -> Iterator[dict[AxisT, np.ndarray]]:
   ```

   Passing a `float` returns positions at a fixed servo-cycle rate. Passing a
   `TriggerPattern` returns positions at each trigger instant (centred on each
   active window). Both paths yield the same `dict[axis → np.ndarray]`
   interface.

## Consequences

- PandA sequence table builders must emit `½·deadtime` pre-delay, `livetime`
  active window, and `½·deadtime` post-delay — not `livetime` then `deadtime`.
- Consumers must not reject `TriggerPattern` instances with `livetime=0.0`.
- `Window.positions()` implementations must branch on the argument type and
  compute trigger instants from the centred-livetime formula when a
  `TriggerPattern` is passed.
- `core.py` docstrings on `TriggerPattern` must document both the centred
  semantics and the `livetime=0.0` case.
