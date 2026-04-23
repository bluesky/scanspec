# 5. Trigger timing baked at compile time

Date: 2026-04-23

## Status

Accepted

## Context

In scanspec 1.x, detector timing was not part of the scan specification.
`DetectorGroup` described which detectors to trigger and how fast, but the
actual trigger patterns (how many repeats per window, livetime, deadtime)
were computed by each consumer independently. This meant:

- PandA sequence table builders duplicated the logic of "inner dimension
  length × exposures_per_collection = repeats".
- Step scan orchestrators had their own version.
- Any new consumer (PMAC, motor record) had to reimplement the same formula.
- Multi-rate detectors (e.g. encoders at 10× the Pilatus rate) required
  each consumer to handle rate ratios.

## Decision

`Acquire.compile()` bakes `DetectorGroup` descriptions into `TriggerGroup`
objects with concrete `TriggerPattern`s at compile time:

- Each `DetectorGroup` becomes a `TriggerGroup` with
  `trigger_patterns = [TriggerPattern(repeats, livetime, deadtime)]`.
- For fly scans, `repeats = inner_length × exposures_per_collection`.
- For step scans, `repeats = exposures_per_collection`.
- Multi-rate groups (e.g. 10× encoders) each get their own `TriggerGroup`
  in the same window.
- `livetime` and `deadtime` must be set (not `None`) before `compile()` —
  it raises `ValueError` otherwise. `None` means "ophyd-async fills this in"
  and must be resolved before compilation.

The duration of each window is derived from the slowest detector group:
`duration = max(sum(p.repeats * (p.livetime + p.deadtime) for p in tg.trigger_patterns) for tg in trigger_groups)`.
An explicitly supplied `duration` on `Acquire` must be ≥ this derived value.

`TriggerGroup`s are set on the innermost `WindowGenerator` and applied to
every `Window` it yields, regardless of fly vs step.

## Consequences

- **Consumers read, don't compute**: PandA reads `pattern.repeats`,
  `pattern.livetime`, `pattern.deadtime` directly into `SeqTable` rows.
- **Single-rate and multi-rate are uniform**: A `Window` may have multiple
  `TriggerGroup`s; each consumer finds its group by matching detector names
  via `frozenset(group.detectors)`.
- **Ptychography extensible**: Variable-gap patterns use multiple
  `TriggerPattern` entries per group — same structure, no special case.
- **Validation at compile time**: Rate ratio and uniqueness invariants are
  checked when `Acquire.compile()` runs, not at consumption time.
