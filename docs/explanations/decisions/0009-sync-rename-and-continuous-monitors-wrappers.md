# 9. Rename `Acquire` to `Sync`; pull `continuous_streams`/`monitors` out into wrapper specs

Date: 2026-09-17

## Status

Accepted

## Context

`Acquire` binds three independent things together: the windowed trigger
plan (`trigger_plan`, ADR 0008), `continuous_streams`, and `monitors`. Only
the first of these is actually about the relationship the name is meant to
suggest — detector triggering bound to motion. `continuous_streams` and
`monitors` are explicitly *not* frame-coupled: they run for the whole scan
regardless of windows, and the existing code already treats them as
scan-wide rather than local to wherever an `Acquire` happens to sit in the
combinator tree — `_reject_continuous_and_monitors` (`specs.py`), called by
every combinator that merges specs (`Product`, `Zip`, `Concat`), already
raises if a *non-outermost* `Acquire` carries either field, with the
docstring stating outright: *"must be attached at the outermost `Acquire`,
not nested inside combinators."* That rule already exists; it just lives as
a runtime check on a field, not as something the type system expresses.

Separately, `Acquire` is not a precise name for what the class does. It
binds a motion spec to a description of what to do at each point along
it — motion domain and trigger domain, married together. "Acquire" reads as
specifically about data acquisition, which is what happens downstream at
execution time, not the spec-level relationship this node declares.

A related design thread (variable-spacing/phased trigger authoring for
ptychography-style use cases) was explored and *not* adopted here — see
Consequences. This ADR intentionally keeps `Sync`'s trigger-plan surface
exactly as broad as ADR 0008 shipped it (a single root, optionally with
parallel children), not broader.

## Decision

### 1. `Acquire` is renamed to `Sync`

Purely a rename — no field, method, or behavioural change to the parts of
`Acquire` this ADR doesn't otherwise touch. `Sync.spec`, `.fly`,
`.stream_name`, `.trigger_plan`, `.duration` and their semantics
(ADR 0008) are unchanged.

### 2. `continuous_streams` and `monitors` move off `Sync` onto two new outermost wrapper specs

```python
class Monitors(Spec[AxisT, DetectorT, MonitorT]):
    """Attach free-running PV monitors to the whole scan."""

    spec: AnySpec[AxisT, DetectorT, MonitorT]
    monitors: Sequence[MonitorStream[MonitorT]] = ()

    def compile(self) -> Scan[AxisT, DetectorT, MonitorT]:
        scan = self.spec.compile()
        _reject_continuous_and_monitors(scan, "Monitors")
        scan.monitors = list(self.monitors)
        return scan


class ContinuousStreams(Spec[AxisT, DetectorT, MonitorT]):
    """Attach continuously-acquired detector streams to the whole scan."""

    spec: AnySpec[AxisT, DetectorT, MonitorT]
    continuous_streams: Sequence[ContinuousStream[DetectorT]] = ()

    def compile(self) -> Scan[AxisT, DetectorT, MonitorT]:
        scan = self.spec.compile()
        _reject_continuous_and_monitors(scan, "ContinuousStreams")
        scan.continuous_streams = list(self.continuous_streams)
        return scan
```

`Sync` drops both fields entirely. Usage:

```python
ContinuousStreams(
    Monitors(
        Sync(motion, trigger_plan=trigger_plan),
        monitors=[MonitorStream("temperature", "tc1")],
    ),
    continuous_streams=[ContinuousStream("cameras", [...])],
)
```

Each wrapper compiles its inner spec, then attaches its own list to the
resulting `Scan` — the same pattern `Acquire.compile()` already used for
these two fields, just relocated. The two wrappers commute: nesting order
between `ContinuousStreams` and `Monitors` doesn't matter, since each only
ever touches its own field on the compiled `Scan`.

Each wrapper guards against double-wrapping on its own field before
overwriting (`ContinuousStreams(ContinuousStreams(...))`,
`Monitors(Monitors(...))`) via a new field-specific check
(`_reject_existing_monitors`/`_reject_existing_continuous_streams`) rather
than the existing `_reject_continuous_and_monitors`. That function checks
*both* fields at once — correct for a combinator, which must never see
either field already attached since combinators are never outermost, but
wrong for a wrapper: it would reject the ADR's own usage example above
(`ContinuousStreams(Monitors(...), ...)`), since by the time
`ContinuousStreams.compile()` runs, the inner `Monitors` has already
attached `monitors` to the scan, and a monitors-vs-continuous_streams
cross-check has no business living in a check that's meant to guard one
field's own idempotency. `Product`/`Zip`/`Concat` keep using the original
`_reject_continuous_and_monitors` unchanged — a wrapper nested inside a
combinator instead of outside it is still caught there, at the combinator's
own boundary, exactly as before.

### 3. Both previously-open questions about this design are resolved by the wrapper's placement, not left open

- **Do multiple `Sync` nodes inside one wrapper share one scan-wide set, or
  could different sub-trees want their own?** Resolved by construction:
  since `ContinuousStreams`/`Monitors` sit *outside* the whole spec tree,
  there is structurally only one `continuous_streams` list and one
  `monitors` list for the entire compiled `Scan`, regardless of how many
  `Sync` nodes exist inside (e.g. under a `Concat` of several `Sync`s for
  different windowed streams). This is the direct consequence of choosing
  the wrapper shape over a per-node field, and matches why the maintainer
  proposed wrappers in the first place: these are scan-wide concerns, not
  sub-tree-local ones.
- **Does this interact with multiple independent-rate continuous
  streams?** No. `continuous_streams: Sequence[ContinuousStream[DetectorT]]`
  was already a sequence before this change and still is — the pull-out
  only moves *where* that sequence is authored (a wrapper spec field
  instead of a `Sync`/`Acquire` field), not its own multiplicity. Multiple
  `ContinuousStream` entries at independent rates were, and remain, fully
  expressible.

## Consequences

### Code changes required

1. **Rename** `Acquire` -> `Sync` throughout `specs.py`, `core.py` (any
   type references), all six affected test files, and cross-references in
   ADR 0008, PRD.md, and API_SPEC.md.
2. **`Sync`** (`specs.py`): remove `continuous_streams`/`monitors` fields
   and their `Field(...)` descriptions; remove the corresponding lines in
   `compile()` (`scan.continuous_streams = ...`, `scan.monitors = ...`);
   remove the `continuous_streams`/`monitors` branches from
   `_validate_unique_detectors`.

   Detector-uniqueness checking against those two moves to a shared helper,
   called from the end of both `Monitors.compile()` and
   `ContinuousStreams.compile()`, after each attaches its own field. The
   helper re-checks global uniqueness across whatever is currently on the
   `Scan` (windowed-stream detector groups, `continuous_streams`,
   `monitors`) rather than trying to reason about which wrapper is
   "outermost". Because wrapper nesting is sequential — an inner wrapper's
   `compile()` always returns, with its field already attached, before an
   outer wrapper's `compile()` runs — whichever wrapper actually ends up
   outermost naturally sees the complete picture by the time its own check
   runs, regardless of nesting order or whether only one wrapper is used at
   all. No coordination between the two wrapper classes is needed beyond
   calling the same helper.
3. **`Monitors`, `ContinuousStreams`** (`specs.py`, new classes): as
   drafted in Decision 2.
4. **Tests**: update every `Acquire(...)` construction across
   `test_specs.py`, `test_use_cases.py`, `test_compile.py`, `test_core.py`,
   `test_type_inference.py` to `Sync(...)`. Move
   `test_acquire_compile_continuous_streams_and_monitors` (and any test
   exercising `continuous_streams`/`monitors` on `Acquire`) to construct
   `ContinuousStreams(Monitors(Sync(...)))` instead. Existing
   `test_concat_rejects_continuous_streams`/`test_concat_rejects_monitors`
   need no behavioural change (`_reject_continuous_and_monitors` is
   untouched), just updated to the new construction shape. New tests
   needed: double-wrapping rejection
   (`ContinuousStreams(ContinuousStreams(...))`); wrapper nested inside a
   combinator instead of outside it, rejected the same way a non-outermost
   `Acquire` is today; commutativity of `ContinuousStreams`/`Monitors`
   nesting order.
5. **`API_SPEC.md`, PRD.md**: update trigger-model and `Acquire` references
   to `Sync`; document the new wrapper specs.

### A real type-inference cost, discovered during implementation

Before this ADR, `MonitorT` flowed from `monitors=` on `Sync`'s own
constructor call, so `Sync(motion, trigger_plan=..., monitors=[...])`
inferred `Sync[str, str, str]` with no annotation needed. After the
pull-out, `Sync` has no field that mentions `MonitorT` at all, so a bare
`Sync(...)` call always infers `MonitorT=Never` -- fixed the moment that
call returns. Wrapping it in `Monitors(Sync(...), monitors=[...])`, even as
one nested expression, does not recover the inference: pyright evaluates
nested calls argument-first, not bidirectionally, so `Sync(...)`'s
`Never` is already locked in before `Monitors(...)` ever runs, and pyright
correctly reports a type error on the `monitors=` argument passed with a
non-`Never` element type. An explicit annotation on the assignment target
(`spec: Monitors[str, str, str] = Monitors(...)`) is now required to get
`MonitorT` right. `DetectorT` is unaffected -- it still flows from
`Sync.trigger_plan` with no annotation needed, since `trigger_plan` stayed
on `Sync`. See `tests/scanspec/v2/test_type_inference.py`.

### Explicitly out of scope

- **Variable-spacing/phased trigger authoring** (ptychography's
  burst/spacer/burst pattern). Investigated in depth: `Concat` was found
  structurally unable to express it (window-level composition, always a
  turnaround boundary between children, verified against `Window`'s own
  docstring and `WindowGenerator.windows()`'s concat-handling code); the
  compiled side (`Window.trigger_sequences: list[TriggerSequence]`) already
  supports it and was purpose-built for exactly this case, missing only a
  spec-level authoring surface. Two authoring-surface alternatives were
  sketched (a separate `PhasedTriggerPlan`/`List[TriggerGroup]` type, and a
  single `TriggerPlan` with mutually-exclusive `root`/`children` vs.
  `phases` fields). Decision: park this entirely rather than build either.
  `Sync`'s `trigger_plan` stays exactly as broad as ADR 0008 shipped it —
  root, optionally with parallel children, no phases dimension. Nothing
  about this ADR forecloses adding phased authoring later; the compiled
  side remains ready for it whenever a concrete need arises.
- **`num`/`livetime`/`deadtime` optionality rework** (which of these a
  caller may leave for ophyd-async to resolve, and how). Unrelated to
  `Sync`'s outer boundary; belongs with ADR 0008's own thread whenever it's
  ready, not this ADR.

### A naming collision worth handling deliberately, not a blocker

ADR 0007 Decision 4 already uses "sync" for something else — *"BITA is
reserved for motion-controller sync"*, a literal hardware synchronization
signal for the pause/resume mechanism. That is a related but distinct
concept: `Sync` (this ADR) is the spec-level node binding motion to
triggering; motion-controller sync is a hardware-level signal on a
specific consumer's pause-gate implementation. The two are not in tension,
but a reader moving between ADR 0007 and this one could conflate them if
it's not called out — hence this note, and consistent backticking
(`` `Sync` `` for the class, plain "sync" for the hardware signal)
throughout future docs referencing both.
