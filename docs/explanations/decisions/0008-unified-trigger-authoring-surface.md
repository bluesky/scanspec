# 8. Unified trigger-authoring surface: `TriggerGroup` and `TriggerPlan`

Date: 2026-09-11

## Status

Accepted

## Context

`Acquire` currently authors its windowed stream through two independent
fields: `detectors: Sequence[DetectorGroup[DetectorT]]` and
`trigger_sequence: TriggerSequence[DetectorT] | None`. This has two related
problems.

**Collapsing `num` loses information needed elsewhere.** Triggering a
detector only needs the total repeat count
(`exposures_per_collection * collections_per_event`, folded into
`TriggerRepeat.num`). Reshaping collected data back into events needs the two
factors kept separate — PRD §5 uses `collections_per_event > 1` to add an
extra reshaping dimension downstream, and `(2, 3)` vs `(6, 1)` give the same
trigger count but different output shapes. Because `DetectorGroup` is the
only place both factors are recorded, `WindowedStream.detector_groups` must
keep its own `list[DetectorGroup]`, entirely separate from whatever
`trigger_sequence` the caller supplies — it cannot be derived from the
already-collapsed `TriggerSequence` after the fact.

**With more than one `DetectorGroup`, the caller writes the same hierarchy
twice.** `_bake_trigger_sequence` only auto-derives a `TriggerSequence` for
the zero-or-one-group case; picking which group becomes the parent is
otherwise ambiguous. Any multi-group `Acquire` — the common case for
multi-rate detector triggering — requires hand-authoring both `detectors`
and `trigger_sequence`, with a validator
(`_validate_trigger_sequence_detectors_match`) whose only job is catching
drift between the two afterwards. The caller still explicitly decides the
hierarchy (this remains unchanged — see Decision 1 below); the problem is
only that they must express that same decision twice, in two
independently-validated structures.

A single richer authoring type, capturing detector identity, both shape
factors, and trigger timing together, resolves both problems at once: one
object written down, two different things (`DetectorGroup` and
`TriggerSequence`) mechanically derived from it, each given exactly the
information it needs.

Separately, `DetectorGroup.livetime`/`deadtime` are also the only place
`ContinuousStream.detector_groups` expresses its own rate — `ContinuousStream`
has no timing field of its own. Any change that removes timing from
`DetectorGroup` entirely would leave continuous streams with no way to
express a rate at all. `ContinuousStream`/`MonitorStream` are otherwise out
of scope for this decision (see Consequences).

## Decision

### 1. New authoring types: `TriggerGroup` and `TriggerPlan`

```python
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
    """Caller-authored trigger hierarchy for one Acquire's windowed stream.

    Replaces Acquire.detectors + Acquire.trigger_sequence: the caller
    authors the detector/timing hierarchy once, and compile() derives both
    the compiled TriggerSequence tree and the list[DetectorGroup] the
    windowed stream needs, instead of requiring the caller to hand-write
    and keep both in sync.

    root/children mirror TriggerSequence's own parent/children shape one
    level up: children fire during every root repeat, in parallel with each
    other, each at its own integer-multiple rate. Detector sets across root
    and all children must be disjoint -- checked here structurally at
    construction time, since timing may still be unresolved. Physical
    checks (integer-ratio rates, child duration fitting inside the root's
    livetime, concrete timing) happen at compile() time against the derived
    TriggerSequence, unchanged from before.
    """

    model_config = ConfigDict(frozen=True)

    root: TriggerGroup[DetectorT]
    children: list[TriggerGroup[DetectorT]] = []

    @classmethod
    def single(cls, root: TriggerGroup[DetectorT]) -> Self:
        """Sugar for the common no-children case: TriggerPlan(root=root)."""
        return cls(root=root)

    @model_validator(mode="after")
    def _disjoint_detectors(self) -> Self: ...  # structural check only
```

The caller still explicitly decides the hierarchy — which group is the
root, which are children, how they nest — exactly as before. Only how many
times that decision must be expressed changes: once, into one object that
generates both derived structures, instead of two independently-authored
objects that must each separately express the same hierarchy.

### 2. `Acquire.trigger_plan` replaces `Acquire.detectors` and `Acquire.trigger_sequence`

```python
class Acquire(Spec[AxisT, DetectorT, MonitorT]):
    spec: AnySpec[AxisT, Any, Any]
    fly: bool = False
    stream_name: str = "primary"
    trigger_plan: TriggerPlan[DetectorT] | None = None
    continuous_streams: Sequence[ContinuousStream[DetectorT]] = ()  # unchanged
    monitors: Sequence[MonitorStream[MonitorT]] = ()                # unchanged
    duration: float | None = None
```

A bare `TriggerGroup` passed as `trigger_plan` is eagerly normalized (a
`mode="before"` validator) into `TriggerPlan(root=that_group)` before
pydantic's own field validation runs. The field's annotated type stays
`TriggerPlan[DetectorT] | None` — no union fan-out — and the stored/
serialized form is always the wrapped `TriggerPlan` shape regardless of
which form the caller used to construct it, so there is exactly one
canonical wire format for a given plan, not two.

`_validate_trigger_sequence_detectors_match` is deleted: with one field
instead of two, there is nothing left to cross-check for drift.
`_validate_unique_detectors` is retargeted to walk
`trigger_plan.root`/`trigger_plan.children` instead of `self.detectors`.

### 3. `compile()` derives two structures from `trigger_plan`, neither lossy

Two module-level derivation functions (not methods on `TriggerPlan`, since
they need `inner_length`/`fly` from the compiled motion generators — the
same reason `_bake_trigger_sequence` takes `gens` today):

- `TriggerGroup -> TriggerRepeat`: same `num` formula as today
  (`exposures_per_event * inner_length if fly else exposures_per_event`),
  reading from `TriggerGroup` instead of `DetectorGroup`. Feeds
  `WindowGenerator`/`Window.trigger_sequences`.
- `TriggerGroup -> DetectorGroup`: no `num` involved at all — both shape
  factors pass through unchanged. Feeds `WindowedStream.detector_groups`.

Both derivations read from the same `TriggerPlan`, so the two outputs can
never drift out of sync with each other — there is no validator needed to
catch that drift, because there are no longer two independently-authored
sources for it to drift between.

### 4. `DetectorGroup` keeps `livetime`/`deadtime`; `ContinuousStream` is unaffected

`DetectorGroup`'s shape is unchanged. What changes is how it is produced for
the windowed case: it is no longer directly caller-authored (it moves off
`Acquire.detectors` onto the derived output described in Decision 3) — it
becomes purely compiled output for `WindowedStream.detector_groups`, the
same role it already plays today. `ContinuousStream.detector_groups`
continues to be directly caller-authored via `Acquire.continuous_streams`,
completely unchanged by this decision — it still has no way to express
detector identity and rate other than through `DetectorGroup`, so
`DetectorGroup` cannot be reduced to pure inventory without breaking it.

### 5. `TriggerChild` is removed; `TriggerRepeat` absorbs it

`TriggerChild.repeats: list[TriggerRepeat]` is always a one-element list
everywhere it is constructed in the current codebase — dead capacity. The
mechanism it was meant to support (ptychography's variable exposure
spacing) already lives one level up, on `Window.trigger_sequences`: a
window can hold several sequential `TriggerSequence` entries (burst,
zero-livetime spacer, burst), and motion stays continuous across all of
them within the window. A child-level list duplicates a capability the
parent-level list already provides, at a level where it is never used.

`TriggerRepeat` is extended with a `detectors` field and absorbs
`TriggerChild`'s role directly as `TriggerSequence`'s child entries:

```python
@dataclass(frozen=True)
class TriggerRepeat(Generic[DetectorT]):
    """One resolved, repeating trigger block within a TriggerSequence.

    detectors: the set of detectors this block fires.
    num:       number of times this block repeats.
    livetime:  detector exposure time in seconds.
    deadtime:  detector readout/spacing time in seconds.

    Centred-livetime semantics apply: execution order per repeat is
    ½·deadtime -> livetime -> ½·deadtime.
    """

    detectors: frozenset[DetectorT]
    num: int
    livetime: float
    deadtime: float


@dataclass(frozen=True)
class TriggerSequence(Generic[DetectorT]):
    """Detector triggering description for one sequential entry in a window.

    parent fires first; children fire in parallel with each other during
    every parent repeat. All child detector sets must be disjoint from each
    other and from parent.detectors.

    Window.trigger_sequences is an ordered list; entries execute one after
    another within the window.
    """

    parent: TriggerRepeat[DetectorT]
    children: list[TriggerRepeat[DetectorT]]
```

`TriggerSequence`'s own `detectors` field is dropped — it was always
identical to `parent.detectors` once `TriggerRepeat` carries detector
identity, so keeping both was redundant.

### 6. Compiled `TriggerRepeat`/`TriggerSequence` become plain dataclasses

Under ADR 0007, `TriggerRepeat`/`TriggerChild`/`TriggerSequence` were
pydantic `BaseModel`s specifically because `TriggerSequence` doubled as
caller-authored input to `Acquire.trigger_sequence` and had to survive a
JSON round trip (unresolved `livetime`/`deadtime` filled in downstream
before `compile()`).

That round trip now happens entirely on `TriggerGroup`/`TriggerPlan`, before
`compile()` — never on the compiled output. `compile()` and
`validate_trigger_sequence` already reject unresolved (`None`) timing, so
once a `TriggerRepeat`/`TriggerSequence` exists, there is no remaining
ambiguity for a round trip to preserve. `TriggerRepeat.livetime`/`deadtime`
drop `| None` and become plain `float`. With no round-trip requirement
left, `TriggerRepeat` and `TriggerSequence` revert to plain
`@dataclass(frozen=True)`, matching every other compiled-output type
(`DetectorGroup`, `WindowGenerator`, `Window`, `Scan`, `WindowedStream`).
This supersedes ADR 0007 Decision 1's pydantic carve-out for these types.

## Consequences

### Code changes required

1. **`TriggerGroup`, `TriggerPlan`** (`core.py`): add as pydantic
   `BaseModel`s, per Decision 1.
2. **`TriggerRepeat`, `TriggerSequence`** (`core.py`): convert to plain
   `@dataclass(frozen=True)`; add `detectors` to `TriggerRepeat`; drop
   `| None` from `livetime`/`deadtime`; drop `TriggerSequence.detectors`;
   rename `TriggerSequence.trigger_repeat` to `parent`. Remove
   `TriggerChild` entirely.
3. **`Acquire`** (`specs.py`): replace `detectors`/`trigger_sequence` fields
   with `trigger_plan: TriggerPlan[DetectorT] | None`, plus the
   `mode="before"` bare-`TriggerGroup`-normalization validator (Decision 2).
   Delete `_validate_trigger_sequence_detectors_match`. Retarget
   `_validate_unique_detectors` to walk `trigger_plan`.
4. **`compile()`** (`specs.py`): add the two module-level derivation
   functions from Decision 3, replacing `_bake_trigger_sequence`'s
   single-group-only logic. `WindowedStream.detector_groups` is now always
   populated from the derived `list[DetectorGroup]`, not `self.detectors`
   directly.
5. **`validate_trigger_sequence`, `_truncate_trigger_sequence`,
   `trigger_sequences_duration`** (`core.py`): drop the inner
   `for r in child.repeats` loop everywhere it appears — children are now a
   flat `list[TriggerRepeat]`, not `list[TriggerChild]`. Field accesses move
   from `.trigger_repeat`/`.detectors` to `.parent`/`.parent.detectors`.
6. **Tests**: update every construction of `TriggerRepeat`/`TriggerChild`/
   `TriggerSequence` (`tests/scanspec/v2/test_specs.py`,
   `test_use_cases.py`, `test_compile.py`, `test_core.py`) to the new
   shape. Remove any test exercising `TriggerChild.repeats` with more than
   one entry (none currently exist). Add: `TriggerPlan` disjointness
   validation; bare-`TriggerGroup` normalization on `Acquire.trigger_plan`;
   the multi-group flagship case authored once instead of twice; the
   derived `DetectorGroup` list matches what hand-authoring `detectors`
   used to produce.
7. **`API_SPEC.md`, PRD.md §4**: update trigger-model examples and prose to
   the new types once implemented — out of scope for this ADR itself.

### Explicitly out of scope

- **`continuous_streams`/`monitors`.** Whether these stay on `Acquire` or
  move to a standalone construct outside the spec tree is a separate,
  undecided question, tracked independently. `ContinuousStream`/
  `MonitorStream` are unaffected by this ADR (Decision 4).
- **PandA SEQ-block chaining / multi-rate continuous-stream
  representation.** Unrelated to the authoring-surface question this ADR
  resolves.

### Assumptions carried forward unchanged from ADR 0007

Two-level depth cap (root + one layer of children, no self-nesting),
centred-livetime semantics, checkpoint/pause-resume on root-level repeats,
and `Scan.active_stream_sets` are all unaffected by this decision — they
operate on `TriggerSequence`, whose *shape* (parent + children) is
unchanged, only its field names and the removal of the dead child-repeats
list.
