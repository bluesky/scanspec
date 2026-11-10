# 8. Unified trigger-authoring surface: `TriggerGroup` and `TriggerFollower`

Date: 2026-09-11

## Status

Accepted

`Acquire`, referenced in this ADR's original Context below, was renamed to
`Sync` — see ADR 0009 (implemented). The rest of this document uses `Sync`
throughout, including in sections describing decisions made before that
rename landed, to stay consistent with the current code.

## Context

`Sync` originally authored its windowed stream through two independent
fields: `detectors: Sequence[DetectorGroup[DetectorT]]` and
`trigger_sequence: TriggerSequence[DetectorT] | None`. This has two related
problems.

**Collapsing `repeats` loses information needed elsewhere.** Triggering a
detector only needs the total repeat count
(`exposures_per_collection * collections_per_event`, folded into
`TriggerRepeat.repeats`). Reshaping collected data back into events needs the two
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
otherwise ambiguous. Any multi-group `Sync` — the common case for
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

### 1. New authoring types: `TriggerGroup` and `TriggerFollower`

`TriggerGroup` carries its own base-rate detector identity and timing
directly — `detectors`, `exposures_per_collection`, `collections_per_event`,
`livetime`, `deadtime` — plus a `followers` list. `followers` fire in
parallel with each other during every one of the group's own repeats, each
at its own integer-multiple rate. Detector sets across the group's own
`detectors` and all followers must be disjoint — checked here structurally
at construction time, since timing may still be unresolved. Physical checks
(integer-ratio rates, follower duration fitting inside the group's own
livetime, concrete timing) happen at `compile()` time against the derived
`TriggerSequence`.

```python
class TriggerGroup(BaseModel, Generic[DetectorT]):
    """Caller-authored trigger hierarchy for one Sync's windowed stream.

    Replaces Sync.detectors + Sync.trigger_sequence: the caller authors
    the detector/timing hierarchy once, and compile() derives both the
    compiled TriggerSequence tree and the list[DetectorGroup] the
    windowed stream needs, instead of requiring the caller to hand-write
    and keep both in sync.

    detectors/exposures_per_collection/collections_per_event/livetime/
    deadtime describe the group's own base-rate detectors -- also exactly
    the shape DetectorGroup is derived from (compile(), Decision 3), so a
    TriggerGroup with no followers at all is already a complete,
    meaningful authoring unit on its own. followers is how the group
    grows to cover detectors triggering at other integer-multiple rates
    within the same collective, not a separate hierarchical tier.
    """

    model_config = ConfigDict(frozen=True)

    detectors: frozenset[DetectorT]
    exposures_per_collection: int
    collections_per_event: int
    livetime: float | None
    deadtime: float | None
    followers: list[TriggerFollower[DetectorT]] = []

    @property
    def exposures_per_event(self) -> int:
        """Total triggers per event: each collection needs its own trigger."""
        return self.exposures_per_collection * self.collections_per_event

    @model_validator(mode="after")
    def _disjoint_detectors(self) -> Self: ...  # structural check only


class TriggerFollower(BaseModel, Generic[DetectorT]):
    """Authoring-time detector group firing at its own rate within a
    TriggerGroup, driven by the group's own repeat cadence.

    repeats/livetime/deadtime may all be left unresolved (None) at
    authoring time -- a downstream process (e.g. ophyd-async) fills in
    whichever are missing before compile(), which requires all three
    concrete (Decision 3). Unlike the group's own detectors, repeats is a
    real field here: a follower's rate relative to the group is genuinely
    external information (detector hardware timing), not something
    derivable purely from the spec tree. Named repeats, not num -- "num"
    alone doesn't say what it counts; "repeats" matches the compiled
    TriggerRepeat.repeats it flows into (Decision 5).
    """

    model_config = ConfigDict(frozen=True)

    detectors: frozenset[DetectorT]
    exposures_per_collection: int
    collections_per_event: int
    livetime: float | None
    deadtime: float | None
    repeats: int | None

    @property
    def exposures_per_event(self) -> int:
        """Total triggers per event: each collection needs its own trigger."""
        return self.exposures_per_collection * self.collections_per_event
```

**Why the group's own repeat count is never a field, only ever derived**:
it's `exposures_per_event × inner_length` (fly) or `exposures_per_event`
(step), and both factors are already fully known from the spec tree itself
(the motion generators, the `fly` flag). Nothing about it depends on
hardware; it's the literal point where the motion domain and the trigger
domain bind into one number — the computation `Sync` exists to perform,
not incidental to it. A caller-suppliable count here would force a choice
between silently overwriting a caller-supplied value (hides a mistake) or
cross-checking it
against the derived one (reintroducing the exact "two things kept in sync
by hand" problem this ADR exists to eliminate) — so there is no such field
at all.

**Naming**: the first implemented version of this decision used
`TriggerGroup` identically for both a root group and its children
(`TriggerPlan.root: TriggerGroup`, `TriggerPlan.children:
list[TriggerGroup]`), inside a separate `TriggerPlan` container. Merging
the container and the root group into one class (this decision) briefly
considered keeping the container named `TriggerPlan`, but that collides
with Bluesky's own "plan" vocabulary (a `RunEngine` executes "plans" —
`count()`, `scan()`, `grid_scan()` — a term scanspec, living in the same
ecosystem, shouldn't shadow). `TriggerGroup` for the merged container reads
correctly for the same reason it was already the right name before: it
groups detectors, and that grouping now literally includes the root's own
detectors as well as any followers — not a narrower meaning than before,
just a more complete one. `TriggerFollower` (not `TriggerChild`, which is
purely positional — says where it sits, not what it does) names the actual
relationship: fires at a rate driven by the group's own cadence, not chosen
independently, without implying (the way a leader/follower pairing would)
that a group with no followers is somehow incomplete.

The caller still explicitly decides the hierarchy — what the group's own
rate is, which detectors follow it, at what rate — exactly as before. Only
how many times that decision must be expressed changes: once, into one
object that generates both derived structures, instead of two
independently-authored objects that must each separately express the same
hierarchy.

### 2. `Sync.trigger_group` replaces `Sync.detectors` and `Sync.trigger_sequence`

```python
class Sync(Spec[AxisT, DetectorT, MonitorT]):
    spec: AnySpec[AxisT, Any, Any]
    fly: bool = False
    stream_name: str = "primary"
    trigger_group: TriggerGroup[DetectorT] | None = None
    duration: float | None = None
```

(`continuous_streams`/`monitors` are not `Sync` fields — see ADR 0009.)

The field is named `trigger_group`, not `trigger_plan`, for the same reason
the type itself isn't `TriggerPlan` — avoiding Bluesky's "plan" vocabulary
matters at the field name too, not just the class name.

Merging the root group into `TriggerGroup` itself (Decision 1) means this
field has exactly one shape: `TriggerGroup[DetectorT] | None`, nothing
else. The first implemented design kept `TriggerGroup` as a type reused
for both a root group and its children, paired with a separate `TriggerPlan`
container — which meant this field had to accept a bare `TriggerGroup` too
(the trivial no-children case) alongside a full `TriggerPlan`, a
`TriggerPlan[DetectorT] | TriggerGroup[DetectorT] | None` union with its
own normalization helper (`_as_trigger_plan()`) and a
lossless-vs-canonical-JSON tradeoff to reason about (whether a bare
`TriggerGroup` should be stored/serialized as-is or eagerly wrapped into a
`TriggerPlan`). Merging root into one type removes the second shape
entirely, and with it the whole question: there is only ever one type to
store, serialize, or normalize.

`_validate_trigger_sequence_detectors_match` is deleted: with one field
instead of two, there is nothing left to cross-check for drift. Detector
uniqueness across `trigger_group` and whatever else lives on the compiled
`Scan` (`continuous_streams`, `monitors`) is not `Sync`'s concern at all —
see ADR 0009's `Monitors`/`ContinuousStreams` wrappers, which check it
post-compile instead.

### 3. `compile()` derives two structures from `trigger_group`, neither lossy

Two module-level derivation functions (not methods on `TriggerGroup`, since
they need `inner_length`/`fly` from the compiled motion generators — the
same reason `_bake_trigger_sequence` takes `gens` today):

- `-> TriggerRepeat`: `repeats` is derived differently for the group's own
  detectors than for a follower, since the two answer different questions.
  - **The group's own detectors**: `exposures_per_event * inner_length if
    fly else exposures_per_event` — tied to the scan's own geometry,
    always computed, never caller-suppliable (Decision 1's rationale: this
    is the literal motion↔trigger binding `Sync` exists to perform, not an
    authoring input).
  - **A follower**: `repeats`, `livetime`, and `deadtime` must all be
    concrete (non-`None`) by this point, or `compile()` raises `ValueError`
    immediately — a follower's rate relative to the group is genuinely
    external information (detector hardware timing), so nothing here is
    derived from the spec tree. No attempt is made to compute a missing
    `repeats` from `livetime`/`deadtime` (e.g. `round(group_livetime /
    follower_period)`, mechanically available but deliberately not
    applied) — a caller or ophyd-async must supply all three explicitly.
    A future "repair" pass that fills in whichever *one* of the three is
    still missing, where mechanically derivable, is explicitly out of
    scope for this decision (see Consequences).

  Either way, the existing integer-ratio and duration checks
  (`validate_trigger_sequence`) still run against the derived
  `TriggerRepeat` afterwards, regardless of whether `repeats` came from a
  caller/ophyd-async or (in a future repair pass) from scanspec's own
  calculation — physical consistency doesn't care about provenance.

  Feeds `WindowGenerator`/`Window.trigger_sequences`.
- `-> DetectorGroup`: no `repeats` involved at all — both shape factors pass
  through unchanged, for both the group's own detectors and each follower.
  Feeds `WindowedStream.detector_groups`.

Both derivations read from the same `TriggerGroup`, so the two outputs can
never drift out of sync with each other — there is no validator needed to
catch that drift, because there are no longer two independently-authored
sources for it to drift between.

### 4. `DetectorGroup` keeps `livetime`/`deadtime`; `ContinuousStream` is unaffected

`DetectorGroup`'s shape is unchanged. What changes is how it is produced for
the windowed case: it is no longer directly caller-authored (it moves off
`Sync.detectors` onto the derived output described in Decision 3) — it
becomes purely compiled output for `WindowedStream.detector_groups`, the
same role it already plays today. `ContinuousStream.detector_groups`
continues to be directly caller-authored, now via
`ContinuousStreams.continuous_streams` (ADR 0009),
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
    repeats:   how many times this block executes.
    livetime:  detector exposure time in seconds.
    deadtime:  detector readout/spacing time in seconds.

    Centred-livetime semantics apply: execution order per repeat is
    ½·deadtime -> livetime -> ½·deadtime.
    """

    detectors: frozenset[DetectorT]
    repeats: int
    livetime: float
    deadtime: float


@dataclass(frozen=True)
class TriggerSequence(Generic[DetectorT]):
    """Detector triggering description for one sequential entry in a window.

    root fires first; children fire in parallel with each other during
    every root repeat. All child detector sets must be disjoint from each
    other and from root.detectors.

    Window.trigger_sequences is an ordered list; entries execute one after
    another within the window.
    """

    root: TriggerRepeat[DetectorT]
    children: list[TriggerRepeat[DetectorT]]
```

`TriggerSequence`'s own `detectors` field is dropped — it was always
identical to `root.detectors` once `TriggerRepeat` carries detector
identity, so keeping both was redundant. The compiled side keeps
`root`/`children` naming (not `TriggerGroup`/`TriggerFollower`'s
group-plus-followers naming) — `TriggerSequence` is pure compiled output,
one level removed from the authoring surface's own vocabulary, and `root`
still reads correctly there: a field named `parent` on a container reads
as "the thing that owns this," not "the primary entry held alongside
`children`" — `root` can't be misread that way. Relational
"parent"/"child" language remains fine in prose and local variable names
describing how `root` and `children` relate to each other, even though the
authoring-side types now use "group"/"follower" instead.

### 6. Compiled `TriggerRepeat`/`TriggerSequence` become plain dataclasses

Under ADR 0007, `TriggerRepeat`/`TriggerChild`/`TriggerSequence` were
pydantic `BaseModel`s specifically because `TriggerSequence` doubled as
caller-authored input to `Sync.trigger_sequence` and had to survive a
JSON round trip (unresolved `livetime`/`deadtime` filled in downstream
before `compile()`).

That round trip now happens entirely on `TriggerGroup`/`TriggerFollower`,
before `compile()` — never on the compiled output. `compile()` and
`validate_trigger_sequence` already reject unresolved (`None`) timing, so
once a `TriggerRepeat`/`TriggerSequence` exists, there is no remaining
ambiguity for a round trip to preserve. `TriggerRepeat.livetime`/`deadtime`
drop `| None` and become plain `float`. With no round-trip requirement
left, `TriggerRepeat` and `TriggerSequence` revert to plain dataclasses,
like `DetectorGroup` and the rest of the compiled-output shapes — `frozen=
True` specifically because instances are shared by reference across
multiple `Window`s (e.g. `_truncate_trigger_sequence` always constructs a
new `TriggerSequence`/`TriggerRepeat` rather than mutating one in place),
where accidental mutation of a shared instance would silently affect every
`Window` holding it. This supersedes ADR 0007 Decision 1's pydantic
carve-out for these types.

## Consequences

**Implementation status**: Decisions 1-3's `TriggerGroup`/`TriggerFollower`
merge (replacing the first implemented two-type `TriggerGroup`+
`TriggerPlan` design) and the repeats/livetime/deadtime-all-required
compile-time policy are design-accepted but **not yet implemented** as of
this amendment — Decisions 4-6 (compiled-side shapes, `DetectorGroup`
handling) are already implemented, under the original design, and are
unaffected by this amendment. `scanspec.v2` has no external consumers yet
(it only becomes `scanspec` at the final 2.0 migration, PRD §12), so none
of this carries release/backward-compatibility weight — the code changes
below describe the remaining delta from what is currently in
`core.py`/`specs.py` to what Decisions 1-3 now describe.

### Code changes required

1. **`TriggerGroup`, `TriggerFollower`** (`core.py`): replace the
   currently-implemented `TriggerGroup`+`TriggerPlan` pair with the merged
   shape from Decision 1 — move
   `detectors`/`exposures_per_collection`/`collections_per_event`/
   `livetime`/`deadtime` directly onto the (renamed) container; rename
   `children` to `followers`; add `TriggerFollower` (same fields as the
   old per-child `TriggerGroup`, plus a new `repeats: int | None`).
   Retarget `_disjoint_detectors` to check the group's own `detectors`
   against each follower instead of `root` against each child.
2. **`Sync`** (`specs.py`): rename the field from `trigger_plan` to
   `trigger_group`; narrow its type from `TriggerPlan[DetectorT] |
   TriggerGroup[DetectorT] | None` to `TriggerGroup[DetectorT] | None`
   (Decision 2). Delete `_as_trigger_plan()` entirely — every call site can
   read `trigger_group` directly, no normalization needed.
3. **`TriggerRepeat`** (`core.py`): rename the already-implemented
   `num: int` field to `repeats: int` (Decision 5), for consistency with
   `TriggerFollower.repeats` — the authoring-time field it flows from is
   more important to keep unambiguous than the compiled one, so the
   compiled side takes the (mildly redundant-sounding) name change too.
   Update every construction and `.num` access across `core.py`/`specs.py`
   (`_trigger_group_to_repeat`, `_truncate_trigger_sequence`,
   `validate_trigger_sequence`, `trigger_sequences_duration`) and every
   test/doc example.
4. **`compile()`** (`specs.py`): the group's own `TriggerRepeat`
   derivation is unchanged (still reads the now-inline fields off
   `TriggerGroup` directly instead of `.root`). Follower derivation
   changes from always computing `repeats = round(group_livetime /
   follower_period)` to requiring `repeats`/`livetime`/`deadtime` all
   already concrete on the `TriggerFollower` — raise `ValueError`
   immediately if any is still `None`, no derivation attempted (Decision
   3).
5. **Tests**: update every construction of the old `TriggerGroup`/
   `TriggerPlan` pair (`tests/scanspec/v2/test_specs.py`,
   `test_use_cases.py`, `test_compile.py`, `test_type_inference.py`, plus
   `PRD.md`/`API_SPEC.md` examples) to the merged `TriggerGroup`/
   `TriggerFollower` shape and the `repeats` rename. Add: a test asserting
   `compile()` rejects a follower with any of `repeats`/`livetime`/
   `deadtime` still `None`; a construction-time test that setting fields
   directly on `TriggerGroup` (no separate root object) still validates
   disjointness against followers correctly.

### Explicitly out of scope

- **A "repair" pass for follower `repeats`.** Decision 3 deliberately
  rejects rather than derives a missing follower `repeats`, even though
  `round(group_livetime / follower_period)` is mechanically available
  (it's exactly the formula the first implemented version always used).
  Adding that back as an optional fallback — attempted only when `repeats`
  is missing, after requiring `livetime`/`deadtime` regardless — is real
  follow-up work, not resolved here.
- **`continuous_streams`/`monitors`.** Resolved by ADR 0009: moved off
  `Sync` onto the outermost `Monitors`/`ContinuousStreams` wrapper specs,
  not decided by this ADR. `ContinuousStream`/`MonitorStream` are
  unaffected by this ADR (Decision 4).
- **PandA SEQ-block chaining / multi-rate continuous-stream
  representation.** Unrelated to the authoring-surface question this ADR
  resolves.

### Assumptions carried forward unchanged from ADR 0007

Two-level depth cap (root + one layer of children, no self-nesting),
centred-livetime semantics, checkpoint/pause-resume on root-level repeats,
and `Scan.active_stream_sets` are all unaffected by this decision — they
operate on `TriggerSequence`, whose *shape* (root + children) is
unchanged, only its field names and the removal of the dead child-repeats
list.
