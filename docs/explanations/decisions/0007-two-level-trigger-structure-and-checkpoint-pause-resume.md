# 7. Two-level trigger structure and checkpoint-based pause/resume

Date: 2026-06-26

## Status

Accepted

## Context

The existing `TriggerPattern` / `TriggerGroup` split (ADR 0005) cannot express
detector groups that fire at integer-multiple rates within the same window (e.g.
SAXS/WAXS at 100 Hz with a Tetramm electrometer at 8 kHz and a PandA encoder at
5 kHz reading out in parallel): the faster groups are siblings with no structural
relationship to the master group, so there is no invariant linking their repeat
counts at any checkpoint during the scan.

This ADR supersedes ADR 0005. It also modifies ADR 0006: centred-livetime
semantics are unchanged, but the `Window.positions()` argument changes to a
plain `times: np.ndarray` (Assumption A4) — no longer `TriggerPattern`, and
no longer the structured `TriggerRepeat` type defined below either.

## Decision

### 1. New trigger structures

Replace `TriggerPattern` and `TriggerGroup` with two
new types:

```python
@dataclass
class TriggerRepeat:
    num: int          # number of times this pattern repeats
    livetime: float   # detector exposure time in seconds
    deadtime: float   # detector readout/spacing time in seconds

@dataclass
class TriggerSequence(Generic[DetectorT]):
    detectors: frozenset[DetectorT]
    trigger_repeat: TriggerRepeat
    children: dict[frozenset[DetectorT], list[TriggerRepeat]]
```

`Window.trigger_groups` becomes `Window.trigger_sequences: list[TriggerSequence]`.
The list is **sequential** — entries execute one after another within the window.

`children` is a **parallel** dict: each key is a frozenset of child detectors,
each value is a **sequential** `list[TriggerRepeat]` that those detectors execute
during *each* parent repeat. All child detector sets must be disjoint from each
other and from the parent `detectors`; this is validated at compile time.

Centred-livetime semantics (ADR 0006) apply to every `TriggerRepeat`: execution
order per repeat is `½·deadtime → livetime → ½·deadtime`.

Child `TriggerRepeat` durations must not exceed the parent livetime — the sum of
`num × (livetime + deadtime)` across a child's `list[TriggerRepeat]` must be ≤
`trigger_repeat.livetime`; validated at compile time.

This design means two unrelated detector groups cannot be triggered at the top
level with no structural relationship between them. This is intentional — there is
no current use case for it.

### 2. Integer-multiple rate groups as parallel children

For example, consider a beamline collecting SAXS and WAXS (two X-ray
scattering detectors) at 100 Hz as the master rate, while simultaneously
reading out a Tetramm electrometer (a fast current amplifier) at 8 kHz and a
PandA encoder at 5 kHz during each SAXS exposure. Each SAXS frame has a 9 ms livetime and 1 ms
deadtime (10 ms period, 100 Hz). For a 1-second flyscan row the parent fires
100 times. During each 9 ms SAXS livetime, Tetramm fires 72 times and PandA
fires 45 times in parallel — each child runs at its full advertised rate but
only within the parent's 9 ms livetime (8000 Hz × 0.009 s = 72;
5000 Hz × 0.009 s = 45):

```python
TriggerSequence(
    detectors = frozenset({saxs, waxs}),
    trigger_repeat = TriggerRepeat(num=100, livetime=0.009, deadtime=0.001),
    # 100 × 0.010 s = 1.000 s total window duration
    children = {
        frozenset({tetramm}): [TriggerRepeat(num=72,  livetime=0.000124, deadtime=0.000001)],
        # 8000 Hz × 0.009 s livetime = 72 triggers per parent repeat
        frozenset({panda}):   [TriggerRepeat(num=45,  livetime=0.000190, deadtime=0.000010)],
        # 5000 Hz × 0.009 s livetime = 45 triggers per parent repeat
    }
)
```

Tetramm and PandA fire in parallel during each 9 ms SAXS livetime, with no
structural relationship to each other — they start at the same instant (the
beginning of the parent livetime) but their individual trigger instants are
independent.

This configuration has two children (Tetramm, PandA), so it costs two SEQ
blocks — one parent-plus-first-child pair, plus one more block for the
second child (§5).

If a detector needed to fire at a rate nested inside one of these children —
e.g. an 80 kHz detector triggered within each Tetramm exposure — it would
not be expressed as a child of a child. Since 80 kHz is still an integer
multiple of the 100 Hz parent rate, its timing is computed directly against
the parent's 9 ms livetime and added as a third parallel entry in the same
`children` dict, not nested inside Tetramm's (A2).

### 3. Spacers for variable spacing

`livetime = 0.0` on `trigger_repeat` is valid (per ADR 0006): it is a pure
spacing-only repeat — no detector fires, the hardware inserts a timed gap.

```python
# Variable-spacing ptychography pattern in trigger_sequences:
[
    TriggerSequence(frozenset({det}), TriggerRepeat(N1, lt1, dt), {}),
    TriggerSequence(frozenset(),      TriggerRepeat(1,  0.0, spacing), {}),
    TriggerSequence(frozenset({det}), TriggerRepeat(N2, lt2, dt), {}),
]
```

**Validation**: if `trigger_repeat.livetime == 0.0` then `children` must be `{}`.
Non-empty children on a spacer is a compile-time error — there is no livetime
window for children to execute within.

### 4. Checkpoint-based pause/resume

**scanspec2's role** is to define the unit of progress and the resume API:

- Each root-level repeat of a `TriggerSequence`'s `trigger_repeat` is a
  **checkpoint**, except blank/spacer repeats (`livetime == 0.0`), which are
  never counted and never truncated — a paused blank always replays in full
  on resume. `trigger_index` counts completed *live* root-level parent
  repeats across all `TriggerSequence`s in the window. Gaps are minimum
  requirements, not exact durations, so overshooting a gap (however long
  elapsed pre-pause, plus the pause itself) is harmless, while undershooting
  it — which counting blanks would risk, since resume could then skip the
  unexecuted remainder — is not. (Assumes at most one blank between live
  bursts per window, per Assumption A1 — a window with several blanks before
  the true resume point would replay all of them; currently unreachable via
  `compile()`.)
- `Scan.with_start(window, trigger_index)` returns a new `Scan` whose first
  window has the first `trigger_index` repeats removed, via
  `_truncate_trigger_sequence(sequences, trigger_index)`.
- Resume is always **forward-only** — emitted detector data is final. If data
  is bad, cancel and restart the scan; there is no rewind mechanism.

**The consumer's role** (ophyd-async / PandA) is to determine `trigger_index`
at pause time and call `with_start`. The expected protocol is:

```python
await seq.bitb.set(0)
try:
    # Poll seq.state (not TABLE_LINE/LINE_REPEAT) for the stall.
    await wait_for_value(seq.state, WAIT_TRIGGER, timeout=0.2)
except TimeoutError:
    if not is_blank(await seq.table_line.get_value()):
        # Live row: wait for the hardware to reach the next checkpoint.
        await wait_for_value(seq.state, WAIT_TRIGGER,
                             timeout=max_time_between_checkpoints)
    # Blank row (livetime = 0): read table_line immediately — no data in
    # flight, any mid-blank position is a safe resume point.
trigger_index = lookup_checkpoint_index(await seq.table_line.get_value())
```

BITB is used for the pause gate; BITA is reserved for motion-controller sync
at window boundaries. Key consumer behaviour:

- Stall detection polls `seq.state` for `WAIT_TRIGGER`, not `TABLE_LINE` or
  `LINE_REPEAT`.
- **Live rows**: pause latency is bounded by one root-level parent repeat
  period (`max_time_between_checkpoints`).
- **Blank rows** (`livetime = 0`): `table_line` is read immediately — no data
  in flight, any mid-blank position is safe. Long blank spacers do **not**
  need to be broken into short repeats.

After stalling, the consumer aborts the sequence and reloads it from the
checkpoint via `with_start`, rather than rewriting the remaining table in
place while PandA holds mid-table. This is a consumer implementation
decision; it does not affect the scanspec data model.

### 5. `Scan.active_stream_sets`

Add `active_stream_sets: list[frozenset[str]]` to `Scan`. Enables consumers to
validate sequencer-table capacity up front without iterating windows.

**Algorithm** (compile-time, no window iteration):

- `Acquire` → `[frozenset({self.stream_name})]`
- `Concat` → deduplicated union of each child's list
- All other wrappers → pass inner value through unchanged

The two axes of PandA resource consumption are orthogonal:

- **Nesting depth** (parent + children in a `TriggerSequence`) determines how
  many SEQ blocks a single active set requires: one parent plus one child
  fits in a single SEQ block, and each additional child requires an
  additional SEQ block. This ADR fixes depth at two (parent + one child
  layer); the SAXS/WAXS + Tetramm + PandA example (§2) has two children, so
  it costs two SEQ blocks, not one.
- **Number of distinct simultaneous active sets** determines how many independent
  SEQ tables are needed across the full scan. A `Concat` of two differently-named
  streams produces two singleton sets — one table used alternately, not
  simultaneously. A SEQ block has 6 outputs, so independent streams can
  reuse different outputs of the same block rather than each needing a
  separate one — up to 6 streams per block.

## Consequences

### Code changes required

1. **`TriggerRepeat` and `TriggerSequence`** (`core.py`): add the two new
   dataclasses. Remove `TriggerPattern` and `TriggerGroup`.
   `Window.trigger_groups` → `Window.trigger_sequences: list[TriggerSequence]`.

2. **`Window.positions()`** (`core.py`): signature is
   `positions(times: np.ndarray) -> dict[axis, np.ndarray]`. The
   `TriggerRepeat` branch, `max_duration`, and all internal chunking are
   gone — the caller supplies explicit time instants and owns iteration
   entirely. See Assumption A4 for the rationale: generating those time
   instants (including any PandA-specific row/edge structure) is a
   consumer-layer concern, not something `Window.positions()` should know
   about.

3. **`_truncate_trigger_sequence`** (`core.py`): walk the flat
   `list[TriggerSequence]`, accumulating `trigger_repeat.num` counts. Skip
   completed sequences; for the in-progress one, produce a replacement with
   `trigger_repeat.num = original_num - completed`. `detectors` and `children`
   are carried unchanged into the remainder.

4. **`Scan.with_start(window, trigger_index)`** (`core.py`): already present;
   internal call changes from `_truncate_trigger_patterns` to
   `_truncate_trigger_sequence`.

5. **`Scan.active_stream_sets`** (`core.py` / `specs.py`): add as a field to
   `Scan`, populated at compile time by `Acquire` and `Concat` as described
   above.

6. **`_bake_trigger_sequence`** (`specs.py`): replaces `_bake_trigger_groups`.
   Takes `list[DetectorGroup]`, selects the lowest-rate group as parent (all
   groups at the same rate merge into `parent.detectors`), maps remaining groups
   to parallel children keyed by their detector frozensets. Validates at compile
   time: each child's total duration ≤ parent livetime; detector sets are
   disjoint; depth does not exceed one child layer.

7. **`_compute_duration`** (`specs.py`): sum `trigger_repeat.num × (livetime +
   deadtime)` across the root `trigger_sequences` list. Children run inside the
   parent livetime and do not contribute to window duration.

8. **Tests**:
   - Remove `test_with_start_trigger_index_multi_group_raises`.
   - Update `test_use_cases.py` to use `TriggerSequence`/`TriggerRepeat` and
     verify the SAXS/Tetramm/PandA example produces the correct parallel children
     structure.
   - Add: `TriggerSequence` truncation preserves `detectors` and `children`;
     spacer compile-time error when `children` is non-empty and `livetime=0.0`;
     `active_stream_sets` covering (a) single `Acquire` → one singleton, (b)
     `Concat` of two differently-named `Acquire`s → two singletons, (c) `Concat`
     of two same-named `Acquire`s → one singleton (deduplication).

## Assumptions

### A1 — Compiled specs always produce one `TriggerSequence`; the spacer pattern (Decision §3) has no way to be authored yet

Today, `_bake_trigger_sequence` always produces a single-entry
`list[TriggerSequence]` for a compiled window: it takes a flat
`list[DetectorGroup]` and bakes it into one parent/children structure, with
no way to express a *sequence* of several `TriggerSequence`s in one window.
This means the variable-spacing spacer pattern from Decision §3 — which
needs multiple `TriggerSequence` entries (burst, spacer, burst) — cannot be
produced by `compile()` at all today.

The only way to get a multi-entry `list[TriggerSequence]` is to construct a
`Window` by hand, bypassing `compile()` entirely. Adding a spec-facing way to
author the spacer pattern is out of scope for this ADR (see PRD §2.5, §11).
Even though compiled specs can't produce multi-entry lists yet,
`_truncate_trigger_sequence` and `_compute_duration` must still handle them
correctly, since manually constructed `Window`s are a supported input to
both.

### A2 — The one-child-layer limit exists for checkpointing, not SEQ block capacity

Pause/resume needs a single top-level stream to checkpoint on (Decision §4).
That requirement — not SEQ block capacity, which §5 covers separately — is
why this ADR caps nesting at one child layer. Any deeper nesting is
expressible within this structure: since every detector group fires at a
fixed integer-multiple rate of the same top-level parent (Decision §2), a
group that might conceptually sit two or more levels deep can always be
re-derived directly against the parent's livetime and added as another
entry in the same flat `children` dict, rather than nested inside another
child.

### A3 — Pause gate composes with position-compare as separate, interleaved rows; exact row/table structure is a consumer-side concern

A PandA SEQ row supports only one trigger condition, so BITB cannot be
combined with a position-compare condition on the same row. Repeats also
cannot be rolled into a single `REPEATS=N` row for this purpose — the pause
gate must be checked on every repeat, not just once at the end of a
collapsed block of them. The direction is to trigger on position first, then
on BITB, as two separate, interleaved rows.

The precise row/table structure this produces — exactly how many rows it
costs beyond A4's N+1, which edges get an interleaved BITB check, and how it
interacts with the per-child SEQ-block-capacity accounting in §5 — is **not**
something scanspec or this ADR needs to resolve. It is the not-yet-written
ophyd-async PandA driver's job; scanspec's obligation stops at documenting
the constraint (separate interleaved rows, checked per repeat, not
collapsible) clearly enough for that implementation to be built correctly.

This choice affects only the PandA consumer/driver's row-generation logic.
It does not change scanspec's API or data model: `Window.positions()`,
`TriggerSequence`, `trigger_index`, and `Scan.with_start` are unaffected
either way, since none of this row-level detail is represented in
scanspec's code.

### A4 — PandA position-compare row/edge encoding is a consumer-side concern, distinct from A3's BITB pause gate; it does not live in `Window.positions()`

PandA's own SEQ position-compare encoding for the *N* live exposures within a
window requires **N+1 table rows and 2N edges**, not a flat repeat of one row
pattern: every exposure needs its own gate-open (HIGH) edge and gate-close
(LOW) edge — 2N edges, unavoidable. Consecutive exposures share the boundary
between them (the LOW that closes exposure *n* and the HIGH that opens
exposure *n+1* land on the same row), so only the first and last edges are
unpaired, giving 1 (opening, HIGH-only) + (N−1) (middle, LOW+HIGH each) + 1
(closing, LOW-only) = N+1 rows:

```
windows:  /‾‾‾‾‾‾‾‾‾‾‾‾‾‾\/‾‾‾‾‾‾‾‾‾‾‾‾‾‾\/‾‾‾‾‾‾‾‾‾‾‾‾‾‾\
livetime: __‾‾‾‾‾‾‾‾‾‾‾‾____‾‾‾‾‾‾‾‾‾‾‾‾____‾‾‾‾‾‾‾‾‾‾‾‾__
pcomp:      |           |               |               |
action:     high        low+high        low+high        low
```

This row count is fixed by the structure of the gate signal itself — it is
independent of whether livetime placement is centred (ADR 0006) or
leading-edge; centred-livetime only changes *which numeric position* each row
compares against, not how many rows/edges are structurally required.

This is a **related but distinct mechanism from A3**. A4 is about the
position-compare rows that generate the exposure gate itself (HIGH/LOW per
frame) — these exist regardless of whether pause/resume is in play at all.
A3 is about how the BITB pause gate composes with those same rows once they
exist: per A3, a single PandA SEQ row supports only one trigger condition,
so pause-gating cannot be combined into A4's position-compare rows directly
— the direction is to trigger on position first, then BITB, as separate,
interleaved rows, checked on every repeat. The exact row/table
structure this composition requires is a consumer-side (ophyd-async PandA
driver) concern, not something this ADR resolves; see A3.

Because this row/edge encoding is fundamentally PandA-hardware-specific — a
PMAC consumer has no concept of "SEQ blocks" or position-compare rows at all,
it just wants a dense position stream — it belongs in the hardware-specific
consumer/driver layer (ophyd-async), not in scanspec's hardware-agnostic
`Window`/`TriggerSequence` model. scanspec's job stays "what position is the
axis at, at time T" — a pure function of the compiled spec — not "how does
PandA's SEQ table format encode N trigger events." This is the rationale for
the `Window.positions()` signature change described in Consequences item 2:
the caller, not `Window.positions()`, now owns generating whatever row/edge
instants its hardware needs and supplies them as an explicit time array.

### A5 — `TriggerRepeat.livetime`/`deadtime` do not yet support unresolved values, unlike their ADR 0005 predecessor

ADR 0005's `TriggerPattern.livetime`/`deadtime` were `float | None`, with
`None` meaning the value is not yet known at authoring time and must be
filled in by a downstream process (e.g. ophyd-async, which has visibility
into a device's real limits) before `compile()` — `compile()` raised if
either was still `None`. `TriggerRepeat.livetime`/`deadtime` (Decision §1)
carry this forward as plain `float`, with no equivalent unresolved-value
representation.

This is a gap, not an intentional narrowing: nothing in this ADR's Context
or Decision revisits or retracts ADR 0005's requirement, and `thoughts.md`'s
original statement of the requirement ("we still need the ability to just
specify duration and ophyd will fill in livetime and deadtime from it")
covers both fields, not only `deadtime`. Restoring `float | None` parity —
or an equivalent unresolved-value representation suited to the two-level
`TriggerRepeat`/`TriggerSequence` structure — is unresolved and needs to be
addressed before the `Acquire` authoring-surface redesign lands (Decision
§1 and Consequences), since that redesign is exactly where a caller-supplied
`TriggerSequence` would need to carry not-yet-resolved timing values.
