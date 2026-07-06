# 7. Two-level trigger structure and checkpoint-based pause/resume

Date: 2026-06-26

## Status

Proposed

## Context

The existing `TriggerPattern` / `TriggerGroup` split (ADR 0005) cannot express
detector groups that fire at integer-multiple rates within the same window (e.g.
SAXS/WAXS at 100 Hz with a Tetramm electrometer at 8 kHz and a PandA encoder at
5 kHz reading out in parallel): the faster groups are siblings with no structural
relationship to the master group, so there is no invariant linking their repeat
counts at any checkpoint during the scan.

This ADR supersedes ADR 0005. It also modifies ADR 0006: centred-livetime
semantics are unchanged, but the `Window.positions()` argument type changes from
`TriggerPattern` to `TriggerRepeat` (defined below).

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
  **checkpoint**. `trigger_index` counts completed root-level parent repeats
  across all `TriggerSequence`s in the window.
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
place while PandA holds on the gate row. This is a consumer implementation
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
  many SEQ block levels a single active set requires. This ADR fixes depth at
  two (parent + one child layer), which fits in a single SEQ block.
- **Number of distinct simultaneous active sets** determines how many independent
  SEQ tables are needed across the full scan. A `Concat` of two differently-named
  streams produces two singleton sets — one table used alternately, not
  simultaneously.

## Consequences

### Code changes required

1. **`TriggerRepeat` and `TriggerSequence`** (`core.py`): add the two new
   dataclasses. Remove `TriggerPattern` and `TriggerGroup`.
   `Window.trigger_groups` → `Window.trigger_sequences: list[TriggerSequence]`.

2. **`Window.positions()`** (`core.py`): update signature from
   `float | TriggerPattern` to `float | TriggerRepeat`. When a `TriggerRepeat`
   is passed, compute trigger instants using the centred-livetime formula for
   that repeat's `livetime` and `deadtime`.

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

### A1 — `_bake_trigger_sequence` always produces a single-entry list; the spacer pattern has no authoring surface yet

`_bake_trigger_sequence` always produces a single-entry `list[TriggerSequence]`
per compiled window — it has no `DetectorGroup`-level input for the
variable-spacing spacer pattern (Decision §3), so compiled specs cannot
produce it. Multi-entry lists only arise via manual `Window` construction
until a spec-facing authoring surface is added (out of scope for this ADR;
see PRD §2.5, §11). `_truncate_trigger_sequence` and `_compute_duration` must
still handle multi-entry lists correctly in all cases, since manually
constructed `Window`s are a supported input to both.

### A2 — Two nesting levels fit in a single PandA SEQ block

The parent `TriggerSequence.trigger_repeat` and its `children` both encode into a
single SEQ block — no chained tables are required. The one-child-layer limit in
this ADR is therefore structurally exact, not conservative.

### A3 — SEQ encoding inserts one BITB gate row per root-level parent repeat

The pause/resume guarantee for live rows (max latency = one root-level repeat
period) requires a `TRIGGER=BITB=1` gate row before each root-level parent repeat.
The consumer must not collapse N repeats into a single `REPEATS=N` SEQ row — that
reduces N checkpoints to one. A valid minimal encoding is a two-row sub-table
`[gate_row (TRIGGER=BITB=1, REPEATS=1), data_row (TRIGGER=Immediate, REPEATS=1)]`
iterated via the table-level `SEQ.REPEATS` field. The precise row layout is a
PandA consumer implementation detail.
