# scanspec 2.0 — Implementation Plan

Baseline: 94 tests passing, 0 pyright errors, 0 ruff errors.

Each phase runs from a clean context. All code in `src/scanspec2/`,
all tests in `tests/scanspec2/`. Never modify `src/scanspec/` or
`tests/scanspec2/test_use_cases.py`.

Phase exit criteria: all tests pass, 0 pyright errors, 0 ruff errors.

---

## Phase A — WindowGenerator refactor

**Goal**: Replace `Scan(motion_dims=list[Dimension], fly=bool)` with
`Scan(generators=list[WindowGenerator])`. This decouples iteration from
the fly/step decision, enabling Phase B (multi-stream Concat of fly + step
Acquires) without tree surgery.

### Why

Currently `Scan.__iter__` uses a global `fly: bool` to decide whether the
innermost Dimension is stepped or swept. This breaks when Concat joins a
fly Acquire with a step Acquire — the Scan can't be both.

Windows don't have a `fly` concept. They only have `static_axes` and
`moving_axes`. The `fly` flag only controls *how windows are generated*
from Dimensions. The fix: make each generator own its step/fly behavior.

### What is a WindowGenerator

A `WindowGenerator` is an internal (private) object that knows how to
produce windows for one dimension of the scan.

Two flavours, distinguished by linearity (matching `Dimension.non_linear`):

```python
class WindowGenerator(Generic[AxisT]):
    """Abstract base for one dimension of window generation."""
    length: int       # how many setpoints in this dimension
    snake: bool       # whether alternate passes reverse
    axes: list[AxisT] # axes this generator moves
    fly: bool         # if True, sweep continuously; if False, step

    def positions(self, index: int) -> dict[AxisT, float]:
        """Return setpoint positions for a given index (step mode)."""
        ...

    def make_fly_window_data(self, reversed: bool) -> FlyData:
        """Return AxisMotion + positions_fn for a fly sweep."""
        ...
```

Whether a generator is linear or non-linear is determined by `Dimension`'s
existing `non_linear` flag (i.e. `isinstance(position_fn, LinearPositions)`).
This does NOT need two subclasses — a single `WindowGenerator` class wrapping
a `Dimension` is sufficient, since `Dimension` already carries the position
function and linearity flag.

### How Specs build generators

Each `Spec.compile()` produces a `Scan` with a list of `WindowGenerator`s.
The mapping:

| Spec | Generators |
|------|-----------|
| `Linspace` / `Static` / `Spiral` | `[WindowGenerator(dim, fly=False)]` |
| `Product(outer, inner)` | `outer.generators + inner.generators` |
| `Snake(spec)` | Same generators, innermost gets `snake=True` |
| `Zip(left, right)` | Merge innermost generators from both |
| `Concat(left, right)` | **Currently**: merge innermost dims. **Phase B**: `ConcatGenerator(left_gens, right_gens)` |
| `Repeat(spec, n)` | Prepend a new `WindowGenerator(empty_dim, fly=False)` of length `n` |
| `Acquire(spec, fly=False)` | Same generators as inner spec (all stepped) |
| `Acquire(spec, fly=True)` | Same generators, but sets `fly=True` on the innermost |

The key: `Acquire(fly=True)` just flips a flag on the last generator.
No restructuring. No tree surgery.

### How Scan.__iter__ uses generators

```python
def __iter__(self):
    generators = self._generators  # list[WindowGenerator]
    # Iterate outer generators as nested loops (step mode).
    # Innermost generator: step or fly depending on its .fly flag.
    # Same logic as current __iter__, just reading .fly from the
    # generator instead of from self.fly.
```

The current `__iter__` logic stays almost identical:
- `outer_dims` → `generators[:-1]` (always stepped, regardless of `.fly`)
- `inner_dim` → `generators[-1]` — if `.fly`, make a fly window; else step

The *only* difference from current code: `self.fly` disappears from `Scan`,
replaced by `generators[-1].fly`.

### Dimension stays for analysis

`Dimension` is NOT replaced. It remains the public analysis abstraction:
- `Scan.dimensions: list[Dimension]` — convenience property
- `WindowedStream.dimensions: list[Dimension]` — per-stream analysis shape

`WindowGenerator` wraps a `Dimension` (or something equivalent) internally.
Analysis consumers use `dim.setpoints(axis)` as before.

### Scan.fly becomes computed

```python
@property
def fly(self) -> bool:
    """True if any generator produces fly windows."""
    return any(g.fly for g in self._generators)
```

Existing tests that check `scan.fly` continue to pass.

### Implementation details

**Files to modify**: `src/scanspec2/core.py`, `src/scanspec2/specs.py`

**Step 1 — Define WindowGenerator in core.py**

```python
class WindowGenerator(Generic[AxisT]):
    """One dimension of window generation. Wraps a Dimension."""

    def __init__(
        self,
        dim: Dimension[AxisT],
        fly: bool = False,
    ) -> None:
        self.dim = dim
        self.fly = fly

    @property
    def length(self) -> int:
        return self.dim.length

    @property
    def snake(self) -> bool:
        return self.dim.snake

    @property
    def axes(self) -> list[AxisT]:
        return self.dim.axes
```

This is intentionally thin — just a Dimension + fly flag. The step/fly
window creation logic stays in Scan (in `_make_step_window` and
`_make_fly_window`, unchanged from current code).

**Step 2 — Update Scan.__init__**

Replace `motion_dims + fly` with `generators`:

```python
class Scan:
    def __init__(
        self,
        generators: Sequence[WindowGenerator[AxisT]],
        windowed_streams: Sequence[WindowedStream[AxisT, DetectorT]] = (),
        continuous_streams: Sequence[ContinuousStream[DetectorT]] = (),
        monitors: Sequence[MonitorStream[MonitorT]] = (),
        duration: float | None = None,
        start_window: int = 0,
        start_time: float = 0.0,
    ):
        self._generators = list(generators)
        ...

    @property
    def fly(self) -> bool:
        return any(g.fly for g in self._generators)

    @property
    def dimensions(self) -> list[Dimension[AxisT]]:
        return [g.dim for g in self._generators]
```

**Step 3 — Update Scan.__iter__**

Replace `self.fly` with `self._generators[-1].fly`. Replace
`dims[:-1]` / `dims[-1]` with generators. The rest is identical:

```python
if self._generators[-1].fly:
    outer_gens = self._generators[:-1]
    inner_gen = self._generators[-1]
else:
    outer_gens = self._generators
    inner_gen = None
```

`outer_dims` in the current code becomes `[g.dim for g in outer_gens]`.
`_make_fly_window` takes `inner_gen.dim` instead of `inner_dim`.
Duration logic unchanged.

**Step 4 — Update all Spec.compile() methods**

Each compile() currently returns `Scan(motion_dims=[...])`. Change to
`Scan(generators=[WindowGenerator(dim) for dim in ...])`.

`Acquire.compile()` changes from:
```python
return Scan(motion_dims=inner_dims, ..., fly=self.fly)
```
to:
```python
gens = [WindowGenerator(d) for d in inner_dims]
if self.fly and gens:
    gens[-1] = WindowGenerator(gens[-1].dim, fly=True)
return Scan(generators=gens, ...)
```

**Step 5 — Update with_start and helpers**

`with_start` copies generators instead of motion_dims + fly.
`_motion_dims` helper becomes `_generators` or similar.

### What does NOT change

- `Dimension` class — unchanged
- `LinearPositions` — unchanged
- `Window` class — unchanged
- `_make_step_window` — unchanged (reads dims from generators)
- `_make_fly_window` — unchanged (reads dim from generator)
- Snake nesting logic — unchanged (operates on outer generators)
- All existing tests pass — the output (windows) is identical

### Tests

Existing tests continue to pass — this is a refactor. Add in
`test_compile.py`:

- `test_generator_fly_flag`: `Acquire(Linspace(...), fly=True).compile()` →
  last generator has `fly=True`, others have `fly=False`.
- `test_generator_from_motion_spec`: `Linspace(...).compile()` → single
  generator with `fly=False`.
- `test_scan_fly_property`: `scan.fly` returns True iff any generator is fly.

### Open questions for implementation

1. Should `WindowGenerator` be a simple class (as shown) or a `@dataclass`?
   Leaning toward regular class for consistency with `Dimension` and `Window`.

2. The `_motion_dims()` helper in specs.py currently does
   `spec.compile().dimensions`. This needs to become something that returns
   generators, since Product/Snake/etc. need to prepend/modify generators,
   not dimensions. Rename to `_generators(spec)` → `spec.compile()._generators`?
   Or add a public `scan.generators` property?

3. `Scan.dimensions` currently returns `self._motion_dims`. After refactor
   it returns `[g.dim for g in self._generators]`. This is semantically
   identical. But should `dimensions` be the Dimensions from generators, or
   from `windowed_streams[0].dimensions`? Currently they're the same object
   references. Keep them the same for now, revisit in Phase B when streams
   may have different dimensions.

4. Current `Concat.compile()` merges innermost Dimensions (summing lengths).
   After this refactor it still does the same thing — it merges the innermost
   generators' dims. Phase B will change this to support different fly modes.
   For now, Concat works identically to before.

**Context to load for Phase A**: This section of `plan.md`.
`src/scanspec2/core.py` (all). `src/scanspec2/specs.py` (all).
`tests/scanspec2/test_compile.py` (all). `tests/scanspec2/test_use_cases.py`
(read-only reference). `AGENTS.md`.

---

## Phase B — trigger_groups + Multi-stream Concat

**Goal**: Two things that interlock:

1. Populate `window.trigger_groups` from `Acquire.detectors`.
2. Enable Concat of Acquires with different `stream_name` and different
   fly modes. Resolves API_SPEC open question 1.

### trigger_groups

Bake each `DetectorGroup` into a `TriggerGroup` with `TriggerPattern`:
- Step scan: `TriggerPattern(repeats=1, livetime, deadtime)` per group.
- Fly scan: `TriggerPattern(repeats=inner_dim.length, livetime, deadtime)`.
  `exposures_per_collection` multiplies `repeats`.

Pass trigger_groups through `Scan` → `Window`. When detectors are present,
derive `duration` from trigger timing.

### Multi-stream Concat

The flagship use case (thoughts.md L59–66):

```python
spec = Repeat(
    Acquire(Static("e", e0), detectors=[diff_det], stream_name="diff")
    .concat(Acquire(Linspace("e", e1, e2, 1000), fly=True,
                    detectors=[spec_det], stream_name="spec"))
    .concat(Acquire(Linspace("e", e2, e1, 1000), fly=True,
                    detectors=[spec_det], stream_name="spec")),
    num=200,
)
```

With WindowGenerator from Phase A, each Acquire compiles to generators
with its own fly flag. `Concat.compile()` concatenates the generator
lists from its children. When the children have different fly modes or
different stream names, the generators stay separate — Scan iterates
them sequentially.

This requires a new generator type:

```python
class ConcatGenerator(Generic[AxisT]):
    """Concatenates window sequences from multiple child generators."""
    children: list[list[WindowGenerator[AxisT]]]
```

`Scan.__iter__` needs to handle `ConcatGenerator` in its iteration.
When it encounters one, it runs each child's generators sequentially.

The `windowed_streams` on the resulting `Scan` must reflect per-stream
dimensions:
- Streams with different names → separate `WindowedStream` entries.
- Streams with same name → merged (new outer concatenation dimension).

### Tests (in `test_compile.py`)

- `test_step_scan_trigger_groups`: Acquire with detectors, fly=False.
- `test_fly_scan_trigger_groups`: Acquire with detectors, fly=True.
- `test_multirate_trigger_groups`: Two detector groups at different rates.
- `test_duration_derived_from_detectors`: window.duration from triggers.
- `test_concat_fly_step_windows`: Concat of fly + step Acquires.
- `test_concat_different_streams`: Two Acquires, different stream_names.
- `test_repeat_concat_flagship`: Full flagship pattern.

**Context to load**: `plan.md` Phase B. `API_SPEC.md` (all).
`src/scanspec2/core.py` (post Phase A). `src/scanspec2/specs.py` (post
Phase A). `tests/scanspec2/test_compile.py`.

---

## Phase C — Ellipse and Polygon specs

**Goal**: Port `Ellipse` and `Polygon` from 1.x as motion spec nodes.
No API_SPEC addition needed.

**Files**: `src/scanspec2/specs.py`

**Work**:

1. **Ellipse**: Grid of points masked to an elliptical footprint. Two axes,
   centre, diameters, step sizes. `compile()` produces generators with a
   single Dimension containing the masked grid points.

2. **Polygon**: Same pattern, polygon mask (even-odd ray-casting).

3. Both expose `snake` and `vertical` (which axis is fast).

**Tests** (in `test_specs.py`):
- `test_ellipse_basic`, `test_polygon_triangle`, `test_ellipse_fly`,
  `test_polygon_snake`.

**Context to load**: `src/scanspec/specs.py` (1.x Ellipse + Polygon).
`src/scanspec2/specs.py` (post Phase B). Existing tests.

---

## Phase D — Range + `__init__.py` + final validation

**Goal**: Fill remaining gaps.

1. **Range**: `Range(axis, start, stop, step)` → Linspace with
   `num = abs(round((stop - start) / step)) + 1`.

2. **`__init__.py`**: Define `__all__`, verify `from scanspec2 import *`.

3. **Full validation**: `tox -p`.

**Tests** (in `test_specs.py`):
- `test_range_basic`, `test_range_negative_step`.

---

## Phase ordering

```
Phase A (WindowGenerator refactor)
    ↓
Phase B (trigger_groups + multi-stream Concat)
    ↓
Phase C (Ellipse / Polygon)
    ↓
Phase D (Range + exports + final validation)
```
