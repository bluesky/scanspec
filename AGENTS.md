# Agent Guidelines

## Read first

- **`PRD.md`** — requirements and current design intent. Authoritative; read
  it before designing or implementing anything.
- **`API_SPEC.md`** — annotated consumption-API examples. Partially stale
  (see PRD §10); where it disagrees with `src/scanspec/v2/` + PRD, the
  latter win.
- **`docs/explanations/decisions/`** — ADRs. 0001–0007 all accepted; several
  are partially or fully superseded by later ADRs in the set (0003, 0005,
  0006 — see each file's own Status field). ADR 0003 Decision 6 has a
  carve-out: `TriggerRepeat`/`TriggerSequence`/`TriggerChild` are pydantic
  `BaseModel`s, not plain dataclasses, because `TriggerSequence` is also
  caller-authored input that must survive a JSON round trip (PRD §9).

## Repository structure

1.x and 2.0 coexist during 2.0 development, as sibling packages under one
top-level `scanspec` distribution:

- `src/scanspec/` (everything except `v2/`) — the 1.x package. **Do not
  modify.** Reference only; don't load it into context unless porting a
  specific algorithm.
- `src/scanspec/v2/` — the 2.0 package, nested as a submodule
  (`import scanspec.v2`). All new work goes here. Not yet the top-level
  `scanspec` name — that happens at final 2.0 release (PRD §12, Phase 2).

Tests mirror this: `tests/` (flat, everything except `scanspec/v2/`) covers
1.x (do not modify); `tests/scanspec/v2/` is where all new tests go.

Branch flow: feature branches → PRs against `bluesky/scanspec:v2-dev` →
`v2-dev` merges to `main` only at the final 2.0 migration (PRD §12).

## Known churn — check before building on these

- `Acquire.detectors: Sequence[DetectorGroup]` may be replaced —
  `_bake_trigger_sequence` currently guesses the parent/child hierarchy by
  ranking `DetectorGroup`s by duration; the maintainer wants `Acquire` to
  stop making that decision, with the caller supplying an already-built
  hierarchy instead. Not yet designed (PRD §11). Avoid new code that
  deepens coupling to the current auto-ranking logic.
- Naming that the docs sometimes get wrong: the code uses
  `Window.non_linear` (not `non_linear_move`), `Scan.has_moving_axes` /
  `Scan.non_linear` (there is no `Scan.fly`), and
  `Scan.with_start(window, trigger_index)` (not `time`).

## Testing conventions

- pytest-style **functions**, not `unittest` classes.
- Simple, direct assertions; test **public interfaces**; avoid mocks unless
  there is no other way.
- No serialisation tests for plain dataclasses (they carry none).
- **`tests/scanspec/v2/test_use_cases.py` is the maintainer's file.** Never
  add, remove, or modify tests in it without explicit permission. Put your
  tests in `test_compile.py`, `test_core.py`, `test_specs.py`, etc.
- **Assert real, independently-derived expected values — not just shape,
  direction, or internal consistency.** A test that only checks
  `len(...) > 0`, `a < b`, or that two derived quantities agree with each
  other (e.g. `start_velocity == end_velocity`) will pass even if the
  underlying computation is wrong by a constant factor or unit — it can
  never catch a bug that scales or shifts every value equally. Derive the
  expected value independently (by hand, from the spec/math), not by
  running the implementation and asserting on its own output. See
  `e2207568` for a concrete example: a velocity/position unit-conversion
  bug survived undetected through the whole implementation because every
  existing test either used a masking special case or asserted only
  shape/consistency.

## Quality gates — all three must pass after every change

```bash
pytest tests/scanspec/v2/ -v
python -m pyright src/scanspec/v2/ tests/scanspec/v2/   # 0 errors
ruff check src/scanspec/v2/ tests/scanspec/v2/          # 0 errors
```

## Type annotations and lint

- No `# type: ignore`. If a type error can't be fixed structurally, leave it
  unsuppressed and report the remaining pyright errors at the end of the task.
- `# noqa: <code>` only for genuinely unfixable violations (e.g. `UP007` on
  a dynamic `Union[tuple(...)]`).

## Working style

- Raise questions or errors rather than guessing on design ambiguity (e.g.
  mismatched snake flags in `Zip` raise; they are not silently reconciled).
- Scratch/prototype files go in `/workspaces/scanspec/scratch/`, never
  `/tmp`; delete or incorporate them after verification.
- `CONTEXT.*.md` files (if present locally) are private working notes — never
  commit them or reference them in committed files.
