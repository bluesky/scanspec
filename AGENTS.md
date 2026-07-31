# Agent Guidelines

## Read first

- **`PRD.md`** — requirements and current design intent. Authoritative; read
  it before designing or implementing anything.
- **`API_SPEC.md`** — annotated consumption-API examples. Partially stale
  (see PRD §10); where it disagrees with `src/scanspec2/` + PRD, the latter
  win.
- **`docs/explanations/decisions/`** — ADRs. 0001–0005 accepted; 0006
  tentative; 0007 proposed with pending review corrections (PRD §9). Do not
  treat 0006/0007 as settled.

## Repository structure

Two packages coexist during 2.0 development:

- `src/scanspec/` — the 1.x package. **Do not modify.** Reference only;
  don't load it into context unless porting a specific algorithm.
- `src/scanspec2/` — the 2.0 package. All new work goes here.

Tests mirror this: `tests/` covers 1.x (do not modify); `tests/scanspec2/`
is where all new tests go.

Branch flow: feature branches → PRs against `bluesky/scanspec:v2-dev` →
`v2-dev` merges to `main` only at the final 2.0 migration (PRD §12).

## Known churn — check before building on these

- `TriggerPattern`/`TriggerGroup` will merge into a `TriggerNode` tree and
  `Window.trigger_groups` → `Window.trigger_nodes` when ADR 0007 is
  accepted. Avoid new code that deepens coupling to the current split.
- Naming that the docs sometimes get wrong: the code uses
  `Window.non_linear` (not `non_linear_move`), `Scan.has_moving_axes` /
  `Scan.non_linear` (there is no `Scan.fly`), and
  `Scan.with_start(window, trigger_index)` (not `time`).

## Testing conventions

- pytest-style **functions**, not `unittest` classes.
- Simple, direct assertions; test **public interfaces**; avoid mocks unless
  there is no other way.
- No serialisation tests for plain dataclasses (they carry none).
- **`tests/scanspec2/test_use_cases.py` is the maintainer's file.** Never
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
pytest tests/scanspec2/ -v
python -m pyright src/scanspec2/ tests/scanspec2/   # 0 errors
ruff check src/scanspec2/ tests/scanspec2/          # 0 errors
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
