# Agent Guidelines

## Read first

- **`PRD.md`** — requirements and current design intent. Authoritative; read
  it before designing or implementing anything.
- **`API_SPEC.md`** — annotated consumption-API examples.
- **`docs/explanations/decisions/`** — ADRs. Historical records, several
  partially superseded by later ones (see each file's Status field); they
  will be consolidated later. Where anything disagrees with
  `src/scanspec/v2/` + `PRD.md`, the latter win.

## Repository structure

- `src/scanspec/v2/` + `tests/scanspec/v2/` — the 2.0 package and its tests.
  All new work goes here. Becomes the top-level `scanspec` at the final 2.0
  release (PRD §12).
- Everything else under `src/scanspec/` and `tests/` is 1.x. **Do not
  modify**; don't load it into context unless porting a specific algorithm.

Branch flow: feature branches → PRs against `bluesky/scanspec:v2-dev`.

## Testing conventions

- pytest-style **functions**; simple, direct assertions against **public
  interfaces**; avoid mocks unless there is no other way.
- **`tests/scanspec/v2/test_use_cases.py` is the maintainer's file.** Never
  add, remove, or modify tests in it without explicit permission.
- **Assert independently-derived expected values** (by hand, from the
  spec/math) — not just shape, direction, or agreement between two derived
  quantities, which can't catch a bug that scales every value equally (see
  `e2207568`).

## Quality gate

`tox -p` must pass after every change (pre-commit/ruff, pyright, pytest,
docs — the same envs CI runs).

Prefer structural fixes over `# type: ignore` / `# noqa`; when suppressing is
genuinely necessary (e.g. a test deliberately passing a wrong type), always
name the specific code.

## Working style

Raise questions or errors on design ambiguity rather than guessing (e.g.
mismatched snake flags in `Zip` raise; they are not silently reconciled).
