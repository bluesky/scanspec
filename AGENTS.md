# Agent Guidelines

## Repository structure

This repository contains two packages during the scanspec 2.0 development period:

- `src/scanspec/` — the original 1.x package. **Do not modify this.** It is kept as a reference implementation.
- `src/scanspec2/` — the new 2.0 package under active development. All new work goes here.

When `scanspec2` is feature-complete (all PRD requirements met and tests passing), the migration will be:
1. Delete `src/scanspec/`.
2. Rename `src/scanspec2/` → `src/scanspec/`.
3. Update `pyproject.toml` and any import references accordingly.

## Where to work

**Always make changes in `src/scanspec2/`**, never in `src/scanspec/`.

All new tests go in `tests/scanspec2/`. Do not modify tests in `tests/` (those cover the 1.x package).

The PRD is in `prd.md`. The design notes and user stories are in `thoughts.md`.
