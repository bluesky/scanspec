from typing import Any

import pytest


def approx(
    expected: Any,
    rel: float | None = None,
    abs: float | None = None,
    nan_ok: bool = False,
) -> Any:
    """
    Temporary loosely typed wrapper around approx.
    To be removed pending:
    https://github.com/pytest-dev/pytest/issues/7469

    Args:
        expected: Expected value
        rel: Relative tolerance. Defaults to None.
        abs: Absolute tolerance. Defaults to None.
        nan_ok: Permit nan. Defaults to False.

    Returns:
        Any: Approximate comparator
    """

    return pytest.approx(expected, rel=rel, abs=abs, nan_ok=nan_ok)  # type: ignore
