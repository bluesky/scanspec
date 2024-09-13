from typing import Any

import pytest

from scanspec.regions import Region
from scanspec.specs import Line, Spec, Squash


def test_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        Region().axis_sets()
    with pytest.raises(NotImplementedError):
        region: Region[Any] = Region()
        region.mask({})
    with pytest.raises(NotImplementedError):
        Spec().axes()
    with pytest.raises(NotImplementedError):
        Spec().calculate()
    with pytest.raises(TypeError):
        Spec() * Region()  # type: ignore


def test_non_snake_not_allowed_inside_snaking_dim() -> None:
    spec = Line("z", 1, 2, 2) * Squash(~Line("y", 1, 3, 3) * Line("x", 0, 2, 3))
    with pytest.raises(ValueError) as cm:
        spec.calculate()
    assert "['x'] would run backwards" in cm.value.args[0]


def test_snake_not_allowed_inside_odd_nested() -> None:
    spec = Line("z", 1, 2, 2) * Squash(Line("y", 1, 3, 3) * ~Line("x", 0, 2, 3))
    with pytest.raises(ValueError) as cm:
        spec.calculate()
    assert "['x'] would jump in position" in cm.value.args[0]
