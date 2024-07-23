import pytest

from scanspec.regions import Region
from scanspec.specs import Spec, line, squash


def test_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        Region().axis_sets()
    with pytest.raises(NotImplementedError):
        Region().mask({})
    with pytest.raises(NotImplementedError):
        Spec().axes()
    with pytest.raises(NotImplementedError):
        Spec().calculate()
    with pytest.raises(TypeError):
        Spec() * Region()


def test_non_snake_not_allowed_inside_snaking_dim() -> None:
    spec = line("z", 1, 2, 2) * squash(~line("y", 1, 3, 3) * line("x", 0, 2, 3))
    with pytest.raises(ValueError) as cm:
        spec.calculate()
    assert "['x'] would run backwards" in cm.value.args[0]


def test_snake_not_allowed_inside_odd_nested() -> None:
    spec = line("z", 1, 2, 2) * squash(line("y", 1, 3, 3) * ~line("x", 0, 2, 3))
    with pytest.raises(ValueError) as cm:
        spec.calculate()
    assert "['x'] would jump in position" in cm.value.args[0]
