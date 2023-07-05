import pytest

from scanspec.regions import Region
from scanspec.specs import Line, Spec, Squash


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
    spec = Line(axis="z", start=1, stop=2, num=2) * \
           Squash(spec=~Line(axis="y", start=1, stop=3, num=3) * Line(axis="x", start=0, stop=2, num=3))
    with pytest.raises(ValueError) as cm:
        spec.calculate()
    assert "['x'] would run backwards" in cm.value.args[0]


def test_snake_not_allowed_inside_odd_nested() -> None:
    spec = Line(axis="z", start=1, stop=2, num=2) * \
           Squash(spec=Line(axis="y", start=1, stop=3, num=3) * ~Line(axis="x", start=0, stop=2, num=3))
    with pytest.raises(ValueError) as cm:
        spec.calculate()
    assert "['x'] would jump in position" in cm.value.args[0]
