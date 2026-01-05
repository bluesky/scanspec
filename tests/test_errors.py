import pytest

from scanspec.specs import Line, Squash


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
