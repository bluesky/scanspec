import pytest

from scanspec.core import View
from scanspec.specs import Line


def test_line_view() -> None:
    x = object()
    inst = Line(x, 0, 1, 5)
    view = View(inst.create_dimensions())
    assert len(view) == 5
    dim = view.consume()
    assert dim.positions == {x: pytest.approx([0, 0.25, 0.5, 0.75, 1.0])}
    assert len(view) == 0


def test_line_positions() -> None:
    x = object()
    inst = Line(x, 0, 1, 5)
    it = inst.positions()
    assert it.keys == [x]
    assert len(it) == 5
    positions = [d[x] for d in it]
    assert positions == pytest.approx([0, 0.25, 0.5, 0.75, 1.0])
