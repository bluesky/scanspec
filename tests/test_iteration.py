import pytest

from scanspec.specs import Line


def test_line_iterator() -> None:
    x = object()
    inst = Line(x, 0, 1, 5)
    view = inst.create_view()
    assert len(view) == 5
    positions = [d[x] for d in view]
    assert positions == pytest.approx([0, 0.25, 0.5, 0.75, 1.0])
    assert len(view) == 0
