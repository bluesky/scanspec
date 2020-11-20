import pytest

from scanspec.core import View
from scanspec.specs import Concat, Line, Snake


def test_line_iterator() -> None:
    x = object()
    inst = Line(x, 0, 1, 5)
    view = inst.create_view()
    assert len(view) == 5
    positions = [d[x] for d in view]
    assert positions == pytest.approx([0, 0.25, 0.5, 0.75, 1.0])
    assert len(view) == 0


def test_zipped_lines() -> None:
    x, y = object(), object()
    lx, ly = Line(x, 0, 1, 5), Line(y, 1, 2, 5)
    inst = lx + ly
    assert inst.keys == [x, y]
    dims = inst.create_dimensions()
    assert len(dims) == 1
    assert dims[0].positions == {
        x: pytest.approx([0, 0.25, 0.5, 0.75, 1]),
        y: pytest.approx([1, 1.25, 1.5, 1.75, 2]),
    }


def test_product_lines() -> None:
    x, y = object(), object()
    lx, ly = Line(x, 0, 1, 2), Line(y, 1, 2, 3)
    inst = ly * lx
    assert inst.keys == [y, x]
    dims = inst.create_dimensions()
    assert len(dims) == 2
    view = View(dims)
    batch = view.create_batch(1000)
    assert batch.positions == {
        x: pytest.approx([0, 1, 0, 1, 0, 1]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert batch.lower == {
        x: pytest.approx([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert batch.upper == {
        x: pytest.approx([0.5, 1.5, 0.5, 1.5, 0.5, 1.5]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2]),
    }


def test_product_snaking_lines() -> None:
    x, y = object(), object()
    lx, ly = Line(x, 0, 1, 2), Line(y, 1, 2, 3)
    inst = Snake(ly * lx)
    assert inst.keys == [y, x]
    dims = inst.create_dimensions()
    assert len(dims) == 2
    view = View(dims)
    batch = view.create_batch(1000)
    assert batch.positions == {
        x: pytest.approx([0, 1, 1, 0, 0, 1]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert batch.lower == {
        x: pytest.approx([-0.5, 0.5, 1.5, 0.5, -0.5, 0.5]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert batch.upper == {
        x: pytest.approx([0.5, 1.5, 0.5, -0.5, 0.5, 1.5]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2]),
    }


def test_concat_lines() -> None:
    x = object()
    inst = Concat(Line(x, 0, 1, 2), Line(x, 1, 2, 3))
    view = inst.create_view()
    batch = view.create_batch(1000)
    assert batch.positions == {x: pytest.approx([0, 1, 1, 1.5, 2])}
    assert batch.lower == {x: pytest.approx([-0.5, 0.5, 0.75, 1.25, 1.75])}
    assert batch.upper == {x: pytest.approx([0.5, 1.5, 1.25, 1.75, 2.25])}
