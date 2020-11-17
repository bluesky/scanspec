import pytest

from scanspec.core import View
from scanspec.specs import Concat, Line, Snake


def test_one_point_line() -> None:
    x = object()
    inst = Line(x, 0, 1, 1)
    dim = inst.create_dimensions()[0]
    assert dim.positions == {x: pytest.approx([0])}
    assert dim.lower == {x: pytest.approx([-0.5])}
    assert dim.upper == {x: pytest.approx([0.5])}
    assert dim.snake is False


def test_two_point_line() -> None:
    x = object()
    inst = Line(x, 0, 1, 2)
    dim = inst.create_dimensions()[0]
    assert dim.positions == {x: pytest.approx([0, 1])}
    assert dim.lower == {x: pytest.approx([-0.5, 0.5])}
    assert dim.upper == {x: pytest.approx([0.5, 1.5])}


def test_many_point_line() -> None:
    x = object()
    inst = Line(x, 0, 1, 5)
    dim = inst.create_dimensions()[0]
    assert dim.positions == {x: pytest.approx([0, 0.25, 0.5, 0.75, 1])}
    assert dim.lower == {x: pytest.approx([-0.125, 0.125, 0.375, 0.625, 0.875])}
    assert dim.upper == {x: pytest.approx([0.125, 0.375, 0.625, 0.875, 1.125])}


def test_one_point_bounded_line() -> None:
    x = object()
    inst = Line.bounded(x, 0, 1, 1)
    assert inst == Line(x, 0.5, 1.5, 1)
    dim = inst.create_dimensions()[0]
    assert dim.positions == {x: pytest.approx([0.5])}
    assert dim.lower == {x: pytest.approx([0])}
    assert dim.upper == {x: pytest.approx([1])}


def test_many_point_bounded_line() -> None:
    x = object()
    inst = Line.bounded(x, 0, 1, 4)
    assert inst == Line(x, 0.125, 0.875, 4)
    dim = inst.create_dimensions()[0]
    assert dim.positions == {x: pytest.approx([0.125, 0.375, 0.625, 0.875])}
    assert dim.lower == {x: pytest.approx([0, 0.25, 0.5, 0.75])}
    assert dim.upper == {x: pytest.approx([0.25, 0.5, 0.75, 1])}
    assert dim.snake is False


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
