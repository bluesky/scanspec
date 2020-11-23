import pytest

from scanspec.specs import Line, Spiral


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


def test_spiral() -> None:
    x, y = object(), object()
    inst = Spiral(x, y, 0, 10, 5, 50, 10)
    dim = inst.create_dimensions()[0]
    assert dim.positions == {
        y: pytest.approx(
            [5.4, 6.4, 19.7, 23.8, 15.4, 1.7, -8.6, -10.7, -4.1, 8.3], abs=0.1
        ),
        x: pytest.approx(
            [0.3, -0.9, -0.7, 0.5, 1.5, 1.6, 0.7, -0.6, -1.8, -2.4], abs=0.1
        ),
    }
    assert dim.lower == {
        y: pytest.approx(
            [10.0, 2.7, 13.3, 23.5, 20.9, 8.7, -4.2, -10.8, -8.4, 1.6], abs=0.1
        ),
        x: pytest.approx(
            [0.0, -0.3, -1.0, -0.1, 1.1, 1.7, 1.3, 0.0, -1.2, -2.2], abs=0.1
        ),
    }
    assert dim.upper == {
        y: pytest.approx(
            [2.7, 13.3, 23.5, 20.9, 8.7, -4.2, -10.8, -8.4, 1.6, 15.3], abs=0.1
        ),
        x: pytest.approx(
            [-0.3, -1.0, -0.1, 1.1, 1.7, 1.3, 0.0, -1.2, -2.2, -2.4], abs=0.1
        ),
    }
    assert dim.snake is False


def test_spaced_spiral() -> None:
    x, y = object(), object()
    inst = Spiral.spaced(x, y, 0, 10, 5, 1)
    assert inst == Spiral(x, y, 0, 10, 5, 5, 78)
