import pytest

from scanspec.core import Path
from scanspec.regions import Circle, Rectangle
from scanspec.specs import TIME, Concat, Line, Mask, Spiral, Squash, Static, fly, step


def test_one_point_line() -> None:
    x = object()
    inst = Line(x, 0, 1, 1)
    (dim,) = inst.create_dimensions()
    assert dim.positions == {x: pytest.approx([0])}
    assert dim.lower == {x: pytest.approx([-0.5])}
    assert dim.upper == {x: pytest.approx([0.5])}
    assert dim.snake is False


def test_two_point_line() -> None:
    x = object()
    inst = Line(x, 0, 1, 2)
    (dim,) = inst.create_dimensions()
    assert dim.positions == {x: pytest.approx([0, 1])}
    assert dim.lower == {x: pytest.approx([-0.5, 0.5])}
    assert dim.upper == {x: pytest.approx([0.5, 1.5])}


def test_two_point_stepped_line() -> None:
    x = object()
    inst = step(Line(x, 0, 1, 2), 0.1)
    dimx, dimt = inst.create_dimensions()
    assert dimx.positions == dimx.lower == dimx.upper == {x: pytest.approx([0, 1])}
    assert dimt.positions == dimt.lower == dimt.upper == {TIME: pytest.approx([0.1])}


def test_two_point_fly_line() -> None:
    x = object()
    inst = fly(Line(x, 0, 1, 2), 0.1)
    (dim,) = inst.create_dimensions()
    assert dim.positions == {x: pytest.approx([0, 1]), TIME: pytest.approx([0.1, 0.1])}
    assert dim.lower == {x: pytest.approx([-0.5, 0.5]), TIME: pytest.approx([0.1, 0.1])}
    assert dim.upper == {x: pytest.approx([0.5, 1.5]), TIME: pytest.approx([0.1, 0.1])}


def test_many_point_line() -> None:
    x = object()
    inst = Line(x, 0, 1, 5)
    (dim,) = inst.create_dimensions()
    assert dim.positions == {x: pytest.approx([0, 0.25, 0.5, 0.75, 1])}
    assert dim.lower == {x: pytest.approx([-0.125, 0.125, 0.375, 0.625, 0.875])}
    assert dim.upper == {x: pytest.approx([0.125, 0.375, 0.625, 0.875, 1.125])}


def test_one_point_bounded_line() -> None:
    x = object()
    inst = Line.bounded(x, 0, 1, 1)
    assert inst == Line(x, 0.5, 1.5, 1)


def test_many_point_bounded_line() -> None:
    x = object()
    inst = Line.bounded(x, 0, 1, 4)
    assert inst == Line(x, 0.125, 0.875, 4)


def test_spiral() -> None:
    x, y = object(), object()
    inst = Spiral(x, y, 0, 10, 5, 50, 10)
    (dim,) = inst.create_dimensions()
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


def test_zipped_lines() -> None:
    x, y = object(), object()
    inst = Line(x, 0, 1, 5) + Line(y, 1, 2, 5)
    assert inst.keys() == [x, y]
    (dim,) = inst.create_dimensions()
    assert dim.positions == {
        x: pytest.approx([0, 0.25, 0.5, 0.75, 1]),
        y: pytest.approx([1, 1.25, 1.5, 1.75, 2]),
    }


def test_product_lines() -> None:
    x, y = object(), object()
    inst = Line(y, 1, 2, 3) * Line(x, 0, 1, 2)
    assert inst.keys() == [y, x]
    dims = inst.create_dimensions()
    assert len(dims) == 2
    dim = Path(dims).consume()
    assert dim.positions == {
        x: pytest.approx([0, 1, 0, 1, 0, 1]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert dim.lower == {
        x: pytest.approx([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert dim.upper == {
        x: pytest.approx([0.5, 1.5, 0.5, 1.5, 0.5, 1.5]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2]),
    }


def test_zipped_product_lines() -> None:
    x, y, z = object(), object(), object()
    inst = Line(x, 0, 1, 5) + Line(y, 1, 2, 3) * Line(z, 2, 3, 5)
    assert inst.keys() == [x, y, z]
    dimy, dimxz = inst.create_dimensions()
    assert dimxz.positions == {
        x: pytest.approx([0, 0.25, 0.5, 0.75, 1]),
        z: pytest.approx([2, 2.25, 2.5, 2.75, 3]),
    }
    assert dimy.positions == {
        y: pytest.approx([1, 1.5, 2]),
    }


def test_squashed_product() -> None:
    x, y = object(), object()
    inst = Squash(Line(y, 1, 2, 3) * Line(x, 0, 1, 2))
    assert inst.keys() == [y, x]
    (dim,) = inst.create_dimensions()
    assert dim.positions == {
        x: pytest.approx([0, 1, 0, 1, 0, 1]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert dim.lower == {
        x: pytest.approx([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert dim.upper == {
        x: pytest.approx([0.5, 1.5, 0.5, 1.5, 0.5, 1.5]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2]),
    }


def test_squashed_multiplied_snake_scan() -> None:
    x, y, z = object(), object(), object()
    inst = Line(z, 1, 2, 2) * Squash(
        Line(y, 1, 2, 2) * ~Line.bounded(x, 3, 7, 2) * Static(TIME, 9, 2)
    )
    assert inst.keys() == [z, y, x, TIME]
    dimz, dimxyt = inst.create_dimensions()
    for d in dimxyt.positions, dimxyt.lower, dimxyt.upper:
        assert d == {
            x: pytest.approx([4, 4, 6, 6, 6, 6, 4, 4]),
            y: pytest.approx([1, 1, 1, 1, 2, 2, 2, 2]),
            TIME: pytest.approx([9, 9, 9, 9, 9, 9, 9, 9]),
        }
    assert (
        dimz.positions == dimz.lower == dimz.upper == {z: pytest.approx([1, 2])}
    )


def test_product_snaking_lines() -> None:
    x, y = object(), object()
    inst = Line(y, 1, 2, 3) * ~Line(x, 0, 1, 2)
    assert inst.keys() == [y, x]
    dims = inst.create_dimensions()
    assert len(dims) == 2
    dim = Path(dims).consume()
    assert dim.positions == {
        x: pytest.approx([0, 1, 1, 0, 0, 1]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert dim.lower == {
        x: pytest.approx([-0.5, 0.5, 1.5, 0.5, -0.5, 0.5]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert dim.upper == {
        x: pytest.approx([0.5, 1.5, 0.5, -0.5, 0.5, 1.5]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2]),
    }


def test_concat_lines() -> None:
    x = object()
    inst = Concat(Line(x, 0, 1, 2), Line(x, 1, 2, 3))
    assert inst.keys() == [x]
    (dim,) = inst.create_dimensions()
    assert dim.positions == {x: pytest.approx([0, 1, 1, 1.5, 2])}
    assert dim.lower == {x: pytest.approx([-0.5, 0.5, 0.75, 1.25, 1.75])}
    assert dim.upper == {x: pytest.approx([0.5, 1.5, 1.25, 1.75, 2.25])}


def test_rect_region() -> None:
    x, y = object(), object()
    inst = Line(y, 1, 3, 5) * Line(x, 0, 2, 3) & Rectangle(x, y, 0, 1, 1.5, 2.2)
    assert inst.keys() == [y, x]
    (dim,) = inst.create_dimensions()
    assert dim.positions == {
        x: pytest.approx([0, 1, 0, 1, 0, 1]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert dim.lower == {
        x: pytest.approx([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert dim.upper == {
        x: pytest.approx([0.5, 1.5, 0.5, 1.5, 0.5, 1.5]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2]),
    }


def test_rect_region_3D() -> None:
    x, y, z = object(), object(), object()
    inst = Static(z, 3.2, 2) * Line(y, 1, 3, 5) * Line(x, 0, 2, 3) & Rectangle(
        x, y, 0, 1, 1.5, 2.2
    )
    assert inst.keys() == [z, y, x]
    zdim, xydim = inst.create_dimensions()
    assert zdim.positions == {z: pytest.approx([3.2, 3.2])}
    assert zdim.positions is zdim.upper
    assert zdim.positions is zdim.lower
    assert xydim.positions == {
        x: pytest.approx([0, 1, 0, 1, 0, 1]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert xydim.lower == {
        x: pytest.approx([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert xydim.upper == {
        x: pytest.approx([0.5, 1.5, 0.5, 1.5, 0.5, 1.5]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2]),
    }


def test_rect_region_union() -> None:
    x, y = object(), object()
    inst = Line(y, 1, 3, 5) * Line(x, 0, 2, 3) & Rectangle(
        x, y, 0, 1, 1.5, 2.2
    ) | Rectangle(x, y, 0.5, 1.5, 2, 2.5)
    assert inst.keys() == [y, x]
    (dim,) = inst.create_dimensions()
    assert dim.positions == {
        x: pytest.approx([0, 1, 0, 1, 2, 0, 1, 2, 1, 2]),
        y: pytest.approx([1, 1, 1.5, 1.5, 1.5, 2, 2, 2, 2.5, 2.5]),
    }


def test_rect_region_intersection() -> None:
    x, y = object(), object()
    inst = (
        Line(y, 1, 3, 5) * Line(x, 0, 2, 3)
        & Rectangle(x, y, 0, 1, 1.5, 2.2)
        & Rectangle(x, y, 0.5, 1.5, 2, 2.5)
    )
    assert inst.keys() == [y, x]
    (dim,) = inst.create_dimensions()
    assert dim.positions == {
        x: pytest.approx([1, 1]),
        y: pytest.approx([1.5, 2]),
    }


def test_rect_region_difference() -> None:
    x, y, t = object(), object(), object()
    # Bracket to
    inst = (
        Line(y, 1, 3, 5) * Line(x, 0, 2, 3) + Static(t, 0.1)
        & Rectangle(x, y, 0, 1, 1.5, 2.2)
    ) - Rectangle(x, y, 0.5, 1.5, 2, 2.5)
    assert inst.keys() == [y, x, t]
    (dim,) = inst.create_dimensions()
    assert dim.positions == {
        x: pytest.approx([0, 1, 0, 0]),
        y: pytest.approx([1, 1, 1.5, 2]),
        t: pytest.approx([0.1, 0.1, 0.1, 0.1]),
    }


def test_rect_region_symmetricdifference() -> None:
    x, y = object(), object()
    inst = Line(y, 1, 3, 5) * Line(x, 0, 2, 3) & Rectangle(
        x, y, 0, 1, 1.5, 2.2
    ) ^ Rectangle(x, y, 0.5, 1.5, 2, 2.5)
    assert inst.keys() == [y, x]
    (dim,) = inst.create_dimensions()
    assert dim.positions == {
        x: pytest.approx([0, 1, 0, 2, 0, 2, 1, 2]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2, 2.5, 2.5]),
    }


def test_circle_region() -> None:
    x, y = object(), object()
    inst = Line(y, 1, 3, 3) * Line(x, 0, 2, 3) & Circle(x, y, 1, 2, 1)
    assert inst.keys() == [y, x]
    (dim,) = inst.create_dimensions()
    assert dim.positions == {
        x: pytest.approx([1, 0, 1, 2, 1]),
        y: pytest.approx([1, 2, 2, 2, 3]),
    }
    assert dim.lower == {
        x: pytest.approx([0.5, -0.5, 0.5, 1.5, 0.5]),
        y: pytest.approx([1, 2, 2, 2, 3]),
    }
    assert dim.upper == {
        x: pytest.approx([1.5, 0.5, 1.5, 2.5, 1.5]),
        y: pytest.approx([1, 2, 2, 2, 3]),
    }


def test_circle_snaked_region() -> None:
    x, y = object(), object()
    inst = Mask(
        Line(y, 1, 3, 3) * ~Line(x, 0, 2, 3),
        Circle(x, y, 1, 2, 1),
        check_path_changes=False,
    )
    assert inst.keys() == [y, x]
    (dim,) = inst.create_dimensions()
    assert dim.positions == {
        x: pytest.approx([1, 2, 1, 0, 1]),
        y: pytest.approx([1, 2, 2, 2, 3]),
    }
    assert dim.lower == {
        x: pytest.approx([0.5, 2.5, 1.5, 0.5, 0.5]),
        y: pytest.approx([1, 2, 2, 2, 3]),
    }
    assert dim.upper == {
        x: pytest.approx([1.5, 1.5, 0.5, -0.5, 1.5]),
        y: pytest.approx([1, 2, 2, 2, 3]),
    }
