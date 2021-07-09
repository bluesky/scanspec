import pytest

from scanspec.core import Path
from scanspec.regions import Circle, Ellipse, Polygon, Rectangle
from scanspec.specs import (
    DURATION,
    Concat,
    Line,
    Mask,
    Spiral,
    Squash,
    Static,
    fly,
    repeat,
    step,
)

x, y, z = "x", "y", "z"


def test_one_point_duration() -> None:
    duration = Static.duration(1.0)
    (dim,) = duration.create_dimensions()
    assert dim.midpoints == {DURATION: pytest.approx([1.0])}
    assert dim.lower == {DURATION: pytest.approx([1.0])}
    assert dim.upper == {DURATION: pytest.approx([1.0])}
    assert dim.snake is False


def test_one_point_line() -> None:
    inst = Line(x, 0, 1, 1)
    (dim,) = inst.create_dimensions()
    assert dim.midpoints == {x: pytest.approx([0])}
    assert dim.lower == {x: pytest.approx([-0.5])}
    assert dim.upper == {x: pytest.approx([0.5])}
    assert dim.snake is False


def test_two_point_line() -> None:
    inst = Line(x, 0, 1, 2)
    (dim,) = inst.create_dimensions()
    assert dim.midpoints == {x: pytest.approx([0, 1])}
    assert dim.lower == {x: pytest.approx([-0.5, 0.5])}
    assert dim.upper == {x: pytest.approx([0.5, 1.5])}


def test_two_point_stepped_line() -> None:
    inst = step(Line(x, 0, 1, 2), 0.1)
    dimx, dimt = inst.create_dimensions()
    assert dimx.midpoints == dimx.lower == dimx.upper == {x: pytest.approx([0, 1])}
    assert (
        dimt.midpoints == dimt.lower == dimt.upper == {DURATION: pytest.approx([0.1])}
    )


def test_two_point_fly_line() -> None:
    inst = fly(Line(x, 0, 1, 2), 0.1)
    (dim,) = inst.create_dimensions()
    assert dim.midpoints == {
        x: pytest.approx([0, 1]),
        DURATION: pytest.approx([0.1, 0.1]),
    }
    assert dim.lower == {
        x: pytest.approx([-0.5, 0.5]),
        DURATION: pytest.approx([0.1, 0.1]),
    }
    assert dim.upper == {
        x: pytest.approx([0.5, 1.5]),
        DURATION: pytest.approx([0.1, 0.1]),
    }


def test_many_point_line() -> None:
    inst = Line(x, 0, 1, 5)
    (dim,) = inst.create_dimensions()
    assert dim.midpoints == {x: pytest.approx([0, 0.25, 0.5, 0.75, 1])}
    assert dim.lower == {x: pytest.approx([-0.125, 0.125, 0.375, 0.625, 0.875])}
    assert dim.upper == {x: pytest.approx([0.125, 0.375, 0.625, 0.875, 1.125])}


def test_one_point_bounded_line() -> None:
    inst = Line.bounded(x, 0, 1, 1)
    assert inst == Line(x, 0.5, 1.5, 1)


def test_many_point_bounded_line() -> None:
    inst = Line.bounded(x, 0, 1, 4)
    assert inst == Line(x, 0.125, 0.875, 4)


def test_spiral() -> None:
    inst = Spiral(x, y, 0, 10, 5, 50, 10)
    (dim,) = inst.create_dimensions()
    assert dim.midpoints == {
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
    inst = Spiral.spaced(x, y, 0, 10, 5, 1)
    assert inst == Spiral(x, y, 0, 10, 10, 10, 78)


def test_zipped_lines() -> None:
    inst = Line(x, 0, 1, 5) + Line(y, 1, 2, 5)
    assert inst.axes() == [x, y]
    (dim,) = inst.create_dimensions()
    assert dim.midpoints == {
        x: pytest.approx([0, 0.25, 0.5, 0.75, 1]),
        y: pytest.approx([1, 1.25, 1.5, 1.75, 2]),
    }


def test_product_lines() -> None:
    inst = Line(y, 1, 2, 3) * Line(x, 0, 1, 2)
    assert inst.axes() == [y, x]
    dims = inst.create_dimensions()
    assert len(dims) == 2
    dim = Path(dims).consume()
    assert dim.midpoints == {
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
    inst = Line(y, 1, 2, 3) * Line(x, 0, 1, 5) + Line(z, 2, 3, 5)
    assert inst.axes() == [y, x, z]
    dimy, dimxz = inst.create_dimensions()
    assert dimxz.midpoints == {
        x: pytest.approx([0, 0.25, 0.5, 0.75, 1]),
        z: pytest.approx([2, 2.25, 2.5, 2.75, 3]),
    }
    assert dimy.midpoints == {
        y: pytest.approx([1, 1.5, 2]),
    }


def test_squashed_product() -> None:
    inst = Squash(Line(y, 1, 2, 3) * Line(x, 0, 1, 2))
    assert inst.axes() == [y, x]
    (dim,) = inst.create_dimensions()
    assert dim.midpoints == {
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
    inst = Line(z, 1, 2, 2) * Squash(
        Line(y, 1, 2, 2) * ~Line.bounded(x, 3, 7, 2) * Static.duration(9, 2)
    )
    assert inst.axes() == [z, y, x, DURATION]
    dimz, dimxyt = inst.create_dimensions()
    for d in dimxyt.midpoints, dimxyt.lower, dimxyt.upper:
        assert d == {
            x: pytest.approx([4, 4, 6, 6, 6, 6, 4, 4]),
            y: pytest.approx([1, 1, 1, 1, 2, 2, 2, 2]),
            DURATION: pytest.approx([9, 9, 9, 9, 9, 9, 9, 9]),
        }
    assert dimz.midpoints == dimz.lower == dimz.upper == {z: pytest.approx([1, 2])}


def test_product_snaking_lines() -> None:
    inst = Line(y, 1, 2, 3) * ~Line(x, 0, 1, 2)
    assert inst.axes() == [y, x]
    dims = inst.create_dimensions()
    assert len(dims) == 2
    dim = Path(dims).consume()
    assert dim.midpoints == {
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
    inst = Concat(Line(x, 0, 1, 2), Line(x, 1, 2, 3))
    assert inst.axes() == [x]
    (dim,) = inst.create_dimensions()
    assert dim.midpoints == {x: pytest.approx([0, 1, 1, 1.5, 2])}
    assert dim.lower == {x: pytest.approx([-0.5, 0.5, 0.75, 1.25, 1.75])}
    assert dim.upper == {x: pytest.approx([0.5, 1.5, 1.25, 1.75, 2.25])}


def test_rect_region() -> None:
    inst = Line(y, 1, 3, 5) * Line(x, 0, 2, 3) & Rectangle(x, y, 0, 1, 1.5, 2.2)
    assert inst.axes() == [y, x]
    (dim,) = inst.create_dimensions()
    assert dim.midpoints == {
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
    inst = Static(z, 3.2, 2) * Line(y, 1, 3, 5) * Line(x, 0, 2, 3) & Rectangle(
        x, y, 0, 1, 1.5, 2.2
    )
    assert inst.axes() == [z, y, x]
    zdim, xydim = inst.create_dimensions()
    assert zdim.midpoints == {z: pytest.approx([3.2, 3.2])}
    assert zdim.midpoints is zdim.upper
    assert zdim.midpoints is zdim.lower
    assert xydim.midpoints == {
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
    inst = Line(y, 1, 3, 5) * Line(x, 0, 2, 3) & Rectangle(
        x, y, 0, 1, 1.5, 2.2
    ) | Rectangle(x, y, 0.5, 1.5, 2, 2.5)
    assert inst.axes() == [y, x]
    (dim,) = inst.create_dimensions()
    assert dim.midpoints == {
        x: pytest.approx([0, 1, 0, 1, 2, 0, 1, 2, 1, 2]),
        y: pytest.approx([1, 1, 1.5, 1.5, 1.5, 2, 2, 2, 2.5, 2.5]),
    }


def test_rect_region_intersection() -> None:
    inst = (
        Line(y, 1, 3, 5) * Line(x, 0, 2, 3)
        & Rectangle(x, y, 0, 1, 1.5, 2.2)
        & Rectangle(x, y, 0.5, 1.5, 2, 2.5)
    )
    assert inst.axes() == [y, x]
    (dim,) = inst.create_dimensions()
    assert dim.midpoints == {
        x: pytest.approx([1, 1]),
        y: pytest.approx([1.5, 2]),
    }


def test_rect_region_difference() -> None:
    t = "t"
    # Bracket to
    inst = (
        Line(y, 1, 3, 5) * Line(x, 0, 2, 3) + Static(t, 0.1)
        & Rectangle(x, y, 0, 1, 1.5, 2.2)
    ) - Rectangle(x, y, 0.5, 1.5, 2, 2.5)
    assert inst.axes() == [y, x, t]
    (dim,) = inst.create_dimensions()
    assert dim.midpoints == {
        x: pytest.approx([0, 1, 0, 0]),
        y: pytest.approx([1, 1, 1.5, 2]),
        t: pytest.approx([0.1, 0.1, 0.1, 0.1]),
    }


def test_rect_region_symmetricdifference() -> None:
    inst = Line(y, 1, 3, 5) * Line(x, 0, 2, 3) & Rectangle(
        x, y, 0, 1, 1.5, 2.2
    ) ^ Rectangle(x, y, 0.5, 1.5, 2, 2.5)
    assert inst.axes() == [y, x]
    (dim,) = inst.create_dimensions()
    assert dim.midpoints == {
        x: pytest.approx([0, 1, 0, 2, 0, 2, 1, 2]),
        y: pytest.approx([1, 1, 1.5, 1.5, 2, 2, 2.5, 2.5]),
    }


def test_circle_region() -> None:
    inst = Line(y, 1, 3, 3) * Line(x, 0, 2, 3) & Circle(x, y, 1, 2, 1)
    assert inst.axes() == [y, x]
    (dim,) = inst.create_dimensions()
    assert dim.midpoints == {
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
    inst = Mask(
        Line(y, 1, 3, 3) * ~Line(x, 0, 2, 3),
        Circle(x, y, 1, 2, 1),
        check_path_changes=False,
    )
    assert inst.axes() == [y, x]
    (dim,) = inst.create_dimensions()
    assert dim.midpoints == {
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


def test_ellipse_region() -> None:
    inst = Line("y", 1, 3, 3) * Line("x", 0, 2, 3) & Ellipse(x, y, 1, 2, 2, 1, 45)
    assert inst.axes() == [y, x]
    (dim,) = inst.create_dimensions()
    assert dim.midpoints == {
        x: pytest.approx([0, 1, 0, 1, 2, 1, 2]),
        y: pytest.approx([1, 1, 2, 2, 2, 3, 3]),
    }
    assert dim.lower == {
        x: pytest.approx([-0.5, 0.5, -0.5, 0.5, 1.5, 0.5, 1.5]),
        y: pytest.approx([1, 1, 2, 2, 2, 3, 3]),
    }
    assert dim.upper == {
        x: pytest.approx([0.5, 1.5, 0.5, 1.5, 2.5, 1.5, 2.5]),
        y: pytest.approx([1, 1, 2, 2, 2, 3, 3]),
    }


def test_polygon_region() -> None:
    x_verts = [0, 0.5, 4.0, 2.5]
    y_verts = [0, 3.5, 3.5, 0.5]
    inst = Line("y", 1, 3, 3) * Line("x", 0, 4, 5) & Polygon(x, y, x_verts, y_verts)
    assert inst.axes() == [y, x]
    (dim,) = inst.create_dimensions()
    assert dim.midpoints == {
        x: pytest.approx([1, 2, 1, 2, 3, 1, 2, 3]),
        y: pytest.approx([1, 1, 2, 2, 2, 3, 3, 3]),
    }
    assert dim.lower == {
        x: pytest.approx([0.5, 1.5, 0.5, 1.5, 2.5, 0.5, 1.5, 2.5]),
        y: pytest.approx([1, 1, 2, 2, 2, 3, 3, 3]),
    }
    assert dim.upper == {
        x: pytest.approx([1.5, 2.5, 1.5, 2.5, 3.5, 1.5, 2.5, 3.5]),
        y: pytest.approx([1, 1, 2, 2, 2, 3, 3, 3]),
    }


def test_beam_selector() -> None:
    # Beam selector scan moves bounded between midpoints and lower and upper bounds at
    # maximum speed. Turnaround sections are where it sends the triggers
    bs = str()
    spec = repeat(~Line.bounded(bs, 11, 19, 1), 10)
    dim = spec.path().consume()
    assert len(dim) == 10
    assert dim.lower == {
        bs: pytest.approx([11, 19, 11, 19, 11, 19, 11, 19, 11, 19]),
        "REPEAT": pytest.approx([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    }
    assert dim.upper == {
        bs: pytest.approx([19, 11, 19, 11, 19, 11, 19, 11, 19, 11]),
        "REPEAT": pytest.approx([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    }
    assert dim.midpoints == {
        bs: pytest.approx([15, 15, 15, 15, 15, 15, 15, 15, 15, 15]),
        "REPEAT": pytest.approx([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    }


def test_blended_repeat() -> None:
    # Check that if we blend the REPEATS don't change
    bs = str()
    spec = repeat(~Line.bounded(bs, 11, 19, 1), 10, blend=True)
    dim = spec.path().consume()
    assert len(dim) == 10
    assert dim.lower == {
        bs: pytest.approx([11, 19, 11, 19, 11, 19, 11, 19, 11, 19]),
        "REPEAT": pytest.approx([10, 10, 10, 10, 10, 10, 10, 10, 10, 10]),
    }
    assert dim.upper == {
        bs: pytest.approx([19, 11, 19, 11, 19, 11, 19, 11, 19, 11]),
        "REPEAT": pytest.approx([10, 10, 10, 10, 10, 10, 10, 10, 10, 10]),
    }
    assert dim.midpoints == {
        bs: pytest.approx([15, 15, 15, 15, 15, 15, 15, 15, 15, 15]),
        "REPEAT": pytest.approx([10, 10, 10, 10, 10, 10, 10, 10, 10, 10]),
    }


def test_multiple_statics():
    part_1 = Static("y", 2) * Static("z", 3) * Line("x", 0, 10, 2)
    part_2 = Static("y", 4) * Static("z", 5) * Line("x", 0, 10, 2)
    spec = Concat(part_1, part_2)

    assert list(spec.midpoints()) == [
        {"x": 0.0, "y": 2, "z": 3},
        {"x": 10.0, "y": 2, "z": 3},
        {"x": 0.0, "y": 4, "z": 5},
        {"x": 10.0, "y": 4, "z": 5},
    ]
