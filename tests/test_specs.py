from typing import Any

import pytest

from scanspec.core import Dimension, Path, SnakedDimension
from scanspec.regions import Circle, Ellipse, Polygon, Rectangle
from scanspec.specs import (
    VARIABLE_DURATION,
    Concat,
    ConstantDuration,
    Fly,
    Line,
    Mask,
    Product,
    Spec,
    Spiral,
    Squash,
    Static,
    Zip,
    fly,
    get_constant_duration,
    step,
)

from . import approx

x, y, z = "x", "y", "z"


def ints(s: str) -> Any:
    return approx([int(t) for t in s])


def test_one_point_line() -> None:
    inst = Line(x, 0, 1, 1)
    (dim,) = inst.calculate(bounds=True)
    assert dim.midpoints == {x: approx([0])}
    assert dim.lower == {x: approx([-0.5])}
    assert dim.upper == {x: approx([0.5])}
    assert not isinstance(dim, SnakedDimension)
    assert dim.gap == ints("1")
    assert dim.duration is None


def test_one_point_duration() -> None:
    duration = ConstantDuration[Any](1.0)
    (dim,) = duration.calculate()
    assert dim.duration == approx([1.0])
    assert duration.axes() == []


def test_two_point_line() -> None:
    inst = Line(x, 0, 1, 2)
    (dim,) = inst.calculate(bounds=True)
    assert dim.midpoints == {x: approx([0, 1])}
    assert dim.lower == {x: approx([-0.5, 0.5])}
    assert dim.upper == {x: approx([0.5, 1.5])}
    assert dim.gap == ints("10")


def test_two_point_stepped_line() -> None:
    inst = 0.1 @ Line("x", 0, 1, 2)
    (dim,) = inst.calculate()
    assert dim.midpoints == dim.lower == dim.upper == {x: approx([0, 1])}
    assert dim.gap == ints("11")
    assert dim.duration == approx([0.1, 0.1])


def test_two_point_fly_line() -> None:
    inst = Fly(0.1 @ Line(x, 0, 1, 2))
    (dim,) = inst.calculate()
    assert dim.midpoints == {
        x: approx([0, 1]),
    }
    assert dim.lower == {
        x: approx([-0.5, 0.5]),
    }
    assert dim.upper == {
        x: approx([0.5, 1.5]),
    }
    assert dim.gap == ints("10")
    assert dim.duration == approx([0.1, 0.1])


def test_many_point_line() -> None:
    inst = Line(x, 0, 1, 5)
    (dim,) = inst.calculate(bounds=True)
    assert dim.midpoints == {x: approx([0, 0.25, 0.5, 0.75, 1])}
    assert dim.lower == {x: approx([-0.125, 0.125, 0.375, 0.625, 0.875])}
    assert dim.upper == {x: approx([0.125, 0.375, 0.625, 0.875, 1.125])}
    assert dim.gap == ints("10000")


def test_empty_dimension() -> None:
    with pytest.raises(ValueError) as msg:
        Dimension(midpoints={}, upper={}, lower={}, gap=None, duration=None)
    assert "self.gap is undefined" in str(msg.value)


def test_concat() -> None:
    dim1 = Fly(1.0 @ Line("x", 0, 1, 2))
    dim2 = Line("x", 3, 4, 2)

    with pytest.raises(ValueError) as msg:
        Concat(dim1, dim2)
    assert "Only one of left and right defines a duration" in str(msg.value)

    dim2 = Fly(1.0 @ Line("x", 3, 4, 2))

    spec = Concat(dim1, dim2)

    assert spec.frames().duration == approx([1, 1, 1, 1])

    # Check that concat on the Dimension class works as expected
    (dim1,) = Line("x", 3, 4, 2).calculate()
    (dim2,) = (1.0 @ Line("x", 3, 4, 2)).calculate()

    with pytest.raises(ValueError) as msg:
        dim1.concat(dim2)
    assert "Can't concatenate dimensions unless all or none provide durations" in str(
        msg.value
    )


def test_zip() -> None:
    dim1 = Fly(1.0 @ Line("x", 0, 1, 2))
    dim2 = Fly(2.0 @ Line("y", 3, 4, 2))

    with pytest.raises(ValueError) as cm:
        Zip(dim1, dim2)
    assert "Both left and right define a duration" in str(cm.value)

    # Forcing the Specs into dimensions and trying to zip them
    (dim1,) = dim1.calculate()
    (dim2,) = dim2.calculate()

    with pytest.raises(ValueError) as cm:
        dim1.zip(dim2)
    assert "Can't have more than one durations array" in str(cm.value)


def test_one_point_bounded_line() -> None:
    inst = Line.bounded(x, 0, 1, 1)
    assert inst == Line(x, 0.5, 1.5, 1)


def test_many_point_bounded_line() -> None:
    inst = Line.bounded(x, 0, 1, 4)
    assert inst == Line(x, 0.125, 0.875, 4)


def test_spiral() -> None:
    inst = Spiral(x, y, 0, 10, 5, 50, 10)
    (dim,) = inst.calculate(bounds=True)
    assert dim.midpoints == {
        y: approx([5.4, 6.4, 19.7, 23.8, 15.4, 1.7, -8.6, -10.7, -4.1, 8.3], abs=0.1),
        x: approx([0.3, -0.9, -0.7, 0.5, 1.5, 1.6, 0.7, -0.6, -1.8, -2.4], abs=0.1),
    }
    assert dim.lower == {
        y: approx([10.0, 2.7, 13.3, 23.5, 20.9, 8.7, -4.2, -10.8, -8.4, 1.6], abs=0.1),
        x: approx([0.0, -0.3, -1.0, -0.1, 1.1, 1.7, 1.3, 0.0, -1.2, -2.2], abs=0.1),
    }
    assert dim.upper == {
        y: approx([2.7, 13.3, 23.5, 20.9, 8.7, -4.2, -10.8, -8.4, 1.6, 15.3], abs=0.1),
        x: approx([-0.3, -1.0, -0.1, 1.1, 1.7, 1.3, 0.0, -1.2, -2.2, -2.4], abs=0.1),
    }
    assert not isinstance(dim, SnakedDimension)
    assert dim.gap == ints("1000000000")


def test_spaced_spiral() -> None:
    inst = Spiral.spaced(x, y, 0, 10, 5, 1)
    assert inst == Spiral(x, y, 0, 10, 10, 10, 78)


def test_zipped_lines() -> None:
    inst = Line(x, 0, 1, 5).zip(Line(y, 1, 2, 5))
    assert inst.axes() == [x, y]
    (dim,) = inst.calculate(bounds=True)
    assert dim.midpoints == {
        x: approx([0, 0.25, 0.5, 0.75, 1]),
        y: approx([1, 1.25, 1.5, 1.75, 2]),
    }
    assert dim.gap == ints("10000")


def test_product_lines() -> None:
    inst = Line(y, 1, 2, 3) * Line(x, 0, 1, 2)
    assert inst.axes() == [y, x]
    dims = inst.calculate(bounds=True)
    assert len(dims) == 2
    dim = Path(dims).consume()
    assert dim.midpoints == {
        x: approx([0, 1, 0, 1, 0, 1]),
        y: approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert dim.lower == {
        x: approx([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]),
        y: approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert dim.upper == {
        x: approx([0.5, 1.5, 0.5, 1.5, 0.5, 1.5]),
        y: approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert dim.gap == ints("101010")


def test_zipped_product_lines() -> None:
    inst = Line(y, 1, 2, 3) * Line(x, 0, 1, 5).zip(Line(z, 2, 3, 5))
    assert inst.axes() == [y, x, z]
    dimy, dimxz = inst.calculate(bounds=True)
    assert dimxz.midpoints == {
        x: approx([0, 0.25, 0.5, 0.75, 1]),
        z: approx([2, 2.25, 2.5, 2.75, 3]),
    }
    assert dimy.midpoints == {
        y: approx([1, 1.5, 2]),
    }
    assert inst.frames(bounds=True).gap == ints("100001000010000")


def test_squashed_product() -> None:
    inst = Squash(Line(y, 1, 2, 3) * Line(x, 0, 1, 2))
    assert inst.axes() == [y, x]
    (dim,) = inst.calculate(bounds=True)
    assert dim.midpoints == {
        x: approx([0, 1, 0, 1, 0, 1]),
        y: approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert dim.lower == {
        x: approx([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]),
        y: approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert dim.upper == {
        x: approx([0.5, 1.5, 0.5, 1.5, 0.5, 1.5]),
        y: approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert dim.gap == ints("101010")


def test_squashed_multiplied_snake_scan() -> None:
    inst = Line(z, 1, 2, 2) * Squash(
        9.0 @ Line(y, 1, 2, 2) * ~Line.bounded(x, 3, 7, 2) * 2
    )
    assert inst.axes() == [z, y, x]
    (dimz, dimxyt) = inst.calculate()
    for d in dimxyt.midpoints, dimxyt.lower, dimxyt.upper:
        assert d == {
            x: approx([4, 4, 6, 6, 6, 6, 4, 4]),
            y: approx([1, 1, 1, 1, 2, 2, 2, 2]),
        }
    assert dimxyt.duration == approx([9, 9, 9, 9, 9, 9, 9, 9])
    assert dimz.midpoints == dimz.lower == dimz.upper == {z: approx([1, 2])}
    assert inst.frames(bounds=True).gap == ints("1010101010101010")


def test_product_snaking_lines() -> None:
    inst = Line(y, 1, 2, 3) * ~Line(x, 0, 1, 2)
    assert inst.axes() == [y, x]
    dims = inst.calculate(bounds=True)
    assert len(dims) == 2
    dim = Path(dims).consume()
    assert dim.midpoints == {
        x: approx([0, 1, 1, 0, 0, 1]),
        y: approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert dim.lower == {
        x: approx([-0.5, 0.5, 1.5, 0.5, -0.5, 0.5]),
        y: approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert dim.upper == {
        x: approx([0.5, 1.5, 0.5, -0.5, 0.5, 1.5]),
        y: approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert dim.gap == ints("101010")


def test_product_duration() -> None:
    with pytest.raises(ValueError) as msg:
        _ = Fly(1.0 @ Line(y, 1, 2, 3)) * Fly(1.0 @ ~Line(x, 0, 1, 2))
    assert "Both inner and outer specs defined a duration" in str(msg.value)


def test_concat_lines() -> None:
    inst = Concat(Line(x, 0, 1, 2), Line(x, 1, 2, 3))
    assert inst.axes() == [x]
    (dim,) = inst.calculate(bounds=True)
    assert dim.midpoints == {x: approx([0, 1, 1, 1.5, 2])}
    assert dim.lower == {x: approx([-0.5, 0.5, 0.75, 1.25, 1.75])}
    assert dim.upper == {x: approx([0.5, 1.5, 1.25, 1.75, 2.25])}
    assert dim.gap == ints("10100")

    # Test concating one Spec with duration and another one without
    with pytest.raises(ValueError) as msg:
        Concat((1.0 @ Line(x, 0, 1, 2)), Line(x, 1, 2, 3))
    assert "Only one of left and right defines a duration" in str(msg.value)

    # Variable duration concat
    spec = Concat(1.0 @ Line(x, 0, 1, 2), 2.0 @ Line(x, 1, 2, 3))
    assert spec.duration() == VARIABLE_DURATION


def test_rect_region() -> None:
    inst = Line(y, 1, 3, 5) * Line(x, 0, 2, 3) & Rectangle(x, y, 0, 1, 1.5, 2.2)
    assert inst.axes() == [y, x]
    (dim,) = inst.calculate(bounds=True)
    assert dim.midpoints == {
        x: approx([0, 1, 0, 1, 0, 1]),
        y: approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert dim.lower == {
        x: approx([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]),
        y: approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert dim.upper == {
        x: approx([0.5, 1.5, 0.5, 1.5, 0.5, 1.5]),
        y: approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert dim.gap == ints("101010")


def test_rect_region_3D() -> None:
    inst = Static(z, 3.2, 2) * Line(y, 1, 3, 5) * Line(x, 0, 2, 3) & Rectangle(
        x, y, 0, 1, 1.5, 2.2
    )
    assert inst.axes() == [z, y, x]
    zdim, xydim = inst.calculate(bounds=True)
    assert zdim.midpoints == {z: approx([3.2, 3.2])}
    assert zdim.midpoints is zdim.upper
    assert zdim.midpoints is zdim.lower
    assert xydim.midpoints == {
        x: approx([0, 1, 0, 1, 0, 1]),
        y: approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert xydim.lower == {
        x: approx([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]),
        y: approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert xydim.upper == {
        x: approx([0.5, 1.5, 0.5, 1.5, 0.5, 1.5]),
        y: approx([1, 1, 1.5, 1.5, 2, 2]),
    }
    assert inst.frames(bounds=True).gap == ints("101010101010")


def test_rect_region_union() -> None:
    inst = Line(y, 1, 3, 5) * Line(x, 0, 2, 3) & Rectangle(
        x, y, 0, 1, 1.5, 2.2
    ) | Rectangle(x, y, 0.5, 1.5, 2, 2.5)
    assert inst.axes() == [y, x]
    (dim,) = inst.calculate(bounds=True)
    assert dim.midpoints == {
        x: approx([0, 1, 0, 1, 2, 0, 1, 2, 1, 2]),
        y: approx([1, 1, 1.5, 1.5, 1.5, 2, 2, 2, 2.5, 2.5]),
    }
    assert dim.gap == ints("1010010010")


def test_rect_region_intersection() -> None:
    inst = (
        Line(y, 1, 3, 5) * Line(x, 0, 2, 3)
        & Rectangle(x, y, 0, 1, 1.5, 2.2)
        & Rectangle(x, y, 0.5, 1.5, 2, 2.5)
    )
    assert inst.axes() == [y, x]
    (dim,) = inst.calculate()
    assert dim.midpoints == {
        x: approx([1, 1]),
        y: approx([1.5, 2]),
    }
    assert dim.gap == ints("11")


def test_rect_region_difference() -> None:
    # Bracket to force testing Mask.__sub__ rather than Region.__sub__
    spec = (
        Line(y, 1, 3, 5) * Line(x, 0, 2, 3) & Rectangle(x, y, 0, 1, 1.5, 2.2)
    ) - Rectangle(x, y, 0.5, 1.5, 2, 2.5)

    inst = Fly(0.1 @ spec)
    assert inst.axes() == [y, x]
    (dim,) = inst.calculate()
    assert dim.midpoints == {
        x: approx([0, 1, 0, 0]),
        y: approx([1, 1, 1.5, 2]),
    }
    assert dim.duration == approx([0.1, 0.1, 0.1, 0.1])
    assert dim.gap == ints("1011")


def test_rect_region_symmetricdifference() -> None:
    inst = Line(y, 1, 3, 5) * Line(x, 0, 2, 3) & Rectangle(
        x, y, 0, 1, 1.5, 2.2
    ) ^ Rectangle(x, y, 0.5, 1.5, 2, 2.5)
    assert inst.axes() == [y, x]
    (dim,) = inst.calculate(bounds=True)
    assert dim.midpoints == {
        x: approx([0, 1, 0, 2, 0, 2, 1, 2]),
        y: approx([1, 1, 1.5, 1.5, 2, 2, 2.5, 2.5]),
    }
    assert dim.gap == ints("10111110")


def test_circle_region() -> None:
    inst = Line(y, 1, 3, 3) * Line(x, 0, 2, 3) & Circle(x, y, 1, 2, 1)
    assert inst.axes() == [y, x]
    (dim,) = inst.calculate(bounds=True)
    assert dim.midpoints == {
        x: approx([1, 0, 1, 2, 1]),
        y: approx([1, 2, 2, 2, 3]),
    }
    assert dim.lower == {
        x: approx([0.5, -0.5, 0.5, 1.5, 0.5]),
        y: approx([1, 2, 2, 2, 3]),
    }
    assert dim.upper == {
        x: approx([1.5, 0.5, 1.5, 2.5, 1.5]),
        y: approx([1, 2, 2, 2, 3]),
    }
    assert dim.gap == ints("11001")


def test_circle_snaked_region() -> None:
    inst = Mask(
        Line(y, 1, 3, 3) * ~Line(x, 0, 2, 3),
        Circle(x, y, 1, 2, 1),
        check_path_changes=False,
    )
    assert inst.axes() == [y, x]
    (dim,) = inst.calculate(bounds=True)
    assert dim.midpoints == {
        x: approx([1, 2, 1, 0, 1]),
        y: approx([1, 2, 2, 2, 3]),
    }
    assert dim.lower == {
        x: approx([0.5, 2.5, 1.5, 0.5, 0.5]),
        y: approx([1, 2, 2, 2, 3]),
    }
    assert dim.upper == {
        x: approx([1.5, 1.5, 0.5, -0.5, 1.5]),
        y: approx([1, 2, 2, 2, 3]),
    }
    assert dim.gap == ints("11001")


def test_ellipse_region() -> None:
    inst = Line("y", 1, 3, 3) * Line("x", 0, 2, 3) & Ellipse(x, y, 1, 2, 2, 1, 45)
    assert inst.axes() == [y, x]
    (dim,) = inst.calculate(bounds=True)
    assert dim.midpoints == {
        x: approx([0, 1, 0, 1, 2, 1, 2]),
        y: approx([1, 1, 2, 2, 2, 3, 3]),
    }
    assert dim.lower == {
        x: approx([-0.5, 0.5, -0.5, 0.5, 1.5, 0.5, 1.5]),
        y: approx([1, 1, 2, 2, 2, 3, 3]),
    }
    assert dim.upper == {
        x: approx([0.5, 1.5, 0.5, 1.5, 2.5, 1.5, 2.5]),
        y: approx([1, 1, 2, 2, 2, 3, 3]),
    }
    assert dim.gap == ints("1010010")


def test_polygon_region() -> None:
    x_verts = [0, 0.5, 4.0, 2.5]
    y_verts = [0, 3.5, 3.5, 0.5]
    inst = Line("y", 1, 3, 3) * Line("x", 0, 4, 5) & Polygon(x, y, x_verts, y_verts)
    assert inst.axes() == [y, x]
    (dim,) = inst.calculate(bounds=True)
    assert dim.midpoints == {
        x: approx([1, 2, 1, 2, 3, 1, 2, 3]),
        y: approx([1, 1, 2, 2, 2, 3, 3, 3]),
    }
    assert dim.lower == {
        x: approx([0.5, 1.5, 0.5, 1.5, 2.5, 0.5, 1.5, 2.5]),
        y: approx([1, 1, 2, 2, 2, 3, 3, 3]),
    }
    assert dim.upper == {
        x: approx([1.5, 2.5, 1.5, 2.5, 3.5, 1.5, 2.5, 3.5]),
        y: approx([1, 1, 2, 2, 2, 3, 3, 3]),
    }
    assert dim.gap == ints("10100100")


def test_xyz_stack() -> None:
    # Beam selector scan moves bounded between midpoints and lower and upper bounds at
    # maximum speed. Turnaround sections are where it sends the triggers
    spec = Line(z, 0, 1, 2) * ~Line(y, 0, 2, 3) * ~Line(x, 0, 3, 4)
    dim = spec.frames(bounds=True)
    assert len(dim) == 24
    assert dim.lower == {
        z: ints("000000000000111111111111"),
        y: ints("000011112222222211110000"),
        x: approx([-0.5, 0.5, 1.5, 2.5, 3.5, 2.5, 1.5, 0.5] * 3),
    }
    assert dim.upper == {
        z: ints("000000000000111111111111"),
        y: ints("000011112222222211110000"),
        x: approx([0.5, 1.5, 2.5, 3.5, 2.5, 1.5, 0.5, -0.5] * 3),
    }
    assert dim.midpoints == {
        z: ints("000000000000111111111111"),
        y: ints("000011112222222211110000"),
        x: ints("012332100123321001233210"),
    }
    assert dim.gap == ints("100010001000100010001000")
    # Check that it still works if you consume then start on a point that should
    # be False
    p = Path(spec.calculate(bounds=True))
    assert p.consume(4).gap == ints("1000")
    assert p.consume(4).gap == ints("1000")
    assert p.consume(5).gap == ints("10001")
    assert p.consume(2).gap == ints("00")
    assert p.consume().gap == ints("010001000")
    assert p.consume().gap == ints("")


def test_beam_selector() -> None:
    # Beam selector scan moves bounded between midpoints and lower and upper bounds at
    # maximum speed. Turnaround sections are where it sends the triggers
    spec: Spec[str] = 10 * ~Line.bounded(x, 11, 19, 1)
    dim = spec.frames(bounds=True)
    assert len(dim) == 10
    assert dim.lower == {x: approx([11, 19, 11, 19, 11, 19, 11, 19, 11, 19])}
    assert dim.upper == {x: approx([19, 11, 19, 11, 19, 11, 19, 11, 19, 11])}
    assert dim.midpoints == {x: approx([15, 15, 15, 15, 15, 15, 15, 15, 15, 15])}
    assert dim.gap == ints("1111111111")


def test_gap_repeat() -> None:
    # Check that no gap propogates to dim.gap for snaked axis
    spec = Product(10, ~Line.bounded(x, 11, 19, 1), gap=False)
    dim = spec.frames(bounds=True)
    assert len(dim) == 10
    assert dim.lower == {x: approx([11, 19, 11, 19, 11, 19, 11, 19, 11, 19])}
    assert dim.upper == {x: approx([19, 11, 19, 11, 19, 11, 19, 11, 19, 11])}
    assert dim.midpoints == {x: approx([15, 15, 15, 15, 15, 15, 15, 15, 15, 15])}
    assert dim.gap == ints("0000000000")


def test_gap_repeat_non_snake() -> None:
    # Check that no gap doesn't propogate to dim.gap for non-snaked axis
    spec = Product(3, Line.bounded(x, 11, 19, 1), gap=False)
    dim = spec.frames(bounds=True)
    assert len(dim) == 3
    assert dim.lower == {x: approx([11, 11, 11])}
    assert dim.upper == {x: approx([19, 19, 19])}
    assert dim.midpoints == {x: approx([15, 15, 15])}
    assert dim.gap == ints("111")


@pytest.mark.parametrize("gap", [True, False])
def test_gap_repeat_right_hand_side(gap: bool) -> None:
    # Check that 2 repeats of each frame means a gap on each change in x, no
    # matter what the setting of the gap argument
    spec = Product(Line(x, 11, 19, 2), 2, gap=gap)
    dim = spec.frames(bounds=True)
    assert len(dim) == 4
    assert dim.lower == dim.midpoints == dim.upper == {x: approx([11, 11, 19, 19])}
    assert dim.gap == ints("1010")


def test_multiple_statics():
    part_1 = Static("y", 2) * Static("z", 3) * Line("x", 0, 10, 2)
    part_2 = Static("y", 4) * Static("z", 5) * Line("x", 0, 10, 2)
    spec = part_1.concat(part_2)
    assert list(spec.midpoints()) == [
        {"x": 0.0, "y": 2, "z": 3},
        {"x": 10.0, "y": 2, "z": 3},
        {"x": 0.0, "y": 4, "z": 5},
        {"x": 10.0, "y": 4, "z": 5},
    ]
    assert spec.frames(bounds=True).gap == ints("1010")


def test_multiple_statics_with_grid():
    grid = Line("y", 0, 10, 2) * Line("x", 0, 10, 2)
    part_1 = grid.zip(Static("a", 2)).zip(Static("b", 3))
    part_2 = grid.zip(Static("a", 4)).zip(Static("b", 5))
    spec = part_1.concat(part_2)
    assert list(spec.midpoints()) == [
        {"x": 0.0, "y": 0.0, "a": 2, "b": 3},
        {"x": 10.0, "y": 0.0, "a": 2, "b": 3},
        {"x": 0.0, "y": 10.0, "a": 2, "b": 3},
        {"x": 10.0, "y": 10.0, "a": 2, "b": 3},
        {"x": 0.0, "y": 0.0, "a": 4, "b": 5},
        {"x": 10.0, "y": 0.0, "a": 4, "b": 5},
        {"x": 0.0, "y": 10.0, "a": 4, "b": 5},
        {"x": 10.0, "y": 10.0, "a": 4, "b": 5},
    ]
    assert spec.frames(bounds=True).gap == ints("10101010")


@pytest.mark.parametrize(
    "spec,expected_shape",
    [
        (Line("x", 0.0, 1.0, 1), (1,)),
        (Line("x", 0.0, 1.0, 5), (5,)),
        (Spiral("x", "y", 0.0, 0.0, 1.0, 1.0, 5, 0.0), (5,)),
        (Line("x", 0.0, 1.0, 2) * Line("y", 0.0, 1.0, 2), (2, 2)),
        (Squash(Line("x", 0.0, 1.0, 2) * Line("y", 0.0, 1.0, 2)), (4,)),
        (Zip(Line("x", 0.0, 1.0, 2), Line("y", 0.0, 1.0, 2)), (2,)),
        (Concat(Line("x", 0.0, 1.0, 2), Line("x", 0.0, 1.0, 2)), (4,)),
        (
            Line("x", 0.0, 1.0, 2) * Line("y", 0.0, 1.0, 2) * Line("z", 0.0, 2.0, 2),
            (2, 2, 2),
        ),
        (
            Zip(Line("x", 0.0, 1.0, 2), Line("y", 0.0, 1.0, 2))
            * Line("z", 0.0, 2.0, 2),
            (2, 2),
        ),
        (
            Concat(Line("x", 0.0, 1.0, 2), Line("x", 0.0, 1.0, 2))
            * Line("z", 0.0, 2.0, 2),
            (4, 2),
        ),
    ],
)
def test_shape(spec: Spec[Any], expected_shape: tuple[int, ...]):
    assert expected_shape == spec.shape()


def test_constant_duration():
    spec1 = Fly(1.0 @ Line("x", 0, 1, 2))
    spec2 = 2.0 @ Line("x", 0, 1, 2)

    with pytest.raises(ValueError) as msg:
        2.0 @ spec1  # type: ignore
    assert f"{spec1} already defines a duration" in str(msg.value)

    with pytest.raises(ValueError) as msg:
        spec1.zip(spec2)
    assert "Both left and right define a duration" in str(msg.value)

    with pytest.raises(ValueError) as msg:
        spec1.concat(Line("x", 0, 1, 2))
    assert "Only one of left and right defines a duration" in str(msg.value)


def test_int_duration():
    spec1 = 1 @ Line("x", 0, 1, 2)
    assert spec1.duration() == 1.0


@pytest.mark.filterwarnings("ignore:fly")
def test_fly():
    spec = fly(Line("x", 0, 1, 5), 0.1)
    (dim,) = spec.calculate()
    assert dim.midpoints == {x: approx([0, 0.25, 0.5, 0.75, 1])}
    assert dim.upper == {x: approx([0.125, 0.375, 0.625, 0.875, 1.125])}
    assert dim.lower == {x: approx([-0.125, 0.125, 0.375, 0.625, 0.875])}
    assert dim.duration == approx(
        [
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
        ]
    )


@pytest.mark.filterwarnings("ignore:step")
def test_step():
    (dim,) = step(Line("x", 0, 1, 5), 0.1).calculate()
    assert (
        dim.midpoints == dim.lower == dim.upper == {x: approx([0, 0.25, 0.5, 0.75, 1])}
    )
    assert dim.duration == approx(
        [
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
        ]
    )


@pytest.mark.filterwarnings("ignore:get_constant_duration")
def test_get_constant_duration():
    spec = Fly(1.0 @ Line("x", 0, 1, 4)).calculate()
    assert get_constant_duration(spec) == 1

    spec = Concat(1.0 @ Line(x, 0, 1, 2), 2.0 @ Line(x, 1, 2, 3)).calculate()

    assert get_constant_duration(spec) is None
