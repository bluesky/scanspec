from typing import Any

import pytest

from scanspec.core import Axis, Dimension, Path, SnakedDimension
from scanspec.specs import (
    VARIABLE_DURATION,
    Concat,
    ConstantDuration,
    Ellipse,
    Fly,
    Linspace,
    Polygon,
    Product,
    Range,
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


def test_default_num_linspace() -> None:
    inst = Linspace(x, 0, 1)
    (dim,) = inst.calculate(bounds=True)
    assert dim.midpoints == {x: approx([0])}
    assert dim.lower == {x: approx([-0.5])}
    assert dim.upper == {x: approx([0.5])}
    assert not isinstance(dim, SnakedDimension)
    assert dim.gap == ints("1")
    assert dim.duration is None


def test_one_point_linspace() -> None:
    inst = Linspace(x, 0, 1, 1)
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


def test_two_point_linspace() -> None:
    inst = Linspace(x, 0, 1, 2)
    (dim,) = inst.calculate(bounds=True)
    assert dim.midpoints == {x: approx([0, 1])}
    assert dim.lower == {x: approx([-0.5, 0.5])}
    assert dim.upper == {x: approx([0.5, 1.5])}
    assert dim.gap == ints("10")


def test_two_point_stepped_linspace() -> None:
    inst = 0.1 @ Linspace("x", 0, 1, 2)
    (dim,) = inst.calculate()
    assert dim.midpoints == dim.lower == dim.upper == {x: approx([0, 1])}
    assert dim.gap == ints("11")
    assert dim.duration == approx([0.1, 0.1])


def test_two_point_fly_linspace() -> None:
    inst = Fly(0.1 @ Linspace(x, 0, 1, 2))
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


def test_many_point_linspace() -> None:
    inst = Linspace(x, 0, 1, 5)
    (dim,) = inst.calculate(bounds=True)
    assert dim.midpoints == {x: approx([0, 0.25, 0.5, 0.75, 1])}
    assert dim.lower == {x: approx([-0.125, 0.125, 0.375, 0.625, 0.875])}
    assert dim.upper == {x: approx([0.125, 0.375, 0.625, 0.875, 1.125])}
    assert dim.gap == ints("10000")


def test_zero_step_range() -> None:
    with pytest.raises(ValueError):
        Range(x, 0, 1, 0)


def test_default_step_range() -> None:
    inst = Range(x, 0, 1)
    (dim,) = inst.calculate(bounds=True)
    assert dim.midpoints == {x: approx([0, 1])}
    assert dim.lower == {x: approx([-0.5, 0.5])}
    assert dim.upper == {x: approx([0.5, 1.5])}
    assert not isinstance(dim, SnakedDimension)
    assert dim.gap == ints("10")
    assert dim.duration is None


def test_one_point_range() -> None:
    inst = Range(x, 0, 1, 2)
    (dim,) = inst.calculate(bounds=True)
    assert dim.midpoints == {x: approx([0])}
    assert dim.lower == {x: approx([-1])}
    assert dim.upper == {x: approx([1])}
    assert not isinstance(dim, SnakedDimension)
    assert dim.gap == ints("1")
    assert dim.duration is None


@pytest.mark.parametrize("step", [1, 1 + 1e-8])
def test_two_point_range(step: float) -> None:
    inst = Range(x, 0, 1, step)
    (dim,) = inst.calculate(bounds=True)
    assert dim.midpoints == {x: approx([0, 1])}
    assert dim.lower == {x: approx([-0.5, 0.5])}
    assert dim.upper == {x: approx([0.5, 1.5])}
    assert dim.gap == ints("10")


def test_two_point_fly_range() -> None:
    inst = Fly(0.1 @ Range(x, 0, 1, 1))
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


@pytest.mark.parametrize("step", [0.25, 0.25 + 1e-8])
def test_many_point_range(step: float) -> None:
    inst = Range(x, 0, 1, step)
    (dim,) = inst.calculate(bounds=True)
    assert dim.midpoints == {x: approx([0, 0.25, 0.5, 0.75, 1])}
    assert dim.lower == {x: approx([-0.125, 0.125, 0.375, 0.625, 0.875])}
    assert dim.upper == {x: approx([0.125, 0.375, 0.625, 0.875, 1.125])}
    assert dim.gap == ints("10000")


@pytest.mark.parametrize(
    "range_args,mid,lower,upper,gap",
    [
        ((0, 1, 0.8), 0.4, 0, 0.8, "1"),  # step smaller than range
        ((0, 1, 1), 0.5, 0, 1, "1"),  # step same as range
        ((0, 1, 1.2), 0.5, 0, 1, "1"),  # step larger than range
    ],
)
def test_one_point_bounded_range(
    range_args: list[float], mid: float, lower: float, upper: float, gap: str
) -> None:
    inst = Range.bounded(x, *range_args)
    (dim,) = inst.calculate(bounds=True)
    assert dim.midpoints == {x: approx([mid])}
    assert dim.lower == {x: approx([lower])}
    assert dim.upper == {x: approx([upper])}
    assert not isinstance(dim, SnakedDimension)
    assert dim.gap == ints(gap)
    assert dim.duration is None


@pytest.mark.parametrize("step", [0.25, 0.25 + 1e-8])
def test_many_point_bounded_range(step: float):
    inst = Range.bounded(x, 0, 1, step)
    (dim,) = inst.calculate(bounds=True)
    assert dim.midpoints == {x: approx([0.125, 0.375, 0.625, 0.875])}
    assert dim.lower == {x: approx([0.0, 0.25, 0.5, 0.75])}
    assert dim.upper == {x: approx([0.25, 0.5, 0.75, 1.0])}
    assert dim.gap == ints("1000")


def test_empty_dimension() -> None:
    with pytest.raises(ValueError) as msg:
        Dimension(midpoints={}, upper={}, lower={}, gap=None, duration=None)
    assert "self.gap is undefined" in str(msg.value)


def test_concat() -> None:
    dim1 = Fly(1.0 @ Linspace("x", 0, 1, 2))
    dim2 = Linspace("x", 3, 4, 2)

    with pytest.raises(ValueError) as msg:
        Concat(dim1, dim2)
    assert "Only one of left and right defines a duration" in str(msg.value)

    dim2 = Fly(1.0 @ Linspace("x", 3, 4, 2))

    spec = Concat(dim1, dim2)

    assert spec.frames().duration == approx([1, 1, 1, 1])

    # Check that concat on the Dimension class works as expected
    (dim1,) = Linspace("x", 3, 4, 2).calculate()
    (dim2,) = (1.0 @ Linspace("x", 3, 4, 2)).calculate()

    with pytest.raises(ValueError) as msg:
        dim1.concat(dim2)
    assert "Can't concatenate dimensions unless all or none provide durations" in str(
        msg.value
    )


def test_zip() -> None:
    dim1 = Fly(1.0 @ Linspace("x", 0, 1, 2))
    dim2 = Fly(2.0 @ Linspace("y", 3, 4, 2))

    with pytest.raises(ValueError) as cm:
        Zip(dim1, dim2)
    assert "Both left and right define a duration" in str(cm.value)

    # Forcing the Specs into dimensions and trying to zip them
    (dim1,) = dim1.calculate()
    (dim2,) = dim2.calculate()

    with pytest.raises(ValueError) as cm:
        dim1.zip(dim2)
    assert "Can't have more than one durations array" in str(cm.value)


def test_one_point_bounded_linspace() -> None:
    inst = Linspace.bounded(x, 0, 1, 1)
    assert inst == Linspace(x, 0.5, 1.5, 1)


def test_many_point_bounded_linspace() -> None:
    inst = Linspace.bounded(x, 0, 1, 4)
    assert inst == Linspace(x, 0.125, 0.875, 4)


def test_spiral() -> None:
    inst = Spiral(x, 0, 5, 2, y, 10, 10)
    (dim,) = inst.calculate(bounds=True)
    assert dim.midpoints == {
        y: approx([9.0, 9.2, 11.9, 12.7, 11.0, 8.3, 6.2, 5.8, 7.1, 9.6], abs=0.1),
        x: approx([0.3, -0.9, -0.7, 0.5, 1.5, 1.6, 0.7, -0.6, -1.8, -2.4], abs=0.1),
    }
    assert dim.lower == {
        y: approx([10.0, 8.5, 10.6, 12.7, 12.1, 9.7, 7.1, 5.8, 6.3, 8.3], abs=0.1),
        x: approx([0.0, -0.3, -1.0, -0.1, 1.1, 1.7, 1.3, 0.0, -1.2, -2.2], abs=0.1),
    }
    assert dim.upper == {
        y: approx([8.5, 10.6, 12.7, 12.1, 9.7, 7.1, 5.8, 6.3, 8.3, 11.0], abs=0.1),
        x: approx([-0.3, -1.0, -0.1, 1.1, 1.7, 1.3, 0.0, -1.2, -2.2, -2.4], abs=0.1),
    }
    assert not isinstance(dim, SnakedDimension)
    assert dim.gap == ints("1000000000")


def test_zipped_linspaces() -> None:
    inst = Linspace(x, 0, 1, 5).zip(Linspace(y, 1, 2, 5))
    assert inst.axes() == [x, y]
    (dim,) = inst.calculate(bounds=True)
    assert dim.midpoints == {
        x: approx([0, 0.25, 0.5, 0.75, 1]),
        y: approx([1, 1.25, 1.5, 1.75, 2]),
    }
    assert dim.gap == ints("10000")


def test_product_linspaces() -> None:
    inst = Linspace(y, 1, 2, 3) * Linspace(x, 0, 1, 2)
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


def test_zipped_product_linspaces() -> None:
    inst = Linspace(y, 1, 2, 3) * Linspace(x, 0, 1, 5).zip(Linspace(z, 2, 3, 5))
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
    inst = Squash(Linspace(y, 1, 2, 3) * Linspace(x, 0, 1, 2))
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
    inst = Linspace(z, 1, 2, 2) * Squash(
        9.0 @ Linspace(y, 1, 2, 2) * ~Linspace.bounded(x, 3, 7, 2) * 2
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


def test_product_snaking_linspaces() -> None:
    inst = Linspace(y, 1, 2, 3) * ~Linspace(x, 0, 1, 2)
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
        _ = Fly(1.0 @ Linspace(y, 1, 2, 3)) * Fly(1.0 @ ~Linspace(x, 0, 1, 2))
    assert "Both inner and outer specs defined a duration" in str(msg.value)


def test_concat_linspaces() -> None:
    inst = Concat(Linspace(x, 0, 1, 2), Linspace(x, 1, 2, 3))
    assert inst.axes() == [x]
    (dim,) = inst.calculate(bounds=True)
    assert dim.midpoints == {x: approx([0, 1, 1, 1.5, 2])}
    assert dim.lower == {x: approx([-0.5, 0.5, 0.75, 1.25, 1.75])}
    assert dim.upper == {x: approx([0.5, 1.5, 1.25, 1.75, 2.25])}
    assert dim.gap == ints("10100")

    # Test concating one Spec with duration and another one without
    with pytest.raises(ValueError) as msg:
        Concat((1.0 @ Linspace(x, 0, 1, 2)), Linspace(x, 1, 2, 3))
    assert "Only one of left and right defines a duration" in str(msg.value)

    # Variable duration concat
    spec = Concat(1.0 @ Linspace(x, 0, 1, 2), 2.0 @ Linspace(x, 1, 2, 3))
    assert spec.duration() == VARIABLE_DURATION


def test_xyz_stack() -> None:
    # Beam selector scan moves bounded between midpoints and lower and upper bounds at
    # maximum speed. Turnaround sections are where it sends the triggers
    spec = Linspace(z, 0, 1, 2) * ~Linspace(y, 0, 2, 3) * ~Linspace(x, 0, 3, 4)
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
    spec: Spec[str] = 10 * ~Linspace.bounded(x, 11, 19, 1)
    dim = spec.frames(bounds=True)
    assert len(dim) == 10
    assert dim.lower == {x: approx([11, 19, 11, 19, 11, 19, 11, 19, 11, 19])}
    assert dim.upper == {x: approx([19, 11, 19, 11, 19, 11, 19, 11, 19, 11])}
    assert dim.midpoints == {x: approx([15, 15, 15, 15, 15, 15, 15, 15, 15, 15])}
    assert dim.gap == ints("1111111111")


def test_gap_repeat() -> None:
    # Check that no gap propogates to dim.gap for snaked axis
    spec = Product(10, ~Linspace.bounded(x, 11, 19, 1), gap=False)
    dim = spec.frames(bounds=True)
    assert len(dim) == 10
    assert dim.lower == {x: approx([11, 19, 11, 19, 11, 19, 11, 19, 11, 19])}
    assert dim.upper == {x: approx([19, 11, 19, 11, 19, 11, 19, 11, 19, 11])}
    assert dim.midpoints == {x: approx([15, 15, 15, 15, 15, 15, 15, 15, 15, 15])}
    assert dim.gap == ints("0000000000")


def test_gap_repeat_non_snake() -> None:
    # Check that no gap doesn't propogate to dim.gap for non-snaked axis
    spec = Product(3, Linspace.bounded(x, 11, 19, 1), gap=False)
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
    spec = Product(Linspace(x, 11, 19, 2), 2, gap=gap)
    dim = spec.frames(bounds=True)
    assert len(dim) == 4
    assert dim.lower == dim.midpoints == dim.upper == {x: approx([11, 11, 19, 19])}
    assert dim.gap == ints("1010")


def test_multiple_statics():
    part_1 = Static("y", 2) * Static("z", 3) * Linspace("x", 0, 10, 2)
    part_2 = Static("y", 4) * Static("z", 5) * Linspace("x", 0, 10, 2)
    spec = part_1.concat(part_2)
    assert list(spec.midpoints()) == [
        {"x": 0.0, "y": 2, "z": 3},
        {"x": 10.0, "y": 2, "z": 3},
        {"x": 0.0, "y": 4, "z": 5},
        {"x": 10.0, "y": 4, "z": 5},
    ]
    assert spec.frames(bounds=True).gap == ints("1010")


def test_multiple_statics_with_grid():
    grid = Linspace("y", 0, 10, 2) * Linspace("x", 0, 10, 2)
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
        (Linspace("x", 0.0, 1.0, 1), (1,)),
        (Linspace("x", 0.0, 1.0, 5), (5,)),
        (Spiral("x", 0.0, 1.0, 0.5, "y", 0.0), (4,)),
        (Linspace("x", 0.0, 1.0, 2) * Linspace("y", 0.0, 1.0, 2), (2, 2)),
        (Squash(Linspace("x", 0.0, 1.0, 2) * Linspace("y", 0.0, 1.0, 2)), (4,)),
        (Zip(Linspace("x", 0.0, 1.0, 2), Linspace("y", 0.0, 1.0, 2)), (2,)),
        (Concat(Linspace("x", 0.0, 1.0, 2), Linspace("x", 0.0, 1.0, 2)), (4,)),
        (
            Linspace("x", 0.0, 1.0, 2)
            * Linspace("y", 0.0, 1.0, 2)
            * Linspace("z", 0.0, 2.0, 2),
            (2, 2, 2),
        ),
        (
            Zip(Linspace("x", 0.0, 1.0, 2), Linspace("y", 0.0, 1.0, 2))
            * Linspace("z", 0.0, 2.0, 2),
            (2, 2),
        ),
        (
            Concat(Linspace("x", 0.0, 1.0, 2), Linspace("x", 0.0, 1.0, 2))
            * Linspace("z", 0.0, 2.0, 2),
            (4, 2),
        ),
    ],
)
def test_shape(spec: Spec[Any], expected_shape: tuple[int, ...]):
    assert expected_shape == spec.shape()


def test_constant_duration():
    spec1 = Fly(1.0 @ Linspace("x", 0, 1, 2))
    spec2 = 2.0 @ Linspace("x", 0, 1, 2)

    with pytest.raises(ValueError) as msg:
        2.0 @ spec1  # type: ignore
    assert f"{spec1} already defines a duration" in str(msg.value)

    with pytest.raises(ValueError) as msg:
        spec1.zip(spec2)
    assert "Both left and right define a duration" in str(msg.value)

    with pytest.raises(ValueError) as msg:
        spec1.concat(Linspace("x", 0, 1, 2))
    assert "Only one of left and right defines a duration" in str(msg.value)


def test_int_duration():
    spec1 = 1 @ Linspace("x", 0, 1, 2)
    assert spec1.duration() == 1.0


@pytest.mark.filterwarnings("ignore:fly")
def test_fly():
    spec = fly(Linspace("x", 0, 1, 5), 0.1)
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
    (dim,) = step(Linspace("x", 0, 1, 5), 0.1).calculate()
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
    spec = Fly(1.0 @ Linspace("x", 0, 1, 4)).calculate()
    assert get_constant_duration(spec) == 1

    spec = Concat(1.0 @ Linspace(x, 0, 1, 2), 2.0 @ Linspace(x, 1, 2, 3)).calculate()

    assert get_constant_duration(spec) is None


@pytest.mark.parametrize(
    "inst,exp_mid,exp_lower,exp_upper,exp_gap",
    [
        (
            Ellipse(x, 5, 1, 0.5, y, 0, snake=False, vertical=False),
            {
                y: approx([-0.5, 0.0, 0.0, 0.0, 0.5]),
                x: approx([5.0, 4.5, 5.0, 5.5, 5.0]),
            },
            {
                y: approx([-0.5, 0.0, 0.0, 0.0, 0.5]),
                "x": approx([4.75, 4.25, 4.75, 5.25, 4.75]),
            },
            {
                y: approx([-0.5, 0.0, 0.0, 0.0, 0.5]),
                x: approx([5.25, 4.75, 5.25, 5.75, 5.25]),
            },
            "11001",
        ),
        (
            Ellipse(x, 5, 1, 0.5, y, 0, snake=False, vertical=True),
            {
                x: approx([4.5, 5.0, 5.0, 5.0, 5.5]),
                y: approx([0.0, -0.5, 0.0, 0.5, 0.0]),
            },
            {
                x: approx([4.5, 5.0, 5.0, 5.0, 5.5]),
                y: approx([-0.25, -0.75, -0.25, 0.25, -0.25]),
            },
            {
                x: approx([4.5, 5.0, 5.0, 5.0, 5.5]),
                y: approx([0.25, -0.25, 0.25, 0.75, 0.25]),
            },
            "11001",
        ),
        (
            Ellipse(x, 5, 1, 0.5, y, 0, snake=True, vertical=False),
            {
                y: approx([-0.5, 0.0, 0.0, 0.0, 0.5]),
                x: approx([5.0, 5.5, 5.0, 4.5, 5.0]),
            },
            {
                y: approx([-0.5, 0.0, 0.0, 0.0, 0.5]),
                x: approx([4.75, 5.75, 5.25, 4.75, 4.75]),
            },
            {
                y: approx([-0.5, 0.0, 0.0, 0.0, 0.5]),
                x: approx([5.25, 5.25, 4.75, 4.25, 5.25]),
            },
            "11001",
        ),
        (
            Ellipse(x, 5, 1, 0.5, y, 0, snake=True, vertical=True),
            {
                x: approx([4.5, 5.0, 5.0, 5.0, 5.5]),
                y: approx([0.0, 0.5, 0.0, -0.5, 0.0]),
            },
            {
                x: approx([4.5, 5.0, 5.0, 5.0, 5.5]),
                y: approx([-0.25, 0.75, 0.25, -0.25, -0.25]),
            },
            {
                x: approx([4.5, 5.0, 5.0, 5.0, 5.5]),
                y: approx([0.25, 0.25, -0.25, -0.75, 0.25]),
            },
            "11001",
        ),
    ],
)
def test_ellipse(
    inst: Ellipse[Spec[Axis]],
    exp_mid: dict[str, list[float]],
    exp_lower: dict[str, list[float]],
    exp_upper: dict[str, list[float]],
    exp_gap: str,
):
    (dim,) = inst.calculate(bounds=True)
    assert inst.axes() == [y, x]
    assert dim.midpoints == exp_mid
    assert dim.lower == exp_lower
    assert dim.upper == exp_upper
    assert dim.gap == ints(exp_gap)


@pytest.mark.parametrize(
    "inst,exp_mid,exp_lower,exp_upper,exp_gap",
    [
        (
            Polygon(
                x, y, [(0, 0), (5, 0), (2.5, 4)], 1, 2, snake=False, vertical=False
            ),
            {
                y: approx([0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0]),
                x: approx([0.0, 1.0, 2.0, 3.0, 4.0, 2.0, 3.0]),
            },
            {
                y: approx([0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0]),
                x: approx([-0.5, 0.5, 1.5, 2.5, 3.5, 1.5, 2.5]),
            },
            {
                y: approx([0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0]),
                x: approx([0.5, 1.5, 2.5, 3.5, 4.5, 2.5, 3.5]),
            },
            "1000010",
        ),
        (
            Polygon(
                x,
                y,
                [(0, 0), (1, 0), (1, 1), (0, 2)],
                0.5,
                0.5,
                snake=False,
                vertical=True,
            ),
            {
                x: approx([0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5]),
                y: approx([0.0, 0.5, 1.0, 1.5, 0.0, 0.5, 1.0]),
            },
            {
                x: approx([0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5]),
                y: approx([-0.25, 0.25, 0.75, 1.25, -0.25, 0.25, 0.75]),
            },
            {
                x: approx([0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5]),
                y: approx([0.25, 0.75, 1.25, 1.75, 0.25, 0.75, 1.25]),
            },
            "1000100",
        ),
        (
            Polygon(
                x,
                y,
                [(-1, 0), (1, 0), (1, 1), (0, 2), (-1, 1)],
                0.5,
                0.5,
                snake=True,
                vertical=True,
            ),
            {
                x: approx(
                    [
                        -1.0,
                        -1.0,
                        -1.0,
                        -0.5,
                        -0.5,
                        -0.5,
                        -0.5,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.5,
                        0.5,
                        0.5,
                    ]
                ),
                y: approx(
                    [
                        0.0,
                        0.5,
                        1.0,
                        1.5,
                        1.0,
                        0.5,
                        0.0,
                        0.0,
                        0.5,
                        1.0,
                        1.5,
                        1.0,
                        0.5,
                        0.0,
                    ]
                ),
            },
            {
                x: approx(
                    [
                        -1.0,
                        -1.0,
                        -1.0,
                        -0.5,
                        -0.5,
                        -0.5,
                        -0.5,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.5,
                        0.5,
                        0.5,
                    ]
                ),
                y: approx(
                    [
                        -0.25,
                        0.25,
                        0.75,
                        1.75,
                        1.25,
                        0.75,
                        0.25,
                        -0.25,
                        0.25,
                        0.75,
                        1.25,
                        1.25,
                        0.75,
                        0.25,
                    ]
                ),
            },
            {
                x: approx(
                    [
                        -1.0,
                        -1.0,
                        -1.0,
                        -0.5,
                        -0.5,
                        -0.5,
                        -0.5,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.5,
                        0.5,
                        0.5,
                    ]
                ),
                y: approx(
                    [
                        0.25,
                        0.75,
                        1.25,
                        1.25,
                        0.75,
                        0.25,
                        -0.25,
                        0.25,
                        0.75,
                        1.25,
                        1.75,
                        0.75,
                        0.25,
                        -0.25,
                    ]
                ),
            },
            "10010001000100",
        ),
        (
            Polygon(x, y, [(0, 0), (5, 0), (2.5, 4)], 1, 2, snake=True, vertical=True),
            {
                x: approx([0.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0]),
                y: approx([0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0]),
            },
            {
                x: approx([0.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0]),
                y: approx([-1.0, 1.0, -1.0, 1.0, 3.0, 1.0, -1.0]),
            },
            {
                x: approx([0.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0]),
                y: approx([1.0, -1.0, 1.0, 3.0, 1.0, -1.0, 1.0]),
            },
            "1110101",
        ),
    ],
)
def test_polygon(
    inst: Polygon[Spec[Axis]],
    exp_mid: dict[str, list[float]],
    exp_lower: dict[str, list[float]],
    exp_upper: dict[str, list[float]],
    exp_gap: str,
):
    (dim,) = inst.calculate(bounds=True)
    assert inst.axes() == [y, x]
    assert dim.midpoints == exp_mid
    assert dim.lower == exp_lower
    assert dim.upper == exp_upper
    assert dim.gap == ints(exp_gap)
