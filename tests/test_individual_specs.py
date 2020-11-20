from typing import Dict
import pytest

from scanspec.specs import Line


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


def _get_spiral_data(start_x: float, start_y: float) -> Dict[str, float]:
    return [{'motor2': start_y + 0.000, 'motor1': start_x + 0.100},
            {'motor2': start_y + 0.000, 'motor1': start_x + 0.200},
            {'motor2': start_y + 0.000, 'motor1': start_x - 0.200},
            {'motor2': start_y + 0.000, 'motor1': start_x + 0.300},
            {'motor2': start_y + 0.260, 'motor1': start_x - 0.150},
            {'motor2': start_y - 0.260, 'motor1': start_x - 0.150},
            {'motor2': start_y + 0.000, 'motor1': start_x + 0.400},
            {'motor2': start_y + 0.400, 'motor1': start_x + 0.000},
            {'motor2': start_y + 0.000, 'motor1': start_x - 0.400},
            {'motor2': start_y - 0.400, 'motor1': start_x - 0.000},
            {'motor2': start_y + 0.000, 'motor1': start_x + 0.500},
            {'motor2': start_y + 0.476, 'motor1': start_x + 0.155},
            {'motor2': start_y + 0.294, 'motor1': start_x - 0.405},
            {'motor2': start_y - 0.294, 'motor1': start_x - 0.405},
            {'motor2': start_y - 0.476, 'motor1': start_x + 0.155},
            ]


def test_spiral() -> None:
    scan = bp.spiral([det], motor1, motor2, 0.0, 0.0, 1.0, 1.0, 0.1, 1.0,
                     tilt=0.0)