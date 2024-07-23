from enum import Enum, auto
from typing import Any, Mapping

import pytest
from pydantic import ValidationError

from scanspec.regions import Circle, Rectangle, Region, UnionOf, circle, union_of
from scanspec.specs import Line, Mask, Spec, Spiral, line, mask, spiral


class _SerOrDe(Enum):
    SER = auto()
    DE = auto()


@pytest.fixture(
    params=[_SerOrDe.SER, _SerOrDe.DE], ids=["serialization", "deserialization"]
)
def assert_serialized_deserialized(request: pytest.FixtureRequest):
    def _assert_serialized_deserialized(expected_serialized: dict[str, Any], obj: Spec):
        if request.param == _SerOrDe.SER:
            serialized = obj.serialize()
            assert serialized == expected_serialized
        else:
            assert Spec.deserialize(expected_serialized) == obj

    return _assert_serialized_deserialized


def test_line_serializes(assert_serialized_deserialized) -> None:
    assert_serialized_deserialized(
        {"type": "Line", "axis": "x", "start": 0.0, "stop": 1.0, "num": 4},
        line("x", 0, 1, 4),
    )


def test_spiral_serializes(assert_serialized_deserialized) -> None:
    assert_serialized_deserialized(
        {
            "x_axis": "x",
            "y_axis": "y",
            "x_start": 1.0,
            "y_start": 1.0,
            "x_range": 2.0,
            "y_range": 2.0,
            "num": 50,
            "rotate": 0.0,
            "type": "Spiral",
        },
        spiral("x", "y", 1, 1, 2, 2, 50),
    )


def test_masked_circle_serializes(assert_serialized_deserialized) -> None:
    assert_serialized_deserialized(
        {
            "type": "Mask",
            "spec": {"type": "Line", "axis": "x", "start": 0, "stop": 1, "num": 4},
            "region": {
                "x_axis": "x",
                "y_axis": "y",
                "x_middle": 0.0,
                "y_middle": 1.0,
                "radius": 4.0,
                "type": "Circle",
            },
            "check_path_changes": True,
        },
        mask(line("x", 0, 1, 4), circle("x", "y", x_middle=0, y_middle=1, radius=4)),
    )


def test_product_lines_serializes(assert_serialized_deserialized) -> None:
    assert_serialized_deserialized(
        {
            "type": "Product",
            "outer": {
                "type": "Product",
                "outer": {
                    "type": "Line",
                    "axis": "z",
                    "start": 4.0,
                    "stop": 5.0,
                    "num": 6,
                },
                "inner": {
                    "type": "Line",
                    "axis": "y",
                    "start": 2.0,
                    "stop": 3.0,
                    "num": 5,
                },
            },
            "inner": {"type": "Line", "axis": "x", "start": 0.0, "stop": 1.0, "num": 4},
        },
        line("z", 4, 5, 6) * line("y", 2, 3, 5) * line("x", 0, 1, 4),
    )


def test_complex_nested_serializes(assert_serialized_deserialized) -> None:
    assert_serialized_deserialized(
        {
            "spec": {
                "x_axis": "x",
                "y_axis": "y",
                "x_start": 0.0,
                "y_start": 0.0,
                "x_range": 20.0,
                "y_range": 20.0,
                "num": 34,
                "rotate": 0.0,
                "type": "Spiral",
            },
            "region": {
                "left": {
                    "x_axis": "x",
                    "y_axis": "y",
                    "x_middle": 0.0,
                    "y_middle": 1.0,
                    "radius": 4.0,
                    "type": "Circle",
                },
                "right": {
                    "x_axis": "x",
                    "y_axis": "y",
                    "x_min": 0.0,
                    "y_min": 1.1,
                    "x_max": 1.5,
                    "y_max": 2.1,
                    "angle": 30.0,
                    "type": "Rectangle",
                },
                "type": "UnionOf",
            },
            "check_path_changes": True,
            "type": "Mask",
        },
        mask(
            Spiral.spaced("x", "y", 0, 0, 10, 3),
            union_of(
                circle("x", "y", x_middle=0, y_middle=1, radius=4),
                Rectangle("x", "y", 0, 1.1, 1.5, 2.1, 30),
            ),
        ),
    )


@pytest.mark.parametrize(
    "serialized",
    [
        {
            "axis": "x",
            "start": 0.0,
            "stop": 1.0,
            "num": 4,
            "foo": "bar",
            "type": "Line",
        },
        {
            "axis": "x",
            "start": 0.0,
            "stop": 1.0,
            "type": "Line",
        },
        {
            "axis": "x",
            "start": 0.0,
            "stop": {},
            "num": 4,
            "type": "Line",
        },
        {
            "axis": "x",
            "start": 0.0,
            "stop": None,
            "num": 4,
            "type": "Line",
        },
        {
            "type": "Product",
            "outer": None,
            "inner": {"type": "Line", "axis": "x", "start": 0.0, "stop": 1.0, "num": 4},
        },
    ],
    ids=["extra arg", "missing arg", "wrong type", "null value", "null spec"],
)
def test_detects_invalid_serialized(serialized: Mapping[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Spec.deserialize(serialized)
