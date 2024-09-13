from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from scanspec.regions import Circle, Rectangle, Region, UnionOf
from scanspec.specs import Line, Mask, Spec, Spiral


def test_line_serializes() -> None:
    ob = Line("x", 0, 1, 4)
    serialized = {"type": "Line", "axis": "x", "start": 0.0, "stop": 1.0, "num": 4}
    assert ob.serialize() == serialized
    assert Spec.deserialize(serialized) == ob


def test_circle_serializes() -> None:
    ob = Circle("x", "y", x_middle=0, y_middle=1, radius=4)
    serialized = {
        "x_axis": "x",
        "y_axis": "y",
        "x_middle": 0.0,
        "y_middle": 1.0,
        "radius": 4.0,
        "type": "Circle",
    }
    assert ob.serialize() == serialized
    assert Region.deserialize(serialized) == ob


def test_masked_circle_serializes() -> None:
    ob = Mask(Line("x", 0, 1, 4), Circle("x", "y", x_middle=0, y_middle=1, radius=4))
    serialized = {
        "type": "Mask",
        "spec": {"type": "Line", "axis": "x", "start": 0, "stop": 1, "num": 4},
        "region": {
            "x_axis": "x",
            "y_axis": "y",
            "x_middle": 0,
            "y_middle": 1,
            "radius": 4,
            "type": "Circle",
        },
        "check_path_changes": True,
    }
    assert ob.serialize() == serialized
    assert Spec.deserialize(serialized) == ob


def test_product_lines_serializes() -> None:
    ob = Line("z", 4, 5, 6) * Line("y", 2, 3, 5) * Line("x", 0, 1, 4)
    serialized = {
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
            "inner": {"type": "Line", "axis": "y", "start": 2.0, "stop": 3.0, "num": 5},
        },
        "inner": {"type": "Line", "axis": "x", "start": 0.0, "stop": 1.0, "num": 4},
    }

    assert ob.serialize() == serialized
    assert Spec.deserialize(serialized) == ob


def test_complex_nested_serializes() -> None:
    ob = Mask(
        Spiral.spaced("x", "y", 0, 0, 10, 3),
        UnionOf(
            Circle("x", "y", x_middle=0, y_middle=1, radius=4),
            Rectangle("x", "y", 0, 1.1, 1.5, 2.1, 30),
        ),
    )
    serialized = {
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
    }
    assert ob.serialize() == serialized
    assert Spec.deserialize(serialized) == ob


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


def test_vanilla_serialization():
    ob = Mask(
        Spiral.spaced("x", "y", 0, 0, 10, 3),
        UnionOf(
            Circle("x", "y", x_middle=0, y_middle=1, radius=4),
            Rectangle("x", "y", 0, 1.1, 1.5, 2.1, 30),
        ),
    )

    adapter = TypeAdapter(Spec[str])
    serialized = adapter.dump_json(ob)
    deserialized = adapter.validate_json(serialized)
    assert deserialized == ob
