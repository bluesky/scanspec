from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from scanspec.specs import Ellipse, Linspace, Spec, Spiral


def test_line_serializes() -> None:
    ob = Linspace("x", 0, 1, 4)
    serialized = {"type": "Linspace", "axis": "x", "start": 0.0, "stop": 1.0, "num": 4}
    assert ob.serialize() == serialized
    assert Spec.deserialize(serialized) == ob


def test_ellipse_serializes() -> None:
    ob = Ellipse(
        x_axis="x",
        x_centre=0,
        x_diameter=1,
        x_step=0.1,
        y_axis="y",
        y_centre=2,
        y_diameter=3,
        y_step=0.5,
        snake=True,
    )
    serialized = {
        "type": "Ellipse",
        "x_axis": "x",
        "x_centre": 0.0,
        "x_diameter": 1.0,
        "x_step": 0.1,
        "y_axis": "y",
        "y_centre": 2.0,
        "y_diameter": 3.0,
        "y_step": 0.5,
        "snake": True,
        "vertical": False,
    }
    assert ob.serialize() == serialized
    assert Ellipse.deserialize(serialized) == ob


def test_spiral_serializes():
    ob = Spiral("x", 0, 1, 0.1, "y", 2, 3)

    serialized = {
        "x_axis": "x",
        "x_centre": 0.0,
        "x_diameter": 1.0,
        "x_step": 0.1,
        "y_axis": "y",
        "y_centre": 2.0,
        "y_diameter": 3.0,
        "type": "Spiral",
    }
    assert ob.serialize() == serialized
    assert Spec.deserialize(serialized) == ob


def test_product_lines_serializes() -> None:
    ob = Linspace("z", 4, 5, 6) * Linspace("y", 2, 3, 5) * Linspace("x", 0, 1, 4)
    serialized = {
        "type": "Product",
        "gap": True,
        "outer": {
            "type": "Product",
            "gap": True,
            "outer": {
                "type": "Linspace",
                "axis": "z",
                "start": 4.0,
                "stop": 5.0,
                "num": 6,
            },
            "inner": {
                "type": "Linspace",
                "axis": "y",
                "start": 2.0,
                "stop": 3.0,
                "num": 5,
            },
        },
        "inner": {"type": "Linspace", "axis": "x", "start": 0.0, "stop": 1.0, "num": 4},
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
            "type": "Linspace",
        },
        {
            "axis": "x",
            "start": 0.0,
            "type": "Linspace",
        },
        {
            "axis": "x",
            "start": 0.0,
            "stop": {},
            "num": 4,
            "type": "Linspace",
        },
        {
            "axis": "x",
            "start": 0.0,
            "stop": None,
            "num": 4,
            "type": "Linspace",
        },
        {
            "type": "Product",
            "outer": None,
            "inner": {
                "type": "Linspace",
                "axis": "x",
                "start": 0.0,
                "stop": 1.0,
                "num": 4,
            },
        },
    ],
    ids=["extra arg", "missing arg", "wrong type", "null value", "null spec"],
)
def test_detects_invalid_serialized(serialized: Mapping[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Spec.deserialize(serialized)


def test_vanilla_serialization():
    ob = Spiral("x", 0, 1, 0.1, "y", 2, 3)

    adapter = TypeAdapter(Spec[str])
    serialized = adapter.dump_json(ob)
    deserialized = adapter.validate_json(serialized)
    assert deserialized == ob
