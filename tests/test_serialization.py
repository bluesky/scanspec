from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
from ophyd_async.core import Device
from pydantic import TypeAdapter, ValidationError

from scanspec.specs import Array, Ellipse, Fly, Linspace, Spec, Spiral


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


def test_spec_with_ophyd_async_device_axis_serializes():
    motor = Device(name="motor")
    spec = Fly(Linspace(motor, 1, 2, 5))
    serialized = spec.serialize()
    assert serialized["type"] == "Fly"
    assert serialized["spec"]["type"] == "Linspace"
    assert serialized["spec"]["axis"] == motor.name
    assert serialized["spec"]["start"] == 1.0
    assert serialized["spec"]["stop"] == 2.0
    assert serialized["spec"]["num"] == 5


def test_spec_with_nameless_axis_falls_back_to_repr():
    class NamelessAxis:
        pass

    axis = NamelessAxis()
    spec = Fly(Linspace(axis, 1, 2, 5))
    serialized = spec.serialize()
    assert serialized["spec"]["axis"] == repr(axis)


def test_spec_with_ophyd_async_device_axis_is_json_serializable():
    import json

    motor = Device(name="motor")
    spec = Fly(Linspace(motor, 1, 2, 5))
    serialized = spec.serialize()
    assert serialized["spec"]["axis"] == motor.name
    # Should not raise
    json.dumps(serialized)


def test_array_serializes() -> None:
    ob = Array(
        "x",
        array=np.array([0.0, 1.0, 3.0]),
        bounds=np.array([-0.5, 0.5, 2.0, 4.0]),
        durations=np.array([0.1, 0.2, 0.3]),
    )
    serialized = {
        "type": "Array",
        "axis": "x",
        "array": [0.0, 1.0, 3.0],
        "bounds": [-0.5, 0.5, 2.0, 4.0],
        "durations": [0.1, 0.2, 0.3],
    }
    assert ob.serialize() == serialized
    roundtrip = Spec.deserialize(serialized)
    assert isinstance(roundtrip, Array)
    assert roundtrip.array is not None
    np.testing.assert_array_equal(roundtrip.array, ob.array)
    np.testing.assert_array_equal(roundtrip.bounds, ob.bounds)
    np.testing.assert_array_equal(roundtrip.durations, ob.durations)


def test_array_accepts_lists_and_ints() -> None:
    ob = Array("x", array=[0, 1, 3])  # type: ignore
    assert ob.array is not None
    assert ob.array.dtype == np.float64
    assert ob.serialize()["array"] == [0.0, 1.0, 3.0]
    assert ob.bounds is None


def test_array_equality() -> None:
    def array(
        axis: str = "x",
        array: list[float] | None = None,
        bounds: list[float] | None = None,
        durations: list[float] | None = None,
    ) -> Array[str]:
        return Array(
            axis,
            None if array is None else np.array(array),
            None if bounds is None else np.array(bounds),
            None if durations is None else np.array(durations),
        )

    mid, bounds, durations = [0.0, 1.0, 3.0], [-0.5, 0.5, 2.0, 4.0], [1.0, 1.0, 1.0]
    a = array(array=mid, bounds=bounds, durations=durations)
    assert a == array(array=mid, bounds=bounds, durations=durations)
    assert a != array(array=[0.0, 1.0, 4.0], bounds=bounds, durations=durations)
    assert a != array(array=mid, durations=durations)
    assert a != array(array=mid, bounds=bounds, durations=[1.0, 1.0, 2.0])
    assert a != array(array=mid, bounds=bounds)
    assert a != array("y", array=mid, bounds=bounds, durations=durations)
    assert a != "not a spec"


def test_array_roundtrips_inside_fly() -> None:
    spec = Fly(Array("x", np.array([0.0, 1.0, 3.0])))
    assert Spec.deserialize(spec.serialize()) == spec
