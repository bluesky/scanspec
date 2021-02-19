import json

import pytest
from apischema.validation.errors import ValidationError

from scanspec.regions import Circle
from scanspec.specs import Line, Mask, spec_from_dict, spec_from_json


def test_line_serializes() -> None:
    ob = Line("x", 0, 1, 4)
    serialized = '{"Line": {"key": "x", "start": 0.0, "stop": 1.0, "num": 4}}'
    assert ob.serialize() == json.loads(serialized)
    assert spec_from_json(serialized) == ob
    assert spec_from_dict(json.loads(serialized)) == ob


def test_masked_circle_serializes() -> None:
    ob = Mask(Line("x", 0, 1, 4), Circle("x", "y", x_centre=0, y_centre=1, radius=4))
    serialized = (
        '{"Mask": {"spec": {"Line": {"key": "x", "start": 0, "stop": 1, "num": 4}}, '
        '"region": {"Circle": {"x_key": "x", "y_key": "y", "x_centre": 0, '
        '"y_centre": 1, "radius": 4}}, "check_path_changes": true}}'
    )
    assert ob.serialize() == json.loads(serialized)
    assert spec_from_json(serialized) == ob


def test_product_lines_serializes() -> None:
    ob = Line("y", 2, 3, 5) * Line("x", 0, 1, 4)
    serialized = (
        '{"Product": {"outer": {"Line": {"key": "y", "start": 2.0, "stop": 3.0, '
        '"num": 5}}, "inner": {"Line": {"key": "x", "start": 0.0, "stop": 1.0, '
        '"num": 4}}}}'
    )
    assert ob.serialize() == json.loads(serialized)
    assert spec_from_json(serialized) == ob


def test_extra_arg_fails() -> None:
    with pytest.raises(ValidationError):
        ob = Line("x", 0, 1, 4)
        serialized = (
            '{"Line": {"key": "x", "start": 0.0, "stop": 1.0, "num": 4, "foo": "bar"}}'
        )
        assert spec_from_json(serialized) == ob
