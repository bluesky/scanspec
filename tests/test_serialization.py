from pydantic.main import BaseModel
from pydantic.typing import display_as_type

from scanspec.regions import Circle
from scanspec.specs import Line, Mask, spec_from_json


def test_line_serializes() -> None:
    ob = Line("x", 0, 1, 4)
    serialized = '{"type": "Line", "key": "x", "start": 0.0, "stop": 1.0, "num": 4}'
    assert ob.json() == serialized
    assert spec_from_json(serialized) == ob


def test_masked_circle_serializes() -> None:
    ob = Mask(Line("x", 0, 1, 4), Circle("x", "y", x_center=0, y_center=1, radius=4))
    serialized = (
        '{"type": "Mask", '
        '"spec": {"type": "Line", "key": "x", "start": 0.0, "stop": 1.0, "num": 4}, '
        '"region": {"type": "Circle", "x_key": "x", "y_key": "y", "x_centre": 0.0, '
        '"y_centre": 1.0, "radius": 4.0}'
        "}"
    )
    assert ob.json() == serialized
    assert spec_from_json(serialized) == ob


def test_product_lines_serializes() -> None:
    ob = Line("y", 2, 3, 5) * Line("x", 0, 1, 4)
    serialized = (
        '{"type": "Product", '
        '"outer": {"type": "Line", "key": "y", "start": 2.0, "stop": 3.0, "num": 5}, '
        '"inner": {"type": "Line", "key": "x", "start": 0.0, "stop": 1.0, "num": 4}'
        "}"
    )
    assert ob.json() == serialized
    assert spec_from_json(serialized) == ob


def test_pydantic_inspections():
    assert issubclass(Line, BaseModel)
    assert list(Line.__fields__) == ["type", "key", "start", "stop", "num"]
    type_field = Line.__fields__["type"]
    assert type_field.field_info.const
    assert not type_field.required
    assert type_field.default == "Line"
    num_field = Line.__fields__["num"]
    assert not num_field.field_info.const
    assert num_field.field_info.description == "Number of points to produce"
    assert num_field.required
    assert display_as_type(num_field.type_) == "ConstrainedIntValue"
    assert hasattr(Line.bounded, "model") and issubclass(Line.bounded.model, BaseModel)
    assert list(Line.bounded.model.__fields__) == [
        "cls",
        "key",
        "lower",
        "upper",
        "num",
        "args",
        "kwargs",
    ]
    num_field = Line.__fields__["num"]
    assert not num_field.field_info.const
    assert num_field.field_info.description == "Number of points to produce"
    assert num_field.required
    assert display_as_type(num_field.type_) == "ConstrainedIntValue"
