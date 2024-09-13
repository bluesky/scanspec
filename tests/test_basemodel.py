import pytest
from pydantic import BaseModel, TypeAdapter

from scanspec.specs import Line, Spec


class Foo(BaseModel):
    spec: Spec[str]


simple_foo = Foo(spec=Line("x", 1, 2, 5))
nested_foo = Foo(spec=Line("x", 1, 2, 5) * Line("y", 1, 2, 5))


@pytest.mark.parametrize("model", [simple_foo, nested_foo])
def test_model_validation(model: Foo):
    # To/from Python dict
    as_dict = model.model_dump()
    deserialized = Foo.model_validate(as_dict)
    assert deserialized == model

    # To/from Json dict
    as_json = model.model_dump_json()
    deserialized = Foo.model_validate_json(as_json)
    assert deserialized == model


@pytest.mark.parametrize("model", [simple_foo, nested_foo])
def test_type_adapter(model: Foo):
    type_adapter = TypeAdapter(Foo)

    # To/from Python dict
    as_dict = model.model_dump()
    deserialized = type_adapter.validate_python(as_dict)
    assert deserialized == model

    # To/from Json dict
    as_json = model.model_dump_json()
    deserialized = type_adapter.validate_json(as_json)
    assert deserialized == model
