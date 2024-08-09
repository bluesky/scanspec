import pytest
from pydantic import BaseModel, TypeAdapter
from pydantic.dataclasses import dataclass

from scanspec.core import StrictConfig, uses_tagged_union
from scanspec.specs import Line, Spec


@uses_tagged_union
class Foo(BaseModel):
    spec: Spec


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


def test_schema_updates_with_new_values():
    old_schema = TypeAdapter(Foo).json_schema()

    @dataclass(config=StrictConfig)
    class Splat(Spec[str]):  # NOSONAR
        def axes(self) -> list[str]:
            return ["*"]

    new_schema = TypeAdapter(Foo).json_schema()

    assert new_schema != old_schema
