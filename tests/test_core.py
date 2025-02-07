from typing import Annotated, Any, Generic, TypeVar

import pytest
from pydantic import TypeAdapter
from pydantic.dataclasses import dataclass

from scanspec.core import (
    UnsupportedSubclass,
    discriminated_union_of_subclasses,
)

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")

B = TypeVar("B", int, float)


@discriminated_union_of_subclasses
class Parent(Generic[T]):
    pass


@dataclass
class Child(Parent[U]):
    a: U


@dataclass
class AnnotatedChild(Parent[Annotated[U, "comment"]]):
    b: U


@dataclass
class GrandChild(Child[V]):
    # TODO: subclasses with fields?
    pass


@discriminated_union_of_subclasses
class NonGenericParent:
    pass


@dataclass
class NonGenericChild(NonGenericParent):
    a: int
    b: float


def test_specific_implementation_child():
    with pytest.warns(UnsupportedSubclass):

        @dataclass
        class Specific(Parent[int]):
            b: int

    with pytest.warns(UnsupportedSubclass):

        @dataclass
        class SubSpecific(Specific):  # type: ignore
            pass


def test_extra_generic_parameters():
    with pytest.warns(UnsupportedSubclass):

        @dataclass
        class ExtraGeneric(Parent[U], Generic[U, V]):  # type: ignore
            c: U
            d: V


def test_unrelated_generic_parameters():
    with pytest.warns(UnsupportedSubclass):

        @dataclass
        class UnrelatedGeneric(Parent[int], Generic[U]):  # type: ignore
            e: int
            f: U


def test_reordered_generics():
    with pytest.warns(UnsupportedSubclass):

        @dataclass
        class DisorderedGeneric(Parent[U], Generic[T, U, V]):  # type: ignore
            g: T
            h: U
            i: V


@pytest.mark.skip("Unsure if this case should be valid or not")
def test_unionised_child():
    with pytest.warns(UnsupportedSubclass):

        @dataclass
        class UnionSubclass(Parent[int | U]):  # type: ignore
            a: U


def test_untyped_child():
    with pytest.warns(UnsupportedSubclass):

        @dataclass
        class UnmarkedChild(Parent):  # type: ignore we're testing the bad type annotations
            a: int


def test_additional_type_bounds():
    with pytest.warns(UnsupportedSubclass):
        # Adding bounds to the generic parameter is not supported
        @dataclass
        class ConstrainedChild(Parent[B]):  # type: ignore
            cc: B


def test_adding_generics_to_nongeneric():
    with pytest.warns(UnsupportedSubclass):

        @dataclass
        class NewGenerics(NonGenericParent, Generic[T]):  # type: ignore
            a: T


def deserialize(target: type[Any], source: Any) -> Any:
    return TypeAdapter(target).validate_python(source)  # type: ignore


def test_child():
    ch = deserialize(Parent[int], {"type": "Child", "a": "42"})
    assert ch.a == 42

    ch = deserialize(Parent[str], {"type": "Child", "a": "42"})
    assert ch.a == "42"

    ch = deserialize(Parent[list[int]], {"type": "Child", "a": ["1", "2", "3"]})
    assert ch.a == [1, 2, 3]


def test_annotated_child():
    ch = deserialize(Parent[int], {"type": "AnnotatedChild", "b": "42"})
    assert ch.b == 42


@pytest.mark.xfail(reason="Pydantic #11363")
def test_grandchild():
    ch = deserialize(Parent[int], {"type": "GrandChild", "a": "42"})
    assert ch.a == 42


def test_non_generic_child():
    ngc = deserialize(
        NonGenericParent, {"type": "NonGenericChild", "a": "42", "b": "3.14"}
    )
    assert ngc.a == 42
    assert ngc.b == pytest.approx(3.14)  # type: ignore
