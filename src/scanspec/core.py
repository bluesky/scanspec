"""Core classes like `Dimension` and `Path`."""

from __future__ import annotations

import itertools
import sys
import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache
from inspect import isclass
from typing import (
    Any,
    Generic,
    Literal,
    TypeVar,
    get_args,
    get_origin,
)

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, GetCoreSchemaHandler
from pydantic.dataclasses import is_pydantic_dataclass, rebuild_dataclass
from pydantic_core import CoreSchema
from pydantic_core.core_schema import tagged_union_schema

if sys.version_info >= (3, 12):
    from types import get_original_bases
else:
    # function added to stdlib in 3.12
    def get_original_bases(cls: type, /) -> tuple[Any, ...]:
        try:
            return cls.__dict__.get("__orig_bases__", cls.__bases__)
        except AttributeError:
            raise TypeError(
                f"Expected an instance of type, not {type(cls).__name__!r}"
            ) from None


__all__ = [
    "Axis",
    "OtherAxis",
    "if_instance_do",
    "AxesPoints",
    "Dimension",
    "SnakedDimension",
    "gap_between_frames",
    "squash_frames",
    "Path",
    "Midpoints",
    "discriminated_union_of_subclasses",
    "StrictConfig",
    "DURATION",
]

#: Can be used as a special key to indicate how long each point should be
DURATION = "DURATION"

#: Used to ensure pydantic dataclasses error if given extra arguments
StrictConfig: ConfigDict = {"extra": "forbid", "arbitrary_types_allowed": True}

C = TypeVar("C")
T = TypeVar("T")

GapArray = npt.NDArray[np.bool_]
DurationArray = npt.NDArray[np.float64]


class UnsupportedSubclass(RuntimeWarning):
    """Warning for subclasses that are not simple extensions of generic types."""

    pass


def discriminated_union_of_subclasses(
    super_cls: type[C],
    discriminator: str = "type",
) -> type[C]:
    """Add all subclasses of super_cls to a discriminated union.

    For all subclasses of super_cls, add a discriminator field to identify
    the type. Raw JSON should look like {<discriminator>: <type name>, params for
    <type name>...}.

    Subclasses that extend this class must be Pydantic dataclasses, and types that
    need their schema to be updated when a new type that extends super_cls is
    created must be either Pydantic dataclasses or BaseModels.

    Example::

        @discriminated_union_of_subclasses
        class Expression(ABC):
            @abstractmethod
            def calculate(self) -> int:
                ...


        @dataclass
        class Add(Expression):
            left: Expression
            right: Expression

            def calculate(self) -> int:
                return self.left.calculate() + self.right.calculate()


        @dataclass
        class Subtract(Expression):
            left: Expression
            right: Expression

            def calculate(self) -> int:
                return self.left.calculate() - self.right.calculate()


        @dataclass
        class IntLiteral(Expression):
            value: int

            def calculate(self) -> int:
                return self.value


        my_sum = Add(IntLiteral(5), Subtract(IntLiteral(10), IntLiteral(2)))
        assert my_sum.calculate() == 13

        assert my_sum == parse_obj_as(
            Expression,
            {
                "type": "Add",
                "left": {"type": "IntLiteral", "value": 5},
                "right": {
                    "type": "Subtract",
                    "left": {"type": "IntLiteral", "value": 10},
                    "right": {"type": "IntLiteral", "value": 2},
                },
            },
        )

    Args:
        super_cls: The superclass of the union, Expression in the above example
        discriminator: The discriminator that will be inserted into the
            serialized documents for type determination. Defaults to "type".

    Returns:
        Type: decorated superclass with handling for subclasses to be added
            to its discriminated union for deserialization

    """
    tagged_union = _TaggedUnion(super_cls, discriminator)
    _tagged_unions[super_cls] = tagged_union

    def add_subclass_to_union(subclass: type[C]):
        # Add a discriminator field to a subclass so it can
        # be identified when deserializing
        subclass.__annotations__ = {
            **subclass.__annotations__,
            discriminator: Literal[subclass.__name__],  # type: ignore
        }
        setattr(subclass, discriminator, Field(subclass.__name__, repr=False))  # type: ignore

    def get_schema_of_union(
        cls: type[C], actual_type: type, handler: GetCoreSchemaHandler
    ):
        super(super_cls, cls).__init_subclass__()
        if cls is not super_cls:
            tagged_union.add_member(cls)
            return handler(cls)
        # Rebuild any dataclass (including this one) that references this union
        # Note that this has to be done after the creation of the dataclass so that
        # previously created classes can refer to this newly created class
        return tagged_union.schema(actual_type, handler)

    super_cls.__init_subclass__ = classmethod(add_subclass_to_union)  # type: ignore
    super_cls.__get_pydantic_core_schema__ = classmethod(get_schema_of_union)  # type: ignore
    return super_cls


_tagged_unions: dict[type, _TaggedUnion] = {}


class _TaggedUnion:
    def __init__(self, base_class: type[Any], discriminator: str):
        self._base_class = base_class
        # Classes and their field names that refer to this tagged union
        self._discriminator = discriminator
        # The members of the tagged union, i.e. subclasses of the baseclass
        self._subclasses: list[type] = []
        # The type parameters expected for the base class of the union
        self._generics = _parameters(base_class)

    def add_member(self, cls: type):
        if cls in self._subclasses:
            return
        elif not self._support_subclass(cls):
            warnings.warn(
                f"Subclass {cls} has unsupported generics and will not be part "
                "of the tagged union",
                UnsupportedSubclass,
                stacklevel=2,
            )
            return
        self._subclasses.append(cls)
        for member in self._subclasses:
            if member is not cls:
                _TaggedUnion._rebuild(member)

    @staticmethod
    def _rebuild(cls_or_func: Callable[..., T]) -> None:
        if isclass(cls_or_func):
            if is_pydantic_dataclass(cls_or_func):
                rebuild_dataclass(cls_or_func, force=True)
            if issubclass(cls_or_func, BaseModel):
                cls_or_func.model_rebuild(force=True)

    def schema(self, actual_type: type, handler: GetCoreSchemaHandler) -> CoreSchema:
        return tagged_union_schema(
            _make_schema(
                tuple(
                    self._specify_generics(sub, actual_type) for sub in self._subclasses
                ),
                handler,
            ),
            discriminator=self._discriminator,
            ref=self._base_class.__name__,
        )

    def _support_subclass(self, subcls: type) -> bool:
        if subcls == self._base_class:
            return True
        sub_params = _parameters(subcls)
        if len(self._generics) != len(sub_params):
            return False
        if not all(
            _compatible_types(actual, target)
            for actual, target in zip(self._generics, sub_params, strict=True)
        ):
            return False
        if any(
            not self._support_subclass(get_origin(base) or base)
            for base in get_original_bases(subcls)
        ):
            return False
        return True

    def _specify_generics(self, subcls: type, actual_type: type) -> type:
        args = get_args(actual_type)
        if args:
            return subcls[args]  # type: ignore
        return subcls


def _parameters(possibly_generic: type) -> tuple[Any, ...]:
    return getattr(possibly_generic, "__parameters__", ())


def _compatible_types(left: TypeVar, right: TypeVar) -> bool:
    return (
        left.__bound__ == right.__bound__
        and left.__constraints__ == right.__constraints__
        and left.__covariant__ == right.__covariant__
        and left.__contravariant__ == right.__contravariant__
    )


@lru_cache(1)
def _make_schema(
    members: tuple[type[Any], ...], handler: Callable[[Any], CoreSchema]
) -> dict[str, CoreSchema]:
    return {member.__name__: handler(member) for member in members}


def if_instance_do(x: C, cls: type[C], func: Callable[[C], T]) -> T:
    """If x is of type cls then return func(x), otherwise return NotImplemented.

    Used as a helper when implementing operator overloading.
    """
    if isinstance(x, cls):
        return func(x)
    else:
        return NotImplemented


#: A type variable for an `axis_` that can be specified for a scan
Axis = TypeVar("Axis", covariant=True)

#: Alternative axis variable to be used when two are required in the same type binding
OtherAxis = TypeVar("OtherAxis")

#: Map of axes to float ndarray of points
#: E.g. {xmotor: array([0, 1, 2]), ymotor: array([2, 2, 2])}
AxesPoints = dict[Axis, npt.NDArray[np.float64]]


@dataclass
class Slice(Generic[Axis]):
    """Generalization of the Dimensions class.

    Only holds information but no methods to handle it.
    """

    midpoints: AxesPoints[Axis]
    gap: GapArray
    lower: AxesPoints[Axis]
    upper: AxesPoints[Axis]
    duration: DurationArray | None = None

    def __len__(self) -> int:
        """The number of frames in this section of the scan."""
        # All axespoints arrays are same length, pick the first one
        return len(self.gap)

    def axes(self) -> list[Axis]:
        return list(self.midpoints.keys())


class Dimension(Generic[Axis]):
    """Represents a series of scan frames along a number of axes.

    During a scan each axis will traverse lower-midpoint-upper for each frame.

    Args:
        midpoints: The midpoints of scan frames for each axis
        lower: Lower bounds of scan frames if different from midpoints
        upper: Upper bounds of scan frames if different from midpoints
        gap: If supplied, define if there is a gap between frame and previous
            otherwise it is calculated by looking at lower and upper bounds

    Typically used in two ways:

    - A list of Dimension objects returned from `Spec.calculate` represents a scan
      as a linear stack of frames. Interpreted as nested from slowest moving to
      fastest moving, so each faster Dimension object will iterate once per
      position of the slower Dimension object. It is passed to a `Path` for
      calculation of the actual scan path.
    - A single Dimension object returned from `Path.consume` represents a chunk of
      frames forming part of a scan path, for interpretation by the code
      that will actually perform the scan.

    See Also:
        `technical-terms`

    """

    def __init__(
        self,
        midpoints: AxesPoints[Axis],
        lower: AxesPoints[Axis] | None = None,
        upper: AxesPoints[Axis] | None = None,
        gap: GapArray | None = None,
    ):
        #: The midpoints of scan frames for each axis
        self.midpoints = midpoints
        #: The lower bounds of each scan frame in each axis for fly-scanning
        self.lower = lower or midpoints
        #: The upper bounds of each scan frame in each axis for fly-scanning
        self.upper = upper or midpoints
        if gap is not None:
            #: Whether there is a gap between this frame and the previous. First
            #: element is whether there is a gap between the last frame and the first
            self.gap = gap
        else:
            # Need to calculate gap as not passed one
            # We have a gap if upper[i] != lower[i+1] for any axes
            axes_gap = [
                np.roll(upper, 1) != lower
                for upper, lower in zip(
                    self.upper.values(), self.lower.values(), strict=False
                )
            ]
            self.gap = np.logical_or.reduce(axes_gap)
        # Check all axes and ordering are the same
        assert list(self.midpoints) == list(self.lower) == list(self.upper), (
            f"Mismatching axes "
            f"{list(self.midpoints)} != {list(self.lower)} != {list(self.upper)}"
        )
        # Check all lengths are the same
        lengths = {
            len(arr)
            for d in (self.midpoints, self.lower, self.upper)
            for arr in d.values()
        }
        lengths.add(len(self.gap))
        assert len(lengths) <= 1, f"Mismatching lengths {list(lengths)}"

    def axes(self) -> list[Axis]:
        """The axes which will move during the scan.

        These will be present in `midpoints`, `lower` and `upper`.
        """
        return list(self.midpoints.keys())

    def __len__(self) -> int:
        """The number of frames in this section of the scan."""
        # All axespoints arrays are same length, pick the first one
        return len(self.gap)

    def extract(
        self, indices: npt.NDArray[np.signedinteger[Any]], calculate_gap: bool = True
    ) -> Dimension[Axis]:
        """Return a new Dimension object restricted to the indices provided.

        Args:
            indices: The indices of the frames to extract, modulo scan length
            calculate_gap: If True then recalculate the gap from upper and lower

        >>> frames = Dimension({"x": np.array([1, 2, 3])})
        >>> frames.extract(np.array([1, 0, 1])).midpoints
        {'x': array([2, 1, 2])}

        """
        dim_indices = indices % len(self)

        def extract_dict(ds: Iterable[AxesPoints[Axis]]) -> AxesPoints[Axis]:
            for d in ds:
                return {k: v[dim_indices] for k, v in d.items()}
            return {}

        def extract_gap(gaps: Iterable[GapArray]) -> GapArray | None:
            for gap in gaps:
                if not calculate_gap:
                    return gap[dim_indices]
            return None

        return _merge_frames(self, dict_merge=extract_dict, gap_merge=extract_gap)

    def concat(self, other: Dimension[Axis], gap: bool = False) -> Dimension[Axis]:
        """Return a new Dimension object concatenating self and other.

        Requires both Dimension objects to have the same axes, but not necessarily in
        the same order. The order is inherited from self, so other may be reordered.

        Args:
            other: The Dimension to concatenate to self
            gap: Whether to force a gap between the two Dimension objects

        >>> frames = Dimension({"x": np.array([1, 2, 3]), "y": np.array([6, 5, 4])})
        >>> frames2 = Dimension({"y": np.array([3, 2, 1]), "x": np.array([4, 5, 6])})
        >>> frames.concat(frames2).midpoints
        {'x': array([1, 2, 3, 4, 5, 6]), 'y': array([6, 5, 4, 3, 2, 1])}

        """
        assert set(self.axes()) == set(other.axes()), (
            f"axes {self.axes()} != {other.axes()}"
        )

        def concat_dict(ds: Sequence[AxesPoints[Axis]]) -> AxesPoints[Axis]:
            # Concat each array in midpoints, lower, upper. E.g.
            # lower[ax] = np.concatenate(self.lower[ax], other.lower[ax])
            return {a: np.concatenate([d[a] for d in ds]) for a in self.axes()}

        def concat_gap(gaps: Sequence[GapArray]) -> GapArray:
            g = np.concatenate(gaps)
            # Calc the first frame
            g[0] = gap_between_frames(other, self)
            # And the join frame
            g[len(self)] = gap or gap_between_frames(self, other)
            return g

        return _merge_frames(self, other, dict_merge=concat_dict, gap_merge=concat_gap)

    def zip(self, other: Dimension[Axis]) -> Dimension[Axis]:
        """Return a new Dimension object merging self and other.

        Require both Dimension objects to not share axes.

        >>> fx = Dimension({"x": np.array([1, 2, 3])})
        >>> fy = Dimension({"y": np.array([5, 6, 7])})
        >>> fx.zip(fy).midpoints
        {'x': array([1, 2, 3]), 'y': array([5, 6, 7])}
        """
        overlapping = list(set(self.axes()).intersection(other.axes()))
        assert not overlapping, f"Zipping would overwrite axes {overlapping}"

        def zip_dict(ds: Sequence[AxesPoints[Axis]]) -> AxesPoints[Axis]:
            # Merge dicts for midpoints, lower, upper. E.g.
            # lower[ax] = {**self.lower[ax], **other.lower[ax]}
            return dict(kv for d in ds for kv in d.items())

        def zip_gap(gaps: Sequence[GapArray]) -> GapArray:
            # Gap if either frames has a gap. E.g.
            # gap[i] = self.gap[i] | other.gap[i]
            return np.logical_or.reduce(gaps)

        return _merge_frames(self, other, dict_merge=zip_dict, gap_merge=zip_gap)


def _merge_frames(
    *stack: Dimension[Axis],
    dict_merge: Callable[[Sequence[AxesPoints[Axis]]], AxesPoints[Axis]],  # type: ignore
    gap_merge: Callable[[Sequence[GapArray]], GapArray | None],
) -> Dimension[Axis]:
    types = {type(fs) for fs in stack}
    assert len(types) == 1, f"Mismatching types for {stack}"
    cls = types.pop()

    # Apply to midpoints, force calculation of gap
    return cls(
        midpoints=dict_merge([fs.midpoints for fs in stack]),
        gap=gap_merge([fs.gap for fs in stack]),
        # If any lower or upper are different, apply to those
        lower=dict_merge([fs.lower for fs in stack])
        if any(fs.midpoints is not fs.lower for fs in stack)
        else None,
        upper=dict_merge([fs.upper for fs in stack])
        if any(fs.midpoints is not fs.upper for fs in stack)
        else None,
    )


def stack2dimension(
    dimensions: list[Dimension[Axis]],
    indices: npt.NDArray[np.signedinteger] | None = None,
    lengths: npt.NDArray[np.signedinteger] | None = None,
) -> Dimension[Axis]:
    if lengths is None:
        lengths = np.array([len(f) for f in dimensions])

    if indices is None:
        indices = np.arange(0, int(np.prod(lengths)))

    stack: Dimension[Axis] = Dimension(
        {},
        {},
        {},
        np.zeros(indices.shape, dtype=np.bool_),
    )
    # Example numbers below from a 2x3x4 ZxYxX scan
    for i, frames in enumerate(dimensions):
        # Number of times each frame will repeat: Z:12, Y:4, X:1
        repeats = np.prod(lengths[i + 1 :])
        # Scan indices mapped to indices within Dimension object:
        # Z:000000000000111111111111
        # Y:000011112222000011112222
        # X:012301230123012301230123
        if repeats > 1:
            dim_indices = indices // repeats
        else:
            dim_indices = indices
        # Create the sliced frames
        sliced = frames.extract(dim_indices, calculate_gap=False)
        if repeats > 1:
            # Whether this frames contributes to the gap bit
            # Z:000000000000100000000000
            # Y:000010001000100010001000
            # X:111111111111111111111111
            in_gap = (indices % repeats) == 0
            # If in_gap, then keep the relevant gap bit
            sliced.gap &= in_gap
        # Zip it with the output Dimension object
        stack = stack.zip(sliced)

    return stack


def dimension2slice(
    dimension: Dimension[Axis], duration: DurationArray | None
) -> Slice[Axis]:
    return Slice(
        midpoints=dimension.midpoints,
        gap=dimension.gap,
        upper=dimension.upper,
        lower=dimension.lower,
        duration=duration,
    )


class SnakedDimension(Dimension[Axis]):
    """Like a `Dimension` object, but each alternate repetition will run in reverse."""

    def __init__(
        self,
        midpoints: AxesPoints[Axis],
        lower: AxesPoints[Axis] | None = None,
        upper: AxesPoints[Axis] | None = None,
        gap: GapArray | None = None,
    ):
        super().__init__(midpoints, lower=lower, upper=upper, gap=gap)
        # Override first element of gap to be True, as subsequent runs
        # of snake scans are always joined end -> start
        self.gap[0] = False

    @classmethod
    def from_frames(
        cls: type[SnakedDimension[Any]], frames: Dimension[OtherAxis]
    ) -> SnakedDimension[OtherAxis]:
        """Create a snaked version of a `Dimension` object."""
        return cls(frames.midpoints, frames.lower, frames.upper, frames.gap)

    def extract(
        self, indices: npt.NDArray[np.signedinteger[Any]], calculate_gap: bool = True
    ) -> Dimension[Axis]:
        """Return a new Dimension object restricted to the indices provided.

        Args:
            indices: The indices of the frames to extract, can extend past len(self)
            calculate_gap: If True then recalculate the gap from upper and lower

        >>> frames = SnakedDimension({"x": np.array([1, 2, 3])})
        >>> frames.extract(np.array([0, 1, 2, 3, 4, 5])).midpoints
        {'x': array([1, 2, 3, 3, 2, 1])}

        """
        # Calculate the indices
        # E.g for len = 4
        # indices:       0123456789
        # backwards:     0000111100
        # snake_indices: 0123321001
        # gap_indices:   0123032101
        length = len(self)
        backwards = (indices // length) % 2
        snake_indices = np.where(backwards, (length - 1) - indices, indices) % length
        cls: type[Dimension[Any]]
        if not calculate_gap:
            cls = Dimension
            gap = self.gap[np.where(backwards, length - indices, indices) % length]
        else:
            cls = type(self)
            gap = None

        # Apply to midpoints
        return cls(
            {k: v[snake_indices] for k, v in self.midpoints.items()},
            gap=gap,
            # If lower or upper are different, apply to those
            lower={
                k: np.where(backwards, self.upper[k][snake_indices], v[snake_indices])
                for k, v in self.lower.items()
            }
            if self.midpoints is not self.lower
            else None,
            upper={
                k: np.where(backwards, self.lower[k][snake_indices], v[snake_indices])
                for k, v in self.upper.items()
            }
            if self.midpoints is not self.upper
            else None,
        )


def gap_between_frames(frames1: Dimension[Axis], frames2: Dimension[Axis]) -> bool:
    """Is there a gap between end of frames1 and start of frames2."""
    return any(frames1.upper[a][-1] != frames2.lower[a][0] for a in frames1.axes())


def squash_frames(
    stack: list[Dimension[Axis]], check_path_changes: bool = True
) -> Dimension[Axis]:
    """Squash a stack of nested Dimension into a single one.

    Args:
        stack: The Dimension stack to squash, from slowest to fastest moving
        check_path_changes: If True then check that nesting the output
            Dimension object within others will provide the same path
            as nesting the input Dimension stack within others

    See Also:
        `why-squash-can-change-path`

    >>> fx = SnakedDimension({"x": np.array([1, 2])})
    >>> fy = Dimension({"y": np.array([3, 4])})
    >>> squash_frames([fy, fx]).midpoints
    {'y': array([3, 3, 4, 4]), 'x': array([1, 2, 2, 1])}

    """
    # Consuming a Path through these Dimension performs the squash
    squashed = stack2dimension(stack)
    # Check that the squash is the same as the original
    if stack and isinstance(stack[0], SnakedDimension):
        squashed = SnakedDimension.from_frames(squashed)
        # The top level is snaking, so this Dimension object will run backwards
        # This means any non-snaking axes will run backwards, which is
        # surprising, so don't allow it
        if check_path_changes:
            non_snaking = [
                k for d in stack for k in d.axes() if not isinstance(d, SnakedDimension)
            ]
            if non_snaking:
                raise ValueError(
                    f"Cannot squash non-snaking Dimension inside a SnakingFrames "
                    f"otherwise {non_snaking} would run backwards"
                )
    elif check_path_changes:
        # The top level is not snaking, so make sure there is an even
        # number of iterations of any snaking axis within it so it
        # doesn't jump when this frames object is iterated a second time
        for i, frames in enumerate(stack):
            # A SnakedDimension within a non-snaking top level must repeat
            # an even number of times
            if (
                isinstance(frames, SnakedDimension)
                and np.prod(np.array([len(f) for f in stack])[:i]) % 2
            ):
                raise ValueError(
                    f"Cannot squash SnakingFrames inside a non-snaking Dimension "
                    f"when they do not repeat an even number of times "
                    f"otherwise {frames.axes()} would jump in position"
                )
    return squashed


class Path(Generic[Axis]):
    """A consumable route through a stack of Dimension, representing a scan path.

    Args:
        stack: The Dimension stack describing the scan, from slowest to fastest
            moving
        start: The index of where in the Path to start
        num: The number of scan frames to produce after start. None means up to
            the end

    See Also:
        `iterate-a-spec`

    """

    def __init__(
        self, stack: list[Dimension[Axis]], start: int = 0, num: int | None = None
    ):
        #: The Dimension stack describing the scan, from slowest to fastest moving
        self.stack = stack
        #: Index that is next to be consumed
        self.index = start
        #: The lengths of all the stack
        self.lengths = np.array([len(f) for f in stack])
        #: Index of the end frame, one more than the last index that will be
        #: produced
        self.end_index = int(np.prod(self.lengths))
        if num is not None and start + num < self.end_index:
            self.end_index = start + num

    def consume(self, num: int | None = None) -> Slice[Axis]:
        """Consume at most num frames from the Path and return as a Dimension object.

        >>> fx = SnakedDimension({"x": np.array([1, 2])})
        >>> fy = Dimension({"y": np.array([3, 4])})
        >>> path = Path([fy, fx])
        >>> path.consume(3).midpoints
        {'y': array([3, 3, 4]), 'x': array([1, 2, 2])}
        >>> path.consume(3).midpoints
        {'y': array([4]), 'x': array([1])}
        >>> path.consume(3).midpoints
        {'y': array([], dtype=int64), 'x': array([], dtype=int64)}
        """
        if num is None:
            end_index = self.end_index
        else:
            end_index = min(self.index + num, self.end_index)
        indices = np.arange(self.index, end_index)
        self.index = end_index

        stack = stack2dimension(self.stack, indices, self.lengths)

        duration = None
        if DURATION in stack.axes():
            duration = stack.midpoints.pop(DURATION)

        return dimension2slice(stack, duration)

    def __len__(self) -> int:
        """Number of frames left in a scan, reduces when `consume` is called."""
        return self.end_index - self.index


class Midpoints(Generic[Axis]):
    """Convenience iterable that produces the scan midpoints for each axis.

    For better performance, consume from a `Path` instead.

    Args:
        stack: The stack of Dimension describing the scan, from slowest to fastest
            moving

    See Also:
        `iterate-a-spec`

    >>> fx = SnakedDimension({"x": np.array([1, 2])})
    >>> fy = Dimension({"y": np.array([3, 4])})
    >>> mp = Midpoints([fy, fx])
    >>> for p in mp: print(p)
    {'y': np.int64(3), 'x': np.int64(1)}
    {'y': np.int64(3), 'x': np.int64(2)}
    {'y': np.int64(4), 'x': np.int64(2)}
    {'y': np.int64(4), 'x': np.int64(1)}

    """

    def __init__(self, stack: list[Dimension[Axis]]):
        #: The stack of Dimension describing the scan, from slowest to fastest moving
        self.stack = stack

    @property
    def axes(self) -> list[Axis]:
        """The axes that will be present in each points dictionary."""
        return list(itertools.chain(*(frames.axes() for frames in self.stack)))

    def __len__(self) -> int:
        """The number of dictionaries that will be produced if iterated over."""
        return int(np.prod([len(frames) for frames in self.stack]))

    def __iter__(self) -> Iterator[dict[Axis, float]]:
        """Yield {axis: midpoint} for each frame in the scan."""
        path = Path(self.stack)
        while len(path):
            frames = path.consume(1)
            yield {a: frames.midpoints[a][0] for a in frames.axes()}
