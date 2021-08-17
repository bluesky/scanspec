from types import new_class
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterator,
    List,
    Mapping,
    Type,
    TypeVar,
)

import numpy as np
from apischema import deserialize, deserializer, serialize, serializer, type_name
from apischema.conversions import Conversion
from apischema.metadata import conversion
from apischema.objects import object_deserialization
from apischema.tagged_unions import Tagged, TaggedUnion, get_tagged

__all__ = [
    "S",
    "Serializable",
    "AxesPoints",
    "if_instance_do",
    "Dimension",
    "squash_dimensions",
    "Path",
    "Midpoints",
]


def rec_subclasses(cls: type) -> Iterator[type]:
    """Recursive implementation of type.__subclasses__"""
    for sub_cls in cls.__subclasses__():
        yield sub_cls
        yield from rec_subclasses(sub_cls)


# {cls_name: [functions]}
_alternative_constructors: Dict[str, List[Callable]] = {}

if TYPE_CHECKING:
    # Close enough for mypy
    alternative_constructor = staticmethod
else:

    def alternative_constructor(f):
        """Register an alternative constructor for this class. This will be returned
        as a staticmethod so the signature should not include self/cls.

        >>> import dataclasses
        >>> @dataclasses.dataclass
        ... class Foo:
        ...     a: int
        ...     @alternative_constructor
        ...     def doubled(b: int) -> "Foo":
        ...         return Foo(b * 2)
        ...
        >>> Foo.doubled(2)
        Foo(a=4)
        """
        cls_name = f.__qualname__.split(".")[0]
        _alternative_constructors.setdefault(cls_name, []).append(f)
        return staticmethod(f)


def as_tagged_union(cls: Type):
    def serialization() -> Conversion:
        serialization_union = new_class(
            f"Tagged{cls.__name__}Union",
            (TaggedUnion,),
            exec_body=lambda ns: ns.update(
                {
                    "__annotations__": {
                        sub.__name__: Tagged[sub]  # type: ignore
                        for sub in rec_subclasses(cls)
                    }
                }
            ),
        )
        return Conversion(
            lambda obj: serialization_union(**{obj.__class__.__name__: obj}),
            source=cls,
            target=serialization_union,
            # Conversion must not be inherited because it would lead to infinite
            # recursion otherwise
            inherited=False,
        )

    def deserialization() -> Conversion:
        annotations: Dict[str, Any] = {}
        deserialization_namespace: Dict[str, Any] = {"__annotations__": annotations}
        for sub in rec_subclasses(cls):
            annotations[sub.__name__] = Tagged[sub]  # type: ignore
            # Add tagged fields for all its alternative constructors
            for constructor in _alternative_constructors.get(sub.__name__, ()):
                # Build the alias of the field
                alias = (
                    "".join(map(str.capitalize, constructor.__name__.split("_")))
                    + sub.__name__
                )
                # object_deserialization uses get_type_hints, but the constructor
                # return type is stringified and the class not defined yet,
                # so it must be assigned manually
                constructor.__annotations__["return"] = sub
                # Add constructor tagged field with its conversion
                annotations[alias] = Tagged[sub]  # type: ignore
                deserialization_namespace[alias] = Tagged(
                    conversion(
                        # Use object_deserialization to wrap constructor as deserializer
                        deserialization=object_deserialization(
                            constructor, type_name(alias)
                        )
                    )
                )
        # Create the deserialization tagged union class
        deserialization_union = new_class(
            f"Tagged{cls.__name__}Union",
            (TaggedUnion,),
            exec_body=lambda ns: ns.update(deserialization_namespace),
        )
        return Conversion(
            lambda obj: get_tagged(obj)[1], source=deserialization_union, target=cls
        )

    deserializer(lazy=deserialization, target=cls)
    serializer(lazy=serialization, source=cls)


#: A subclass of Serializable
S = TypeVar("S", bound="Serializable")


class Serializable:
    """Base class for registering apischema (de)serialization conversions.
    Each direct subclass will be registered for (de)serialization as a tagged union
    of its subclasses, using the pattern documented here:
    https://wyfo.github.io/apischema/examples/subclass_tagged_union/"""

    # Base class which will directly inherit from Serializable
    _base_serializable: ClassVar[Type["Serializable"]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if Serializable in cls.__bases__:
            cls._base_serializable = cls
            as_tagged_union(cls)

    def serialize(self) -> Mapping[str, Any]:
        """Serialize to a dictionary representation"""
        # Base serializable class must be passed to serialize in order to use its
        # registered conversion (which is not inherited)
        return serialize(self._base_serializable, self)

    @classmethod
    def deserialize(cls: Type[S], serialization: Mapping[str, Any]) -> S:
        """Deserialize from a dictionary representation"""
        inst = deserialize(cls._base_serializable, serialization)
        assert isinstance(inst, cls)
        return inst


#: Map of axes to points_ndarray
#: E.g. {xmotor: array([0, 1, 2]), ymotor: array([2, 2, 2])}
AxesPoints = Dict[str, np.ndarray]


def if_instance_do(x, cls: Type, func: Callable):
    """If x is of type cls then return func(x), otherwise return NotImplemented.
    Used as a helper when implementing operator overloading"""
    if isinstance(x, cls):
        return func(x)
    else:
        return NotImplemented


def last_first_gap(upper: AxesPoints, lower: AxesPoints) -> bool:
    """Return if there is a gap between last point of upper and first point
    of lower"""
    return any(upper[a][-1] != lower[a][0] for a in lower)


class Dimension:
    """A dimension is a repeatable, possibly snaking structure of frames along a
    number of axes.

    Represents a linear stack of frames. A list of Dimensions
    is interpreted as nested from slowest moving to fastest moving, so each
    faster Dimension will iterate once per position of the slower Dimension.
    When fly-scanning the axis will traverse lower-midpoint-upper on the fastest
    Dimension for each point in the scan.

    Args:
        midpoints: The centre points of the scan for each axis
        lower: Lower bounds if different from midpoints
        upper: Upper bounds if different from midpoints
        gap: Whether there is a gap between this frame and the previous
        snake: If True then every other iteration of this Dimension within a
            slower moving Dimension will be reversed

    See Also:
        `what-are-dimensions`
    """

    def __init__(
        self,
        midpoints: AxesPoints,
        lower: AxesPoints = None,
        upper: AxesPoints = None,
        gap: np.ndarray = None,
        snake: bool = False,
    ):
        #: The centre points of the scan for each axis
        self.midpoints = midpoints
        #: The lower bounds of each scan point in each axis for fly-scanning
        self.lower = lower or midpoints
        #: The upper bounds of each scan point in each axis for fly-scanning
        self.upper = upper or midpoints
        # Initialize snake value
        self._snake = snake
        #: Whether there is a gap between this frame and the previous. First
        #: element is whether there is a gap between the last frame and the first
        if gap is None:
            # We have a gap if upper[i] != lower[i+1] for any axes
            axes_gap = [
                np.roll(u, 1) != l
                for u, l in zip(self.upper.values(), self.lower.values())
            ]
            self.gap = np.logical_or.reduce(axes_gap)
            # If we are snaking then make sure self.gap is updated
            if snake:
                self.snake = True
        else:
            # We pre-calculated gap
            self.gap = gap
        # Check all axes and ordering are the same
        assert list(self.midpoints) == list(self.lower) == list(self.upper), (
            f"Mismatching axes "
            f"{list(self.midpoints)} != {list(self.lower)} != {list(self.upper)}"
        )
        # Check all lengths are the same
        lengths = set(
            len(arr)
            for d in (self.midpoints, self.lower, self.upper)
            for arr in d.values()
        )
        lengths.add(len(self.gap))
        assert len(lengths) <= 1, f"Mismatching lengths {list(lengths)}"

    @property
    def snake(self) -> bool:
        """Whether every other iteration of this Dimension within a slower
        moving Dimension will be reversed"""
        return self._snake

    @snake.setter
    def snake(self, snake: bool):
        assert snake, "Can only set snake=True after init"
        self._snake = True
        # If we are snaking then there is never a gap between end and start
        if len(self.gap) > 0:
            self.gap[0] = False

    def axes(self) -> List:
        """The axes that are present in `midpoints`, `lower` and `upper`
        which will move during the scan"""
        return list(self.midpoints.keys())

    def __len__(self) -> int:
        """The number of `frames` in the scan"""
        # All axespoints arrays are same length, pick the first one
        return len(self.gap)

    def _dim_from_indices(self, indices: np.ndarray, snake: bool) -> "Dimension":
        def apply_to_dict(d: AxesPoints) -> AxesPoints:
            return {k: v[indices] for k, v in d.items()}

        # If lower or upper are different, apply to those
        kwargs = {
            a: apply_to_dict(getattr(self, a))
            for a in ("lower", "upper")
            if self.midpoints is not getattr(self, a)
        }

        # Apply to midpoints, inherit snaked, force calculation of gap
        return Dimension(
            midpoints=apply_to_dict(self.midpoints), gap=None, snake=snake, **kwargs,
        )

    def __getitem__(self, indices: np.ndarray) -> "Dimension":
        """Return a new Dimension that produces this dimension
        restricted to the slice"""
        return self._dim_from_indices(indices, self.snake)

    def tile(self, reps: int, snake: bool = False) -> "Dimension":
        """Return a new Dimension that iterates self reps times

        Args:
            reps: The number of times the dimension will repeat in its entirity
            snake: Whether the output dimension will snake

        >>> dim = Dimension({"x": np.array([1, 2, 3])})
        >>> dim.tile(reps=2).midpoints
        {'x': array([1, 2, 3, 1, 2, 3])}
        """
        return self._dim_from_indices(np.tile(np.arange(len(self)), reps), snake)

    def repeat(self, reps: int, snake: bool = False) -> "Dimension":
        """Return a new Dimension that repeats each point in self reps times

        Args:
            reps: The number of times each point of the dimension will repeat
            snake: Whether the output dimension will snake

        >>> dim = Dimension({"x": np.array([1, 2, 3])})
        >>> dim.repeat(reps=2).midpoints
        {'x': array([1, 1, 2, 2, 3, 3])}
        """
        return self._dim_from_indices(np.repeat(np.arange(len(self)), reps), snake)

    def mask(self, mask: np.ndarray) -> "Dimension":
        """Return a new Dimension that produces only points from self in the
        mask

        >>> dim = Dimension({"x": np.array([1, 2, 3])})
        >>> dim.mask(np.array([1, 0, 1])).midpoints
        {'x': array([1, 3])}
        """
        return self._dim_from_indices(mask.nonzero()[0], self.snake)

    def _merge_dims(
        self,
        other: "Dimension",
        dict_merge: Callable[[AxesPoints, AxesPoints], AxesPoints],
        gap_merge: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ):
        assert isinstance(other, Dimension), f"Expected Dimension, got {other}"
        assert self.snake == other.snake, "Snake settings don't match"
        # If lower or upper are different, apply to those
        kwargs = {
            a: dict_merge(getattr(self, a), getattr(other, a))
            for a in ("lower", "upper")
            if self.midpoints is not getattr(self, a)
            or other.midpoints is not getattr(other, a)
        }

        # Apply to midpoints, inherit snaked, force calculation of gap
        return Dimension(
            midpoints=dict_merge(self.midpoints, other.midpoints),
            gap=gap_merge(self.gap, other.gap),
            snake=self.snake,
            **kwargs,
        )

    def concat(self, other: "Dimension", gap: bool = False) -> "Dimension":
        """Return a new Dimension with arrays from self and other concatenated
        together. Require both Dimensions to have the same axes and snake
        settings

        Args:
            other: The dimension to concatenate to self
            snake: Whether to force a gap between the two dimensions

        >>> dim = Dimension({"x": np.array([1, 2, 3])})
        >>> dim2 = Dimension({"x": np.array([5, 6, 7])})
        >>> dim.concat(dim2).midpoints
        {'x': array([1, 2, 3, 5, 6, 7])}
        """

        def concat_gap(gap1: np.ndarray, gap2: np.ndarray) -> np.ndarray:
            g = np.concatenate((gap1, gap2))
            # Calc the first point
            g[0] = last_first_gap(other.upper, self.lower)
            # And the join point
            g[len(self)] = gap or last_first_gap(self.upper, other.lower)
            return g

        assert self.axes() == other.axes(), f"axes {self.axes()} != {other.axes()}"
        dim = self._merge_dims(
            other,
            # Concat each array in midpoints, lower, upper. E.g.
            # lower[ax] = np.concatenate(self.lower[ax], other.lower[ax])
            lambda d1, d2: {k: np.concatenate((v, d2[k])) for k, v in d1.items()},
            # Gap should be concatted, but calc the join points above
            concat_gap,
        )
        return dim

    def zip(self, other: "Dimension") -> "Dimension":
        """Return a new Dimension with arrays from axes of self and other
        merged together. Require both Dimensions to not share axes, and
        to have the snake settings

        >>> dimx = Dimension({"x": np.array([1, 2, 3])})
        >>> dimy = Dimension({"y": np.array([5, 6, 7])})
        >>> dimx.zip(dimy).midpoints
        {'x': array([1, 2, 3]), 'y': array([5, 6, 7])}
        """
        overlapping = list(set(self.axes()).intersection(other.axes()))
        assert not overlapping, f"Zipping would overwrite axes {overlapping}"
        return self._merge_dims(
            other,
            # Merge dicts for midpoints, lower, upper. E.g.
            # lower[ax] = {**self.lower[ax], **other.lower[ax]}
            lambda d1, d2: {**d1, **d2},
            # Gap if either dim has a gap. E.g.
            # gap[i] = self.gap[i] | other.gap[i]
            np.logical_or,
        )


def squash_dimensions(
    dimensions: List[Dimension], check_path_changes=True
) -> Dimension:
    """Squash a list of nested Dimensions into a single one.

    Args:
        dimensions: The Dimensions to squash, from slowest to fastest moving
        check_path_changes: If True then check that nesting the output
            Dimension within other Dimensions will provide the same path
            as nesting the input Dimension within other Dimensions

    See Also:
        `why-squash-can-change-path`

    >>> dimx = Dimension({"x": np.array([1, 2])}, snake=True)
    >>> dimy = Dimension({"y": np.array([3, 4])})
    >>> squash_dimensions([dimy, dimx]).midpoints
    {'y': array([3, 3, 4, 4]), 'x': array([1, 2, 2, 1])}
    """
    path = Path(dimensions)
    # Comsuming a Path of these dimensions performs the squash
    # TODO: dim.tile might give better performance but is much longer
    squashed = path.consume()
    # Check that the squash is the same as the original
    if dimensions and dimensions[0].snake:
        squashed.snake = True
        # The top level is snaking, so this dimension will run backwards
        # This means any non-snaking axes will run backwards, which is
        # surprising, so don't allow it
        if check_path_changes:
            non_snaking = [k for d in dimensions for k in d.axes() if not d.snake]
            if non_snaking:
                raise ValueError(
                    f"Cannot squash non-snaking Specs in a snaking Dimension "
                    f"otherwise {non_snaking} would run backwards"
                )
    elif check_path_changes:
        # The top level is not snaking, so make sure there is an even
        # number of iterations of any snaking axis within it so it
        # doesn't jump when this dimension is iterated a second time
        for i, dim in enumerate(dimensions):
            # A snaking dimension within a non-snaking top level must repeat
            # an even number of times
            if dim.snake and np.product(path._lengths[:i]) % 2:
                raise ValueError(
                    f"Cannot squash snaking Specs in a non-snaking Dimension "
                    f"when they do not repeat an even number of times "
                    f"otherwise {dim.axes()} would jump in position"
                )
    return squashed


class Path:
    """A consumable route through one or more dimensions,
    representing a scan path.

    Args:
        dimensions: The Dimensions describing the scan, from slowest to fastest
            moving
        start: The index of where in the Path to start
        num: The number of scan points to produce after start. None means up to
            the end

    See Also:
        `iterate-a-spec`
    """

    def __init__(
        self, dimensions: List[Dimension], start: int = 0, num: int = None,
    ):
        #: The Dimensions describing the scan, from slowest to fastest moving
        self.dimensions = dimensions
        #: Index that is next to be consumed
        self.index = start
        self._lengths = np.array([len(dim) for dim in dimensions])
        #: Index of the end point, one more than the last index that will be
        #: produced
        self.end_index = np.product(self._lengths)
        if num is not None and start + num < self.end_index:
            self.end_index = start + num

    def consume(self, num: int = None) -> Dimension:
        """Consume at most num points from the Path and return as a Dimension

        >>> dimx = Dimension({"x": np.array([1, 2])}, snake=True)
        >>> dimy = Dimension({"y": np.array([3, 4])})
        >>> path = Path([dimy, dimx])
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
        gap = np.zeros(indices.shape, dtype=np.bool_)
        self.index = end_index
        midpoints, lower, upper = {}, {}, {}
        if len(indices) > 0:
            self.index = indices[-1] + 1
        # Example numbers below from a 2x3x4 ZxYxX scan
        for i, dim in enumerate(self.dimensions):
            # Number of times each point will repeat: Z:12, Y:4, X:1
            repeats = np.product(self._lengths[i + 1 :])
            # How big is this dim: Z:2, Y:3, X:4
            dim_len = self._lengths[i]
            # Scan indices mapped to indices within dimension:
            # Z:000000000000111111111111
            # Y:000011112222000011112222
            # X:012301230123012301230123
            dim_indices = (indices // repeats) % dim_len
            # Whether this dim contributes to the gap bit
            # Z:000000000000100000000000
            # Y:000010001000100010001000
            # X:111111111111111111111111
            in_gap = (indices % repeats) == 0
            if dim.snake:
                # Whether this point is running backwards:
                # Z:000000000000000000000000
                # Y:000000000000111111111111
                # X:000011110000111100001111
                backwards = (indices // (repeats * dim_len)) % 2
                # The scan indices mapped to snaking ones:
                # Z:000000000000111111111111
                # Y:000011112222222211110000
                # X:012332100123321001233210
                snake_indices = np.where(
                    backwards, dim_len - 1 - dim_indices, dim_indices
                )
                # Gap is the difference between subsequent elements,
                # with True at the beginning, so the reverse dimension
                # needs to be one more than the snake_indices
                # Z:000000000000111111111111
                # Y:000011112222333322221111
                # X:012343210123432101233210
                gap_indices = np.where(backwards, dim_len - dim_indices, dim_indices)
                for axis in dim.axes():
                    midpoints[axis] = dim.midpoints[axis][snake_indices]
                    # If going backwards, select from the opposite bound
                    lower[axis] = np.where(
                        backwards,
                        dim.upper[axis][snake_indices],
                        dim.lower[axis][snake_indices],
                    )
                    upper[axis] = np.where(
                        backwards,
                        dim.lower[axis][snake_indices],
                        dim.upper[axis][snake_indices],
                    )
            else:
                gap_indices = dim_indices
                for axis in dim.axes():
                    midpoints[axis] = dim.midpoints[axis][dim_indices]
                    lower[axis] = dim.lower[axis][dim_indices]
                    upper[axis] = dim.upper[axis][dim_indices]
            # If in_gap, logical_or the relevant gap bit from the dim
            # We use % as gap_indices is 1..dim_len, with dim_len rolling
            # over to 0
            gap |= np.where(in_gap, dim.gap[gap_indices % dim_len], False)

        return Dimension(midpoints, lower, upper, gap)

    def __len__(self) -> int:
        """Number of points left in a scan, reduces when `consume` is called"""
        return self.end_index - self.index


class Midpoints:
    """Convenience iterable that produces the scan points for each axis. For
    better performance, consume from a `Path` instead.

    Args:
        dimensions: The Dimensions describing the scan, from slowest to fastest
            moving

    See Also:
        `iterate-a-spec`

    >>> dimx = Dimension({"x": np.array([1, 2])}, snake=True)
    >>> dimy = Dimension({"y": np.array([3, 4])})
    >>> mp = Midpoints([dimy, dimx])
    >>> for p in mp: print(p)
    {'y': 3, 'x': 1}
    {'y': 3, 'x': 2}
    {'y': 4, 'x': 2}
    {'y': 4, 'x': 1}
    """

    def __init__(self, dimensions: List[Dimension]):
        #: The Dimensions describing the scan, from slowest to fastest moving
        self.dimensions = dimensions

    @property
    def axes(self) -> List:
        """The axes that will be present in each points dictionary"""
        axes = []
        for dim in self.dimensions:
            axes += dim.axes()
        return axes

    def __len__(self) -> int:
        """The number of dictionaries that will be produced if iterated over"""
        return np.product([len(dim) for dim in self.dimensions])

    def __iter__(self) -> Iterator[AxesPoints]:
        path = Path(self.dimensions)
        while len(path):
            dim = path.consume(1)
            yield {a: dim.midpoints[a][0] for a in dim.axes()}
