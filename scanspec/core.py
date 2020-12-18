from typing import Any, Callable, Dict, Iterator, List, Type, TypeVar

import numpy as np
from pydantic import BaseModel
from pydantic.fields import Field, FieldInfo


# These are used in the definition of the Schema
# It allows the class to be inferred from the serialized "type" field
class _SerializableMetaClass(type(BaseModel)):  # type: ignore
    def __new__(mcs, name, bases, namespace, **kwargs):
        # Override type in namespace to be the literal value of the class name
        namespace["type"] = Field(name, const=True)
        return super().__new__(mcs, name, bases, namespace, **kwargs)


class Serializable(BaseModel, metaclass=_SerializableMetaClass):
    """pydantic BaseModel that adds support for positional arguments and a type
    parameter from class name."""

    type: str

    class Config:
        # Forbid any extra input
        extra = "forbid"

    def __init__(self, *args, **kwargs):
        # Allow positional args, but don't include type
        keys = [x for x in self.__fields__ if x != "type"]
        kwargs.update(zip(keys, args))
        # Work around the fact that arg=Field(default, ...) doesn't
        # work with validate_arguments
        for k, v in kwargs.items():
            if isinstance(v, FieldInfo):
                kwargs[k] = v.default
        super().__init__(**kwargs)

    def __repr_args__(self):
        return [(a, v) for a, v in super().__repr_args__() if a != "type"]


#: Positions map {key: positions_ndarray}
#: E.g. {xmotor: array([0, 1, 2]), ymotor: array([2, 2, 2])}
Positions = Dict[Any, np.ndarray]

#: The type of class the function will return
T = TypeVar("T")


def if_instance_do(x, cls: Type[T], func: Callable[[T], Any]):
    """If x is of type cls then return func(x), otherwise return NotImplemented.
    Used as a helper when implementing operator overloading"""
    if isinstance(x, cls):
        return func(x)
    else:
        return NotImplemented


class Dimension:
    """Represents a linear stack of positions and bounds. A list of Dimensions
    is interpreted as nested from slowest moving to fastest moving, so each
    faster Dimension will iterate once per position of the slower Dimension.
    When fly-scanning they key will traverse lower-position-upper on the fastest
    Dimension for each point in the scan.

    Args:
        positions: The centre positions of the scan for each key
        lower: Lower bounds if different from positions
        upper: Upper bounds if different from positions
        snake: If True then every other iteration of this Dimension within a
            slower moving Dimension will be reversed

    See Also:
        `what-are-dimensions`
    """

    def __init__(
        self,
        positions: Positions,
        lower: Positions = None,
        upper: Positions = None,
        snake: bool = False,
    ):
        #: The centre positions of the scan for each key
        self.positions = positions
        #: The lower bounds of each scan point for each key for fly-scanning
        self.lower = lower or positions
        #: The upper bounds of each scan point for each key for fly-scanning
        self.upper = upper or positions
        #: Whether every other iteration of this Dimension within a slower
        #: moving Dimension will be reversed
        self.snake = snake
        # Check all keys and ordering are the same
        assert list(self.positions) == list(self.lower) == list(self.upper), (
            f"Mismatching keys "
            f"{list(self.positions)} != {list(self.lower)} != {list(self.upper)}"
        )
        # Check all lengths are the same
        lengths = set(
            len(arr)
            for d in (self.positions, self.lower, self.upper)
            for arr in d.values()
        )
        assert len(lengths) <= 1, f"Mismatching lengths {list(lengths)}"

    def keys(self) -> List:
        """The keys that are present in `positions`, `lower` and `upper`
        which will move during the scan"""
        return list(self.positions.keys())

    def __len__(self) -> int:
        """The number of `positions` in the scan"""
        # All positions arrays are same length, pick the first one
        return len(list(self.positions.values())[0])

    def _dim_with(self, func: Callable[[str, Any], np.ndarray]) -> "Dimension":
        def apply_func(a: str):
            return {k: func(a, k) for k in getattr(self, a)}

        # Apply to every array in positions
        kwargs = dict(positions=apply_func("positions"), snake=self.snake)
        # If lower and upper are different, apply to those too
        if self.lower is not self.positions:
            kwargs["lower"] = apply_func("lower")
        if self.upper is not self.positions:
            kwargs["upper"] = apply_func("upper")
        return Dimension(**kwargs)

    def tile(self, reps: int) -> "Dimension":
        """Return a new Dimension that iterates self reps times

        >>> dim = Dimension({"x": np.array([1, 2, 3])})
        >>> dim.tile(reps=2).positions
        {'x': array([1, 2, 3, 1, 2, 3])}
        """
        return self._dim_with(lambda a, k: np.tile(getattr(self, a)[k], reps))

    def repeat(self, reps: int) -> "Dimension":
        """Return a new Dimension that repeats each point in self reps times

        >>> dim = Dimension({"x": np.array([1, 2, 3])})
        >>> dim.repeat(reps=2).positions
        {'x': array([1, 1, 2, 2, 3, 3])}
        """
        return self._dim_with(lambda a, k: np.repeat(getattr(self, a)[k], reps))

    def mask(self, mask: np.ndarray) -> "Dimension":
        """Return a new Dimension that produces only points from self in the
        mask

        >>> dim = Dimension({"x": np.array([1, 2, 3])})
        >>> dim.mask(np.array([1, 0, 1])).positions
        {'x': array([1, 3])}
        """
        indices = mask.nonzero()[0]
        return self._dim_with(lambda a, k: getattr(self, a)[k][indices])

    def copy(self) -> "Dimension":
        """Return a shallow copy of the current Dimension (dicts copied,
        arrays within them are not)"""
        return self._dim_with(lambda a, k: getattr(self, a)[k])

    def _check_dim(self, other: "Dimension"):
        assert isinstance(other, Dimension), f"Expected Dimension, got {other}"
        assert self.snake == other.snake, "Snake settings don't match"

    def concat(self, other: "Dimension") -> "Dimension":
        """Return a new Dimension with arrays from self and other concatenated
        together. Require both Dimensions to have the same keys and snake
        settings

        >>> dim = Dimension({"x": np.array([1, 2, 3])})
        >>> dim2 = Dimension({"x": np.array([5, 6, 7])})
        >>> dim.concat(dim2).positions
        {'x': array([1, 2, 3, 5, 6, 7])}
        """
        self._check_dim(other)
        assert self.keys() == other.keys(), f"Keys {self.keys()} != {other.keys()}"
        return self._dim_with(
            lambda a, k: np.concatenate((getattr(self, a)[k], getattr(other, a)[k]))
        )

    def zip(self, other: "Dimension") -> "Dimension":
        """Return a new Dimension with arrays from keys of self and other
        merged together. Require both Dimensions to not share keys, and
        to have the snake settings

        >>> dimx = Dimension({"x": np.array([1, 2, 3])})
        >>> dimy = Dimension({"y": np.array([5, 6, 7])})
        >>> dimx.zip(dimy).positions
        {'x': array([1, 2, 3]), 'y': array([5, 6, 7])}
        """
        self._check_dim(other)
        overlapping = list(set(self.keys()).intersection(other.keys()))
        assert not overlapping, f"Zipping would overwrite keys {overlapping}"
        # rely on the constructor to check lengths
        dim = Dimension(
            positions={**self.positions, **other.positions},
            lower={**self.lower, **other.lower},
            upper={**self.upper, **other.upper},
            snake=self.snake,
        )
        return dim


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
    >>> squash_dimensions([dimy, dimx]).positions
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
            non_snaking = [k for d in dimensions for k in d.keys() if not d.snake]
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
                    f"otherwise {dim.keys()} would jump in position"
                )
    return squashed


class Path:
    """Create a consumable Path through a list of Dimensions representing a
    scan path.

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
        >>> path.consume(3).positions
        {'y': array([3, 3, 4]), 'x': array([1, 2, 2])}
        >>> path.consume(3).positions
        {'y': array([4]), 'x': array([1])}
        >>> path.consume(3).positions
        {'y': array([], dtype=int64), 'x': array([], dtype=int64)}
        """
        if num is None:
            end_index = self.end_index
        else:
            end_index = min(self.index + num, self.end_index)
        indices = np.arange(self.index, end_index)
        self.index = end_index
        positions, lower, upper = {}, {}, {}
        if len(indices) > 0:
            self.index = indices[-1] + 1
        # Example numbers below from a 2x3x4 ZxYxX scan
        for i, dim in enumerate(self.dimensions):
            # Number of times each position will repeat: Z:12, Y:4, X:1
            repeats = np.product(self._lengths[i + 1 :])
            # How big is this dim: Z:2, Y:3, X:4
            dim_len = self._lengths[i]
            # Scan indices mapped to indices within dimension:
            # Z:000000000000111111111111
            # Y:000011112222000011112222
            # X:012301230123012301230123
            dim_indices = (indices // repeats) % dim_len
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
                for key in dim.keys():
                    positions[key] = dim.positions[key][snake_indices]
                    # If going backwards, select from the opposite bound
                    lower[key] = np.where(
                        backwards,
                        dim.upper[key][snake_indices],
                        dim.lower[key][snake_indices],
                    )
                    upper[key] = np.where(
                        backwards,
                        dim.lower[key][snake_indices],
                        dim.upper[key][snake_indices],
                    )
            else:
                for key in dim.keys():
                    positions[key] = dim.positions[key][dim_indices]
                    lower[key] = dim.lower[key][dim_indices]
                    upper[key] = dim.upper[key][dim_indices]
        return Dimension(positions, lower, upper)

    def __len__(self) -> int:
        """Number of points left in a scan, reduces when `consume` is called"""
        return self.end_index - self.index


class SpecPositions:
    """Convenience iterable that produces the scan positions for each axis. For
    better performance, consume from a `Path` instead.

    Args:
        dimensions: The Dimensions describing the scan, from slowest to fastest
            moving

    See Also:
        `iterate-a-spec`

    >>> dimx = Dimension({"x": np.array([1, 2])}, snake=True)
    >>> dimy = Dimension({"y": np.array([3, 4])})
    >>> sp = SpecPositions([dimy, dimx])
    >>> for p in sp: print(p)
    {'y': 3, 'x': 1}
    {'y': 3, 'x': 2}
    {'y': 4, 'x': 2}
    {'y': 4, 'x': 1}
    """

    def __init__(self, dimensions: List[Dimension]):
        #: The Dimensions describing the scan, from slowest to fastest moving
        self.dimensions = dimensions

    @property
    def keys(self) -> List:
        """The keys that will be present in each position dictionary"""
        keys = []
        for dim in self.dimensions:
            keys += dim.keys()
        return keys

    def __len__(self) -> int:
        """The number of dictionaries that will be produced if iterated over"""
        return np.product([len(dim) for dim in self.dimensions])

    def __iter__(self) -> Iterator[Positions]:
        path = Path(self.dimensions)
        while len(path):
            dim = path.consume(1)
            yield {k: dim.positions[k][0] for k in dim.keys()}
