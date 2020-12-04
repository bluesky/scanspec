from typing import Any, Callable, Dict, Iterator, List, Type, TypeVar

import numpy as np
from pydantic import BaseModel
from pydantic.fields import Field, FieldInfo


# These are used in the definition of the Schema
# It allows the class to be inferred from the serialized "type" field
class _WithTypeMetaClass(type(BaseModel)):  # type: ignore
    def __new__(mcs, name, bases, namespace, **kwargs):
        # Override type in namespace to be the literal value of the class name
        namespace["type"] = Field(name, const=True)
        return super().__new__(mcs, name, bases, namespace, **kwargs)


class WithType(BaseModel, metaclass=_WithTypeMetaClass):
    """BaseModel that adds a type parameter from class name."""

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


Positions = Dict[Any, np.ndarray]


T = TypeVar("T")


def if_instance_do(x, cls: Type[T], func: Callable[[T], Any]):
    if isinstance(x, cls):
        return func(x)
    else:
        return NotImplemented


class Dimension:
    def __init__(
        self,
        positions: Positions,
        lower: Positions = None,
        upper: Positions = None,
        snake=False,
    ):
        self.positions = positions
        self.lower = lower or positions
        self.upper = upper or positions
        lengths = set(
            len(arr)
            for d in (self.positions, self.lower, self.upper)
            for arr in d.values()
        )
        assert len(lengths) <= 1, f"Mismatching lengths {list(lengths)}"
        self.snake = snake

    def keys(self) -> List:
        return list(self.positions.keys())

    def __len__(self) -> int:
        # All positions arrays are same length, pick the first one
        return len(list(self.positions.values())[0])

    def _dim_with(self, func: Callable[[str, Any], np.ndarray]) -> "Dimension":
        def apply_func(a: str):
            return {k: func(a, k) for k in getattr(self, a)}

        kwargs = dict(positions=apply_func("positions"), snake=self.snake)
        if self.lower is not self.positions:
            kwargs["lower"] = apply_func("lower")
        if self.upper is not self.positions:
            kwargs["upper"] = apply_func("upper")
        return Dimension(**kwargs)

    def tile(self, reps: int) -> "Dimension":
        return self._dim_with(lambda a, k: np.tile(getattr(self, a)[k], reps))

    def repeat(self, reps: int) -> "Dimension":
        return self._dim_with(lambda a, k: np.repeat(getattr(self, a)[k], reps))

    def mask(self, mask: np.ndarray) -> "Dimension":
        indices = mask.nonzero()[0]
        return self._dim_with(lambda a, k: getattr(self, a)[k][indices])

    def _check_dim(self, other: "Dimension"):
        assert isinstance(other, Dimension), f"Expected Dimension, gott {other}"
        assert self.snake == other.snake, "Snake settings don't match"

    def concat(self, other: "Dimension") -> "Dimension":
        self._check_dim(other)
        assert self.keys() == other.keys(), f"Keys {self.keys()} != {other.keys()}"
        return self._dim_with(
            lambda a, k: np.concatenate((getattr(self, a)[k], getattr(other, a)[k]))
        )

    def copy(self) -> "Dimension":
        return self._dim_with(lambda a, k: getattr(self, a)[k])

    def __add__(self, other: "Dimension") -> "Dimension":
        """Zip them together"""
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
    dimensions: List[Dimension], check_path_changes=False
) -> Dimension:
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
            if dim.snake and np.product(path.sizes[:i]) % 2:
                raise ValueError(
                    f"Cannot squash snaking Specs in a non-snaking Dimension "
                    f"when they do not repeat an even number of times "
                    f"otherwise {dim.keys()} would jump in position"
                )
    return squashed


class Path:
    def __init__(
        self, dimensions: List[Dimension], start: int = 0, num: int = None,
    ):
        self.sizes = np.array([len(dim) for dim in dimensions])
        if num is None:
            num = np.product(self.sizes)
        self.dimensions = dimensions
        self.index = start
        self.end_index = start + num

    def consume(self, num: int = None) -> Dimension:
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
            repeats = np.product(self.sizes[i + 1 :])
            # How big is this dim: Z:2, Y:3, X:4
            dim_len = self.sizes[i]
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
        """Number of points in a scan"""
        return self.end_index - self.index


class SpecPositions:
    """Backwards compatibility with Cycler"""

    def __init__(self, dimensions: List[Dimension]):
        self.dimensions = dimensions

    @property
    def keys(self) -> List:
        keys = []
        for dim in self.dimensions:
            keys += dim.keys()
        return keys

    def __len__(self) -> int:
        return np.product([len(dim) for dim in self.dimensions])

    def __iter__(self) -> Iterator[Positions]:
        path = Path(self.dimensions)
        while len(path):
            dim = path.consume(1)
            yield {k: dim.positions[k][0] for k in dim.keys()}
