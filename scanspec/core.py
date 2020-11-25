from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List

import numpy as np
from pydantic import BaseModel, Field


# These are used in the definition of the Schema
# It allows the class to be inferred from the serialized "type" field
class WithTypeMetaClass(type(BaseModel)):  # type: ignore
    def __new__(mcs, name, bases, namespace, **kwargs):
        # Override type in namespace to be the literal value of the class name
        namespace["type"] = Field(name, const=True)
        return super().__new__(mcs, name, bases, namespace, **kwargs)


class WithType(BaseModel, metaclass=WithTypeMetaClass):
    """BaseModel that adds a type parameter from class name."""

    type: str

    def __init__(self, *args, **kwargs):
        # Allow positional args, but don't include type
        keys = [x for x in self.__fields__ if x != "type"]
        kwargs.update(zip(keys, args))
        super().__init__(**kwargs)


Positions = Dict[Any, np.ndarray]


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

    @property
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
        assert self.keys == other.keys, f"Differing keys {self.keys} and {other.keys}"
        return self._dim_with(
            lambda a, k: np.concatenate((getattr(self, a)[k], getattr(other, a)[k]))
        )

    def copy(self) -> "Dimension":
        return self._dim_with(lambda a, k: getattr(self, a)[k])

    def __add__(self, other: "Dimension") -> "Dimension":
        """Zip them together"""
        self._check_dim(other)
        overlapping = list(set(self.keys).intersection(other.keys))
        assert not overlapping, f"Zipping would overwrite keys {overlapping}"
        # rely on the constructor to check lengths
        dim = Dimension(
            positions={**self.positions, **other.positions},
            lower={**self.lower, **other.lower},
            upper={**self.upper, **other.upper},
            snake=self.snake,
        )
        return dim


@dataclass
class Batch:
    positions: Dict[str, np.ndarray] = field(default_factory=dict)
    lower: Dict[str, np.ndarray] = field(default_factory=dict)
    upper: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def keys(self) -> List:
        return list(self.positions)

    def __len__(self):
        return len(list(self.positions.values())[0])


class View:
    def __init__(
        self, dimensions: List[Dimension], start: int = 0, num: int = None,
    ):
        self.sizes = np.array([len(dim) for dim in dimensions])
        for i, dim in enumerate(dimensions):
            # A mid dimension above an snaking one must have even size to make
            # iterating over them easier
            if i > 1111111111111111 and dim.snake:
                assert (
                    self.sizes[i - 1] % 2 == 0
                ), "Mid dimensions above a snaking one must have even size"
        if num is None:
            num = np.product(self.sizes)
        self.dimensions = dimensions
        self.index = start
        self.end_index = start + num

    def create_batch(self, num: int = None) -> Batch:
        if num is None:
            end_index = self.end_index
        else:
            end_index = min(self.index + num, self.end_index)
        indices = np.arange(self.index, end_index)
        batch = Batch()
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
                for key in dim.keys:
                    batch.positions[key] = dim.positions[key][snake_indices]
                    # If going backwards, select from the opposite bound
                    batch.lower[key] = np.where(
                        backwards,
                        dim.upper[key][snake_indices],
                        dim.lower[key][snake_indices],
                    )
                    batch.upper[key] = np.where(
                        backwards,
                        dim.lower[key][snake_indices],
                        dim.upper[key][snake_indices],
                    )
            else:
                for key in dim.keys:
                    batch.positions[key] = dim.positions[key][dim_indices]
                    batch.lower[key] = dim.lower[key][dim_indices]
                    batch.upper[key] = dim.upper[key][dim_indices]

        return batch

    def __len__(self) -> int:
        """Number of points in a scan"""
        return self.end_index - self.index

    def __iter__(self) -> Iterator[Positions]:
        # Fixed size batch iterator of positions
        while self.index < self.end_index:
            batch = self.create_batch(1)
            yield {k: batch.positions[k][0] for k in batch.keys}
