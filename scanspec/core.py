from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List

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
    def keys(self):
        return self.positions.keys()

    def __len__(self) -> int:
        # All positions arrays are same length, pick the first one
        return len(list(self.positions.values())[0])

    def __add__(self, other: "Dimension") -> "Dimension":
        """Zip them together"""
        assert isinstance(other, Dimension), f"Can only add a Dimension, not {other}"
        # rely on the contstructor to check lengths
        dim = Dimension(
            positions={**self.positions, **other.positions},
            lower={**self.lower, **other.lower},
            upper={**self.upper, **other.upper},
        )
        return dim

    def _for_each_array(self, func):
        def apply_func(d):
            for k, arr in d.items():
                d[k] = func(arr)

        apply_func(self.positions)
        if self.lower is not self.positions:
            apply_func(self.lower)
        if self.upper is not self.positions:
            apply_func(self.upper)

    def tile(self, reps: int):
        self._for_each_array(lambda arr: np.tile(arr, reps))

    def repeat(self, reps: int):
        self._for_each_array(lambda arr: np.repeat(arr, reps))


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
        self,
        dimensions: List[Dimension],
        start: int = 0,
        num: int = None,
        default_batch: int = 1000,
    ):
        self.sizes = np.array([len(dim) for dim in dimensions])
        for i, dim in enumerate(dimensions):
            # A mid dimension above an snaking one must have even size to make
            # iterating over them easier
            if i > 2 and dim.snake:
                assert (
                    self.sizes[i - 1] % 2 == 0
                ), "Mid dimensions above a snaking one must have even size"
        if num is None:
            num = np.product(self.sizes)
        self.dimensions = dimensions
        self.index = start
        self.end_index = start + num
        self.default_batch = default_batch

    def get_batch(self, num: int) -> Batch:
        batch = Batch()
        indices = np.arange(self.index, min(self.index + num, self.end_index))
        self.index = indices[-1] + 1
        for i, dim in enumerate(self.dimensions):
            each_point_repeats = np.product(self.sizes[i + 1 :])
            dim_indices = indices // each_point_repeats
            dim_run = indices % each_point_repeats
            backwards = (dim_run % 2 == 1) & dim.snake
            src_indices = np.where(backwards, len(dim) - 1 - dim_indices, dim_indices)
            for key in dim.keys:
                batch.positions[key] = dim.positions[key][src_indices]
                batch.lower[key] = np.where(
                    backwards, dim.upper[key][src_indices], dim.lower[key][src_indices]
                )
                batch.upper[key] = np.where(
                    backwards, dim.lower[key][src_indices], dim.upper[key][src_indices]
                )
        return batch

    def __len__(self) -> int:
        """Number of points in a scan"""
        return self.end_index - self.index

    def __iter__(self) -> Iterator[Positions]:
        # Fixed size batch iterator of positions
        while self.index < self.end_index:
            batch = self.get_batch(self.default_batch)
            for i in range(len(batch)):
                yield {k: batch.positions[k][i] for k in batch.keys}
