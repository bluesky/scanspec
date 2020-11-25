from typing import Any, Callable, Dict, List, Optional, Union, cast

import numpy as np
from pydantic import Field, parse_raw_as, validate_arguments

from scanspec.regions import Region

from .core import Dimension, View, WithType


class Spec(WithType):
    def get_keys(self) -> List:
        raise NotImplementedError(self)

    def create_dimensions(self, bounds=True) -> List[Dimension]:
        raise NotImplementedError(self)

    def create_view(self) -> View:
        """Create view of dimensions that can iterate through points"""
        dims = self.create_dimensions()
        view = View(dims)
        return view

    @property
    def keys(self) -> List:
        return self.get_keys()

    def __mul__(self, other) -> "Spec":
        """Outer product of two Specs"""
        if isinstance(other, Spec):
            return Product(self, other)
        else:
            return NotImplemented

    def __add__(self, other) -> "Spec":
        """Zip together"""
        if isinstance(other, Spec):
            return Zip(self, other)
        else:
            return NotImplemented


class Squash(Spec):
    spec: Spec

    def get_keys(self) -> List:
        return self.spec.keys

    def create_dimensions(self, bounds=True) -> List[Dimension]:
        dims = self.spec.create_dimensions(bounds)
        lengths = [len(dim) for dim in dims]
        ret = dims[0].repeat(np.product(lengths[1:]))
        for i, dim in enumerate(dims):
            if i > 0:
                ntile = np.product(lengths[:i])
                nrepeats = np.product(lengths[i + 1 :])
                ret = ret + dim.repeat(nrepeats).tile(ntile)
        return [ret]


class Mask(Spec):
    spec: Spec
    region: Region

    def get_keys(self) -> List:
        return self.spec.keys

    def create_dimensions(self, bounds=True) -> List[Dimension]:
        # Squash the dimensions together
        dim = Squash(self.spec).create_dimensions(bounds)[0]
        # Generate masks from the positions showing what's inside
        mask = self.region.mask(dim.positions)
        masked_dim = dim.mask(mask)
        return [masked_dim]


class Zip(Spec):
    left: Spec
    right: Spec

    def get_keys(self) -> List:
        return self.left.keys + self.right.keys

    def create_dimensions(self, bounds=True) -> List[Dimension]:
        dims_left = self.left.create_dimensions(bounds)
        dims_right = self.right.create_dimensions(bounds)

        def _pad_dims(
            dims: List[Dimension], others: List[Dimension]
        ) -> List[Optional[Dimension]]:
            # Special case, if only one dim with size 1, expand to the right size
            if len(dims) == 1 and len(dims[0]) == 1:
                dims = [dims[0].repeat(len(others[-1]))]
            # Left pad the dims with Nones so they are the same size
            nones: List[Optional[Dimension]] = [None] * max(len(dims) - len(others), 0)
            return nones + cast(List[Optional[Dimension]], dims)

        # Pad and expand them
        padded_left = _pad_dims(dims_left, dims_right)
        padded_right = _pad_dims(dims_right, dims_left)

        # Work through, zipping them together one by one
        dimensions = []
        for dim_left, dim_right in zip(padded_left, padded_right):
            if dim_left is None:
                dim = dim_right
            elif dim_right is None:
                dim = dim_left
            else:
                dim = dim_left + dim_right
            assert isinstance(
                dim, Dimension
            ), f"Padding went wrong {padded_left} {padded_right}"
            dimensions.append(dim)
        return dimensions


class Product(Spec):
    outer: Spec = Field(..., description="Will be executed once")
    inner: Spec = Field(..., description="Will be executed len(outer) times")

    def get_keys(self) -> List:
        return self.outer.keys + self.inner.keys

    def create_dimensions(self, bounds=True) -> List[Dimension]:
        dims_outer = self.outer.create_dimensions(bounds=False)
        dims_inner = self.inner.create_dimensions(bounds)
        return dims_outer + dims_inner


class Snake(Spec):
    spec: Spec

    def get_keys(self) -> List:
        return self.spec.keys

    def create_dimensions(self, bounds=True) -> List[Dimension]:
        dims = self.spec.create_dimensions(bounds)
        for dim in dims:
            dim.snake = True
        return dims


class Concat(Spec):
    left: Spec
    right: Spec

    def get_keys(self) -> List:
        assert self.left.keys == self.right.keys, "Keys don't match"
        return self.left.keys

    def create_dimensions(self, bounds=True) -> List[Dimension]:
        dims_left = self.left.create_dimensions(bounds)
        dims_right = self.right.create_dimensions(bounds)
        assert len(dims_right) == len(
            dims_left
        ), f"Specs {self.left} and {self.right} don't have same number of dimensions"
        dimensions = []
        for dim_left, dim_right in zip(dims_left, dims_right):
            dimensions.append(dim_left.concat(dim_right))
        return dimensions


def create_dimensions(
    func: Callable[[np.ndarray], Dict[Any, np.ndarray]],
    keys: List,
    num: int,
    bounds: bool,
) -> List[Dimension]:
    positions_calc = func(np.linspace(0.5, num - 0.5, num))
    positions = {k: positions_calc[k] for k in keys}
    if bounds:
        bounds_calc = func(np.linspace(0, num, num + 1))
        lower = {k: bounds_calc[k][:-1] for k in keys}
        upper = {k: bounds_calc[k][1:] for k in keys}
        dimension = Dimension(positions, lower, upper)
    else:
        dimension = Dimension(positions)
    return [dimension]


class Static(Spec):
    key: Any
    value: float = Field(..., description="The value at each point")
    num: int = Field(1, ge=1, description="How many times to repeat this point")

    def get_keys(self) -> List:
        return [self.key]

    def _repeat(self, indexes: np.ndarray) -> Dict[Any, np.ndarray]:
        return {self.key: np.full(len(indexes), self.value)}

    def create_dimensions(self, bounds=True) -> List[Dimension]:
        return create_dimensions(self._repeat, self.keys, self.num, bounds)


class Line(Spec):
    """Thing"""

    # TODO: are start and stop positions, bounds, or different for fly/step
    key: Any
    start: float = Field(..., description="Centre point of the first point of the line")
    stop: float = Field(..., description="Centre point of the last point of the line")
    num: int = Field(..., ge=1, description="Number of points to produce")

    def get_keys(self) -> List:
        return [self.key]

    def _line(self, indexes: np.ndarray) -> Dict[Any, np.ndarray]:
        if self.num == 1:
            # Only one point, stop-start gives length of one point
            step = self.stop - self.start
        else:
            # Multiple points, stop-start gives length of num-1 points
            step = (self.stop - self.start) / (self.num - 1)
        return {self.key: (indexes - 0.5) * step + self.start}

    def create_dimensions(self, bounds=True) -> List[Dimension]:
        return create_dimensions(self._line, self.keys, self.num, bounds)

    @classmethod
    @validate_arguments
    def bounded(
        cls,
        key,
        lower: float = Field(
            ..., description="Lower bound of the first point of the line"
        ),
        upper: float = Field(
            ..., description="Upper bound of the last point of the line"
        ),
        num: int = num,
    ):
        """Specify instance by extreme bounds"""
        half_step = (upper - lower) / num / 2
        start = lower + half_step
        if num == 1:
            # One point, stop will only be used for step size
            stop = upper + half_step
        else:
            # Many points, stop will be produced
            stop = upper - half_step
        return cls(key, start, stop, num)


class Spiral(Spec):
    x_key: Any
    y_key: Any
    x_start: float = Field(..., description="x centre of the spiral")
    y_start: float = Field(..., description="y centre of the spiral")
    x_range: float = Field(..., description="x width of the spiral")
    y_range: float = Field(..., description="y width of the spiral")
    num: int = Field(..., description="Number of points in the spiral")
    rotate: float = Field(0.0, description="How much to rotate the angle of the spiral")

    def get_keys(self) -> List:
        # TODO: reversed from __init__ args, a good idea?
        return [self.y_key, self.x_key]

    def _spiral(self, indexes: np.ndarray) -> Dict[Any, np.ndarray]:
        # simplest spiral equation: r = phi
        # we want point spacing across area to be the same as between rings
        # so: sqrt(area / num) = ring_spacing
        # so: sqrt(pi * phi^2 / num) = 2 * pi
        # so: phi = sqrt(4 * pi * num)
        phi = np.sqrt(4 * np.pi * indexes)
        # indexes are 0..num inclusive, and diameter is 2x biggest phi
        diameter = 2 * np.sqrt(4 * np.pi * self.num)
        # scale so that the spiral is strictly smaller than the range
        x_scale = self.x_range / diameter
        y_scale = self.y_range / diameter
        return {
            self.y_key: self.y_start + y_scale * phi * np.cos(phi + self.rotate),
            self.x_key: self.x_start + x_scale * phi * np.sin(phi + self.rotate),
        }

    def create_dimensions(self, bounds=True) -> List[Dimension]:
        return create_dimensions(self._spiral, self.keys, self.num, bounds)

    @classmethod
    def spaced(
        cls,
        x_key,
        y_key,
        x_start: float,
        y_start: float,
        radius: float,
        dr: float,
        rotate: float = 0.0,
    ):
        # phi = sqrt(4 * pi * num)
        # and: n_rings = phi / (2 * pi)
        # so: n_rings * 2 * pi = sqrt(4 * pi * num)
        # so: num = n_rings^2 * pi
        n_rings = radius / dr
        num = n_rings ** 2 * np.pi
        return cls(x_key, y_key, x_start, y_start, radius, radius, num, rotate)


class _UnionModifier:
    # Modifies all Spec subclasses so Spec->Union[all Spec subclasses]
    def __init__(self):
        _spec_subclasses = tuple(self._all_subclasses(Spec))
        _region_subclasses = tuple(self._all_subclasses(Region))
        self.spec_union = Union[_spec_subclasses]  # type: ignore
        self.region_union = Union[_region_subclasses]  # type: ignore

        for spec in _spec_subclasses + _region_subclasses:
            for _, field in spec.__fields__.items():
                if field.type_ is Spec:
                    # TODO: Is it better to create a new field than modify?
                    field.type_ = self.spec_union
                    field.prepare()
                elif field.type_ is Region:
                    field.type_ = self.region_union
                    field.prepare()

    def _all_subclasses(self, cls):
        return set(cls.__subclasses__()).union(
            [s for c in cls.__subclasses__() for s in self._all_subclasses(c)]
        )


_modifier = _UnionModifier()


def spec_from_json(text: str) -> Spec:
    return parse_raw_as(_modifier.spec_union, text)  # type: ignore
