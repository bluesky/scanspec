from typing import Any, Callable, Dict, List, Optional, Union, cast

import numpy as np
from pydantic import Field, parse_obj_as, parse_raw_as, validate_arguments

from .core import (
    Dimension,
    Path,
    SpecPositions,
    WithType,
    if_instance_do,
    squash_dimensions,
)
from .regions import Region, get_mask


class Spec(WithType):
    def keys(self) -> List:
        raise NotImplementedError(self)

    def create_dimensions(self, bounds=True, nested=False) -> List[Dimension]:
        raise NotImplementedError(self)

    def path(self) -> Path:
        path = Path(self.create_dimensions())
        return path

    def positions(self) -> SpecPositions:
        sp = SpecPositions(self.create_dimensions(bounds=False))
        return sp

    def __add__(self, other) -> "Zip":
        """Zip together"""
        return if_instance_do(other, Spec, lambda o: Zip(self, o))

    def __mul__(self, other) -> "Product":
        """Outer product of two Specs"""
        return if_instance_do(other, Spec, lambda o: Product(self, o))

    def __invert__(self) -> "Snake":
        """Snake this spec"""
        return Snake(self)

    def __and__(self, other) -> "Mask":
        """Mask with a region"""
        return if_instance_do(other, Region, lambda o: Mask(self, o))


class Zip(Spec):
    left: Spec
    right: Spec

    def keys(self) -> List:
        return self.left.keys() + self.right.keys()

    def create_dimensions(self, bounds=True, nested=False) -> List[Dimension]:
        dims_left = self.left.create_dimensions(bounds, nested)
        dims_right = self.right.create_dimensions(bounds, nested)

        def _pad_dims(
            dims: List[Dimension], others: List[Dimension]
        ) -> List[Optional[Dimension]]:
            # Special case, if only one dim with size 1, expand to the right size
            if len(dims) == 1 and len(dims[0]) == 1:
                repeated = dims[0].repeat(len(others[-1]))
                repeated.snake = others[-1].snake
                dims = [repeated]
            # Left pad the dims with Nones so they are the same size
            npad = max(len(dims), len(others)) - len(dims)
            nones: List[Optional[Dimension]] = [None] * npad
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

    def keys(self) -> List:
        return self.outer.keys() + self.inner.keys()

    def create_dimensions(self, bounds=True, nested=False) -> List[Dimension]:
        dims_outer = self.outer.create_dimensions(bounds=False, nested=nested)
        dims_inner = self.inner.create_dimensions(bounds, nested=True)
        return dims_outer + dims_inner


class Snake(Spec):
    spec: Spec

    def keys(self) -> List:
        return self.spec.keys()

    def create_dimensions(self, bounds=True, nested=False) -> List[Dimension]:
        dims = self.spec.create_dimensions(bounds, nested)
        for dim in dims:
            dim.snake = True
        return dims


class Mask(Spec):
    spec: Spec
    region: Region
    check_path_changes: bool = True

    def keys(self) -> List:
        return self.spec.keys()

    def create_dimensions(self, bounds=True, nested=False) -> List[Dimension]:
        dims = self.spec.create_dimensions(bounds, nested)
        for key_set in self.region.key_sets():
            # Squash the dimensions together containing these keys
            matches = [i for i, d in enumerate(dims) if set(d.keys()) & key_set]
            assert matches, f"No Specs match keys {list(key_set)}"
            si, ei = matches[0], matches[-1]
            if si != ei:
                # Span Specs, squash them
                # If the spec to be squashed is nested (inside the Mask or outside)
                # then check the path changes if requested
                check_path_changes = (nested or si) and self.check_path_changes
                squashed = squash_dimensions(dims[si : ei + 1], check_path_changes)
                dims = dims[:si] + [squashed] + dims[ei + 1 :]
        # Generate masks from the positions showing what's inside
        masked_dims = [dim.mask(get_mask(self.region, dim.positions)) for dim in dims]
        return masked_dims

    # *+ bind more tightly than &|^ so without these overrides we
    # would need to add brackets to all combinations of Regions
    def __or__(self, other: "Region") -> "Mask":
        return if_instance_do(other, Region, lambda o: Mask(self.spec, self.region | o))

    def __and__(self, other: "Region") -> "Mask":
        return if_instance_do(other, Region, lambda o: Mask(self.spec, self.region & o))

    def __xor__(self, other: "Region") -> "Mask":
        return if_instance_do(other, Region, lambda o: Mask(self.spec, self.region ^ o))

    # This is here for completeness, tends not to be called as - binds
    # tighter than &
    def __sub__(self, other: "Region") -> "Mask":
        return if_instance_do(other, Region, lambda o: Mask(self.spec, self.region - o))


class Concat(Spec):
    left: Spec
    right: Spec

    def keys(self) -> List:
        left_keys, right_keys = self.left.keys(), self.right.keys()
        assert left_keys == right_keys, f"Keys {left_keys} != {right_keys}"
        return left_keys

    def create_dimensions(self, bounds=True, nested=False) -> List[Dimension]:
        dims_left = self.left.create_dimensions(bounds, nested)
        dims_right = self.right.create_dimensions(bounds, nested)
        assert len(dims_right) == len(
            dims_left
        ), f"Specs {self.left} and {self.right} don't have same number of dimensions"
        dimensions = []
        for dim_left, dim_right in zip(dims_left, dims_right):
            dimensions.append(dim_left.concat(dim_right))
        return dimensions


class Squash(Spec):
    """Squash the Dimensions together of the scan (but not positions) into one
    long line.

    .. example_spec::

        from scanspec.specs import Line, Squash

        spec = Squash(Line("y", 1, 2, 3) * Line("x", 0, 1, 4))
    """

    spec: Spec = Field(..., description="The Spec to squash the dimensions of")
    check_path_changes: bool = Field(
        True, description="If True path through scan will not be modified by squash"
    )

    def keys(self) -> List:
        return self.spec.keys()

    def create_dimensions(self, bounds=True, nested=False) -> List[Dimension]:
        # TODO: if we squash we explode the size, can we avoid this?
        dims = self.spec.create_dimensions(bounds, nested)
        dim = squash_dimensions(dims, nested and self.check_path_changes)
        return [dim]


def _dimensions_from_indexes(
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


class Line(Spec):
    """Linearly spaced points in the given key, with first and last points
    centred on start and stop.

    .. example_spec::

        from scanspec.specs import Line

        spec = Line("x", 1, 2, 5)
    """

    # TODO: are start and stop positions, bounds, or different for fly/step
    key: Any = Field(..., description="An identifier for what to move")
    start: float = Field(..., description="Centre point of the first point of the line")
    stop: float = Field(..., description="Centre point of the last point of the line")
    num: int = Field(..., ge=1, description="Number of points to produce")

    def keys(self) -> List:
        return [self.key]

    def _line_from_indexes(self, indexes: np.ndarray) -> Dict[Any, np.ndarray]:
        if self.num == 1:
            # Only one point, stop-start gives length of one point
            step = self.stop - self.start
        else:
            # Multiple points, stop-start gives length of num-1 points
            step = (self.stop - self.start) / (self.num - 1)
        return {self.key: (indexes - 0.5) * step + self.start}

    def create_dimensions(self, bounds=True, nested=False) -> List[Dimension]:
        return _dimensions_from_indexes(
            self._line_from_indexes, self.keys(), self.num, bounds
        )

    @classmethod
    @validate_arguments
    def bounded(
        cls,
        key=key,
        lower: float = Field(
            ..., description="Lower bound of the first point of the line"
        ),
        upper: float = Field(
            ..., description="Upper bound of the last point of the line"
        ),
        num: int = num,
    ):
        """Specify a Line by extreme bounds instead of centre points.

        .. example_spec::

            from scanspec.specs import Line

            spec = Line.bounded("x", 1, 2, 5)
        """
        half_step = (upper - lower) / num / 2
        start = lower + half_step
        if num == 1:
            # One point, stop will only be used for step size
            stop = upper + half_step
        else:
            # Many points, stop will be produced
            stop = upper - half_step
        return cls(key, start, stop, num)


class Static(Spec):
    """A static point, repeated "num" times, with "key" at "value". Can
    be used to set key=value at every point in a scan.

    .. example_spec::

        from scanspec.specs import Line, Static

        spec = Line("y", 1, 2, 3) + Static("x", 3)
    """

    key: Any = Field(..., description="An identifier for what to move")
    value: float = Field(..., description="The value at each point")
    num: int = Field(1, ge=1, description="How many times to repeat this point")

    def keys(self) -> List:
        return [self.key]

    def _repeats_from_indexes(self, indexes: np.ndarray) -> Dict[Any, np.ndarray]:
        return {self.key: np.full(len(indexes), self.value)}

    def create_dimensions(self, bounds=True, nested=False) -> List[Dimension]:
        return _dimensions_from_indexes(
            self._repeats_from_indexes, self.keys(), self.num, bounds
        )


class Spiral(Spec):
    """Archimedean spiral of "x_key" and "y_key", starting at centre point
    ("x_start", "y_start") with angle "rotate". Produces "num" points
    in a spiral spanning width of "x_range" and height of "y_range"

    .. example_spec::

        from scanspec.specs import Spiral

        spec = Spiral("x", "y", 1, 5, 10, 50, 30)
    """

    x_key: Any = Field(..., description="An identifier for what to move for x")
    y_key: Any = Field(..., description="An identifier for what to move for y")
    # TODO: do we like these names?
    x_start: float = Field(..., description="x centre of the spiral")
    y_start: float = Field(..., description="y centre of the spiral")
    x_range: float = Field(..., description="x width of the spiral")
    y_range: float = Field(..., description="y width of the spiral")
    num: int = Field(..., description="Number of points in the spiral")
    rotate: float = Field(0.0, description="How much to rotate the angle of the spiral")

    def keys(self) -> List:
        # TODO: reversed from __init__ args, a good idea?
        return [self.y_key, self.x_key]

    def _spiral_from_indexes(self, indexes: np.ndarray) -> Dict[Any, np.ndarray]:
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

    def create_dimensions(self, bounds=True, nested=False) -> List[Dimension]:
        return _dimensions_from_indexes(
            self._spiral_from_indexes, self.keys(), self.num, bounds
        )

    @classmethod
    @validate_arguments
    def spaced(
        cls,
        x_key: Any = x_key,
        y_key: Any = y_key,
        x_start: float = x_start,
        y_start: float = y_start,
        radius: float = Field(..., description="radius of the spiral"),
        dr: float = Field(..., description="difference between each ring"),
        rotate: float = rotate,
    ):
        """Specify a Spiral equally spaced in "x_key" and "y_key" by specifying
        the "radius" and difference between each ring of the spiral "dr"

        .. example_spec::

            from scanspec.specs import Spiral

            spec = Spiral.spaced("x", "y", 0, 0, 10, 3, 0)
        """
        # phi = sqrt(4 * pi * num)
        # and: n_rings = phi / (2 * pi)
        # so: n_rings * 2 * pi = sqrt(4 * pi * num)
        # so: num = n_rings^2 * pi
        n_rings = radius / dr
        num = n_rings ** 2 * np.pi
        return cls(x_key, y_key, x_start, y_start, radius, radius, num, rotate)


#: Can be used as a special key to indicate how long each point should be
TIME = "TIME"


def fly(spec: Spec, duration: float):
    """Flyscan, zipping TIME=duration for every point"""
    return spec + Static(TIME, duration)


def step(spec: Spec, duration: float):
    """Step scan, adding TIME=duration as an inner dimension for every point"""
    return spec * Static(TIME, duration)


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


def spec_from_dict(d: Dict) -> Spec:
    return parse_obj_as(_modifier.spec_union, d)  # type: ignore


def spec_from_json(text: str) -> Spec:
    return parse_raw_as(_modifier.spec_union, text)  # type: ignore
