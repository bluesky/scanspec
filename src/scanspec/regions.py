"""`Region` and its subclasses.

.. inheritance-diagram:: scanspec.regions
    :top-classes: scanspec.regions.Region
    :parts: 1
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import is_dataclass
from typing import Any, Generic, cast

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, TypeAdapter
from pydantic.dataclasses import dataclass

from .core import (
    AxesPoints,
    Axis,
    StrictConfig,
    discriminated_union_of_subclasses,
    if_instance_do,
)

__all__ = [
    "Region",
    "get_mask",
    "CombinationOf",
    "UnionOf",
    "IntersectionOf",
    "DifferenceOf",
    "SymmetricDifferenceOf",
    "Range",
    "Rectangle",
    "Polygon",
    "Circle",
    "Ellipse",
    "find_regions",
]

NpMask = npt.NDArray[np.bool_]


@discriminated_union_of_subclasses
class Region(Generic[Axis]):
    """Abstract baseclass for a Region that can `Mask` a `Spec`.

    Supports operators:

    - ``|``: `UnionOf` two Regions, midpoints present in either
    - ``&``: `IntersectionOf` two Regions, midpoints present in both
    - ``-``: `DifferenceOf` two Regions, midpoints present in first not second
    - ``^``: `SymmetricDifferenceOf` two Regions, midpoints present in one not both
    """

    def axis_sets(self) -> list[set[Axis]]:  # noqa: D102
        """Produce the non-overlapping sets of axes this region spans."""
        raise NotImplementedError(self)

    def mask(self, points: AxesPoints[Axis]) -> NpMask:  # noqa: D102
        """Produce a mask of which points are in the region."""
        raise NotImplementedError(self)

    def __or__(self, other: Region[Axis]) -> UnionOf[Axis]:
        return if_instance_do(other, Region, lambda o: UnionOf(self, o))

    def __and__(self, other: Region[Axis]) -> IntersectionOf[Axis]:
        return if_instance_do(other, Region, lambda o: IntersectionOf(self, o))

    def __sub__(self, other: Region[Axis]) -> DifferenceOf[Axis]:
        return if_instance_do(other, Region, lambda o: DifferenceOf(self, o))

    def __xor__(self, other: Region[Axis]) -> SymmetricDifferenceOf[Axis]:
        return if_instance_do(other, Region, lambda o: SymmetricDifferenceOf(self, o))

    def serialize(self) -> Mapping[str, Any]:
        """Serialize the Region to a dictionary."""
        return TypeAdapter(Region[Any]).dump_python(self)

    @staticmethod
    def deserialize(obj: Any) -> Region[Any]:
        """Deserialize a Region from a dictionary."""
        return TypeAdapter(Region[Any]).validate_python(obj)


def get_mask(region: Region[Axis], points: AxesPoints[Axis]) -> NpMask:
    """Return a mask of the points inside the region.

    If there is an overlap of axes of region and points return a
    mask of the points in the region, otherwise return all ones
    """
    axes = set(points)
    needs_mask = any(ks & axes for ks in region.axis_sets())
    if needs_mask:
        return region.mask(points)
    else:
        return np.ones(len(list(points.values())[0]), dtype=np.bool_)


def _merge_axis_sets(axis_sets: list[set[Axis]]) -> Iterator[set[Axis]]:
    # Take overlapping axis sets and merge any that overlap into each
    # other
    for ks in axis_sets:  # ks = key_sets - left over from a previous naming standard
        axis_set = ks.copy()
        # Empty matching sets into this axis_set
        for ks in axis_sets:
            if ks & axis_set:
                while ks:
                    axis_set.add(ks.pop())
        # It might be emptied already, only yield if it isn't
        if axis_set:
            yield axis_set


@dataclass(config=StrictConfig)
class CombinationOf(Region[Axis]):
    """Abstract baseclass for a combination of two regions, left and right."""

    left: Region[Axis] = Field(description="The left-hand Region to combine")
    right: Region[Axis] = Field(description="The right-hand Region to combine")

    def axis_sets(self) -> list[set[Axis]]:  # noqa: D102
        axis_sets = list(
            _merge_axis_sets(self.left.axis_sets() + self.right.axis_sets())
        )
        return axis_sets


# Naming so we don't clash with typing.Union
@dataclass(config=StrictConfig)
class UnionOf(CombinationOf[Axis]):
    """A point is in UnionOf(a, b) if in either a or b.

    Typically created with the ``|`` operator

    >>> r = Range("x", 0.5, 2.5) | Range("x", 1.5, 3.5)
    >>> r.mask({"x": np.array([0, 1, 2, 3, 4])})
    array([False,  True,  True,  True, False])
    """

    def mask(self, points: AxesPoints[Axis]) -> NpMask:  # noqa: D102
        mask = get_mask(self.left, points) | get_mask(self.right, points)
        return mask


@dataclass(config=StrictConfig)
class IntersectionOf(CombinationOf[Axis]):
    """A point is in IntersectionOf(a, b) if in both a and b.

    Typically created with the ``&`` operator.

    >>> r = Range("x", 0.5, 2.5) & Range("x", 1.5, 3.5)
    >>> r.mask({"x": np.array([0, 1, 2, 3, 4])})
    array([False, False,  True, False, False])
    """

    def mask(self, points: AxesPoints[Axis]) -> NpMask:  # noqa: D102
        mask = get_mask(self.left, points) & get_mask(self.right, points)
        return mask


@dataclass(config=StrictConfig)
class DifferenceOf(CombinationOf[Axis]):
    """A point is in DifferenceOf(a, b) if in a and not in b.

    Typically created with the ``-`` operator.

    >>> r = Range("x", 0.5, 2.5) - Range("x", 1.5, 3.5)
    >>> r.mask({"x": np.array([0, 1, 2, 3, 4])})
    array([False,  True, False, False, False])
    """

    def mask(self, points: AxesPoints[Axis]) -> NpMask:  # noqa: D102
        left_mask = get_mask(self.left, points)
        # Return the xor restricted to the left region
        mask = left_mask ^ get_mask(self.right, points) & left_mask
        return mask


@dataclass(config=StrictConfig)
class SymmetricDifferenceOf(CombinationOf[Axis]):
    """A point is in SymmetricDifferenceOf(a, b) if in either a or b, but not both.

    Typically created with the ``^`` operator.

    >>> r = Range("x", 0.5, 2.5) ^ Range("x", 1.5, 3.5)
    >>> r.mask({"x": np.array([0, 1, 2, 3, 4])})
    array([False,  True, False,  True, False])
    """

    def mask(self, points: AxesPoints[Axis]) -> NpMask:  # noqa: D102
        mask = get_mask(self.left, points) ^ get_mask(self.right, points)
        return mask


@dataclass(config=StrictConfig)
class Range(Region[Axis]):
    """Mask contains points of axis >= min and <= max.

    >>> r = Range("x", 1, 2)
    >>> r.mask({"x": np.array([0, 1, 2, 3, 4])})
    array([False,  True,  True, False, False])
    """

    axis: Axis = Field(description="The name matching the axis to mask in spec")
    min: float = Field(description="The minimum inclusive value in the region")
    max: float = Field(description="The minimum inclusive value in the region")

    def axis_sets(self) -> list[set[Axis]]:  # noqa: D102
        return [{self.axis}]

    def mask(self, points: AxesPoints[Axis]) -> NpMask:  # noqa: D102
        v = points[self.axis]
        mask = np.bitwise_and(v >= self.min, v <= self.max)
        return mask


@dataclass(config=StrictConfig)
class Rectangle(Region[Axis]):
    """Mask contains points of axis within a rotated xy rectangle.

    .. example_spec::

        from scanspec.regions import Rectangle
        from scanspec.specs import Line

        grid = Line("y", 1, 3, 10) * ~Line("x", 0, 2, 10)
        spec = grid & Rectangle("x", "y", 0, 1.1, 1.5, 2.1, 30)
    """

    x_axis: Axis = Field(description="The name matching the x axis of the spec")
    y_axis: Axis = Field(description="The name matching the y axis of the spec")
    x_min: float = Field(description="Minimum inclusive x value in the region")
    y_min: float = Field(description="Minimum inclusive y value in the region")
    x_max: float = Field(description="Maximum inclusive x value in the region")
    y_max: float = Field(description="Maximum inclusive y value in the region")
    angle: float = Field(
        description="Clockwise rotation angle of the rectangle", default=0.0
    )

    def axis_sets(self) -> list[set[Axis]]:  # noqa: D102
        return [{self.x_axis, self.y_axis}]

    def mask(self, points: AxesPoints[Axis]) -> NpMask:  # noqa: D102
        x = points[self.x_axis] - self.x_min
        y = points[self.y_axis] - self.y_min
        if self.angle != 0:
            # Rotate src points by -angle
            phi = np.radians(-self.angle)
            rx = x * np.cos(phi) - y * np.sin(phi)
            ry = x * np.sin(phi) + y * np.cos(phi)
            x = rx
            y = ry
        mask_x = np.bitwise_and(x >= 0, x <= (self.x_max - self.x_min))
        mask_y = np.bitwise_and(y >= 0, y <= (self.y_max - self.y_min))
        return mask_x & mask_y


@dataclass(config=StrictConfig)
class Polygon(Region[Axis]):
    """Mask contains points of axis within a rotated xy polygon.

    .. example_spec::

        from scanspec.regions import Polygon
        from scanspec.specs import Line

        grid = Line("y", 3, 8, 10) * ~Line("x", 1 ,8, 10)
        spec = grid & Polygon("x", "y", [1.0, 6.0, 8.0, 2.0], [4.0, 10.0, 6.0, 1.0])
    """

    x_axis: Axis = Field(description="The name matching the x axis of the spec")
    y_axis: Axis = Field(description="The name matching the y axis of the spec")
    x_verts: list[float] = Field(
        description="The Nx1 x coordinates of the polygons vertices", min_length=3
    )
    y_verts: list[float] = Field(
        description="The Nx1 y coordinates of the polygons vertices", min_length=3
    )

    def axis_sets(self) -> list[set[Axis]]:  # noqa: D102
        return [{self.x_axis, self.y_axis}]

    def mask(self, points: AxesPoints[Axis]) -> NpMask:  # noqa: D102
        x = points[self.x_axis]
        y = points[self.y_axis]
        v1x, v1y = self.x_verts[-1], self.y_verts[-1]
        mask = np.full(len(x), False, dtype=np.bool_)
        for v2x, v2y in zip(self.x_verts, self.y_verts, strict=False):
            # skip horizontal edges
            if v2y != v1y:
                vmask = np.full(len(x), False, dtype=np.bool_)
                vmask |= (y < v2y) & (y >= v1y)
                vmask |= (y < v1y) & (y >= v2y)
                t = (y - v1y) / (v2y - v1y)
                vmask &= x < v1x + t * (v2x - v1x)
                mask ^= vmask
            v1x, v1y = v2x, v2y
        return mask


@dataclass(config=StrictConfig)
class Circle(Region[Axis]):
    """Mask contains points of axis within an xy circle of given radius.

    .. example_spec::

        from scanspec.regions import Circle
        from scanspec.specs import Line

        grid = Line("y", 1, 3, 10) * ~Line("x", 0, 2, 10)
        spec = grid & Circle("x", "y", 1, 2, 0.9)
    """

    x_axis: Axis = Field(description="The name matching the x axis of the spec")
    y_axis: Axis = Field(description="The name matching the y axis of the spec")
    x_middle: float = Field(description="The central x point of the circle")
    y_middle: float = Field(description="The central y point of the circle")
    radius: float = Field(description="Radius of the circle", gt=0.0)

    def axis_sets(self) -> list[set[Axis]]:  # noqa: D102
        return [{self.x_axis, self.y_axis}]

    def mask(self, points: AxesPoints[Axis]) -> NpMask:  # noqa: D102
        x = points[self.x_axis] - self.x_middle
        y = points[self.y_axis] - self.y_middle
        mask = x * x + y * y <= (self.radius * self.radius)
        return mask


@dataclass(config=StrictConfig)
class Ellipse(Region[Axis]):
    """Mask contains points of axis within an xy ellipse of given radius.

    .. example_spec::

        from scanspec.regions import Ellipse
        from scanspec.specs import Line

        grid = Line("y", 3, 8, 10) * ~Line("x", 1 ,8, 10)
        spec = grid & Ellipse("x", "y", 5, 5, 2, 3, 75)
    """

    x_axis: Axis = Field(description="The name matching the x axis of the spec")
    y_axis: Axis = Field(description="The name matching the y axis of the spec")
    x_middle: float = Field(description="The central x point of the ellipse")
    y_middle: float = Field(description="The central y point of the ellipse")
    x_radius: float = Field(
        description="The radius along the x axis of the ellipse", gt=0.0
    )
    y_radius: float = Field(
        description="The radius along the y axis of the ellipse", gt=0.0
    )
    angle: float = Field(description="The angle of the ellipse (degrees)", default=0.0)

    def axis_sets(self) -> list[set[Axis]]:  # noqa: D102
        return [{self.x_axis, self.y_axis}]

    def mask(self, points: AxesPoints[Axis]) -> NpMask:  # noqa: D102
        x = points[self.x_axis] - self.x_middle
        y = points[self.y_axis] - self.y_middle
        if self.angle != 0:
            # Rotate src points by -angle
            phi = np.radians(-self.angle)
            tx = x * np.cos(phi) - y * np.sin(phi)
            ty = x * np.sin(phi) + y * np.cos(phi)
            x = tx
            y = ty
        mask = (x / self.x_radius) ** 2 + (y / self.y_radius) ** 2 <= 1
        return mask


def find_regions(obj: Any) -> Iterator[Region[Any]]:  # noqa: D102
    """Recursively yield Regions from obj and its children."""
    if (
        hasattr(obj, "__pydantic_model__")
        and issubclass(obj.__pydantic_model__, BaseModel)
        or is_dataclass(obj)
    ):
        if isinstance(obj, Region):
            yield obj
        for name in obj.__dict__.keys():
            regions: Iterator[Region[Any]] = find_regions(getattr(cast(Any, obj), name))
            yield from regions
