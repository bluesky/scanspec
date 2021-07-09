from dataclasses import dataclass, is_dataclass
from typing import Iterator, List, Set

import numpy as np
from apischema import schema
from typing_extensions import Annotated as A

from .core import AxesPoints, Serializable, if_instance_do

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


@dataclass
class Region(Serializable):
    """Abstract baseclass for a Region that can `Mask` a `Spec`. Supports operators:

    - ``|``: `UnionOf` two Regions, midpoints present in either
    - ``&``: `IntersectionOf` two Regions, points present in both
    - ``-``: `DifferenceOf` two Regions, points present in first not second
    - ``^``: `SymmetricDifferenceOf` two Regions, points present in one not both
    """

    def axis_sets(self) -> List[Set[str]]:
        """Implemented by subclasses to produce the non-overlapping sets of axes
        this region spans"""
        raise NotImplementedError(self)

    def mask(self, points: AxesPoints) -> np.ndarray:
        """Implemented by subclasses to produce a mask of which points are in
        the region"""
        raise NotImplementedError(self)

    def __or__(self, other) -> "UnionOf":
        return if_instance_do(other, Region, lambda o: UnionOf(self, o))

    def __and__(self, other) -> "IntersectionOf":
        return if_instance_do(other, Region, lambda o: IntersectionOf(self, o))

    def __sub__(self, other) -> "DifferenceOf":
        return if_instance_do(other, Region, lambda o: DifferenceOf(self, o))

    def __xor__(self, other) -> "SymmetricDifferenceOf":
        return if_instance_do(other, Region, lambda o: SymmetricDifferenceOf(self, o))


def get_mask(region: Region, points: AxesPoints) -> np.ndarray:
    """If there is an overlap of axes of region and frames return a
    mask of the frames in the region, otherwise return all ones
    """
    axes = set(points)
    needs_mask = any(ks & axes for ks in region.axis_sets())
    if needs_mask:
        return region.mask(points)
    else:
        return np.ones(len(list(points.values())[0]))


def _merge_axis_sets(axis_sets: List[Set[str]]) -> Iterator[Set[str]]:
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


@dataclass
class CombinationOf(Region):
    """Abstract baseclass for a combination of two regions, left and right"""

    left: A[Region, schema(description="The left-hand Region to combine")]
    right: A[Region, schema(description="The right-hand Region to combine")]

    def axis_sets(self) -> List[Set[str]]:
        axis_sets = list(
            _merge_axis_sets(self.left.axis_sets() + self.right.axis_sets())
        )
        return axis_sets


# Naming so we don't clash with typing.Union
@dataclass
class UnionOf(CombinationOf):
    """A point is in UnionOf(a, b) if it is in either a or b. Typically
    created with the ``|`` operator

    >>> r = Range("x", 0.5, 2.5) | Range("x", 1.5, 3.5)
    >>> r.mask({"x": np.array([0, 1, 2, 3, 4])})
    array([False,  True,  True,  True, False])
    """

    def mask(self, points: AxesPoints) -> np.ndarray:
        mask = get_mask(self.left, points) | get_mask(self.right, points)
        return mask


@dataclass
class IntersectionOf(CombinationOf):
    """A point is in IntersectionOf(a, b) if it is in both a and b. Typically
    created with the ``&`` operator

    >>> r = Range("x", 0.5, 2.5) & Range("x", 1.5, 3.5)
    >>> r.mask({"x": np.array([0, 1, 2, 3, 4])})
    array([False, False,  True, False, False])
    """

    def mask(self, points: AxesPoints) -> np.ndarray:
        mask = get_mask(self.left, points) & get_mask(self.right, points)
        return mask


@dataclass
class DifferenceOf(CombinationOf):
    """A point is in DifferenceOf(a, b) if it is in a and not in b. Typically
    created with the ``-`` operator

    >>> r = Range("x", 0.5, 2.5) - Range("x", 1.5, 3.5)
    >>> r.mask({"x": np.array([0, 1, 2, 3, 4])})
    array([False,  True, False, False, False])
    """

    def mask(self, points: AxesPoints) -> np.ndarray:
        left_mask = get_mask(self.left, points)
        # Return the xor restricted to the left region
        mask = left_mask ^ get_mask(self.right, points) & left_mask
        return mask


@dataclass
class SymmetricDifferenceOf(CombinationOf):
    """A point is in SymmetricDifferenceOf(a, b) if it is in either a or b,
    but not both. Typically created with the ``^`` operator

    >>> r = Range("x", 0.5, 2.5) ^ Range("x", 1.5, 3.5)
    >>> r.mask({"x": np.array([0, 1, 2, 3, 4])})
    array([False,  True, False,  True, False])
    """

    def mask(self, points: AxesPoints) -> np.ndarray:
        mask = get_mask(self.left, points) ^ get_mask(self.right, points)
        return mask


@dataclass
class Range(Region):
    """Mask contains points of key >= min and <= max

    >>> r = Range("x", 1, 2)
    >>> r.mask({"x": np.array([0, 1, 2, 3, 4])})
    array([False,  True,  True, False, False])
    """

    axis: A[str, schema(description="The name matching the axis to mask in spec")]
    min: A[float, schema(description="The minimum inclusive value in the region")]
    max: A[float, schema(description="The minimum inclusive value in the region")]

    def axis_sets(self) -> List[Set[str]]:
        return [{self.axis}]

    def mask(self, points: AxesPoints) -> np.ndarray:
        v = points[self.axis]
        mask = np.bitwise_and(v >= self.min, v <= self.max)
        return mask


@dataclass
class Rectangle(Region):
    """Mask contains points of axis within a rotated xy rectangle

    .. example_spec::

        from scanspec.specs import Line
        from scanspec.regions import Rectangle

        grid = Line("y", 1, 3, 10) * ~Line("x", 0, 2, 10)
        spec = grid & Rectangle("x", "y", 0, 1.1, 1.5, 2.1, 30)
    """

    x_axis: A[str, schema(description="The name matching the x axis of the spec")]
    y_axis: A[str, schema(description="The name matching the y axis of the spec")]
    x_min: A[float, schema(description="Minimum inclusive x value in the region")]
    y_min: A[float, schema(description="Minimum inclusive y value in the region")]
    x_max: A[float, schema(description="Maximum inclusive x value in the region")]
    y_max: A[float, schema(description="Maximum inclusive y value in the region")]
    angle: A[
        float, schema(description="Clockwise rotation angle of the rectangle")
    ] = 0.0

    def axis_sets(self) -> List[Set[str]]:
        return [{self.x_axis, self.y_axis}]

    def mask(self, points: AxesPoints) -> np.ndarray:
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


@dataclass
class Polygon(Region):
    """Mask contains points of axis within a rotated xy polygon

    .. example_spec::

        from scanspec.specs import Line
        from scanspec.regions import Polygon

        grid = Line("y", 3, 8, 10) * ~Line("x", 1 ,8, 10)
        spec = grid & Polygon("x", "y", [1.0, 6.0, 8.0, 2.0], [4.0, 10.0, 6.0, 1.0])
    """

    x_axis: A[str, schema(description="The name matching the x axis of the spec")]
    y_axis: A[str, schema(description="The name matching the y axis of the spec")]
    x_verts: A[
        List[float],
        schema(description="The Nx1 x coordinates of the polygons vertices", min_len=3),
    ]
    y_verts: A[
        List[float],
        schema(description="The Nx1 y coordinates of the polygons vertices", min_len=3),
    ]

    def axis_sets(self) -> List[Set[str]]:
        return [{self.x_axis, self.y_axis}]

    def mask(self, points: AxesPoints) -> np.ndarray:
        x = points[self.x_axis]
        y = points[self.y_axis]
        v1x, v1y = self.x_verts[-1], self.y_verts[-1]
        mask = np.full(len(x), False, dtype=np.int8)
        for v2x, v2y in zip(self.x_verts, self.y_verts):
            # skip horizontal edges
            if v2y != v1y:
                vmask = np.full(len(x), False, dtype=np.int8)
                vmask |= (y < v2y) & (y >= v1y)
                vmask |= (y < v1y) & (y >= v2y)
                t = (y - v1y) / (v2y - v1y)
                vmask &= x < v1x + t * (v2x - v1x)
                mask ^= vmask
            v1x, v1y = v2x, v2y
        return mask


@dataclass
class Circle(Region):
    """Mask contains points of axis within an xy circle of given radius

    .. example_spec::

        from scanspec.specs import Line
        from scanspec.regions import Circle

        grid = Line("y", 1, 3, 10) * ~Line("x", 0, 2, 10)
        spec = grid & Circle("x", "y", 1, 2, 0.9)
    """

    x_axis: A[str, schema(description="The name matching the x axis of the spec")]
    y_axis: A[str, schema(description="The name matching the y axis of the spec")]
    x_middle: A[float, schema(description="The central x point of the circle")]
    y_middle: A[float, schema(description="The central y point of the circle")]
    radius: A[float, schema(description="Radius of the circle", exc_min=0)]

    def axis_sets(self) -> List[Set[str]]:
        return [{self.x_axis, self.y_axis}]

    def mask(self, points: AxesPoints) -> np.ndarray:
        x = points[self.x_axis] - self.x_middle
        y = points[self.y_axis] - self.y_middle
        mask = x * x + y * y <= (self.radius * self.radius)
        return mask


@dataclass
class Ellipse(Region):
    """Mask contains points of axis within an xy ellipse of given radius

    .. example_spec::

        from scanspec.specs import Line
        from scanspec.regions import Ellipse

        grid = Line("y", 3, 8, 10) * ~Line("x", 1 ,8, 10)
        spec = grid & Ellipse("x", "y", 5, 5, 2, 3, 75)
    """

    x_axis: A[str, schema(description="The name matching the x axis of the spec")]
    y_axis: A[str, schema(description="The name matching the y axis of the spec")]
    x_middle: A[float, schema(description="The central x point of the ellipse")]
    y_middle: A[float, schema(description="The central y point of the ellipse")]
    x_radius: A[
        float,
        schema(description="The radius along the x axis of the ellipse", exc_min=0),
    ]
    y_radius: A[
        float,
        schema(description="The radius along the y axis of the ellipse", exc_min=0),
    ]
    angle: A[float, schema(description="The angle of the ellipse (degrees)")] = 0.0

    def axis_sets(self) -> List[Set[str]]:
        return [{self.x_axis, self.y_axis}]

    def mask(self, points: AxesPoints) -> np.ndarray:
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


def find_regions(obj) -> Iterator[Region]:
    """Recursively iterate over obj and its children, yielding any Region
    instances found"""
    if is_dataclass(obj):
        if isinstance(obj, Region):
            yield obj
        for name in obj.__dict__.keys():
            yield from find_regions(getattr(obj, name))
