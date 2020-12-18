from typing import Any, Dict, Iterator, List, Set

import numpy as np
from pydantic import Field
from pydantic.main import BaseModel

from .core import Positions, Serializable, if_instance_do


class Region(Serializable):
    """Abstract baseclass for a Region that can `Mask` a `Spec`. Supports operators:

    - ``|``: `UnionOf` two Regions, positions present in either
    - ``&``: `IntersectionOf` two Regions, positions present in both
    - ``-``: `DifferenceOf` two Regions, positions present in first not second
    - ``^``: `SymmetricDifferenceOf` two Regions, positions present in one not both
    """

    def key_sets(self) -> List[Set[str]]:
        """Implemented by subclasses to produce the non-overlapping sets of keys
        this region spans"""
        raise NotImplementedError(self)

    def mask(self, positions: Dict[Any, np.ndarray]) -> np.ndarray:
        """Implemented by subclasses to produce a mask of which positions are in
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


def get_mask(region: Region, positions: Positions) -> np.ndarray:
    """If there is an overlap of keys of region and positions return a
    mask of the positions in the region, otherwise return all ones
    """
    keys = set(positions)
    needs_mask = any(ks & keys for ks in region.key_sets())
    if needs_mask:
        return region.mask(positions)
    else:
        return np.ones(len(list(positions.values())[0]))


def _merge_key_sets(key_sets: List[Set[str]]) -> Iterator[Set[str]]:
    # Take overlapping key sets and merge any that overlap into each
    # other
    for ks in key_sets:
        key_set = ks.copy()
        # Empty matching sets into this key_set
        for ks in key_sets:
            if ks & key_set:
                while ks:
                    key_set.add(ks.pop())
        # It might be emptied already, only yield if it isn't
        if key_set:
            yield key_set


class CombinationOf(Region):
    """Abstract baseclass for a combination of two regions, left and right"""

    left: Region = Field(..., description="The left-hand Region to combine")
    right: Region = Field(..., description="The right-hand Region to combine")

    def key_sets(self) -> List[Set[str]]:
        key_sets = list(_merge_key_sets(self.left.key_sets() + self.right.key_sets()))
        return key_sets


# Naming so we don't clash with typing.Union
class UnionOf(CombinationOf):
    """A position is in UnionOf(a, b) if it is in either a or b. Typically
    created with the ``|`` operator

    >>> r = Range("x", 0.5, 2.5) | Range("x", 1.5, 3.5)
    >>> r.mask({"x": np.array([0, 1, 2, 3, 4])})
    array([False,  True,  True,  True, False])
    """

    def mask(self, positions: Dict[Any, np.ndarray]) -> np.ndarray:
        mask = get_mask(self.left, positions) | get_mask(self.right, positions)
        return mask


class IntersectionOf(CombinationOf):
    """A position is in IntersectionOf(a, b) if it is in both a and b. Typically
    created with the ``&`` operator

    >>> r = Range("x", 0.5, 2.5) & Range("x", 1.5, 3.5)
    >>> r.mask({"x": np.array([0, 1, 2, 3, 4])})
    array([False, False,  True, False, False])
    """

    def mask(self, positions: Dict[Any, np.ndarray]) -> np.ndarray:
        mask = get_mask(self.left, positions) & get_mask(self.right, positions)
        return mask


class DifferenceOf(CombinationOf):
    """A position is in DifferenceOf(a, b) if it is in a and not in b. Typically
    created with the ``-`` operator

    >>> r = Range("x", 0.5, 2.5) - Range("x", 1.5, 3.5)
    >>> r.mask({"x": np.array([0, 1, 2, 3, 4])})
    array([False,  True, False, False, False])
    """

    def mask(self, positions: Dict[Any, np.ndarray]) -> np.ndarray:
        left_mask = get_mask(self.left, positions)
        # Return the xor restricted to the left region
        mask = left_mask ^ get_mask(self.right, positions) & left_mask
        return mask


class SymmetricDifferenceOf(CombinationOf):
    """A position is in SymmetricDifferenceOf(a, b) if it is in either a or b,
    but not both. Typically created with the ``^`` operator

    >>> r = Range("x", 0.5, 2.5) ^ Range("x", 1.5, 3.5)
    >>> r.mask({"x": np.array([0, 1, 2, 3, 4])})
    array([False,  True, False,  True, False])
    """

    def mask(self, positions: Dict[Any, np.ndarray]) -> np.ndarray:
        mask = get_mask(self.left, positions) ^ get_mask(self.right, positions)
        return mask


class Range(Region):
    """Mask contains positions of key >= min and <= max

    >>> r = Range("x", 1, 2)
    >>> r.mask({"x": np.array([0, 1, 2, 3, 4])})
    array([False,  True,  True, False, False])
    """

    key: Any = Field(..., description="The key matching the axis to mask in the spec")
    min: float = Field(..., description="The minimum inclusive value in the region")
    max: float = Field(..., description="The minimum inclusive value in the region")

    def key_sets(self) -> List[Set[str]]:
        return [{self.key}]

    def mask(self, positions: Dict[Any, np.ndarray]) -> np.ndarray:
        v = positions[self.key]
        mask = np.bitwise_and(v >= self.min, v <= self.max)
        return mask


class Rectangle(Region):
    """Mask contains positions of key within a rotated xy rectangle

    .. example_spec::

        from scanspec.specs import Line
        from scanspec.regions import Rectangle

        grid = Line("y", 1, 3, 10) * ~Line("x", 0, 2, 10)
        spec = grid & Rectangle("x", "y", 0, 1.1, 1.5, 2.1, 30)
    """

    x_key: Any = Field(..., description="The key matching the x axis of the spec")
    y_key: Any = Field(..., description="The key matching the y axis of the spec")
    x_min: float = Field(..., description="Minimum inclusive x value in the region")
    y_min: float = Field(..., description="Minimum inclusive y value in the region")
    x_max: float = Field(..., description="Maximum inclusive x value in the region")
    y_max: float = Field(..., description="Maximum inclusive y value in the region")
    angle: float = Field(0, description="Clockwise rotation angle of the rectangle")

    def key_sets(self) -> List[Set[str]]:
        return [{self.x_key, self.y_key}]

    def mask(self, positions: Dict[Any, np.ndarray]) -> np.ndarray:
        x = positions[self.x_key] - self.x_min
        y = positions[self.y_key] - self.y_min
        if self.angle != 0:
            # Rotate src positions by -angle
            phi = np.radians(-self.angle)
            rx = x * np.cos(phi) - y * np.sin(phi)
            ry = x * np.sin(phi) + y * np.cos(phi)
            x = rx
            y = ry
        mask_x = np.bitwise_and(x >= 0, x <= (self.x_max - self.x_min))
        mask_y = np.bitwise_and(y >= 0, y <= (self.y_max - self.y_min))
        return mask_x & mask_y


class Circle(Region):
    """Mask contains positions of key within an xy circle of given radius

    .. example_spec::

        from scanspec.specs import Line
        from scanspec.regions import Circle

        grid = Line("y", 1, 3, 10) * ~Line("x", 0, 2, 10)
        spec = grid & Circle("x", "y", 1, 2, 0.9)
    """

    x_key: Any = Field(..., description="The key matching the x axis of the spec")
    y_key: Any = Field(..., description="The key matching the x axis of the spec")
    x_centre: float = Field(
        ..., description="Minimum inclusive x value in the region", alias="x_center"
    )
    y_centre: float = Field(
        ..., description="Minimum inclusive y value in the region", alias="y_center"
    )
    radius: float = Field(..., description="Radius of the circle")

    class Config:
        # Allow either centre or center
        allow_population_by_field_name = True

    def key_sets(self) -> List[Set[str]]:
        return [{self.x_key, self.y_key}]

    def mask(self, positions: Dict[Any, np.ndarray]) -> np.ndarray:
        x = positions[self.x_key] - self.x_centre
        y = positions[self.y_key] - self.y_centre
        mask = x * x + y * y <= (self.radius * self.radius)
        return mask


def find_regions(obj) -> Iterator[Region]:
    """Recursively iterate over obj and its children, yielding any Region
    instances found"""
    if isinstance(obj, BaseModel):
        if isinstance(obj, Region):
            yield obj
        for name in obj.__fields__:
            yield from find_regions(getattr(obj, name))
