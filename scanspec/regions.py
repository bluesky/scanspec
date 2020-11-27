from typing import Any, Dict, Iterator, List, Set

import numpy as np
from pydantic.fields import Field

from .core import Positions, WithType, if_instance_do


class Region(WithType):
    def key_sets(self) -> List[Set[str]]:
        # Return the non-overlapping sets of keys this region spans
        raise NotImplementedError(self)

    def mask(self, positions: Dict[Any, np.ndarray]) -> np.ndarray:
        # Return a mask of which positions are in the region
        raise NotImplementedError(self)

    def __or__(self, other: "Region") -> "Region":
        return if_instance_do(other, Region, lambda o: UnionOf(self, o))

    def __and__(self, other: "Region") -> "Region":
        return if_instance_do(other, Region, lambda o: IntersectionOf(self, o))

    def __sub__(self, other: "Region") -> "Region":
        return if_instance_do(other, Region, lambda o: DifferenceOf(self, o))

    def __xor__(self, other: "Region") -> "Region":
        return if_instance_do(other, Region, lambda o: SymmetricDifferenceOf(self, o))


def get_mask(region: Region, positions: Positions) -> np.ndarray:
    keys = set(positions)
    needs_mask = any(ks & keys for ks in region.key_sets())
    if needs_mask:
        return region.mask(positions)
    else:
        return np.ones(len(list(positions.values())[0]))


def _merge_key_sets(key_sets: List[Set[str]]) -> Iterator[Set[str]]:
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
    left: Region
    right: Region

    def key_sets(self) -> List[Set[str]]:
        # Return the non-overlapping sets of keys this region spans
        key_sets = list(_merge_key_sets(self.left.key_sets() + self.right.key_sets()))
        return key_sets


# Naming so we don't clash with typing.Union
class UnionOf(CombinationOf):
    def mask(self, positions: Dict[Any, np.ndarray]) -> np.ndarray:
        mask = get_mask(self.left, positions) | get_mask(self.right, positions)
        return mask


class IntersectionOf(CombinationOf):
    def mask(self, positions: Dict[Any, np.ndarray]) -> np.ndarray:
        mask = get_mask(self.left, positions) & get_mask(self.right, positions)
        return mask


class DifferenceOf(CombinationOf):
    def mask(self, positions: Dict[Any, np.ndarray]) -> np.ndarray:
        left_mask = get_mask(self.left, positions)
        # Return the xor restricted to the left region
        mask = left_mask ^ get_mask(self.right, positions) & left_mask
        return mask


class SymmetricDifferenceOf(CombinationOf):
    def mask(self, positions: Dict[Any, np.ndarray]) -> np.ndarray:
        mask = get_mask(self.left, positions) ^ get_mask(self.right, positions)
        return mask


class Rectangle(Region):
    x_key: Any = Field(..., description="The key matching the x axis of the spec")
    y_key: Any = Field(..., description="The key matching the x axis of the spec")
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
