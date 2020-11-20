from typing import Any, List, Optional, cast

import numpy as np
from pydantic import Field

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
    left: Spec
    right: Spec

    def get_keys(self) -> List:
        return self.left.keys + self.right.keys

    def create_dimensions(self, bounds=True) -> List[Dimension]:
        dims_left = self.left.create_dimensions(bounds=False)
        dims_right = self.right.create_dimensions(bounds)
        return dims_left + dims_right


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


class Line(Spec):
    """Thing"""

    key: Any
    start: float
    stop: float
    num: int = Field(..., ge=1)

    @classmethod
    def bounded(cls, key, lower: float, upper: float, num: int):
        """Specify instance by extreme bounds

        Args:
            key: Thing to move
            lower: Lower bound of the first point of the line
            upper: Upper bound of the last point of the line
            num: Number of points in the line
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

    def get_keys(self) -> List:
        return [self.key]

    def create_dimensions(self, bounds=True) -> List[Dimension]:
        positions = {self.key: np.linspace(self.start, self.stop, self.num)}
        if bounds:
            if self.num == 1:
                # Only one point, stop-start gives length of one point
                half_step = (self.stop - self.start) / 2
                stop = self.start + half_step
            else:
                # Multiple points, stop-start gives length of num-1 points
                half_step = (self.stop - self.start) / (self.num - 1) / 2
                stop = self.stop + half_step
            bounds_array = np.linspace(self.start - half_step, stop, self.num + 1)
            lower = {self.key: bounds_array[:-1]}
            upper = {self.key: bounds_array[1:]}
            dimension = Dimension(positions, lower, upper)
        else:
            dimension = Dimension(positions)
        return [dimension]




class Spiral(Spec):
    x_key: Any
    y_key: Any
    x_start: float = Field(..., description="x centre of the spiral")
    y_start: float = Field(..., description="y centre of the spiral")
    x_range: float = Field(..., description="x width of the spiral")
    y_range: float = Field(..., description="y width of the spiral")
    x_radius: float = Field(..., description="x radius of the spiral")
    y_radius: float = Field(None, description="x radius of the spiral")
    y_start: float
    , x_range, y_range, dr, nth, *,
           dr_y=None, tilt=0.0):
    '''Spiral scan, centered around (x_start, y_start)
    Parameters
    ----------
    x_motor : object, optional
        any 'setable' object (motor, temp controller, etc.)
    y_motor : object, optional
        any 'setable' object (motor, temp controller, etc.)
    x_start : float
        x center
    y_start : float
        y center
    x_range : float
        x width of spiral
    y_range : float
        y width of spiral
    dr : float
        Delta radius along the minor axis of the ellipse.
    dr_y : float, optional
        Delta radius along the major axis of the ellipse, if not specifed
        defaults to dr
    nth : float
        Number of theta steps
    tilt : float, optional
        Tilt angle in radians, default 0.0
    Returns
    -------
    cyc : cycler
    '''
    if dr_y is None:
        dr_aspect = 1
    else:
        dr_aspect = dr_y / dr

    half_x = x_range / 2
    half_y = y_range / (2 * dr_aspect)

    r_max = np.sqrt(half_x ** 2 + half_y ** 2)
    num_ring = 1 + int(r_max / dr)
    tilt_tan = np.tan(tilt + np.pi / 2.)

    x_points, y_points = [], []

    for i_ring in range(1, num_ring + 2):
        radius = i_ring * dr
        angle_step = 2. * np.pi / (i_ring * nth)

        for i_angle in range(int(i_ring * nth)):
            angle = i_angle * angle_step
            x = radius * np.cos(angle)
            y = radius * np.sin(angle) * dr_aspect
            if ((abs(x - (y / dr_aspect) / tilt_tan) <= half_x) and
                    (abs(y / dr_aspect) <= half_y)):
                x_points.append(x_start + x)
                y_points.append(y_start + y)