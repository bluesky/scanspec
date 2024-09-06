from __future__ import annotations

from _collections_abc import Callable, Mapping
from typing import Any

import numpy as np
import strawberry

from scanspec.core import (
    Axis,
    Frames,
    gap_between_frames,
)


@strawberry.type
class PointsResponse:
    """Information about the points provided by a spec."""

    total_frames: int
    returned_frames: int

    def __init__(self, chunk: Frames[str], total_frames: int):
        self.total_frames = total_frames
        """The number of frames present across the entire spec"""
        self.returned_frames = len(chunk)
        """The number of frames returned by the getPoints query
            (controlled by the max_points argument)"""
        self._chunk = chunk


@strawberry.interface
class SpecInterface:
    def serialize(self) -> Mapping[str, Any]:
        """Serialize the spec to a dictionary."""
        return "serialized"


def _dimensions_from_indexes(
    func: Callable[[np.ndarray], dict[Axis, np.ndarray]],
    axes: list,
    num: int,
    bounds: bool,
) -> list[Frames[Axis]]:
    # Calc num midpoints (fences) from 0.5 .. num - 0.5
    midpoints_calc = func(np.linspace(0.5, num - 0.5, num))
    midpoints = {a: midpoints_calc[a] for a in axes}
    if bounds:
        # Calc num + 1 bounds (posts) from 0 .. num
        bounds_calc = func(np.linspace(0, num, num + 1))
        lower = {a: bounds_calc[a][:-1] for a in axes}
        upper = {a: bounds_calc[a][1:] for a in axes}
        # Points must have no gap as upper[a][i] == lower[a][i+1]
        # because we initialized it to be that way
        gap = np.zeros(num, dtype=np.bool_)
        dimension = Frames(midpoints, lower, upper, gap)
        # But calc the first point as difference between first
        # and last
        gap[0] = gap_between_frames(dimension, dimension)
    else:
        # Gap can be calculated in Dimension
        dimension = Frames(midpoints)
    return [dimension]


@strawberry.input
class Line(SpecInterface):
    axis: str = strawberry.field(description="An identifier for what to move")
    start: float = strawberry.field(
        description="Midpoint of the first point of the line"
    )
    stop: float = strawberry.field(description="Midpoint of the last point of the line")
    num: int = strawberry.field(description="Number of frames to produce")

    def axes(self) -> list:
        return [self.axis]

    def _line_from_indexes(self, indexes: np.ndarray) -> dict[Axis, np.ndarray]:
        if self.num == 1:
            # Only one point, stop-start gives length of one point
            step = self.stop - self.start
        else:
            # Multiple points, stop-start gives length of num-1 points
            step = (self.stop - self.start) / (self.num - 1)
        # self.start is the first centre point, but we need the lower bound
        # of the first point as this is where the index array starts
        first = self.start - step / 2
        return {self.axis: indexes * step + first}

    def calculate(self, bounds=True, nested=False) -> list[Frames[Axis]]:
        return _dimensions_from_indexes(
            self._line_from_indexes, self.axes(), self.num, bounds
        )
