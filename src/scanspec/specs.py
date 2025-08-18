"""`Spec` and its subclasses.

.. inheritance-diagram:: scanspec.specs
    :top-classes: scanspec.specs.Spec
    :parts: 1
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from typing import Any, Generic, Literal, SupportsFloat

import numpy as np
import numpy.typing as npt
from pydantic import Field, TypeAdapter, validate_call
from pydantic.dataclasses import dataclass

from .core import (
    Axis,
    Dimension,
    Midpoints,
    OtherAxis,
    SnakedDimension,
    StrictConfig,
    discriminated_union_of_subclasses,
    gap_between_frames,
    if_instance_do,
    squash_frames,
    stack2dimension,
)
from .regions import Region, get_mask

__all__ = [
    "ConstantDuration",
    "Spec",
    "Product",
    "Zip",
    "Mask",
    "Snake",
    "Concat",
    "Squash",
    "Line",
    "Static",
    "Spiral",
    "Fly",
    "step",
    "fly",
    "VARIABLE_DURATION",
]

#: A string returned from `Spec.duration` to signify it produces
#: a different duration for each point
VARIABLE_DURATION = "VARIABLE_DURATION"


@discriminated_union_of_subclasses
class Spec(Generic[Axis]):
    """A serializable representation of the type and parameters of a scan.

    Abstract baseclass for the specification of a scan. Supports operators:

    - ``*``: Outer `Product` of two Specs or ints, nesting the second within the first.
    - ``@``: `ConstantDuration` of the Spec, setting a constant duration for each point.
    - ``&``: `Mask` the Spec with a `Region`, excluding midpoints outside of it
    - ``~``: `Snake` the Spec, reversing every other iteration of it
    """

    def __post_init__(self):
        # Call axes and duration as they do error checking for zip, product etc.
        self.axes()
        self.duration()

    def axes(self) -> list[Axis]:  # noqa: D102
        """Return the list of axes that are present in the scan.

        Ordered from slowest moving to fastest moving.
        """
        raise NotImplementedError(self)

    def duration(self) -> float | None | Literal["VARIABLE_DURATION"]:
        """Returns the duration of each scan point.

        Return value will be one of:
        - ``None``: No duration defined
        - ``float``: A constant duration for each point
        - `VARIABLE_DURATION`: A different duration for each point
        """
        return None

    def calculate(
        self, bounds: bool = False, nested: bool = False
    ) -> list[Dimension[Axis]]:  # noqa: D102
        """Produce a stack of nested `Dimension` that form the scan.

        Ordered from slowest moving to fastest moving.
        """
        raise NotImplementedError(self)

    def frames(self, bounds: bool = False) -> Dimension[Axis]:
        """Expand all the scan `Dimension` and return them."""
        return stack2dimension(self.calculate(bounds=bounds))

    def midpoints(self) -> Midpoints[Axis]:
        """Return `Midpoints` that can be iterated point by point."""
        return Midpoints(self.calculate(bounds=False))

    def shape(self) -> tuple[int, ...]:
        """Return the final, simplified shape of the scan."""
        return tuple(len(dim) for dim in self.calculate())

    def __rmatmul__(self, other: SupportsFloat) -> ConstantDuration[Axis]:
        return if_instance_do(
            other, SupportsFloat, lambda o: ConstantDuration(float(o), self)
        )

    def __rmul__(self, other: Spec[Axis] | int) -> Product[Axis]:
        return if_instance_do(other, (Spec, int), lambda o: Product(o, self))

    def __mul__(self, other: Spec[Axis] | int) -> Product[Axis]:
        return if_instance_do(other, (Spec, int), lambda o: Product(self, o))

    def __and__(self, other: Region[Axis]) -> Mask[Axis]:
        return if_instance_do(other, Region, lambda o: Mask(self, o))

    def __invert__(self) -> Snake[Axis]:
        return Snake(self)

    def zip(self, other: Spec[OtherAxis]) -> Zip[Axis | OtherAxis]:
        """`Zip` the Spec with another, iterating in tandem."""
        return Zip(left=self, right=other)

    def concat(self, other: Spec[Axis]) -> Concat[Axis]:
        """`Concat` the Spec with another, iterating one after the other."""
        return Concat(self, other)

    def serialize(self) -> Mapping[str, Any]:
        """Serialize the Spec to a dictionary."""
        return TypeAdapter(Spec[Any]).dump_python(self)

    @staticmethod
    def deserialize(obj: Any) -> Spec[Any]:
        """Deserialize a Spec from a dictionary."""
        return TypeAdapter(Spec[Any]).validate_python(obj)


@dataclass(config=StrictConfig)
class Product(Spec[Axis]):
    """Outer product of two Specs, nesting inner within outer.

    This means that inner will run in its entirety at each point in outer.

    .. example_spec::

        from scanspec.specs import Fly, Line

        spec = Fly(Line("y", 1, 2, 3) * Line("x", 3, 4, 12))

    An inner integer can be used to repeat the same point many times.

    .. example_spec::

        from scanspec.specs import Fly, Line

        spec = Fly(Line("y", 1, 2, 3) * 2)

    An outer integer can be used to repeat the same scan many times.

    .. example_spec::

        from scanspec.specs import Fly, Line

        spec = Fly(2 * ~Line.bounded("x", 3, 4, 1))

    If you want snaked axes to have no gap between iterations you can do:

    .. example_spec::

        from scanspec.specs import Fly, Line, Product

        spec = Fly(Product(2, ~Line.bounded("x", 3, 4, 1), gap=False))

    .. note:: There is no turnaround arrow at x=4
    """

    outer: Spec[Axis] | int = Field(description="Will be executed once")
    inner: Spec[Axis] | int = Field(description="Will be executed len(outer) times")
    gap: bool = Field(
        description="If False and the outer spec is an integer and the inner spec is "
        "snaked then the end and start of consecutive iterations of inner will have no "
        "gap",
        default=True,
    )

    def axes(self) -> list[Axis]:  # noqa: D102
        outer_axes = [] if isinstance(self.outer, int) else self.outer.axes()
        inner_axes = [] if isinstance(self.inner, int) else self.inner.axes()
        return outer_axes + inner_axes

    def duration(self) -> float | None | Literal["VARIABLE_DURATION"]:  # noqa: D102
        outer_duration = None if isinstance(self.outer, int) else self.outer.duration()
        inner_duration = None if isinstance(self.inner, int) else self.inner.duration()
        if outer_duration is not None:
            if inner_duration is not None:
                raise ValueError("Both inner and outer specs defined a duration")
            return outer_duration
        else:
            return inner_duration

    def calculate(  # noqa: D102
        self, bounds: bool = False, nested: bool = False
    ) -> list[Dimension[Axis]]:
        if isinstance(self.outer, int):
            dims_outer = [Dimension[Axis]({}, gap=np.full(self.outer, self.gap))]
        else:
            dims_outer = self.outer.calculate(bounds=False, nested=nested)
        if isinstance(self.inner, int):
            dims_inner = [Dimension[Axis]({}, gap=np.full(self.inner, False))]
        else:
            dims_inner = self.inner.calculate(bounds, nested=True)
        return dims_outer + dims_inner


@dataclass(config=StrictConfig)
class Zip(Spec[Axis]):
    """Run two Specs in parallel, merging their midpoints together.

    Typically formed using `Spec.zip`.

    Stacks of Dimension are merged by:

    - If right creates a stack of a single Dimension object of size 1, expand it to
      the size of the fastest Dimension object created by left
    - Merge individual Dimension objects together from fastest to slowest

    This means that Zipping a Spec producing stack [l2, l1] with a Spec
    producing stack [r1] will assert len(l1)==len(r1), and produce
    stack [l2, l1.zip(r1)].

    .. example_spec::

        from scanspec.specs import Fly, Line

        spec = Fly(Line("z", 1, 2, 3) * Line("y", 3, 4, 5).zip(Line("x", 4, 5, 5)))
    """

    left: Spec[Axis] = Field(
        description="The left-hand Spec to Zip, will appear earlier in axes"
    )
    right: Spec[Axis] = Field(
        description="The right-hand Spec to Zip, will appear later in axes"
    )

    def axes(self) -> list[Axis]:  # noqa: D102
        return self.left.axes() + self.right.axes()

    def duration(self) -> float | None | Literal["VARIABLE_DURATION"]:  # noqa: D102
        left, right = self.left.duration(), self.right.duration()
        if left is not None and right is not None:
            raise ValueError("Both left and right define a duration")
        return left if left is not None else right

    def calculate(  # noqa: D102
        self, bounds: bool = False, nested: bool = False
    ) -> list[Dimension[Axis]]:
        frames_left = self.left.calculate(bounds, nested)
        frames_right = self.right.calculate(bounds, nested)
        assert len(frames_left) >= len(frames_right), (
            f"Zip requires len({self.left}) >= len({self.right})"
        )

        # Pad and expand the right to be the same size as left. Special case, if
        # only one Dimension object with size 1, expand to the right size
        if len(frames_right) == 1 and len(frames_right[0]) == 1:
            # Take the 0th element N times to make a repeated Dimension object
            indices = np.zeros(len(frames_left[-1]), dtype=np.int8)
            repeated = frames_right[0].extract(indices)

            if isinstance(frames_left[-1], SnakedDimension):
                repeated = SnakedDimension.from_frames(repeated)
            frames_right = [repeated]

        # Left pad frames_right with Nones so they are the same size
        npad = len(frames_left) - len(frames_right)
        padded_right: list[Dimension[Axis] | None] = [None] * npad
        # Mypy doesn't like this because lists are invariant:
        # https://github.com/python/mypy/issues/4244
        padded_right += frames_right

        # Work through, zipping them together one by one
        frames: list[Dimension[Axis]] = []
        for left, right in zip(frames_left, padded_right, strict=False):
            if right is None:
                combined = left
            else:
                combined = left.zip(right)
            assert isinstance(combined, Dimension), (
                f"Padding went wrong {frames_left} {padded_right}"
            )
            frames.append(combined)
        return frames


@dataclass(config=StrictConfig)
class Mask(Spec[Axis]):
    """Restrict Spec to only midpoints that fall inside the given Region.

    Typically created with the ``&`` operator. It also pushes down the
    ``& | ^ -`` operators to its `Region` to avoid the need for brackets on
    combinations of Regions.

    If a Region spans multiple Dimension objects, they will be squashed together.

    .. example_spec::

        from scanspec.regions import Circle
        from scanspec.specs import Fly, Line

        region = Circle("x", "y", 4, 2, 1.2)
        spec = Fly(Line("y", 1, 3, 3) * Line("x", 3, 5, 5) & region)

    See Also: `why-squash-can-change-path`
    """

    spec: Spec[Axis] = Field(description="The Spec containing the source midpoints")
    region: Region[Axis] = Field(description="The Region that midpoints will be inside")
    check_path_changes: bool = Field(
        description="If True path through scan will not be modified by squash",
        default=True,
    )

    def axes(self) -> list[Axis]:  # noqa: D102
        return self.spec.axes()

    def duration(self) -> float | None | Literal["VARIABLE_DURATION"]:  # noqa: D102
        return self.spec.duration()

    def calculate(  # noqa: D102
        self, bounds: bool = False, nested: bool = False
    ) -> list[Dimension[Axis]]:
        frames = self.spec.calculate(bounds, nested)
        for axis_set in self.region.axis_sets():
            # Find the start and end index of any dimensions containing these axes
            matches = [i for i, d in enumerate(frames) if set(d.axes()) & axis_set]
            assert matches, f"No Specs match axes {list(axis_set)}"
            si, ei = matches[0], matches[-1]
            if si != ei:
                # The axis_set spans multiple Dimensions, squash them together
                # If the spec to be squashed is nested (inside the Mask or outside)
                # then check the path changes if requested
                check_path_changes = bool(nested or si) and self.check_path_changes
                squashed = squash_frames(frames[si : ei + 1], check_path_changes)
                frames = frames[:si] + [squashed] + frames[ei + 1 :]
        # Generate masks from the midpoints showing what's inside
        masked_frames: list[Dimension[Axis]] = []
        for f in frames:
            indices = get_mask(self.region, f.midpoints).nonzero()[0]
            masked_frames.append(f.extract(indices))
        return masked_frames

    # *+ bind more tightly than &|^ so without these overrides we
    # would need to add brackets to all combinations of Regions
    def __or__(self, other: Region[Axis]) -> Mask[Axis]:
        return if_instance_do(other, Region, lambda o: Mask(self.spec, self.region | o))

    def __and__(self, other: Region[Axis]) -> Mask[Axis]:
        return if_instance_do(other, Region, lambda o: Mask(self.spec, self.region & o))

    def __xor__(self, other: Region[Axis]) -> Mask[Axis]:
        return if_instance_do(other, Region, lambda o: Mask(self.spec, self.region ^ o))

    # This is here for completeness, tends not to be called as - binds
    # tighter than &
    def __sub__(self, other: Region[Axis]) -> Mask[Axis]:
        return if_instance_do(other, Region, lambda o: Mask(self.spec, self.region - o))


@dataclass(config=StrictConfig)
class Snake(Spec[Axis]):
    """Run the Spec in reverse on every other iteration when nested.

    Typically created with the ``~`` operator.

    .. example_spec::

        from scanspec.specs import Fly, Line

        spec = Fly(Line("y", 1, 3, 3) * ~Line("x", 3, 5, 5))
    """

    spec: Spec[Axis] = Field(
        description="The Spec to run in reverse every other iteration"
    )

    def axes(self) -> list[Axis]:  # noqa: D102
        return self.spec.axes()

    def duration(self) -> float | None | Literal["VARIABLE_DURATION"]:  # noqa: D102
        return self.spec.duration()

    def calculate(  # noqa: D102
        self, bounds: bool = False, nested: bool = False
    ) -> list[Dimension[Axis]]:
        return [
            SnakedDimension.from_frames(segment)
            for segment in self.spec.calculate(bounds, nested)
        ]


@dataclass(config=StrictConfig)
class Concat(Spec[Axis]):
    """Concatenate two Specs together, running one after the other.

    Each Dimension of left and right must contain the same axes. Typically
    formed using `Spec.concat`.

    .. example_spec::

        from scanspec.specs import Fly, Line

        spec = Fly(Line("x", 1, 3, 3).concat(Line("x", 4, 5, 5)))
    """

    left: Spec[Axis] = Field(
        description="The left-hand Spec to Concat, midpoints will appear earlier"
    )
    right: Spec[Axis] = Field(
        description="The right-hand Spec to Concat, midpoints will appear later"
    )

    gap: bool = Field(
        description="If True, force a gap in the output at the join", default=False
    )
    check_path_changes: bool = Field(
        description="If True path through scan will not be modified by squash",
        default=True,
    )

    def axes(self) -> list[Axis]:  # noqa: D102
        left_axes, right_axes = self.left.axes(), self.right.axes()
        # Assuming the axes are the same, the order does not matter, we inherit the
        # order from the left-hand side. See also scanspec.core.concat.
        assert set(left_axes) == set(right_axes), f"axes {left_axes} != {right_axes}"
        return left_axes

    def duration(self) -> float | None | Literal["VARIABLE_DURATION"]:  # noqa: D102
        left, right = self.left.duration(), self.right.duration()
        if left == right:
            # They are producing the same duration
            return left
        elif left is None or right is None:
            # They aren't both None, but if one is then raise
            raise ValueError("Only one of left and right defines a duration")
        else:
            # They both exist, but are different, so are variable
            return VARIABLE_DURATION

    def calculate(  # noqa: D102
        self, bounds: bool = False, nested: bool = False
    ) -> list[Dimension[Axis]]:
        dim_left = squash_frames(
            self.left.calculate(bounds, nested), nested and self.check_path_changes
        )
        dim_right = squash_frames(
            self.right.calculate(bounds, nested), nested and self.check_path_changes
        )
        dim = dim_left.concat(dim_right, self.gap)
        return [dim]


@dataclass(config=StrictConfig)
class Squash(Spec[Axis]):
    """Squash a stack of Dimension together into a single expanded Dimension object.

    See Also:
        `why-squash-can-change-path`

    .. example_spec::

        from scanspec.specs import Fly, Line, Squash

        spec = Fly(Squash(Line("y", 1, 2, 3) * Line("x", 0, 1, 4)))

    """

    spec: Spec[Axis] = Field(description="The Spec to squash the dimensions of")
    check_path_changes: bool = Field(
        description="If True path through scan will not be modified by squash",
        default=True,
    )

    def axes(self) -> list[Axis]:  # noqa: D102
        return self.spec.axes()

    def duration(self) -> float | None | Literal["VARIABLE_DURATION"]:  # noqa: D102
        return self.spec.duration()

    def calculate(  # noqa: D102
        self, bounds: bool = False, nested: bool = False
    ) -> list[Dimension[Axis]]:
        dims = self.spec.calculate(bounds, nested)
        dim = squash_frames(dims, nested and self.check_path_changes)
        return [dim]


def _dimensions_from_indexes(
    func: Callable[[npt.NDArray[np.float64]], dict[Axis, npt.NDArray[np.float64]]],
    axes: list[Axis],
    num: int,
    bounds: bool,
) -> list[Dimension[Axis]]:
    # Calc num midpoints (fences) from 0.5 .. num - 0.5
    midpoints_calc = func(np.linspace(0.5, num - 0.5, num, dtype=np.float64))
    midpoints = {a: midpoints_calc[a] for a in axes}
    if bounds:
        # Calc num + 1 bounds (posts) from 0 .. num
        bounds_calc = func(np.linspace(0, num, num + 1, dtype=np.float64))
        lower = {a: bounds_calc[a][:-1] for a in axes}
        upper = {a: bounds_calc[a][1:] for a in axes}
        # Points must have no gap as upper[a][i] == lower[a][i+1]
        # because we initialized it to be that way
        gap = np.zeros(num, dtype=np.bool_)
        dimension = Dimension(midpoints, lower, upper, gap)
        # But calc the first point as difference between first
        # and last
        gap[0] = gap_between_frames(dimension, dimension)
    else:
        # Gap can be calculated in Dimension
        dimension = Dimension(midpoints)
    return [dimension]


@dataclass(config=StrictConfig)
class Line(Spec[Axis]):
    """Linearly spaced frames with start and stop as first and last midpoints.

    .. example_spec::

        from scanspec.specs import Fly, Line

        spec = Fly(Line("x", 1, 2, 5))
    """

    axis: Axis = Field(description="An identifier for what to move")
    start: float = Field(description="Midpoint of the first point of the line")
    stop: float = Field(description="Midpoint of the last point of the line")
    num: int = Field(ge=1, description="Number of frames to produce")

    def axes(self) -> list[Axis]:  # noqa: D102
        return [self.axis]

    def _line_from_indexes(
        self, indexes: npt.NDArray[np.float64]
    ) -> dict[Axis, npt.NDArray[np.float64]]:
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

    def calculate(  # noqa: D102
        self, bounds: bool = False, nested: bool = False
    ) -> list[Dimension[Axis]]:
        return _dimensions_from_indexes(
            self._line_from_indexes, self.axes(), self.num, bounds
        )

    @classmethod
    def bounded(
        cls: type[Line[Any]],
        axis: OtherAxis = Field(description="An identifier for what to move"),
        lower: float = Field(description="Lower bound of the first point of the line"),
        upper: float = Field(description="Upper bound of the last point of the line"),
        num: int = Field(ge=1, description="Number of frames to produce"),
    ) -> Line[OtherAxis]:
        """Specify a Line by extreme bounds instead of midpoints.

        .. example_spec::

            from scanspec.specs import Fly, Line

            spec = Fly(Line.bounded("x", 1, 2, 5))
        """
        half_step = (upper - lower) / num / 2
        start = lower + half_step
        if num == 1:
            # One point, stop will only be used for step size
            stop = upper + half_step
        else:
            # Many points, stop will be produced
            stop = upper - half_step
        return cls(axis, start, stop, num)


"""
Defers wrapping function with validate_call until class is fully instantiated
"""
Line.bounded = validate_call(Line.bounded)  # type:ignore


@dataclass(config=StrictConfig)
class Fly(Spec[Axis]):
    """Move through lower to upper bounds of the Spec rather than stopping.

    This is commonly termed a "fly scan" rather than a "step scan"

    .. example_spec::

        from scanspec.specs import Fly, Line

        spec = Fly(Line("x", 1, 2, 3))
    """

    spec: Spec[Axis] = Field(description="Spec contaning the path to be followed")

    def axes(self) -> list[Axis]:  # noqa: D102
        return self.spec.axes()

    def duration(self) -> float | None | Literal["VARIABLE_DURATION"]:  # noqa: D102
        return self.spec.duration()

    def calculate(  # noqa: D102
        self, bounds: bool = False, nested: bool = False
    ) -> list[Dimension[Axis]]:
        return self.spec.calculate(bounds=True, nested=nested)


@dataclass(config=StrictConfig)
class ConstantDuration(Spec[Axis]):
    """Apply a constant duration to every point in a Spec.

    Typically applied with the ``@`` modifier.

    .. example_spec::

        from scanspec.specs import Line

        spec = 0.1 @ Line("x", 1, 2, 3)
    """

    constant_duration: float = Field(description="The value at each point")
    spec: Spec[Axis] | None = Field(
        description="Spec contaning the path to be followed", default=None
    )

    def axes(self) -> list[Axis]:  # noqa: D102
        if self.spec:
            return self.spec.axes()
        else:
            return []

    def duration(self) -> float | None | Literal["VARIABLE_DURATION"]:  # noqa: D102
        if self.spec and self.spec.duration() is not None:
            raise ValueError(f"{self.spec} already defines a duration")
        return self.constant_duration

    def calculate(  # noqa: D102
        self, bounds: bool = False, nested: bool = False
    ) -> list[Dimension[Axis]]:
        if self.spec:
            dimensions = self.spec.calculate(bounds=bounds)
            dimensions[-1].duration = np.full(
                len(dimensions[-1].gap),
                self.constant_duration,
            )
            return dimensions
        else:
            # Had to do it like this otherwise it will complain about typing
            empty_dim: Dimension[Axis] = Dimension(
                {},
                {},
                {},
                None,
                duration=np.full(1, self.constant_duration),
            )
            return [empty_dim]


@dataclass(config=StrictConfig)
class Static(Spec[Axis]):
    """A static frame, repeated num times, with axis at value.

    Can be used to set axis=value at every point in a scan.

    .. example_spec::

        from scanspec.specs import Fly, Line, Static

        spec = Fly(Line("y", 1, 2, 3).zip(Static("x", 3)))
    """

    axis: Axis = Field(description="An identifier for what to move")
    value: float = Field(description="The value at each point")
    num: int = Field(ge=1, description="Number of frames to produce", default=1)

    def axes(self) -> list[Axis]:  # noqa: D102
        return [self.axis]

    def _repeats_from_indexes(
        self, indexes: npt.NDArray[np.float64]
    ) -> dict[Axis, npt.NDArray[np.float64]]:
        return {self.axis: np.full(len(indexes), self.value)}

    def calculate(  # noqa: D102
        self, bounds: bool = False, nested: bool = False
    ) -> list[Dimension[Axis]]:
        return _dimensions_from_indexes(
            self._repeats_from_indexes, self.axes(), self.num, bounds
        )


@dataclass(config=StrictConfig)
class Spiral(Spec[Axis]):
    """Archimedean spiral of "x_axis" and "y_axis".

    Starts at centre point ("x_start", "y_start") with angle "rotate". Produces
    "num" points in a spiral spanning width of "x_range" and height of "y_range"

    .. example_spec::

        from scanspec.specs import Fly, Spiral

        spec = Fly(Spiral("x", "y", 1, 5, 10, 50, 30))
    """

    # TODO: Make use of typing.Annotated upon fix of
    # https://github.com/pydantic/pydantic/issues/3496
    x_axis: Axis = Field(description="An identifier for what to move for x")
    y_axis: Axis = Field(description="An identifier for what to move for y")
    x_start: float = Field(description="x centre of the spiral")
    y_start: float = Field(description="y centre of the spiral")
    x_range: float = Field(description="x width of the spiral")
    y_range: float = Field(description="y width of the spiral")
    num: int = Field(ge=1, description="Number of frames to produce")
    rotate: float = Field(
        description="How much to rotate the angle of the spiral", default=0.0
    )

    def axes(self) -> list[Axis]:  # noqa: D102
        # TODO: reversed from __init__ args, a good idea?
        return [self.y_axis, self.x_axis]

    def _spiral_from_indexes(
        self, indexes: npt.NDArray[np.float64]
    ) -> dict[Axis, npt.NDArray[np.float64]]:
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
            self.y_axis: self.y_start + y_scale * phi * np.cos(phi + self.rotate),
            self.x_axis: self.x_start + x_scale * phi * np.sin(phi + self.rotate),
        }

    def calculate(  # noqa: D102
        self, bounds: bool = False, nested: bool = False
    ) -> list[Dimension[Axis]]:
        return _dimensions_from_indexes(
            self._spiral_from_indexes, self.axes(), self.num, bounds
        )

    @classmethod
    def spaced(
        cls: type[Spiral[Any]],
        x_axis: OtherAxis = Field(description="An identifier for what to move for x"),
        y_axis: OtherAxis = Field(description="An identifier for what to move for y"),
        x_start: float = Field(description="x centre of the spiral"),
        y_start: float = Field(description="y centre of the spiral"),
        radius: float = Field(description="radius of the spiral"),
        dr: float = Field(description="difference between each ring"),
        rotate: float = Field(
            description="How much to rotate the angle of the spiral", default=0.0
        ),
    ) -> Spiral[OtherAxis]:
        """Specify a Spiral equally spaced in "x_axis" and "y_axis".

        .. example_spec::

            from scanspec.specs import Fly, Spiral

            spec = Fly(Spiral.spaced("x", "y", 0, 0, 10, 3))
        """
        # phi = sqrt(4 * pi * num)
        # and: n_rings = phi / (2 * pi)
        # so: n_rings * 2 * pi = sqrt(4 * pi * num)
        # so: num = n_rings^2 * pi
        n_rings = radius / dr
        num = int(n_rings**2 * np.pi)
        return cls(
            x_axis,
            y_axis,
            x_start,
            y_start,
            radius * 2,
            radius * 2,
            num,
            rotate,
        )


Spiral.spaced = validate_call(Spiral.spaced)  # type:ignore


def fly(spec: Spec[Axis], duration: float) -> Spec[Axis | str]:
    """Flyscan, zipping with fixed duration for every frame.

    Args:
        spec: The source `Spec` to continuously move
        duration: How long to spend at each frame in the spec

    .. deprecated:: 1.0.0

        You should use `Fly` and `ConstantDuration` instead

    """
    warnings.warn(
        f"fly method is deprecated! Use Fly({duration} @ spec) instead",
        DeprecationWarning,
        stacklevel=2,
    )

    return Fly(duration @ spec)


def step(spec: Spec[Axis], duration: float, num: int = 1) -> Spec[Axis]:
    """Step scan, with num frames of given duration at each frame in the spec.

    Args:
        spec: The source `Spec` with midpoints to move to and stop
        duration: The duration of each scan frame
        num: Number of frames to produce with given duration at each of frame
            in the spec

    .. deprecated:: 1.0.0

        You should use `ConstantDuration` instead.

    """
    warnings.warn(
        f"step method is deprecated! Use {duration} @ spec instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return duration @ spec


def get_constant_duration(frames: list[Dimension[Any]]) -> float | None:
    """Returns the duration of a number of ScanSpec frames, if known and consistent.

    Args:
        frames (List[Dimension]): A number of Frame objects

    Returns:
        duration (float): if all frames have a consistent duration
        None: otherwise

    """
    warnings.warn(
        "get_constant_duration method is deprecated! Use spec.duration() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    duration_frames = [f.duration for f in frames if f.duration is not None]
    if len(duration_frames) == 0:
        # List of frames has no frame with duration in it
        return None
    # First element of the first duration array
    first_duration = duration_frames[0][0]
    for frame in duration_frames:
        if np.any(frame != first_duration):
            # Not all durations are the same
            return None
    return first_duration
