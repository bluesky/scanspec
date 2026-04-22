"""Motion spec nodes and Acquire for scanspec 2.0.

All Spec nodes are pydantic BaseModels.  ``type`` is a computed field on
``Spec`` returning ``type(self).__name__`` — no per-subclass literal needed.

Positional constructor args are supported automatically via ``PosargsMeta``,
a custom metaclass that wraps the pydantic-generated ``__init__`` at
class-creation time.  ``@dataclass_transform`` on the metaclass informs
pyright so positional calls are accepted without per-class stub overrides.

There is a single ``AnySpec`` union covering all spec subclasses.  At
type-check time ``AnySpec`` is simply the base ``Spec`` class, so generic
type parameters flow naturally through combinator fields.  At runtime the
union is built dynamically from ``_recursive_subclasses(Spec)`` — no
hardcoded list to keep in sync when new subclasses are added.
"""

from __future__ import annotations

from abc import ABCMeta
from collections.abc import Callable, Iterator, Sequence
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Generic,
    Never,
    Self,
    TypeAlias,
    TypeVar,
    Union,
    cast,
    dataclass_transform,
)

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    GetCoreSchemaHandler,
    Tag,
    computed_field,
    model_validator,
)

from .core import (
    ContinuousStream,
    DetectorGroup,
    Dimension,
    LinearPositions,
    MonitorStream,
    Scan,
    WindowedStream,
)

AxisT = TypeVar("AxisT")
DetectorT = TypeVar("DetectorT")
MonitorT = TypeVar("MonitorT")


def _discriminate_by_type(obj: Any) -> str | None:
    """Callable discriminator: read ``type`` from dict or return class name."""
    if isinstance(obj, dict):
        return cast(dict[str, str], obj).get("type")
    return type(obj).__name__


def _recursive_subclasses(cls: type) -> Iterator[type]:
    """Yield all transitive concrete subclasses of *cls*."""
    for sub in cls.__subclasses__():
        yield sub
        yield from _recursive_subclasses(sub)


# ---------------------------------------------------------------------------
# Metaclass
# ---------------------------------------------------------------------------


@dataclass_transform(field_specifiers=(Field,))
class PosargsMeta(type(BaseModel), ABCMeta):
    """Metaclass that patches each Spec subclass to accept positional args.

    ``@dataclass_transform`` tells pyright that subclasses behave like
    dataclasses, giving correct type inference for positional constructor calls.
    """

    def __new__(  # noqa: D102
        mcs,
        cls_name: str,
        bases: tuple[type[Any], ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> type:
        cls: type[BaseModel] = super().__new__(
            mcs, cls_name, bases, namespace, **kwargs
        )
        original_init = cls.__init__

        def patched_init(self: BaseModel, *args: Any, **kwargs: Any) -> None:
            for k, v in zip(cls.model_fields, args, strict=False):
                kwargs[k] = v
            original_init(self, **kwargs)

        cls.__init__ = patched_init
        return cls


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Spec(BaseModel, Generic[AxisT, DetectorT, MonitorT], metaclass=PosargsMeta):
    """Base class for all scan specs.

    All subclasses carry the same three type parameters:
    ``AxisT`` — axis identifier type,
    ``DetectorT`` — detector identifier type (``Never`` for pure motion specs),
    ``MonitorT`` — monitor identifier type (``Never`` for pure motion specs).

    Positional constructor args are supported on all subclasses via
    ``PosargsMeta``.  The ``type`` computed field is included in JSON
    serialisation and drives the discriminated-union deserialiser.
    """

    model_config = ConfigDict(frozen=True)

    @computed_field
    @property
    def type(self) -> str:
        """Discriminator field: the concrete class name."""
        return type(self).__name__

    def __mul__(
        self, other: Spec[AxisT, DetectorT, MonitorT]
    ) -> Product[AxisT, DetectorT, MonitorT]:
        """``self * other`` -> ``Product(outer=self, inner=other)``."""
        return Product(outer=self, inner=other)

    def __invert__(self) -> Snake[AxisT, DetectorT, MonitorT]:
        """``~self`` -> ``Snake(self)``."""
        return Snake(spec=self)

    def zip(
        self, other: Spec[AxisT, DetectorT, MonitorT]
    ) -> Zip[AxisT, DetectorT, MonitorT]:
        """Interleave axes of ``self`` and ``other`` into one shared dimension."""
        return Zip(left=self, right=other)

    def concat(
        self, other: Spec[AxisT, DetectorT, MonitorT]
    ) -> Concat[AxisT, DetectorT, MonitorT]:
        """Concatenate ``self`` then ``other``."""
        return Concat(left=self, right=other)

    def compile(self) -> Scan[AxisT, DetectorT, MonitorT]:
        """Compile this spec into a Scan.  Subclasses override."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# AnySpec — single discriminated union covering all subclasses.
#
# At TYPE_CHECKING time AnySpec is simply the bare Spec class (no type args).
# AnySpec: TypeAlias = Spec[AxisT, DetectorT, MonitorT]
# so AnySpec[T, D, M] resolves to Spec[T, D, M] for pyright.
#
# At runtime AnySpec is a subscriptable class whose __class_getitem__ ignores
# type params and returns the real discriminated union (so pydantic uses the
# union for validation), and __get_pydantic_core_schema__ makes TypeAdapter
# work directly.  The union is built at the bottom of the module after all
# subclasses are defined.
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    AnySpec: TypeAlias = Spec[AxisT, DetectorT, MonitorT]
# Runtime AnySpec class defined at the bottom of this module.


# ---------------------------------------------------------------------------
# compile() helpers
# ---------------------------------------------------------------------------


def _motion_dims(spec: Spec[AxisT, Any, Any]) -> list[Dimension[AxisT]]:
    """Return the motion dimensions from a compiled Scan."""
    return list(spec.compile().dimensions)


# ---------------------------------------------------------------------------
# Motion primitives
# ---------------------------------------------------------------------------


class Linspace(Spec[AxisT, Never, Never]):
    """Evenly-spaced sweep of one axis."""

    axis: AxisT = Field(description="Axis identifier.")
    start: float = Field(description="Midpoint of the first setpoint.")
    stop: float = Field(description="Midpoint of the last setpoint.")
    num: int = Field(description="Number of setpoints (>= 1).", ge=1)

    def compile(self) -> Scan[AxisT, Never, Never]:
        """Compile into a one-dimension Scan with a linear position function."""
        dim = Dimension(
            axes=[self.axis],
            length=self.num,
            snake=False,
            position_fn=LinearPositions(
                axis_ranges={self.axis: (self.start, self.stop)},
                length=self.num,
            ),
        )
        return Scan(motion_dims=[dim])


class Static(Spec[AxisT, Never, Never]):
    """Single static position for one axis."""

    axis: AxisT = Field(description="Axis identifier.")
    value: float = Field(description="The fixed position.")
    num: int = Field(
        default=1,
        description="How many times this position appears (>= 1).",
        ge=1,
    )

    def compile(self) -> Scan[AxisT, Never, Never]:
        """Compile into a one-dimension Scan with a constant position function."""
        dim = Dimension(
            axes=[self.axis],
            length=self.num,
            snake=False,
            position_fn=LinearPositions(
                axis_ranges={self.axis: (self.value, self.value)},
                length=self.num,
            ),
        )
        return Scan(motion_dims=[dim])


class Spiral(Spec[AxisT, Never, Never]):
    """Archimedean spiral of *x_axis* and *y_axis*.

    Starts at (*x_centre*, *y_centre*).  Produces points in a spiral
    spanning a width of *x_diameter* and height of *y_diameter*
    (defaults to ``abs(x_diameter)``), with rings spaced *x_step* apart.

    Point spacing matches the original scanspec Spiral: integer indexes
    0 … num-1 are shifted by 0.5 internally so they correspond to the
    midpoints of the spiral bands (same convention as 1.x
    ``_dimensions_from_indexes``).
    """

    x_axis: AxisT = Field(description="Axis identifier for x.")
    x_centre: float = Field(description="x centre of the spiral.")
    x_diameter: float = Field(description="x width of the spiral.")
    x_step: float = Field(description="Radial spacing along x.")
    y_axis: AxisT = Field(description="Axis identifier for y.")
    y_centre: float = Field(description="y centre of the spiral.")
    y_diameter: float | None = Field(
        default=None,
        description="y height of the spiral (defaults to abs(x_diameter)).",
    )

    def _eff_y_diameter(self) -> float:
        return self.y_diameter if self.y_diameter is not None else abs(self.x_diameter)

    def _num_points(self) -> int:
        ellipse_area = np.pi * self.x_diameter * self._eff_y_diameter() / 4
        return int(ellipse_area / self.x_step**2) + 1

    def compile(self) -> Scan[AxisT, Never, Never]:
        """Compile into a one-dimension Scan with a spiral position function."""
        num = self._num_points()
        yd = self._eff_y_diameter()
        # diameter of the spiral at index num (outermost ring)
        diameter = 2 * np.sqrt(4 * np.pi * num)
        x_scale = self.x_diameter / diameter
        y_scale = yd / diameter
        # capture as locals for the closure
        xc, yc = self.x_centre, self.y_centre
        x_axis, y_axis = self.x_axis, self.y_axis

        def spiral_fn(indexes: np.ndarray) -> dict[AxisT, np.ndarray]:
            # Offset by 1.0 so that integer index i maps to the midpoint of
            # band i+1, keeping the fly-scan start boundary (i=-0.5) at
            # phi=sqrt(2π) — well away from the singular spiral centre at
            # phi=0 where dy/di diverges.  The 0.5 convention used by 1.x
            # (midpoints = linspace(0.5, num-0.5, num)) places the start
            # boundary exactly at phi=0, making start_velocity undefined.
            phi = np.sqrt(4 * np.pi * (indexes + 1.0))
            return {
                y_axis: yc + y_scale * phi * np.cos(phi),
                x_axis: xc + x_scale * phi * np.sin(phi),
            }

        dim = Dimension(
            axes=[self.y_axis, self.x_axis],
            length=num,
            snake=False,
            position_fn=spiral_fn,
        )
        return Scan(motion_dims=[dim])


# ---------------------------------------------------------------------------
# Combinators — accept any Spec (including Acquire)
# ---------------------------------------------------------------------------


class Repeat(Spec[AxisT, DetectorT, MonitorT]):
    """Repeat an inner spec a fixed number of times as an outer dimension."""

    spec: AnySpec[AxisT, DetectorT, MonitorT] = Field(
        description="Inner spec to repeat."
    )
    num: int = Field(description="Number of repetitions (>= 1).", ge=1)

    def compile(self) -> Scan[AxisT, DetectorT, MonitorT]:
        """Compile by prepending a new outer dimension of length *num*."""
        inner_dims = _motion_dims(self.spec)
        empty: dict[AxisT, tuple[float, float]] = {}
        new_dim: Dimension[AxisT] = Dimension(
            axes=[],
            length=self.num,
            snake=False,
            position_fn=LinearPositions(axis_ranges=empty, length=self.num),
        )
        return Scan(motion_dims=[new_dim] + inner_dims)


class Snake(Spec[AxisT, DetectorT, MonitorT]):
    """Reverse alternate repeats of an inner spec (snake/boustrophedon)."""

    spec: AnySpec[AxisT, DetectorT, MonitorT] = Field(
        description="Inner spec to snake."
    )

    def compile(self) -> Scan[AxisT, DetectorT, MonitorT]:
        """Compile by setting snake=True on the innermost dimension."""
        inner_dims = _motion_dims(self.spec)
        if not inner_dims:
            return Scan(motion_dims=inner_dims)
        last = inner_dims[-1]
        inner_dims[-1] = Dimension(
            axes=last.axes,
            length=last.length,
            snake=True,
            position_fn=last.position_fn,
        )
        return Scan(motion_dims=inner_dims)


class Product(Spec[AxisT, DetectorT, MonitorT]):
    """Outer x inner product: ``outer`` is slow, ``inner`` is fast."""

    outer: AnySpec[AxisT, DetectorT, MonitorT] = Field(description="Slow (outer) spec.")
    inner: AnySpec[AxisT, DetectorT, MonitorT] = Field(description="Fast (inner) spec.")

    def compile(self) -> Scan[AxisT, DetectorT, MonitorT]:
        """Compile by concatenating outer dimensions before inner dimensions."""
        outer_dims = _motion_dims(self.outer)
        inner_dims = _motion_dims(self.inner)
        return Scan(motion_dims=outer_dims + inner_dims)


class Zip(Spec[AxisT, DetectorT, MonitorT]):
    """Merge two specs into one shared dimension (both must have the same length)."""

    left: AnySpec[AxisT, DetectorT, MonitorT] = Field(description="First spec.")
    right: AnySpec[AxisT, DetectorT, MonitorT] = Field(
        description="Second spec (must have the same length as left)."
    )

    def compile(self) -> Scan[AxisT, DetectorT, MonitorT]:
        """Compile by merging the innermost dimensions of left and right."""
        left_dims = _motion_dims(self.left)
        right_dims = _motion_dims(self.right)
        if not left_dims or not right_dims:
            raise ValueError("Zip requires both specs to have at least one dimension")
        l_inner = left_dims[-1]
        r_inner = right_dims[-1]
        if l_inner.length != r_inner.length:
            raise ValueError(
                f"Zip requires equal inner dimension lengths; "
                f"got {l_inner.length} and {r_inner.length}"
            )
        # Merge the two innermost dimensions.
        merged_positions: Callable[[np.ndarray], dict[Any, np.ndarray]]
        l_fn = l_inner.position_fn
        r_fn = r_inner.position_fn
        if isinstance(l_fn, LinearPositions) and isinstance(r_fn, LinearPositions):
            l_ranges = cast(LinearPositions[AxisT], l_fn).axis_ranges
            r_ranges = cast(LinearPositions[AxisT], r_fn).axis_ranges
            merged_positions = LinearPositions(
                axis_ranges={**l_ranges, **r_ranges},
                length=l_inner.length,
            )
        else:

            def _merged(indexes: np.ndarray) -> dict[Any, np.ndarray]:
                result: dict[Any, np.ndarray] = {}
                result.update(l_inner(indexes))
                result.update(r_inner(indexes))
                return result

            merged_positions = _merged
        merged_dim: Dimension[AxisT] = Dimension(
            axes=l_inner.axes + r_inner.axes,
            length=l_inner.length,
            snake=l_inner.snake or r_inner.snake,
            position_fn=merged_positions,
        )
        dims = left_dims[:-1] + [merged_dim]
        return Scan(motion_dims=dims)


class Concat(Spec[AxisT, DetectorT, MonitorT]):
    """Concatenate two specs sequentially.

    Both *left* and *right* may carry detector configuration — useful for
    combining two ``Acquire`` nodes that differ in detector setup.
    """

    left: AnySpec[AxisT, DetectorT, MonitorT] = Field(description="First spec.")
    right: AnySpec[AxisT, DetectorT, MonitorT] = Field(
        description="Second spec (appended after left)."
    )

    def compile(self) -> Scan[AxisT, DetectorT, MonitorT]:
        """Compile by concatenating the innermost dimensions of left and right."""
        left_dims = _motion_dims(self.left)
        right_dims = _motion_dims(self.right)
        if not left_dims or not right_dims:
            raise ValueError(
                "Concat requires both specs to have at least one dimension"
            )
        l_inner = left_dims[-1]
        r_inner = right_dims[-1]
        # Outer dimensions must match (same axes and lengths).
        if left_dims[:-1] and right_dims[:-1]:
            l_outer_axes = [d.axes for d in left_dims[:-1]]
            r_outer_axes = [d.axes for d in right_dims[:-1]]
            if l_outer_axes != r_outer_axes:
                raise ValueError(
                    "Concat: outer dimensions must match between left and right"
                )
        # Build a merged inner dimension: same axes, summed length.
        if l_inner.axes != r_inner.axes:
            raise ValueError(
                f"Concat: innermost axes must match; "
                f"got {l_inner.axes} vs {r_inner.axes}"
            )
        l_len = l_inner.length
        r_len = r_inner.length
        total_len = l_len + r_len

        def concat_fn(indexes: np.ndarray) -> dict[AxisT, np.ndarray]:
            # indexes in [0, total_len); first l_len belong to left, rest to right.
            left_mask = indexes < l_len
            right_mask = ~left_mask
            result: dict[AxisT, np.ndarray] = {}
            if np.any(left_mask):
                for axis, arr in l_inner(indexes[left_mask]).items():
                    result[axis] = np.empty(len(indexes), dtype=arr.dtype)
                    result[axis][left_mask] = arr
            if np.any(right_mask):
                for axis, arr in r_inner(indexes[right_mask] - l_len).items():
                    if axis not in result:
                        dummy = l_inner(np.array([0.0]))[axis]
                        result[axis] = np.empty(len(indexes), dtype=dummy.dtype)
                    result[axis][right_mask] = arr
            return result

        merged_dim: Dimension[AxisT] = Dimension(
            axes=l_inner.axes,
            length=total_len,
            snake=l_inner.snake,
            position_fn=concat_fn,
        )
        dims = left_dims[:-1] + [merged_dim]
        return Scan(motion_dims=dims)


class Acquire(Spec[AxisT, DetectorT, MonitorT]):
    """Outermost spec node: binds detector triggering to a motion spec.

    ``Acquire`` instances can be combined with ``+`` to concatenate two
    detector configurations for the same scan path::

        acq1 + acq2  ->  Concat(acq1, acq2)
    """

    spec: AnySpec[AxisT, Any, Any] = Field(
        description="Inner spec (motion or nested Acquire.)."
    )
    fly: bool = Field(
        default=False,
        description="True for flyscan (innermost dimension sweeps continuously).",
    )
    stream_name: str = Field(
        default="primary",
        description="Bluesky stream name.",
    )
    detectors: Sequence[DetectorGroup[DetectorT]] = Field(
        default_factory=tuple,
        description="DetectorGroups for the windowed stream.",
    )
    continuous_streams: Sequence[ContinuousStream[DetectorT]] = Field(
        default_factory=tuple,
        description="Continuously-acquired detector groups (no scan dimensions).",
    )
    monitors: Sequence[MonitorStream[MonitorT]] = Field(
        default_factory=tuple,
        description="Free-running PV monitors.",
    )

    @model_validator(mode="after")
    def _validate_unique_detectors(self) -> Self:
        """All detector names must be globally unique within this Acquire."""
        seen: set[object] = set()
        duplicates: set[object] = set()
        for dg in self.detectors:
            for d in dg.detectors:
                if d in seen:
                    duplicates.add(d)
                seen.add(d)
        for cs in self.continuous_streams:
            for dg in cs.detector_groups:
                for d in dg.detectors:
                    if d in seen:
                        duplicates.add(d)
                    seen.add(d)
        for ms in self.monitors:
            if ms.detector in seen:
                duplicates.add(ms.detector)
            seen.add(ms.detector)
        if duplicates:
            raise ValueError(
                f"Detector names must be unique across all groups; "
                f"duplicates: {sorted(str(d) for d in duplicates)}"
            )
        return self

    def __add__(
        self, other: Acquire[AxisT, DetectorT, MonitorT]
    ) -> Concat[AxisT, DetectorT, MonitorT]:
        """``acq1 + acq2`` -> ``Concat(acq1, acq2)``."""
        return Concat(left=self, right=other)

    def compile(self) -> Scan[AxisT, DetectorT, MonitorT]:
        """Compile into a Scan with detector groups and motion dims."""
        inner_dims = _motion_dims(self.spec)
        stream = WindowedStream(
            name=self.stream_name,
            dimensions=inner_dims,
            detector_groups=list(self.detectors),
        )
        return Scan(
            motion_dims=inner_dims,
            windowed_streams=[stream],
            continuous_streams=list(self.continuous_streams),
            monitors=list(self.monitors),
            fly=self.fly,
        )


# ---------------------------------------------------------------------------
# Build AnySpec union now that all subclasses are defined
# ---------------------------------------------------------------------------

if not TYPE_CHECKING:
    # Filter out pydantic-internal parametrized classes
    # (e.g. "Spec[~AxisT, Never, Never]").
    _subclasses = [s for s in _recursive_subclasses(Spec) if "[" not in s.__name__]
    _ANYSPEC_UNION = Annotated[
        Union[tuple(Annotated[sub, Tag(sub.__name__)] for sub in _subclasses)],  # noqa: UP007
        Discriminator(_discriminate_by_type),
    ]

    class AnySpec:
        """Runtime subscriptable sentinel for AnySpec.

        At TYPE_CHECKING time, ``AnySpec = Spec`` (the bare base class), so
        ``AnySpec[T, D, M]`` resolves to ``Spec[T, D, M]`` for pyright.

        At runtime this class is used because the discriminated union cannot be
        subscripted with TypeVars.  ``__class_getitem__`` ignores its params
        and returns the real union so pydantic uses the correct schema.
        ``__get_pydantic_core_schema__`` makes ``TypeAdapter(AnySpec)`` work.
        """

        @classmethod
        def __class_getitem__(cls, params: Any) -> Any:
            """Return the pydantic discriminated union regardless of params."""
            return _ANYSPEC_UNION

        @classmethod
        def __get_pydantic_core_schema__(
            cls, source_type: Any, handler: GetCoreSchemaHandler
        ) -> Any:
            """Delegate to the union schema so TypeAdapter(AnySpec) works."""
            return handler(_ANYSPEC_UNION)

    # Pass AnySpec into the rebuild namespace so forward-ref fields resolve correctly.
    for _sub in _subclasses:
        _sub.model_rebuild(_types_namespace={"AnySpec": AnySpec})
