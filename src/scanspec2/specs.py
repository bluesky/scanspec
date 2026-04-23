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
from collections.abc import Iterator, Sequence
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
    MonitorStream,
    Scan,
    TriggerGroup,
    TriggerPattern,
    WindowedStream,
    WindowGenerator,
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
        """Compile this spec into a Scan.  Subclasses override.

        The returned Scan is owned by the caller, who is free to read or
        modify any of its structures (generators, windowed_streams, etc.).
        Consequently, ``compile()`` implementations may mutate the Scan
        produced by an inner ``spec.compile()`` call rather than
        constructing new wrapper objects.
        """
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
        gen = WindowGenerator(
            axes=[self.axis],
            length=self.num,
            axis_ranges={self.axis: (self.start, self.stop)},
        )
        return Scan(generators=[gen])


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
        gen = WindowGenerator(
            axes=[self.axis],
            length=self.num,
            axis_ranges={self.axis: (self.value, self.value)},
        )
        return Scan(generators=[gen])


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
        # The outermost midpoint is at index num-0.5; offset by 0.5 gives
        # an effective spiral index of num.  diameter is 2*phi at that point.
        diameter = 2 * np.sqrt(4 * np.pi * num)
        x_scale = self.x_diameter / diameter
        y_scale = yd / diameter
        # capture as locals for the closure
        xc, yc = self.x_centre, self.y_centre
        x_axis, y_axis = self.x_axis, self.y_axis

        def spiral_fn(indexes: np.ndarray) -> dict[AxisT, np.ndarray]:
            # Uses the 1.x fence/post convention: midpoints at half-integer
            # indexes 0.5..N-0.5, fly boundaries at integer indexes 0..N.
            # Offset by 0.5 so that:
            #   - Fly start boundary (index=0) maps to phi=sqrt(2*pi),
            #     well away from the singular spiral centre at phi=0.
            #   - First midpoint (index=0.5) maps to phi=sqrt(4*pi*1.0).
            phi = np.sqrt(4 * np.pi * (indexes + 0.5))
            return {
                y_axis: yc + y_scale * phi * np.cos(phi),
                x_axis: xc + x_scale * phi * np.sin(phi),
            }

        gen = WindowGenerator(
            axes=[self.y_axis, self.x_axis],
            length=num,
            position_fn=spiral_fn,
        )
        return Scan(generators=[gen])


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
        new_gen: WindowGenerator[AxisT] = WindowGenerator(
            axes=[],
            length=self.num,
            axis_ranges={},
        )
        scan = self.spec.compile()
        new_dim: Dimension[AxisT] = Dimension(
            axes=[], length=self.num, snake=False, position_fn=new_gen.setpoints
        )
        for ws in scan.windowed_streams:
            ws.dimensions.insert(0, new_dim)
        scan.generators.insert(0, new_gen)
        return scan


class Snake(Spec[AxisT, DetectorT, MonitorT]):
    """Reverse alternate repeats of an inner spec (snake/boustrophedon)."""

    spec: AnySpec[AxisT, DetectorT, MonitorT] = Field(
        description="Inner spec to snake."
    )

    def compile(self) -> Scan[AxisT, DetectorT, MonitorT]:
        """Compile by setting snake=True on the innermost generator."""
        scan = self.spec.compile()
        if not scan.generators:
            raise ValueError("Snake requires at least one generator")
        scan.generators[-1].snake = True
        return scan


class Product(Spec[AxisT, DetectorT, MonitorT]):
    """Outer x inner product: ``outer`` is slow, ``inner`` is fast."""

    outer: AnySpec[AxisT, DetectorT, MonitorT] = Field(description="Slow (outer) spec.")
    inner: AnySpec[AxisT, DetectorT, MonitorT] = Field(description="Fast (inner) spec.")

    def compile(self) -> Scan[AxisT, DetectorT, MonitorT]:
        """Compile by prepending outer generators before inner generators."""
        outer_scan = self.outer.compile()
        inner_scan = self.inner.compile()
        outer_dims = [
            Dimension(
                axes=g.axes,
                length=g.length,
                snake=g.snake,
                position_fn=g.setpoints,
            )
            for g in outer_scan.generators
        ]
        for ws in inner_scan.windowed_streams:
            ws.dimensions[:0] = outer_dims
        inner_scan.generators[:0] = outer_scan.generators
        return inner_scan


class Zip(Spec[AxisT, DetectorT, MonitorT]):
    """Merge two specs into one shared dimension (both must have the same length)."""

    left: AnySpec[AxisT, DetectorT, MonitorT] = Field(description="First spec.")
    right: AnySpec[AxisT, DetectorT, MonitorT] = Field(
        description="Second spec (must have the same length as left)."
    )

    def compile(self) -> Scan[AxisT, DetectorT, MonitorT]:
        """Compile by merging the innermost generators of left and right."""
        left_scan = self.left.compile()
        right_scan = self.right.compile()
        if not left_scan.generators or not right_scan.generators:
            raise ValueError("Zip requires both specs to have at least one dimension")
        l_inner = left_scan.generators[-1]
        r_inner = right_scan.generators[-1]
        if l_inner.length != r_inner.length:
            raise ValueError(
                f"Zip requires equal inner dimension lengths; "
                f"got {l_inner.length} and {r_inner.length}"
            )
        if l_inner.snake != r_inner.snake:
            raise ValueError(
                f"Zip requires matching snake flags on inner generators; "
                f"got {l_inner.snake} and {r_inner.snake}"
            )
        # Merge the two innermost generators.
        if l_inner.axis_ranges is not None and r_inner.axis_ranges is not None:
            merged_gen: WindowGenerator[AxisT] = WindowGenerator(
                axes=l_inner.axes + r_inner.axes,
                length=l_inner.length,
                snake=l_inner.snake,
                axis_ranges={**l_inner.axis_ranges, **r_inner.axis_ranges},
            )
        else:

            def _merged(indexes: np.ndarray) -> dict[Any, np.ndarray]:
                result: dict[Any, np.ndarray] = {}
                result.update(l_inner.setpoints(indexes))
                result.update(r_inner.setpoints(indexes))
                return result

            merged_gen = WindowGenerator(
                axes=l_inner.axes + r_inner.axes,
                length=l_inner.length,
                snake=l_inner.snake,
                position_fn=_merged,
            )
        left_scan.generators[-1] = merged_gen
        return left_scan


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
        """Compile by concatenating left and right.

        Two modes:
        - Pure motion (no windowed_streams): merge innermost generators
          into one generator with a combined position function.
        - Detector-aware (any side has windowed_streams or children):
          create a concat WindowGenerator whose children are iterated
          sequentially, and merge windowed_streams.
        """
        left_scan = self.left.compile()
        right_scan = self.right.compile()

        left_has_streams = bool(left_scan.windowed_streams)
        right_has_streams = bool(right_scan.windowed_streams)
        left_has_children = (
            bool(left_scan.generators) and left_scan.generators[-1].children is not None
        )
        right_has_children = (
            bool(right_scan.generators)
            and right_scan.generators[-1].children is not None
        )

        if (
            left_has_streams
            or right_has_streams
            or left_has_children
            or right_has_children
        ):
            return self._compile_concat(left_scan, right_scan)
        return self._compile_merged(left_scan, right_scan)

    def _compile_merged(
        self,
        left_scan: Scan[AxisT, DetectorT, MonitorT],
        right_scan: Scan[AxisT, DetectorT, MonitorT],
    ) -> Scan[AxisT, DetectorT, MonitorT]:
        """Merge innermost generators (pure motion concat)."""
        if len(left_scan.generators) != 1 or len(right_scan.generators) != 1:
            raise ValueError("Concat requires both specs to have exactly one generator")
        l_inner = left_scan.generators[0]
        r_inner = right_scan.generators[0]
        if l_inner.axes != r_inner.axes:
            raise ValueError(
                f"Concat: innermost axes must match; "
                f"got {l_inner.axes} vs {r_inner.axes}"
            )
        l_len = l_inner.length
        r_len = r_inner.length
        total_len = l_len + r_len

        def concat_fn(indexes: np.ndarray) -> dict[AxisT, np.ndarray]:
            left_mask = indexes < l_len
            right_mask = ~left_mask
            result: dict[AxisT, np.ndarray] = {}
            if np.any(left_mask):
                for axis, arr in l_inner.setpoints(indexes[left_mask]).items():
                    result[axis] = np.empty(len(indexes), dtype=arr.dtype)
                    result[axis][left_mask] = arr
            if np.any(right_mask):
                shifted = indexes[right_mask] - l_len
                for axis, arr in r_inner.setpoints(shifted).items():
                    if axis not in result:
                        dummy = l_inner.setpoints(np.array([0.0]))[axis]
                        result[axis] = np.empty(len(indexes), dtype=dummy.dtype)
                    result[axis][right_mask] = arr
            return result

        merged_gen: WindowGenerator[AxisT] = WindowGenerator(
            axes=l_inner.axes,
            length=total_len,
            snake=l_inner.snake,
            position_fn=concat_fn,
        )
        left_scan.generators[:] = [merged_gen]
        return left_scan

    @staticmethod
    def _scan_to_children(
        scan: Scan[AxisT, DetectorT, MonitorT],
    ) -> list[WindowGenerator[AxisT]]:
        """Extract leaf WindowGenerators from a Scan for concat children.

        Each side of a Concat must have exactly one generator.  If that
        generator already has children (from a nested Concat), flatten them;
        otherwise use the generator itself.

        Supporting multi-generator sides would require squashing generators
        and stream dimensions together (the squash_dimensions algorithm from
        scanspec 1.x).  This adds significant complexity and has never been
        needed in practice, so it is not implemented.
        """
        gens = scan.generators
        if len(gens) != 1:
            raise ValueError("Concat requires each side to have exactly one generator")
        gen = gens[0]
        if gen.children is not None:
            return list(gen.children)
        return [gen]

    def _compile_concat(
        self,
        left_scan: Scan[AxisT, DetectorT, MonitorT],
        right_scan: Scan[AxisT, DetectorT, MonitorT],
    ) -> Scan[AxisT, DetectorT, MonitorT]:
        """Build a concat WindowGenerator with children."""
        children = self._scan_to_children(left_scan) + self._scan_to_children(
            right_scan
        )
        concat_gen: WindowGenerator[AxisT] = WindowGenerator(
            axes=[],
            length=0,
            axis_ranges={},
            children=children,
        )
        left_scan.generators[:] = [concat_gen]
        # Merge right's windowed streams into left's (sum inner dim lengths)
        left_by_name = {s.name: s for s in left_scan.windowed_streams}
        for stream in right_scan.windowed_streams:
            if stream.name in left_by_name:
                existing = left_by_name[stream.name]
                if existing.dimensions and stream.dimensions:
                    existing.dimensions[-1].length += stream.dimensions[-1].length
            else:
                left_scan.windowed_streams.append(stream)
        for cs in right_scan.continuous_streams:
            if cs not in left_scan.continuous_streams:
                left_scan.continuous_streams.append(cs)
        for m in right_scan.monitors:
            if m not in left_scan.monitors:
                left_scan.monitors.append(m)
        return left_scan


class Acquire(Spec[AxisT, DetectorT, MonitorT]):
    """Outermost spec node: binds detector triggering to a motion spec."""

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
    duration: float | None = Field(
        default=None,
        description=(
            "Per-point duration in seconds. Required for detector-less fly "
            "scans. When detectors are present, duration is derived from "
            "trigger timing. For step scans without detectors, defaults to 0."
        ),
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

    def compile(self) -> Scan[AxisT, DetectorT, MonitorT]:
        """Compile into a Scan with detector groups and generators."""
        scan = self.spec.compile()
        trigger_groups = self._bake_trigger_groups(scan.generators)
        duration = self._compute_duration(trigger_groups, scan.generators)
        if scan.generators:
            last = scan.generators[-1]
            last.fly = self.fly
            last.trigger_groups = trigger_groups
            last.duration = duration
        dims = [
            Dimension(
                axes=g.axes,
                length=g.length,
                snake=g.snake,
                position_fn=g.setpoints,
            )
            for g in scan.generators
        ]

        stream = WindowedStream(
            name=self.stream_name,
            dimensions=dims,
            detector_groups=list(self.detectors),
        )
        scan.windowed_streams = [stream]
        scan.continuous_streams = list(self.continuous_streams)
        scan.monitors = list(self.monitors)
        return scan

    def _bake_trigger_groups(
        self,
        gens: list[WindowGenerator[AxisT]],
    ) -> list[TriggerGroup[DetectorT]]:
        """Convert DetectorGroups into TriggerGroups with TriggerPatterns."""
        if not self.detectors:
            return []
        inner_length = gens[-1].length if gens else 1
        fly = self.fly and bool(gens)
        result: list[TriggerGroup[DetectorT]] = []
        for dg in self.detectors:
            if dg.livetime is None or dg.deadtime is None:
                raise ValueError(
                    f"livetime and deadtime must be set on DetectorGroup "
                    f"before compile(); got livetime={dg.livetime}, "
                    f"deadtime={dg.deadtime}"
                )
            if fly:
                repeats = inner_length * dg.exposures_per_collection
            else:
                repeats = dg.exposures_per_collection
            pattern = TriggerPattern(
                repeats=repeats,
                livetime=dg.livetime,
                deadtime=dg.deadtime,
            )
            result.append(
                TriggerGroup(
                    detectors=list(dg.detectors),
                    trigger_patterns=[pattern],
                )
            )
        return result

    def _compute_duration(
        self,
        trigger_groups: list[TriggerGroup[DetectorT]],
        gens: list[WindowGenerator[AxisT]],
    ) -> float | None:
        """Derive per-point duration from trigger timing.

        Returns per-point duration. For step scans (1 window per setpoint)
        this equals total window duration. For fly scans the total window
        duration is ``per_point * inner_length``; ``Scan.__iter__``
        multiplies automatically.
        """
        if not trigger_groups:
            return self.duration
        total_dur = 0.0
        for tg in trigger_groups:
            tg_dur = sum(
                tp.repeats * (tp.livetime + tp.deadtime) for tp in tg.trigger_patterns
            )
            total_dur = max(total_dur, tg_dur)
        fly = self.fly and bool(gens)
        inner_length = gens[-1].length if fly else 1
        per_point = total_dur / inner_length
        if self.duration is not None:
            if self.duration < per_point:
                raise ValueError(
                    f"Explicit duration {self.duration} is less than "
                    f"detector-derived per-point duration {per_point}"
                )
            return self.duration
        return per_point


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
