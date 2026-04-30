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
    ConcatSource,
    ContinuousStream,
    DetectorGroup,
    Dimension,
    FunctionSource,
    LinearSource,
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
# Dynamic AnySpec union — rebuilt automatically by PosargsMeta.__new__
# ---------------------------------------------------------------------------

# Mutable module-level state for the union.  Updated every time a new
# concrete Spec subclass is created (including out-of-package ones).
_anyspec_union: Any = None
_anyspec_cls: Any = None  # runtime AnySpec sentinel class, set once below


def _maybe_rebuild_anyspec_union(cls: type) -> None:
    """Rebuild ``_anyspec_union`` if *cls* is a new concrete Spec subclass."""
    global _anyspec_union  # noqa: PLW0603
    # Avoid running during Spec base-class creation or for parametrised generics.
    if not hasattr(cls, "model_fields"):
        return
    # Check cls is a Spec subclass (not Spec itself).
    try:
        if cls.__name__ == "Spec" or not issubclass(cls, Spec):
            return
    except TypeError:
        return
    if "[" in cls.__name__:
        return

    subclasses = [s for s in _recursive_subclasses(Spec) if "[" not in s.__name__]
    if len(subclasses) < 2:
        return

    _anyspec_union = Annotated[
        Union[tuple(Annotated[sub, Tag(sub.__name__)] for sub in subclasses)],  # noqa: UP007
        Discriminator(_discriminate_by_type),
    ]

    # Rebuild all existing subclasses so pydantic picks up the updated union.
    ns: dict[str, Any] = {}
    if _anyspec_cls is not None:
        ns["AnySpec"] = _anyspec_cls
    for sub in subclasses:
        rebuild = getattr(sub, "model_rebuild", None)
        if rebuild is not None:
            try:
                rebuild(force=True, _types_namespace=ns)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Metaclass
# ---------------------------------------------------------------------------


@dataclass_transform(field_specifiers=(Field,))
class PosargsMeta(type(BaseModel), ABCMeta):
    """Metaclass that patches each Spec subclass to accept positional args.

    ``@dataclass_transform`` tells pyright that subclasses behave like
    dataclasses, giving correct type inference for positional constructor calls.

    Also rebuilds ``_anyspec_union`` whenever a new concrete Spec subclass is
    created, so that out-of-package Spec subclasses are automatically included
    in the discriminated union used for (de)serialisation.
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

        # Rebuild AnySpec union if this is a concrete Spec subclass.
        _maybe_rebuild_anyspec_union(cls)

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
# At TYPE_CHECKING time AnySpec is simply the bare Spec class (no type args)
# so AnySpec[T, D, M] resolves to Spec[T, D, M] for pyright.
#
# At runtime the AnySpec class is defined at the bottom of this module,
# AFTER all concrete subclasses, so that pydantic defers annotation
# resolution until model_rebuild is called with the complete union.
# Out-of-package subclasses are handled by _maybe_rebuild_anyspec_union
# which fires in PosargsMeta.__new__.
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    AnySpec: TypeAlias = Spec[AxisT, DetectorT, MonitorT]


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
            source=LinearSource({self.axis: (self.start, self.stop)}, self.num),
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
            source=LinearSource({self.axis: (self.value, self.value)}, self.num),
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
            source=FunctionSource(spiral_fn),
        )
        return Scan(generators=[gen])


# ---------------------------------------------------------------------------
# Combinators — accept any Spec (including Acquire)
# ---------------------------------------------------------------------------


def _reject_continuous_and_monitors(scan: Scan[Any, Any, Any], combinator: str) -> None:
    """Raise if *scan* has continuous_streams or monitors.

    Continuous and monitor streams run in parallel to the entire scan and
    must be attached at the outermost ``Acquire``, not nested inside
    combinators.
    """
    if scan.continuous_streams:
        raise ValueError(
            f"{combinator} does not accept specs with continuous_streams; "
            f"attach them on the outermost Acquire instead"
        )
    if scan.monitors:
        raise ValueError(
            f"{combinator} does not accept specs with monitors; "
            f"attach them on the outermost Acquire instead"
        )


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
            source=LinearSource({}, self.num),
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
        if len(scan.generators) != 1:
            raise ValueError(
                f"Snake requires exactly one generator, got {len(scan.generators)}"
            )
        scan.generators[0].snake = True
        return scan


class Product(Spec[AxisT, DetectorT, MonitorT]):
    """Outer x inner product: ``outer`` is slow, ``inner`` is fast."""

    outer: AnySpec[AxisT, DetectorT, MonitorT] = Field(description="Slow (outer) spec.")
    inner: AnySpec[AxisT, DetectorT, MonitorT] = Field(description="Fast (inner) spec.")

    def compile(self) -> Scan[AxisT, DetectorT, MonitorT]:
        """Compile by prepending outer generators before inner generators."""
        outer_scan = self.outer.compile()
        inner_scan = self.inner.compile()
        _reject_continuous_and_monitors(outer_scan, "Product")
        _reject_continuous_and_monitors(inner_scan, "Product")
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
    """Merge two specs into one shared dimension.

    Supports the same cases as 1.x:

    - Both sides have the same number of generators with matching inner
      lengths: merge innermost generators dimension-by-dimension.
    - Right has more generators than left: left-pad left with right's
      extra outer generators.
    - Right has a single generator of length 1 (e.g. ``Static``): expand
      it to match left's innermost generator length.
    """

    left: AnySpec[AxisT, DetectorT, MonitorT] = Field(description="First spec.")
    right: AnySpec[AxisT, DetectorT, MonitorT] = Field(description="Second spec.")

    def compile(self) -> Scan[AxisT, DetectorT, MonitorT]:
        """Compile by merging generators of left and right."""
        left_scan = self.left.compile()
        right_scan = self.right.compile()
        _reject_continuous_and_monitors(left_scan, "Zip")
        _reject_continuous_and_monitors(right_scan, "Zip")
        if not left_scan.generators or not right_scan.generators:
            raise ValueError("Zip requires both specs to have at least one dimension")
        if len(left_scan.generators) < len(right_scan.generators):
            raise ValueError(
                f"Zip requires len(left.generators) >= len(right.generators); "
                f"got {len(left_scan.generators)} and {len(right_scan.generators)}"
            )

        r_gens = list(right_scan.generators)

        # Special case: right is a single generator of length 1 (e.g. Static).
        # Expand it to match left's innermost length.
        if len(r_gens) == 1 and r_gens[0].length == 1:
            l_inner = left_scan.generators[-1]
            r_gen = r_gens[0]
            # Build an expanded generator that repeats the single value.
            if isinstance(r_gen.source, LinearSource):
                expanded_gen: WindowGenerator[AxisT] = WindowGenerator(
                    axes=r_gen.axes,
                    length=l_inner.length,
                    snake=l_inner.snake,
                    source=LinearSource(r_gen.source.axis_ranges, l_inner.length),
                )
            else:
                r_captured = r_gen

                def _expand(indexes: np.ndarray) -> dict[Any, np.ndarray]:
                    # Always evaluate at index 0.5 (the single setpoint)
                    single = r_captured.setpoints(np.array([0.5]))
                    return {
                        ax: np.full(len(indexes), float(arr[0]))
                        for ax, arr in single.items()
                    }

                expanded_gen = WindowGenerator(
                    axes=r_gen.axes,
                    length=l_inner.length,
                    snake=l_inner.snake,
                    source=FunctionSource(_expand),
                )
            r_gens = [expanded_gen]

        # Left-pad r_gens so both lists are the same length.
        npad = len(left_scan.generators) - len(r_gens)
        padded_right: list[WindowGenerator[AxisT] | None] = [None] * npad + r_gens  # type: ignore[list-item]

        # Merge generator-by-generator from outer to inner.
        for i, (l_gen, r_gen_or_none) in enumerate(
            zip(left_scan.generators, padded_right, strict=True)
        ):
            if r_gen_or_none is None:
                continue
            r_gen = r_gen_or_none
            if l_gen.length != r_gen.length:
                raise ValueError(
                    f"Zip requires equal dimension lengths at position {i}; "
                    f"got {l_gen.length} and {r_gen.length}"
                )
            if l_gen.snake != r_gen.snake:
                raise ValueError(
                    f"Zip requires matching snake flags at position {i}; "
                    f"got {l_gen.snake} and {r_gen.snake}"
                )
            left_scan.generators[i] = self._merge_generators(l_gen, r_gen)
        return left_scan

    @staticmethod
    def _merge_generators(
        left: WindowGenerator[AxisT], right: WindowGenerator[AxisT]
    ) -> WindowGenerator[AxisT]:
        """Merge two generators into one with combined axes."""
        if isinstance(left.source, LinearSource) and isinstance(
            right.source, LinearSource
        ):
            return WindowGenerator(
                axes=left.axes + right.axes,
                length=left.length,
                snake=left.snake,
                source=LinearSource(
                    {**left.source.axis_ranges, **right.source.axis_ranges},
                    left.length,
                ),
            )
        l_cap, r_cap = left, right

        def _merged(indexes: np.ndarray) -> dict[Any, np.ndarray]:
            result: dict[Any, np.ndarray] = {}
            result.update(l_cap.setpoints(indexes))
            result.update(r_cap.setpoints(indexes))
            return result

        return WindowGenerator(
            axes=left.axes + right.axes,
            length=left.length,
            snake=left.snake,
            source=FunctionSource(_merged),
        )


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
        """Compile by concatenating left and right into a concat generator."""
        left_scan = self.left.compile()
        right_scan = self.right.compile()
        _reject_continuous_and_monitors(left_scan, "Concat")
        _reject_continuous_and_monitors(right_scan, "Concat")

        children = self._extract_children(left_scan) + self._extract_children(
            right_scan
        )

        # Validate: pure-motion concats must share the same axes.
        if not left_scan.windowed_streams and not right_scan.windowed_streams:
            l_axes = left_scan.generators[0].axes if left_scan.generators else []
            r_axes = right_scan.generators[0].axes if right_scan.generators else []
            if l_axes != r_axes:
                raise ValueError(
                    f"Concat: innermost axes must match; got {l_axes} vs {r_axes}"
                )

        total_length = sum(c.length for c in children)
        concat_gen: WindowGenerator[AxisT] = WindowGenerator(
            axes=[],
            length=total_length,
            source=ConcatSource(children),
        )
        left_scan.generators[:] = [concat_gen]

        # Merge right's windowed streams into left's (sum inner dim lengths).
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

    @staticmethod
    def _extract_children(
        scan: Scan[AxisT, DetectorT, MonitorT],
    ) -> list[WindowGenerator[AxisT]]:
        """Extract leaf WindowGenerators from a Scan for concat children.

        Each side of a Concat must have exactly one generator.  If that
        generator already has children (from a nested Concat), flatten them;
        otherwise use the generator itself.
        """
        gens = scan.generators
        if len(gens) != 1:
            raise ValueError("Concat requires each side to have exactly one generator")
        gen = gens[0]
        if isinstance(gen.source, ConcatSource):
            return list(gen.source.children)
        return [gen]


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

        # Only create a windowed stream when this Acquire has detectors.
        # A monitor/continuous-only Acquire preserves inner streams.
        if self.detectors:
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
# Runtime AnySpec class — defined after all subclasses so pydantic defers
# annotation resolution until model_rebuild below.
# ---------------------------------------------------------------------------

if not TYPE_CHECKING:

    class AnySpec:
        """Runtime subscriptable sentinel for AnySpec.

        ``__class_getitem__`` returns the discriminated union regardless of
        type params so pydantic uses the correct schema.
        ``__get_pydantic_core_schema__`` makes ``TypeAdapter(AnySpec)`` work.
        """

        @classmethod
        def __class_getitem__(cls, params: Any) -> Any:
            return _anyspec_union

        @classmethod
        def __get_pydantic_core_schema__(
            cls, source_type: Any, handler: GetCoreSchemaHandler
        ) -> Any:
            return handler(_anyspec_union)

    _anyspec_cls = AnySpec

    # Final rebuild: resolve AnySpec annotations now that all subclasses and
    # the runtime AnySpec class exist.
    _maybe_rebuild_anyspec_union(Acquire)
