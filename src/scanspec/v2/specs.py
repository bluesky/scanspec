"""Motion spec nodes and Sync for scanspec 2.0.

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
from math import isclose, prod
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Generic,
    Never,
    Self,
    TypeAlias,
    Union,
    cast,
    dataclass_transform,
)
from typing import (
    TypeVar as StdTypeVar,
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
from typing_extensions import TypeVar

from .core import (
    ConcatSource,
    ContinuousStream,
    DetectorGroup,
    Dimension,
    FunctionSource,
    LinearSource,
    MonitorStream,
    Scan,
    TriggerFollower,
    TriggerGroup,
    TriggerRepeat,
    TriggerSequence,
    WindowedStream,
    WindowGenerator,
    trigger_sequences_duration,
    validate_trigger_sequence,
)

AxisT = StdTypeVar("AxisT")
DetectorT = StdTypeVar("DetectorT")
MonitorT = TypeVar("MonitorT", default=Never)
_BoundedAxisT = StdTypeVar("_BoundedAxisT")


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

    ``arbitrary_types_allowed`` is required because ``MonitorT`` defaults to
    ``Never`` (PEP 696) so pyright can infer it when a bare ``Sync(...)``
    is constructed with no ``monitors=``; pydantic-core cannot generate a
    schema for ``Never`` itself when a subclass is built unparametrized
    (e.g. ``Zip(left=self, right=other)`` inside the combinator methods
    below), and this is the fix pydantic's own error message recommends.
    Concretely-parametrized construction (``Sync[str, str, str](...)``,
    or inference from a real ``monitors=`` value) is unaffected -- full
    validation still applies there.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

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

    @classmethod
    def bounded(
        cls,
        axis: _BoundedAxisT,
        lower: float,
        upper: float,
        num: int = 1,
    ) -> Linspace[_BoundedAxisT]:
        """Construct a Linspace from extreme bounds rather than midpoints.

        ``lower`` and ``upper`` are the outer edges of the first and last
        frames respectively.  The midpoints are half a step inward from each
        edge.
        """
        half_step = (upper - lower) / num / 2
        start = lower + half_step
        if num == 1:
            stop = upper + half_step
        else:
            stop = upper - half_step
        return cast(Linspace[_BoundedAxisT], cls(cast(Any, axis), start, stop, num))


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


class Range(Spec[AxisT, Never, Never]):
    """Evenly-spaced sweep defined by a step size rather than a point count.

    ``start`` and ``stop`` are the midpoints of the first and last frames.
    ``step`` is the spacing between midpoints (must be > 0).  The number of
    points is derived from ``abs(stop - start) / step`` using the same
    rounding rule as the 1.x implementation.
    """

    axis: AxisT = Field(description="Axis identifier.")
    start: float = Field(description="Midpoint of the first frame.")
    stop: float = Field(description="Midpoint of the last frame.")
    step: float = Field(description="Step size between midpoints.", gt=0)

    def _num(self) -> int:
        step = abs(self.step)
        distance = abs(self.stop - self.start)
        num = int(distance // step) + 1
        if isclose(step * num, distance, rel_tol=1e-5, abs_tol=1e-8):
            num = num + 1
        return num

    def compile(self) -> Scan[AxisT, Never, Never]:
        """Compile into a one-dimension Scan with a linear position function."""
        num = self._num()
        # Use the computed last midpoint rather than self.stop: when stop is not
        # an exact multiple of step away from start, LinearSource((start, stop), num)
        # would produce the wrong step spacing.
        sign = np.sign(self.stop - self.start)
        actual_stop = self.start + (num - 1) * sign * self.step
        gen = WindowGenerator(
            axes=[self.axis],
            length=num,
            source=LinearSource({self.axis: (self.start, actual_stop)}, num),
        )
        return Scan(generators=[gen])

    @classmethod
    def bounded(
        cls,
        axis: _BoundedAxisT,
        lower: float,
        upper: float,
        step: float,
    ) -> Range[_BoundedAxisT]:
        """Construct a Range from extreme bounds rather than midpoints.

        ``lower`` and ``upper`` are the outer edges of the bounding box.
        ``step`` is clamped to ``abs(upper - lower)`` so at least one frame
        is always produced.  When ``lower == upper`` the step is used as-is
        and a single point at that position is returned.
        """
        distance = abs(upper - lower)
        if distance == 0.0:
            # Degenerate case: single point, step unchanged
            return cast(
                Range[_BoundedAxisT],
                cls(cast(Any, axis), lower, lower, abs(step)),
            )
        direction = np.sign(upper - lower)
        step = min(distance, abs(step))
        half_step = step / 2 * direction
        start = lower + half_step
        stop = upper - half_step
        if stop == start:
            stop = np.nextafter(start, np.inf * direction)
        return cast(Range[_BoundedAxisT], cls(cast(Any, axis), start, stop, step))


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
    x_diameter: float = Field(description="x width of the spiral.", gt=0)
    x_step: float = Field(description="Radial spacing along x.", gt=0)
    y_axis: AxisT = Field(description="Axis identifier for y.")
    y_centre: float = Field(description="y centre of the spiral.")
    y_diameter: float | None = Field(
        default=None,
        description="y height of the spiral (defaults to abs(x_diameter)).",
        gt=0,
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
# Combinators — accept any Spec (including Sync)
# ---------------------------------------------------------------------------


def _reject_continuous_and_monitors(scan: Scan[Any, Any, Any], combinator: str) -> None:
    """Raise if *scan* has continuous_streams or monitors.

    Continuous and monitor streams run in parallel to the entire scan and
    must be attached via the outermost ``ContinuousStreams``/``Monitors``
    wrapper, not nested inside any other spec.
    """
    if scan.continuous_streams:
        raise ValueError(
            f"{combinator} does not accept specs with continuous_streams; "
            f"attach them via the outermost ContinuousStreams wrapper instead"
        )
    if scan.monitors:
        raise ValueError(
            f"{combinator} does not accept specs with monitors; "
            f"attach them via the outermost Monitors wrapper instead"
        )


def _reject_existing_monitors(scan: Scan[Any, Any, Any]) -> None:
    """Raise if *scan* already has monitors attached.

    Guards against double-wrapping (``Monitors(Monitors(...))``) or a
    ``Monitors`` nested inside a combinator instead of outside it -- the
    latter is also caught by ``_reject_continuous_and_monitors`` at the
    combinator itself, this is belt-and-braces for the wrapper's own
    boundary. Deliberately does *not* check ``continuous_streams`` --
    ``ContinuousStreams`` may legitimately have already run on this same
    scan (the two wrappers commute and compose, e.g.
    ``ContinuousStreams(Monitors(...), ...)``).
    """
    if scan.monitors:
        raise ValueError(
            "Monitors cannot wrap a spec that already has monitors "
            "attached (double-wrapping, or a nested Monitors deeper in "
            "the tree)"
        )


def _reject_existing_continuous_streams(scan: Scan[Any, Any, Any]) -> None:
    """Raise if *scan* already has continuous_streams attached.

    Mirrors ``_reject_existing_monitors`` for ``ContinuousStreams``;
    deliberately does not check ``monitors``.
    """
    if scan.continuous_streams:
        raise ValueError(
            "ContinuousStreams cannot wrap a spec that already has "
            "continuous_streams attached (double-wrapping, or a nested "
            "ContinuousStreams deeper in the tree)"
        )


def _group_keys(groups: list[DetectorGroup[Any]]) -> list[tuple[object, ...]]:
    """Comparable form of detector groups; detector order is not significant."""
    return [
        (
            frozenset(g.detectors),
            g.exposures_per_collection,
            g.collections_per_event,
            g.livetime,
            g.deadtime,
        )
        for g in groups
    ]


def _leaves(stacks: list[list[WindowGenerator[AxisT]]]) -> list[WindowGenerator[AxisT]]:
    """Every generator in *stacks*, looking through nested concats.

    A concat generator's own ``axes`` and trigger sequences are empty; its
    children's generators carry them.
    """
    return [
        leaf
        for stack in stacks
        for gen in stack
        for leaf in (
            _leaves(gen.source.children)
            if isinstance(gen.source, ConcatSource)
            else [gen]
        )
    ]


def _reject_multi_generator_concat(scan: Scan[Any, Any, Any], op: str) -> None:
    """Raise if *scan* has a concat with a multi-generator child anywhere.

    Such a child runs as nested loops, so the concat has no single flat
    dimension for *op* to treat it as.
    """

    def has_one(gens: list[WindowGenerator[Any]]) -> bool:
        return any(
            isinstance(gen.source, ConcatSource)
            and any(len(stack) != 1 or has_one(stack) for stack in gen.source.children)
            for gen in gens
        )

    if has_one(scan.generators):
        raise ValueError(
            f"{op} cannot take a Concat with a multi-generator side; "
            f"put the {op} inside the Concat's sides instead"
        )


def _reject_snaked_concat_side(scan: Scan[Any, Any, Any]) -> None:
    """Raise if a concat in *scan* has a snaked side, at any depth.

    A stream ``Dimension`` has one snake flag for the whole concat, so it
    can't describe a side that reverses on its own on odd passes.
    """

    def has_one(gens: list[WindowGenerator[Any]]) -> bool:
        return any(
            isinstance(gen.source, ConcatSource)
            and any(
                child.snake or has_one([child])
                for stack in gen.source.children
                for child in stack
            )
            for gen in gens
        )

    if has_one(scan.generators):
        raise ValueError(
            "Sync cannot take a Concat with a snaked side: its stream can't "
            "describe one side reversing on its own; give that side its own "
            "stream, e.g. Sync(a).concat(Sync(~b, stream_name='b'))"
        )


def _moved_axes(leaves: list[WindowGenerator[AxisT]]) -> list[AxisT]:
    """The axes *leaves* move, in first-seen order."""
    return list(dict.fromkeys(axis for leaf in leaves for axis in leaf.axes))


def _stream_pieces(
    leaves: list[WindowGenerator[Any]], root: frozenset[Any]
) -> list[WindowGenerator[Any]]:
    """The leaf generators belonging to the stream whose root fires *root*.

    A detector belongs to only one stream, so its root detectors identify it.
    """
    return [
        leaf
        for leaf in leaves
        if any(seq.root.detectors == root for seq in leaf.trigger_sequences)
    ]


def _follower_repeats(
    pieces: list[WindowGenerator[Any]],
) -> set[tuple[tuple[frozenset[Any], int], ...]]:
    """Follower ``repeats`` of a stream's *pieces*.

    ``DetectorGroup`` doesn't carry them, but each piece's compiled trigger
    sequences do, and they don't depend on the motion, so pieces with
    identical trigger groups have identical values here.
    """
    return {
        tuple((c.detectors, c.repeats) for c in seq.children)
        for piece in pieces
        for seq in piece.trigger_sequences
    }


def _check_unique_detectors(scan: Scan[Any, Any, Any]) -> None:
    """Raise if a detector belongs to more than one stream.

    Checks windowed streams, continuous streams and monitors on *scan*.
    Called by ``Concat`` (the only place windowed streams merge) and by
    ``Monitors``/``ContinuousStreams`` after attaching their own field; each
    re-checks the whole scan, so the outermost call sees everything.
    """
    seen: set[object] = set()
    duplicates: set[object] = set()

    def _mark(d: object) -> None:
        if d in seen:
            duplicates.add(d)
        seen.add(d)

    for ws in scan.windowed_streams:
        for dg in ws.detector_groups:
            for d in dg.detectors:
                _mark(d)
    for cs in scan.continuous_streams:
        for dg in cs.detector_groups:
            for d in dg.detectors:
                _mark(d)
    for ms in scan.monitors:
        _mark(ms.detector)

    if duplicates:
        raise ValueError(
            f"A detector can belong to only one stream; "
            f"duplicates: {sorted(str(d) for d in duplicates)}"
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
        _reject_continuous_and_monitors(scan, "Repeat")
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

    @model_validator(mode="after")
    def _no_sync_inside(self) -> Self:
        # A Sync's stream dimensions are fixed when it compiles, so a
        # reversal added outside it would go unrecorded.
        _reject_sync_in("Snake", self.spec)
        return self

    def compile(self) -> Scan[AxisT, DetectorT, MonitorT]:
        """Compile by setting snake=True on the innermost generator."""
        scan = self.spec.compile()
        if len(scan.generators) != 1:
            raise ValueError(
                f"Snake requires exactly one generator, got {len(scan.generators)}"
            )
        _reject_multi_generator_concat(scan, "Snake")
        _reject_continuous_and_monitors(scan, "Snake")
        scan.generators[0].snake = True
        return scan


def _contains_sync(spec: Spec[Any, Any, Any]) -> bool:
    """True if *spec* is a ``Sync`` or has one anywhere beneath it."""
    if isinstance(spec, Sync):
        return True
    for name in type(spec).model_fields:
        child: object = getattr(spec, name)
        if isinstance(child, Spec) and _contains_sync(cast(Spec[Any, Any, Any], child)):
            return True
    return False


def _reject_sync_in(owner: str, *children: Spec[Any, Any, Any]) -> None:
    if any(_contains_sync(c) for c in children):
        raise ValueError(
            f"{owner} cannot contain a Sync: wrap the whole motion in a single "
            f"Sync instead, e.g. Sync({owner}(...))"
        )


class Product(Spec[AxisT, DetectorT, MonitorT]):
    """Outer x inner product: ``outer`` is slow, ``inner`` is fast."""

    outer: AnySpec[AxisT, DetectorT, MonitorT] = Field(description="Slow (outer) spec.")
    inner: AnySpec[AxisT, DetectorT, MonitorT] = Field(description="Fast (inner) spec.")

    @model_validator(mode="after")
    def _no_sync_inside(self) -> Self:
        _reject_sync_in("Product", self.outer, self.inner)
        return self

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
    - Left has more generators than right: left-pad right's generator list
      with placeholders so positions align by innermost generator; left's
      extra outer generators are left untouched.
    - Right has a single generator of length 1 (e.g. ``Static``): expand
      it to match left's innermost generator length.

    A ``Concat`` on either side at a merged position stays a concat: each
    of its sides is zipped with the slice of the other spec covering its
    points, so ``Zip(a.concat(b), y)`` runs as
    ``Zip(a, y[:len(a)]).concat(Zip(b, y[len(a):]))``. Two concats at the
    same position, or a snaked concat side, raise. So do two sides that set
    the same axis, which would give it two values at each point.
    """

    left: AnySpec[AxisT, DetectorT, MonitorT] = Field(description="First spec.")
    right: AnySpec[AxisT, DetectorT, MonitorT] = Field(description="Second spec.")

    @model_validator(mode="after")
    def _no_sync_inside(self) -> Self:
        _reject_sync_in("Zip", self.left, self.right)
        return self

    def compile(self) -> Scan[AxisT, DetectorT, MonitorT]:
        """Compile by merging generators of left and right."""
        left_scan = self.left.compile()
        right_scan = self.right.compile()
        _reject_continuous_and_monitors(left_scan, "Zip")
        _reject_continuous_and_monitors(right_scan, "Zip")
        if not left_scan.generators or not right_scan.generators:
            raise ValueError("Zip requires both specs to have at least one dimension")
        shared = set(_moved_axes(_leaves([left_scan.generators]))) & set(
            _moved_axes(_leaves([right_scan.generators]))
        )
        if shared:
            raise ValueError(
                f"Zip cannot give an axis two values at each point; both "
                f"sides set {sorted(map(str, shared))}"
            )
        _reject_multi_generator_concat(left_scan, "Zip")
        _reject_multi_generator_concat(right_scan, "Zip")
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

    @classmethod
    def _merge_generators(
        cls, left: WindowGenerator[AxisT], right: WindowGenerator[AxisT]
    ) -> WindowGenerator[AxisT]:
        """Merge two equal-length generators into one with combined axes.

        A concat on either side stays a concat: each of its children is
        zipped with the slice of the other side that covers the child's
        index range, so ``Zip(a.concat(b), y)`` runs as
        ``Zip(a, y[:len(a)]).concat(Zip(b, y[len(a):]))``.
        """
        l_concat = isinstance(left.source, ConcatSource)
        r_concat = isinstance(right.source, ConcatSource)
        if l_concat and r_concat:
            raise ValueError(
                "Zip cannot merge two Concats at the same position; "
                "zip inside the Concats' sides instead"
            )
        if l_concat or r_concat:
            concat, other = (left, right) if l_concat else (right, left)
            assert isinstance(concat.source, ConcatSource)
            children: list[list[WindowGenerator[AxisT]]] = []
            offset = 0
            for (child,) in concat.source.children:
                if child.snake:
                    # The other spec is one generator, so it can't reverse
                    # with just this side on odd passes.
                    raise ValueError(
                        "Zip cannot merge a snaked Concat side with another "
                        "spec; zip inside the Concat's sides instead"
                    )
                piece = _slice_generator(other, offset, child.length)
                pair = (child, piece) if l_concat else (piece, child)
                children.append([cls._merge_generators(*pair)])
                offset += child.length
            return WindowGenerator(
                axes=[],
                length=concat.length,
                snake=concat.snake,
                source=ConcatSource(children),
            )
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


def _slice_generator(
    gen: WindowGenerator[AxisT], offset: int, length: int
) -> WindowGenerator[AxisT]:
    """The points ``offset`` to ``offset + length`` of *gen*, as a generator.

    A linear source stays linear, with the same step; anything else is
    evaluated through *gen* at shifted indexes.
    """
    if isinstance(gen.source, LinearSource):
        first = gen.setpoints(np.array([offset + 0.5]))
        second = gen.setpoints(np.array([offset + 1.5]))
        ranges = {
            ax: (
                float(first[ax][0]),
                float(
                    first[ax][0] + max(length - 1, 1) * (second[ax][0] - first[ax][0])
                ),
            )
            for ax in gen.source.axis_ranges
        }
        source: LinearSource[AxisT] | FunctionSource[AxisT] = LinearSource(
            ranges, length
        )
    else:

        def _shifted(indexes: np.ndarray) -> dict[AxisT, np.ndarray]:
            return gen.setpoints(indexes + offset)

        source = FunctionSource(_shifted)
    return WindowGenerator(axes=gen.axes, length=length, source=source)


class Concat(Spec[AxisT, DetectorT, MonitorT]):
    """Concatenate two specs sequentially.

    Both *left* and *right* may carry detector configuration — useful for
    combining two ``Sync`` nodes that differ in detector setup.
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

        left_stacks = self._extract_children(left_scan)
        right_stacks = self._extract_children(right_scan)
        left_leaves = _leaves(left_stacks)
        right_leaves = _leaves(right_stacks)

        # Validate: pure-motion concats must move the same axes, over every
        # generator of each side.
        if not left_scan.windowed_streams and not right_scan.windowed_streams:
            l_axes = _moved_axes(left_leaves)
            r_axes = _moved_axes(right_leaves)
            if set(l_axes) != set(r_axes):
                raise ValueError(
                    f"Concat: both sides must move the same axes; "
                    f"got {l_axes} vs {r_axes}"
                )

        children = left_stacks + right_stacks
        total_length = sum(prod(g.length for g in stack) for stack in children)
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
                root = frozenset(existing.detector_groups[0].detectors)
                left_pieces = _stream_pieces(left_leaves, root)
                right_pieces = _stream_pieces(right_leaves, root)
                if _group_keys(existing.detector_groups) != _group_keys(
                    stream.detector_groups
                ) or _follower_repeats(left_pieces) != _follower_repeats(right_pieces):
                    raise ValueError(
                        f"Concat: Syncs sharing stream name {stream.name!r} "
                        f"must have identical trigger groups"
                    )
                if len(existing.dimensions) > 1 or len(stream.dimensions) > 1:
                    raise ValueError(
                        f"Concat: Syncs sharing stream name {stream.name!r} "
                        f"can only merge single-dimension streams"
                    )
                if existing.dimensions and stream.dimensions:
                    inner, other = existing.dimensions[-1], stream.dimensions[-1]
                    if inner.snake or other.snake:
                        raise ValueError(
                            f"Concat: Syncs sharing stream name {stream.name!r} "
                            f"cannot merge a snaked dimension: each piece "
                            f"reverses on its own, which one merged dimension "
                            f"can't describe; give them different stream names"
                        )
                    if set(inner.axes) != set(other.axes):
                        raise ValueError(
                            f"Concat: Syncs sharing stream name {stream.name!r} "
                            f"must move the same axes; got {inner.axes} and "
                            f"{other.axes}"
                        )
                    existing.dimensions[-1] = Dimension(
                        axes=inner.axes,
                        length=inner.length + other.length,
                        snake=inner.snake,
                        position_fn=ConcatSource(
                            [[p] for p in left_pieces + right_pieces]
                        ).setpoints,
                    )
            else:
                left_scan.windowed_streams.append(stream)
        _check_unique_detectors(left_scan)
        return left_scan

    @staticmethod
    def _extract_children(
        scan: Scan[AxisT, DetectorT, MonitorT],
    ) -> list[list[WindowGenerator[AxisT]]]:
        """The generator stacks *scan* contributes as concat children.

        A side is one stack, its whole generator list. A side that is itself
        a lone, unsnaked concat generator (from a nested Concat) splices that
        concat's stacks in instead; a snaked one stays whole, so its reversal
        on odd passes is kept.
        """
        gens = scan.generators
        if not gens:
            raise ValueError("Concat requires each side to have a generator")
        if (
            len(gens) == 1
            and isinstance(gens[0].source, ConcatSource)
            and not gens[0].snake
        ):
            return list(gens[0].source.children)
        return [list(gens)]


def _resolved_timing(
    detectors: frozenset[DetectorT], livetime: float | None, deadtime: float | None
) -> tuple[float, float]:
    """Return (livetime, deadtime), raising if either is unresolved."""
    if livetime is None or deadtime is None:
        raise ValueError(
            f"livetime and deadtime must be resolved before compile() for "
            f"detectors {sorted(str(d) for d in detectors)}; got "
            f"livetime={livetime}, deadtime={deadtime}"
        )
    return livetime, deadtime


def _group_to_repeat(
    group: TriggerGroup[DetectorT], *, inner_length: int, fly: bool
) -> TriggerRepeat[DetectorT]:
    """Resolve a TriggerGroup's own detectors into the root TriggerRepeat.

    repeats is tied to the scan's own geometry -- exposures_per_event,
    scaled by inner_length for a fly scan. Never caller-suppliable: this is
    the literal motion<->trigger binding Sync exists to perform, not an
    authoring input (ADR 0008 Decision 1).
    """
    lt, dt = _resolved_timing(group.detectors, group.livetime, group.deadtime)
    repeats = group.exposures_per_event * (inner_length if fly else 1)
    return TriggerRepeat(
        detectors=group.detectors, repeats=repeats, livetime=lt, deadtime=dt
    )


def _follower_to_repeat(
    follower: TriggerFollower[DetectorT],
) -> TriggerRepeat[DetectorT]:
    """Resolve a caller-authored TriggerFollower into a compiled TriggerRepeat.

    repeats/livetime/deadtime must all already be concrete -- a follower's
    rate relative to the group is genuinely external information (detector
    hardware timing), so nothing here is derived from the spec tree. No
    attempt is made to compute a missing repeats from livetime/deadtime
    (ADR 0008 Decision 3) -- a caller or ophyd-async must supply all three
    explicitly. The integer-ratio/duration checks against the group's own
    repeat still run afterwards, via validate_trigger_sequence.
    """
    if follower.repeats is None:
        raise ValueError(
            f"repeats must be set on TriggerFollower before compile() for "
            f"detectors {sorted(str(d) for d in follower.detectors)}"
        )
    lt, dt = _resolved_timing(follower.detectors, follower.livetime, follower.deadtime)
    return TriggerRepeat(
        detectors=follower.detectors, repeats=follower.repeats, livetime=lt, deadtime=dt
    )


def _trigger_group_to_detector_group(
    group: TriggerGroup[DetectorT] | TriggerFollower[DetectorT],
) -> DetectorGroup[DetectorT]:
    """Carry a TriggerGroup/TriggerFollower's shape/timing through unchanged.

    No repeats involved -- both shape factors pass through unchanged. Called
    only after the corresponding TriggerRepeat derivation has already run
    (see Sync.compile()), which guarantees livetime/deadtime are resolved
    by this point.
    """
    return DetectorGroup(
        exposures_per_collection=group.exposures_per_collection,
        collections_per_event=group.collections_per_event,
        livetime=group.livetime,
        deadtime=group.deadtime,
        detectors=list(group.detectors),
    )


def _group_to_sequence(
    group: TriggerGroup[DetectorT],
    *,
    inner_length: int,
    fly: bool,
) -> TriggerSequence[DetectorT]:
    """Derive a compiled TriggerSequence from a caller-authored TriggerGroup."""
    root = _group_to_repeat(group, inner_length=inner_length, fly=fly)
    children = [_follower_to_repeat(follower) for follower in group.followers]
    return TriggerSequence(root=root, children=children)


class Sync(Spec[AxisT, DetectorT, MonitorT]):
    """Binds detector triggering to a motion spec.

    ``spec`` must be pure motion: combine several ``Sync``s with ``Concat``,
    never by nesting them or placing them under ``Product``/``Zip``.
    """

    spec: AnySpec[AxisT, Any, Any] = Field(description="Motion spec.")
    fly: bool = Field(
        default=False,
        description="True for flyscan (innermost dimension sweeps continuously).",
    )
    stream_name: str = Field(
        default="primary",
        description="Bluesky stream name.",
    )
    trigger_group: TriggerGroup[DetectorT] | None = Field(
        default=None,
        description=(
            "Caller-authored trigger hierarchy for the windowed stream. "
            "compile() derives both the compiled TriggerSequence and the "
            "list[DetectorGroup] the windowed stream needs from this."
        ),
    )
    duration: float | None = Field(
        default=None,
        description=(
            "Per-point duration in seconds, for scans without a "
            "trigger_group: required for fly scans, defaults to 0 for step "
            "scans. Not allowed with a trigger_group, whose timing "
            "determines the duration."
        ),
    )

    @model_validator(mode="after")
    def _validate_structure(self) -> Self:
        if self.duration is not None and self.trigger_group is not None:
            raise ValueError(
                "duration cannot be combined with trigger_group: the duration "
                "is derived from the trigger timing"
            )
        if self.fly and self.duration is None and self.trigger_group is None:
            raise ValueError(
                "a fly Sync needs a duration or a trigger_group: without one "
                "its velocities and window durations have no time base"
            )
        if _contains_sync(self.spec):
            raise ValueError(
                "Sync cannot contain another Sync: its spec must be pure "
                "motion. Combine Syncs with Concat instead"
            )
        return self

    def compile(self) -> Scan[AxisT, DetectorT, MonitorT]:
        """Compile into a Scan with detector groups and generators.

        The innermost generator's windows carry this Sync's fly flag,
        trigger sequences and duration. A concat's windows come from its
        leaves, so each leaf gets them, derived from its own length:
        ``Sync(a.concat(b))`` runs as ``Sync(a).concat(Sync(b))``.
        """
        scan = self.spec.compile()
        _reject_multi_generator_concat(scan, "Sync")
        _reject_snaked_concat_side(scan)
        _reject_continuous_and_monitors(scan, "Sync")
        targets = _leaves([scan.generators[-1:]])
        for gen in targets:
            trigger_sequences = self._trigger_sequences(gen.length)
            gen.fly = self.fly
            gen.trigger_sequences = trigger_sequences
            gen.duration = self._compute_duration(trigger_sequences, gen.length)
        if not targets:
            # Nothing to attach them to, but the timing must still be valid.
            self._trigger_sequences(1)

        # Only create a windowed stream when this Sync has detectors.
        if self.trigger_group is not None:
            group = self.trigger_group
            dims = [
                Dimension(
                    axes=_moved_axes(_leaves([[g]])),
                    length=g.length,
                    snake=g.snake,
                    position_fn=g.setpoints,
                )
                for g in scan.generators
            ]
            stream = WindowedStream(
                name=self.stream_name,
                dimensions=dims,
                detector_groups=[
                    _trigger_group_to_detector_group(g)
                    for g in [group, *group.followers]
                ],
            )
            scan.windowed_streams = [stream]
        return scan

    def _trigger_sequences(self, inner_length: int) -> list[TriggerSequence[DetectorT]]:
        """Derive and validate the trigger sequences for one generator.

        *inner_length* is that generator's length, which a fly scan's root
        repeats scale by.
        """
        if self.trigger_group is None:
            return []
        seq = _group_to_sequence(
            self.trigger_group, inner_length=inner_length, fly=self.fly
        )
        validate_trigger_sequence(seq)
        return [seq]

    def _compute_duration(
        self, trigger_sequences: list[TriggerSequence[DetectorT]], inner_length: int
    ) -> float | None:
        """Derive per-point duration from trigger timing.

        For step scans (1 window per setpoint) this equals total window
        duration. For fly scans the total window duration is
        ``per_point * inner_length``; ``Scan.__iter__`` multiplies
        automatically.
        """
        if not trigger_sequences:
            return self.duration
        total_dur = trigger_sequences_duration(trigger_sequences)
        return total_dur / (inner_length if self.fly else 1)


class Monitors(Spec[AxisT, DetectorT, MonitorT]):
    """Attach free-running PV monitors to the whole scan.

    Must sit outside the entire spec tree -- ``_reject_continuous_and_monitors``
    (raised by every spec that wraps or merges specs, other than these two
    wrappers) and ``_reject_existing_monitors`` (raised by this class on its
    own inner scan) enforce that a nested ``Monitors``/``ContinuousStreams``
    cannot sit inside a larger tree.
    """

    spec: AnySpec[AxisT, DetectorT, Any] = Field(description="Inner spec.")
    monitors: Sequence[MonitorStream[MonitorT]] = Field(
        default_factory=tuple,
        description="Free-running PV monitors, scan-wide.",
    )

    def compile(self) -> Scan[AxisT, DetectorT, MonitorT]:
        """Compile the inner spec, then attach monitors to the resulting Scan."""
        scan = self.spec.compile()
        _reject_existing_monitors(scan)
        scan.monitors = list(self.monitors)
        _check_unique_detectors(scan)
        return scan


class ContinuousStreams(Spec[AxisT, DetectorT, MonitorT]):
    """Attach continuously-acquired detector streams to the whole scan.

    Must sit outside the entire spec tree -- see ``Monitors`` docstring.
    """

    spec: AnySpec[AxisT, Any, MonitorT] = Field(description="Inner spec.")
    continuous_streams: Sequence[ContinuousStream[DetectorT]] = Field(
        default_factory=tuple,
        description="Continuously-acquired detector groups (no scan dimensions).",
    )

    def compile(self) -> Scan[AxisT, DetectorT, MonitorT]:
        """Compile the inner spec, then attach continuous streams to the Scan."""
        scan = self.spec.compile()
        _reject_existing_continuous_streams(scan)
        scan.continuous_streams = list(self.continuous_streams)
        _check_unique_detectors(scan)
        return scan


# ---------------------------------------------------------------------------
# Masked-grid helpers
# ---------------------------------------------------------------------------


def _eval_full_grid(scan: Scan[Any, Any, Any]) -> dict[Any, np.ndarray]:
    """Return all midpoints of a compiled scan as flat arrays.

    Expands a multi-generator scan into a full Cartesian product, giving one
    array per axis of length ``product(g.length for g in scan.generators)``.
    Snake flags are ignored — the caller cares only about *which* points
    exist, not their traversal order.
    """
    per_axis: dict[Any, np.ndarray] = {}
    outer_len = 1
    for gen in scan.generators:
        inner_len = gen.length
        midpt_idx = np.arange(inner_len, dtype=float) + 0.5
        pts = gen.setpoints(midpt_idx)
        # Each already-collected outer axis repeats inner_len times.
        for ax in list(per_axis.keys()):
            per_axis[ax] = np.repeat(per_axis[ax], inner_len)
        # Each new inner axis tiles outer_len times.
        for ax, arr in pts.items():
            per_axis[ax] = np.tile(arr, outer_len)
        outer_len *= inner_len
    return per_axis


class Ellipse(Spec[AxisT, Never, Never]):
    """Grid of points masked to an elliptical footprint.

    Builds a 2-D rectangular grid from ``Range`` objects spanning the
    bounding box of the ellipse, then keeps only those midpoints that satisfy
    the ellipse equation.  ``vertical`` controls the fast/slow axis selection
    (it does not affect *which* points are retained).
    """

    x_axis: AxisT = Field(description="Axis identifier for x.")
    x_centre: float = Field(description="x centre of the ellipse.")
    x_diameter: float = Field(description="x diameter of the ellipse.", gt=0)
    x_step: float = Field(description="Grid spacing along x.", gt=0)
    y_axis: AxisT = Field(description="Axis identifier for y.")
    y_centre: float = Field(description="y centre of the ellipse.")
    y_diameter: float | None = Field(
        default=None,
        description="y diameter (defaults to abs(x_diameter)).",
        gt=0,
    )
    y_step: float | None = Field(
        default=None,
        description="Grid spacing along y (defaults to x_step).",
        gt=0,
    )
    vertical: bool = Field(default=False, description="If True, y is the fast axis.")

    def _eff_y_diameter(self) -> float:
        return self.y_diameter if self.y_diameter is not None else abs(self.x_diameter)

    def _eff_y_step(self) -> float:
        return self.y_step if self.y_step is not None else self.x_step

    def compile(self) -> Scan[AxisT, Never, Never]:
        """Compile into a flat single-generator Scan of masked midpoints."""
        x_radius = abs(self.x_diameter) / 2
        eff_y_diam = self._eff_y_diameter()
        y_radius = abs(eff_y_diam) / 2
        eff_y_step = self._eff_y_step()

        x_range: Range[AxisT] = Range(
            self.x_axis,
            self.x_centre - x_radius,
            self.x_centre + x_radius,
            self.x_step,
        )
        y_range: Range[AxisT] = Range(
            self.y_axis,
            self.y_centre - y_radius,
            self.y_centre + y_radius,
            eff_y_step,
        )

        if self.vertical:
            grid: Spec[AxisT, Never, Never] = x_range * y_range
        else:
            grid = y_range * x_range

        all_pts = _eval_full_grid(grid.compile())

        x = all_pts[self.x_axis] - self.x_centre
        y = all_pts[self.y_axis] - self.y_centre
        mask = (2 * x / self.x_diameter) ** 2 + (2 * y / eff_y_diam) ** 2 <= 1

        x_masked = all_pts[self.x_axis][mask]
        y_masked = all_pts[self.y_axis][mask]
        num_masked = int(mask.sum())

        x_ax: AxisT = self.x_axis
        y_ax: AxisT = self.y_axis

        def _pos_fn(indexes: np.ndarray) -> dict[AxisT, np.ndarray]:
            idx = (indexes - 0.5).astype(int)
            return {y_ax: y_masked[idx], x_ax: x_masked[idx]}

        gen = WindowGenerator(
            axes=[self.y_axis, self.x_axis],
            length=num_masked,
            source=FunctionSource(_pos_fn),
        )
        return Scan(generators=[gen])


class Polygon(Spec[AxisT, Never, Never]):
    """Grid of points masked to a polygonal footprint.

    Uses an even-odd ray-casting rule to determine which grid midpoints are
    inside the polygon defined by ``vertices``.
    """

    x_axis: AxisT = Field(description="Axis identifier for x.")
    y_axis: AxisT = Field(description="Axis identifier for y.")
    vertices: list[tuple[float, float]] = Field(
        description="Ordered (x, y) vertices of the polygon."
    )
    x_step: float = Field(description="Grid spacing along x.", gt=0)
    y_step: float | None = Field(
        default=None,
        description="Grid spacing along y (defaults to x_step).",
        gt=0,
    )
    vertical: bool = Field(default=False, description="If True, y is the fast axis.")

    def _eff_y_step(self) -> float:
        return self.y_step if self.y_step is not None else self.x_step

    def _poly_mask(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Even-odd ray-casting mask (matches 1.x algorithm exactly)."""
        v1x, v1y = self.vertices[-1]
        mask = np.full(len(x), False, dtype=np.bool_)
        for v2x, v2y in self.vertices:
            if v2y != v1y:
                vmask = np.full(len(x), False, dtype=np.bool_)
                vmask |= (y < v2y) & (y >= v1y)
                vmask |= (y < v1y) & (y >= v2y)
                t = (y - v1y) / (v2y - v1y)
                vmask &= x < v1x + t * (v2x - v1x)
                mask ^= vmask
            v1x, v1y = v2x, v2y
        return mask

    def compile(self) -> Scan[AxisT, Never, Never]:
        """Compile into a flat single-generator Scan of masked midpoints."""
        x_start = min(v[0] for v in self.vertices)
        x_stop = max(v[0] for v in self.vertices)
        y_start = min(v[1] for v in self.vertices)
        y_stop = max(v[1] for v in self.vertices)
        eff_y_step = self._eff_y_step()

        x_range: Range[AxisT] = Range(self.x_axis, x_start, x_stop, self.x_step)
        y_range: Range[AxisT] = Range(self.y_axis, y_start, y_stop, eff_y_step)

        if self.vertical:
            grid: Spec[AxisT, Never, Never] = x_range * y_range
        else:
            grid = y_range * x_range

        all_pts = _eval_full_grid(grid.compile())

        x_arr = all_pts[self.x_axis]
        y_arr = all_pts[self.y_axis]
        mask = self._poly_mask(x_arr, y_arr)

        x_masked = x_arr[mask]
        y_masked = y_arr[mask]
        num_masked = int(mask.sum())

        x_ax: AxisT = self.x_axis
        y_ax: AxisT = self.y_axis

        def _pos_fn(indexes: np.ndarray) -> dict[AxisT, np.ndarray]:
            idx = (indexes - 0.5).astype(int)
            return {y_ax: y_masked[idx], x_ax: x_masked[idx]}

        gen = WindowGenerator(
            axes=[self.y_axis, self.x_axis],
            length=num_masked,
            source=FunctionSource(_pos_fn),
        )
        return Scan(generators=[gen])


# Line is a trivial alias for Linspace (both names are supported).
Line = Linspace


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
    _maybe_rebuild_anyspec_union(Sync)
