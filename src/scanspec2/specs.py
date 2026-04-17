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
    MonitorStream,
    Scan,
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
# Motion primitives
# ---------------------------------------------------------------------------


class Linspace(Spec[AxisT, Never, Never]):
    """Evenly-spaced sweep of one axis."""

    axis: AxisT = Field(description="Axis identifier.")
    start: float = Field(description="Midpoint of the first setpoint.")
    stop: float = Field(description="Midpoint of the last setpoint.")
    num: int = Field(description="Number of setpoints (>= 1).", ge=1)


class Static(Spec[AxisT, Never, Never]):
    """Single static position for one axis."""

    axis: AxisT = Field(description="Axis identifier.")
    value: float = Field(description="The fixed position.")
    num: int = Field(
        default=1,
        description="How many times this position appears (>= 1).",
        ge=1,
    )


# ---------------------------------------------------------------------------
# Combinators — accept any Spec (including Acquire)
# ---------------------------------------------------------------------------


class Repeat(Spec[AxisT, DetectorT, MonitorT]):
    """Repeat an inner spec a fixed number of times as an outer dimension."""

    spec: AnySpec[AxisT, DetectorT, MonitorT] = Field(
        description="Inner spec to repeat."
    )
    num: int = Field(description="Number of repetitions (>= 1).", ge=1)


class Snake(Spec[AxisT, DetectorT, MonitorT]):
    """Reverse alternate repeats of an inner spec (snake/boustrophedon)."""

    spec: AnySpec[AxisT, DetectorT, MonitorT] = Field(
        description="Inner spec to snake."
    )


class Product(Spec[AxisT, DetectorT, MonitorT]):
    """Outer x inner product: ``outer`` is slow, ``inner`` is fast."""

    outer: AnySpec[AxisT, DetectorT, MonitorT] = Field(description="Slow (outer) spec.")
    inner: AnySpec[AxisT, DetectorT, MonitorT] = Field(description="Fast (inner) spec.")


class Zip(Spec[AxisT, DetectorT, MonitorT]):
    """Merge two specs into one shared dimension (both must have the same length)."""

    left: AnySpec[AxisT, DetectorT, MonitorT] = Field(description="First spec.")
    right: AnySpec[AxisT, DetectorT, MonitorT] = Field(
        description="Second spec (must have the same length as left)."
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


# ---------------------------------------------------------------------------
# Acquire — outermost node; composable via + operator
# ---------------------------------------------------------------------------


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
        """Compile into a Scan (stub — implemented in Phase 3)."""
        raise NotImplementedError


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
