"""Type-inference verification for the scanspec.v2 Sync/Monitors API.

Demonstrates that pyright infers ``DetectorT`` for ``Sync[AxisT, DetectorT,
MonitorT]`` from the ``trigger_plan`` argument without any annotation, but
that ``MonitorT`` -- now that ``monitors`` lives on the separate
``Monitors[AxisT, DetectorT, MonitorT]`` wrapper (ADR 0009), not on
``Sync`` -- can no longer be inferred purely from usage, and needs an
explicit annotation on the assignment target. See
``test_monitor_t_requires_explicit_annotation`` for why.

These tests are checked statically by pyright (``tox -e type-checking``) and
executed by pytest (``assert_type`` is a no-op at runtime in Python ≥ 3.11).
"""

from __future__ import annotations

from typing import Never, assert_type

from scanspec.v2.core import MonitorStream, TriggerGroup
from scanspec.v2.specs import Linspace, Monitors, Sync

# ---------------------------------------------------------------------------
# Inference assertions
# ---------------------------------------------------------------------------

motion = Linspace("x", 0.0, 1.0, 100)


def test_detector_t_inferred_monitor_t_defaults_to_never() -> None:
    """Pyright infers DetectorT=str from trigger_plan; MonitorT=Never always.

    AxisT must be provided as an explicit annotation: ``Sync.spec`` is typed
    as the ``MotionSpec`` union (``Union[Linspace[Any], ...]``), so AxisT cannot
    be bound by the synthesised constructor. DetectorT is still inferred
    from ``trigger_plan`` alone, unaffected by ADR 0009's pull-out. MonitorT
    is a PEP 696 TypeVar with ``default=Never``: since ``Sync`` has no field
    that mentions MonitorT at all any more (ADR 0009 moved ``monitors`` to
    the separate ``Monitors`` wrapper), a standalone ``Sync(...)`` call
    always infers MonitorT=Never, regardless of whether it's later wrapped.
    """
    spec = Sync(
        motion,
        trigger_plan=TriggerGroup(
            detectors=frozenset({"saxs", "waxs"}),
            exposures_per_collection=1,
            collections_per_event=1,
            livetime=0.003,
            deadtime=0.001,
        ),
    )
    assert_type(spec, Sync[str, str, Never])


def test_monitor_t_requires_explicit_annotation() -> None:
    """Unlike DetectorT, MonitorT can no longer be inferred purely from
    usage once Monitors wraps Sync -- an explicit annotation is required.

    Sync's own MonitorT is fixed to Never the moment a ``Sync(...)`` call
    returns, since nothing on ``Sync`` mentions MonitorT any more (ADR
    0009 moved ``monitors`` to the separate ``Monitors`` wrapper) -- even
    when the ``Sync(...)`` call sits directly inside a ``Monitors(...)``
    call in the same expression, since pyright evaluates nested calls
    argument-first, not bidirectionally, and locks in Never before
    ``Monitors`` ever runs. An explicit annotation on the assignment
    target is the only way to widen it back to the actual monitor type.
    This is a real ergonomics cost of the ADR 0009 pull-out, not a bug:
    before it, MonitorT flowed from ``monitors=`` directly on ``Sync``'s
    own constructor call, in the same statement that fixed the type.
    """
    spec: Monitors[str, str, str] = Monitors(
        Sync(
            motion,
            trigger_plan=TriggerGroup(
                detectors=frozenset({"saxs", "waxs"}),
                exposures_per_collection=1,
                collections_per_event=1,
                livetime=0.003,
                deadtime=0.001,
            ),
        ),
        monitors=[MonitorStream("dcm_temp", "dcm_temperature")],
    )
    assert_type(spec, Monitors[str, str, str])
