"""Type-inference verification for the scanspec.v2 Sync/Monitors API.

Demonstrates that pyright infers ``DetectorT`` for ``Sync[AxisT, DetectorT,
MonitorT]`` from the ``trigger_group`` argument, and ``MonitorT`` for
``Monitors[AxisT, DetectorT, MonitorT]`` from the ``monitors`` argument,
without any annotation.

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
    """Pyright infers DetectorT=str from trigger_group; MonitorT=Never always.

    AxisT must be provided as an explicit annotation: ``Sync.spec`` is typed
    as the ``MotionSpec`` union (``Union[Linspace[Any], ...]``), so AxisT cannot
    be bound by the synthesised constructor. DetectorT is still inferred
    from ``trigger_group`` alone, unaffected by ADR 0009's pull-out. MonitorT
    is a PEP 696 TypeVar with ``default=Never``: since ``Sync`` has no field
    that mentions MonitorT at all any more (ADR 0009 moved ``monitors`` to
    the separate ``Monitors`` wrapper), a standalone ``Sync(...)`` call
    always infers MonitorT=Never, regardless of whether it's later wrapped.
    """
    spec = Sync(
        motion,
        trigger_group=TriggerGroup(
            detectors=frozenset({"saxs", "waxs"}),
            exposures_per_collection=1,
            collections_per_event=1,
            livetime=0.003,
            deadtime=0.001,
        ),
    )
    assert_type(spec, Sync[str, str, Never])


def test_monitor_t_inferred_from_monitors() -> None:
    """Pyright infers MonitorT=str from ``monitors``, with no annotation.

    The inner spec's own MonitorT (Never for a bare ``Sync``) doesn't
    constrain the wrapper's: ``Monitors.spec`` accepts an inner spec of any
    MonitorT, since an inner spec can never carry monitors itself.
    """
    spec = Monitors(
        Sync(
            motion,
            trigger_group=TriggerGroup(
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
