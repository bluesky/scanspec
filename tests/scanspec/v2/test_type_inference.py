"""Type-inference verification for the scanspec.v2 Acquire API.

Demonstrates that pyright infers ``DetectorT`` and ``MonitorT`` for
``Acquire[AxisT, DetectorT, MonitorT]`` from the constructor arguments, so
explicit annotation is usually not needed at the call site.

These tests are checked statically by pyright (``tox -e type-checking``) and
executed by pytest (``assert_type`` is a no-op at runtime in Python ≥ 3.11).
"""

from __future__ import annotations

from typing import Never, assert_type

from scanspec.v2.core import DetectorGroup, MonitorStream
from scanspec.v2.specs import Acquire, Linspace

# ---------------------------------------------------------------------------
# Inference assertions
# ---------------------------------------------------------------------------

motion = Linspace("x", 0.0, 1.0, 100)


def test_detector_t_and_monitor_t_inferred() -> None:
    """Pyright infers DetectorT=str and MonitorT=str from the argument types.

    AxisT must be provided as an explicit annotation: ``Acquire.spec`` is typed
    as the ``MotionSpec`` union (``Union[Linspace[Any], ...]``), so AxisT cannot
    be bound by the synthesised constructor.  DetectorT and MonitorT are still
    inferred from the ``detectors`` / ``monitors`` list element types.
    """
    spec = Acquire(
        motion,
        detectors=[
            DetectorGroup(
                exposures_per_collection=1,
                collections_per_event=1,
                livetime=0.003,
                deadtime=0.001,
                detectors=["saxs", "waxs"],
            )
        ],
        monitors=[MonitorStream("dcm_temp", "dcm_temperature")],
    )
    assert_type(spec, Acquire[str, str, str])


def test_no_monitors_infers_never() -> None:
    """When monitors is omitted, MonitorT infers to Never -- no annotation.

    MonitorT is a PEP 696 TypeVar with a ``default=Never``, so pyright fills
    it in on its own when nothing constrains it from ``monitors=``.
    """
    spec = Acquire(
        motion,
        detectors=[
            DetectorGroup(
                exposures_per_collection=1,
                collections_per_event=1,
                livetime=0.003,
                deadtime=0.001,
                detectors=["saxs"],
            )
        ],
    )
    assert_type(spec, Acquire[str, str, Never])
