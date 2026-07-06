"""Type-inference verification for the scanspec2 Acquire API.

Demonstrates that pyright infers ``DetectorT`` and ``MonitorT`` for
``Acquire[AxisT, DetectorT, MonitorT]`` from the constructor arguments, so
explicit annotation is usually not needed at the call site.

These tests are checked statically by pyright (``tox -e type-checking``) and
executed by pytest (``assert_type`` is a no-op at runtime in Python ≥ 3.11).
"""

from __future__ import annotations

from typing import Any, assert_type

from scanspec2.core import DetectorGroup, MonitorStream
from scanspec2.specs import Acquire, Linspace

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


def test_no_monitors_requires_explicit_annotation() -> None:
    """When monitors is omitted, MonitorT is unconstrained.

    An explicit annotation of ``Acquire[str, str, Any]`` is needed to tell
    pyright what MonitorT is.
    """
    spec: Acquire[str, str, Any] = Acquire(
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
    _ = spec  # acknowledge spec to silence unused-variable warning
