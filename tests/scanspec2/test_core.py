"""Tests for scanspec2.core data structures."""

from typing import Never

import pytest

from scanspec2.core import (
    AxisMotion,
    ContinuousStream,
    DetectorGroup,
    Dimension,
    LinearPositions,
    MonitorStream,
    Scan,
    TriggerGroup,
    TriggerPattern,
    Window,
    WindowedStream,
)


def test_trigger_pattern():
    tp = TriggerPattern(repeats=500, livetime=0.003, deadtime=0.001)
    assert tp.repeats == 500
    assert tp.livetime == 0.003
    assert tp.deadtime == 0.001


def test_trigger_group():
    tp = TriggerPattern(repeats=100, livetime=0.01, deadtime=0.001)
    tg = TriggerGroup(detectors=["det1", "det2"], trigger_patterns=[tp])
    assert tg.detectors == ["det1", "det2"]
    assert tg.trigger_patterns == [tp]


def test_axis_motion():
    am = AxisMotion(
        start_position=0.0, start_velocity=1.0, end_position=10.0, end_velocity=1.0
    )
    assert am.start_position == 0.0
    assert am.end_position == 10.0


def test_window():
    tp = TriggerPattern(repeats=10, livetime=0.001, deadtime=0.0001)
    tg = TriggerGroup(detectors=["det1"], trigger_patterns=[tp])
    w = Window(
        static_axes={"y": 5.0},
        moving_axes={"x": AxisMotion(0.0, 1.0, 10.0, 1.0)},
        non_linear_move=False,
        duration=0.012,
        trigger_groups=[tg],
        previous=None,
    )
    assert w.static_axes == {"y": 5.0}
    assert "x" in w.moving_axes
    assert w.non_linear_move is False
    assert w.duration == pytest.approx(0.012)  # type: ignore[reportUnknownMemberType]
    assert w.previous is None


def test_window_previous():
    tp = TriggerPattern(repeats=10, livetime=0.001, deadtime=0.0001)
    tg = TriggerGroup(detectors=["det1"], trigger_patterns=[tp])
    first = Window(
        static_axes={"y": 5.0},
        moving_axes={},
        non_linear_move=False,
        duration=0.01,
        trigger_groups=[tg],
        previous=None,
    )
    second = Window(
        static_axes={"y": 6.0},
        moving_axes={},
        non_linear_move=False,
        duration=0.01,
        trigger_groups=[tg],
        previous=first,
    )
    assert second.previous is first


def test_scan_dimension():
    sd = Dimension(
        axes=["x"],
        length=100,
        snake=False,
        position_fn=LinearPositions({"x": (0.0, 99.0)}, length=100),
    )
    assert sd.axes == ["x"]
    assert sd.length == 100
    assert sd.snake is False


def test_scan_dimension_setpoints_with_fn():
    import numpy as np

    def pos_fn(indexes: np.ndarray) -> dict[str, np.ndarray]:
        return {"x": indexes * 2.0}

    sd = Dimension(axes=["x"], length=5, snake=False, position_fn=pos_fn)
    result = next(sd.setpoints("x"))
    np.testing.assert_allclose(result, [0.0, 2.0, 4.0, 6.0, 8.0])
    assert sd.non_linear is True


def test_scan_dimension_setpoints_linear():
    import numpy as np

    sd = Dimension(
        axes=["x"],
        length=5,
        snake=False,
        position_fn=LinearPositions({"x": (0.0, 4.0)}, length=5),
    )
    result = next(sd.setpoints("x"))
    np.testing.assert_allclose(result, [0.0, 1.0, 2.0, 3.0, 4.0])
    assert sd.non_linear is False


def test_scan_dimension_setpoints_chunks():
    import numpy as np

    sd = Dimension(
        axes=["x"],
        length=5,
        snake=False,
        position_fn=LinearPositions({"x": (0.0, 4.0)}, length=5),
    )
    chunks = list(sd.setpoints("x", chunk_size=2))
    np.testing.assert_allclose(chunks[0], [0.0, 1.0])
    np.testing.assert_allclose(chunks[1], [2.0, 3.0])
    np.testing.assert_allclose(chunks[2], [4.0])


def test_detector_group():
    dg = DetectorGroup(
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=0.01,
        deadtime=0.001,
        detectors=["eiger"],
    )
    assert dg.detectors == ["eiger"]
    assert dg.livetime == pytest.approx(0.01)  # type: ignore[reportUnknownMemberType]


def test_detector_group_none_timing():
    dg = DetectorGroup(
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=None,
        deadtime=None,
        detectors=["det"],
    )
    assert dg.livetime is None
    assert dg.deadtime is None


def test_windowed_stream():
    dim = Dimension(
        axes=["x"],
        length=50,
        snake=True,
        position_fn=LinearPositions({"x": (0.0, 49.0)}, length=50),
    )
    dg = DetectorGroup(
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=0.005,
        deadtime=0.0005,
        detectors=["eiger"],
    )
    ws = WindowedStream(name="diffraction", dimensions=[dim], detector_groups=[dg])
    assert ws.name == "diffraction"
    assert ws.dimensions[0].length == 50


def test_continuous_stream():
    dg = DetectorGroup(
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=0.05,
        deadtime=0.005,
        detectors=["front_cam", "side_cam"],
    )
    cs = ContinuousStream(name="cameras", detector_groups=[dg])
    assert cs.name == "cameras"
    assert cs.detector_groups[0].detectors == ["front_cam", "side_cam"]


def test_monitor_stream():
    ms = MonitorStream(name="temperature", detector="BL02I-EA-TEMP-01:TEMP")
    assert ms.name == "temperature"
    assert ms.detector == "BL02I-EA-TEMP-01:TEMP"


def test_scan_step():
    dim = Dimension(
        axes=["x", "y"],
        length=200,
        snake=True,
        position_fn=LinearPositions({"x": (0.0, 1.0), "y": (0.0, 1.0)}, length=200),
    )
    dg = DetectorGroup(
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=0.01,
        deadtime=0.001,
        detectors=["eiger"],
    )
    ws = WindowedStream(name="diffraction", dimensions=[dim], detector_groups=[dg])
    cs: ContinuousStream[str] = ContinuousStream(name="cameras", detector_groups=[])
    mon = MonitorStream(name="temperature", detector="TEMP:PV")
    scan = Scan(
        motion_dims=[],
        windowed_streams=[ws],
        continuous_streams=[cs],
        monitors=[mon],
        fly=False,
    )

    assert scan.fly is False
    assert scan.windowed_streams[0].name == "diffraction"
    assert scan.windowed_streams[0].dimensions[0].axes == ["x", "y"]
    assert scan.continuous_streams[0].name == "cameras"
    assert scan.monitors[0].detector == "TEMP:PV"


def test_scan_fly():
    ws: WindowedStream[Never, Never] = WindowedStream(
        name="diff", dimensions=[], detector_groups=[]
    )
    scan: Scan[Never, Never, Never] = Scan(
        motion_dims=[],
        windowed_streams=[ws],
        continuous_streams=[],
        monitors=[],
        fly=True,
    )
    assert scan.fly is True
