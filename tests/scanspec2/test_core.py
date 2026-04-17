"""Tests for scanspec2.core data structures."""

from typing import Never

import pytest

from scanspec2.core import (
    AxisMotion,
    ContinuousStream,
    DetectorGroup,
    MonitorStream,
    Scan,
    ScanDimension,
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
    sd = ScanDimension(axes=["x"], length=100, snake=False)
    assert sd.axes == ["x"]
    assert sd.length == 100
    assert sd.snake is False


def test_scan_dimension_setpoints_raises():
    sd = ScanDimension(axes=["x"], length=100, snake=False)
    with pytest.raises(NotImplementedError):
        next(sd.setpoints("x"))


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
    dim = ScanDimension(axes=["x"], length=50, snake=True)
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
    dim = ScanDimension(axes=["x", "y"], length=200, snake=True)
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
        windowed_streams=[ws], continuous_streams=[cs], monitors=[mon], fly=False
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
        windowed_streams=[ws], continuous_streams=[], monitors=[], fly=True
    )
    assert scan.fly is True
