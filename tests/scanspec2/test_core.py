"""Tests for scanspec2.core data structures."""

from typing import Never

import pytest

from scanspec2.core import (
    AxisMotion,
    ContinuousStream,
    DetectorGroup,
    Dimension,
    LinearSource,
    MonitorStream,
    Scan,
    TriggerRepeat,
    TriggerSequence,
    Window,
    WindowedStream,
    WindowGenerator,
    _truncate_trigger_sequence,  # type: ignore[reportPrivateUsage]
)


def test_trigger_repeat():
    tr = TriggerRepeat(num=500, livetime=0.003, deadtime=0.001)
    assert tr.num == 500
    assert tr.livetime == 0.003
    assert tr.deadtime == 0.001


def test_trigger_sequence():
    tr = TriggerRepeat(num=100, livetime=0.01, deadtime=0.001)
    ts = TriggerSequence(
        detectors=frozenset({"det1", "det2"}),
        trigger_repeat=tr,
        children={},
    )
    assert ts.detectors == frozenset({"det1", "det2"})
    assert ts.trigger_repeat == tr
    assert ts.children == {}


def test_trigger_sequence_children():
    parent = TriggerRepeat(num=100, livetime=0.009, deadtime=0.001)
    child_a = TriggerRepeat(num=72, livetime=0.000124, deadtime=0.000001)
    child_b = TriggerRepeat(num=45, livetime=0.00019, deadtime=0.00001)
    ts = TriggerSequence(
        detectors=frozenset({"saxs", "waxs"}),
        trigger_repeat=parent,
        children={
            frozenset({"tetramm"}): [child_a],
            frozenset({"panda"}): [child_b],
        },
    )
    assert ts.detectors == frozenset({"saxs", "waxs"})
    assert ts.trigger_repeat == parent
    assert ts.children[frozenset({"tetramm"})] == [child_a]
    assert ts.children[frozenset({"panda"})] == [child_b]


def test_axis_motion():
    am = AxisMotion(
        start_position=0.0, start_velocity=1.0, end_position=10.0, end_velocity=1.0
    )
    assert am.start_position == 0.0
    assert am.end_position == 10.0


def test_window():
    tr = TriggerRepeat(num=10, livetime=0.001, deadtime=0.0001)
    ts = TriggerSequence(
        detectors=frozenset({"det1"}),
        trigger_repeat=tr,
        children={},
    )
    w = Window(
        static_axes={"y": 5.0},
        moving_axes={"x": AxisMotion(0.0, 1.0, 10.0, 1.0)},
        non_linear=False,
        duration=0.012,
        trigger_sequences=[ts],
        previous=None,
    )
    assert w.static_axes == {"y": 5.0}
    assert "x" in w.moving_axes
    assert w.non_linear is False
    assert w.duration == pytest.approx(0.012)  # type: ignore[reportUnknownMemberType]
    assert w.previous is None


def test_window_previous():
    tr = TriggerRepeat(num=10, livetime=0.001, deadtime=0.0001)
    ts = TriggerSequence(
        detectors=frozenset({"det1"}),
        trigger_repeat=tr,
        children={},
    )
    first = Window(
        static_axes={"y": 5.0},
        moving_axes={},
        non_linear=False,
        duration=0.01,
        trigger_sequences=[ts],
        previous=None,
    )
    second = Window(
        static_axes={"y": 6.0},
        moving_axes={},
        non_linear=False,
        duration=0.01,
        trigger_sequences=[ts],
        previous=first,
    )
    assert second.previous is first


def test_window_positions_returns_dict_directly():
    """positions(times) returns a plain dict, not a generator/chunks."""
    import numpy as np

    def pos_fn(times: np.ndarray) -> dict[str, np.ndarray]:
        return {"x": times}

    w: Window[str, Never] = Window(
        static_axes={},
        moving_axes={"x": AxisMotion(0.0, 1.0, 0.1, 1.0)},
        non_linear=False,
        duration=0.1,
        trigger_sequences=[],
        previous=None,
        positions_fn=pos_fn,
    )

    times = (np.arange(10) + 0.5) * 0.01
    result = w.positions(times)
    assert isinstance(result, dict)
    np.testing.assert_allclose(result["x"], times)


def test_window_positions_raises_without_positions_fn():
    """positions() raises RuntimeError on step windows (no positions_fn)."""
    import numpy as np

    w: Window[str, Never] = Window(
        static_axes={"x": 1.0},
        moving_axes={},
        non_linear=False,
        duration=0.0,
        trigger_sequences=[],
        previous=None,
    )
    with pytest.raises(RuntimeError, match="step windows"):
        w.positions(np.array([0.0]))


def test_truncate_trigger_sequence():
    seqs = [
        TriggerSequence(frozenset({"a"}), TriggerRepeat(5, 0.01, 0.001), {}),
        TriggerSequence(frozenset({"b"}), TriggerRepeat(3, 0.02, 0.002), {}),
    ]

    # trigger_index=0 → all sequences unchanged
    t0 = _truncate_trigger_sequence(seqs, 0)
    assert t0 == seqs

    # trigger_index=6 → first seq fully consumed (5),
    # second seq reduced from 3 to 2 (6-5=1 consumed → num=2)
    t6 = _truncate_trigger_sequence(seqs, 6)
    assert len(t6) == 1
    assert t6[0].detectors == frozenset({"b"})
    assert t6[0].trigger_repeat.num == 2
    assert t6[0].children == {}

    # trigger_index=8 → all consumed, empty result
    t8 = _truncate_trigger_sequence(seqs, 8)
    assert t8 == []


def test_truncate_trigger_sequence_blank_replays_in_full():
    burst1: TriggerSequence[str] = TriggerSequence(
        frozenset({"det"}), TriggerRepeat(100, 0.003, 0.001), {}
    )
    blank: TriggerSequence[str] = TriggerSequence(
        frozenset(), TriggerRepeat(1, 0.0, 50.0), {}
    )
    burst2: TriggerSequence[str] = TriggerSequence(
        frozenset({"det"}), TriggerRepeat(200, 0.003, 0.001), {}
    )
    seqs: list[TriggerSequence[str]] = [burst1, blank, burst2]

    # Pause anywhere in or after the blank (burst1 fully done, blank never
    # counted) -> trigger_index=100 regardless of how far into the 50s blank
    # the pause landed. The blank must survive whole, not be dropped or
    # partially truncated, so the realized gap on resume is never shorter
    # than the designed minimum.
    result = _truncate_trigger_sequence(seqs, 100)
    assert result == [blank, burst2]

    # Pause mid-burst1 (60 of 100 done): burst1 truncates as normal, blank
    # and burst2 downstream are untouched.
    result_mid = _truncate_trigger_sequence(seqs, 60)
    assert len(result_mid) == 3
    assert result_mid[0].trigger_repeat.num == 40
    assert result_mid[1] == blank
    assert result_mid[2] == burst2


def test_scan_dimension():
    import numpy as np

    def pos_fn(indexes: np.ndarray) -> dict[str, np.ndarray]:
        return {"x": indexes}

    sd = Dimension(
        axes=["x"],
        length=100,
        snake=False,
        position_fn=pos_fn,
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
    # Midpoints at half-integer indexes: 0.5, 1.5, 2.5, 3.5, 4.5
    np.testing.assert_allclose(result, [1.0, 3.0, 5.0, 7.0, 9.0])


def test_scan_dimension_setpoints_linear():
    import numpy as np

    gen = WindowGenerator(
        axes=["x"], length=5, source=LinearSource({"x": (0.0, 4.0)}, 5)
    )
    sd = Dimension(
        axes=["x"],
        length=5,
        snake=False,
        position_fn=gen.setpoints,
    )
    result = next(sd.setpoints("x"))
    np.testing.assert_allclose(result, [0.0, 1.0, 2.0, 3.0, 4.0])


def test_scan_dimension_setpoints_chunks():
    import numpy as np

    gen = WindowGenerator(
        axes=["x"], length=5, source=LinearSource({"x": (0.0, 4.0)}, 5)
    )
    sd = Dimension(
        axes=["x"],
        length=5,
        snake=False,
        position_fn=gen.setpoints,
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
    gen = WindowGenerator(
        axes=["x"],
        length=50,
        snake=True,
        source=LinearSource({"x": (0.0, 49.0)}, 50),
    )
    dim = Dimension(
        axes=["x"],
        length=50,
        snake=True,
        position_fn=gen.setpoints,
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
    gen = WindowGenerator(
        axes=["x", "y"],
        length=200,
        source=LinearSource({"x": (0.0, 1.0), "y": (0.0, 1.0)}, 200),
    )
    dim = Dimension(
        axes=["x", "y"],
        length=200,
        snake=True,
        position_fn=gen.setpoints,
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
        generators=[],
        windowed_streams=[ws],
        continuous_streams=[cs],
        monitors=[mon],
    )

    assert scan.windowed_streams[0].name == "diffraction"
    assert scan.windowed_streams[0].dimensions[0].axes == ["x", "y"]
    assert scan.continuous_streams[0].name == "cameras"
    assert scan.monitors[0].detector == "TEMP:PV"


def test_scan_fly():
    ws: WindowedStream[Never, Never] = WindowedStream(
        name="diff", dimensions=[], detector_groups=[]
    )
    gen: WindowGenerator[Never] = WindowGenerator(
        axes=[], length=1, fly=True, source=LinearSource({}, 1)
    )
    scan: Scan[Never, Never, Never] = Scan(
        generators=[gen],
        windowed_streams=[ws],
        continuous_streams=[],
        monitors=[],
    )
    assert scan.generators[0].fly is True
