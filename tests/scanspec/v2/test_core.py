"""Tests for scanspec.v2.core data structures."""

from typing import Never

import pytest

from scanspec.v2.core import (
    AxisMotion,
    ContinuousStream,
    DetectorGroup,
    Dimension,
    LinearSource,
    MonitorStream,
    Scan,
    TriggerFollower,
    TriggerGroup,
    TriggerRepeat,
    TriggerSequence,
    Window,
    WindowedStream,
    WindowGenerator,
    _truncate_trigger_sequence,  # type: ignore[reportPrivateUsage]
    validate_trigger_sequence,
)
from scanspec.v2.specs import Linspace, Product, Sync


def test_trigger_repeat():
    tr = TriggerRepeat(
        detectors=frozenset({"det1"}), repeats=500, livetime=0.003, deadtime=0.001
    )
    assert tr.detectors == frozenset({"det1"})
    assert tr.repeats == 500
    assert tr.livetime == 0.003
    assert tr.deadtime == 0.001


def test_trigger_sequence():
    tr = TriggerRepeat(
        detectors=frozenset({"det1", "det2"}),
        repeats=100,
        livetime=0.01,
        deadtime=0.001,
    )
    ts = TriggerSequence(root=tr, children=[])
    assert ts.root.detectors == frozenset({"det1", "det2"})
    assert ts.root == tr
    assert ts.children == []


def test_trigger_sequence_children():
    root = TriggerRepeat(
        detectors=frozenset({"saxs", "waxs"}),
        repeats=100,
        livetime=0.009,
        deadtime=0.001,
    )
    child_a = TriggerRepeat(
        detectors=frozenset({"tetramm"}),
        repeats=72,
        livetime=0.000124,
        deadtime=0.000001,
    )
    child_b = TriggerRepeat(
        detectors=frozenset({"panda"}), repeats=45, livetime=0.00019, deadtime=0.00001
    )
    ts = TriggerSequence(root=root, children=[child_a, child_b])
    assert ts.root.detectors == frozenset({"saxs", "waxs"})
    assert ts.root == root
    by_det = {child.detectors: child for child in ts.children}
    assert by_det[frozenset({"tetramm"})] == child_a
    assert by_det[frozenset({"panda"})] == child_b


def test_trigger_group_json_round_trip_unresolved_timing():
    """The motivating case: a draft with unresolved livetime/deadtime must
    still round-trip -- e.g. sent to ophyd-async, which fills in real
    device limits and sends it back. This lives on the authoring type
    (TriggerGroup) now, not the compiled TriggerSequence/TriggerRepeat --
    see ADR 0008 Decision 6."""
    draft = TriggerGroup(
        detectors=frozenset({"saxs", "waxs"}),
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=None,
        deadtime=None,
    )
    wire = draft.model_dump_json()
    assert '"livetime":null' in wire
    round_tripped = TriggerGroup[str].model_validate_json(wire)
    assert round_tripped == draft
    assert round_tripped.livetime is None


def test_axis_motion():
    am = AxisMotion(
        start_position=0.0, start_velocity=1.0, end_position=10.0, end_velocity=1.0
    )
    assert am.start_position == 0.0
    assert am.end_position == 10.0


def test_window():
    tr = TriggerRepeat(
        detectors=frozenset({"det1"}), repeats=10, livetime=0.001, deadtime=0.0001
    )
    ts = TriggerSequence(root=tr, children=[])
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
    tr = TriggerRepeat(
        detectors=frozenset({"det1"}), repeats=10, livetime=0.001, deadtime=0.0001
    )
    ts = TriggerSequence(root=tr, children=[])
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
        TriggerSequence(
            root=TriggerRepeat(
                detectors=frozenset({"a"}), repeats=5, livetime=0.01, deadtime=0.001
            ),
            children=[],
        ),
        TriggerSequence(
            root=TriggerRepeat(
                detectors=frozenset({"b"}), repeats=3, livetime=0.02, deadtime=0.002
            ),
            children=[],
        ),
    ]

    # trigger_index=0 → all sequences unchanged
    t0 = _truncate_trigger_sequence(seqs, 0)
    assert t0 == seqs

    # trigger_index=6 → first seq fully consumed (5),
    # second seq reduced from 3 to 2 (6-5=1 consumed → repeats=2)
    t6 = _truncate_trigger_sequence(seqs, 6)
    assert len(t6) == 1
    assert t6[0].root.detectors == frozenset({"b"})
    assert t6[0].root.repeats == 2
    assert t6[0].children == []

    # trigger_index=8 → all consumed, empty result
    t8 = _truncate_trigger_sequence(seqs, 8)
    assert t8 == []


def test_truncate_trigger_sequence_blank_replays_in_full():
    burst1: TriggerSequence[str] = TriggerSequence(
        root=TriggerRepeat(
            detectors=frozenset({"det"}), repeats=100, livetime=0.003, deadtime=0.001
        ),
        children=[],
    )
    blank: TriggerSequence[str] = TriggerSequence(
        root=TriggerRepeat(
            detectors=frozenset(), repeats=1, livetime=0.0, deadtime=50.0
        ),
        children=[],
    )
    burst2: TriggerSequence[str] = TriggerSequence(
        root=TriggerRepeat(
            detectors=frozenset({"det"}), repeats=200, livetime=0.003, deadtime=0.001
        ),
        children=[],
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
    assert result_mid[0].root.repeats == 40
    assert result_mid[1] == blank
    assert result_mid[2] == burst2


def test_child_duration_exceeds_parent_livetime_raises():
    """A clean integer ratio (period=0.0003, ratio=10) but a hand-set
    repeats=11 makes total child duration 0.0033 > the 0.003 root livetime.

    Constructs TriggerRepeat/TriggerSequence directly to isolate
    validate_trigger_sequence's own check. Under ADR 0008 (as amended for
    the TriggerGroup/TriggerFollower merge), this exact combination is also
    reachable through the normal authoring surface now: TriggerFollower's
    repeats is caller-supplied, not derived from the ratio, so nothing on
    TriggerGroup prevents authoring an inconsistent repeats/timing
    combination -- validate_trigger_sequence (via compile()) is what
    catches it either way.
    """
    root = TriggerRepeat(
        detectors=frozenset({"saxs"}), repeats=100, livetime=0.003, deadtime=0.001
    )
    child = TriggerRepeat(
        detectors=frozenset({"enc"}), repeats=11, livetime=0.0002, deadtime=0.0001
    )
    seq = TriggerSequence(root=root, children=[child])
    with pytest.raises(ValueError, match="exceeds parent livetime"):
        validate_trigger_sequence(seq)


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


def test_windowed_stream_number_of_events_step_product():
    """Two step dimensions: outer * inner."""
    outer = Dimension(axes=["y"], length=5, snake=False, position_fn=lambda _: {})
    inner = Dimension(axes=["x"], length=10, snake=False, position_fn=lambda _: {})
    ws: WindowedStream[str, Never] = WindowedStream(
        name="diff", dimensions=[outer, inner], detector_groups=[]
    )
    assert ws.number_of_events == 50


def test_windowed_stream_number_of_events_fly_agnostic():
    """A flown dimension's length is a real point count, not collapsed to 1.

    Regression test: number_of_events used to be computed from generator
    window counts, which collapse a fly generator's contribution to 1
    regardless of its length. Dimension.length is fly-agnostic, so the
    product below must be 5 * 100, not 5 * 1.
    """
    outer = Dimension(axes=["y"], length=5, snake=False, position_fn=lambda _: {})
    inner_fly = Dimension(axes=["x"], length=100, snake=False, position_fn=lambda _: {})
    ws: WindowedStream[str, Never] = WindowedStream(
        name="diff", dimensions=[outer, inner_fly], detector_groups=[]
    )
    assert ws.number_of_events == 500


def test_windowed_stream_number_of_events_matches_iteration():
    """number_of_events agrees with actually counting a matching Scan's output."""
    outer_gen = WindowGenerator(
        axes=["y"], length=4, source=LinearSource({"y": (0.0, 1.0)}, 4)
    )
    inner_gen = WindowGenerator(
        axes=["x"], length=7, source=LinearSource({"x": (0.0, 1.0)}, 7)
    )
    scan: Scan[str, Never, Never] = Scan(generators=[outer_gen, inner_gen])
    outer_dim = Dimension(axes=["y"], length=4, snake=False, position_fn=lambda _: {})
    inner_dim = Dimension(axes=["x"], length=7, snake=False, position_fn=lambda _: {})
    ws: WindowedStream[str, Never] = WindowedStream(
        name="diff", dimensions=[outer_dim, inner_dim], detector_groups=[]
    )
    assert ws.number_of_events == len(list(scan))


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


def test_ophyd_async_trigger_info():
    """PRD §8: ophyd-async — map WindowedStream to TriggerInfo.

    The consumer maps DetectorGroup + WindowedStream.number_of_events onto
    ophyd_async.core.TriggerInfo for StandardDetector.prepare(). The root
    group's total comes straight from number_of_events (fly-agnostic,
    O(1)). A follower's own rate isn't in the compiled DetectorGroup (it
    has no `repeats` field), so its total is found by walking windows: a
    follower's `repeats` is per *root repeat*, not per window (TriggerGroup
    docstring: "each fires during every one of the group's own repeats"),
    so each window contributes `root.repeats * child.repeats`, summed
    across all windows. DetectorTrigger (electrical trigger mode) has no
    scanspec analogue at all -- it's hardware wiring, supplied by the
    consumer, never derived from the spec. It isn't a free choice per
    group either: root has nothing upstream of it (INTERNAL), while a
    follower is physically gated by the root's own SEQ output (see the
    "chained pair of SEQ blocks" worked example in API_SPEC.md) -- SEQ2's
    OA1 there is held high for the child's own livetime per repeat, a
    constant-width gate, so CONSTANT_GATE, not INTERNAL or EDGE_TRIGGER.
    """
    from ophyd_async.core import DetectorTrigger, TriggerInfo

    root_group = TriggerGroup(
        detectors=frozenset({"det1"}),
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=0.003,
        deadtime=0.001,
        followers=[
            TriggerFollower(
                detectors=frozenset({"enc"}),
                exposures_per_collection=10,
                collections_per_event=1,
                livetime=0.000299992,
                deadtime=8e-9,
                repeats=10,
            ),
        ],
    )

    for fly in (True, False):
        spec: Sync[str, str, Never] = Sync(
            Product(Linspace("y", 0, 5, 3), ~Linspace("x", 0, 10, 5)),
            fly=fly,
            trigger_group=root_group,
        )
        scan = spec.compile()
        stream = scan.windowed_streams[0]

        det_dg = next(g for g in stream.detector_groups if g.detectors == ["det1"])
        enc_dg = next(g for g in stream.detector_groups if g.detectors == ["enc"])

        # Root: O(1), fly-agnostic. Hand-derived: 3 y-points * 5 x-points = 15,
        # regardless of whether x is flown or stepped.
        assert det_dg.livetime is not None
        assert det_dg.deadtime is not None
        det_info = TriggerInfo(
            number_of_events=stream.number_of_events,
            trigger=DetectorTrigger.INTERNAL,  # nothing upstream of root
            livetime=det_dg.livetime,
            deadtime=det_dg.deadtime,
            exposures_per_event=det_dg.exposures_per_event,
        )
        assert det_info.number_of_events == 15
        assert det_info.total_number_of_exposures == 15

        # Follower: walk windows, since the compiled DetectorGroup has no
        # `repeats` field. Hand-derived: 15 root events * 10 follower
        # repeats-per-root-repeat = 150, independent of fly.
        enc_total = 0
        for window in scan:
            seq = next(
                s
                for s in window.trigger_sequences
                if s.root.detectors == frozenset({"det1"})
            )
            child = next(c for c in seq.children if c.detectors == frozenset({"enc"}))
            enc_total += seq.root.repeats * child.repeats
        assert enc_dg.livetime is not None
        assert enc_dg.deadtime is not None
        enc_info = TriggerInfo(
            number_of_events=enc_total,
            # Gated by root's own SEQ output, not free-running -- see
            # SEQ2's BITA <- SEQ1.OA wiring in the chained-SEQ-blocks
            # example in API_SPEC.md.
            trigger=DetectorTrigger.CONSTANT_GATE,
            livetime=enc_dg.livetime,
            deadtime=enc_dg.deadtime,
            exposures_per_event=enc_dg.exposures_per_event,
        )
        assert enc_info.number_of_events == 150
        # 150 * exposures_per_event(10)
        assert enc_info.total_number_of_exposures == 1500
