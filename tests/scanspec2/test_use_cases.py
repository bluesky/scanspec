"""End-to-end use-case tests for scanspec2."""

from __future__ import annotations

from typing import Never, cast

import pytest

from scanspec2.core import (
    ContinuousStream,
    DetectorGroup,
    MonitorStream,
    Scan,
    TriggerPattern,
)
from scanspec2.specs import Acquire, Linspace, Product, Repeat, Spiral, Static

from .. import approx


def test_linspace_step_scan():
    """Iterate Linspace(x, 0, 1, 5) — check all window fields."""
    linspace = Linspace("x", 0, 1, 5)
    windows = list(linspace.compile())
    assert len(windows) == 5

    expected_x = [0.0, 0.25, 0.5, 0.75, 1.0]
    for i, (w, x) in enumerate(zip(windows, expected_x, strict=True)):
        # static_axes: x moves to the target position
        assert w.static_axes == {"x": approx(x)}
        # step scan: no moving axes
        assert w.moving_axes == {}
        # step scan: not a continuous trajectory
        assert w.non_linear_move is False
        # step scan: no detectors, duration is 0
        assert w.duration == approx(0.0)
        # no detectors configured on bare Linspace
        assert w.trigger_groups == []
        # previous window linkage
        if i == 0:
            assert w.previous is None
        else:
            assert w.previous is windows[i - 1]


def test_linspace_fly_scan():
    """Acquire(fly=True, Linspace(x, 0, 1, 5), duration=0.004) — one fly window."""
    linspace = Linspace("x", 0, 1, 5)
    scan = cast(
        Scan[str, Never, Never],
        Acquire(linspace, fly=True, duration=0.004).compile(),
    )
    windows = list(scan)
    assert len(windows) == 1

    w = windows[0]
    # fly scan: no static axes (only window, so no prior)
    assert w.static_axes == {}
    # fly scan: x moves continuously
    assert "x" in w.moving_axes
    am = w.moving_axes["x"]
    # f(i) = 0 + i * 0.25; boundary at i=-0.5 and i=4.5
    assert am.start_position == approx(-0.125)
    assert am.end_position == approx(1.125)
    assert am.start_velocity == approx(0.25)
    assert am.end_velocity == approx(0.25)
    # linear sweep
    assert w.non_linear_move is False
    # 5 points × 0.004 s/point = 0.02 s
    assert w.duration == approx(0.02)
    # no detectors configured
    assert w.trigger_groups == []
    assert w.previous is None


def test_spiral_step_scan():
    """Spiral(x, 0, 5, 2, y, 10, 10) step scan — check positions and non_linear flag."""
    scan = Spiral("x", 0, 5, 2, "y", 10, 10).compile()
    windows = list(scan)
    assert len(windows) == 10

    assert windows[0].non_linear_move is False

    x_pos = [w.static_axes["x"] for w in windows]
    y_pos = [w.static_axes["y"] for w in windows]

    # Midpoints with phi-offset=1.0 (first ring, not the degenerate centre)
    assert x_pos == approx(
        [-0.31, -1.07, -0.20, 1.14, 1.76, 1.31, 0.10, -1.27, -2.22, -2.44], abs=0.01
    )
    assert y_pos == approx(
        [8.55, 10.66, 12.71, 12.19, 9.74, 7.14, 5.82, 6.31, 8.33, 11.06], abs=0.01
    )


def test_spiral_fly_scan():
    """Acquire(fly=True, Spiral(...)) — one fly window covering all 10 spiral points."""
    scan = cast(
        Scan[str, Never, Never],
        Acquire(Spiral("x", 0, 5, 2, "y", 10, 10), fly=True).compile(),
    )
    windows = list(scan)
    assert len(windows) == 1

    w = windows[0]
    # The spiral position function is non-linear
    assert w.non_linear_move is True
    # duration = |9.5 - (-0.5)| = 10.0 index units (10 points)
    assert w.duration == approx(10.0)
    # only window, so no prior outer position to report
    assert w.static_axes == {}

    # Start boundary i=-0.5: phi=sqrt(2π) — first ring approach, well-defined
    assert w.moving_axes["x"].start_position == approx(0.332, abs=0.001)
    assert w.moving_axes["y"].start_position == approx(9.100, abs=0.001)
    assert w.moving_axes["x"].start_velocity == approx(-0.797, abs=0.001)
    assert w.moving_axes["y"].start_velocity == approx(-2.562, abs=0.001)

    # End boundary i=9.5: outermost ring exit
    assert w.moving_axes["x"].end_position == approx(-2.259, abs=0.001)
    assert w.moving_axes["y"].end_position == approx(12.417, abs=0.001)
    assert w.moving_axes["x"].end_velocity == approx(0.553, abs=0.001)
    assert w.moving_axes["y"].end_velocity == approx(2.586, abs=0.001)


def test_flagship_multi_stream_concat():
    """Flagship pattern: repeat(diff_step + spec_fly_fwd + spec_fly_rev).

    200 iterations of:
      1. Step to e=7.0, take 1 diffraction image (stream "diff")
      2. Fly e 7.0→7.1, 1000 spectroscopy frames (stream "spec")
      3. Fly e 7.1→7.0, 1000 spectroscopy frames (stream "spec")

    600 windows total: 200 × (1 step + 1 fly + 1 fly).
    """
    diff_det = DetectorGroup(1, 1, 0.01, 0.001, ["diffraction"])
    spec_det = DetectorGroup(1, 1, 0.003, 0.001, ["spectroscopy"])

    diff_acq: Acquire[str, str, Never] = Acquire(
        Static("e", 7.0),
        detectors=[diff_det],
        stream_name="diff",
    )
    spec_fwd: Acquire[str, str, Never] = Acquire(
        Linspace("e", 7.0, 7.1, 1000),
        fly=True,
        detectors=[spec_det],
        stream_name="spec",
    )
    spec_rev: Acquire[str, str, Never] = Acquire(
        Linspace("e", 7.1, 7.0, 1000),
        fly=True,
        detectors=[spec_det],
        stream_name="spec",
    )
    spec: Repeat[str, str, Never] = Repeat(
        diff_acq.concat(spec_fwd).concat(spec_rev),
        num=200,
    )
    scan: Scan[str, str, Never] = spec.compile()

    # --- Windowed streams ---
    streams_by_name = {s.name: s for s in scan.windowed_streams}
    assert set(streams_by_name) == {"diff", "spec"}

    diff_stream = streams_by_name["diff"]
    assert len(diff_stream.dimensions) == 2
    assert diff_stream.dimensions[0].length == 200  # repeat
    assert diff_stream.dimensions[1].length == 1  # static
    assert diff_stream.dimensions[1].axes == ["e"]

    spec_stream = streams_by_name["spec"]
    assert len(spec_stream.dimensions) == 2
    assert spec_stream.dimensions[0].length == 200  # repeat
    assert spec_stream.dimensions[1].length == 2000  # 1000 + 1000 concat
    assert spec_stream.dimensions[1].axes == ["e"]

    # --- Window iteration ---
    windows = list(scan)
    assert len(windows) == 600

    # Each group of 3 windows: step, fly forward, fly reverse
    for rep in range(200):
        base = rep * 3
        w_diff = windows[base]
        w_fwd = windows[base + 1]
        w_rev = windows[base + 2]

        # Window 0: step to e=7.0
        assert w_diff.static_axes["e"] == approx(7.0)
        assert w_diff.moving_axes == {}
        assert w_diff.non_linear_move is False
        # duration from detector: 1 × (0.01 + 0.001) = 0.011
        assert w_diff.duration == approx(0.011)
        # trigger_groups from diff_det
        assert len(w_diff.trigger_groups) == 1
        tg = w_diff.trigger_groups[0]
        assert tg.detectors == ["diffraction"]
        assert tg.trigger_patterns == [TriggerPattern(1, 0.01, 0.001)]

        # Window 1: fly e 7.0 → 7.1
        assert w_fwd.moving_axes != {}
        assert "e" in w_fwd.moving_axes
        am_fwd = w_fwd.moving_axes["e"]
        assert am_fwd.start_position < am_fwd.end_position
        assert w_fwd.non_linear_move is False  # linear Linspace
        # duration from detector: 1000 × (0.003 + 0.001) = 4.0
        assert w_fwd.duration == approx(4.0)
        assert len(w_fwd.trigger_groups) == 1
        assert w_fwd.trigger_groups[0].detectors == ["spectroscopy"]
        assert w_fwd.trigger_groups[0].trigger_patterns == [
            TriggerPattern(1000, 0.003, 0.001)
        ]

        # Window 2: fly e 7.1 → 7.0
        assert w_rev.moving_axes != {}
        assert "e" in w_rev.moving_axes
        am_rev = w_rev.moving_axes["e"]
        assert am_rev.start_position > am_rev.end_position
        assert w_rev.non_linear_move is False
        assert w_rev.duration == approx(4.0)
        assert len(w_rev.trigger_groups) == 1
        assert w_rev.trigger_groups[0].detectors == ["spectroscopy"]
        assert w_rev.trigger_groups[0].trigger_patterns == [
            TriggerPattern(1000, 0.003, 0.001)
        ]

    # Previous chain is connected across all 600 windows
    assert windows[0].previous is None
    for i in range(1, len(windows)):
        assert windows[i].previous is windows[i - 1]


@pytest.mark.parametrize("fly", [True, False])
def test_maximal_fly_step(fly: bool):
    """Maximal example: y(50) * ~x(100) with multi-rate detectors, cameras, monitor.

    fly=True: 50 fly windows (one per y row), x sweeps continuously.
    fly=False: 5000 step windows (50 y × 100 x).
    """
    spec = Acquire(
        Product(Linspace("y", 0, 5, 50), ~Linspace("x", 0, 10, 100)),
        fly=fly,
        detectors=[
            DetectorGroup(1, 1, 0.003, 0.001, ["saxs", "waxs"]),
            DetectorGroup(10, 1, 0.0003, 8e-9, ["timestamp", "x_enc", "y_enc"]),
        ],
        continuous_streams=[
            ContinuousStream(
                "cameras",
                [DetectorGroup(1, 1, 0.048, 0.001, ["front_cam", "side_cam"])],
            ),
        ],
        monitors=[
            MonitorStream("temperature", "tc1"),
        ],
    )
    scan = spec.compile()

    # --- Stream structure ---
    assert len(scan.windowed_streams) == 1
    assert scan.windowed_streams[0].name == "primary"
    assert len(scan.continuous_streams) == 1
    assert scan.continuous_streams[0].name == "cameras"
    assert len(scan.monitors) == 1
    assert scan.monitors[0].detector == "tc1"

    # --- Dimensions ---
    dims = scan.windowed_streams[0].dimensions
    assert len(dims) == 2
    assert dims[0].axes == ["y"]
    assert dims[0].length == 50
    assert dims[1].axes == ["x"]
    assert dims[1].length == 100
    assert dims[1].snake is True

    # --- Window count ---
    windows = list(scan)
    if fly:
        assert len(windows) == 50
    else:
        assert len(windows) == 5000

    # --- Trigger groups on every window ---
    for w in windows:
        assert len(w.trigger_groups) == 2
        tg_saxs = w.trigger_groups[0]
        tg_enc = w.trigger_groups[1]
        assert tg_saxs.detectors == ["saxs", "waxs"]
        assert tg_enc.detectors == ["timestamp", "x_enc", "y_enc"]

        if fly:
            # fly: repeats = inner_length × exposures_per_collection
            assert tg_saxs.trigger_patterns == [TriggerPattern(100, 0.003, 0.001)]
            assert tg_enc.trigger_patterns == [TriggerPattern(1000, 0.0003, 8e-9)]
        else:
            # step: repeats = exposures_per_collection
            assert tg_saxs.trigger_patterns == [TriggerPattern(1, 0.003, 0.001)]
            assert tg_enc.trigger_patterns == [TriggerPattern(10, 0.0003, 8e-9)]

    # --- Duration ---
    # saxs: repeats × (livetime + deadtime):
    #   fly: 100 × 0.004 = 0.4
    #   step: 1 × 0.004 = 0.004
    # encoder: fly: 1000 × 0.0003008 ≈ 0.3008, step: 10 × 0.0003008 ≈ 0.003008
    # per_point = max(total_dur) / inner_length (fly) or max(total_dur) (step)
    if fly:
        # total_dur: max(0.4, 0.3008) = 0.4; per_point = 0.4 / 100 = 0.004
        # window duration = 100 × 0.004 = 0.4
        for w in windows:
            assert w.duration == approx(0.4)
    else:
        # total_dur: max(0.004, 0.003008) = 0.004; per_point = 0.004
        for w in windows:
            assert w.duration == approx(0.004)

    # --- Fly: moving_axes, snake direction ---
    if fly:
        prev_y: dict[str, float] = {}
        for i, w in enumerate(windows):
            prev_y.update(w.static_axes)
            assert "x" in w.moving_axes
            assert "y" not in w.moving_axes
            am = w.moving_axes["x"]
            # x snakes: even rows forward, odd rows backward
            if i % 2 == 0:
                assert am.start_position < am.end_position
            else:
                assert am.start_position > am.end_position
        # y values should be 0..5 in 50 steps
        assert prev_y["y"] == approx(5.0)
    else:
        # Step: all windows have empty moving_axes
        for w in windows:
            assert w.moving_axes == {}

    # --- Previous chain ---
    assert windows[0].previous is None
    for i in range(1, len(windows)):
        assert windows[i].previous is windows[i - 1]
