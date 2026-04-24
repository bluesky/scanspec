"""End-to-end use-case tests for scanspec2."""

from __future__ import annotations

from typing import Never, cast

import numpy as np
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
        assert w.non_linear is False
        # step scan: no detectors, duration is 0
        assert w.duration == approx(0.0)
        # no detectors configured on bare Linspace
        assert w.trigger_groups == []
        # previous window linkage
        if i == 0:
            assert w.previous is None
        else:
            assert w.previous is windows[i - 1]

    # Step scan windows have no position function — positions() should error.
    with pytest.raises(RuntimeError, match="No position function"):
        next(windows[0].positions(dt=1.0))


def test_linspace_fly_scan():
    """Acquire(fly=True, Linspace(x, 0, 1, 5)) — one fly window."""
    linspace = Linspace("x", 0, 1, 5)
    scan = cast(
        Scan[str, Never, Never],
        Acquire(linspace, fly=True).compile(),
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
    assert w.non_linear is False
    # index-based duration: 5 points
    assert w.duration == approx(5.0)
    # no detectors configured
    assert w.trigger_groups == []
    assert w.previous is None

    # Fly window positions: dt=1.0 → 5 points at integer indexes.
    all_chunks = list(w.positions(dt=1.0))
    assert len(all_chunks) == 1
    full_x = np.concatenate([ch["x"] for ch in all_chunks])
    # Positions at indexes [0,1,2,3,4]: boundary-to-boundary sweep
    assert full_x == approx([-0.125, 0.1875, 0.5, 0.8125, 1.125])


def test_spiral_step_scan():
    """Spiral(x, 0, 5, 2, y, 10, 10) step scan — check positions and non_linear flag."""
    scan = Spiral("x", 0, 5, 2, "y", 10, 10).compile()
    windows = list(scan)
    assert len(windows) == 10

    assert windows[0].non_linear is False

    # Step scan windows should error on positions().
    with pytest.raises(RuntimeError, match="No position function"):
        next(windows[0].positions(dt=1.0))

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
    assert w.non_linear is True
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

    # Fly window should have a positions function for spiral trajectory.
    # dt=1.0, duration=10.0 → 10 points in 1 chunk.
    all_chunks = list(w.positions(dt=1.0))
    assert len(all_chunks) == 1
    full_x = np.concatenate([ch["x"] for ch in all_chunks])
    full_y = np.concatenate([ch["y"] for ch in all_chunks])
    assert full_x == approx(
        [0.332, -0.981, -0.549, 0.946, 1.757, 1.255, -0.138, -1.590, -2.401, -2.259],
        abs=0.001,
    )
    assert full_y == approx(
        [9.100, 9.576, 12.366, 12.451, 9.900, 7.028, 5.776, 6.748, 9.355, 12.417],
        abs=0.001,
    )


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
    scan: Scan[str, str, str] = Acquire(  # type: ignore[reportUnknownVariableType]
        spec,
        monitors=[MonitorStream("temperature", "tc1")],
    ).compile()  # type: ignore[reportArgumentType]

    # --- Monitors ---
    assert len(scan.monitors) == 1
    assert scan.monitors[0].name == "temperature"
    assert scan.monitors[0].detector == "tc1"

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
        assert w_diff.non_linear is False
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
        assert w_fwd.non_linear is False  # linear Linspace
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
        assert w_rev.non_linear is False
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


# ---------- Consumption use cases from API_SPEC ----------


def test_panda_sequence_table():
    """Use case 2: PandA flyscan — build a sequence table from trigger_groups.

    The consumer receives a Scan, finds its trigger group by detector name,
    and reads trigger_patterns + moving_axes to populate a PandA sequence table.
    """
    spec: Acquire[str, str, Never] = Acquire(
        Product(Linspace("y", 0, 5, 3), ~Linspace("x", 0, 10, 50)),
        fly=True,
        detectors=[
            DetectorGroup(1, 1, 0.003, 0.001, ["saxs", "waxs"]),
        ],
    )
    scan = spec.compile()
    det_key = frozenset(["saxs", "waxs"])

    for window in scan:
        # Consumer locates its group by matching detector names
        group = next(
            g for g in window.trigger_groups if frozenset(g.detectors) == det_key
        )

        # Trigger patterns are baked — consumer reads them directly for SeqTable
        assert len(group.trigger_patterns) == 1
        pattern = group.trigger_patterns[0]
        assert pattern.repeats == 50
        assert pattern.livetime == approx(0.003)
        assert pattern.deadtime == approx(0.001)

        # PandA needs start_velocity to pick compare axis
        assert len(window.moving_axes) == 1
        assert "x" in window.moving_axes
        am = window.moving_axes["x"]
        assert am.start_velocity != 0.0

        # Consumer-side: time1 = int(livetime * 1e6), time2 = int(deadtime * 1e6)
        time1 = int(pattern.livetime * 1e6)
        time2 = int(pattern.deadtime * 1e6)
        assert time1 == 3000
        assert time2 == 1000


def test_motor_record_fly():
    """Use case 3: Motor record — single-axis constant-velocity flyscan.

    Consumer reads moving_axes for one axis, computes acceleration ramp
    from boundary kinematics, then drives the motor.
    """
    spec: Acquire[str, str, Never] = Acquire(
        Linspace("x", 0, 10, 100),
        fly=True,
        detectors=[DetectorGroup(1, 1, 0.003, 0.001, ["det1"])],
    )
    scan = spec.compile()
    windows = list(scan)
    assert len(windows) == 1

    w = windows[0]
    # Must be linear for motor record
    assert w.non_linear is False
    # Exactly one moving axis
    assert len(w.moving_axes) == 1
    axis, motion = next(iter(w.moving_axes.items()))
    assert axis == "x"

    # Constant velocity: start_velocity == end_velocity
    assert motion.start_velocity == approx(motion.end_velocity)
    velocity = motion.start_velocity

    # Consumer computes acceleration ramp endpoints
    acceleration_time = 0.5  # hypothetical motor parameter
    ramp_up_start = motion.start_position - acceleration_time * velocity / 2
    ramp_down_end = motion.end_position + acceleration_time * velocity / 2
    # Ramp extends beyond the collection region
    assert ramp_up_start < motion.start_position
    assert ramp_down_end > motion.end_position

    # Duration is available for timeout calculation
    assert w.duration > 0


def test_pmac_trajectory_positions():
    """Use case 4: PMAC — consume window.positions() in servo-rate chunks.

    Consumer calls window.positions(dt, max_duration) to get chunked
    position arrays for the trajectory scan.
    """
    spec: Acquire[str, str, Never] = Acquire(
        Product(Linspace("y", 0, 1, 2), ~Linspace("x", 0, 10, 100)),
        fly=True,
        detectors=[DetectorGroup(1, 1, 0.003, 0.001, ["det1"])],
    )
    scan = spec.compile()
    windows = list(scan)
    assert len(windows) == 2

    for i, window in enumerate(windows):
        assert "x" in window.moving_axes

        # Consume positions in chunks — emulates PMAC servo-rate loading
        all_x: list[np.ndarray] = []
        for arrays in window.positions(dt=0.01, max_duration=0.05):
            assert "x" in arrays
            all_x.append(arrays["x"])

        # Concatenate all chunks — should cover the full sweep
        full_x = np.concatenate(all_x)
        assert len(full_x) > 0

        # X direction alternates due to snake
        if i == 0:
            assert full_x[0] < full_x[-1]  # forward
        else:
            assert full_x[0] > full_x[-1]  # reverse

    # Turnaround: consumer uses boundary kinematics between windows
    prev_end = windows[0].moving_axes["x"].end_position
    curr_start = windows[1].moving_axes["x"].start_position
    # The second window starts near where the first ended
    assert prev_end == approx(curr_start, abs=0.5)


def test_pause_resume():
    """Use case 5: Pause/resume via scan.with_start().

    On pause the consumer records window_index. Resume constructs a new
    Scan that starts iteration from that point.
    """
    spec = Linspace("x", 0, 1, 10)
    scan = spec.compile()

    all_windows = list(scan)
    assert len(all_windows) == 10

    # Simulate pause at window 5
    pause_at = 5
    resumed_scan = scan.with_start(window=pause_at)
    resumed_windows = list(resumed_scan)

    # Should yield windows from index 5 onwards
    assert len(resumed_windows) == 10 - pause_at

    # The resumed windows should have the same static_axes positions as
    # the corresponding windows in the full scan (positions are deterministic)
    for rw, aw in zip(resumed_windows, all_windows[pause_at:], strict=True):
        # static_axes may differ in keys (delta vs full) but the values present
        # should match the same positions
        for axis in rw.static_axes:
            assert rw.static_axes[axis] == approx(aw.static_axes[axis])

    # Resume at the last window
    last_scan = scan.with_start(window=9)
    last_windows = list(last_scan)
    assert len(last_windows) == 1

    # Resume past the end yields nothing
    empty_scan = scan.with_start(window=10)
    assert list(empty_scan) == []


def test_analysis_reshaping():
    """Analysis use case: reshape detector data using stream dimensions.

    Consumer uses scan.windowed_streams[].dimensions to determine scan
    shape, then calls dim.setpoints() to get axis coordinates.
    """
    spec: Acquire[str, str, Never] = Acquire(
        Product(Linspace("y", 0, 5, 3), ~Linspace("x", 0, 10, 5)),
        fly=True,
        detectors=[
            DetectorGroup(1, 1, 0.003, 0.001, ["det1"]),
            DetectorGroup(10, 1, 0.0003, 8e-9, ["enc"]),
        ],
    )
    scan = spec.compile()

    stream = scan.windowed_streams[0]
    assert stream.name == "primary"

    # Base scan shape
    base_shape = [dim.length for dim in stream.dimensions]
    assert base_shape == [3, 5]

    # Per-group reshaping: collections_per_event adds an extra inner dimension
    for group in stream.detector_groups:
        if group.collections_per_event > 1:
            shape = base_shape + [group.collections_per_event]
        else:
            shape = list(base_shape)

        if group.detectors == ["det1"]:
            # exposures_per_collection=1, collections_per_event=1
            assert shape == [3, 5]
        elif group.detectors == ["enc"]:
            # exposures_per_collection=10, collections_per_event=1
            # collections_per_event is 1, so no extra dim
            assert shape == [3, 5]

    # Axis coordinates via setpoints
    y_dim = stream.dimensions[0]
    x_dim = stream.dimensions[1]
    assert y_dim.axes == ["y"]
    assert x_dim.axes == ["x"]

    y_coords = next(y_dim.setpoints("y"))
    assert len(y_coords) == 3
    assert y_coords == approx([0.0, 2.5, 5.0])

    x_coords = next(x_dim.setpoints("x"))
    assert len(x_coords) == 5
    assert x_coords == approx([0.0, 2.5, 5.0, 7.5, 10.0])

    # De-snake info is available
    assert x_dim.snake is True
    assert y_dim.snake is False
