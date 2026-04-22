"""End-to-end use-case tests for scanspec2."""

from __future__ import annotations

from typing import Never, cast

from scanspec2.core import Scan
from scanspec2.specs import Acquire, Linspace, Spiral

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
        # step scan: nominal unit duration
        assert w.duration == approx(1.0)
        # no detectors configured on bare Linspace
        assert w.trigger_groups == []
        # previous window linkage
        if i == 0:
            assert w.previous is None
        else:
            assert w.previous is windows[i - 1]


def test_linspace_fly_scan():
    """Acquire(fly=True, Linspace(x, 0, 1, 5)) — one fly window, check all fields."""
    linspace = Linspace("x", 0, 1, 5)
    scan = cast(Scan[str, Never, Never], Acquire(linspace, fly=True).compile())
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
    # duration = |4.5 - (-0.5)| = 5.0 index units
    assert w.duration == approx(5.0)
    # no detectors configured
    assert w.trigger_groups == []
    assert w.previous is None


def test_spiral_step_scan():
    """Spiral(x, 0, 5, 2, y, 10, 10) step scan — check positions and non_linear flag."""
    scan = Spiral("x", 0, 5, 2, "y", 10, 10).compile()
    windows = list(scan)
    assert len(windows) == 10

    # Spiral dimension is non-linear (uses a custom position function)
    assert scan.dimensions[0].non_linear is True

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
