"""Tests for spec.compile() — window geometry, setpoints, and iteration."""

from __future__ import annotations

from typing import Any, Never

import numpy as np
import pytest

from scanspec2.core import AxisMotion, Dimension, Scan, Window
from scanspec2.specs import (
    Acquire,
    Linspace,
    Product,
    Repeat,
    Snake,
    Static,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def dims(scan: Scan[Any, Any, Any]) -> list[Dimension[Any]]:
    return scan.dimensions


def windows(scan: Scan[Any, Any, Any]) -> list[Window[Any, Any]]:
    return list(iter(scan))


# ---------------------------------------------------------------------------
# Linspace
# ---------------------------------------------------------------------------


def test_linspace_compile_dimensions():
    sc = Linspace("x", 0.0, 10.0, 100).compile()
    assert len(sc.windowed_streams) == 0  # pure motion: no streams
    d = dims(sc)
    assert len(d) == 1
    assert d[0].axes == ["x"]
    assert d[0].length == 100
    assert d[0].snake is False
    assert sc.fly is False


def test_linspace_setpoints():
    sc = Linspace("x", 0.0, 10.0, 5).compile()
    pts = next(dims(sc)[0].setpoints("x"))
    np.testing.assert_allclose(pts, [0.0, 2.5, 5.0, 7.5, 10.0])


def test_linspace_setpoints_single_point():
    sc = Linspace("x", 3.0, 3.0, 1).compile()
    pts = next(dims(sc)[0].setpoints("x"))
    np.testing.assert_allclose(pts, [3.0])


def test_linspace_setpoints_chunked():
    sc = Linspace("x", 0.0, 9.0, 10).compile()
    chunks = list(dims(sc)[0].setpoints("x", chunk_size=3))
    assert len(chunks) == 4  # 3+3+3+1
    assert len(chunks[0]) == 3
    assert len(chunks[-1]) == 1


# ---------------------------------------------------------------------------
# Static
# ---------------------------------------------------------------------------


def test_static_compile_dimensions():
    sc = Static("y", 5.0).compile()
    d = dims(sc)
    assert len(d) == 1
    assert d[0].axes == ["y"]
    assert d[0].length == 1


def test_static_compile_num():
    sc = Static("y", 5.0, 3).compile()
    assert dims(sc)[0].length == 3


def test_static_setpoints():
    sc = Static("y", 7.0, 4).compile()
    pts = next(dims(sc)[0].setpoints("y"))
    np.testing.assert_allclose(pts, [7.0, 7.0, 7.0, 7.0])


# ---------------------------------------------------------------------------
# Snake
# ---------------------------------------------------------------------------


def test_snake_sets_inner_snake_flag():
    sc = Snake(Linspace("x", 0.0, 10.0, 10)).compile()
    assert dims(sc)[-1].snake is True


def test_snake_preserves_length():
    sc = Snake(Linspace("x", 0.0, 10.0, 10)).compile()
    assert dims(sc)[0].length == 10


def test_snake_setpoints_unchanged():
    # setpoints always return forward direction; snaking is caller's concern
    sc = Snake(Linspace("x", 0.0, 9.0, 10)).compile()
    pts = next(dims(sc)[0].setpoints("x"))
    np.testing.assert_allclose(pts, np.linspace(0, 9, 10))


def test_invert_operator_snakes():
    sc = (~Linspace("x", 0.0, 10.0, 5)).compile()
    assert dims(sc)[0].snake is True


# ---------------------------------------------------------------------------
# Product
# ---------------------------------------------------------------------------


def test_product_dimensions_order():
    sc = (Linspace("y", 0.0, 4.0, 5) * Linspace("x", 0.0, 10.0, 10)).compile()
    d = dims(sc)
    assert len(d) == 2
    assert d[0].axes == ["y"]
    assert d[0].length == 5
    assert d[1].axes == ["x"]
    assert d[1].length == 10


def test_product_mul_operator():
    outer = Linspace("y", 0.0, 4.0, 5)
    inner = Linspace("x", 0.0, 10.0, 10)
    sc = (outer * inner).compile()
    assert dims(sc)[0].axes == ["y"]
    assert dims(sc)[1].axes == ["x"]


def test_product_setpoints():
    sc = (Linspace("y", 0.0, 4.0, 3) * Linspace("x", 0.0, 10.0, 5)).compile()
    y_pts = next(dims(sc)[0].setpoints("y"))
    x_pts = next(dims(sc)[1].setpoints("x"))
    np.testing.assert_allclose(y_pts, [0.0, 2.0, 4.0])
    np.testing.assert_allclose(x_pts, [0.0, 2.5, 5.0, 7.5, 10.0])


# ---------------------------------------------------------------------------
# Zip
# ---------------------------------------------------------------------------


def test_zip_merges_innermost_dimension():
    sc = Linspace("x", 0.0, 10.0, 5).zip(Linspace("y", 0.0, 4.0, 5)).compile()
    d = dims(sc)
    assert len(d) == 1
    assert set(d[0].axes) == {"x", "y"}
    assert d[0].length == 5


def test_zip_setpoints_both_axes():
    sc = Linspace("x", 0.0, 10.0, 5).zip(Linspace("y", 0.0, 4.0, 5)).compile()
    x_pts = next(dims(sc)[0].setpoints("x"))
    y_pts = next(dims(sc)[0].setpoints("y"))
    np.testing.assert_allclose(x_pts, [0.0, 2.5, 5.0, 7.5, 10.0])
    np.testing.assert_allclose(y_pts, [0.0, 1.0, 2.0, 3.0, 4.0])


def test_zip_length_mismatch_raises():
    with pytest.raises(ValueError, match="equal inner dimension lengths"):
        Linspace("x", 0.0, 10.0, 5).zip(Linspace("y", 0.0, 4.0, 3)).compile()


# ---------------------------------------------------------------------------
# Concat
# ---------------------------------------------------------------------------


def test_concat_sums_inner_length():
    sc = Linspace("x", 0.0, 4.0, 5).concat(Linspace("x", 5.0, 9.0, 5)).compile()
    d = dims(sc)
    assert len(d) == 1
    assert d[0].length == 10


def test_concat_setpoints_combined():
    sc = Linspace("x", 0.0, 4.0, 5).concat(Linspace("x", 5.0, 9.0, 5)).compile()
    pts = next(dims(sc)[0].setpoints("x"))
    assert len(pts) == 10
    np.testing.assert_allclose(pts[:5], [0.0, 1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(pts[5:], [5.0, 6.0, 7.0, 8.0, 9.0])


def test_concat_axes_mismatch_raises():
    with pytest.raises(ValueError, match="innermost axes must match"):
        Linspace("x", 0.0, 5.0, 5).concat(Linspace("y", 0.0, 5.0, 5)).compile()


# ---------------------------------------------------------------------------
# Repeat
# ---------------------------------------------------------------------------


def test_repeat_prepends_outer_dimension():
    sc = Repeat(Linspace("x", 0.0, 10.0, 5), 3).compile()
    d = dims(sc)
    assert len(d) == 2
    assert d[0].length == 3  # repeat outer
    assert d[1].axes == ["x"]
    assert d[1].length == 5


# ---------------------------------------------------------------------------
# Acquire
# ---------------------------------------------------------------------------


def test_acquire_compile_stream_name():
    spec: Acquire[str, Never, Never] = Acquire(
        Linspace("x", 0.0, 10.0, 5), stream_name="custom"
    )
    sc = spec.compile()
    assert sc.windowed_streams[0].name == "custom"


def test_acquire_compile_fly_flag():
    spec: Acquire[str, Never, Never] = Acquire(Linspace("x", 0.0, 10.0, 5), fly=True)
    sc = spec.compile()
    assert sc.fly is True


def test_acquire_compile_continuous_streams_and_monitors():
    from scanspec2.core import ContinuousStream, DetectorGroup, MonitorStream

    spec = Acquire(
        Linspace("x", 0.0, 10.0, 5),
        continuous_streams=[
            ContinuousStream("cam", [DetectorGroup(1, 1, 0.048, 0.001, ["cam1"])])
        ],
        monitors=[MonitorStream("temp", "tc1")],
    )
    sc = spec.compile()
    assert len(sc.continuous_streams) == 1
    assert sc.continuous_streams[0].name == "cam"
    assert len(sc.monitors) == 1
    assert sc.monitors[0].detector == "tc1"


# ---------------------------------------------------------------------------
# Operator chaining
# ---------------------------------------------------------------------------


def test_product_then_snake():
    sc = (Linspace("y", 0.0, 4.0, 5) * ~Linspace("x", 0.0, 10.0, 10)).compile()
    d = dims(sc)
    assert d[0].axes == ["y"]
    assert d[0].snake is False
    assert d[1].axes == ["x"]
    assert d[1].snake is True


def test_three_level_product():
    sc = (
        Linspace("z", 0.0, 2.0, 3)
        * Linspace("y", 0.0, 4.0, 5)
        * Linspace("x", 0.0, 10.0, 10)
    ).compile()
    d = dims(sc)
    assert len(d) == 3
    assert [dim.axes for dim in d] == [["z"], ["y"], ["x"]]
    assert [dim.length for dim in d] == [3, 5, 10]


# ---------------------------------------------------------------------------
# Maximal example (from API_SPEC.md)
# ---------------------------------------------------------------------------


def test_maximal_example_dimensions():
    from scanspec2.core import ContinuousStream, DetectorGroup, MonitorStream

    energy = Linspace("energy", 7.0, 7.1, 20)
    xy = Product(Linspace("y", 0.0, 5.0, 50), ~Linspace("x", 0.0, 10.0, 100))
    full_motion = energy * xy

    spec = Acquire(
        full_motion,
        fly=True,
        stream_name="primary",
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
        monitors=[MonitorStream("dcm_temp", "dcm_temperature")],
    )
    sc = spec.compile()

    assert sc.fly is True
    assert len(sc.windowed_streams) == 1
    stream = sc.windowed_streams[0]
    assert stream.name == "primary"
    assert len(stream.dimensions) == 3
    assert stream.dimensions[0].axes == ["energy"]
    assert stream.dimensions[0].length == 20
    assert stream.dimensions[0].snake is False
    assert stream.dimensions[1].axes == ["y"]
    assert stream.dimensions[1].length == 50
    assert stream.dimensions[1].snake is False
    assert stream.dimensions[2].axes == ["x"]
    assert stream.dimensions[2].length == 100
    assert stream.dimensions[2].snake is True
    assert len(sc.continuous_streams) == 1
    assert sc.continuous_streams[0].name == "cameras"
    assert len(sc.monitors) == 1
    assert sc.monitors[0].detector == "dcm_temperature"


# ---------------------------------------------------------------------------
# Window iteration — step scan
# ---------------------------------------------------------------------------


def test_step_scan_window_count():
    sc = (Linspace("y", 0.0, 4.0, 3) * Linspace("x", 0.0, 10.0, 5)).compile()
    ws = windows(sc)
    assert len(ws) == 15


def test_step_scan_first_window_has_all_axes():
    sc = (Linspace("y", 0.0, 4.0, 3) * Linspace("x", 0.0, 10.0, 5)).compile()
    ws = windows(sc)
    assert "y" in ws[0].static_axes
    assert "x" in ws[0].static_axes
    assert ws[0].moving_axes == {}
    assert ws[0].previous is None


def test_step_scan_second_window_only_changed_axes():
    sc = (Linspace("y", 0.0, 4.0, 3) * Linspace("x", 0.0, 10.0, 5)).compile()
    ws = windows(sc)
    # Second window: only x changed (x moved from 0→2.5; y stayed at 0)
    assert "x" in ws[1].static_axes
    assert "y" not in ws[1].static_axes


def test_step_scan_new_row_includes_y():
    sc = (Linspace("y", 0.0, 4.0, 3) * Linspace("x", 0.0, 10.0, 5)).compile()
    ws = windows(sc)
    # Window 5 is first window of second y row
    assert "y" in ws[5].static_axes
    assert "x" in ws[5].static_axes


def test_step_scan_window_positions():
    sc = (Linspace("y", 0.0, 2.0, 3) * Linspace("x", 0.0, 4.0, 3)).compile()
    ws = windows(sc)
    # Window 0: y=0, x=0
    assert ws[0].static_axes["y"] == pytest.approx(0.0)  # type: ignore[reportUnknownMemberType]
    assert ws[0].static_axes["x"] == pytest.approx(0.0)  # type: ignore[reportUnknownMemberType]
    # Window 1: x=2; y unchanged
    assert ws[1].static_axes["x"] == pytest.approx(2.0)  # type: ignore[reportUnknownMemberType]
    assert "y" not in ws[1].static_axes
    # Window 4: y=1, x=0
    assert ws[3].static_axes["y"] == pytest.approx(1.0)  # type: ignore[reportUnknownMemberType]


def test_step_scan_previous_chain():
    sc = Linspace("x", 0.0, 10.0, 3).compile()
    ws = windows(sc)
    assert ws[0].previous is None
    assert ws[1].previous is ws[0]
    assert ws[2].previous is ws[1]


def test_step_scan_no_moving_axes():
    sc = (Linspace("y", 0.0, 4.0, 3) * Linspace("x", 0.0, 10.0, 5)).compile()
    for w in windows(sc):
        assert w.moving_axes == {}
        assert w.non_linear_move is False


# ---------------------------------------------------------------------------
# Window iteration — fly scan
# ---------------------------------------------------------------------------


def test_fly_scan_window_count():
    sc: Scan[str, Never, Never] = Acquire(  # type: ignore[reportUnknownVariableType]
        Linspace("y", 0.0, 4.0, 3) * ~Linspace("x", 0.0, 10.0, 5), fly=True
    ).compile()  # type: ignore[reportArgumentType]  # noqa: E501
    ws = windows(sc)
    assert len(ws) == 3  # 3 y rows, each row is one fly window


def test_fly_scan_moving_axes():
    sc: Scan[str, Never, Never] = Acquire(  # type: ignore[reportUnknownVariableType]
        Linspace("y", 0.0, 4.0, 3) * ~Linspace("x", 0.0, 10.0, 5), fly=True
    ).compile()  # type: ignore[reportArgumentType]  # noqa: E501
    ws = windows(sc)
    for w in ws:
        assert "x" in w.moving_axes
        assert "y" not in w.moving_axes
        assert "y" in w.static_axes or w.previous is None or "y" not in w.static_axes


def test_fly_scan_static_axes_delta():
    sc: Scan[str, Never, Never] = Acquire(  # type: ignore[reportUnknownVariableType]
        Linspace("y", 0.0, 4.0, 3) * Linspace("x", 0.0, 10.0, 5), fly=True
    ).compile()  # type: ignore[reportArgumentType]  # noqa: E501
    ws = windows(sc)
    # Window 0: y is new, so it should be in static_axes
    assert "y" in ws[0].static_axes
    # Window 1: y changed value → present
    assert "y" in ws[1].static_axes
    # Window 2: y changed again → present
    assert "y" in ws[2].static_axes


def test_fly_scan_forward_sweep_kinematics():
    # Linspace(x, 0, 10, 5): positions at 0, 2.5, 5, 7.5, 10
    # Boundary convention: sweep from f(-0.5) to f(4.5)
    # f(i) = 0 + i * 10/4 = 2.5*i
    # f(-0.5)= -1.25, f(4.5)=11.25
    sc: Scan[str, Never, Never] = Acquire(  # type: ignore[reportUnknownVariableType]
        Linspace("x", 0.0, 10.0, 5), fly=True
    ).compile()  # type: ignore[reportArgumentType]  # noqa: E501
    ws = windows(sc)
    assert len(ws) == 1
    am: AxisMotion = ws[0].moving_axes["x"]
    assert am.start_position == pytest.approx(-1.25)  # type: ignore[reportUnknownMemberType]
    assert am.end_position == pytest.approx(11.25)  # type: ignore[reportUnknownMemberType]
    assert am.start_velocity == pytest.approx(2.5)  # type: ignore[reportUnknownMemberType]
    assert am.end_velocity == pytest.approx(2.5)  # type: ignore[reportUnknownMemberType]


def test_fly_scan_snake_reverses_direction():
    sc: Scan[str, Never, Never] = Acquire(  # type: ignore[reportUnknownVariableType]
        Linspace("y", 0.0, 2.0, 2) * ~Linspace("x", 0.0, 10.0, 5), fly=True
    ).compile()  # type: ignore[reportArgumentType]  # noqa: E501
    ws = windows(sc)
    assert len(ws) == 2
    am0 = ws[0].moving_axes["x"]
    am1 = ws[1].moving_axes["x"]
    # Forward then backward
    assert am0.start_position < am0.end_position
    assert am1.start_position > am1.end_position


def test_fly_scan_previous_link():
    sc: Scan[str, Never, Never] = Acquire(  # type: ignore[reportUnknownVariableType]
        Linspace("y", 0.0, 4.0, 3) * Linspace("x", 0.0, 10.0, 5), fly=True
    ).compile()  # type: ignore[reportArgumentType]  # noqa: E501
    ws = windows(sc)
    assert ws[0].previous is None
    assert ws[1].previous is ws[0]
    assert ws[2].previous is ws[1]


# ---------------------------------------------------------------------------
# with_start — resume from a given window
# ---------------------------------------------------------------------------


def test_with_start_skips_windows():
    sc = Linspace("x", 0.0, 10.0, 5).compile()
    sc2 = sc.with_start(window=2)
    ws = windows(sc2)
    assert len(ws) == 3  # skipped first 2


def test_with_start_zero_is_full_scan():
    sc = Linspace("x", 0.0, 10.0, 5).compile()
    sc2 = sc.with_start(window=0)
    assert len(windows(sc2)) == 5


def test_with_start_does_not_mutate_original():
    sc = Linspace("x", 0.0, 10.0, 5).compile()
    sc2 = sc.with_start(window=3)
    assert len(windows(sc)) == 5
    assert len(windows(sc2)) == 2
