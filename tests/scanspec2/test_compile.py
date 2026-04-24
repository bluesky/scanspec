"""Tests for spec.compile() — window geometry, setpoints, and iteration."""

from __future__ import annotations

from typing import Any, Never

import numpy as np
import pytest

from scanspec2.core import (
    AxisMotion,
    ConcatSource,
    DetectorGroup,
    Scan,
    TriggerPattern,
    Window,
    WindowGenerator,
)
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


def gens(scan: Scan[Any, Any, Any]) -> list[WindowGenerator[Any]]:
    return scan.generators


def windows(scan: Scan[Any, Any, Any]) -> list[Window[Any, Any]]:
    return list(iter(scan))


# ---------------------------------------------------------------------------
# Linspace
# ---------------------------------------------------------------------------


def test_linspace_compile_dimensions():
    sc = Linspace("x", 0.0, 10.0, 100).compile()
    assert len(sc.windowed_streams) == 0  # pure motion: no streams
    g = gens(sc)
    assert len(g) == 1
    assert g[0].axes == ["x"]
    assert g[0].length == 100
    assert g[0].snake is False
    assert g[0].fly is False


def test_linspace_setpoints():
    sc = Linspace("x", 0.0, 10.0, 5).compile()
    pts = gens(sc)[0].setpoints(np.array([0.5, 1.5, 2.5, 3.5, 4.5]))["x"]
    np.testing.assert_allclose(pts, [0.0, 2.5, 5.0, 7.5, 10.0])


def test_linspace_setpoints_single_point():
    sc = Linspace("x", 3.0, 3.0, 1).compile()
    pts = gens(sc)[0].setpoints(np.array([0.5]))["x"]
    np.testing.assert_allclose(pts, [3.0])


def test_linspace_setpoints_chunked():
    sc = Linspace("x", 0.0, 9.0, 10).compile()
    g = gens(sc)[0]
    # Test generator produces correct positions for 3-point chunks
    pts0 = g.setpoints(np.array([0.5, 1.5, 2.5]))["x"]
    pts3 = g.setpoints(np.array([9.5]))["x"]
    np.testing.assert_allclose(pts0, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(pts3, [9.0])


# ---------------------------------------------------------------------------
# Static
# ---------------------------------------------------------------------------


def test_static_compile_dimensions():
    sc = Static("y", 5.0).compile()
    g = gens(sc)
    assert len(g) == 1
    assert g[0].axes == ["y"]
    assert g[0].length == 1


def test_static_compile_num():
    sc = Static("y", 5.0, 3).compile()
    assert gens(sc)[0].length == 3


def test_static_setpoints():
    sc = Static("y", 7.0, 4).compile()
    pts = gens(sc)[0].setpoints(np.array([0.5, 1.5, 2.5, 3.5]))["y"]
    np.testing.assert_allclose(pts, [7.0, 7.0, 7.0, 7.0])


# ---------------------------------------------------------------------------
# Snake
# ---------------------------------------------------------------------------


def test_snake_sets_inner_snake_flag():
    sc = Snake(Linspace("x", 0.0, 10.0, 10)).compile()
    assert gens(sc)[-1].snake is True


def test_snake_preserves_length():
    sc = Snake(Linspace("x", 0.0, 10.0, 10)).compile()
    assert gens(sc)[0].length == 10


def test_snake_setpoints_unchanged():
    # setpoints always return forward direction; snaking is caller's concern
    sc = Snake(Linspace("x", 0.0, 9.0, 10)).compile()
    pts = gens(sc)[0].setpoints(np.arange(10) + 0.5)["x"]
    np.testing.assert_allclose(pts, np.linspace(0, 9, 10))


def test_invert_operator_snakes():
    sc = (~Linspace("x", 0.0, 10.0, 5)).compile()
    assert gens(sc)[0].snake is True


def test_snake_empty_generators_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Snake on a Scan with no generators is an error."""
    spec = ~Linspace("x", 0, 1, 1)

    def _empty_compile(self: Any) -> Scan[str, Never, Never]:
        return Scan(generators=[])

    monkeypatch.setattr(Linspace, "compile", _empty_compile)
    with pytest.raises(ValueError, match="exactly one generator"):
        spec.compile()


def test_snake_multiple_generators_raises():
    """Snake on a spec with multiple generators is an error."""
    # Product of two Linspaces produces 2 generators.  Snake requires 1.
    spec = ~(Linspace("y", 0, 1, 3) * Linspace("x", 0, 1, 5))
    with pytest.raises(ValueError, match="exactly one generator"):
        spec.compile()


def test_snake_with_concat_children():
    """Snake on a Concat'd Acquire sets snake on the concat generator (children).

    The old Snake.compile had an if/else branch because reconstructing a
    WindowGenerator required different args for children vs non-children
    generators.  With mutation (just set .snake = True), no branch is needed.
    """
    dg: DetectorGroup[str] = DetectorGroup(1, 1, 0.001, 0.001, ["det"])
    acq1: Acquire[str, str, Never] = Acquire(Linspace("x", 0, 5, 3), detectors=[dg])
    acq2: Acquire[str, str, Never] = Acquire(Linspace("x", 10, 15, 2), detectors=[dg])
    spec = ~acq1.concat(acq2)
    sc = spec.compile()
    g = gens(sc)
    assert len(g) == 1
    assert g[0].snake is True
    assert isinstance(g[0].source, ConcatSource)
    assert len(g[0].source.children) == 2
    # Verify iteration still works — snake reverses on odd outer iterations
    ws = windows(sc)
    assert len(ws) == 5  # 3 + 2 points


# ---------------------------------------------------------------------------
# Product
# ---------------------------------------------------------------------------


def test_product_dimensions_order():
    sc = (Linspace("y", 0.0, 4.0, 5) * Linspace("x", 0.0, 10.0, 10)).compile()
    g = gens(sc)
    assert len(g) == 2
    assert g[0].axes == ["y"]
    assert g[0].length == 5
    assert g[1].axes == ["x"]
    assert g[1].length == 10


def test_product_mul_operator():
    outer = Linspace("y", 0.0, 4.0, 5)
    inner = Linspace("x", 0.0, 10.0, 10)
    sc = (outer * inner).compile()
    assert gens(sc)[0].axes == ["y"]
    assert gens(sc)[1].axes == ["x"]


def test_product_setpoints():
    sc = (Linspace("y", 0.0, 4.0, 3) * Linspace("x", 0.0, 10.0, 5)).compile()
    g = gens(sc)
    y_pts = g[0].setpoints(np.array([0.5, 1.5, 2.5]))["y"]
    x_pts = g[1].setpoints(np.array([0.5, 1.5, 2.5, 3.5, 4.5]))["x"]
    np.testing.assert_allclose(y_pts, [0.0, 2.0, 4.0])
    np.testing.assert_allclose(x_pts, [0.0, 2.5, 5.0, 7.5, 10.0])


# ---------------------------------------------------------------------------
# Zip
# ---------------------------------------------------------------------------


def test_zip_merges_innermost_dimension():
    sc = Linspace("x", 0.0, 10.0, 5).zip(Linspace("y", 0.0, 4.0, 5)).compile()
    g = gens(sc)
    assert len(g) == 1
    assert set(g[0].axes) == {"x", "y"}
    assert g[0].length == 5


def test_zip_setpoints_both_axes():
    sc = Linspace("x", 0.0, 10.0, 5).zip(Linspace("y", 0.0, 4.0, 5)).compile()
    g = gens(sc)[0]
    idx = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    positions = g.setpoints(idx)
    np.testing.assert_allclose(positions["x"], [0.0, 2.5, 5.0, 7.5, 10.0])
    np.testing.assert_allclose(positions["y"], [0.0, 1.0, 2.0, 3.0, 4.0])


def test_zip_length_mismatch_raises():
    with pytest.raises(ValueError, match="equal dimension lengths"):
        Linspace("x", 0.0, 10.0, 5).zip(Linspace("y", 0.0, 4.0, 3)).compile()


def test_zip_static_expands_to_match():
    """Zip with Static(num=1) expands to match left's innermost length."""
    sc = Linspace("y", 0.0, 4.0, 5).zip(Static("x", 3.0)).compile()
    g = gens(sc)
    assert len(g) == 1
    assert set(g[0].axes) == {"y", "x"}
    assert g[0].length == 5
    pts = g[0].setpoints(np.arange(5) + 0.5)
    np.testing.assert_allclose(pts["y"], [0.0, 1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(pts["x"], [3.0, 3.0, 3.0, 3.0, 3.0])


def test_zip_product_left_single_right():
    """z(3) * y(5).zip(x(5)) — z outer, y+x merged inner."""
    sc = (
        Linspace("z", 0.0, 2.0, 3)
        * Linspace("y", 0.0, 4.0, 5).zip(Linspace("x", 0.0, 8.0, 5))
    ).compile()
    g = gens(sc)
    assert len(g) == 2
    assert g[0].axes == ["z"]
    assert g[0].length == 3
    assert set(g[1].axes) == {"y", "x"}
    assert g[1].length == 5


# ---------------------------------------------------------------------------
# Concat
# ---------------------------------------------------------------------------


def test_concat_sums_inner_length():
    sc = Linspace("x", 0.0, 4.0, 5).concat(Linspace("x", 5.0, 9.0, 5)).compile()
    g = gens(sc)
    assert len(g) == 1
    assert g[0].length == 10


def test_concat_setpoints_combined():
    sc = Linspace("x", 0.0, 4.0, 5).concat(Linspace("x", 5.0, 9.0, 5)).compile()
    g = gens(sc)[0]
    pts = g.setpoints(np.arange(10) + 0.5)["x"]
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
    g = gens(sc)
    assert len(g) == 2
    assert g[0].length == 3  # repeat outer
    assert g[1].axes == ["x"]
    assert g[1].length == 5


# ---------------------------------------------------------------------------
# Acquire
# ---------------------------------------------------------------------------


def test_acquire_compile_stream_name():
    det = DetectorGroup(1, 1, 0.01, 0.001, ["det1"])
    spec: Acquire[str, str, Never] = Acquire(
        Linspace("x", 0.0, 10.0, 5), stream_name="custom", detectors=[det]
    )
    sc = spec.compile()
    assert sc.windowed_streams[0].name == "custom"


def test_acquire_compile_fly_flag():
    spec: Acquire[str, Never, Never] = Acquire(Linspace("x", 0.0, 10.0, 5), fly=True)
    sc = spec.compile()
    assert sc.generators[-1].fly is True


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
    g = gens(sc)
    assert g[0].axes == ["y"]
    assert g[0].snake is False
    assert g[1].axes == ["x"]
    assert g[1].snake is True


def test_three_level_product():
    sc = (
        Linspace("z", 0.0, 2.0, 3)
        * Linspace("y", 0.0, 4.0, 5)
        * Linspace("x", 0.0, 10.0, 10)
    ).compile()
    g = gens(sc)
    assert len(g) == 3
    assert [gen.axes for gen in g] == [["z"], ["y"], ["x"]]
    assert [gen.length for gen in g] == [3, 5, 10]


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

    assert sc.generators[-1].fly is True
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
        assert w.non_linear is False


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


# ---------------------------------------------------------------------------
# Snake nesting — ported from 1.x test_specs.test_product_snaking_linspaces
# and test_specs.test_xyz_stack
# ---------------------------------------------------------------------------


def _all_positions(
    scan: Scan[str, Any, Any],
) -> list[dict[str, float]]:
    """Accumulate setpoint positions across windows (step scan)."""
    pos: dict[str, float] = {}
    result: list[dict[str, float]] = []
    for w in scan:
        pos = dict(pos)
        pos.update(w.static_axes)
        result.append(pos)
    return result


def test_snake_step_y_snakex():
    """Port of 1.x test_product_snaking_linspaces: y(3) * ~x(2)."""
    sc = (Linspace("y", 1.0, 2.0, 3) * ~Linspace("x", 0.0, 1.0, 2)).compile()
    pts = _all_positions(sc)
    assert len(pts) == 6
    assert [p["x"] for p in pts] == pytest.approx([0, 1, 1, 0, 0, 1])  # type: ignore[reportUnknownMemberType]
    assert [p["y"] for p in pts] == pytest.approx([1, 1, 1.5, 1.5, 2, 2])  # type: ignore[reportUnknownMemberType]


def test_snake_step_xyz():
    """Port of 1.x test_xyz_stack: z(2) * ~y(3) * ~x(4)."""
    sc = (
        Linspace("z", 0.0, 1.0, 2)
        * ~Linspace("y", 0.0, 2.0, 3)
        * ~Linspace("x", 0.0, 3.0, 4)
    ).compile()
    pts = _all_positions(sc)
    assert len(pts) == 24
    assert [p["x"] for p in pts] == pytest.approx(  # type: ignore[reportUnknownMemberType]
        [0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0]
    )
    assert [p["y"] for p in pts] == pytest.approx(  # type: ignore[reportUnknownMemberType]
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0]
    )
    assert [p["z"] for p in pts] == pytest.approx([0] * 12 + [1] * 12)  # type: ignore[reportUnknownMemberType]


def test_snake_fly_y_snakex():
    """Fly scan: y(3) * ~x(4) — 3 windows, x alternates direction."""
    spec: Acquire[str, Never, Never] = Acquire(
        Linspace("y", 0.0, 2.0, 3) * ~Linspace("x", 0.0, 3.0, 4),
        fly=True,
    )
    sc = spec.compile()
    ws = windows(sc)
    assert len(ws) == 3
    # x forward in window 0, reversed in window 1, forward in window 2
    assert ws[0].moving_axes["x"].start_position < ws[0].moving_axes["x"].end_position
    assert ws[1].moving_axes["x"].start_position > ws[1].moving_axes["x"].end_position
    assert ws[2].moving_axes["x"].start_position < ws[2].moving_axes["x"].end_position


def test_snake_fly_xyz():
    """Fly scan: z(2) * ~y(3) * ~x(4) — 6 windows, both y and x snake."""
    spec: Acquire[str, Never, Never] = Acquire(
        Linspace("z", 0.0, 1.0, 2)
        * ~Linspace("y", 0.0, 2.0, 3)
        * ~Linspace("x", 0.0, 3.0, 4),
        fly=True,
    )
    sc = spec.compile()
    ws = windows(sc)
    assert len(ws) == 6

    # Collect y positions (from static_axes) and x directions.
    y_vals: list[float] = []
    x_fwd: list[bool] = []
    prev: dict[str, float] = {}
    for w in ws:
        prev.update(w.static_axes)
        y_vals.append(prev["y"])
        x_fwd.append(
            w.moving_axes["x"].start_position < w.moving_axes["x"].end_position
        )

    # y snakes: 0,1,2 then 2,1,0
    assert y_vals == pytest.approx([0, 1, 2, 2, 1, 0])  # type: ignore[reportUnknownMemberType]
    # x alternates direction each window
    assert x_fwd == [True, False, True, False, True, False]


# ---------------------------------------------------------------------------
# Phase B — trigger_groups
# ---------------------------------------------------------------------------


def test_step_scan_trigger_groups():
    det = DetectorGroup(1, 1, 0.01, 0.001, ["det1"])
    sc: Scan[str, str, Never] = Acquire(  # type: ignore[reportUnknownVariableType]
        Linspace("x", 0.0, 10.0, 5), detectors=[det]
    ).compile()  # type: ignore[reportArgumentType]  # noqa: E501
    ws = windows(sc)
    assert len(ws) == 5
    for w in ws:
        assert len(w.trigger_groups) == 1
        tg = w.trigger_groups[0]
        assert tg.detectors == ["det1"]
        assert tg.trigger_patterns == [TriggerPattern(1, 0.01, 0.001)]


def test_fly_scan_trigger_groups():
    det = DetectorGroup(1, 1, 0.003, 0.001, ["det1"])
    sc: Scan[str, str, Never] = Acquire(  # type: ignore[reportUnknownVariableType]
        Linspace("x", 0.0, 10.0, 5), fly=True, detectors=[det]
    ).compile()  # type: ignore[reportArgumentType]  # noqa: E501
    ws = windows(sc)
    assert len(ws) == 1
    tg = ws[0].trigger_groups[0]
    assert tg.detectors == ["det1"]
    # fly: repeats = length * exposures_per_collection = 5 * 1
    assert tg.trigger_patterns == [TriggerPattern(5, 0.003, 0.001)]


def test_multirate_trigger_groups():
    det1 = DetectorGroup(1, 1, 0.003, 0.001, ["saxs"])
    det2 = DetectorGroup(10, 1, 0.0003, 8e-9, ["encoder"])
    sc: Scan[str, str, Never] = Acquire(  # type: ignore[reportUnknownVariableType]
        Linspace("x", 0.0, 10.0, 100),
        fly=True,
        detectors=[det1, det2],
    ).compile()  # type: ignore[reportArgumentType]  # noqa: E501
    ws = windows(sc)
    tgs = ws[0].trigger_groups
    assert len(tgs) == 2
    assert tgs[0].trigger_patterns == [TriggerPattern(100, 0.003, 0.001)]
    assert tgs[1].trigger_patterns == [TriggerPattern(1000, 0.0003, 8e-9)]


def test_duration_derived_from_detectors():
    det = DetectorGroup(1, 1, 0.01, 0.001, ["det1"])
    sc: Scan[str, str, Never] = Acquire(  # type: ignore[reportUnknownVariableType]
        Linspace("x", 0.0, 10.0, 5), detectors=[det]
    ).compile()  # type: ignore[reportArgumentType]  # noqa: E501
    ws = windows(sc)
    # step: 1 × (0.01 + 0.001) = 0.011
    for w in ws:
        assert w.duration == pytest.approx(0.011)  # type: ignore[reportUnknownMemberType]


def test_duration_derived_from_fly_detectors():
    det = DetectorGroup(1, 1, 0.003, 0.001, ["det1"])
    sc: Scan[str, str, Never] = Acquire(  # type: ignore[reportUnknownVariableType]
        Linspace("x", 0.0, 10.0, 5), fly=True, detectors=[det]
    ).compile()  # type: ignore[reportArgumentType]  # noqa: E501
    ws = windows(sc)
    # fly: 5 × (0.003 + 0.001) = 0.02
    assert ws[0].duration == pytest.approx(0.02)  # type: ignore[reportUnknownMemberType]


def test_explicit_duration_must_be_ge_derived():
    det = DetectorGroup(1, 1, 0.01, 0.001, ["det1"])
    with pytest.raises(ValueError, match="less than"):
        Acquire(
            Linspace("x", 0.0, 10.0, 5),
            detectors=[det],
            duration=0.005,  # too small
        ).compile()


# ---------------------------------------------------------------------------
# Scan capability properties
# ---------------------------------------------------------------------------


def test_scan_has_moving_axes_fly():
    sc: Scan[str, Never, Never] = Acquire(  # type: ignore[reportUnknownVariableType]
        Linspace("x", 0, 10, 5), fly=True
    ).compile()  # type: ignore[reportArgumentType]  # noqa: E501
    assert sc.has_moving_axes is True
    assert sc.non_linear is False


def test_scan_has_moving_axes_step():
    sc = Linspace("x", 0, 10, 5).compile()
    assert sc.has_moving_axes is False
    assert sc.non_linear is False


def test_scan_non_linear_spiral():
    from scanspec2.specs import Spiral

    sc: Scan[str, Never, Never] = Acquire(  # type: ignore[reportUnknownVariableType]
        Spiral("x", 0, 5, 2, "y", 10, 10), fly=True
    ).compile()  # type: ignore[reportArgumentType]  # noqa: E501
    assert sc.has_moving_axes is True
    assert sc.non_linear is True


# ---------------------------------------------------------------------------
# AnySpec: out-of-package subclass included in union
# ---------------------------------------------------------------------------


def test_anyspec_extension():
    """Spec subclass defined outside specs.py is included in the AnySpec union."""
    from typing import Never as Nv

    from pydantic import TypeAdapter

    from scanspec2.core import LinearSource as LinSrc
    from scanspec2.core import Scan as Sc
    from scanspec2.core import WindowGenerator as WinGen
    from scanspec2.specs import AnySpec, Spec

    class CustomLinspace(Spec[str, Nv, Nv]):
        axis: str
        num: int = 3

        def compile(self) -> Sc[str, Nv, Nv]:
            gen = WinGen(
                axes=[self.axis],
                length=self.num,
                source=LinSrc({self.axis: (0, 1)}, self.num),
            )
            return Sc(generators=[gen])

    # The custom class can be serialised and deserialised via AnySpec.
    original = CustomLinspace(axis="q", num=7)
    ta = TypeAdapter(AnySpec)
    serialized = ta.dump_python(original)
    assert serialized["type"] == "CustomLinspace"
    restored = ta.validate_python(serialized)
    assert isinstance(restored, CustomLinspace)
    assert restored.axis == "q"
    assert restored.num == 7

    # And it can compile.
    sc = original.compile()
    assert len(sc.generators) == 1
    assert sc.generators[0].length == 7


def test_explicit_duration_overrides_when_larger():
    det = DetectorGroup(1, 1, 0.01, 0.001, ["det1"])
    sc: Scan[str, str, Never] = Acquire(  # type: ignore[reportUnknownVariableType]
        Linspace("x", 0.0, 10.0, 5),
        detectors=[det],
        duration=0.05,  # larger than derived 0.011
    ).compile()  # type: ignore[reportArgumentType]  # noqa: E501
    ws = windows(sc)
    for w in ws:
        assert w.duration == pytest.approx(0.05)  # type: ignore[reportUnknownMemberType]


def test_none_livetime_raises():
    det = DetectorGroup(1, 1, None, 0.001, ["det1"])
    with pytest.raises(ValueError, match="livetime"):
        Acquire(Linspace("x", 0.0, 10.0, 5), detectors=[det]).compile()


def test_none_deadtime_raises():
    det = DetectorGroup(1, 1, 0.01, None, ["det1"])
    with pytest.raises(ValueError, match="deadtime"):
        Acquire(Linspace("x", 0.0, 10.0, 5), detectors=[det]).compile()


# ---------------------------------------------------------------------------
# Phase B — multi-stream Concat
# ---------------------------------------------------------------------------


def test_concat_fly_step_windows():
    """Concat of step + fly Acquires: 2 windows (1 step + 1 fly)."""
    det = DetectorGroup(1, 1, 0.01, 0.001, ["det1"])
    step_acq: Acquire[str, str, Never] = Acquire(
        Static("x", 5.0), detectors=[det], stream_name="s1"
    )
    fly_acq: Acquire[str, str, Never] = Acquire(
        Linspace("x", 0.0, 10.0, 5),
        fly=True,
        detectors=[det],
        stream_name="s2",
    )
    sc = step_acq.concat(fly_acq).compile()
    ws = windows(sc)
    assert len(ws) == 2
    # First: step window
    assert ws[0].moving_axes == {}
    assert ws[0].static_axes["x"] == pytest.approx(5.0)  # type: ignore[reportUnknownMemberType]
    # Second: fly window
    assert "x" in ws[1].moving_axes


def test_concat_different_streams():
    """Two Acquires with different stream names."""
    det1 = DetectorGroup(1, 1, 0.01, 0.001, ["det1"])
    det2 = DetectorGroup(1, 1, 0.003, 0.001, ["det2"])
    a1: Acquire[str, str, Never] = Acquire(
        Static("x", 5.0), detectors=[det1], stream_name="diff"
    )
    a2: Acquire[str, str, Never] = Acquire(
        Linspace("x", 0.0, 10.0, 100),
        fly=True,
        detectors=[det2],
        stream_name="spec",
    )
    sc = a1.concat(a2).compile()
    names = {s.name for s in sc.windowed_streams}
    assert names == {"diff", "spec"}


def test_concat_same_stream_sums_inner():
    """Two Acquires with same stream name → inner length summed."""
    det = DetectorGroup(1, 1, 0.003, 0.001, ["det1"])
    a1: Acquire[str, str, Never] = Acquire(
        Linspace("x", 0.0, 5.0, 500),
        fly=True,
        detectors=[det],
        stream_name="primary",
    )
    a2: Acquire[str, str, Never] = Acquire(
        Linspace("x", 5.0, 0.0, 500),
        fly=True,
        detectors=[det],
        stream_name="primary",
    )
    sc = a1.concat(a2).compile()
    assert len(sc.windowed_streams) == 1
    stream = sc.windowed_streams[0]
    assert stream.name == "primary"
    assert stream.dimensions[-1].length == 1000


def test_repeat_concat_windows():
    """Repeat wrapping a concat: n_repeat x groups_per_concat."""
    det = DetectorGroup(1, 1, 0.01, 0.001, ["det1"])
    a1: Acquire[str, str, Never] = Acquire(
        Static("x", 5.0), detectors=[det], stream_name="s1"
    )
    a2: Acquire[str, str, Never] = Acquire(
        Linspace("x", 0.0, 10.0, 5),
        fly=True,
        detectors=[det],
        stream_name="s2",
    )
    sc = Repeat(a1.concat(a2), num=3).compile()
    ws = windows(sc)
    # 3 repeats × 2 groups = 6 windows
    assert len(ws) == 6
    # Pattern: step, fly, step, fly, step, fly
    for i in range(3):
        assert ws[i * 2].moving_axes == {}
        assert "x" in ws[i * 2 + 1].moving_axes


def test_repeat_concat_streams_have_outer_dim():
    """Repeat wrapping concat: windowed_streams get outer repeat dim."""
    det = DetectorGroup(1, 1, 0.01, 0.001, ["det1"])
    a1: Acquire[str, str, Never] = Acquire(
        Static("x", 5.0), detectors=[det], stream_name="s1"
    )
    a2: Acquire[str, str, Never] = Acquire(
        Linspace("x", 0.0, 10.0, 5),
        fly=True,
        detectors=[det],
        stream_name="s2",
    )
    sc = Repeat(a1.concat(a2), num=10).compile()
    for stream in sc.windowed_streams:
        # outermost dim is the repeat
        assert stream.dimensions[0].length == 10


def test_concat_previous_chain():
    """Previous chain across all windows from grouped concat."""
    det = DetectorGroup(1, 1, 0.01, 0.001, ["det1"])
    a1: Acquire[str, str, Never] = Acquire(
        Static("x", 5.0), detectors=[det], stream_name="s1"
    )
    a2: Acquire[str, str, Never] = Acquire(
        Static("x", 10.0), detectors=[det], stream_name="s2"
    )
    sc = Repeat(a1.concat(a2), num=3).compile()
    ws = windows(sc)
    assert ws[0].previous is None
    for i in range(1, len(ws)):
        assert ws[i].previous is ws[i - 1]


# ---------------------------------------------------------------------------
# Combinators reject continuous_streams and monitors
# ---------------------------------------------------------------------------


def test_concat_rejects_continuous_streams():
    from scanspec2.core import ContinuousStream

    a: Acquire[str, str, Never] = Acquire(
        Linspace("x", 0, 1, 5),
        continuous_streams=[
            ContinuousStream("cam", [DetectorGroup(1, 1, 0.01, 0.001, ["c1"])])
        ],
    )
    with pytest.raises(ValueError, match="Concat does not accept.*continuous"):
        a.concat(Acquire(Linspace("x", 1, 2, 5))).compile()


def test_concat_rejects_monitors():
    from scanspec2.core import MonitorStream

    a: Acquire[str, Never, str] = Acquire(
        Linspace("x", 0, 1, 5),
        monitors=[MonitorStream("temp", "tc1")],
    )
    with pytest.raises(ValueError, match="Concat does not accept.*monitors"):
        a.concat(Acquire(Linspace("x", 1, 2, 5))).compile()


def test_product_rejects_continuous_streams():
    from scanspec2.core import ContinuousStream

    a: Acquire[str, str, Never] = Acquire(
        Linspace("x", 0, 1, 5),
        continuous_streams=[
            ContinuousStream("cam", [DetectorGroup(1, 1, 0.01, 0.001, ["c1"])])
        ],
    )
    with pytest.raises(ValueError, match="Product does not accept.*continuous"):
        (Linspace("y", 0, 1, 3) * a).compile()  # type: ignore[reportOperatorIssue]


def test_zip_rejects_monitors():
    from scanspec2.core import MonitorStream

    a: Acquire[str, Never, str] = Acquire(
        Linspace("x", 0, 1, 5),
        monitors=[MonitorStream("temp", "tc1")],
    )
    with pytest.raises(ValueError, match="Zip does not accept.*monitors"):
        Linspace("y", 0, 1, 5).zip(a).compile()  # type: ignore[reportArgumentType]
