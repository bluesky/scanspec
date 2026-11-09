"""Tests for spec.compile() — window geometry, setpoints, and iteration."""

from __future__ import annotations

from typing import Any, Never

import numpy as np
import pytest

from scanspec.v2.core import (
    AxisMotion,
    DetectorGroup,
    Scan,
    TriggerFollower,
    TriggerGroup,
    TriggerRepeat,
    Window,
    WindowGenerator,
)
from scanspec.v2.specs import (
    Concat,
    Linspace,
    Product,
    Repeat,
    Snake,
    Spec,
    Spiral,
    Static,
    Sync,
    Zip,
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
    with pytest.raises(ValueError, match="must move the same axes"):
        Linspace("x", 0.0, 5.0, 5).concat(Linspace("y", 0.0, 5.0, 5)).compile()


@pytest.mark.parametrize("nest", ["left", "right"])
def test_chained_concat_of_motion(nest: str):
    a, b, c = (Linspace("x", lo, lo + 1.0, 2) for lo in (0.0, 1.0, 2.0))
    spec = Concat(Concat(a, b), c) if nest == "left" else Concat(a, Concat(b, c))
    xs = [p["x"] for p in _all_positions(spec.compile())]
    assert xs == pytest.approx([0, 1, 1, 2, 2, 3])  # type: ignore[reportUnknownMemberType]


def test_chained_concat_axes_mismatch_raises():
    xs = Linspace("x", 0.0, 1.0, 2).concat(Linspace("x", 1.0, 2.0, 2))
    with pytest.raises(ValueError, match="must move the same axes"):
        xs.concat(Linspace("y", 0.0, 1.0, 2)).compile()


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
# Sync
# ---------------------------------------------------------------------------


def test_sync_compile_stream_name():
    tg = TriggerGroup(
        detectors=frozenset({"det1"}),
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=0.01,
        deadtime=0.001,
    )
    spec: Sync[str, str, Never] = Sync(
        Linspace("x", 0.0, 10.0, 5), stream_name="custom", trigger_group=tg
    )
    sc = spec.compile()
    assert sc.windowed_streams[0].name == "custom"


def test_sync_compile_fly_flag():
    spec: Sync[str, Never, Never] = Sync(Linspace("x", 0.0, 10.0, 5), fly=True)
    sc = spec.compile()
    assert sc.generators[-1].fly is True


def test_sync_compile_continuous_streams_and_monitors():
    from scanspec.v2.core import ContinuousStream, DetectorGroup, MonitorStream
    from scanspec.v2.specs import ContinuousStreams, Monitors

    sync: Sync[str, str, str] = Sync(Linspace("x", 0.0, 10.0, 5))
    spec = ContinuousStreams(
        Monitors(
            sync,
            monitors=[MonitorStream("temp", "tc1")],
        ),
        continuous_streams=[
            ContinuousStream("cam", [DetectorGroup(1, 1, 0.048, 0.001, ["cam1"])])
        ],
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
    from scanspec.v2.core import ContinuousStream, DetectorGroup, MonitorStream
    from scanspec.v2.specs import ContinuousStreams, Monitors

    energy = Linspace("energy", 7.0, 7.1, 20)
    xy = Product(Linspace("y", 0.0, 5.0, 50), ~Linspace("x", 0.0, 10.0, 100))
    full_motion = energy * xy

    sync: Sync[str, str, str] = Sync(
        full_motion,
        fly=True,
        stream_name="primary",
        # inner_length=100, the innermost `x` dimension, drives the derived
        # group repeats=100.
        trigger_group=TriggerGroup(
            detectors=frozenset({"saxs", "waxs"}),
            exposures_per_collection=1,
            collections_per_event=1,
            livetime=0.003,
            deadtime=0.001,
            followers=[
                # See test_maximal_fly_step (test_use_cases.py) for how this
                # livetime is derived.
                TriggerFollower(
                    detectors=frozenset({"timestamp", "x_enc", "y_enc"}),
                    exposures_per_collection=10,
                    collections_per_event=1,
                    livetime=0.000299992,
                    deadtime=8e-9,
                    repeats=10,
                ),
            ],
        ),
    )
    spec = ContinuousStreams(
        Monitors(
            sync,
            monitors=[MonitorStream("dcm_temp", "dcm_temperature")],
        ),
        continuous_streams=[
            ContinuousStream(
                "cameras",
                [DetectorGroup(1, 1, 0.048, 0.001, ["front_cam", "side_cam"])],
            ),
        ],
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
    sc: Scan[str, Never, Never] = Sync(  # type: ignore[reportUnknownVariableType]
        Linspace("y", 0.0, 4.0, 3) * ~Linspace("x", 0.0, 10.0, 5), fly=True
    ).compile()  # type: ignore[reportArgumentType]  # noqa: E501
    ws = windows(sc)
    assert len(ws) == 3  # 3 y rows, each row is one fly window


def test_fly_scan_moving_axes():
    sc: Scan[str, Never, Never] = Sync(  # type: ignore[reportUnknownVariableType]
        Linspace("y", 0.0, 4.0, 3) * ~Linspace("x", 0.0, 10.0, 5), fly=True
    ).compile()  # type: ignore[reportArgumentType]  # noqa: E501
    ws = windows(sc)
    for w in ws:
        assert "x" in w.moving_axes
        assert "y" not in w.moving_axes
        assert "y" in w.static_axes or w.previous is None or "y" not in w.static_axes


def test_fly_scan_static_axes_delta():
    sc: Scan[str, Never, Never] = Sync(  # type: ignore[reportUnknownVariableType]
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
    sc: Scan[str, Never, Never] = Sync(  # type: ignore[reportUnknownVariableType]
        Linspace("x", 0.0, 10.0, 5), fly=True
    ).compile()  # type: ignore[reportArgumentType]  # noqa: E501
    ws = windows(sc)
    assert len(ws) == 1
    am: AxisMotion = ws[0].moving_axes["x"]
    assert am.start_position == pytest.approx(-1.25)  # type: ignore[reportUnknownMemberType]
    assert am.end_position == pytest.approx(11.25)  # type: ignore[reportUnknownMemberType]
    assert am.start_velocity == pytest.approx(2.5)  # type: ignore[reportUnknownMemberType]
    assert am.end_velocity == pytest.approx(2.5)  # type: ignore[reportUnknownMemberType]


def test_fly_scan_velocity_uses_real_seconds_not_index_units():
    """AxisMotion velocities must be reported in real position-units-per-
    second. A detector-derived per-point duration (0.004s, not the masking
    default of 1.0s/index) is required to expose this: velocity computed per
    index-step instead of per real second would be off by exactly
    1/0.004 = 250x.
    """
    tg = TriggerGroup(
        detectors=frozenset({"det"}),
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=0.003,
        deadtime=0.001,
    )
    sc: Scan[str, str, Never] = Sync(  # type: ignore[reportUnknownVariableType]
        Linspace("x", 0.0, 10.0, 100), fly=True, trigger_group=tg
    ).compile()  # type: ignore[reportArgumentType]
    w = windows(sc)[0]
    am = w.moving_axes["x"]

    # Linspace(x, 0, 10, 100) fence-post: step = 10/99, spanning index -0.5..99.5
    step = 10.0 / 99.0
    expected_start_position = -step / 2
    expected_end_position = expected_start_position + 100 * step
    # window duration = 100 points * (0.003 + 0.001)s/point = 0.4s
    expected_velocity = (expected_end_position - expected_start_position) / 0.4

    assert am.start_position == pytest.approx(expected_start_position)  # type: ignore[reportUnknownMemberType]
    assert am.end_position == pytest.approx(expected_end_position)  # type: ignore[reportUnknownMemberType]
    assert am.start_velocity == pytest.approx(expected_velocity)  # type: ignore[reportUnknownMemberType]
    assert am.end_velocity == pytest.approx(expected_velocity)  # type: ignore[reportUnknownMemberType]


def test_fly_scan_reversed_velocity_has_correct_sign():
    """AxisMotion velocity for a snake-reversed window must be negative when
    position decreases with time, not just correct in magnitude. Requires a
    detector-derived duration (not the masking 1.0s/index default) combined
    with a reversed window -- the sign bug only manifested when both apply.
    """
    tg = TriggerGroup(
        detectors=frozenset({"det"}),
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=0.003,
        deadtime=0.001,
    )
    sc: Scan[str, str, Never] = Sync(  # type: ignore[reportUnknownVariableType]
        Product(Linspace("y", 0.0, 1.0, 2), ~Linspace("x", 0.0, 10.0, 100)),
        fly=True,
        trigger_group=tg,
    ).compile()  # type: ignore[reportArgumentType]
    ws = windows(sc)
    assert len(ws) == 2

    forward, reverse = ws[0].moving_axes["x"], ws[1].moving_axes["x"]
    # Forward window: position increases with time -> positive velocity.
    assert forward.start_position < forward.end_position
    assert forward.start_velocity > 0
    # Reversed window: position decreases with time -> negative velocity,
    # not the same positive magnitude the pre-fix code reported.
    assert reverse.start_position > reverse.end_position
    assert reverse.start_velocity < 0
    # Both windows traverse the same physical range at the same rate, so
    # their velocities must be exact negatives of each other.
    assert reverse.start_velocity == pytest.approx(-forward.start_velocity)  # type: ignore[reportUnknownMemberType]
    assert reverse.end_velocity == pytest.approx(-forward.end_velocity)  # type: ignore[reportUnknownMemberType]


def test_window_positions_times_maps_real_seconds_to_physical_position():
    """window.positions(times) must return the correct physical position at
    each given real-second time -- not the position at index==time.
    """
    tg = TriggerGroup(
        detectors=frozenset({"det"}),
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=0.003,
        deadtime=0.001,
    )
    sc: Scan[str, str, Never] = Sync(  # type: ignore[reportUnknownVariableType]
        Linspace("x", 0.0, 10.0, 100), fly=True, trigger_group=tg
    ).compile()  # type: ignore[reportArgumentType]
    w = windows(sc)[0]
    assert w.duration == pytest.approx(0.4)  # type: ignore[reportUnknownMemberType]

    # Sample the whole 0.4s window at 4 real-time instants, caller-supplied.
    times = np.linspace(0.0, w.duration, 4)
    all_x = w.positions(times)["x"]
    assert len(all_x) == 4

    step = 10.0 / 99.0
    expected_start_position = -step / 2
    expected_end_position = expected_start_position + 100 * step
    # First sample (t=0) must be at the physical start; last sample (t=0.4,
    # the full duration) must be at the physical end -- not clustered near
    # the start, which is what the index/time unit-confusion bug produced.
    assert all_x[0] == pytest.approx(expected_start_position)  # type: ignore[reportUnknownMemberType]
    assert all_x[-1] == pytest.approx(expected_end_position)  # type: ignore[reportUnknownMemberType]


def test_fly_scan_snake_reverses_direction():
    sc: Scan[str, Never, Never] = Sync(  # type: ignore[reportUnknownVariableType]
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
    sc: Scan[str, Never, Never] = Sync(  # type: ignore[reportUnknownVariableType]
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


def test_with_start_trigger_index_truncates():
    tg = TriggerGroup(
        detectors=frozenset({"det"}),
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=0.003,
        deadtime=0.001,
    )
    sc: Scan[str, str, Never] = Sync(  # type: ignore[reportUnknownVariableType]
        Linspace("x", 0.0, 10.0, 10), fly=True, trigger_group=tg
    ).compile()  # type: ignore[reportArgumentType]
    resumed = sc.with_start(window=0, trigger_index=3)
    windows_list = list(resumed)
    assert len(windows_list) == 1
    ts = windows_list[0].trigger_sequences[0]
    assert ts.root.repeats == 7  # 10 - 3 = 7
    assert ts.root.detectors == frozenset({"det"})


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
    spec: Sync[str, Never, Never] = Sync(
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
    spec: Sync[str, Never, Never] = Sync(
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


def test_snake_step_inside_concat_reverses_on_odd_passes():
    """A snaked Sync inside a Concat reverses on the outer loop's odd passes."""
    a = Sync(~Linspace("x", 0.0, 1.0, 3), trigger_group=_tg("det1"), stream_name="a")
    b = Sync(Static("y", 5.0), trigger_group=_tg("det2"), stream_name="b")
    sc = Repeat(a.concat(b), num=3).compile()
    # Each pass: three "a" windows, then one "b" window (x stays put).
    xs = [p["x"] for p in _all_positions(sc)]
    assert xs == pytest.approx([0, 0.5, 1, 1, 1, 0.5, 0, 0, 0, 0.5, 1, 1])  # type: ignore[reportUnknownMemberType]
    a_dims = sc.windowed_streams[0].dimensions
    assert [(d.length, d.snake) for d in a_dims] == [(3, False), (3, True)]


def test_snake_fly_inside_concat_reverses_on_odd_passes():
    a = Sync(
        ~Linspace("x", 0.0, 1.0, 3),
        fly=True,
        trigger_group=_tg("det1"),
        stream_name="a",
    )
    b = Sync(Static("y", 5.0), trigger_group=_tg("det2"), stream_name="b")
    sc = Repeat(a.concat(b), num=2).compile()
    motions = [w.moving_axes["x"] for w in sc if w.moving_axes]
    # Midpoints 0, 0.5, 1 at indexes 0.5..2.5, so the sweep spans -0.25..1.25.
    ends = [p for m in motions for p in (m.start_position, m.end_position)]
    assert ends == pytest.approx([-0.25, 1.25, 1.25, -0.25])  # type: ignore[reportUnknownMemberType]


def test_snaked_concat_runs_snaked_child_sequence_backwards():
    """~Concat on an odd pass reverses what that pass would otherwise run.

    Pass 1 would otherwise run ~a reversed (1, 0.5, 0) then b (2, 3), so
    reversed it is b backwards (3, 2) then a forwards (0, 0.5, 1).
    """
    inner = Concat(~Linspace("x", 0.0, 1.0, 3), Linspace("x", 2.0, 3.0, 2))
    sc = Repeat(~inner, num=2).compile()
    xs = [p["x"] for p in _all_positions(sc)]
    assert xs == pytest.approx([0, 0.5, 1, 2, 3, 3, 2, 0, 0.5, 1])  # type: ignore[reportUnknownMemberType]


def test_snaked_concat_as_concat_side_keeps_its_snake():
    """~Concat as a side of another Concat is kept whole, not spliced in.

    Pass 1 runs the snaked side backwards (3, 2, 1, 0), then x5 as normal.
    """
    snaked = ~Concat(Linspace("x", 0.0, 1.0, 2), Linspace("x", 2.0, 3.0, 2))
    sc = Repeat(Concat(snaked, Linspace("x", 5.0, 6.0, 2)), num=2).compile()
    xs = [p["x"] for p in _all_positions(sc)]
    assert xs == pytest.approx([0, 1, 2, 3, 5, 6, 3, 2, 1, 0, 5, 6])  # type: ignore[reportUnknownMemberType]


# ---------------------------------------------------------------------------
# Phase B — trigger_sequences
# ---------------------------------------------------------------------------


def test_step_scan_trigger_sequences():
    tg = TriggerGroup(
        detectors=frozenset({"det1"}),
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=0.01,
        deadtime=0.001,
    )
    sc: Scan[str, str, Never] = Sync(  # type: ignore[reportUnknownVariableType]
        Linspace("x", 0.0, 10.0, 5), trigger_group=tg
    ).compile()  # type: ignore[reportArgumentType]  # noqa: E501
    ws = windows(sc)
    assert len(ws) == 5
    for w in ws:
        assert len(w.trigger_sequences) == 1
        ts = w.trigger_sequences[0]
        assert ts.root.detectors == frozenset({"det1"})
        assert ts.root == TriggerRepeat(
            detectors=frozenset({"det1"}), repeats=1, livetime=0.01, deadtime=0.001
        )


def test_fly_scan_trigger_sequences():
    tg = TriggerGroup(
        detectors=frozenset({"det1"}),
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=0.003,
        deadtime=0.001,
    )
    sc: Scan[str, str, Never] = Sync(  # type: ignore[reportUnknownVariableType]
        Linspace("x", 0.0, 10.0, 5), fly=True, trigger_group=tg
    ).compile()  # type: ignore[reportArgumentType]  # noqa: E501
    ws = windows(sc)
    assert len(ws) == 1
    ts = ws[0].trigger_sequences[0]
    assert ts.root.detectors == frozenset({"det1"})
    # fly: repeats = length * exposures_per_collection = 5 * 1
    assert ts.root == TriggerRepeat(
        detectors=frozenset({"det1"}), repeats=5, livetime=0.003, deadtime=0.001
    )


def test_multirate_trigger_sequences():
    trigger_group = TriggerGroup(
        detectors=frozenset({"saxs"}),
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=0.003,
        deadtime=0.001,
        followers=[
            # Encoder triggers 10x per saxs repeat: period must divide the
            # group's 0.003s livetime exactly, so livetime = 0.003/10 - deadtime.
            TriggerFollower(
                detectors=frozenset({"encoder"}),
                exposures_per_collection=10,
                collections_per_event=1,
                livetime=0.000299992,
                deadtime=8e-9,
                repeats=10,
            ),
        ],
    )
    sc: Scan[str, str, Never] = Sync(  # type: ignore[reportUnknownVariableType]
        Linspace("x", 0.0, 10.0, 100),
        fly=True,
        trigger_group=trigger_group,
    ).compile()  # type: ignore[reportArgumentType]
    ws = windows(sc)
    tss = ws[0].trigger_sequences
    assert len(tss) == 1  # single TriggerSequence with children
    ts = tss[0]
    assert ts.root.detectors == frozenset({"saxs"})
    assert ts.root == TriggerRepeat(
        detectors=frozenset({"saxs"}), repeats=100, livetime=0.003, deadtime=0.001
    )
    assert len(ts.children) == 1
    assert ts.children[0] == TriggerRepeat(
        detectors=frozenset({"encoder"}),
        repeats=10,
        livetime=0.000299992,
        deadtime=8e-9,
    )


def test_collections_per_event_multiplies_parent_repeats():
    # exposures_per_event = exposures_per_collection * collections_per_event = 2 * 3 = 6
    tg_step = TriggerGroup(
        detectors=frozenset({"det1"}),
        exposures_per_collection=2,
        collections_per_event=3,
        livetime=0.01,
        deadtime=0.001,
    )
    step_scan: Scan[str, str, Never] = Sync(  # type: ignore[reportUnknownVariableType]
        Linspace("x", 0.0, 10.0, 5), trigger_group=tg_step
    ).compile()  # type: ignore[reportArgumentType]  # noqa: E501
    for w in windows(step_scan):
        ts = w.trigger_sequences[0]
        assert ts.root == TriggerRepeat(
            detectors=frozenset({"det1"}), repeats=6, livetime=0.01, deadtime=0.001
        )

    tg_fly = TriggerGroup(
        detectors=frozenset({"det1"}),
        exposures_per_collection=2,
        collections_per_event=3,
        livetime=0.003,
        deadtime=0.001,
    )
    fly_scan: Scan[str, str, Never] = Sync(  # type: ignore[reportUnknownVariableType]
        Linspace("x", 0.0, 10.0, 100), fly=True, trigger_group=tg_fly
    ).compile()  # type: ignore[reportArgumentType]  # noqa: E501
    ts = windows(fly_scan)[0].trigger_sequences[0]
    # fly: repeats = length * exposures_per_event = 100 * 6
    assert ts.root == TriggerRepeat(
        detectors=frozenset({"det1"}), repeats=600, livetime=0.003, deadtime=0.001
    )


def test_non_integer_rate_ratio_raises():
    # child_period = 0.004 -> group_livetime/child_period = 0.75, not an
    # integer. Checked by validate_trigger_sequence (core.py) against the
    # derived TriggerRepeat at compile() time -- independent of whatever
    # repeats the follower declares, since the ratio check is purely
    # timing-based (ADR 0008 Decision 3).
    trigger_group = TriggerGroup(
        detectors=frozenset({"saxs"}),
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=0.003,
        deadtime=0.001,
        followers=[
            TriggerFollower(
                detectors=frozenset({"enc"}),
                exposures_per_collection=10,
                collections_per_event=1,
                livetime=0.003,
                deadtime=0.001,
                repeats=1,
            ),
        ],
    )
    with pytest.raises(ValueError, match="integer ratio"):
        Sync(  # type: ignore[reportUnknownVariableType]
            Linspace("x", 0.0, 10.0, 100),
            fly=True,
            trigger_group=trigger_group,
        ).compile()  # type: ignore[reportArgumentType]


def test_follower_missing_repeats_raises():
    # ADR 0008 Decision 3: repeats is caller-supplied on TriggerFollower,
    # never derived -- compile() rejects outright if it's still None rather
    # than falling back to a ratio-based calculation.
    trigger_group = TriggerGroup(
        detectors=frozenset({"saxs"}),
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=0.003,
        deadtime=0.001,
        followers=[
            TriggerFollower(
                detectors=frozenset({"enc"}),
                exposures_per_collection=10,
                collections_per_event=1,
                livetime=0.0003,
                deadtime=0.0,
                repeats=None,
            ),
        ],
    )
    with pytest.raises(ValueError, match="repeats must be set"):
        Sync(  # type: ignore[reportUnknownVariableType]
            Linspace("x", 0.0, 10.0, 100),
            fly=True,
            trigger_group=trigger_group,
        ).compile()  # type: ignore[reportArgumentType]


def test_follower_missing_livetime_raises():
    trigger_group = TriggerGroup(
        detectors=frozenset({"saxs"}),
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=0.003,
        deadtime=0.001,
        followers=[
            TriggerFollower(
                detectors=frozenset({"enc"}),
                exposures_per_collection=10,
                collections_per_event=1,
                livetime=None,
                deadtime=0.0,
                repeats=10,
            ),
        ],
    )
    with pytest.raises(ValueError, match="livetime and deadtime must be resolved"):
        Sync(  # type: ignore[reportUnknownVariableType]
            Linspace("x", 0.0, 10.0, 100),
            fly=True,
            trigger_group=trigger_group,
        ).compile()  # type: ignore[reportArgumentType]


def test_group_missing_livetime_raises():
    # Same check applies to the group's own (root) timing, not just followers.
    trigger_group = TriggerGroup(
        detectors=frozenset({"saxs"}),
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=None,
        deadtime=0.001,
    )
    with pytest.raises(ValueError, match="livetime and deadtime must be resolved"):
        Sync(  # type: ignore[reportUnknownVariableType]
            Linspace("x", 0.0, 10.0, 100),
            fly=True,
            trigger_group=trigger_group,
        ).compile()  # type: ignore[reportArgumentType]


# Spacer/overlap/duration checks live in core.validate_trigger_sequence,
# exercised above via Sync(trigger_group=...) and directly in test_core.py.


# ---------------------------------------------------------------------------
# active_stream_sets
# ---------------------------------------------------------------------------


def test_active_stream_sets_single_sync():
    tg = TriggerGroup(
        detectors=frozenset({"det"}),
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=0.01,
        deadtime=0.001,
    )
    sc: Scan[str, str, Never] = Sync(  # type: ignore[reportUnknownVariableType]
        Linspace("x", 0.0, 10.0, 5), trigger_group=tg
    ).compile()  # type: ignore[reportArgumentType]
    assert sc.active_stream_sets == [frozenset({"primary"})]


def _tg(detector: str) -> TriggerGroup[str]:
    return TriggerGroup(
        detectors=frozenset({detector}),
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=0.01,
        deadtime=0.001,
    )


def test_active_stream_sets_concat_different_names():
    sc = Concat(
        Sync(Linspace("x", 0.0, 5.0, 3), trigger_group=_tg("d1"), stream_name="diff"),
        Sync(Linspace("x", 5.0, 10.0, 5), trigger_group=_tg("d2"), stream_name="spec"),
    ).compile()
    assert sc.active_stream_sets == [frozenset({"diff"}), frozenset({"spec"})]


def test_active_stream_sets_concat_same_name_deduplicates():
    sc = Concat(
        Sync(Linspace("x", 0.0, 5.0, 3), trigger_group=_tg("det")),
        Sync(Linspace("x", 5.0, 10.0, 5), trigger_group=_tg("det")),
    ).compile()
    assert sc.active_stream_sets == [frozenset({"primary"})]


def test_active_stream_sets_flagship_under_wrappers():
    """Wrappers around a Repeat of a Concat still report every stream."""
    from scanspec.v2.core import MonitorStream
    from scanspec.v2.specs import Monitors

    diff = Sync(Static("e", 7.0), trigger_group=_tg("d1"), stream_name="diff")
    up = Sync(
        Linspace("e", 7.0, 7.1, 10),
        fly=True,
        trigger_group=_tg("d2"),
        stream_name="spec",
    )
    spec: Repeat[str, str, Never] = Repeat(diff.concat(up), num=2)
    sc: Scan[str, str, str] = Monitors(
        spec, monitors=[MonitorStream("temp", "tc1")]
    ).compile()
    assert sc.active_stream_sets == [frozenset({"diff"}), frozenset({"spec"})]
    assert sc.with_start(1).active_stream_sets == sc.active_stream_sets


def test_active_stream_sets_detector_less_sync_is_empty():
    sc: Scan[str, Never, Never] = Sync(  # type: ignore[reportUnknownVariableType]
        Linspace("x", 0.0, 1.0, 5), fly=True, duration=0.1
    ).compile()
    assert sc.active_stream_sets == []


# ---------------------------------------------------------------------------
# Sync placement
# ---------------------------------------------------------------------------


def test_sync_rejects_nested_sync():
    inner = Sync(Linspace("x", 0.0, 1.0, 5), trigger_group=_tg("det"))
    with pytest.raises(ValueError, match="Sync cannot contain another Sync"):
        Sync(inner)


def test_sync_rejects_sync_nested_under_concat():
    both = Concat(
        Sync(Linspace("x", 0.0, 1.0, 3), trigger_group=_tg("d1"), stream_name="a"),
        Sync(Linspace("x", 1.0, 2.0, 3), trigger_group=_tg("d2"), stream_name="b"),
    )
    with pytest.raises(ValueError, match="Sync cannot contain another Sync"):
        Sync(both, trigger_group=_tg("d3"), stream_name="c")


@pytest.mark.parametrize("side", ["outer", "inner"])
def test_product_rejects_sync(side: str):
    sync = Sync(Linspace("x", 0.0, 1.0, 3), trigger_group=_tg("det"))
    motion = Linspace("y", 0.0, 1.0, 2)
    with pytest.raises(ValueError, match="Product cannot contain a Sync"):
        if side == "outer":
            Product(sync, motion)  # type: ignore[reportArgumentType]
        else:
            Product(motion, sync)  # type: ignore[reportArgumentType]


def test_product_rejects_sync_below_repeat():
    sync = Sync(Linspace("x", 0.0, 1.0, 3), trigger_group=_tg("det"))
    with pytest.raises(ValueError, match="Product cannot contain a Sync"):
        Product(Linspace("y", 0.0, 1.0, 2), Repeat(sync, num=2))  # type: ignore[reportArgumentType]


def test_zip_rejects_sync():
    sync = Sync(Linspace("x", 0.0, 1.0, 3), trigger_group=_tg("det"))
    with pytest.raises(ValueError, match="Zip cannot contain a Sync"):
        Zip(sync, Static("z", 1.0))  # type: ignore[reportArgumentType]


def test_snake_rejects_sync():
    # The reversal would be missing from the Sync's stream dimensions.
    sync = Sync(Linspace("x", 0.0, 1.0, 3), trigger_group=_tg("det"))
    with pytest.raises(ValueError, match="Snake cannot contain a Sync"):
        Snake(sync)


def test_snake_rejects_concat_of_syncs():
    a = Sync(Linspace("x", 0.0, 1.0, 3), trigger_group=_tg("d1"), stream_name="a")
    b = Sync(Linspace("x", 1.0, 2.0, 2), trigger_group=_tg("d2"), stream_name="b")
    with pytest.raises(ValueError, match="Snake cannot contain a Sync"):
        Snake(a.concat(b))


def test_sync_placements_that_stay_allowed():
    """Repeat of a Sync, and a Concat mixing a Sync with plain motion."""
    sync = Sync(Linspace("x", 0.0, 1.0, 3), trigger_group=_tg("det"))
    assert len(windows(Repeat(sync, num=2).compile())) == 6
    # Concat ties DetectorT across both sides, so plain motion (DetectorT=Never)
    # doesn't type-check beside a Sync, although it's valid at runtime.
    mixed = Concat(sync, Linspace("x", 1.0, 2.0, 2)).compile()  # type: ignore[reportArgumentType]
    triggered = [w for w in windows(mixed) if w.trigger_sequences]
    assert len(triggered) == 3


# Sync over a Concat: each child gets the Sync's settings, from its own
# length. x1 = Linspace("x", 0, 1, 3) has midpoints 0, 0.5, 1 (step 0.5);
# x2 = Linspace("x", 1, 2, 4) has midpoints 1, 4/3, 5/3, 2 (step 1/3). A
# fly sweep runs from half a step before the first midpoint to half a step
# after the last; _tg's points take 0.01 + 0.001 = 0.011 s each.


def _x1_x2() -> Concat[str, Never, Never]:
    return Linspace("x", 0.0, 1.0, 3).concat(Linspace("x", 1.0, 2.0, 4))


def test_sync_over_concat_flies_each_child():
    ws = windows(Sync(_x1_x2(), fly=True, trigger_group=_tg("det")).compile())
    assert [w.moving_axes["x"].start_position for w in ws] == pytest.approx(  # type: ignore[reportUnknownMemberType]
        [-0.25, 5 / 6]
    )
    assert [w.moving_axes["x"].end_position for w in ws] == pytest.approx(  # type: ignore[reportUnknownMemberType]
        [1.25, 13 / 6]
    )
    assert [w.duration for w in ws] == pytest.approx([0.033, 0.044])  # type: ignore[reportUnknownMemberType]
    assert [w.trigger_sequences[0].root.repeats for w in ws] == [3, 4]


def test_sync_over_concat_steps_each_child_with_its_duration():
    spec: Sync[str, Never, Never] = Sync(_x1_x2(), duration=0.5)
    ws = windows(spec.compile())
    assert [w.duration for w in ws] == [0.5] * 7
    assert not any(w.moving_axes for w in ws)


def test_sync_over_concat_stream_dimension_moves_the_childrens_axes():
    sc = Sync(_x1_x2(), fly=True, trigger_group=_tg("det")).compile()
    (dim,) = sc.windowed_streams[0].dimensions
    assert dim.axes == ["x"]
    assert dim.length == 7
    np.testing.assert_allclose(
        next(dim.setpoints("x")), [0.0, 0.5, 1.0, 1.0, 4 / 3, 5 / 3, 2.0]
    )


def test_sync_over_concat_matches_concat_of_syncs():
    kw: dict[str, Any] = {"fly": True, "trigger_group": _tg("det")}
    x1, x2 = Linspace("x", 0.0, 1.0, 3), Linspace("x", 1.0, 2.0, 4)
    distributed = Sync(x1.concat(x2), **kw).compile()
    separate = Sync(x1, **kw).concat(Sync(x2, **kw)).compile()
    for a, b in zip(windows(distributed), windows(separate), strict=True):
        assert a.moving_axes == b.moving_axes
        assert a.duration == b.duration
        assert a.trigger_sequences == b.trigger_sequences
    assert [s.number_of_events for s in distributed.windowed_streams] == [
        s.number_of_events for s in separate.windowed_streams
    ]


def test_sync_over_product_with_inner_concat_flies_each_child_per_row():
    spec = Linspace("y", 0.0, 1.0, 2) * _x1_x2()
    sc = Sync(spec, fly=True, trigger_group=_tg("det")).compile()
    ws = windows(sc)
    assert [w.trigger_sequences[0].root.repeats for w in ws] == [3, 4, 3, 4]
    assert [d.axes for d in sc.windowed_streams[0].dimensions] == [["y"], ["x"]]


def test_concat_rejects_same_stream_with_different_trigger_groups():
    slow = _tg("det")
    fast = slow.model_copy(update={"livetime": 0.005})
    with pytest.raises(ValueError, match="'primary' must have identical trigger"):
        Concat(
            Sync(Linspace("x", 0.0, 1.0, 3), trigger_group=slow),
            Sync(Linspace("x", 1.0, 2.0, 3), trigger_group=fast),
        ).compile()


def test_concat_rejects_same_stream_with_different_follower_repeats():
    def tg(repeats: int) -> TriggerGroup[str]:
        return TriggerGroup(
            detectors=frozenset({"saxs"}),
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
                    repeats=repeats,
                )
            ],
        )

    same = Concat(
        Sync(Linspace("x", 0.0, 1.0, 3), fly=True, trigger_group=tg(10)),
        Sync(Linspace("x", 1.0, 2.0, 3), fly=True, trigger_group=tg(10)),
    ).compile()
    assert same.windowed_streams[0].dimensions[-1].length == 6
    with pytest.raises(ValueError, match="'primary' must have identical trigger"):
        Concat(
            Sync(Linspace("x", 0.0, 1.0, 3), fly=True, trigger_group=tg(10)),
            Sync(Linspace("x", 1.0, 2.0, 3), fly=True, trigger_group=tg(5)),
        ).compile()


def test_concat_same_stream_setpoints_follow_each_piece():
    """Merged stream setpoints come from each piece, not the left one extended.

    ``up`` sweeps 7.0 -> 7.1 and ``down`` 7.1 -> 7.0 over 1000 points each;
    Linspace's start/stop are its first/last midpoints.
    """
    up = Sync(Linspace("e", 7.0, 7.1, 1000), fly=True, trigger_group=_tg("spec_det"))
    down = Sync(Linspace("e", 7.1, 7.0, 1000), fly=True, trigger_group=_tg("spec_det"))
    dim = Repeat(up.concat(down), num=2).compile().windowed_streams[0].dimensions[-1]
    assert dim.length == 2000
    full = next(dim.setpoints("e"))
    assert full[[0, 999, 1000, 1999]] == pytest.approx([7.0, 7.1, 7.1, 7.0])  # type: ignore[reportUnknownMemberType]
    np.testing.assert_allclose(np.concatenate(list(dim.setpoints("e", 300))), full)


def test_concat_rejects_same_stream_on_different_axes():
    with pytest.raises(ValueError, match="must move the same axes"):
        Concat(
            Sync(Linspace("e", 0.0, 1.0, 3), trigger_group=_tg("det")),
            Sync(Linspace("x", 0.0, 1.0, 3), trigger_group=_tg("det")),
        ).compile()


def test_concat_rejects_detector_in_two_streams():
    with pytest.raises(ValueError, match=r"only one stream.*'det'"):
        Concat(
            Sync(Linspace("x", 0.0, 1.0, 3), trigger_group=_tg("det"), stream_name="a"),
            Sync(Linspace("x", 1.0, 2.0, 3), trigger_group=_tg("det"), stream_name="b"),
        ).compile()


def test_duration_derived_from_detectors():
    tg = TriggerGroup(
        detectors=frozenset({"det1"}),
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=0.01,
        deadtime=0.001,
    )
    sc: Scan[str, str, Never] = Sync(  # type: ignore[reportUnknownVariableType]
        Linspace("x", 0.0, 10.0, 5), trigger_group=tg
    ).compile()  # type: ignore[reportArgumentType]  # noqa: E501
    ws = windows(sc)
    # step: 1 × (0.01 + 0.001) = 0.011
    for w in ws:
        assert w.duration == pytest.approx(0.011)  # type: ignore[reportUnknownMemberType]


def test_duration_derived_from_fly_detectors():
    tg = TriggerGroup(
        detectors=frozenset({"det1"}),
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=0.003,
        deadtime=0.001,
    )
    sc: Scan[str, str, Never] = Sync(  # type: ignore[reportUnknownVariableType]
        Linspace("x", 0.0, 10.0, 5), fly=True, trigger_group=tg
    ).compile()  # type: ignore[reportArgumentType]  # noqa: E501
    ws = windows(sc)
    # fly: 5 × (0.003 + 0.001) = 0.02
    assert ws[0].duration == pytest.approx(0.02)  # type: ignore[reportUnknownMemberType]


def test_sync_rejects_duration_with_trigger_group():
    tg = TriggerGroup(
        detectors=frozenset({"det1"}),
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=0.01,
        deadtime=0.001,
    )
    with pytest.raises(ValueError, match="duration cannot be combined"):
        Sync(Linspace("x", 0.0, 10.0, 5), trigger_group=tg, duration=0.05)


# ---------------------------------------------------------------------------
# Scan capability properties
# ---------------------------------------------------------------------------


def test_scan_has_moving_axes_fly():
    sc: Scan[str, Never, Never] = Sync(  # type: ignore[reportUnknownVariableType]
        Linspace("x", 0, 10, 5), fly=True
    ).compile()  # type: ignore[reportArgumentType]  # noqa: E501
    assert sc.has_moving_axes is True
    assert sc.non_linear is False


def test_scan_has_moving_axes_step():
    sc = Linspace("x", 0, 10, 5).compile()
    assert sc.has_moving_axes is False
    assert sc.non_linear is False


def test_scan_non_linear_spiral():
    sc: Scan[str, Never, Never] = Sync(  # type: ignore[reportUnknownVariableType]
        Spiral("x", 0, 5, 2, "y", 10, 10), fly=True
    ).compile()  # type: ignore[reportArgumentType]  # noqa: E501
    assert sc.has_moving_axes is True
    assert sc.non_linear is True


def _step_then_fly(fly_motion: Spec[str, Never, Never]) -> Scan[str, str, Never]:
    """A step Sync then a flown Sync, on separate streams."""
    step = Sync(Static("x", 0.0), trigger_group=_tg("det1"), stream_name="step")
    fly = Sync(fly_motion, fly=True, trigger_group=_tg("det2"), stream_name="fly")
    return step.concat(fly).compile()


def test_scan_has_moving_axes_concat_child_flies():
    # The fly window comes from a Concat child, not a top-level generator.
    sc = _step_then_fly(Linspace("x", 0, 10, 5))
    assert sc.has_moving_axes is True
    assert sc.non_linear is False


def test_scan_non_linear_concat_child_spiral():
    sc = _step_then_fly(Spiral("x", 0, 5, 2, "y", 10, 10))
    assert sc.has_moving_axes is True
    assert sc.non_linear is True


# ---------------------------------------------------------------------------
# AnySpec: out-of-package subclass included in union
# ---------------------------------------------------------------------------


def test_anyspec_extension():
    """Spec subclass defined outside specs.py is included in the AnySpec union."""
    from typing import Never as Nv

    from pydantic import TypeAdapter

    from scanspec.v2.core import LinearSource as LinSrc
    from scanspec.v2.core import Scan as Sc
    from scanspec.v2.core import WindowGenerator as WinGen
    from scanspec.v2.specs import AnySpec

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


def test_none_livetime_raises():
    tg = TriggerGroup(
        detectors=frozenset({"det1"}),
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=None,
        deadtime=0.001,
    )
    with pytest.raises(ValueError, match="livetime"):
        Sync(Linspace("x", 0.0, 10.0, 5), trigger_group=tg).compile()


def test_none_deadtime_raises():
    tg = TriggerGroup(
        detectors=frozenset({"det1"}),
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=0.01,
        deadtime=None,
    )
    with pytest.raises(ValueError, match="deadtime"):
        Sync(Linspace("x", 0.0, 10.0, 5), trigger_group=tg).compile()


# ---------------------------------------------------------------------------
# Phase B — multi-stream Concat
# ---------------------------------------------------------------------------


def test_concat_fly_step_windows():
    """Concat of step + fly Syncs: 2 windows (1 step + 1 fly)."""
    tg = TriggerGroup(
        detectors=frozenset({"det1"}),
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=0.01,
        deadtime=0.001,
    )
    tg2 = tg.model_copy(update={"detectors": frozenset({"det2"})})
    step_acq: Sync[str, str, Never] = Sync(
        Static("x", 5.0), trigger_group=tg, stream_name="s1"
    )
    fly_acq: Sync[str, str, Never] = Sync(
        Linspace("x", 0.0, 10.0, 5),
        fly=True,
        trigger_group=tg2,
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
    """Two Syncs with different stream names."""
    tg1 = TriggerGroup(
        detectors=frozenset({"det1"}),
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=0.01,
        deadtime=0.001,
    )
    tg2 = TriggerGroup(
        detectors=frozenset({"det2"}),
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=0.003,
        deadtime=0.001,
    )
    a1: Sync[str, str, Never] = Sync(
        Static("x", 5.0), trigger_group=tg1, stream_name="diff"
    )
    a2: Sync[str, str, Never] = Sync(
        Linspace("x", 0.0, 10.0, 100),
        fly=True,
        trigger_group=tg2,
        stream_name="spec",
    )
    sc = a1.concat(a2).compile()
    names = {s.name for s in sc.windowed_streams}
    assert names == {"diff", "spec"}


def test_concat_same_stream_sums_inner():
    """Two Syncs with same stream name → inner length summed."""
    tg = TriggerGroup(
        detectors=frozenset({"det1"}),
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=0.003,
        deadtime=0.001,
    )
    a1: Sync[str, str, Never] = Sync(
        Linspace("x", 0.0, 5.0, 500),
        fly=True,
        trigger_group=tg,
        stream_name="primary",
    )
    a2: Sync[str, str, Never] = Sync(
        Linspace("x", 5.0, 0.0, 500),
        fly=True,
        trigger_group=tg,
        stream_name="primary",
    )
    sc = a1.concat(a2).compile()
    assert len(sc.windowed_streams) == 1
    stream = sc.windowed_streams[0]
    assert stream.name == "primary"
    assert stream.dimensions[-1].length == 1000


def test_repeat_concat_windows():
    """Repeat wrapping a concat: n_repeat x groups_per_concat."""
    tg = TriggerGroup(
        detectors=frozenset({"det1"}),
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=0.01,
        deadtime=0.001,
    )
    tg2 = tg.model_copy(update={"detectors": frozenset({"det2"})})
    a1: Sync[str, str, Never] = Sync(
        Static("x", 5.0), trigger_group=tg, stream_name="s1"
    )
    a2: Sync[str, str, Never] = Sync(
        Linspace("x", 0.0, 10.0, 5),
        fly=True,
        trigger_group=tg2,
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
    tg = TriggerGroup(
        detectors=frozenset({"det1"}),
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=0.01,
        deadtime=0.001,
    )
    tg2 = tg.model_copy(update={"detectors": frozenset({"det2"})})
    a1: Sync[str, str, Never] = Sync(
        Static("x", 5.0), trigger_group=tg, stream_name="s1"
    )
    a2: Sync[str, str, Never] = Sync(
        Linspace("x", 0.0, 10.0, 5),
        fly=True,
        trigger_group=tg2,
        stream_name="s2",
    )
    sc = Repeat(a1.concat(a2), num=10).compile()
    for stream in sc.windowed_streams:
        # outermost dim is the repeat
        assert stream.dimensions[0].length == 10


def test_concat_previous_chain():
    """Previous chain across all windows from grouped concat."""
    tg = TriggerGroup(
        detectors=frozenset({"det1"}),
        exposures_per_collection=1,
        collections_per_event=1,
        livetime=0.01,
        deadtime=0.001,
    )
    tg2 = tg.model_copy(update={"detectors": frozenset({"det2"})})
    a1: Sync[str, str, Never] = Sync(
        Static("x", 5.0), trigger_group=tg, stream_name="s1"
    )
    a2: Sync[str, str, Never] = Sync(
        Static("x", 10.0), trigger_group=tg2, stream_name="s2"
    )
    sc = Repeat(a1.concat(a2), num=3).compile()
    ws = windows(sc)
    assert ws[0].previous is None
    for i in range(1, len(ws)):
        assert ws[i].previous is ws[i - 1]


# ---------------------------------------------------------------------------
# Concat with multi-generator sides
# ---------------------------------------------------------------------------


def _flagship() -> Repeat[str, str, Never]:
    """200 x (one diffraction image, then a snaked up/down spectroscopy sweep)."""
    diff = Sync(Static("e", 7.0), trigger_group=_tg("diffraction"), stream_name="diff")
    spec = Sync(
        Repeat(~Linspace("e", 7.0, 7.1, 1000), 2),
        fly=True,
        trigger_group=_tg("spectroscopy"),
        stream_name="spec",
    )
    return Repeat(diff.concat(spec), 200)


def _sweep(w: Window[str, Any]) -> tuple[float, float]:
    m = w.moving_axes["e"]
    return (m.start_position, m.end_position)


def test_concat_multi_generator_flagship_streams():
    sc = _flagship().compile()
    dims = {
        s.name: [(d.length, d.snake) for d in s.dimensions] for s in sc.windowed_streams
    }
    assert dims == {
        "diff": [(200, False), (1, False)],
        "spec": [(200, False), (2, False), (1000, True)],
    }
    assert sc.has_moving_axes


def test_concat_multi_generator_flagship_windows():
    ws = windows(_flagship().compile())
    assert len(ws) == 600
    detectors = [w.trigger_sequences[0].root.detectors for w in ws]
    assert (
        detectors
        == [
            frozenset({"diffraction"}),
            frozenset({"spectroscopy"}),
            frozenset({"spectroscopy"}),
        ]
        * 200
    )
    # 1000 midpoints from 7.0 to 7.1 are s = 0.1/999 apart; a sweep covers
    # half a step beyond the first and last.
    s = 0.1 / 999
    up, down = (7.0 - s / 2, 7.1 + s / 2), (7.1 + s / 2, 7.0 - s / 2)
    assert ws[0].moving_axes == {}
    assert ws[0].static_axes == {"e": 7.0}
    for i in (0, 199):
        assert _sweep(ws[3 * i + 1]) == pytest.approx(up)  # type: ignore[reportUnknownMemberType]
        assert _sweep(ws[3 * i + 2]) == pytest.approx(down)  # type: ignore[reportUnknownMemberType]


def test_concat_multi_generator_with_start_mid_flagship():
    """Window 301 is iteration 100's up sweep."""
    sc = _flagship().compile()
    rest = windows(sc.with_start(301))
    assert len(rest) == 299
    s = 0.1 / 999
    assert _sweep(rest[0]) == pytest.approx((7.0 - s / 2, 7.1 + s / 2))  # type: ignore[reportUnknownMemberType]
    assert rest[2].trigger_sequences[0].root.detectors == frozenset({"diffraction"})


def test_concat_multi_generator_snake_parity_continues_across_passes():
    """Spec dims [2, 3, 3 snake]: sweep k of pass n has flat index 3n + k.

    So an odd inner repeat runs up/down/up then down/up/down.
    """
    diff = Sync(Static("x", 5.0), trigger_group=_tg("det1"), stream_name="diff")
    spec = Sync(
        Repeat(~Linspace("x", 0.0, 1.0, 3), 3),
        fly=True,
        trigger_group=_tg("det2"),
        stream_name="spec",
    )
    sc = Repeat(diff.concat(spec), 2).compile()
    spec_dims = sc.windowed_streams[1].dimensions
    assert [(d.length, d.snake) for d in spec_dims] == [
        (2, False),
        (3, False),
        (3, True),
    ]
    motions = [w.moving_axes["x"] for w in sc if w.moving_axes]
    # Midpoints 0, 0.5, 1 at indexes 0.5..2.5, so a sweep spans -0.25..1.25.
    ends = [p for m in motions for p in (m.start_position, m.end_position)]
    up, down = [-0.25, 1.25], [1.25, -0.25]
    assert ends == pytest.approx([*up, *down, *up, *down, *up, *down])  # type: ignore[reportUnknownMemberType]


def test_concat_multi_generator_step_side_positions():
    """A y(2) * ~x(2) side runs as nested loops, its y folded into the windows."""
    a = Sync(Static("x", 9.0), trigger_group=_tg("det1"), stream_name="a")
    b = Sync(
        Linspace("y", 0.0, 1.0, 2) * ~Linspace("x", 0.0, 1.0, 2),
        trigger_group=_tg("det2"),
        stream_name="b",
    )
    sc = Repeat(a.concat(b), 2).compile()
    one_pass = [
        {"x": 9.0},
        {"x": 0.0, "y": 0.0},
        {"x": 1.0, "y": 0.0},
        {"x": 1.0, "y": 1.0},
        {"x": 0.0, "y": 1.0},
    ]
    second_a = {"x": 9.0, "y": 1.0}
    assert _all_positions(sc) == [*one_pass, second_a, *one_pass[1:]]
    b_dims = sc.windowed_streams[1].dimensions
    assert [(d.length, d.snake) for d in b_dims] == [
        (2, False),
        (2, False),
        (2, True),
    ]


def test_concat_motion_only_multi_generator_side():
    sc = (
        Linspace("x", 0.0, 1.0, 2)
        .concat(Repeat(Linspace("x", 2.0, 3.0, 2), 2))
        .compile()
    )
    xs = [p["x"] for p in _all_positions(sc)]
    assert xs == [0.0, 1.0, 2.0, 3.0, 2.0, 3.0]


def test_concat_motion_only_multi_generator_side_axes_mismatch_raises():
    with pytest.raises(ValueError, match="must move the same axes"):
        Linspace("x", 0.0, 1.0, 2).concat(
            Linspace("y", 0.0, 1.0, 2) * Linspace("x", 0.0, 1.0, 2)
        ).compile()


def _multi_generator_concat() -> Concat[str, Never, Never]:
    return Static("e", 7.0).concat(Repeat(Linspace("e", 7.0, 7.1, 5), 2))


@pytest.mark.parametrize(
    "op,build",
    [
        ("Sync", lambda c: Sync(c, trigger_group=_tg("det"))),  # type: ignore[reportUnknownLambdaType]
        ("Zip", lambda c: Zip(c, Linspace("y", 0.0, 1.0, 11))),  # type: ignore[reportUnknownLambdaType]
        ("Zip", lambda c: Zip(Linspace("y", 0.0, 1.0, 11), c)),  # type: ignore[reportUnknownLambdaType]
        ("Snake", lambda c: ~c),  # type: ignore[reportUnknownLambdaType]
    ],
)
def test_ops_reject_concat_with_multi_generator_side(op: str, build: Any):
    spec = build(_multi_generator_concat())
    with pytest.raises(ValueError, match=f"{op} cannot take a Concat"):
        spec.compile()


def test_multi_generator_concat_has_no_flat_setpoints():
    (gen,) = _multi_generator_concat().compile().generators
    with pytest.raises(ValueError, match="no flat position function"):
        gen.setpoints(np.array([0.5]))


def test_concat_rejects_merging_multi_dimension_same_stream():
    single = Sync(Linspace("e", 7.0, 7.1, 5), trigger_group=_tg("det"))
    double = Sync(Repeat(Linspace("e", 7.1, 7.0, 5), 2), trigger_group=_tg("det"))
    with pytest.raises(ValueError, match="only merge single-dimension streams"):
        single.concat(double).compile()


# ---------------------------------------------------------------------------
# Combinators reject continuous_streams and monitors
# ---------------------------------------------------------------------------


def test_concat_rejects_continuous_streams():
    from scanspec.v2.core import ContinuousStream
    from scanspec.v2.specs import ContinuousStreams

    a: ContinuousStreams[str, str, Never] = ContinuousStreams(
        Sync(Linspace("x", 0, 1, 5)),
        continuous_streams=[
            ContinuousStream("cam", [DetectorGroup(1, 1, 0.01, 0.001, ["c1"])])
        ],
    )
    with pytest.raises(ValueError, match="Concat does not accept.*continuous"):
        a.concat(Sync(Linspace("x", 1, 2, 5))).compile()  # type: ignore[reportArgumentType]


def test_concat_rejects_monitors():
    from scanspec.v2.core import MonitorStream
    from scanspec.v2.specs import Monitors

    a: Monitors[str, Never, str] = Monitors(
        Sync(Linspace("x", 0, 1, 5)),
        monitors=[MonitorStream("temp", "tc1")],
    )
    with pytest.raises(ValueError, match="Concat does not accept.*monitors"):
        a.concat(Sync(Linspace("x", 1, 2, 5))).compile()  # type: ignore[reportArgumentType]


def test_product_rejects_continuous_streams():
    from scanspec.v2.core import ContinuousStream
    from scanspec.v2.specs import ContinuousStreams

    a: ContinuousStreams[str, str, Never] = ContinuousStreams(
        Linspace("x", 0, 1, 5),
        continuous_streams=[
            ContinuousStream("cam", [DetectorGroup(1, 1, 0.01, 0.001, ["c1"])])
        ],
    )
    with pytest.raises(ValueError, match="Product does not accept.*continuous"):
        (Linspace("y", 0, 1, 3) * a).compile()  # type: ignore[reportOperatorIssue]


def test_zip_rejects_monitors():
    from scanspec.v2.core import MonitorStream
    from scanspec.v2.specs import Monitors

    a: Monitors[str, Never, str] = Monitors(
        Linspace("x", 0, 1, 5),
        monitors=[MonitorStream("temp", "tc1")],
    )
    with pytest.raises(ValueError, match="Zip does not accept.*monitors"):
        Linspace("y", 0, 1, 5).zip(a).compile()  # type: ignore[reportArgumentType]


# ---------------------------------------------------------------------------
# Monitors/ContinuousStreams wrappers (ADR 0009)
# ---------------------------------------------------------------------------


def test_monitors_rejects_double_wrapping():
    from scanspec.v2.core import MonitorStream
    from scanspec.v2.specs import Monitors

    inner: Monitors[str, Never, str] = Monitors(
        Sync(Linspace("x", 0, 1, 5)),
        monitors=[MonitorStream("temp", "tc1")],
    )
    with pytest.raises(ValueError, match="already has monitors attached"):
        Monitors(inner, monitors=[MonitorStream("temp2", "tc2")]).compile()


def test_continuous_streams_rejects_double_wrapping():
    from scanspec.v2.core import ContinuousStream
    from scanspec.v2.specs import ContinuousStreams

    inner: ContinuousStreams[str, str, Never] = ContinuousStreams(
        Sync(Linspace("x", 0, 1, 5)),
        continuous_streams=[
            ContinuousStream("cam1", [DetectorGroup(1, 1, 0.01, 0.001, ["c1"])])
        ],
    )
    with pytest.raises(ValueError, match="already has continuous_streams attached"):
        ContinuousStreams(
            inner,
            continuous_streams=[
                ContinuousStream("cam2", [DetectorGroup(1, 1, 0.01, 0.001, ["c2"])])
            ],
        ).compile()


def test_continuous_streams_and_monitors_wrappers_commute():
    """Nesting order between ContinuousStreams and Monitors doesn't matter --
    each only ever touches its own field on the compiled Scan (ADR 0009)."""
    from scanspec.v2.core import ContinuousStream, MonitorStream
    from scanspec.v2.specs import ContinuousStreams, Monitors

    cs = [ContinuousStream("cam", [DetectorGroup(1, 1, 0.01, 0.001, ["c1"])])]
    mons = [MonitorStream("temp", "tc1")]

    sync_a: Sync[str, str, str] = Sync(Linspace("x", 0.0, 1.0, 5))
    outer_continuous = ContinuousStreams(
        Monitors(sync_a, monitors=mons),
        continuous_streams=cs,
    ).compile()
    sync_b: Sync[str, str, str] = Sync(Linspace("x", 0.0, 1.0, 5))
    outer_monitors = Monitors(
        ContinuousStreams(sync_b, continuous_streams=cs),
        monitors=mons,
    ).compile()

    assert outer_continuous.continuous_streams == outer_monitors.continuous_streams
    assert outer_continuous.monitors == outer_monitors.monitors
    assert outer_continuous.windowed_streams == outer_monitors.windowed_streams


# ---------------------------------------------------------------------------
# Linspace.bounded — compile
# ---------------------------------------------------------------------------


def test_linspace_bounded_compile_one_point():
    sc = Linspace.bounded("x", 0.0, 1.0, 1).compile()
    g = gens(sc)
    assert len(g) == 1
    assert g[0].length == 1
    pts = g[0].setpoints(np.array([0.5]))["x"]
    np.testing.assert_allclose(pts, [0.5])


def test_linspace_bounded_compile_many_points():
    sc = Linspace.bounded("x", 0.0, 1.0, 4).compile()
    g = gens(sc)
    assert g[0].length == 4
    pts = g[0].setpoints(np.arange(4) + 0.5)["x"]
    np.testing.assert_allclose(pts, [0.125, 0.375, 0.625, 0.875])


def test_linspace_bounded_compile_symmetric():
    # bounded(x, 3, 7, 2) → Linspace(x, 4, 6, 2) → midpoints [4, 6]
    sc = Linspace.bounded("x", 3.0, 7.0, 2).compile()
    pts = gens(sc)[0].setpoints(np.arange(2) + 0.5)["x"]
    np.testing.assert_allclose(pts, [4.0, 6.0])


# ---------------------------------------------------------------------------
# Range — compile
# ---------------------------------------------------------------------------


def test_range_compile_dimensions():
    from scanspec.v2.specs import Range

    sc = Range("x", 0.0, 1.0, 0.25).compile()
    g = gens(sc)
    assert len(g) == 1
    assert g[0].axes == ["x"]
    assert g[0].length == 5  # 0, 0.25, 0.5, 0.75, 1.0


@pytest.mark.parametrize("step", [0.25, 0.25 + 1e-8])
def test_range_setpoints_match_linspace(step: float) -> None:
    from scanspec.v2.specs import Range

    sc_range = Range("x", 0.0, 1.0, step).compile()
    sc_linspace = Linspace("x", 0.0, 1.0, 5).compile()
    idx = np.arange(5) + 0.5
    pts_range = gens(sc_range)[0].setpoints(idx)["x"]
    pts_linspace = gens(sc_linspace)[0].setpoints(idx)["x"]
    np.testing.assert_allclose(pts_range, pts_linspace)


def test_range_one_point():
    from scanspec.v2.specs import Range

    # step > (stop - start) → only one midpoint at start
    sc = Range("x", 0.0, 1.0, 2.0).compile()
    g = gens(sc)
    assert g[0].length == 1
    pts = g[0].setpoints(np.array([0.5]))["x"]
    np.testing.assert_allclose(pts, [0.0])


def test_range_stop_not_on_grid():
    """Last setpoint should be start + (num-1)*step even if stop is mid-interval."""
    from scanspec.v2.specs import Range

    # stop=1.1 is not a grid point; actual last midpoint is 0.0 + 2*0.5 = 1.0
    sc = Range("x", 0.0, 1.1, 0.5).compile()
    g = gens(sc)
    assert g[0].length == 3
    pts = g[0].setpoints(np.arange(3) + 0.5)["x"]
    np.testing.assert_allclose(pts, [0.0, 0.5, 1.0])


def test_range_descending_stop_not_on_grid():
    """Descending range: last point is start - (num-1)*step."""
    from scanspec.v2.specs import Range

    # start=5, stop=2.5, step=1 → 3 points at 5, 4, 3 (not 2.5)
    sc = Range("x", 5.0, 2.5, 1.0).compile()
    g = gens(sc)
    assert g[0].length == 3
    pts = g[0].setpoints(np.arange(3) + 0.5)["x"]
    np.testing.assert_allclose(pts, [5.0, 4.0, 3.0])


@pytest.mark.parametrize("step", [1.0, 1.0 + 1e-8])
def test_range_two_points(step: float) -> None:
    from scanspec.v2.specs import Range

    sc = Range("x", 0.0, 1.0, step).compile()
    assert gens(sc)[0].length == 2
    pts = gens(sc)[0].setpoints(np.arange(2) + 0.5)["x"]
    np.testing.assert_allclose(pts, [0.0, 1.0])


def test_range_snake_flag():
    from scanspec.v2.specs import Range

    sc = (~Range("x", 0.0, 1.0, 0.25)).compile()
    assert gens(sc)[-1].snake is True


# ---------------------------------------------------------------------------
# Range.bounded — compile
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("step", [0.25, 0.25 + 1e-8])
def test_range_bounded_setpoints(step: float) -> None:
    from scanspec.v2.specs import Range

    sc = Range.bounded("x", 0.0, 1.0, step).compile()
    g = gens(sc)
    assert g[0].length == 4
    pts = g[0].setpoints(np.arange(4) + 0.5)["x"]
    np.testing.assert_allclose(pts, [0.125, 0.375, 0.625, 0.875])


@pytest.mark.parametrize(
    "lower,upper,step,expected_mid",
    [
        (0.0, 1.0, 0.8, [0.4]),  # step < range → one frame
        (0.0, 1.0, 1.0, [0.5]),  # step == range → one frame
        (0.0, 1.0, 1.2, [0.5]),  # step > range → clamped to one frame
    ],
)
def test_range_bounded_one_point_setpoints(
    lower: float, upper: float, step: float, expected_mid: list[float]
) -> None:
    from scanspec.v2.specs import Range

    sc = Range.bounded("x", lower, upper, step).compile()
    g = gens(sc)
    assert g[0].length == 1
    pts = g[0].setpoints(np.array([0.5]))["x"]
    np.testing.assert_allclose(pts, expected_mid)


def test_range_bounded_lower_equals_upper_compile():
    """lower == upper must produce exactly one point at that position."""
    from scanspec.v2.specs import Range

    sc = Range.bounded("x", 7.0, 7.0, 0.5).compile()
    g = gens(sc)
    assert g[0].length == 1
    pts = g[0].setpoints(np.array([0.5]))["x"]
    np.testing.assert_allclose(pts, [7.0])


def test_line_compile_same_as_linspace():
    from scanspec.v2.specs import Line

    sc_line = Line("x", 0.0, 10.0, 5).compile()
    sc_linspace = Linspace("x", 0.0, 10.0, 5).compile()
    idx = np.arange(5) + 0.5
    pts_line = gens(sc_line)[0].setpoints(idx)["x"]
    pts_linspace = gens(sc_linspace)[0].setpoints(idx)["x"]
    np.testing.assert_allclose(pts_line, pts_linspace)


# ---------------------------------------------------------------------------
# Ellipse — compile
# ---------------------------------------------------------------------------


def test_ellipse_compile_returns_single_generator():
    from scanspec.v2.specs import Ellipse

    sc = Ellipse("x", 5.0, 1.0, 0.5, "y", 0.0).compile()
    assert len(gens(sc)) == 1


def test_ellipse_compile_point_count():
    from scanspec.v2.specs import Ellipse

    # x: [4.5, 5.0, 5.5], y: [-0.5, 0.0, 0.5] → 9-point grid, 5 inside ellipse
    sc = Ellipse("x", 5.0, 1.0, 0.5, "y", 0.0).compile()
    assert gens(sc)[0].length == 5


def test_ellipse_compile_all_points_inside():
    from scanspec.v2.specs import Ellipse

    sc = Ellipse("x", 5.0, 1.0, 0.5, "y", 0.0).compile()
    g = gens(sc)[0]
    idx = np.arange(g.length) + 0.5
    pts = g.setpoints(idx)
    x = pts["x"] - 5.0
    y = pts["y"] - 0.0
    # All points must satisfy the ellipse equation <= 1
    np.testing.assert_array_less(
        (2 * x / 1.0) ** 2 + (2 * y / 1.0) ** 2,
        np.ones(g.length) + 1e-9,
    )


def test_ellipse_compile_axes_present():
    from scanspec.v2.specs import Ellipse

    sc = Ellipse("x", 5.0, 1.0, 0.5, "y", 0.0).compile()
    assert set(gens(sc)[0].axes) == {"x", "y"}


def test_ellipse_compile_vertical_swaps_fast_slow():
    from scanspec.v2.specs import Ellipse

    # vertical=False: y is slow, x is fast → midpoints ordered by y first
    # vertical=True: x is slow, y is fast → midpoints ordered by x first
    # Both have same set of 5 points, just in different order
    sc_h = Ellipse("x", 5.0, 1.0, 0.5, "y", 0.0, vertical=False).compile()
    sc_v = Ellipse("x", 5.0, 1.0, 0.5, "y", 0.0, vertical=True).compile()
    assert gens(sc_h)[0].length == gens(sc_v)[0].length == 5


# ---------------------------------------------------------------------------
# Polygon — compile
# ---------------------------------------------------------------------------


def test_polygon_compile_returns_single_generator():
    from scanspec.v2.specs import Polygon

    sc = Polygon("x", "y", [(0, 0), (5, 0), (2.5, 4)], 1.0, 2.0).compile()
    assert len(gens(sc)) == 1


def test_polygon_compile_triangle_point_count():
    from scanspec.v2.specs import Polygon

    # Triangle (0,0),(5,0),(2.5,4), x_step=1, y_step=2
    # y rows: 0, 2, 4; x cols: 0..5 → 7 masked points
    sc = Polygon("x", "y", [(0, 0), (5, 0), (2.5, 4)], 1.0, 2.0).compile()
    assert gens(sc)[0].length == 7


def test_polygon_compile_axes_present():
    from scanspec.v2.specs import Polygon

    sc = Polygon("x", "y", [(0, 0), (5, 0), (2.5, 4)], 1.0, 2.0).compile()
    assert set(gens(sc)[0].axes) == {"x", "y"}


def test_polygon_compile_square_all_inside():
    from scanspec.v2.specs import Polygon

    # Unit square [0,1]×[0,1], step=0.5 → 3×3=9 grid, all 9 are inside the square
    vertices = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    sc = Polygon("x", "y", vertices, 0.5).compile()
    g = gens(sc)[0]
    idx = np.arange(g.length) + 0.5
    pts = g.setpoints(idx)
    x, y = pts["x"], pts["y"]
    assert np.all((x >= 0.0) & (x <= 1.0) & (y >= 0.0) & (y <= 1.0))
