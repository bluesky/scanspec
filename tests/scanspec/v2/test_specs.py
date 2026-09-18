"""Tests for scanspec.v2.specs — motion nodes, operator algebra, Sync validation."""

from typing import Any

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from scanspec.v2.core import (
    ContinuousStream,
    DetectorGroup,
    MonitorStream,
    TriggerFollower,
    TriggerGroup,
)
from scanspec.v2.specs import (
    AnySpec,
    Concat,
    ContinuousStreams,
    Ellipse,
    Linspace,
    Monitors,
    Polygon,
    Product,
    Range,
    Repeat,
    Snake,
    Spiral,
    Static,
    Sync,
    Zip,
)


def _first_gen_positions(spec: Any) -> dict[Any, np.ndarray]:
    """Public-API helper: midpoints of a spec's first compiled generator."""
    gen = spec.compile().generators[0]
    return gen.setpoints(np.arange(gen.length) + 0.5)


# ---------------------------------------------------------------------------
# Primitives — instantiation (positional and keyword)
# ---------------------------------------------------------------------------


def test_linspace_positional():
    ls = Linspace("x", 0.0, 10.0, 100)
    assert ls.axis == "x"
    assert ls.start == 0.0
    assert ls.stop == 10.0
    assert ls.num == 100
    assert ls.type == "Linspace"


def test_linspace_keyword():
    ls = Linspace(axis="x", start=0.0, stop=10.0, num=100)
    assert ls.num == 100


def test_static_positional():
    s = Static("y", 3.0)
    assert s.axis == "y"
    assert s.value == 3.0
    assert s.num == 1


def test_static_num():
    s = Static("y", 3.0, 5)
    assert s.num == 5


def test_repeat_positional():
    inner = Linspace("x", 0.0, 1.0, 10)
    r: Repeat[str, Any, Any] = Repeat(inner, 3)
    assert r.num == 3
    assert r.spec is inner


# ---------------------------------------------------------------------------
# Combinators
# ---------------------------------------------------------------------------


def test_snake_positional():
    ls = Linspace("x", 0.0, 1.0, 10)
    sn: Snake[str, Any, Any] = Snake(ls)
    assert sn.spec is ls
    assert sn.type == "Snake"


def test_product_positional():
    outer = Linspace("y", 0.0, 5.0, 50)
    inner = Linspace("x", 0.0, 10.0, 100)
    p: Product[str, Any, Any] = Product(outer, inner)
    assert p.outer is outer
    assert p.inner is inner
    assert p.type == "Product"


def test_zip_positional():
    a = Linspace("x", 0.0, 1.0, 10)
    b = Linspace("y", 0.0, 1.0, 10)
    z: Zip[str, Any, Any] = Zip(a, b)
    assert z.left is a
    assert z.right is b


def test_concat_positional():
    a = Linspace("x", 0.0, 5.0, 5)
    b = Linspace("x", 5.0, 10.0, 5)
    c: Concat[str, Any, Any] = Concat(a, b)
    assert c.left is a
    assert c.right is b


# ---------------------------------------------------------------------------
# Operator algebra
# ---------------------------------------------------------------------------


def test_mul_produces_product():
    y = Linspace("y", 0.0, 5.0, 50)
    x = Linspace("x", 0.0, 10.0, 100)
    p = y * x
    assert isinstance(p, Product)
    assert p.outer is y
    assert p.inner is x


def test_invert_produces_snake():
    x = Linspace("x", 0.0, 10.0, 100)
    sn = ~x
    assert isinstance(sn, Snake)
    assert sn.spec is x


def test_zip_method():
    a = Linspace("x", 0.0, 1.0, 10)
    b = Linspace("y", 0.0, 1.0, 10)
    z = a.zip(b)
    assert isinstance(z, Zip)
    assert z.left is a
    assert z.right is b


def test_concat_method():
    a = Linspace("x", 0.0, 5.0, 5)
    b = Linspace("x", 5.0, 10.0, 5)
    c = a.concat(b)
    assert isinstance(c, Concat)
    assert c.left is a
    assert c.right is b


def test_chained_operators():
    y = Linspace("y", 0.0, 5.0, 50)
    x = Linspace("x", 0.0, 10.0, 100)
    expr = y * ~x
    assert isinstance(expr, Product)
    assert isinstance(expr.inner, Snake)
    assert expr.inner.spec is x


# ---------------------------------------------------------------------------
# JSON round-trips (motion nodes)
# ---------------------------------------------------------------------------


def test_linspace_json_round_trip():
    ls = Linspace("x", 0.0, 10.0, 100)
    ta: TypeAdapter[AnySpec[Any, Any, Any]] = TypeAdapter(AnySpec)
    restored = ta.validate_json(ta.dump_json(ls))
    assert isinstance(restored, Linspace)
    assert restored.axis == "x"
    assert restored.num == 100


def test_product_json_round_trip():
    p = Linspace("y", 0.0, 5.0, 50) * ~Linspace("x", 0.0, 10.0, 100)
    ta: TypeAdapter[AnySpec[Any, Any, Any]] = TypeAdapter(AnySpec)
    restored = ta.validate_json(ta.dump_json(p))
    assert isinstance(restored, Product)
    assert isinstance(restored.inner, Snake)
    assert isinstance(restored.inner.spec, Linspace)


@pytest.mark.parametrize(
    ("spec", "expected_type"),
    [
        pytest.param(Static("y", 3.0), Static, id="Static"),
        pytest.param(Range("x", 0.0, 1.0, 0.25), Range, id="Range"),
        pytest.param(Spiral("x", 0.0, 5.0, 2.0, "y", 10.0, 10.0), Spiral, id="Spiral"),
        pytest.param(Ellipse("x", 5.0, 1.0, 0.5, "y", 0.0), Ellipse, id="Ellipse"),
        pytest.param(
            Polygon("x", "y", [(0.0, 0.0), (5.0, 0.0), (2.5, 4.0)], 1.0),
            Polygon,
            id="Polygon",
        ),
        pytest.param(
            Linspace("x", 0.0, 1.0, 10).zip(Linspace("y", 0.0, 1.0, 10)),
            Zip,
            id="Zip",
        ),
        pytest.param(
            Linspace("x", 0.0, 5.0, 5).concat(Linspace("x", 5.0, 10.0, 5)),
            Concat,
            id="Concat",
        ),
        pytest.param(Repeat(Linspace("x", 0.0, 1.0, 10), 3), Repeat, id="Repeat"),
        pytest.param(Snake(Linspace("x", 0.0, 1.0, 10)), Snake, id="Snake"),
    ],
)
def test_motion_spec_json_round_trip(
    spec: AnySpec[Any, Any, Any], expected_type: type
) -> None:
    ta: TypeAdapter[AnySpec[Any, Any, Any]] = TypeAdapter(AnySpec)
    restored = ta.validate_json(ta.dump_json(spec))
    assert isinstance(restored, expected_type)
    assert restored == spec


def test_sync_json_round_trip():
    spec: Monitors[str, str, str] = Monitors(
        Sync(
            Linspace("x", 0.0, 10.0, 100),
            trigger_group=TriggerGroup(
                detectors=frozenset({"saxs", "waxs"}),
                exposures_per_collection=1,
                collections_per_event=1,
                livetime=0.003,
                deadtime=0.001,
            ),
        ),
        monitors=[MonitorStream("temp", "tc1")],
    )
    ta: TypeAdapter[AnySpec[Any, Any, Any]] = TypeAdapter(AnySpec)
    json_bytes = ta.dump_json(spec)
    restored = ta.validate_json(json_bytes)
    assert isinstance(restored, Monitors)
    assert isinstance(restored.spec, Sync)
    assert isinstance(restored.spec.trigger_group, TriggerGroup)
    assert restored.spec.trigger_group.detectors == frozenset({"saxs", "waxs"})
    assert restored.monitors[0].name == "temp"


def test_sync_json_round_trip_full():
    """Round-trip a wrapped Sync exercising every optional field, including a
    trigger_group with a non-empty followers list."""
    spec: ContinuousStreams[str, str, str] = ContinuousStreams(
        Monitors(
            Sync(
                Linspace("x", 0.0, 10.0, 100),
                fly=True,
                trigger_group=TriggerGroup(
                    detectors=frozenset({"saxs"}),
                    exposures_per_collection=1,
                    collections_per_event=1,
                    livetime=0.003,
                    deadtime=0.001,
                    followers=[
                        TriggerFollower(
                            detectors=frozenset({"encoder"}),
                            exposures_per_collection=10,
                            collections_per_event=1,
                            livetime=0.0003,
                            deadtime=0.0,
                            repeats=10,
                        ),
                    ],
                ),
                duration=0.5,
            ),
            monitors=[MonitorStream("temp", "tc1")],
        ),
        continuous_streams=[
            ContinuousStream(
                "cameras",
                [DetectorGroup(1, 1, 0.048, 0.001, ["front_cam", "side_cam"])],
            ),
        ],
    )
    ta: TypeAdapter[AnySpec[Any, Any, Any]] = TypeAdapter(AnySpec)
    restored = ta.validate_json(ta.dump_json(spec))
    assert isinstance(restored, ContinuousStreams)
    assert restored == spec
    assert isinstance(restored.spec, Monitors)
    assert isinstance(restored.spec.spec, Sync)
    assert isinstance(restored.spec.spec.trigger_group, TriggerGroup)
    assert restored.spec.spec.trigger_group.followers[0].detectors == frozenset(
        {"encoder"}
    )
    assert restored.continuous_streams[0].name == "cameras"
    assert restored.spec.spec.duration == 0.5


# ---------------------------------------------------------------------------
# Sync — validation
# ---------------------------------------------------------------------------


def test_sync_duplicate_detector_in_same_continuous_group():
    """A duplicate name within one DetectorGroup's list still raises.

    Only reachable via continuous_streams now: the windowed authoring path's
    TriggerGroup.detectors is a frozenset, which structurally dedupes rather
    than preserving a duplicate to be caught by validation. Raised by
    ContinuousStreams.compile() (ADR 0009) -- continuous_streams is no
    longer a Sync field, so this is only caught at compile() time now, not
    at construction time.
    """
    spec = ContinuousStreams(
        Sync(Linspace("x", 0.0, 1.0, 10)),
        continuous_streams=[
            ContinuousStream(
                "cameras", [DetectorGroup(1, 1, 0.01, 0.001, ["det1", "det1"])]
            )
        ],
    )
    with pytest.raises(ValueError):
        spec.compile()


def test_sync_duplicate_detector_across_groups():
    # Raised by TriggerGroup's own disjointness check at construction time,
    # before Sync(...) is even reached -- unaffected by the ADR 0009 pull-out.
    with pytest.raises(ValueError, match="det1"):
        Sync(
            Linspace("x", 0.0, 1.0, 10),
            trigger_group=TriggerGroup(
                detectors=frozenset({"det1"}),
                exposures_per_collection=1,
                collections_per_event=1,
                livetime=0.01,
                deadtime=0.001,
                followers=[
                    TriggerFollower(
                        detectors=frozenset({"det1"}),
                        exposures_per_collection=1,
                        collections_per_event=1,
                        livetime=0.01,
                        deadtime=0.001,
                        repeats=1,
                    ),
                ],
            ),
        )


def test_sync_duplicate_between_windowed_and_continuous():
    spec = ContinuousStreams(
        Sync(
            Linspace("x", 0.0, 1.0, 10),
            trigger_group=TriggerGroup(
                detectors=frozenset({"cam1"}),
                exposures_per_collection=1,
                collections_per_event=1,
                livetime=0.01,
                deadtime=0.001,
            ),
        ),
        continuous_streams=[
            ContinuousStream("cameras", [DetectorGroup(1, 1, 0.05, 0.005, ["cam1"])])
        ],
    )
    with pytest.raises(ValueError, match="cam1"):
        spec.compile()


def test_sync_duplicate_with_monitor():
    # Explicit annotation: MonitorT can't be inferred from Sync's own call
    # (monitors is no longer a Sync field, ADR 0009) -- without it, Sync's
    # MonitorT defaults to Never, conflicting with Monitors' str-typed
    # monitors= below.
    sync: Sync[str, str, str] = Sync(
        Linspace("x", 0.0, 1.0, 10),
        trigger_group=TriggerGroup(
            detectors=frozenset({"tc1"}),
            exposures_per_collection=1,
            collections_per_event=1,
            livetime=0.01,
            deadtime=0.001,
        ),
    )
    spec = Monitors(sync, monitors=[MonitorStream("temp", "tc1")])
    with pytest.raises(ValueError, match="tc1"):
        spec.compile()


def test_sync_valid_no_trigger_group():
    # No trigger_group is allowed -- validation only checks uniqueness.
    a: Sync[str, Any, Any] = Sync(Linspace("x", 0.0, 1.0, 10))
    assert a.trigger_group is None


def test_sync_defaults():
    a: Sync[str, Any, Any] = Sync(Linspace("x", 0.0, 1.0, 10))
    assert a.fly is False
    assert a.stream_name == "primary"


def test_monitors_defaults():
    m: Monitors[str, Any, Any] = Monitors(Sync(Linspace("x", 0.0, 1.0, 10)))
    assert m.monitors == ()


def test_continuous_streams_defaults():
    cs: ContinuousStreams[str, Any, Any] = ContinuousStreams(
        Sync(Linspace("x", 0.0, 1.0, 10))
    )
    assert cs.continuous_streams == ()


def test_sync_fly_true():
    a: Sync[str, Any, Any] = Sync(Linspace("x", 0.0, 1.0, 10), fly=True)
    assert a.fly is True


def test_sync_frozen():
    a: Sync[str, Any, Any] = Sync(Linspace("x", 0.0, 1.0, 10))
    with pytest.raises(ValidationError):
        a.fly = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Linspace.bounded — construction
# ---------------------------------------------------------------------------


def test_linspace_bounded_one_point():
    inst = Linspace.bounded("x", 0.0, 1.0, 1)
    assert isinstance(inst, Linspace)
    assert inst.axis == "x"
    assert inst.num == 1
    assert inst.start == 0.5
    # stop encodes the step size for num=1: stop = upper + half_step = 1.5
    assert inst.stop == 1.5


def test_linspace_bounded_many_points():
    inst = Linspace.bounded("x", 0.0, 1.0, 4)
    assert isinstance(inst, Linspace)
    assert inst.start == 0.125
    assert inst.stop == 0.875
    assert inst.num == 4


def test_linspace_bounded_symmetric():
    inst = Linspace.bounded("x", 3.0, 7.0, 2)
    assert isinstance(inst, Linspace)
    assert inst.start == 4.0
    assert inst.stop == 6.0
    assert inst.num == 2


# ---------------------------------------------------------------------------
# Range — construction and validation
# ---------------------------------------------------------------------------


def test_range_positional():
    from scanspec.v2.specs import Range

    r = Range("x", 0.0, 1.0, 0.25)
    assert r.axis == "x"
    assert r.start == 0.0
    assert r.stop == 1.0
    assert r.step == 0.25


def test_range_keyword():
    from scanspec.v2.specs import Range

    r = Range(axis="x", start=0.0, stop=10.0, step=2.0)
    assert r.step == 2.0


def test_range_zero_step_raises():
    from scanspec.v2.specs import Range

    with pytest.raises(ValueError):
        Range("x", 0.0, 1.0, 0.0)


def test_range_negative_step_raises():
    from scanspec.v2.specs import Range

    with pytest.raises(ValueError):
        Range("x", 0.0, 1.0, -0.5)


def test_range_type_field():
    from scanspec.v2.specs import Range

    r = Range("x", 0.0, 1.0, 0.5)
    assert r.type == "Range"


# ---------------------------------------------------------------------------
# Range.bounded — construction
# ---------------------------------------------------------------------------


def test_range_bounded_many_points():
    from scanspec.v2.specs import Range

    inst = Range.bounded("x", 0.0, 1.0, 0.25)
    assert isinstance(inst, Range)
    assert inst.start == 0.125
    assert inst.stop == 0.875
    assert inst.step == 0.25


@pytest.mark.parametrize(
    "lower,upper,step,expected_start",
    [
        (0.0, 1.0, 0.8, 0.4),  # step smaller than range → one frame
        (0.0, 1.0, 1.0, 0.5),  # step equals range → one frame
        (0.0, 1.0, 1.2, 0.5),  # step larger than range → clamped to one frame
    ],
)
def test_range_bounded_one_point(
    lower: float, upper: float, step: float, expected_start: float
) -> None:
    from scanspec.v2.specs import Range

    inst = Range.bounded("x", lower, upper, step)
    assert isinstance(inst, Range)
    assert inst.start == expected_start


def test_range_bounded_lower_equals_upper():
    """lower == upper must not crash and must produce a single point."""
    from scanspec.v2.specs import Range

    inst = Range.bounded("x", 5.0, 5.0, 0.5)
    assert isinstance(inst, Range)
    assert inst.start == 5.0
    assert inst.stop == 5.0
    assert inst.step == 0.5  # step kept as-is, not clamped to 0


def test_line_is_linspace():
    from scanspec.v2.specs import Line

    assert Line is Linspace


def test_line_instantiation():
    from scanspec.v2.specs import Line

    ln = Line("x", 0.0, 10.0, 5)
    assert isinstance(ln, Linspace)
    assert ln.axis == "x"
    assert ln.num == 5


# ---------------------------------------------------------------------------
# Spiral — construction and validation
# ---------------------------------------------------------------------------


def test_spiral_positional():
    from scanspec.v2.specs import Spiral

    s = Spiral("x", 0.0, 5.0, 2.0, "y", 10.0, 10.0)
    assert s.x_diameter == 5.0
    assert s.x_step == 2.0
    assert s.y_diameter == 10.0


def test_spiral_y_diameter_defaults_to_x_diameter():
    from scanspec.v2.specs import Spiral

    s_implicit = Spiral("x", 0.0, 5.0, 2.0, "y", 10.0)
    s_explicit = Spiral("x", 0.0, 5.0, 2.0, "y", 10.0, y_diameter=5.0)
    assert s_implicit.y_diameter is None
    implicit_pos = _first_gen_positions(s_implicit)
    explicit_pos = _first_gen_positions(s_explicit)
    for axis, arr in implicit_pos.items():
        assert np.array_equal(arr, explicit_pos[axis])


@pytest.mark.parametrize("bad_value", [0.0, -1.0])
def test_spiral_x_diameter_not_positive_raises(bad_value: float):
    from scanspec.v2.specs import Spiral

    with pytest.raises(ValueError):
        Spiral("x", 0.0, bad_value, 2.0, "y", 10.0)


@pytest.mark.parametrize("bad_value", [0.0, -1.0])
def test_spiral_x_step_not_positive_raises(bad_value: float):
    from scanspec.v2.specs import Spiral

    with pytest.raises(ValueError):
        Spiral("x", 0.0, 5.0, bad_value, "y", 10.0)


@pytest.mark.parametrize("bad_value", [0.0, -1.0])
def test_spiral_y_diameter_not_positive_raises(bad_value: float):
    from scanspec.v2.specs import Spiral

    with pytest.raises(ValueError):
        Spiral("x", 0.0, 5.0, 2.0, "y", 10.0, y_diameter=bad_value)


# ---------------------------------------------------------------------------
# Ellipse — construction
# ---------------------------------------------------------------------------


def test_ellipse_positional():
    from scanspec.v2.specs import Ellipse

    e = Ellipse("x", 5.0, 1.0, 0.5, "y", 0.0)
    assert e.x_axis == "x"
    assert e.x_centre == 5.0
    assert e.x_diameter == 1.0
    assert e.x_step == 0.5
    assert e.y_axis == "y"
    assert e.y_centre == 0.0


def test_ellipse_y_diameter_defaults_to_x_diameter():
    from scanspec.v2.specs import Ellipse

    e_implicit = Ellipse("x", 0.0, 2.0, 0.5, "y", 0.0)
    e_explicit = Ellipse("x", 0.0, 2.0, 0.5, "y", 0.0, y_diameter=2.0)
    assert e_implicit.y_diameter is None
    implicit_pos = _first_gen_positions(e_implicit)
    explicit_pos = _first_gen_positions(e_explicit)
    for axis, arr in implicit_pos.items():
        assert np.array_equal(arr, explicit_pos[axis])


def test_ellipse_y_step_defaults_to_x_step():
    from scanspec.v2.specs import Ellipse

    e_implicit = Ellipse("x", 0.0, 2.0, 0.5, "y", 0.0)
    e_explicit = Ellipse("x", 0.0, 2.0, 0.5, "y", 0.0, y_step=0.5)
    assert e_implicit.y_step is None
    implicit_pos = _first_gen_positions(e_implicit)
    explicit_pos = _first_gen_positions(e_explicit)
    for axis, arr in implicit_pos.items():
        assert np.array_equal(arr, explicit_pos[axis])


def test_ellipse_vertical_default():
    from scanspec.v2.specs import Ellipse

    e = Ellipse("x", 0.0, 2.0, 0.5, "y", 0.0)
    assert e.vertical is False


def test_ellipse_explicit_y_diameter_and_y_step():
    from scanspec.v2.specs import Ellipse

    e = Ellipse("x", 0.0, 4.0, 1.0, "y", 0.0, y_diameter=2.0, y_step=0.5)
    assert e.y_diameter == 2.0
    assert e.y_step == 0.5


@pytest.mark.parametrize("bad_value", [0.0, -1.0])
def test_ellipse_x_step_not_positive_raises(bad_value: float):
    from scanspec.v2.specs import Ellipse

    with pytest.raises(ValueError):
        Ellipse("x", 0.0, 2.0, bad_value, "y", 0.0)


@pytest.mark.parametrize("bad_value", [0.0, -1.0])
def test_ellipse_x_diameter_not_positive_raises(bad_value: float):
    from scanspec.v2.specs import Ellipse

    with pytest.raises(ValueError):
        Ellipse("x", 0.0, bad_value, 1.0, "y", 0.0)


@pytest.mark.parametrize("bad_value", [0.0, -1.0])
def test_ellipse_y_diameter_not_positive_raises(bad_value: float):
    from scanspec.v2.specs import Ellipse

    with pytest.raises(ValueError):
        Ellipse("x", 0.0, 2.0, 1.0, "y", 0.0, y_diameter=bad_value)


@pytest.mark.parametrize("bad_value", [0.0, -1.0])
def test_ellipse_y_step_not_positive_raises(bad_value: float):
    from scanspec.v2.specs import Ellipse

    with pytest.raises(ValueError):
        Ellipse("x", 0.0, 2.0, 1.0, "y", 0.0, y_step=bad_value)


# ---------------------------------------------------------------------------
# Polygon — construction
# ---------------------------------------------------------------------------


def test_polygon_positional():
    from scanspec.v2.specs import Polygon

    vertices = [(0.0, 0.0), (5.0, 0.0), (2.5, 4.0)]
    p = Polygon("x", "y", vertices, 1.0)
    assert p.x_axis == "x"
    assert p.y_axis == "y"
    assert p.vertices == vertices
    assert p.x_step == 1.0


def test_polygon_y_step_defaults_to_x_step():
    from scanspec.v2.specs import Polygon

    p_implicit = Polygon("x", "y", [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)], 0.25)
    p_explicit = Polygon(
        "x", "y", [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)], 0.25, y_step=0.25
    )
    assert p_implicit.y_step is None
    implicit_pos = _first_gen_positions(p_implicit)
    explicit_pos = _first_gen_positions(p_explicit)
    for axis, arr in implicit_pos.items():
        assert np.array_equal(arr, explicit_pos[axis])


def test_polygon_explicit_y_step():
    from scanspec.v2.specs import Polygon

    p = Polygon("x", "y", [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)], 0.5, 0.25)
    assert p.x_step == 0.5
    assert p.y_step == 0.25


def test_polygon_vertical_default():
    from scanspec.v2.specs import Polygon

    p = Polygon("x", "y", [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)], 0.25)
    assert p.vertical is False


@pytest.mark.parametrize("bad_value", [0.0, -1.0])
def test_polygon_x_step_not_positive_raises(bad_value: float):
    from scanspec.v2.specs import Polygon

    with pytest.raises(ValueError):
        Polygon("x", "y", [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)], bad_value)


@pytest.mark.parametrize("bad_value", [0.0, -1.0])
def test_polygon_y_step_not_positive_raises(bad_value: float):
    from scanspec.v2.specs import Polygon

    with pytest.raises(ValueError):
        Polygon("x", "y", [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)], 1.0, y_step=bad_value)
