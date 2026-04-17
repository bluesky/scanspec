"""Tests for scanspec2.specs — motion nodes, operator algebra, Acquire validation."""

from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from scanspec2.core import ContinuousStream, DetectorGroup, MonitorStream
from scanspec2.specs import (
    Acquire,
    AnySpec,
    Concat,
    Linspace,
    Product,
    Repeat,
    Snake,
    Static,
    Zip,
)

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


def test_acquire_json_round_trip():
    spec: Acquire[str, str, str] = Acquire(
        Linspace("x", 0.0, 10.0, 100),
        detectors=[
            DetectorGroup(
                exposures_per_collection=1,
                collections_per_event=1,
                livetime=0.003,
                deadtime=0.001,
                detectors=["saxs", "waxs"],
            )
        ],
        monitors=[MonitorStream("temp", "tc1")],
    )
    ta: TypeAdapter[AnySpec[Any, Any, Any]] = TypeAdapter(AnySpec)
    json_bytes = ta.dump_json(spec)
    restored = ta.validate_json(json_bytes)
    assert isinstance(restored, Acquire)
    assert restored.detectors[0].detectors == ["saxs", "waxs"]
    assert restored.monitors[0].name == "temp"


# ---------------------------------------------------------------------------
# Acquire — validation
# ---------------------------------------------------------------------------


def test_acquire_duplicate_detector_in_same_group():
    with pytest.raises(ValueError):
        Acquire(
            Linspace("x", 0.0, 1.0, 10),
            detectors=[
                DetectorGroup(1, 1, 0.01, 0.001, ["det1", "det1"]),
            ],
        )


def test_acquire_duplicate_detector_across_groups():
    with pytest.raises(ValueError, match="det1"):
        Acquire(
            Linspace("x", 0.0, 1.0, 10),
            detectors=[
                DetectorGroup(1, 1, 0.01, 0.001, ["det1"]),
                DetectorGroup(1, 1, 0.01, 0.001, ["det1"]),
            ],
        )


def test_acquire_duplicate_between_windowed_and_continuous():
    with pytest.raises(ValueError, match="cam1"):
        Acquire(
            Linspace("x", 0.0, 1.0, 10),
            detectors=[DetectorGroup(1, 1, 0.01, 0.001, ["cam1"])],
            continuous_streams=[
                ContinuousStream(
                    "cameras", [DetectorGroup(1, 1, 0.05, 0.005, ["cam1"])]
                )
            ],
        )


def test_acquire_duplicate_with_monitor():
    with pytest.raises(ValueError, match="tc1"):
        Acquire(
            Linspace("x", 0.0, 1.0, 10),
            detectors=[DetectorGroup(1, 1, 0.01, 0.001, ["tc1"])],
            monitors=[MonitorStream("temp", "tc1")],
        )


def test_acquire_valid_no_detectors():
    # Empty detectors list is allowed — validation only checks uniqueness.
    a: Acquire[str, Any, Any] = Acquire(Linspace("x", 0.0, 1.0, 10))
    assert a.detectors == ()


def test_acquire_defaults():
    a: Acquire[str, Any, Any] = Acquire(Linspace("x", 0.0, 1.0, 10))
    assert a.fly is False
    assert a.stream_name == "primary"
    assert a.continuous_streams == ()
    assert a.monitors == ()


def test_acquire_fly_true():
    a: Acquire[str, Any, Any] = Acquire(Linspace("x", 0.0, 1.0, 10), fly=True)
    assert a.fly is True


def test_acquire_frozen():
    a: Acquire[str, Any, Any] = Acquire(Linspace("x", 0.0, 1.0, 10))
    with pytest.raises(ValidationError):
        a.fly = True  # type: ignore[misc]
