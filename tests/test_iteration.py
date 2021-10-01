from typing import Any

import pytest

from scanspec.core import Frames, Path
from scanspec.specs import Line, Spec


def test_line_path() -> None:
    x = "x"
    inst: Spec[Any] = Line(x, 0, 1, 5)
    dims = inst.calculate()
    path = Path(dims)
    assert len(path) == 5
    dim: Frames[Any] = path.consume()
    assert dim.midpoints == {x: pytest.approx([0, 0.25, 0.5, 0.75, 1.0])}
    assert len(path) == 0


def test_line_midpoints() -> None:
    x: str = "x"
    inst: Spec[Any] = Line(x, 0, 1, 5)
    it = inst.midpoints()
    assert it.axes == [x]
    assert len(it) == 5
    midpoints = [d[x] for d in it]
    assert midpoints == pytest.approx([0, 0.25, 0.5, 0.75, 1.0])
