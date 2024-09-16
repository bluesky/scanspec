import json
import pathlib
import subprocess
import sys
from typing import cast
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from click.testing import CliRunner
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.art3d import Line3D  # type: ignore

from scanspec import __version__, cli
from scanspec.plot import Arrow3D

from . import approx


def assert_min_max_2d(
    line: Line2D,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    length: float | None = None,
):
    line_data = cast(npt.NDArray[np.float64], line.get_data())
    if length is not None:
        assert len(line_data[0]) == length
    mins = np.min(line_data, axis=1)
    maxs = np.max(line_data, axis=1)
    assert list(zip(mins, maxs, strict=False)) == [
        approx([xmin, xmax]),
        approx([ymin, ymax]),
    ]


def assert_min_max_3d(
    line: Line3D,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
    length: float | None = None,
):
    data_3d = cast(npt.NDArray[np.float64], line.get_data_3d())
    if length is not None:
        assert len(data_3d[0]) == length
    mins = np.min(data_3d, axis=1)
    maxs = np.max(data_3d, axis=1)
    assert list(zip(mins, maxs, strict=False)) == [
        approx([xmin, xmax]),
        approx([ymin, ymax]),
        approx([zmin, zmax]),
    ]


def assert_3d_arrow(
    artist: Line3D,
    x: float,
    y: float,
    z: float,
):
    assert isinstance(artist, Arrow3D)
    assert artist.verts3d[0][1] == approx(x)
    assert artist.verts3d[1][1] == approx(y)
    assert artist.verts3d[2][1] == approx(z)


def test_plot_1D_line() -> None:
    runner = CliRunner()
    spec = 'Line("x", 1, 2, 2)'
    with patch("scanspec.plot.plt.show"):
        result = runner.invoke(cli.cli, ["plot", spec])
    assert result.stdout == ""
    axes = plt.gcf().axes[0]
    lines = axes.lines
    assert len(lines) == 4
    # Splines
    assert_min_max_2d(lines[0], 0.5, 1.5, 0, 0)
    assert_min_max_2d(lines[1], 1.5, 2.5, 0, 0)
    # Capture points
    assert_min_max_2d(lines[2], 1, 2, 0, 0, length=2)
    # End
    assert_min_max_2d(lines[3], 2.5, 2.5, 0, 0)
    # Arrows
    texts = cast(list[Annotation], axes.texts)
    assert len(texts) == 1
    assert tuple(texts[0].xy) == (0.5, 0)


def test_plot_1D_line_snake_repeat() -> None:
    runner = CliRunner()
    spec = '2 * ~Line.bounded("x", 1, 2, 1)'
    with patch("scanspec.plot.plt.show"):
        result = runner.invoke(cli.cli, ["plot", spec])
    assert result.stdout == ""
    axes = plt.gcf().axes[0]
    lines = axes.lines
    assert len(lines) == 5
    # First repeat
    assert_min_max_2d(lines[0], 1, 2, 0, 0)
    # Turnaround
    assert_min_max_2d(lines[1], 2.0, 2.005585, 0, 0)
    # Second repeat
    assert_min_max_2d(lines[2], 1, 2, 0, 0)
    # Capture points
    assert_min_max_2d(lines[3], 1.5, 1.5, 0, 0, length=2)
    # End
    assert_min_max_2d(lines[4], 1, 1, 0, 0)
    # Arrows
    texts = cast(list[Annotation], axes.texts)
    assert len(texts) == 2
    assert tuple(texts[0].xy) == (1, 0)
    assert tuple(texts[1].xy) == approx([2, 0])


def test_plot_1D_step() -> None:
    runner = CliRunner()
    spec = 'step(Line("x", 1, 2, 2), 0.1)'
    with patch("scanspec.plot.plt.show"):
        result = runner.invoke(cli.cli, ["plot", spec])
    assert result.stdout == ""
    axes = plt.gcf().axes[0]
    lines = axes.lines
    assert len(lines) == 4
    # Start marker
    assert_min_max_2d(lines[0], 1, 1, 0, 0)
    # Step
    assert_min_max_2d(lines[1], 1, 2, 0, 0)
    # Capture points
    assert_min_max_2d(lines[2], 1, 2, 0, 0, length=2)
    # End
    assert_min_max_2d(lines[3], 2, 2, 0, 0)
    # Arrows
    texts = cast(list[Annotation], axes.texts)
    assert len(texts) == 1
    assert tuple(texts[0].xy) == (2, 0)


def test_plot_2D_line() -> None:
    runner = CliRunner()
    spec = 'Line("y", 2, 3, 2) * Snake(Line("x", 1, 2, 2))'
    with patch("scanspec.plot.plt.show"):
        result = runner.invoke(cli.cli, ["plot", spec])
    assert result.exit_code == 0
    axes = plt.gcf().axes[0]
    lines = axes.lines
    assert len(lines) == 7
    # First row
    assert_min_max_2d(lines[0], 0.5, 1.5, 2, 2)
    assert_min_max_2d(lines[1], 1.5, 2.5, 2, 2)
    # Turnaround
    assert_min_max_2d(lines[2], 2.5, 2.7537562, 1.9999880, 3.0000119)
    # Second row
    assert_min_max_2d(lines[3], 1.5, 2.5, 3, 3)
    assert_min_max_2d(lines[4], 0.5, 1.5, 3, 3)
    # Capture points
    assert_min_max_2d(lines[5], 1, 2, 2, 3, length=4)
    # End
    assert_min_max_2d(lines[6], 0.5, 0.5, 3, 3)
    # Arrows
    texts = cast(list[Annotation], axes.texts)
    assert len(texts) == 2
    assert tuple(texts[0].xy) == (0.5, 2)
    assert tuple(texts[1].xy) == approx([2.5, 3])


def test_plot_2D_line_rect_region() -> None:
    runner = CliRunner()
    spec = "Line(y, 1, 3, 5) * Line(x, 0, 2, 3) & Rectangle(x, y, 0, 1.1, 1.5, 2.1, 30)"
    with patch("scanspec.plot.plt.show"):
        result = runner.invoke(cli.cli, ["plot", spec])
    assert result.exit_code == 0
    axes = plt.gcf().axes[0]
    lines = axes.lines
    assert len(lines) == 6
    # First row
    assert_min_max_2d(lines[0], -0.5, 0.5, 1.5, 1.5)
    # Turnaround
    assert_min_max_2d(lines[1], -0.6071045, 0.60710456, 1.4999969, 2.000003)
    # Second row
    assert_min_max_2d(lines[2], -0.5, 0.5, 2, 2)
    assert_min_max_2d(lines[3], 0.5, 1.5, 2, 2)
    # Capture points
    assert_min_max_2d(lines[4], 0, 1, 1.5, 2, length=3)
    # End
    assert_min_max_2d(lines[5], 1.5, 1.5, 2, 2)
    # Arrows
    texts = cast(list[Annotation], axes.texts)
    assert len(texts) == 2
    assert tuple(texts[0].xy) == (-0.5, 1.5)
    assert tuple(texts[1].xy) == (-0.5, 2)
    # Regions
    patches = axes.patches
    assert len(patches) == 1
    p = patches[0]
    assert isinstance(p, Rectangle)
    assert p.get_xy() == (0, 1.1)
    assert p.get_height() == 1.0
    assert p.get_width() == 1.5
    assert p.angle == 30


def test_plot_3D_line() -> None:
    runner = CliRunner()
    spec = 'Snake(Line("z", 5, 6, 2) * Line("y", 2, 3, 2) * Line("x", 1, 2, 2))'
    with patch("scanspec.plot.plt.show"):
        result = runner.invoke(cli.cli, ["plot", spec])
    assert result.exit_code == 0
    axes = plt.gcf().axes[0]
    lines = cast(list[Line3D], axes.lines)
    assert len(lines) == 13
    # First grid
    # First row
    assert_min_max_3d(lines[0], 0.5, 1.5, 2, 2, 5, 5)
    assert_min_max_3d(lines[1], 1.5, 2.5, 2, 2, 5, 5)
    # Turnaround
    assert_min_max_3d(lines[2], 2.5, 2.7537562, 1.9999880, 3.0000119, 5, 5)
    # Second row
    assert_min_max_3d(lines[3], 1.5, 2.5, 3, 3, 5, 5)
    assert_min_max_3d(lines[4], 0.5, 1.5, 3, 3, 5, 5)
    # Turnaround
    assert_min_max_3d(lines[5], 0.2462437, 0.5, 3, 3, 4.9999880, 6.0000178)
    # Second grid
    # First row
    assert_min_max_3d(lines[6], 0.5, 1.5, 3, 3, 6, 6)
    assert_min_max_3d(lines[7], 1.5, 2.5, 3, 3, 6, 6)
    # Turnaround
    assert_min_max_3d(lines[8], 2.5, 2.7537562, 1.9999880, 3.0000119, 6, 6)
    # Second row
    assert_min_max_3d(lines[9], 1.5, 2.5, 2, 2, 6, 6)
    assert_min_max_3d(lines[10], 0.5, 1.5, 2, 2, 6, 6)
    # Capture points
    assert_min_max_3d(lines[11], 1, 2, 2, 3, 5, 6, length=8)
    # End
    assert_min_max_3d(lines[12], 0.5, 0.5, 2, 2, 6, 6)
    # Arrows
    extra_artists = axes.get_children()

    arrow_artists = cast(
        list[Line3D],
        list(
            filter(
                lambda artist: isinstance(artist, Arrow3D)
                and artist.get_visible()
                and artist.get_in_layout(),
                extra_artists,
            )
        ),
    )
    assert len(arrow_artists) == 4
    assert_3d_arrow(arrow_artists[0], 0.5, 2, 5)
    assert_3d_arrow(arrow_artists[1], 2.5, 3, 5)
    assert_3d_arrow(arrow_artists[2], 0.5, 3, 6)
    assert_3d_arrow(arrow_artists[3], 2.5, 2, 6)


def test_schema() -> None:
    # If this test fails, regenerate the schema by running
    # scanspec schema > schema.json

    runner = CliRunner()
    result = runner.invoke(cli.schema)
    assert result.exit_code == 0
    schema_path = pathlib.Path(__file__).resolve().parent.parent / "schema.json"
    with open(schema_path) as file:
        data = json.load(file)
        assert data == json.loads(result.output)


def test_cli_version():
    cmd = [sys.executable, "-m", "scanspec", "--version"]
    assert subprocess.check_output(cmd).decode().strip() == __version__
