from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from click.testing import CliRunner

from scanspec import cli


def assert_min_max_2d(line, xmin, xmax, ymin, ymax):
    mins = np.min(line.get_data(), axis=1)
    maxs = np.max(line.get_data(), axis=1)
    assert list(zip(mins, maxs)) == [
        pytest.approx([xmin, xmax]),
        pytest.approx([ymin, ymax]),
    ]


def assert_min_max_3d(line, xmin, xmax, ymin, ymax, zmin, zmax):
    mins = np.min(line.get_data_3d(), axis=1)
    maxs = np.max(line.get_data_3d(), axis=1)
    assert list(zip(mins, maxs)) == [
        pytest.approx([xmin, xmax]),
        pytest.approx([ymin, ymax]),
        pytest.approx([zmin, zmax]),
    ]


def test_plot_1D_line() -> None:
    runner = CliRunner()
    f = plt.figure()
    spec = 'Line("x", 1, 2, 2)'
    with patch("scanspec.plot.plt.show"):
        result = runner.invoke(cli.cli, ["plot", spec])
    assert result.stdout == ""
    lines = f.axes[0].lines
    assert len(lines) == 5
    # Splines
    assert_min_max_2d(lines[0], 0.5, 1.5, 0, 0)
    assert_min_max_2d(lines[1], 1.5, 2.5, 0, 0)
    # Arrow
    assert_min_max_2d(lines[2], 0.5, 0.5, 0, 0)
    # End
    assert_min_max_2d(lines[3], 2.5, 2.5, 0, 0)
    # Capture points
    assert len(lines[-1].get_data()[0]) == 2
    assert_min_max_2d(lines[-1], 1, 2, 0, 0)


def test_plot_2D_line() -> None:
    runner = CliRunner()
    f = plt.figure()
    spec = 'Line("y", 2, 3, 2) * Snake(Line("x", 1, 2, 2))'
    with patch("scanspec.plot.plt.show"):
        result = runner.invoke(cli.cli, ["plot", spec])
    assert result.exit_code == 0
    lines = f.axes[0].lines
    assert len(lines) == 9
    # First row
    assert_min_max_2d(lines[0], 0.5, 1.5, 2, 2)
    assert_min_max_2d(lines[1], 1.5, 2.5, 2, 2)
    # Arrow
    assert_min_max_2d(lines[2], 0.5, 0.5, 2, 2)
    # Turnaround
    assert_min_max_2d(lines[3], 2.5, 2.7537562, 1.9999880, 3.0000119)
    # Second row
    assert_min_max_2d(lines[4], 1.5, 2.5, 3, 3)
    assert_min_max_2d(lines[5], 0.5, 1.5, 3, 3)
    # Arrow
    assert_min_max_2d(lines[6], 2.5, 2.5, 3, 3)
    # End
    assert_min_max_2d(lines[7], 0.5, 0.5, 3, 3)
    # Capture points
    assert len(lines[-1].get_data()[0]) == 4
    assert_min_max_2d(lines[-1], 1, 2, 2, 3)


def test_plot_3D_line() -> None:
    runner = CliRunner()
    f = plt.figure()
    spec = 'Snake(Line("z", 5, 6, 2) * Line("y", 2, 3, 2) * Line("x", 1, 2, 2))'
    with patch("scanspec.plot.plt.show"):
        result = runner.invoke(cli.cli, ["plot", spec],)
    assert result.exit_code == 0
    lines = f.axes[0].lines
    assert len(lines) == 17
    # First grid
    # First row
    assert_min_max_3d(lines[0], 0.5, 1.5, 2, 2, 5, 5)
    assert_min_max_3d(lines[1], 1.5, 2.5, 2, 2, 5, 5)
    # Arrow
    assert_min_max_3d(lines[2], 0.5, 0.5, 2, 2, 5, 5)
    # Turnaround
    assert_min_max_3d(lines[3], 2.5, 2.7537562, 1.9999880, 3.0000119, 5, 5)
    # Second row
    assert_min_max_3d(lines[4], 1.5, 2.5, 3, 3, 5, 5)
    assert_min_max_3d(lines[5], 0.5, 1.5, 3, 3, 5, 5)
    # Arrow
    assert_min_max_3d(lines[6], 2.5, 2.5, 3, 3, 5, 5)
    # Turnaround
    assert_min_max_3d(lines[7], 0.2462437, 0.5, 3, 3, 4.9999880, 6.0000178)
    # Second grid
    # First row
    assert_min_max_3d(lines[8], 0.5, 1.5, 3, 3, 6, 6)
    assert_min_max_3d(lines[9], 1.5, 2.5, 3, 3, 6, 6)
    # Arrow
    assert_min_max_3d(lines[10], 0.5, 0.5, 3, 3, 6, 6)
    # Turnaround
    assert_min_max_3d(lines[11], 2.5, 2.7537562, 1.9999880, 3.0000119, 6, 6)
    # Second row
    assert_min_max_3d(lines[12], 1.5, 2.5, 2, 2, 6, 6)
    assert_min_max_3d(lines[13], 0.5, 1.5, 2, 2, 6, 6)
    # Arrow
    assert_min_max_3d(lines[14], 2.5, 2.5, 2, 2, 6, 6)
    # End
    assert_min_max_3d(lines[15], 0.5, 0.5, 2, 2, 6, 6)
    # Capture points
    assert len(lines[-1].get_data()[0]) == 8
    assert_min_max_3d(lines[-1], 1, 2, 2, 3, 5, 6)
