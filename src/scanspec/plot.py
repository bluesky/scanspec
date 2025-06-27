"""`plot_spec` to visualize a scan."""

from collections.abc import Iterable, Iterator
from itertools import cycle
from typing import Any

import numpy as np
import numpy.typing as npt
from matplotlib import colors, patches
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D, proj3d  # type: ignore
from scipy import interpolate  # type: ignore

from .core import stack2dimension
from .regions import Circle, Ellipse, Polygon, Rectangle, Region, find_regions
from .specs import DURATION, Spec

__all__ = ["plot_spec"]


def _plot_arrays(axes: Axes, arrays: list[npt.NDArray[np.float64]], **kwargs: Any):
    if len(arrays) > 2:
        axes.plot3D(arrays[2], arrays[1], arrays[0], **kwargs)  # type: ignore
    elif len(arrays) == 2:
        axes.plot(arrays[1], arrays[0], **kwargs)  # type: ignore
    else:
        axes.plot(arrays[0], np.zeros(len(arrays[0])), **kwargs)  # type: ignore


# https://stackoverflow.com/a/11156353
class Arrow3D(patches.FancyArrowPatch):
    def __init__(
        self,
        xs: npt.NDArray[np.float64],
        ys: npt.NDArray[np.float64],
        zs: npt.NDArray[np.float64],
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__((0, 0), (0, 0), *args, **kwargs)  # type: ignore
        self._verts3d = xs, ys, zs

    # Added here because of https://github.com/matplotlib/matplotlib/issues/21688
    def do_3d_projection(self, renderer: Any = None):  # type: ignore
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)  # type: ignore
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))  # type: ignore

        return np.min(zs)  # type: ignore

    @property
    def verts3d(
        self,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        return self._verts3d


def _plot_arrow(axes: Axes, arrays: list[npt.NDArray[np.float64]]):
    if len(arrays) == 1:
        arrays = [np.array([0, 0])] + arrays
    if len(arrays) == 2:
        head = [a[-1] for a in reversed(arrays)]
        tail = [a[-1] - (a[-1] - a[-2]) * 0.1 for a in reversed(arrays)]
        axes.annotate(  # type: ignore
            "",
            tuple(head[:2]),
            tuple(tail[:2]),
            arrowprops={"color": "lightgrey", "arrowstyle": "-|>"},
        )
    elif len(arrays) == 3:
        arrows = [a[-2:] for a in reversed(arrays)]
        a = Arrow3D(*arrows[:3], mutation_scale=10, arrowstyle="-|>", color="lightgrey")
        axes.add_artist(a)


def _plot_spline(
    axes: Axes,
    ranges: list[float],
    arrays: list[npt.NDArray[np.float64]],
    index_colours: dict[int, str],
) -> Iterable[list[npt.NDArray[np.float64]]]:
    scaled_arrays = [a / r for a, r in zip(arrays, ranges, strict=False)]
    # Define curves parametrically
    t = np.zeros(len(arrays[0]))
    t[1:] = np.sqrt(sum((arr[1:] - arr[:-1]) ** 2 for arr in scaled_arrays))
    t = np.cumsum(t)
    if t[-1] > 0:
        # Can't make a spline that starts and ends in the same place, so add a small
        # delta
        for s, r in zip(scaled_arrays, ranges, strict=False):
            if s[0] == s[-1]:
                s += np.linspace(0, r * 1e-7, len(s))
        # There are no duplicated points, plot a spline
        t /= t[-1]
        # Scale the arrays so splines don't favour larger scaled axes
        tck, _ = interpolate.splprep(scaled_arrays, k=2, s=0)  # type: ignore
        starts = sorted(index_colours)
        stops = starts[1:] + [len(arrays[0]) - 1]
        for start, stop in zip(starts, stops, strict=False):
            start_value: float = t[start]
            stop_value: float = t[stop]
            tnew = np.linspace(start_value, stop_value, num=1001)
            spline: npt.NDArray[np.float64] = interpolate.splev(tnew, tck)  # type: ignore
            # Scale the splines back to the original scaling
            unscaled_splines = [a * r for a, r in zip(spline, ranges, strict=False)]
            _plot_arrays(axes, list(unscaled_splines), color=index_colours[start])  # type: ignore
            yield unscaled_splines  # type: ignore


def plot_spec(spec: Spec[Any], title: str | None = None):
    """Plot a spec, drawing the path taken through the scan.

    Uses a different colour for each frame, grey for the turnarounds, and
    marks the midpoints with a filled circle if there are less than 200 of
    them. If the scan is 2D then 2D regions are shown in black.

    .. example_spec::

        from scanspec.specs import Line
        from scanspec.regions import Circle

        cube = Line("z", 1, 3, 3) * Line("y", 1, 3, 10) * ~Line("x", 0, 2, 10)
        spec = cube & Circle("x", "y", 1, 2, 0.9)
    """
    dims = spec.calculate()
    dim = stack2dimension(dims)
    axes = [a for a in spec.axes() if a is not DURATION]
    ndims = len(axes)

    # Setup axes
    if ndims > 2:
        plt.figure(figsize=(6, 6))  # type: ignore
        plt_axes: Axes = plt.axes(projection="3d")  # type: ignore
        plt_axes.grid(False)  # type: ignore
        if isinstance(plt_axes, Axes3D):
            plt_axes.set_zlabel(axes[-3])  # type: ignore
            plt_axes.set_ylabel(axes[-2])  # type: ignore
            plt_axes.view_init(elev=15)  # type: ignore
        else:
            raise TypeError(
                "Expected matplotlib to create an Axes3D object, "
                f"instead got: {plt_axes}"
            )
    elif ndims == 2:
        plt.figure(figsize=(6, 6))  # type: ignore
        plt_axes = plt.axes()  # type: ignore
        plt_axes.set_ylabel(axes[-2])  # type: ignore
    else:
        plt.figure(figsize=(6, 2))  # type: ignore
        plt_axes = plt.axes()  # type: ignore
        plt_axes.yaxis.set_visible(False)
    plt_axes.set_xlabel(axes[-1])  # type: ignore

    # Title with dimension sizes
    title = title or ", ".join(f"Dim[{' '.join(d.axes())} len={len(d)}]" for d in dims)
    plt.title(title)  # type: ignore

    # Plot any Regions
    if ndims <= 2:
        regions: Iterator[Region[Any]] = find_regions(spec)
        for region in regions:
            if isinstance(region, Rectangle):
                xy = (region.x_min, region.y_min)
                width = region.x_max - region.x_min
                height = region.y_max - region.y_min
                plt_axes.add_patch(
                    patches.Rectangle(xy, width, height, angle=region.angle, fill=False)
                )
            elif isinstance(region, Circle):
                xy = (region.x_middle, region.y_middle)
                plt_axes.add_patch(patches.Circle(xy, region.radius, fill=False))
            elif isinstance(region, Ellipse):
                xy = (region.x_middle, region.y_middle)
                width = region.x_radius * 2
                height = region.y_radius * 2
                angle = region.angle
                plt_axes.add_patch(
                    patches.Ellipse(xy, width, height, angle=angle, fill=False)
                )
            elif isinstance(region, Polygon):
                # *xy_verts* is a numpy array with shape Nx2.
                xy_verts = np.column_stack((region.x_verts, region.y_verts))
                plt_axes.add_patch(patches.Polygon(xy_verts, fill=False))

    # Plot the splines
    tail: dict[str, npt.NDArray[np.float64] | None] = dict.fromkeys(axes)
    ranges = [max(float(np.max(v) - np.min(v)), 0.0001) for v in dim.midpoints.values()]
    seg_col = cycle(colors.TABLEAU_COLORS)
    last_index = 0
    splines = None
    # The first element of gap is undefined (as there is no previous frame)
    # so discard it
    gap_indices = list(np.nonzero(dim.gap[1:])[0] + 1)
    for index in gap_indices + [len(dim)]:
        num_points = index - last_index
        arrays: list[npt.NDArray[np.float64]] = []
        turnaround: list[npt.NDArray[np.float64]] = []
        for a in axes:
            # Add the midpoints and the lower and upper bounds
            arr = np.empty(num_points * 2 + 1)
            arr[:-1:2] = dim.lower[a][last_index:index]
            arr[1::2] = dim.midpoints[a][last_index:index]
            arr[-1] = dim.upper[a][index - 1]
            arrays.append(arr)
            # Add the turnaround
            axis_tail = tail[a]
            if axis_tail is not None:
                # Already had a tail, add lead in points
                axis_tail[2:] = np.linspace(-0.01, 0, 2) * (arr[1] - arr[0]) + arr[0]
                turnaround.append(axis_tail)
            # Add tail off points
            axis_tail = np.empty(4)
            axis_tail[:2] = np.linspace(0, 0.01, 2) * (arr[-1] - arr[-2]) + arr[-1]
            tail[a] = axis_tail
        last_index = index

        arrow_arr = None
        if turnaround:
            # If we didn't move then plot a straight line from start to stop
            if all(t[1] - t[0] == 0 for t in turnaround):
                for t in turnaround:
                    t[1] += (t[2] - t[1]) / 4
            if all(t[3] - t[2] == 0 for t in turnaround):
                for t in turnaround:
                    t[2] -= (t[2] - t[1]) / 4
            # Plot the turnaround
            arrow_arr = list(
                _plot_spline(plt_axes, ranges, turnaround, {0: "lightgrey"})
            )[0]

        # Plot the points
        index_colours = {2 * i: next(seg_col) for i in range(num_points)}
        splines = list(_plot_spline(plt_axes, ranges, arrays, index_colours))

        if arrow_arr:
            # Plot the arrow on the turnaround
            _plot_arrow(plt_axes, arrow_arr)
        elif splines:
            # Plot the starting arrow in the direction of the first point
            arrow_arr = [np.array([2 * a[0] - a[1], a[0]]) for a in splines[0]]
            _plot_arrow(plt_axes, arrow_arr)
        else:
            # First point isn't moving, put a right caret marker
            _plot_arrays(
                plt_axes,
                [np.array([dim.lower[a][0]]) for a in axes],
                marker=5,
                color="lightgrey",
            )

    # Plot the capture points
    if len(dim) < 200:
        arrays = [dim.midpoints[a] for a in axes]
        _plot_arrays(plt_axes, arrays, linestyle="", marker=".", color="k")

    # Plot the end
    _plot_arrays(
        plt_axes,
        [np.array([dim.upper[a][-1]]) for a in axes],
        marker="x",
        color="lightgrey",
    )

    plt.show()  # type: ignore
