from itertools import cycle
from typing import Any, Dict, List

import numpy as np
from matplotlib import colors, patches
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from scipy import interpolate

from .core import Dimension, Path
from .regions import Circle, Rectangle, find_regions
from .specs import TIME, Spec


def _find_breaks(dim: Dimension):
    breaks = []
    for key in dim.keys():
        breaks.append(dim.lower[key][1:] != dim.upper[key][:-1])
    same = np.logical_or.reduce(breaks)
    break_indices = np.nonzero(same)[0] + 1
    return list(break_indices)


def _plot_arrays(axes, arrays: List[np.ndarray], **kwargs):
    if len(arrays) > 2:
        axes.plot3D(arrays[2], arrays[1], arrays[0], **kwargs)
    elif len(arrays) == 2:
        axes.plot(arrays[1], arrays[0], **kwargs)
    else:
        axes.plot(arrays[0], np.zeros(len(arrays[0])), **kwargs)


# https://stackoverflow.com/a/11156353
class _Arrow3D(patches.FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, _ = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)


def _plot_arrow(axes, arrays: List[np.ndarray]):
    if len(arrays) == 1:
        arrays = [[0, 0]] + arrays
    if len(arrays) == 2:
        head = [a[-1] for a in reversed(arrays)]
        tail = [a[-1] - (a[-1] - a[-2]) * 0.1 for a in reversed(arrays)]
        axes.annotate(
            "", head[:2], tail[:2], arrowprops=dict(color="lightgrey", arrowstyle="-|>")
        )
    elif len(arrays) == 3:
        arrows = [a[-2:] for a in reversed(arrays)]
        a = _Arrow3D(
            *arrows[:3], mutation_scale=10, arrowstyle="-|>", color="lightgrey"
        )
        axes.add_artist(a)


def _plot_spline(axes, ranges, arrays: List[np.ndarray], index_colours: Dict[int, str]):
    scaled_arrays = [a / r for a, r in zip(arrays, ranges)]
    # Define curves parametrically
    t = np.zeros(len(arrays[0]))
    t[1:] = np.sqrt(sum((arr[1:] - arr[:-1]) ** 2 for arr in scaled_arrays))
    t = np.cumsum(t)
    if t[-1] > 0:
        # There are no duplicated points, plot a spline
        t /= t[-1]
        # Scale the arrays so splines don't favour larger scaled axes
        tck, _ = interpolate.splprep(scaled_arrays, k=2, s=0)
        starts = sorted(list(index_colours))
        stops = starts[1:] + [len(arrays[0]) - 1]
        for start, stop in zip(starts, stops):
            tnew = np.linspace(t[start], t[stop], num=1001)
            spline = interpolate.splev(tnew, tck)
            # Scale the splines back to the original scaling
            unscaled_splines = [a * r for a, r in zip(spline, ranges)]
            _plot_arrays(axes, unscaled_splines, color=index_colours[start])
            yield unscaled_splines


def plot_spec(spec: Spec):
    """Plot a spec, drawing the path taken through the scan, using a different
    colour for each point, grey for the turnarounds, and marking the
    centrepoints with a filled circle if there are less than 200 of them. If the
    scan is 2D then 2D regions are shown in black.

    .. example_spec::

        from scanspec.specs import Line
        from scanspec.regions import Circle

        cube = Line("z", 1, 3, 3) * Line("y", 1, 3, 10) * ~Line("x", 0, 2, 10)
        spec = cube & Circle("x", "y", 1, 2, 0.9)
    """
    dims = spec.create_dimensions()
    dim = Path(dims).consume()
    keys = [k for k in spec.keys() if k is not TIME]
    ndims = len(keys)

    # Setup axes
    if ndims > 2:
        plt.figure(figsize=(6, 6))
        axes = plt.axes(projection="3d")
        axes.grid(False)
        axes.set_zlabel(keys[-3])
        axes.set_ylabel(keys[-2])
        axes.view_init(elev=15)
    elif ndims == 2:
        plt.figure(figsize=(6, 6))
        axes = plt.axes()
        axes.set_ylabel(keys[-2])
    else:
        plt.figure(figsize=(6, 2))
        axes = plt.axes()
        axes.yaxis.set_visible(False)
    axes.set_xlabel(keys[-1])

    # Title with dimension sizes
    plt.title(", ".join(f"Dim[{' '.join(d.keys())} len={len(d)}]" for d in dims))

    # Plot any Regions
    if ndims <= 2:
        for region in find_regions(spec):
            if isinstance(region, Rectangle):
                xy = (region.x_min, region.y_min)
                width = region.x_max - region.x_min
                height = region.y_max - region.y_min
                axes.add_patch(
                    patches.Rectangle(xy, width, height, region.angle, fill=False)
                )
            elif isinstance(region, Circle):
                xy = (region.x_centre, region.y_centre)
                axes.add_patch(patches.Circle(xy, region.radius, fill=False))

    # Plot the splines
    tail: Any = {k: None for k in keys}
    ranges = [max(np.max(v) - np.min(v), 0.0001) for k, v in dim.positions.items()]
    seg_col = cycle(colors.TABLEAU_COLORS)
    last_index = 0
    splines = None
    for index in _find_breaks(dim) + [len(dim)]:
        num_points = index - last_index
        arrays = []
        turnaround = []
        for k in keys:
            # Add the lower and positions
            arr = np.empty(num_points * 2 + 1)
            arr[:-1:2] = dim.lower[k][last_index:index]
            arr[1::2] = dim.positions[k][last_index:index]
            arr[-1] = dim.upper[k][index - 1]
            arrays.append(arr)
            # Add the turnaround
            if tail[k] is not None:
                # Already had a tail, add lead in points
                tail[k][2:] = np.linspace(-0.01, 0, 2) * (arr[1] - arr[0]) + arr[0]
                turnaround.append(tail[k])
            # Add tail off points
            tail[k] = np.empty(4)
            tail[k][:2] = np.linspace(0, 0.01, 2) * (arr[-1] - arr[-2]) + arr[-1]
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
            arrow_arr = list(_plot_spline(axes, ranges, turnaround, {0: "lightgrey"}))[
                0
            ]

        # Plot the points
        index_colours = {2 * i: next(seg_col) for i in range(num_points)}
        splines = list(_plot_spline(axes, ranges, arrays, index_colours))

        if arrow_arr:
            # Plot the arrow on the turnaround
            _plot_arrow(axes, arrow_arr)
        elif splines:
            # Plot the starting arrow in the direction of the first point
            arrow_arr = [(2 * a[0] - a[1], a[0]) for a in splines[0]]
            _plot_arrow(axes, arrow_arr)
        else:
            # First point isn't moving, put a right caret marker
            _plot_arrays(
                axes, [[dim.lower[k][0]] for k in keys], marker=5, color="lightgrey"
            )

    # Plot the capture points
    if len(dim) < 200:
        arrays = [dim.positions[k] for k in keys]
        _plot_arrays(axes, arrays, linestyle="", marker=".", color="k")

    # Plot the end
    _plot_arrays(
        axes, [[dim.upper[k][-1]] for k in keys], marker="x", color="lightgrey"
    )

    plt.show()
