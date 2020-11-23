from itertools import cycle
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TABLEAU_COLORS
from scipy import interpolate

from .core import Batch
from .specs import Spec

# Number of padding points to make the spline look nice
PAD = 2


def find_breaks(batch: Batch):
    breaks = []
    for key in batch.keys:
        breaks.append(batch.lower[key][1:] != batch.upper[key][:-1])
    same = np.logical_or.reduce(breaks)
    break_indices = np.nonzero(same)[0] + 1
    return list(break_indices)


def plot_arrays(axes, arrays: List[np.ndarray], **kwargs):
    if len(arrays) > 2:
        axes.plot3D(arrays[2], arrays[1], arrays[0], **kwargs)
    elif len(arrays) == 2:
        axes.plot(arrays[1], arrays[0], **kwargs)
    else:
        axes.plot(arrays[0], np.zeros(len(arrays[0])), **kwargs)


def plot_arrow(axes, arrays: List[np.ndarray]):
    diffs = [a[1] - a[0] for a in arrays]
    if len(diffs) == 1:
        diffs = [0] + diffs
    if diffs[-1] != 0:
        angle = np.degrees(np.arctan(diffs[-2] / diffs[-1]))
        if diffs[-1] > 0:
            angle -= 90
        else:
            angle += 90
        plot_arrays(
            axes, [[a[0]] for a in arrays], marker=(3, 0, angle), color="lightgrey"
        )


def plot_spline(axes, ranges, arrays: List[np.ndarray], index_colours: Dict[int, str]):
    scaled_arrays = [a / r for a, r in zip(arrays, ranges)]
    # Define curves parametrically
    t = np.zeros(len(arrays[0]))
    t[1:] = np.sqrt(sum((arr[1:] - arr[:-1]) ** 2 for arr in scaled_arrays))
    t = np.cumsum(t)
    t /= t[-1]
    # Scale the arrays so splines don't favour larger scaled axes
    tck, _ = interpolate.splprep(scaled_arrays, s=0)
    starts = sorted(list(index_colours))
    stops = starts[1:] + [len(arrays[0]) - 1]
    for start, stop in zip(starts, stops):
        tnew = np.linspace(t[start], t[stop], num=1001)
        spline = interpolate.splev(tnew, tck)
        # Scale the splines back to the original scaling
        unscaled_splines = [a * r for a, r in zip(spline, ranges)]
        plot_arrays(axes, unscaled_splines, color=index_colours[start])


def plot_spec(spec: Spec):
    batch = spec.create_view().create_batch()
    ndims = len(spec.keys)
    if ndims > 2:
        axes = plt.axes(projection="3d")
        z, y, x = list(spec.keys)[:3]
        plt.ylabel(y)
        axes.set_zlabel(z)
    else:
        axes = plt.axes()
        if ndims == 1:
            x = list(spec.keys)[0]
            plt.tick_params(left="off", labelleft="off")
        else:
            y, x = list(spec.keys)[:2]
            plt.ylabel(y)
    plt.xlabel(x)
    last_index = 0
    tail: Any = {k: None for k in spec.keys}
    ranges = [max(np.max(v) - np.min(v), 0.0001) for k, v in batch.positions.items()]
    seg_col = cycle(TABLEAU_COLORS)
    for index in find_breaks(batch) + [len(batch)]:
        num_points = index - last_index
        arrays = []
        turnaround = []
        for k in spec.keys:
            # Add the lower and positions
            arr = np.empty(num_points * 2 + 1)
            arr[:-1:2] = batch.lower[k][last_index:index]
            arr[1::2] = batch.positions[k][last_index:index]
            arr[-1] = batch.upper[k][index - 1]
            arrays.append(arr)
            # Add the turnaround
            if tail[k] is not None:
                # Already had a tail, add lead in points
                tail[k][PAD:] = np.linspace(-0.01, 0, PAD) * (arr[1] - arr[0]) + arr[0]
                turnaround.append(tail[k])
            # Add tail off points
            tail[k] = np.empty(PAD * 2)
            tail[k][:PAD] = np.linspace(0, 0.01, PAD) * (arr[-1] - arr[-2]) + arr[-1]

        if turnaround:
            plot_spline(axes, ranges, turnaround, {0: "lightgrey"})
        index_colours = {2 * i: next(seg_col) for i in range(num_points)}
        plot_spline(axes, ranges, arrays, index_colours)
        plot_arrow(axes, arrays)
        last_index = index

    # Plot the end
    plot_arrays(
        axes, [[batch.upper[k][-1]] for k in spec.keys], marker="x", color="lightgrey"
    )

    # Plot the capture points
    if len(batch) < 200:
        arrays = [batch.positions[k] for k in spec.keys]
        plot_arrays(axes, arrays, linestyle="", marker=".", color="k")

    plt.show()
