from itertools import cycle

from scanspec.core import Batch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from scipy import interpolate

from scanspec.specs import Concat, Line, Snake, Spec


def find_breaks(batch: Batch):
    breaks = []
    for key in batch.keys:
        breaks.append(batch.lower[key][1:] != batch.upper[key][:-1])
    same = np.logical_or.reduce(breaks)
    break_indices = np.nonzero(same)[0] + 1
    return list(break_indices)


def plot_arrays(axes, arrays, **kwargs):
    arrays = list(arrays)
    if len(arrays) > 2:
        axes.plot3D(arrays[2], arrays[1], arrays[0], **kwargs)
    elif len(arrays) == 2:
        axes.plot(arrays[1], arrays[0], **kwargs)
    else:
        axes.plot(np.zeros(len(arrays[0])), arrays[0], **kwargs)


def plot_spec(spec: Spec):
    batch = spec.create_view().create_batch()
    ndims = len(spec.keys)
    if ndims > 2:
        axes = plt.axes(projection="3d")
    else:
        axes = plt.axes()
    spline_points = [[] for _ in range(ndims)]
    segment_indexes = []
    turnaround_indexes = set()
    last_index = 0
    seg_col = cycle(TABLEAU_COLORS)
    for index in find_breaks(batch) + [len(batch)]:
        for i, (k, p) in enumerate(zip(spec.keys, spline_points)):
            # Add 9 lead in points
            diff = batch.positions[k][last_index] - batch.lower[k][last_index]
            p.append(np.arange(-0.1, -0.01, 0.01) * diff + batch.lower[k][last_index])
            # Add the segments
            if i == 0:
                start = sum(len(x) for x in p)
                segment_indexes.append(start + np.arange(index-last_index)*2)
            combined = np.empty((index-last_index)*2 + 1)
            combined[0:-1:2] = batch.lower[k][last_index:index]
            combined[1::2] = batch.positions[k][last_index:index]
            combined[-1] = batch.upper[k][index-1]
            p.append(combined)
            # Add 9 tail off points
            if i == 0:
                turnaround_start = segment_indexes[-1][-1] + 2
                segment_indexes.append([turnaround_start])
                turnaround_indexes.add(turnaround_start)
            diff = batch.upper[k][index-1] - batch.positions[k][index-1]
            p.append(np.arange(0.01, 0.1, 0.01) * diff + batch.upper[k][index-1])
        last_index = index
    # Concat them to use as input
    spline_points = [np.concatenate(x) for x in spline_points]
    # Define curves parametrically
    t = np.zeros(len(spline_points[0]))
    t[1:] = np.sqrt(sum((arr[1:] - arr[:-1])**2 for arr in spline_points))
    t = np.cumsum(t)
    t /= t[-1]
    tck, _ = interpolate.splprep(spline_points, k=2, s=0)

    # Plot the segments
    segment_indexes = np.concatenate(segment_indexes)
    for i, index in enumerate(segment_indexes):
        if i + 1 < len(segment_indexes):
            end = segment_indexes[i+1]
        else:
            end = len(spline_points[-1]) - 1
        tnew = np.linspace(t[index], t[end], num=1001, endpoint=True)
        color = "lightgrey" if index in turnaround_indexes else next(seg_col)
        plot_arrays(axes, interpolate.splev(tnew, tck), color=color)

    # Plot the capture points
    if len(batch) < 100:
        plot_arrays(axes, batch.positions.values(), linestyle="", marker="x", color="k")
    plt.show()

plot_spec(Line("z", 3, 4, 3) * Snake(Line("y", 0, 1, 3) + Line("x", 2, 3, 3)))
