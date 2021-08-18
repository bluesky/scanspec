.. _iterate-a-spec:

How to Iterate a Spec
=====================

A Spec is only the specification for a scan. To execute the scan we need the
`Frames <frame_>`. We can do this in a few ways.

If you only need the midpoints
------------------------------

If you are conducting a step scan, you only need the midpoints of each
scan frame. You can get these by using the `Spec.midpoints()` method to produce a
`Midpoints` iterator of scan `Points <point_>`:

>>> from scanspec.specs import Line
>>> spec = Line("x", 1, 2, 3)
>>> for d in spec.midpoints():
...     print(d)
...
{'x': 1.0}
{'x': 1.5}
{'x': 2.0}

This is simple, but not particularly performant, as the numpy arrays of
points are unpacked point by point into point dictionaries

If you need to do a fly scan
----------------------------

If you are conducting a fly scan then you need the `path` that the motor moves
through. You can get that from the lower and upper bounds of each point. It is
more efficient to consume this Path in numpy array chunks that can be processed
into a trajectory:

>>> path = spec.path()
>>> len(path)
3
>>> chunk = path.consume()
>>> chunk.lower
{'x': array([0.75, 1.25, 1.75])}
>>> chunk.midpoints
{'x': array([1. , 1.5, 2. ])}
>>> chunk.upper
{'x': array([1.25, 1.75, 2.25])}
>>> len(path)
0

If ``upper[i] == lower[i+1]`` then we say that two scan points are joined, and
will be blended together in a motion trajectory, otherwise the scanning system
should insert a gap where the motors move between segments


If you need to do multiple runs of the same scan
------------------------------------------------

A `Path` instance is a one-shot consumable view of the list of `Dimension`
instances created by `Spec.create_dimensions()`. Once you have consumed it you
should create a new instance of it. For performance reasons, you can keep the
intermediate `Dimensions <dimension>` and create as many `Path` wrappers to them
as you need. You can also give a maximum size to `Path.consume()`

>>> from scanspec.core import Path
>>> dims = spec.create_dimensions()
>>> path = Path(dims)
>>> len(path)
3
>>> chunk = path.consume(2)
>>> len(chunk)
2
>>> len(path)
1
>>> chunk = path.consume(2)
>>> len(chunk)
1
>>> len(path)
0

You can also use this method to only run a subset of the scan:

>>> path = Path(dims, start=1, num=1)
>>> len(path)
1
>>> chunk = path.consume()
>>> len(chunk)
1
>>> len(path)
0


If you need to know whether there is a gap between points
---------------------------------------------------------

You may need to know where there is a gap between points, so that you can do
something in the turnaround. For example, if we take the x axis of a grid scan,
you can see it snakes back and forth:

>>> from scanspec.specs import Line, fly
>>> grid = fly(Line("y", 0, 1, 2) * ~Line("x", 1, 2, 3), 0.1)
>>> chunk = grid.path().consume()
>>> chunk.midpoints["x"]
array([1. , 1.5, 2. , 2. , 1.5, 1. ])

You can check where the gaps are by using the `Dimension.gap` attribute:

>>> chunk.gap
array([ True, False, False,  True, False, False])

This says whether there is a gap between this frame and the previous frame. In
the example we see 2 gaps:

- On the very first frame (as there is a gap between the last and first frames)
- Between the 2 frames with midpoint of 2.0

You could use this information to work out when to insert turnaround between
rows for the motors