.. _iterate-a-spec:

How to Iterate a Spec
=====================

A Spec is only the specification for a scan. To execute the scan we need the
`Dimension <frame_>`. We can do this in a few ways.

If you only need the midpoints
------------------------------

If you are conducting a step scan, you only need the midpoints of each scan
frame. You can get these by using the `Spec.midpoints()` method to produce a
`Midpoints` iterator of scan `Points <point_>`:

>>> from scanspec.specs import Line
>>> spec = Line("x", 1, 2, 3)
>>> for d in spec.midpoints():
...     print(d)
...
{'x': np.float64(1.0)}
{'x': np.float64(1.5)}
{'x': np.float64(2.0)}

This is simple, but not particularly performant, as the numpy arrays of
points are unpacked point by point into point dictionaries

If you need to do a fly scan
----------------------------

If you are conducting a fly scan then you need the frames that the motor moves
through. You can get that from the lower and upper bounds of each point. If the
scan is small enough to fit in memory on the machine you can use the `Spec.frames()`
method to produce a single `Dimension` object containing the entire scan:

>>> segment = spec.frames()
>>> len(segment)
3
>>> segment.lower
{'x': array([0.75, 1.25, 1.75])}
>>> segment.midpoints
{'x': array([1. , 1.5, 2. ])}
>>> segment.upper
{'x': array([1.25, 1.75, 2.25])}


If you want the most performant option
--------------------------------------

A `Path` instance is a one-shot consumable view of the stack of `Dimension`
objects created by `Spec.calculate()`. Once you have consumed it you
should create a new instance of it. For performance reasons, you can keep the
intermediate `Dimension` stack and create as many `Path` wrappers to them
as you need. You can also give a maximum size to `Path.consume()`

>>> from scanspec.core import Path
>>> stack = spec.calculate()
>>> path = Path(stack)
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

>>> path = Path(stack, start=1, num=1)
>>> len(path)
1
>>> chunk = path.consume()
>>> len(chunk)
1
>>> len(path)
0

.. seealso:: `../explanations/why-stack-frames`

If you need to know whether there is a gap between points
---------------------------------------------------------

You may need to know where there is a gap between points, so that you can do
something in the turnaround. For example, if we take the x axis of a grid scan,
you can see it snakes back and forth:

>>> from scanspec.specs import Line, fly
>>> grid = fly(Line("y", 0, 1, 2) * ~Line("x", 1, 2, 3), 0.1)
>>> chunk = grid.frames()
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
