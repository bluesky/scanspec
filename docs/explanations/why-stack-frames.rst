Why create a stack of Frames?
=============================

If a `Spec` tells you the parameters of a scan, `Dimension` gives you the `Points
<Point_>` that will let you actually execute the scan. A stack of Frames is
interpreted as nested from slowest moving to fastest moving, so each faster
Frames object will iterate once per position of the slower Frames object. When
fly-scanning the axis will traverse lower-midpoint-upper on the fastest Frames
object for each point in the scan.

An Example
----------

>>> from scanspec.specs import Line
>>> spec = Line("y", 4, 5, 6) * Line("x", 1, 2, 3)
>>> stack = spec.calculate()
>>> len(stack)
2
>>> len(stack[0])
6
>>> len(stack[1])
3
>>> stack[0].midpoints
{'y': array([4. , 4.2, 4.4, 4.6, 4.8, 5. ])}
>>> stack[1].midpoints
{'x': array([1. , 1.5, 2. ])}

So the `Product` of two `Lines <Line>` creates a stack of 2 Frames objects, the first
having the same size as the outer line, and the second having the same size as
the inner. Executing the scan will iterate the inner Frames object 6 times, once for
each point in the outer Frames object, 18 points in all.

Why not squash them into a flat sequence?
-----------------------------------------

A stack of Frames objects are created to give the most compact representation of
the scan. Imagine a 100x2000x2000 point scan, which creates a list of 3 Frames
objects. Considering just the midpoint arrays, they would take 100 + 2000 + 2000
float64 = 32.8kB RAM in our list form. If we squashed the list into a single
Frames object it would take 100 * 2000 * 2000 float64 = 3.2GB of RAM. The scan
itself is likely to be executed over a long period of time, so it makes sense to
save on the memory and calculate the squashed points as they are needed.

What about Regions?
-------------------

Regions will stop the regularity of the nested Frames objects, so will cause
them to be squashed into a single Frames object. Taking our example above, if we
`Mask` the grid with a `Circle`, then the `Line` in ``x`` won't have 3 points in
each iteration, the number of points is dependent on ``y``. This means that a
`Mask` will squash any Specs together referred to by its Regions.

How does this stack relate to HDF5 Dimensions?
----------------------------------------------

A Spec by itself does not specify how the scan points should be written to disk.
The simplest strategy is to simply write a stack of images, along with the
setpoint positions specified in the Frames objects, and let the reader of the data
pick out the required frames from the positions. However, the stack of Frames objects
contains all the information required to reshape a sequence of HDF5 frames into the
dimensionality of the scan using a VDS. This holds until we turn snake on, at
which point it destroys the performance of the VDS. For this reason, it is
advisable to `Squash` any snaking Specs with the first non-snaking axis above it
so that the HDF Dimension will not be snaking. See `./why-squash-can-change-path` for
some details on this.
