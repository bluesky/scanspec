Technical Terms
===============

This page draws together some of the technical terms used throughout
the documentation. Consider a 1D line scan:

.. image:: ../images/definitions.png

.. _axis_:

Axis
----

A fixed reference that can be scanned, i.e. a motor or the `frame_` duration.
In the diagram above, the Axis is ``x``. `Spec.axes` will return
the Axes that should be scanned for a given Spec.

.. _dimension_:

Dimension
---------

A repeatable, possibly snaking, series of `Frames <frame_>` along a number of
`Axes <axis_>`. In the diagram above, the whole Line produces a single Dimension.
A stack of `Dimension` instances are produced by `Spec.create_dimensions`.

.. seealso:: `./what-are-dimensions`

.. _frame_:

Frame
-----

A vector of three `Points <point_>` in scan space: lower, midpoint, upper. They
describe the trajectory that should be taken while a detector is active while
fly scanning. In the diagram above each Frame is denoted by a different coloured
section.

.. _path_:

Path
----

A consumable route through one or more `Dimensions <dimension_>`. If the Line in
the above diagram was stacked within another line of length 5, then the Path
would contain 15 `Frames <frame_>`. A `Path` is produced from `Spec.path` or
can be created from a `list` of `Dimension` objects.

.. _point_:

Point
-----

A single or multidimensional location in scan space. In the diagram above,
each `frame_` is made up of lower, midpoint and upper Points. Arrays of
these are available as `Dimension.lower`, `Dimension.midpoints` and
`Dimension.upper`.
