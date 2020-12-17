.. _iterate-a-spec:

How to Iterate a Spec
=====================

A Spec is only the specification for a scan. To execute the scan we need the positions. We can do this in a few ways.

If you only need the positions
------------------------------

If you are conducting a step scan, you only need the central positions of each scan point. You can get these by using
the `Spec.positions` method to produce an `SpecPositions` iterator of scan positions:

>>> for d in Line("x", 1, 2, 3).positions():
...     print(d)
...
{'x': 1.0}
{'x': 1.5}
{'x': 2.0}

This is simple, but not particularly performant, as the numpy arrays of positions are unpacked point by point into
position dictionaries

If you need to do a fly scan
----------------------------

If you are conducting a fly scan then you need the Path that the motor moves through. You can get that from
the lower and upper bounds of each point. It is more efficient to get batches of points for these,


