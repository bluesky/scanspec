.. _creating-a-spec:

Creating a Scan Spec
====================

This tutorial shows how to create Scan Specs of increasing complexity, plotting
the results.

Line
----

We'll start with a simple one, a `Line`. If you enter the following
code into an interactive Python terminal, it should plot a graph of a 1D line:

.. plot::


    from scanspec.specs import Line
    from scanspec.plot import plot_spec

    spec = Line("x", 1, 2, 5)
    plot_spec(spec)

This will create a Spec with 5 points, the first centred on 1, the last centred
on 2. The black dots mark the centre positions on the Path. The coloured
lines mark the motion from the lower to upper bound of each point on the Path.
The grey arrowhead marks the start of the scan, and the cross marks the end.

You can see the points if you create the Path yourself, then consume all its points:

.. code:: python

    >>> chunk = spec.path().consume()
    >>> chunk.lower
    {'x': array([0.875, 1.125, 1.375, 1.625, 1.875])}
    >>> chunk.positions
    {'x': array([1.  , 1.25, 1.5 , 1.75, 2.  ])}
    >>> chunk.upper
    {'x': array([1.125, 1.375, 1.625, 1.875, 2.125])}

The array of positions is between the lower and upper bounds, with the upper bound
and lower bound of adjacent points being equal. This consumable `Path` object is
what a scanning program should use to execute the scan described by a `Spec`.

Plotting from the commandline
-----------------------------

To quickly plot Scan Specs you can use the commandline client. The input is
evaluated with variables ``a`` to ``z`` defined and the output plotted.

For example, for the Line example above you would type::

    $ scanspec plot 'Line(x, 1, 2, 5)'


Line with 2 axes
----------------

If we want to plot a Line in two axes, we can do this with `Zip`, or the ``+`` operator:

.. plot::

    from scanspec.specs import Line
    from scanspec.plot import plot_spec

    spec = Line("y", 3, 4, 5) + Line("x", 1, 2, 5)
    plot_spec(spec)

Grid
----

We can make a grid by creating the `Product` of 2 Lines with the ``*`` operator:

.. plot::

    from scanspec.specs import Line
    from scanspec.plot import plot_spec

    spec = Line("y", 3, 4, 3) * Line("x", 1, 2, 5)
    plot_spec(spec)

The plot shows grey arrowed lines marking the turnarounds. These are added by
the plotting function as an indication of what a scanning program might do between
two disjoint points, it is not a guarantee of the path that will be taken.

Examining the spec shows that it is a `Product`, whose length is
the product of the lengths of the two input Specs:

    >>> spec
    Product(type='Product', outer=Line(type='Line', key='y', start=3.0, stop=4.0, num=3), inner=Line(type='Line', key='x', start=1.0, stop=2.0, num=5))
    >>> len(spec.path())
    15

Snaked Grid
-----------

We can `Snake` a Spec with the ``~`` operator. If we apply this to the inner Spec of
our grid we get:

.. plot::


    from scanspec.specs import Line
    from scanspec.plot import plot_spec

    spec = Line("y", 3, 4, 3) * ~Line("x", 1, 2, 5)
    plot_spec(spec)

Masking with Regions
--------------------

We can apply a `Mask` to only include positions within a given `Region` using
the ``&`` operator:

.. plot::


    from scanspec.specs import Line
    from scanspec.regions import Circle
    from scanspec.plot import plot_spec

    spec = Line("y", 3, 4, 3) * ~Line("x", 1, 2, 5) & Circle("x", "y", 1.5, 3.5, 0.6)
    plot_spec(spec)


Masking with Multiple Regions
-----------------------------

We can apply set-like operators to Masked Specs:

- ``|``: `UnionOf` two Regions
- ``&``: `IntersectionOf` two Regions
- ``-``: `DifferenceOf` two Regions
- ``^``: `SymmetricDifferenceOf` two Regions

For example:

.. plot::


    from scanspec.specs import Line
    from scanspec.regions import Circle
    from scanspec.plot import plot_spec

    spec = Line("y", 3, 4, 3) * ~Line("x", 1, 2, 5) & Circle("x", "y", 1.5, 3.5, 0.6) - Circle("x", "y", 1.4, 3.5, 0.2)
    plot_spec(spec)

Conclusion
----------

This tutorial has demonstrated some Specs and combinations of them. From here
you may like to read `iterate-a-spec` to see how a scanning system could use
these Specs and `serialize-a-spec` to see how you might send one to such a
scanning system.
