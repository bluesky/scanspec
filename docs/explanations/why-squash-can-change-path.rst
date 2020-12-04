Why Squash (and Mask) can change the Path
=========================================

A Spec creates a number of Dimensions that can be set to snake. Stuff...


.. plot::

    from scanspec.specs import Line
    from scanspec.plot import plot_spec

    spec = Line("z", 0, 1, 3) * ~Line("y", 0, 1, 3) * Line("x", 0, 1, 3)
    plot_spec(spec)


.. plot::

    from scanspec.specs import Line, Squash
    from scanspec.plot import plot_spec

    spec = Line("z", 0, 1, 3) * Squash(~Line("y", 0, 1, 3) * Line("x", 0, 1, 3), check_path_changes=False)
    plot_spec(spec)


.. plot::

    from scanspec.specs import Line
    from scanspec.plot import plot_spec

    spec = Line("z", 0, 1, 3) * Line("y", 0, 1, 3) * ~Line("x", 0, 1, 3)
    plot_spec(spec)


.. plot::

    from scanspec.specs import Line, Squash
    from scanspec.plot import plot_spec

    spec = Line("z", 0, 1, 3) * Squash(Line("y", 0, 1, 3) * ~Line("x", 0, 1, 3), check_path_changes=False)
    plot_spec(spec)
