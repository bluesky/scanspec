.. _API:

API
===

The top level scanspec module contains a number of packages that can be used
from code:

- `scanspec.core`: Core classes like `Dimension` and `Path`
- `scanspec.specs`: `Spec` and its subclasses
- `scanspec.regions`: `Region` and its subclasses
- `scanspec.plot`: `plot_spec` to visualize a scan

``scanspec``
------------

.. data:: scanspec.__version__
    :type: str

    Version number as calculated by https://github.com/dls-controls/versiongit

.. automodule:: scanspec.core
    :members:

    ``scanspec.core``
    -----------------

.. automodule:: scanspec.specs
    :members:

    ``scanspec.specs``
    ------------------

    .. inheritance-diagram:: scanspec.specs
        :top-classes: scanspec.specs.Spec
        :parts: 1

.. automodule:: scanspec.regions
    :members:

    ``scanspec.regions``
    --------------------

    .. inheritance-diagram:: scanspec.regions
        :top-classes: scanspec.regions.Region
        :parts: 1

.. automodule:: scanspec.plot
    :members:

    ``scanspec.plot``
    -----------------