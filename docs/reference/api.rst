API
===

.. automodule:: scanspec

    ``scanspec``
    ------------


The top level scanspec module contains a number of packages that can be used
from code:

- `scanspec.core`: Core classes like `Frames` and `Path`
- `scanspec.specs`: `Spec` and its subclasses
- `scanspec.regions`: `Region` and its subclasses
- `scanspec.plot`: `plot_spec` to visualize a scan
- `scanspec.service`: Defines queries and field structure in graphQL such as `PointsResponse`

.. data:: scanspec.__version__
    :type: str

    Version number as calculated by https://github.com/pypa/setuptools_scm

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

.. automodule:: scanspec.service
    :members:
    :exclude-members: abs_diffs, Points, AxisFrames, PointsResponse

    ``scanspec.service``
    --------------------

    .. autoclass:: scanspec.service.Points
        :members:

    .. autoclass:: scanspec.service.AxisFrames
        :members:

    .. autoclass:: scanspec.service.PointsResponse
        :members:
