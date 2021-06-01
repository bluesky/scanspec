.. _API:

API
===

The top level scanspec module contains a number of packages that can be used
from code:

- `scanspec.core`: Core classes like `Dimension` and `Path`
- `scanspec.specs`: `Spec` and its subclasses
- `scanspec.regions`: `Region` and its subclasses
- `scanspec.plot`: `plot_spec` to visualize a scan
- `scanspec.service`: Defines queries and field structure in graphQL such as `PointsResponse`

``scanspec``
------------

.. data:: scanspec.__version__
    :type: str

    Version number as calculated by https://github.com/dls-controls/versiongit

.. automodule:: scanspec.core
    :members:
    :exclude-members: Serializable

    ``scanspec.core``
    -----------------

    .. autoclass:: scanspec.core.Serializable
        :no-show-inheritance:
        :members:

        .. seealso::
            https://github.com/wyfo/apischema/discussions/56#discussioncomment-336580
            for discussion on tagged unions and alternative constructors

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
    :exclude-members: abs_diffs

    ``scanspec.service``
    --------------------
