# API

```{eval-rst}
.. automodule:: scanspec

    ``scanspec``
    ------------

```

The top level scanspec module contains a number of packages that can be used
from code:

- [](#scanspec.core): Core classes like [](#Frames) and [](#Path)
- [](#scanspec.specs): [](#Spec) and its subclasses
- [](#scanspec.regions): [](#Region) and its subclasses
- [](#scanspec.plot): [](#plot_spec) to visualize a scan
- [](#scanspec.service): Defines queries and field structure in REST such as [](#MidpointsResponse)

```{eval-rst}
.. data:: scanspec.__version__
    :type: str

    Version number as calculated by https://github.com/dls-controls/versiongit
```

```{eval-rst}
.. automodule:: scanspec.core
    :members:

    ``scanspec.core``
    -----------------
```

```{eval-rst}
.. automodule:: scanspec.specs
    :members:

    ``scanspec.specs``
    ------------------

    .. inheritance-diagram:: scanspec.specs
        :top-classes: scanspec.specs.Spec
        :parts: 1
```

```{eval-rst}
.. automodule:: scanspec.regions
    :members:

    ``scanspec.regions``
    --------------------

    .. inheritance-diagram:: scanspec.regions
        :top-classes: scanspec.regions.Region
        :parts: 1
```

```{eval-rst}
.. automodule:: scanspec.plot
    :members:

    ``scanspec.plot``
    -----------------
```

```{eval-rst}
.. automodule:: scanspec.service
    :members:

    ``scanspec.service``
    --------------------
```
