scanspec
========

|build_status| |coverage| |pypi_version| |license|

Specify step and flyscan Paths using combinations of:

- Specs like Line or Spiral
- Optionally Snaking
- Zip, Product and Concat to compose
- Masks with multiple Regions to restrict

Serialize the Spec rather than the expanded Path and reconstruct it on the
server. It can them be iterated over like a cycler_, or scan Dimensions
can be produced and expanded Paths created to consume chunk by chunk.

.. _cycler: https://matplotlib.org/cycler/

============== ==============================================================
PyPI           ``pip install scanspec``
Source code    https://github.com/dls-controls/scanspec
Documentation  https://dls-controls.github.io/scanspec
============== ==============================================================

An example ScanSpec of a 2D snaked grid flyscan inside a circle spending 0.4s at each point looks like:

.. code:: python

    from scanspec.specs import Line, fly
    from scanspec.regions import Circle

    grid = Line(ymotor, 2.1, 3.8, 12) * ~Line(xmotor, 0.5, 1.5, 10)
    spec = fly(grid, 0.4) & Circle(xmotor, ymotor, 1.0, 2.8, radius=0.5)

|plot|

You can then either iterate through the scan positions directly for convenience:

.. code:: python

    for positions in spec.positions():
        print(positions)
    # ...
    # {ymotor: 3.2813559322033896, xmotor: 0.8838383838383839, "TIME": 0.4}
    # {ymotor: 3.2813559322033896, xmotor: 0.8737373737373737, "TIME": 0.4}

or create a Path from the Dimensions and consume chunks of a given length from it for performance:

.. code:: python

    from scanspec.core import Path

    dims = spec.create_dimensions()
    len(dims[0].shape)  # 44
    dims[0].keys()  # (ymotor, xmotor, "TIME")

    path = Path(dims, start=5, num=30)
    chunk = path.consume(10)
    chunk.positions  # {xmotor: <ndarray len=10>, ymotor: <ndarray len=10>, "TIME": <ndarray len=10>}
    chunk.upper  # bounds are same dimensionality as positions


.. |build_status| image:: https://github.com/dls-controls/scanspec/workflows/Python%20CI/badge.svg?branch=master
    :target: https://github.com/dls-controls/scanspec/actions?query=workflow%3A%22Python+CI%22
    :alt: Build Status

.. |coverage| image:: https://dls-controls.github.io/scanspec/coverage.svg
    :target: https://github.com/dls-controls/scanspec/actions?query=workflow%3A%22Python+CI%22
    :alt: Test Coverage

.. |pypi_version| image:: https://img.shields.io/pypi/v/scanspec.svg
    :target: https://pypi.org/project/scanspec
    :alt: Latest PyPI version

.. |license| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
    :target: https://opensource.org/licenses/Apache-2.0
    :alt: Apache License

..
    These definitions are used when viewing README.rst and will be replaced
    when included in index.rst

.. |plot| image:: https://raw.githubusercontent.com/dls-controls/scanspec/0.1/docs/images/plot_spec.png
