scanspec
========

|build_status| |coverage| |pypi_version| |readthedocs| |license|

Specify step and flyscan Paths using combinations of:

- Specs like Line or Spiral
- Optionally Snaking
- Zip or Product to compose
- Masks with multiple Regions

Serialize the Spec rather than the expanded Path and reconstruct it on the
server. It can them be iterated over like a cycler_, or scan Dimensions
can be produced and sliced Paths created to consume chunk by chunk.

.. _cycler: https://matplotlib.org/cycler/

============== ==============================================================
PyPI           ``pip install scanspec``
Source code    https://github.com/dls-controls/scanspec
Documentation  http://scanspec.readthedocs.io
============== ==============================================================

An example ScanSpec of a 2D snaked grid flyscan inside a circle spending 0.4s at each point looks like:

.. code:: python

    from scanspec.specs import Line, fly
    from scanspec.regions import Circle

    grid = Line(ymotor, 2.1, 3.8, 60) * ~Line(xmotor, 0.5, 1.5, 100)
    spec = fly(grid, 0.4) & Circle(xmotor, ymotor, 1.0, 2.8, radius=0.5)

You can then either iterate through the scan positions directly for convenience:
.. code:: python

    for positions in spec.positions():
        print(positions)
    # ...
    # {'y': 3.2813559322033896, 'x': 0.8838383838383839, Time(): 0.4}
    # {'y': 3.2813559322033896, 'x': 0.8737373737373737, Time(): 0.4}

or create a Path from the Dimensions and consume chunks of a given length from it for performance:

.. code:: python

    from scanspec.core import Path

    dims = spec.create_dimensions()
    len(dims[0].shape)  # 2696
    dims[0].keys()  # (ymotor, xmotor, Time())

    path = Path(dims, start=5, num=60)
    chunk = path.consume(30)
    chunk.positions  # {xmotor: <ndarray len=30>, ymotor: <ndarray len=30>, TIME: <ndarray len=30>}
    chunk.upper  # bounds are same dimensionality as positions


.. |build_status| image:: https://travis-ci.com/dls-controls/scanspec.svg?branch=master
    :target: https://travis-ci.com/dls-controls/scanspec
    :alt: Build Status

.. |coverage| image:: https://coveralls.io/repos/github/dls-controls/scanspec/badge.svg?branch=master
    :target: https://coveralls.io/github/dls-controls/scanspec?branch=master
    :alt: Test Coverage

.. |pypi_version| image:: https://badge.fury.io/py/scanspec.svg
    :target: https://badge.fury.io/py/scanspec
    :alt: Latest PyPI version

.. |readthedocs| image:: https://readthedocs.org/projects/scanspec/badge/?version=latest
    :target: http://scanspec.readthedocs.io
    :alt: Documentation

.. |license| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
    :target: https://opensource.org/licenses/Apache-2.0
    :alt: Apache License
