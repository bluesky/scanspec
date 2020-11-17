scanspec
========

|build_status| |coverage| |pypi_version| |readthedocs| |license|

Specify step and flyscan paths using combinations of:

- Single or Multidimensional ScanSpecs like Line or Spiral
- Optionally Snaking
- Masks with multiple Regions

Serialize the ScanSpec rather than the points and reconstruct it on the
server. It can them be iterated over like a cycler_, or scan Dimensions
can be produced and sliced Views created to consume chunk by chunk.

.. _cycler: https://matplotlib.org/cycler/

============== ==============================================================
PyPI           ``pip install scanspec``
Source code    https://github.com/dls-controls/scanspec
Documentation  http://scanspec.readthedocs.io
============== ==============================================================

An example ScanSpec of a 2D snaked grid flyscan inside a circle spending 0.4s at each point looks like:

.. code:: python

    from scanspec.specs import Line, Snake, Mask, Static, TIME
    from scanspec.regions import Circle

    grid = Line(ymotor, 2.1, 3.8, 60) * Snake(Line(xmotor, 0.5, 1.5, 100))
    spec = Mask(grid + Static(TIME, 0.4), Circle(1.0, 2.8, radius=0.5))

You can then either iterate through it directly for convenience, or produce chunked Dimensions for performance:

.. code:: python

    for positions in spec:
        print(positions)
        # {ymotor: 2.1, xmotor: 0.5, TIME: 0.4}

    dims = spec.create_dimensions()
    dims[0].shape  # (4600,)
    dims[0].keys  # (ymotor, xmotor)

    view = View(dims, start=5, num=60)
    batch = view.get_batch(30)
    batch.positions # {xmotor: <ndarray len=30>, ymotor: <ndarray len=30>, TIME: <ndarray len=30>}
    batch.upper_bounds # bounds are same dimensionality as positions


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
