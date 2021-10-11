Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.


Unreleased_
-----------

Nothing yet


0.5.1_ - 2021-10-11
-------------------

Fixed:

- Pipenv issue fixed to allow internal DLS build


0.5_ - 2021-10-08
-----------------

Added:

- `Dimension.gap` to see whether there is a gap between frames
- `Concat.gap` to allow a gap to be explicitly inserted between sections
- Generics: `Frames`, `AxesPoints`, `Path`, `Midpoints`, `Spec` and `Regions` (and their subclasses) can 
  now be defined in terms of a type of axis (e.g. `str` or `Motor`)
- `GraphQL serialization function for Spec <../../pull/36>`_

Changed:

- Removed `repeat(spec, 3)` and replaced with `Repeat(spec, 3)` and `3 * spec` shortcut
- Renamed `Dimension -> Frames` and `Dimension(snake=True) -> SnakedFrames`


0.4_ - 2021-08-13
-----------------

Fixed:

- Concat now behaves correctly on stacked dimensions

Changed:

- 'semiaxis' -> 'radius' in regions.Ellipse
- Use sub sample method of reducing points in service
- TIME is now DURATION


0.3_ - 2021-06-01
-----------------

Added:

- 'get_points' query that can be used to return a list of plottable/scannable points from GraphQL
- data structure 'PointsRequest' that includes multiple fields directly relating to fields within GDA
- new naming standard
- documentation on how to use get_points in graphiql


0.2_ - 2020-12-18
-----------------

Added:

- repeat() Spec
- more documentation


0.1.1_ - 2020-12-10
-------------------

Fixed:

- README so it works on PyPI


0.1_ - 2020-12-10
-----------------

- Initial release


.. _Unreleased: ../../compare/0.5.1...HEAD
.. _0.5.1: ../../compare/0.5...0.5.1
.. _0.5: ../../compare/0.4...0.5
.. _0.4: ../../compare/0.3...0.4
.. _0.3: ../../compare/0.2...0.3
.. _0.2: ../../compare/0.1.1...0.2
.. _0.1.1: ../../compare/0.1...0.1.1
.. _0.1: ../../releases/tag/0.1
