.. _serialize-a-spec:

How to Serialize and Deserialize a Spec
=======================================

Lets start with an example `Spec`.

>>> from scanspec.specs import Line, Spec
>>> spec = Line("y", 4, 5, 6) * Line("x", 1, 2, 3)

This Spec has a `repr` that shows its parameters it was instantiated with:

>>> spec
Product(outer=Line(axis='y', start=4.0, stop=5.0, num=6), inner=Line(axis='x', start=1.0, stop=2.0, num=3))


How to Serialize
----------------

We can recursively serialize it to a dictionary:

>>> spec.serialize()
{'outer': {'axis': 'y', 'start': 4.0, 'stop': 5.0, 'num': 6, 'type': 'Line'}, 'inner': {'axis': 'x', 'start': 1.0, 'stop': 2.0, 'num': 3, 'type': 'Line'}, 'type': 'Product'}

How to Deserialize
------------------

We can turn this back into a spec using `Spec.deserialize`:

>>> Spec.deserialize({'outer': {'axis': 'y', 'start': 4.0, 'stop': 5.0, 'num': 6, 'type': 'Line'}, 'inner': {'axis': 'x', 'start': 1.0, 'stop': 2.0, 'num': 3, 'type': 'Line'}, 'type': 'Product'})
Product(outer=Line(axis='y', start=4.0, stop=5.0, num=6), inner=Line(axis='x', start=1.0, stop=2.0, num=3))
