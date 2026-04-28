.. _serialize-a-spec:

How to Serialize and Deserialize a Spec
=======================================

Lets start with an example `Spec`.

>>> from scanspec.specs import Linspace, Spec
>>> spec = Linspace("y", 4, 5, 6) * Linspace("x", 1, 2, 3)

This Spec has a `repr` that shows its parameters it was instantiated with:

>>> spec
Product(outer=Linspace(axis='y', start=4.0, stop=5.0, num=6), inner=Linspace(axis='x', start=1.0, stop=2.0, num=3), gap=True)


How to Serialize
----------------

We can recursively serialize it to a dictionary:

>>> spec.serialize()
{'outer': {'axis': 'y', 'start': 4.0, 'stop': 5.0, 'num': 6, 'type': 'Linspace'}, 'inner': {'axis': 'x', 'start': 1.0, 'stop': 2.0, 'num': 3, 'type': 'Linspace'}, 'gap': True, 'type': 'Product'}

How to Deserialize
------------------

We can turn this back into a spec using `Spec.deserialize`:

>>> Spec.deserialize({'outer': {'axis': 'y', 'start': 4.0, 'stop': 5.0, 'num': 6, 'type': 'Linspace'}, 'inner': {'axis': 'x', 'start': 1.0, 'stop': 2.0, 'num': 3, 'type': 'Linspace'}, 'type': 'Product'})
Product(outer=Linspace(axis='y', start=4.0, stop=5.0, num=6), inner=Linspace(axis='x', start=1.0, stop=2.0, num=3), gap=True)
