.. _serialize-a-spec:

How to Serialize and Deserialize a Spec
=======================================

Lets start with an example `Spec`.

>>> spec = Line("y", 4, 5, 6) * Line("x", 1, 2, 3)

This Spec has a `repr` that shows its parameters it was instantiated with:

>>> spec
Product(outer=Line(key='y', start=4.0, stop=5.0, num=6), inner=Line(key='x', start=1.0, stop=2.0, num=3))

How to Serialize
----------------

We can recursively serialize it to a dictionary:

>>> spec.dict()
{'type': 'Product', 'outer': {'type': 'Line', 'key': 'y', 'start': 4.0, 'stop': 5.0, 'num': 6}, 'inner': {'type': 'Line', 'key': 'x', 'start': 1.0, 'stop': 2.0, 'num': 3}}

Or directly to JSON:

>>> spec.json()
'{"type": "Product", "outer": {"type": "Line", "key": "y", "start": 4.0, "stop": 5.0, "num": 6}, "inner": {"type": "Line", "key": "x", "start": 1.0, "stop": 2.0, "num": 3}}'

How to Deserialize
------------------

We can turn this back into a spec using `spec_from_json` or `spec_from_dict`:

>>> spec_from_json('{"type": "Product", "outer": {"type": "Line", "key": "y", "start": 4.0, "stop": 5.0, "num": 6}, "inner": {"type": "Line", "key": "x", "start": 1.0, "stop": 2.0, "num": 3}}')
Product(outer=Line(key='y', start=4.0, stop=5.0, num=6), inner=Line(key='x', start=1.0, stop=2.0, num=3))
>>> spec_from_dict({'type': 'Product', 'outer': {'type': 'Line', 'key': 'y', 'start': 4.0, 'stop': 5.0, 'num': 6}, 'inner': {'type': 'Line', 'key': 'x', 'start': 1.0, 'stop': 2.0, 'num': 3}})
Product(outer=Line(key='y', start=4.0, stop=5.0, num=6), inner=Line(key='x', start=1.0, stop=2.0, num=3))

How to Specify JSON from the class definitions
----------------------------------------------

Every Spec lists in its documentation a list of parameters and types. The JSON
representation is the dictionary of these parameters with one additional entry,
``type`` which must be equal to the classname. This allows the deserialization
code to pick the correct class to deserialize to.
