.. _rest-service:

Running a REST service for generating points
============================================

The `creating-a-spec` tutorial shows how you would use the commandline client to
plot a `Spec`. This maps nicely to using scanspec in a commandline utility, but
not if you want to expose those points to a web GUI. To do this we will bring up
a RESTful service, compatible with OpenAPI_, that allows a web GUI to request the 
points it would like to plot.

Running the server
------------------

In a terminal, run::

    $ scanspec service --cors
    ======== Running on http://localhost:8080 ========
    (Press CTRL+C to quit)

You can now open a browser to http://localhost:8080/docs and see a `Swagger UI`_ editor
which will allow you to send requests to the server using, for example, the ``curl`` command. 

.. seealso:: `../reference/rest_api`

.. _OpenAPI: https://www.openapis.org/
.. _`Swagger UI`: https://swagger.io/tools/swagger-ui/


Validating a Spec
-----------------

You can ensure a spec is valid by passing it to the ``/valid`` endpoint.

.. code:: shell

  curl -X 'POST' \
    'http://localhost:8080/valid' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "outer": {
      "axis": "y",
      "start": 0,
      "stop": 10,
      "num": 3,
      "type": "Line"
    },
    "inner": {
      "axis": "x",
      "start": 0,
      "stop": 10,
      "type": "Line"
    },
    "type": "Product"
  }'

Should return a 422 error code because the inner axis is missing a parameter.


Generating Midpoints
--------------------

There are several endpoints to inspect generated points. The most useful is ``/midpoints``, 
which gives the middle coordinate of each scan point, organised by axis name.

.. code:: shell

  curl -X 'POST' \
    'http://localhost:8080/midpoints' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "spec": {
      "outer": {
        "axis": "y",
        "start": 0,
        "stop": 10,
        "num": 3,
        "type": "Line"
      },
      "inner": {
        "axis": "x",
        "start": 0,
        "stop": 10,
        "num": 4,
        "type": "Line"
      },
      "type": "Product"
    },
    "max_frames": 1024,
    "format": "FLOAT_LIST"
  }'

Should output:

.. code:: JSON

  {"format": "FLOAT_LIST",
  "midpoints": {"x": [0.0,
                      3.333333333333333,
                      6.666666666666667,
                      10.000000000000002,
                      0.0,
                      3.333333333333333,
                      6.666666666666667,
                      10.000000000000002,
                      0.0,
                      3.333333333333333,
                      6.666666666666667,
                      10.000000000000002],
                "y": [0.0,
                      0.0,
                      0.0,
                      0.0,
                      5.0,
                      5.0,
                      5.0,
                      5.0,
                      10.0,
                      10.0,
                      10.0,
                      10.0]},
  "returned_frames": 1024,
  "total_frames": 12}
