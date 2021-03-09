.. _graphql-service:

Running a GraphQL service for generating points
===============================================

The `creating-a-spec` tutorial shows how you would use the commandline client to
plot a `Spec`. This maps nicely to using scanspec in a commanline utility, but
not if you want to expose those points to a web GUI. To do this we will bring up
a GraphQL_ service that allows a web GUI to request the points it would like to
plot.

Running the server
------------------

In a terminal, run:

    $ scanspec service --cors
    ======== Running on http://0.0.0.0:8080 ========
    (Press CTRL+C to quit)

You can now open a browser to http://0.0.0.0:8080/graphql and see a GraphiQL_ editor
which will allow you to send GraphQL queries to the server

.. note::

    All the examples below will embed a GraphiQL_ editor in your browser talking
    to the scanspec service, they are live and you can modify the queries and press
    the play button to get an updated response from the server. If the server is
    not running (or you didn't pass ``--cors`` to it) then the play button will not
    be visible.


Validating a Spec
-----------------

The first use case for our service is producing the canonical serialization of a
Spec. Some of the Specs like `Line.bounded` will return an instance of Line that
has different parameters to those that are passed.

For example, on the commandline we might do::

    >>> from scanspec.specs import Line
    >>> Line.bounded("x", 0, 1, 5).serialize()
    {'Line': {'key': 'x', 'start': 0.1, 'stop': 0.9, 'num': 5}}

.. seealso:: `serialize-a-spec`

The equivalent in our service is:

.. graphiql:: http://localhost:8080/graphql
    :query:
      {
        validateSpec(
          spec: {
            BoundedLine: {
              key: "x"
              lower: 0
              upper: 1
              num: 5
            }
          }
        )
      }
    :response:
      {
        "data": {
          "validateSpec": {
            "Line": {
              "key": "x",
              "start": 0.1,
              "stop": 0.9,
              "num": 5
            }
          }
        }
      }

Getting Points from a Spec
--------------------------

TODO

Content to move
---------------

When we move this sphinx extension into its own repo we will use the following to demo it:

.. graphiql:: https://countries.trevorblades.com/
    :query:
      {
        country(code: "BR") {
          name
          native
          capital
          emoji
          currency
          languages {
            code
            name
          }
        }
      }
    :response:
      {
        "data": {
          "country": {
            "name": "Brazil",
            "native": "Brasil",
            "capital": "Brasília",
            "emoji": "🇧🇷",
            "currency": "BRL",
            "languages": [
              {
                "code": "pt",
                "name": "Portuguese"
              }
            ]
          }
        }
      }


.. graphiql::
    :query:
      {
        country(code: "BR") {
          name
          native
          capital
          emoji
          currency
          languages {
            code
            name
          }
        }
      }
    :response:
      {
        "data": {
          "country": {
            "name": "Brazil",
            "native": "Brasil",
            "capital": "Brasília",
            "emoji": "🇧🇷",
            "currency": "BRL",
            "languages": [
              {
                "code": "pt",
                "name": "Portuguese"
              }
            ]
          }
        }
      }

.. _GraphQL: https://www.graphql.com/
.. _GraphiQL: https://github.com/graphql/graphiql/tree/main/packages/graphiql#readme