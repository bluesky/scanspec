.. _graphql-service:

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
which will allow you to send requests to the server.

.. _OpenAPI: https://www.openapis.org/
.. _`Swagger UI`: https://swagger.io/tools/swagger-ui/
