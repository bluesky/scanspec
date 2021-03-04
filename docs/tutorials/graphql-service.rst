.. _graphql-service:

Running a GraphQL service for generating points
===============================================

The `creating-a-spec` tutorial shows how you would use the commandline client to
plot a `Spec`. This works for local usage, but if you had a web GUI that allowed
scans to be specified and results plotted you might want to overlay the data
with the requested scan points. To do this we will bring up a GraphQL_ service
that allows a web GUI to request

.. note::

    localhost


For example, our `snaked-grid` example from the `creating-a-spec` tutorial looks like:

.. graphiql::
    :query:
        {}
    :response:
        {}

.. _GraphQL: https://www.graphql.com/