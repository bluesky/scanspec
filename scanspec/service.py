from typing import Any
from aiohttp import web
from apischema.graphql import graphql_schema
from graphql_server.aiohttp import GraphQLView

from scanspec.specs import Spec


def validate_spec(spec: Spec) -> Any:
    # apischema will do all the validation for us
    return spec.serialize()


def run_app():
    schema = graphql_schema(query=[validate_spec])
    app = web.Application()

    GraphQLView.attach(app, schema=schema, graphiql=True)

    # Optional, for adding batch query support (used in Apollo-Client)
    # GraphQLView.attach(app, schema=schema, batch=True, route_path="/graphql/batch")

    web.run_app(app)
