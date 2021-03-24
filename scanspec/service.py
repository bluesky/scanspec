from typing import Any, Dict, List

import aiohttp_cors
import numpy as np
from aiohttp import web
from apischema.graphql import graphql_schema
from graphql_server.aiohttp.graphqlview import GraphQLView, _asyncify

from scanspec.specs import Spec


def validate_spec(spec: Spec) -> Any:
    # apischema will do all the validation for us
    return spec.serialize()


# TODO adjust to return a reduced set of scanPoints
def get_points(spec: Spec) -> List[Dict[str, np.ndarray]]:
    # apischema will do all the validation for us

    # Grab positions from spec
    dims = spec.create_dimensions()
    # Take positions at each dimension and output as a list
    scanPoints = []
    for i in range(len(dims)):
        scanPoints.append(dims[i].positions)

    return scanPoints


schema = graphql_schema(query=[validate_spec, get_points])


def run_app(cors=False):
    app = web.Application()

    view = GraphQLView(schema=schema, graphiql=True)

    # Make GraphQLView compatible with aiohttp-cors
    # https://github.com/aio-libs/aiohttp-cors/issues/241#issuecomment-514278001
    for method in ("GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"):  # no OPTIONS
        app.router.add_route(method, "/graphql", _asyncify(view), name="graphql")

    # Optional, for adding batch query support (used in Apollo-Client)
    # GraphQLView.attach(app, schema=schema, batch=True, route_path="/graphql/batch")

    if cors:
        # Configure default CORS settings.
        cors_config = aiohttp_cors.setup(
            app,
            defaults={
                "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True, expose_headers="*", allow_headers="*",
                )
            },
        )

        # Configure CORS on all routes.
        for route in list(app.router.routes()):
            cors_config.add(route)

    web.run_app(app)
