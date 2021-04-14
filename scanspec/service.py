from dataclasses import dataclass
from typing import Any, List, Optional

import aiohttp_cors
import graphql
from aiohttp import web
from apischema.graphql import graphql_schema, resolver
from graphql_server.aiohttp.graphqlview import GraphQLView, _asyncify
from numpy import array2string, ndarray

from scanspec.core import Path
from scanspec.specs import Spec

# Current naming convention:

# - A position is a region in 1D space that is defined by its
#   upper, middle and lower bounds.

# - A bound defines the limitations of the region that a position encompasses. This
#   is controlled by its upper and lower bounds and centralized by its middle bound.

# - A point is the combination of two or more bounds, each from
#   a different position, to form a single location in space.


# Allows the user to specify the return type of a bound
@dataclass
class Bound:
    def __init__(self, bound: Optional[ndarray]):
        self._bound = bound

    @resolver
    def string(self) -> Optional[str]:
        return array2string(self._bound)

    @resolver
    def float_list(self) -> Optional[List[float]]:
        # Check required for mypy to detect that self._bound is able to use .tolist
        # when populated
        if self._bound is None:
            return None
        else:
            a = self._bound.tolist()
            return a


# Defines a Position: A region in space defined by its bounds
@dataclass
class Position:
    key: str
    lower: Optional[Bound]
    middle: Optional[Bound]
    upper: Optional[Bound]


def validate_spec(spec: Spec) -> Any:
    # apischema will do all the validation for us
    return spec.serialize()


# Returns a full list of points for every position in the scan
# TODO adjust to return a reduced set of scanPoints
def get_points(spec: Spec) -> List[Position]:
    # Grab positions from spec
    dims = spec.create_dimensions()
    # Take positions and convert to a list
    path = Path(dims)
    chunk = path.consume()

    scanPoints = []
    # For every dimension of the scan...
    for key in chunk.positions:
        # Assign the properties of that axis to a dataclass
        a = Position(
            key,
            Bound(chunk.lower.get(key)),
            Bound(chunk.positions.get(key)),  # TODO: CHANGE TO MIDDLE IN CORE.PY
            Bound(chunk.upper.get(key)),
        )
        # Append the information to a list of points for that axis
        scanPoints.append(a)
    return scanPoints


schema = graphql_schema(query=[validate_spec, get_points])


def schema_text() -> str:
    return graphql.utilities.print_schema(schema)


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
