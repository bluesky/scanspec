import base64
from dataclasses import dataclass
from typing import Any, List, Optional

import aiohttp_cors
import graphql
from aiohttp import web
from apischema.graphql import graphql_schema, resolver
from graphql_server.aiohttp.graphqlview import GraphQLView, _asyncify
from numpy import array2string, dtype, float64, frombuffer, ndarray

from scanspec.core import Path
from scanspec.specs import Spec

# See confluence page for clarity on naming conventions and data structure:
# https://confluence.diamond.ac.uk/display/SSCC/Data+Structure+and+Naming+Conventions


# The lowest query level
@dataclass
class Points:
    def __init__(self, points: Optional[ndarray]):
        self._points = points

    @resolver
    def string(self) -> Optional[str]:
        return array2string(self._points)

    @resolver
    def float_list(self) -> Optional[List[float]]:
        if self._points is None:
            return None
        else:
            return self._points.tolist()

    @resolver
    def b64(self) -> Optional[str]:
        if self._points is None:
            return None
        else:
            # make sure the data is sent as float64
            assert dtype(self._points[0]) == dtype(float64)
            return base64.b64encode(self._points.tobytes()).decode("utf-8")

    # Self b64 decoder for testing purposes
    @resolver
    def b64Decode(self) -> Optional[str]:
        if self._points is None:
            return None
        else:
            r = dtype(self._points[0])
            s = base64.decodebytes(base64.b64encode(self._points.tobytes()))
            t = frombuffer(s, dtype=r)
            return array2string(t)


@dataclass
class axisFrames:
    axis: str
    lower: Optional[Points]
    middle: Optional[Points]
    upper: Optional[Points]


# The highest query level
@dataclass
class pointsRequest:
    axes: List[axisFrames]
    num_points: int


# Chacks that the spec will produce a valid scan
def validate_spec(spec: Spec) -> Any:
    # apischema will do all the validation for us
    return spec.serialize()


# Returns a full list of points for each axis in the scan
# TODO adjust to return a reduced set of scanPoints
def get_points(spec: Spec) -> pointsRequest:

    dims = spec.create_dimensions()  # Grab dimensions from spec
    path = Path(dims)  # Convert to a path
    num_points = len(path)  # Capture the length of the path

    # WARNING: path object is consumed after this line
    chunk = path.consume()

    # POINTS #
    scan_points = []
    # For every dimension of the scan...
    for axis in chunk.middle:
        # Extract the upper, lower and middle points
        a = axisFrames(
            axis,
            Points(chunk.lower.get(axis)),
            Points(chunk.middle.get(axis)),
            Points(chunk.upper.get(axis)),
        )
        # Append the information as a list of frames per axis
        scan_points.append(a)

    return pointsRequest(scan_points, num_points)


# Define the schema
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
