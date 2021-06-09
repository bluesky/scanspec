import base64
from dataclasses import dataclass
from typing import Any, List, Optional

import aiohttp_cors
import graphql
import numpy as np
from aiohttp import web
from apischema.graphql import graphql_schema, resolver
from graphql_server.aiohttp.graphqlview import GraphQLView, _asyncify

from scanspec.core import Dimension, Path
from scanspec.specs import Spec


@dataclass
class Points:
    """ A collection of singular or multidimensional locations in scan space"""

    def __init__(self, points: Optional[np.ndarray]):
        self._points = points

    @resolver
    def string(self) -> Optional[str]:
        return np.array2string(self._points)

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
            assert np.dtype(self._points[0]) == np.dtype(np.float64)
            return base64.b64encode(self._points.tobytes()).decode("utf-8")

    # Self b64 decoder for testing purposes
    @resolver
    def b64Decode(self) -> Optional[str]:
        if self._points is None:
            return None
        else:
            r = np.dtype(self._points[0])
            s = base64.decodebytes(base64.b64encode(self._points.tobytes()))
            t = np.frombuffer(s, dtype=r)
            return np.array2string(t)


@dataclass
class AxisFrames:
    """ A collection of frames (comprising midpoints with lower and upper bounds)
    present in each axis of the Spec
    """

    axis: str
    """A fixed reference that can be scanned. i.e. a motor, time or
    number of repetitions.
    """
    lower: Optional[Points]
    """The lower bounds of each midpoint (used when fly scanning)"""
    midpoints: Optional[Points]
    """The centre points of the scan"""
    upper: Optional[Points]
    """The upper bounds of each midpoint (used when fly scanning)"""


@dataclass
class PointsResponse:
    """ The highest level of the getPoints query, allowing users to customise their
    return data from the points present in the scan to some metadata about them
    """

    axes: List[AxisFrames]
    total_frames: int
    returned_frames: int


# Chacks that the spec will produce a valid scan
def validate_spec(spec: Spec) -> Any:
    """ A query used to confirm whether or not the Spec will produce a viable scan"""
    # apischema will do all the validation for us
    return spec.serialize()


# Returns a full list of points for each axis in the scan
# TODO Update max_frames with a more sophisticated method of reducing scan points
def get_points(spec: Spec, max_frames: Optional[int] = 200000) -> PointsResponse:
    """A query that takes a Spec and calculates the points present in the scan
    (for each axis) plus some metadata about the points.

    Arguments:
            [spec]: [The specification of the scan]
            [max_frames]: [The maximum number of frames the user wishes to receive]

    Returns:
        [PointsResponse]: [A dataclass containing information about the scan points
                            present in the spec]
    """
    dims = spec.create_dimensions()  # Grab dimensions from spec
    path = Path(dims)  # Convert to a path
    total_frames = len(path)  # Capture the total length of the path

    # Limit the consumed data to the max_frames argument
    # # WARNING: path object is consumed after this statement
    if max_frames is None:
        # Return as many frames as possible
        returned_frames = len(path)
        chunk = path.consume(len(path))

    elif max_frames >= len(path):
        # Return all of the frames within that selection
        returned_frames = len(path)
        chunk = path.consume(len(path))

    else:
        # Cap the frames by the max limit
        path = reduce_frames(dims, max_frames)
        returned_frames = len(path)
        chunk = path.consume()

    # POINTS
    scan_points = [
        AxisFrames(
            axis,
            Points(chunk.lower.get(axis)),
            Points(chunk.midpoints.get(axis)),
            Points(chunk.upper.get(axis)),
        )
        for axis in spec.axes()
    ]

    return PointsResponse(scan_points, total_frames, returned_frames)


# Define the schema
schema = graphql_schema(query=[validate_spec, get_points])


def reduce_frames(dims: List[Dimension], max_frames: int) -> Path:
    """Removes frames from a spec such that it produces a number that is
    closest to the max points value

    Args:
        dims (List[Dimension]): A dimension object created by a spec
        max_frames (int): The maximum number of frames the user wishes to be returned

    Returns:
        Path: A consumable object containing the expanded dimension with reduced frames
    """
    # Calculate the total number of frames
    num_frames = 1
    for dim in dims:
        num_frames *= len(dim)

    # Need each dim to be this much smaller
    ratio = 1 / np.power(max_frames / num_frames, 1 / len(dims))

    sub_dims = [sub_sample(d, ratio) for d in dims]
    return Path(sub_dims)


def sub_sample(dim: Dimension, ratio: float) -> Dimension:
    """Removes frames from a dimension whilst preserving its core structure

    Args:
        dim (Dimension): the dimension object to be reduced
        ratio (float): the reduction ratio of the dimension
    Returns:
        Dimension: the reduced dimension
    """
    num_indexes = int(len(dim) / ratio)
    indexes = np.linspace(0, len(dim) - 1, num_indexes).astype(np.int32)
    return dim[indexes]


def schema_text() -> str:
    return graphql.utilities.print_schema(schema)


def run_app(cors=False, port=8080):
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

    web.run_app(app, port=port)
