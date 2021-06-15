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

    def __init__(self, points: np.ndarray):
        self._points = points

    @resolver
    def string(self) -> str:
        return np.array2string(self._points)

    @resolver
    def float_list(self) -> List[float]:
        return self._points.tolist()

    @resolver
    def b64(self) -> str:
        # make sure the data is sent as float64
        assert self._points.dtype == np.dtype(np.float64)
        return base64.b64encode(self._points.tobytes()).decode()

    def get_points(self) -> np.ndarray:
        return self._points


@dataclass
class AxisFrames:
    """ A collection of frames (comprising midpoints with lower and upper bounds)
    present in each axis of the Spec
    """

    axis: str
    """A fixed reference that can be scanned. i.e. a motor, time or
    number of repetitions.
    """
    lower: Points
    """The lower bounds of each frame (used when fly scanning)"""
    midpoints: Points
    """The midpoints of each frame"""
    upper: Points
    """The upper bounds of each frame (used when fly scanning)"""

    @resolver
    def smallest_step(self) -> float:
        """The smallest step between midpoints in this axis"""
        return calc_smallest_step([self.midpoints.get_points()])


@dataclass
class PointsResponse:
    """ The highest level of the getPoints query, allowing users to customise their
    return data from the points present in the scan to some metadata about them
    """

    total_frames: int
    returned_frames: int

    def __init__(self, chunk: Dimension, total_frames: int):
        self.total_frames = total_frames
        """The number of frames present across the entire spec"""
        self.returned_frames = len(chunk)
        """The number of frames returned by the getPoints query
        (controlled by the max_points argument)"""
        self._chunk = chunk

    @resolver
    def axes(self) -> List[AxisFrames]:
        """A list of all of the points present in the spec per axis"""
        return [
            AxisFrames(
                axis,
                Points(self._chunk.lower[axis]),
                Points(self._chunk.midpoints[axis]),
                Points(self._chunk.upper[axis]),
            )
            for axis in self._chunk.midpoints
        ]

    @resolver
    def smallest_abs_step(self) -> float:
        """The smallest step between midpoints across ALL axes in the scan"""
        return calc_smallest_step(list(self._chunk.midpoints.values()))


def calc_smallest_step(points: List[np.ndarray]) -> float:
    # Calc abs diffs of all axes
    absolute_diffs = [abs_diffs(axis_midpoints) for axis_midpoints in points]
    # Return the smallest value (Aka. smallest step)
    return np.amin(np.linalg.norm(absolute_diffs, axis=0))


def abs_diffs(array: np.ndarray) -> np.ndarray:
    """Calculates the absolute differences between adjacent elements in the array
    used as part of the smallest step calculation for each axis

    Args:
        array (ndarray): A 1xN array of numerical values

    Returns:
        ndarray: A newly constucted array of absolute differences
    """
    # [array[1] - array[0], array[2] - array[1], ...]
    adjacent_diffs = array[1:] - array[:-1]
    return np.absolute(adjacent_diffs)


# Checks that the spec will produce a valid scan
def validate_spec(spec: Spec) -> Any:
    """ A query used to confirm whether or not the Spec will produce a viable scan"""
    # apischema will do all the validation for us
    return spec.serialize()


# Returns a full list of points for each axis in the scan
# TODO Update max_frames with a more sophisticated method of reducing scan points
def get_points(spec: Spec, max_frames: Optional[int] = 100000) -> PointsResponse:
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

    # TOTAL FRAMES
    total_frames = len(path)  # Capture the total length of the path

    # MAX FRAMES
    # Limit the consumed data by the max_frames argument
    if max_frames and (max_frames < len(path)):
        # Cap the frames by the max limit
        path = reduce_frames(dims, max_frames)
    # WARNING: path object is consumed after this statement
    chunk = path.consume(max_frames)

    return PointsResponse(chunk, total_frames)


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
