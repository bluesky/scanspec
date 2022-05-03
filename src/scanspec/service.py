import base64
from dataclasses import dataclass
from typing import Any, List, Optional

import aiohttp_cors
import graphql
import numpy as np
from aiohttp import web
from apischema import schema
from apischema.graphql import graphql_schema, resolver
from graphql_server.aiohttp.graphqlview import GraphQLView, _asyncify
from typing_extensions import Annotated as A

from scanspec.core import Frames, Path
from scanspec.specs import Spec


@dataclass
class Points:
    """A collection of singular or multidimensional points in scan space."""

    def __init__(self, points: np.ndarray):
        self._points = points

    @resolver
    def string(self) -> str:
        """Truncated string representation of array for debugging."""
        return np.array2string(self._points)

    @resolver
    def float_list(self) -> List[float]:
        """Float list representation of array."""
        return self._points.tolist()

    @resolver
    def b64(self) -> str:
        """Base64 encoded string representation of array."""
        # make sure the data is sent as float64
        assert self._points.dtype == np.dtype(np.float64)
        return base64.b64encode(self._points.tobytes()).decode()

    def get_points(self) -> np.ndarray:
        return self._points


@dataclass
class AxisFrames:
    """The scan points restricted to one particular axis."""

    axis: A[str, schema(description="An identifier for what to move")]
    lower: A[
        Points, schema(description="The lower bounds of each scan frame in each axis")
    ]
    midpoints: A[
        Points, schema(description="The midpoints of scan frames for each axis")
    ]
    upper: A[
        Points, schema(description="The upper bounds of each scan frame in each axis")
    ]

    @resolver
    def smallest_step(self) -> float:
        """The smallest step between midpoints in this axis."""
        return _calc_smallest_step([self.midpoints.get_points()])


@dataclass
class PointsResponse:
    """Information about the points provided by a spec."""

    total_frames: int
    returned_frames: int

    def __init__(self, chunk: Frames[str], total_frames: int):
        self.total_frames = total_frames
        """The number of frames present across the entire spec"""
        self.returned_frames = len(chunk)
        """The number of frames returned by the getPoints query
        (controlled by the max_points argument)"""
        self._chunk = chunk

    @resolver
    def axes(self) -> List[AxisFrames]:
        """A list of all of the points present in the spec per axis."""
        return [
            AxisFrames(
                axis,
                Points(self._chunk.lower[axis]),
                Points(self._chunk.midpoints[axis]),
                Points(self._chunk.upper[axis]),
            )
            for axis in self._chunk.axes()
        ]

    @resolver
    def smallest_abs_step(self) -> float:
        """The smallest step between midpoints across ALL axes in the scan."""
        return _calc_smallest_step(list(self._chunk.midpoints.values()))


def _calc_smallest_step(points: List[np.ndarray]) -> float:
    # Calc abs diffs of all axes
    absolute_diffs = [abs_diffs(axis_midpoints) for axis_midpoints in points]
    # Return the smallest value (Aka. smallest step)
    return np.amin(np.linalg.norm(absolute_diffs, axis=0))


def abs_diffs(array: np.ndarray) -> np.ndarray:
    """Calculates the absolute differences between adjacent elements in the array.

    Args:
        array: A 1xN array of numerical values

    Returns:
        A newly constucted array of absolute differences
    """
    # [array[1] - array[0], array[2] - array[1], ...]
    adjacent_diffs = array[1:] - array[:-1]
    return np.absolute(adjacent_diffs)


# Checks that the spec will produce a valid scan
def validate_spec(spec: Spec[str]) -> Any:
    """A query used to confirm whether or not the Spec will produce a viable scan."""
    # apischema will do all the validation for us
    return spec.serialize()


# Returns a full list of points for each axis in the scan
def get_points(spec: Spec[str], max_frames: Optional[int] = 100000) -> PointsResponse:
    """Calculate the frames present in the scan plus some metadata about the points.

    Args:
        spec: The specification of the scan
        max_frames: The maximum number of frames the user wishes to receive
    """
    dims = spec.calculate()  # Grab dimensions from spec
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
scanspec_schema = graphql_schema(query=[validate_spec, get_points])


def reduce_frames(stack: List[Frames[str]], max_frames: int) -> Path:
    """Removes frames from a spec so len(path) < max_frames.

    Args:
        stack: A stack of Frames created by a spec
        max_frames: The maximum number of frames the user wishes to be returned
    """
    # Calculate the total number of frames
    num_frames = 1
    for frames in stack:
        num_frames *= len(frames)

    # Need each dim to be this much smaller
    ratio = 1 / np.power(max_frames / num_frames, 1 / len(stack))

    sub_frames = [sub_sample(f, ratio) for f in stack]
    return Path(sub_frames)


def sub_sample(frames: Frames[str], ratio: float) -> Frames:
    """Provides a sub-sample Frames object whilst preserving its core structure.

    Args:
        frames: the Frames object to be reduced
        ratio: the reduction ratio of the dimension
    """
    num_indexes = int(len(frames) / ratio)
    indexes = np.linspace(0, len(frames) - 1, num_indexes, dtype=np.int32)
    return frames.extract(indexes, calculate_gap=False)


def scanspec_schema_text() -> str:
    """Return the text representation of the GraphQL schema."""
    return graphql.utilities.print_schema(scanspec_schema)


def run_app(cors=False, port=8080):
    """Run an application providing the scanspec service."""
    app = web.Application()

    view = GraphQLView(schema=scanspec_schema, graphiql=True)

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
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*",
                )
            },
        )

        # Configure CORS on all routes.
        for route in list(app.router.routes()):
            cors_config.add(route)

    web.run_app(app, port=port)
