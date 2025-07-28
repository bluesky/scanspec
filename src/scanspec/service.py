"""FastAPI service to query information about Specs."""

import base64
import json
from collections.abc import Mapping
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from pydantic import Field
from pydantic.dataclasses import dataclass

from scanspec.core import AxesPoints, Dimension, Path, stack2dimension

from .specs import Line, Spec

app = FastAPI(version="0.1.1")

#
# Data Model
#


#: A set of points, that can be returned in various formats
Points = str | list[float]


@dataclass
class ValidResponse:
    """Response model for spec validation."""

    input_spec: Spec[str] = Field(description="The input scanspec")
    valid_spec: Spec[str] = Field(description="The validated version of the spec")


class PointsFormat(str, Enum):
    """Formats in which we can return points."""

    STRING = "STRING"
    FLOAT_LIST = "FLOAT_LIST"
    BASE64_ENCODED = "BASE64_ENCODED"


@dataclass
class PointsRequest:
    """A request for generated scan points."""

    spec: Spec[str] = Field(description="The spec from which to generate points")
    max_frames: int | None = Field(
        description="The maximum number of points to return, if None will return "
        "as many as calculated",
        default=100000,
    )
    format: PointsFormat = Field(
        description="The format in which to output the points data",
        default=PointsFormat.FLOAT_LIST,
    )


@dataclass
class GeneratedPointsResponse:
    """Base class for responses that include generated point data."""

    total_frames: int = Field(description="Total number of frames in spec")
    returned_frames: int = Field(
        description="Total of number of frames in this response, may be "
        "less than total_frames due to downsampling etc."
    )
    format: PointsFormat = Field(description="Format of returned point data")


@dataclass
class MidpointsResponse(GeneratedPointsResponse):
    """Midpoints of a generated scan."""

    midpoints: Mapping[str, Points] = Field(
        description="The midpoints of scan frames for each axis"
    )


@dataclass
class BoundsResponse(GeneratedPointsResponse):
    """Bounds of a generated scan."""

    lower: Mapping[str, Points] = Field(
        description="Lower bounds of scan frames if different from midpoints"
    )
    upper: Mapping[str, Points] = Field(
        description="Upper bounds of scan frames if different from midpoints"
    )


@dataclass
class GapResponse:
    """Presence of gaps in a generated scan."""

    gap: list[bool] = Field(
        description="Boolean array indicating if there is a gap between each frame"
    )


@dataclass
class SmallestStepResponse:
    """Information about the smallest steps between points in a spec."""

    absolute: float = Field(
        description="Absolute smallest distance between two points on a single axis"
    )
    per_axis: Mapping[str, float] = Field(
        description="Smallest distance between two points on each axis"
    )


#
# API Routes
#

_EXAMPLE_SPEC = Line("y", 0.0, 10.0, 3) * Line("x", 0.0, 10.0, 4)
_EXAMPLE_POINTS_REQUEST = PointsRequest(
    _EXAMPLE_SPEC, max_frames=1024, format=PointsFormat.FLOAT_LIST
)


@app.post("/valid", response_model=ValidResponse)
def valid(
    spec: Spec[str] = Body(..., examples=[_EXAMPLE_SPEC]),
) -> ValidResponse:
    """Validate wether a ScanSpec[str] can produce a viable scan.

    Args:
        spec: The scanspec to validate

    Returns:
        ValidResponse: A canonical version of the spec if it is valid.
            An error otherwise.

    """
    valid_spec = Spec.deserialize(spec.serialize())
    return ValidResponse(spec, valid_spec)


@app.post("/midpoints", response_model=MidpointsResponse)
def midpoints(
    request: PointsRequest = Body(
        ...,
        examples=[_EXAMPLE_POINTS_REQUEST],
    ),
) -> MidpointsResponse:
    """Generate midpoints from a scanspec.

    A scanspec can produce bounded points (i.e. a point is valid if an
    axis is between a minimum and and a maximum, see /bounds). The midpoints
    are the middle of each set of bounds.

    Args:
        request: Scanspec and formatting info.

    Returns:
        MidpointsResponse: Midpoints of the scan

    """
    chunk, total_frames = _to_chunk(request)
    return MidpointsResponse(
        total_frames,
        request.max_frames or total_frames,
        request.format,
        _format_axes_points(chunk.midpoints, request.format),
    )


@app.post("/bounds", response_model=BoundsResponse)
def bounds(
    request: PointsRequest = Body(
        ...,
        examples=[_EXAMPLE_POINTS_REQUEST],
    ),
) -> BoundsResponse:
    """Generate bounds from a scanspec.

    A scanspec can produce points with lower and upper bounds.

    Args:
        request: Scanspec and formatting info.

    Returns:
        BoundsResponse: Bounds of the scan

    """
    chunk, total_frames = _to_chunk(request)
    return BoundsResponse(
        total_frames,
        request.max_frames or total_frames,
        request.format,
        _format_axes_points(chunk.lower, request.format),
        _format_axes_points(chunk.upper, request.format),
    )


@app.post("/gap", response_model=GapResponse)
def gap(
    spec: Spec[str] = Body(
        ...,
        examples=[_EXAMPLE_SPEC],
    ),
) -> GapResponse:
    """Generate gaps from a scanspec.

    A scanspec may indicate if there is a gap between two frames.
    The array returned corresponds to whether or not there is a gap
    after each frame.

    Args:
        spec: Scanspec and formatting info.

    Returns:
        GapResponse: Bounds of the scan

    """
    dims = spec.calculate()  # Grab dimensions from spec
    path = Path(dims)  # Convert to a path
    gap = list(path.consume().gap)
    return GapResponse(gap)


@app.post("/smalleststep", response_model=SmallestStepResponse)
def smallest_step(
    spec: Spec[str] = Body(..., examples=[_EXAMPLE_SPEC]),
) -> SmallestStepResponse:
    """Calculate the smallest step in a scan, both absolutely and per-axis.

    Ignore any steps of size 0.

    Args:
        spec: The spec of the scan

    Returns:
        SmallestStepResponse: A description of the smallest steps in the spec

    """
    dims = spec.calculate()  # Grab dimensions from spec
    path = Path(dims)  # Convert to a path
    chunk = path.consume()

    absolute = _calc_smallest_step(list(chunk.midpoints.values()))
    per_axis = {
        axis: _calc_smallest_step([chunk.midpoints[axis]]) for axis in chunk.axes()
    }

    return SmallestStepResponse(absolute, per_axis)


#
# Utility Functions
#


def _to_chunk(request: PointsRequest) -> tuple[Dimension[str], int]:
    spec = Spec.deserialize(request.spec)
    dims = spec.calculate()  # Grab dimensions from spec
    path = Path(dims)  # Convert to a path

    # TOTAL FRAMES
    total_frames = len(path)  # Capture the total length of the path

    # MAX FRAMES
    # Limit the consumed data by the max_frames argument
    max_frames = request.max_frames
    if max_frames and (max_frames < len(path)):
        # Cap the frames by the max limit
        path = _reduce_frames(dims, max_frames)

    lengths = np.array([len(f) for f in path.stack])
    indices = np.arange(0, int(np.prod(lengths)))
    # WARNING: path object is consumed after this statement
    return stack2dimension(path.stack, indices, lengths), total_frames


def _format_axes_points(
    axes_points: AxesPoints[str], format: PointsFormat
) -> Mapping[str, Points]:
    """Convert points to a requested format.

    Args:
        axes_points: The points to convert
        format: The target format

    Raises:
        KeyError: If the function does not support the given format

    Returns:
        Mapping[str, Points]: A mapping of axis to formatted points.

    """
    if format is PointsFormat.FLOAT_LIST:
        return {axis: list(points) for axis, points in axes_points.items()}
    elif format is PointsFormat.STRING:
        return {axis: str(points) for axis, points in axes_points.items()}
    elif format is PointsFormat.BASE64_ENCODED:
        return {
            axis: base64.b64encode(points.tobytes()).decode()
            for axis, points in axes_points.items()
        }
    else:
        raise KeyError(f"Unknown format: {format}")


def _reduce_frames(stack: list[Dimension[str]], max_frames: int) -> Path[str]:
    """Removes frames from a spec so len(path) < max_frames.

    Args:
        stack: A stack of Dimension created by a spec
        max_frames: The maximum number of frames the user wishes to be returned

    """
    # Calculate the total number of frames
    num_frames = 1
    for frames in stack:
        num_frames *= len(frames)

    # Need each dim to be this much smaller
    ratio = 1 / np.power(max_frames / num_frames, 1 / len(stack))

    sub_frames = [_sub_sample(f, ratio) for f in stack]
    return Path(sub_frames)


def _sub_sample(frames: Dimension[str], ratio: float) -> Dimension[str]:
    """Provides a sub-sample Dimension object whilst preserving its core structure.

    Args:
        frames: the Dimension object to be reduced
        ratio: the reduction ratio of the dimension

    """
    num_indexes = int(len(frames) / ratio)
    indexes = np.linspace(0, len(frames) - 1, num_indexes, dtype=np.int32)
    return frames.extract(indexes, calculate_gap=False)


def _calc_smallest_step(points: list[npt.NDArray[np.float64]]) -> float:
    # Calc abs diffs of all axes, ignoring any zero values
    absolute_diffs = [_abs_diffs(axis_midpoints) for axis_midpoints in points]
    # Normalize and remove zeros
    norm_diffs = np.linalg.norm(absolute_diffs, axis=0)
    norm_diffs = norm_diffs[norm_diffs > 0.0]
    # Return the smallest value (Aka. smallest step)
    return np.amin(norm_diffs)


def _abs_diffs(array: npt.NDArray[np.number[Any]]) -> npt.NDArray[np.number[Any]]:
    """Calculates the absolute differences between adjacent elements in the array.

    Args:
        array: A 1xN array of numerical values

    Returns:
        A newly constucted array of absolute differences

    """
    # [array[1] - array[0], array[2] - array[1], ...]
    adjacent_diffs = array[1:] - array[:-1]
    return np.absolute(adjacent_diffs)


def run_app(cors: bool = False, port: int = 8080) -> None:
    """Run an application providing the scanspec service."""
    if cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    import uvicorn

    uvicorn.run(app=app, port=port)


def scanspec_schema_text() -> str:
    """Generate the OpenAPI schema for the service as a string.

    Returns:
        str: The OpenAPI schema

    """
    return json.dumps(
        get_openapi(
            title=app.title,
            version=app.version,
            openapi_version=app.openapi_version,
            description=app.description,
            routes=app.routes,
        )
    )
