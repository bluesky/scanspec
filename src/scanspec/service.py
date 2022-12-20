from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
from fastapi import Body, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import Field
from pydantic.dataclasses import dataclass

from scanspec.core import AxesPoints, Frames, Path

from .specs import Line, Spec

app = FastAPI()

#
# Data Model
#


#: A set of points, that can be returned in various formats
Points = Union[str, List[float]]


@dataclass
class ValidResponse:
    """Response model for spec validation."""

    input_spec: Mapping[str, Any] = Field(
        description="The input scanspec, potentially invalid"
    )
    valid_spec: Spec = Field(description="The validated version of the spec")


class PointsFormat(Enum):
    """Formats in which we can return points."""

    STRING = "STRING"
    FLOAT_LIST = "FLOAT_LIST"
    BASE64_ENCODED = "BASE64_ENCODED"


@dataclass
class PointsRequest:
    """A request for generated scan points."""

    spec: Spec = Field(description="The spec from which to generate points")
    format: PointsFormat = Field(
        description="The format in which to output the points data",
        default=PointsFormat.FLOAT_LIST,
    )
    max_frames: Optional[int] = Field(
        description="The maximum number of points to return, if None will return "
        "as many as calculated",
        default=100000,
    )


@dataclass
class PointsResponse:
    """Generated scan points with metadata."""

    total_frames: int = Field(description="Total number of frames in spec")
    returned_frames: int = Field(
        description="Total of number of frames in this response, may be "
        "less than total_frames due to downsampling etc."
    )
    format: PointsFormat = Field(description="Format of returned point data")
    axes: List[str] = Field(description="Names of axes")
    lower: Dict[str, Points] = Field(
        description="Lower bounds of scan frames if different from midpoints"
    )
    midpoints: Dict[str, Points] = Field(
        description="The midpoints of scan frames for each axis"
    )
    upper: Dict[str, Points] = Field(
        description="Upper bounds of scan frames if different from midpoints"
    )
    gap: Optional[List[bool]] = Field(
        description="Boolean array indicating if there is a gap between each frame"
    )

    @classmethod
    def from_path(cls, frames: Frames) -> "PointsResponse":
        return cls(
            len(frames.midpoints),
            len(frames.midpoints),
            frames.axes,
            frames.lower,
            frames.midpoints,
            frames.upper,
            frames.gap,
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

EXAMPLE_SPEC = Line("y", 0.0, 10.0, 16) * Line("x", 0.0, 10.0, 8)


@app.get("/valid", response_model=ValidResponse)
def valid(spec: Mapping[str, Any] = Body(..., example=EXAMPLE_SPEC)) -> ValidResponse:
    """Validate wether a ScanSpec can produce a viable scan.

    Args:
        spec (Spec): The scanspec to validate

    Returns:
        ValidResponse: A canonical version of the spec if it is valid.
            An error otherwise.
    """
    valid_spec = Spec.deserialize(spec)
    return ValidResponse(spec, valid_spec)


@app.get("/points", response_model=PointsResponse)
def points(
    request: PointsRequest = Body(
        ...,
        example=PointsRequest(EXAMPLE_SPEC, PointsFormat.FLOAT_LIST, max_frames=1024),
    )
) -> PointsResponse:
    """Generate the points of a scanspec.

    Points are generated in "frames" which map axes to coordinates at a particular time.

    Args:
        spec (Spec): ScanSpec to validate
        format (PointsFormat, optional): Format of returned points.
            Defaults to FLOAT_LIST.
        max_frames (Optional[int], optional): Maximum number of frames to return.
            If the spec generates more points than the maximum, the return value
            will be downsampled. If None is passed, all frames will be returned
            no matter what. Defaults to 100000.

    Returns:
        PointsResponse: _description_
    """
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
    # WARNING: path object is consumed after this statement
    chunk = path.consume(max_frames)

    return PointsResponse(
        total_frames,
        max_frames or total_frames,
        request.format,
        chunk.axes,
        _format_axes_points(chunk.lower),
        _format_axes_points(chunk.midpoints),
        _format_axes_points(chunk.upper),
        list(chunk.gap),
    )


@app.get("/smallest-step", response_model=SmallestStepResponse)
def smallest_step(
    spec: Mapping[str, Any] = Body(..., example=EXAMPLE_SPEC)
) -> SmallestStepResponse:
    """Calculate the smallest step in a scan, both absolutely and per-axis.

    Args:
        spec (Spec): The spec of the scan

    Returns:
        SmallestStepResponse: A description of the smallest steps in the spec
    """
    spec = Spec.deserialize(spec)
    dims = spec.calculate()  # Grab dimensions from spec
    path = Path(dims)  # Convert to a path

    absolute = _calc_smallest_step(list(path.midpoints.values()))
    per_axis = {axis: _calc_smallest_step(path.midpoints[axis]) for axis in path.axes()}

    return SmallestStepResponse(absolute, per_axis)


#
# Utility Functions
#


def _format_axes_points(
    axes_points: AxesPoints[str], format: PointsFormat
) -> Mapping[str, Points]:
    """Convert points to a requested format.

    Args:
        axes_points (AxesPoints[str]): The points to convert
        format (PointsFormat): The target format

    Raises:
        KeyError: If the function does not support the given format

    Returns:
        Mapping[str, Points]: A mapping of axis to formatted points.
    """
    if format is PointsFormat.FLOAT_LIST:
        return axes_points
    elif format is PointsFormat.STRING:
        return {axis: str(points) for axis, points in axes_points.items()}
    elif format is PointsFormat.BASE64_ENCODED:
        return {axis: str(points) for axis, points in axes_points.items()}
    else:
        raise KeyError(f"Unknown format: {format}")


def _reduce_frames(stack: List[Frames[str]], max_frames: int) -> Path:
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


def _calc_smallest_step(points: List[np.ndarray]) -> float:
    # Calc abs diffs of all axes
    absolute_diffs = [_abs_diffs(axis_midpoints) for axis_midpoints in points]
    # Return the smallest value (Aka. smallest step)
    return np.amin(np.linalg.norm(absolute_diffs, axis=0))


def _abs_diffs(array: np.ndarray) -> np.ndarray:
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

    uvicorn.run(app, port=port)
