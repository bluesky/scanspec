from typing import Any

import numpy as np
from specs import Line

from scanspec.core import (
    Frames,
    Path,
)


def reduce_frames(stack: list[Frames[str]], max_frames: int) -> Path:
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


def validate_spec(spec: Line) -> Any:
    """A query used to confirm whether or not the Spec will produce a viable scan."""
    # TODO apischema will do all the validation for us
    return spec.serialize()
