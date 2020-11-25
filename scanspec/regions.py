from typing import Any, Dict

import numpy as np
from pydantic.fields import Field

from scanspec.core import WithType


class Region(WithType):
    def mask(self, positions: Dict[Any, np.ndarray]) -> np.ndarray:
        # Return a mask of which positions are in the region
        raise NotImplementedError(self)


class Rectangle(Region):
    x_key: Any = Field(..., description="The key matching the x axis of the spec")
    x_min: float = Field(..., description="Minimum inclusive x value in the region")
    x_max: float = Field(..., description="Maximum inclusive x value in the region")
    y_key: Any = Field(..., description="The key matching the x axis of the spec")
    y_min: float = Field(..., description="Minimum inclusive y value in the region")
    y_max: float = Field(..., description="Maximum inclusive y value in the region")
    angle: float = Field(0, description="Clockwise rotation angle of the rectangle")

    def mask(self, positions: Dict[Any, np.ndarray]) -> np.ndarray:
        y = positions[self.y_key] - self.y_min
        x = positions[self.x_key] - self.x_min
        if self.angle != 0:
            rx = x * np.cos(self.angle) + y * np.sin(self.angle)
            ry = y * np.cos(self.angle) - x * np.sin(self.angle)
            x = rx
            y = ry
        mask_x = np.logical_and(x >= 0, x <= (self.x_max - self.x_min))
        mask_y = np.logical_and(y >= 0, y <= (self.y_max - self.y_min))
        return np.logical_and(mask_x, mask_y)

