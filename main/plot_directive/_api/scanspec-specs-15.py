# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Polygon, Fly

# A triangular region on axes "x" and "y", stepped by 0.2 units
# in both directions.
spec = Fly(
    Polygon(
        x_axis="x",
        y_axis="y",
        vertices=[(0, 0), (5, 0), (2.5, 4)],
        x_step=0.2,
        y_step=0.2,
        snake=True,
        vertical=False,
    )
)
plot_spec(spec)