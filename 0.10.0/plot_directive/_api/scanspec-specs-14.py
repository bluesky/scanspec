# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Ellipse, Fly

# An elliptical region centred at (0, 0) on axes "x" and "y",
# with 10x6 diameters and steps of 0.5 in both directions.
spec = Fly(
    Ellipse(
        "x", 0, 10, 0.5,
        "y", 0, 6,
        snake=True,
        vertical=False,
    )
)
plot_spec(spec)