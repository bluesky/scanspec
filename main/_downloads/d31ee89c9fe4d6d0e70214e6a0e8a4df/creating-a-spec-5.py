# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Ellipse

spec = Ellipse("x", 0, 1, 0.1, "y", 5, 10, 0.5, snake=True)
plot_spec(spec)