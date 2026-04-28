# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Linspace, Ellipse

spec = Linspace("z", 1, 3, 3) * Ellipse("x", 1, 01.8, 0.2, "y", 2, snake=True)
plot_spec(spec)