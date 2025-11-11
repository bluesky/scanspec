# Example Spec

from scanspec.plot import plot_spec
from scanspec.regions import Ellipse
from scanspec.specs import Linspace

grid = Linspace("y", 3, 8, 10) * ~Linspace("x", 1 ,8, 10)
spec = grid & Ellipse("x", "y", 5, 5, 2, 3, 75)
plot_spec(spec)