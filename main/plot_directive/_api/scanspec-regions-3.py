# Example Spec

from scanspec.plot import plot_spec
from scanspec.regions import Circle
from scanspec.specs import Linspace

grid = Linspace("y", 1, 3, 10) * ~Linspace("x", 0, 2, 10)
spec = grid & Circle("x", "y", 1, 2, 0.9)
plot_spec(spec)