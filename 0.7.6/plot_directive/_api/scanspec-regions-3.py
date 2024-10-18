# Example Spec

from scanspec.plot import plot_spec
from scanspec.regions import Circle
from scanspec.specs import Line

grid = Line("y", 1, 3, 10) * ~Line("x", 0, 2, 10)
spec = grid & Circle("x", "y", 1, 2, 0.9)
plot_spec(spec)