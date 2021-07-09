# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Line
from scanspec.regions import Circle

grid = Line("y", 1, 3, 10) * ~Line("x", 0, 2, 10)
spec = grid & Circle("x", "y", 1, 2, 0.9)
plot_spec(spec)