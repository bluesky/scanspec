# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Line
from scanspec.regions import Rectangle

grid = Line("y", 1, 3, 10) * ~Line("x", 0, 2, 10)
spec = grid & Rectangle("x", "y", 0, 1.1, 1.5, 2.1, 30)
plot_spec(spec)