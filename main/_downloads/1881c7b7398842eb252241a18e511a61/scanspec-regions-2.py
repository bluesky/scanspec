# Example Spec

from scanspec.plot import plot_spec
from scanspec.regions import Polygon
from scanspec.specs import Linspace

grid = Linspace("y", 3, 8, 10) * ~Linspace("x", 1 ,8, 10)
spec = grid & Polygon("x", "y", [1.0, 6.0, 8.0, 2.0], [4.0, 10.0, 6.0, 1.0])
plot_spec(spec)