# Example Spec

from scanspec.plot import plot_spec
from scanspec.regions import Circle
from scanspec.specs import Line

spec = Line("y", 1, 3, 3) * Line("x", 3, 5, 5) & Circle("x", "y", 4, 2, 1.2)
plot_spec(spec)