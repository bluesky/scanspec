# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Line
from scanspec.regions import Circle

spec = Line("x", 0, 10, 5) * Line("y", 0, 10, 5) & Circle("x", "y", 5, 5, 3)
plot_spec(spec)