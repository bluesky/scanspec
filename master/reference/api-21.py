# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Line
from scanspec.regions import Circle

cube = Line("z", 1, 3, 3) * Line("y", 1, 3, 10) * ~Line("x", 0, 2, 10)
spec = cube & Circle("x", "y", 1, 2, 0.9)
plot_spec(spec)