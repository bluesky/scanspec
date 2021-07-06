from scanspec.specs import Line
from scanspec.regions import Circle
from scanspec.plot import plot_spec

spec = Line("y", 3, 4, 3) * ~Line("x", 1, 2, 5) & Circle("x", "y", 1.5, 3.5, 0.6) - Circle("x", "y", 1.4, 3.5, 0.2)
plot_spec(spec)