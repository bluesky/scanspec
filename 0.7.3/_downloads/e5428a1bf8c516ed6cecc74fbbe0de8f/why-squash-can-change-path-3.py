from scanspec.specs import Line
from scanspec.plot import plot_spec

spec = Line("z", 0, 1, 3) * Line("y", 0, 1, 3) * ~Line("x", 0, 1, 3)
plot_spec(spec)