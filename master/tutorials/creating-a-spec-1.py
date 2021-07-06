from scanspec.specs import Line
from scanspec.plot import plot_spec

spec = Line("x", 1, 2, 5)
plot_spec(spec)