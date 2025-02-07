# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Line

spec = Line("y", 3, 4, 3) * ~Line("x", 1, 2, 5)
plot_spec(spec)