# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Line

spec = Line("y", 1, 3, 3) * ~Line("x", 3, 5, 5)
plot_spec(spec)