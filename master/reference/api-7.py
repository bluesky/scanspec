# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Line

spec = Line("x", 1, 3, 3).concat(Line("x", 4, 5, 5))
plot_spec(spec)