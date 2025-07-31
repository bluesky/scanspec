# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Line, step

spec = step(Line("x", 1, 2, 3), 0.1)
plot_spec(spec)