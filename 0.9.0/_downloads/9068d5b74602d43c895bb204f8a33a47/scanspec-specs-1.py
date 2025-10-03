# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Line

spec = 0.1 @ Line("x", 1, 2, 3)
plot_spec(spec)