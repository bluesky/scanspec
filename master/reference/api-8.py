# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Line, Squash

spec = Squash(Line("y", 1, 2, 3) * Line("x", 0, 1, 4))
plot_spec(spec)