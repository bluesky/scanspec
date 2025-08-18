# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Fly, Line, Squash

spec = Fly(Squash(Line("y", 1, 2, 3) * Line("x", 0, 1, 4)))
plot_spec(spec)