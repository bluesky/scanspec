# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Fly, Linspace, Squash

spec = Fly(Squash(Linspace("y", 1, 2, 3) * Linspace("x", 0, 1, 4)))
plot_spec(spec)