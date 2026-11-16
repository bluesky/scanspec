# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Fly, Linspace

spec = Fly(Linspace("y", 1, 2, 3) * Linspace("x", 3, 4, 12))
plot_spec(spec)