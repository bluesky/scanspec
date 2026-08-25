# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Fly, Linspace

spec = Fly(Linspace("x", 1, 3, 3).concat(Linspace("x", 4, 5, 5)))
plot_spec(spec)