# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Fly, Linspace

spec = Fly(Linspace("y", 1, 3, 3) * ~Linspace("x", 3, 5, 5))
plot_spec(spec)