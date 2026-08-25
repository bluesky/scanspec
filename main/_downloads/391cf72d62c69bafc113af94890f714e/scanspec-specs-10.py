# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Fly, Linspace

spec = Fly(2 * ~Linspace.bounded("x", 3, 4, 1))
plot_spec(spec)