# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Linspace

spec = Linspace("y", 3, 4, 5).zip(Linspace("x", 1, 2, 5))
plot_spec(spec)