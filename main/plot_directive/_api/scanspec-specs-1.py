# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Linspace

spec = 0.1 @ Linspace("x", 1, 2, 3)
plot_spec(spec)