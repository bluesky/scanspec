# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Linspace
from scanspec.regions import Circle

spec = Linspace("y", 3, 4, 3) * ~Linspace("x", 1, 2, 5) & Circle("x", "y", 1.5, 3.5, 0.6)
plot_spec(spec)