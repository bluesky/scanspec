# Example Spec

from scanspec.plot import plot_spec
from scanspec.regions import Circle
from scanspec.specs import Fly, Linspace

region = Circle("x", "y", 4, 2, 1.2)
spec = Fly(Linspace("y", 1, 3, 3) * Linspace("x", 3, 5, 5) & region)
plot_spec(spec)