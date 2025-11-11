# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Linspace
from scanspec.regions import Circle

cube = (
    Linspace("z", 1, 3, 3) * Linspace("y", 1, 3, 10) * ~Linspace("x", 0, 2, 10)
)
spec = cube & Circle("x", "y", 1, 2, 0.9)
plot_spec(spec)