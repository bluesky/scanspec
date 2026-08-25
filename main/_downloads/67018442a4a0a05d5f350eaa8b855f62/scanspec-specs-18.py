# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Fly, Linspace

spec = Fly(
    Linspace("z", 1, 2, 3) * Linspace("y", 3, 4, 5).zip(Linspace("x", 4, 5, 5))
)
plot_spec(spec)