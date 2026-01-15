# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Fly, Spiral

spec = Fly(Spiral("x", 1, 10, 2.5, "y", 5, 50))
plot_spec(spec)