# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Fly, Spiral

spec = Fly(Spiral.spaced("x", "y", 0, 0, 10, 3))
plot_spec(spec)