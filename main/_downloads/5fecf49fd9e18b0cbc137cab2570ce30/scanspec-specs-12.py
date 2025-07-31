# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Spiral

spec = Spiral("x", "y", 1, 5, 10, 50, 30)
plot_spec(spec)