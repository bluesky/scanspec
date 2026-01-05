# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Fly, Range

spec = Fly(Range("x", 1, 2, 0.25))
plot_spec(spec)