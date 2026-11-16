# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Fly, Range

spec = Fly(Range.bounded("x", 1, 5, 2))
plot_spec(spec)