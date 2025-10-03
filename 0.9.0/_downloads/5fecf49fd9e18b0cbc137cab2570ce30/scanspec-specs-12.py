# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Fly, Line

spec = Fly(Line.bounded("x", 1, 2, 5))
plot_spec(spec)