# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Fly, Line

spec = Fly(Line("y", 1, 2, 3) * Line("x", 3, 4, 12))
plot_spec(spec)