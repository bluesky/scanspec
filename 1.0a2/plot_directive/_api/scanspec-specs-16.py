# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Fly, Line

spec = Fly(Line("x", 1, 2, 3))
plot_spec(spec)