# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Fly, Line

spec = Fly(2 * ~Line.bounded("x", 3, 4, 1))
plot_spec(spec)