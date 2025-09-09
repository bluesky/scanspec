# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Fly, Line

spec = Fly(Line("z", 1, 2, 3) * Line("y", 3, 4, 5).zip(Line("x", 4, 5, 5)))
plot_spec(spec)