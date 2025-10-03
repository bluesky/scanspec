# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Fly, Line, Static

spec = Fly(Line("y", 1, 2, 3).zip(Static("x", 3)))
plot_spec(spec)