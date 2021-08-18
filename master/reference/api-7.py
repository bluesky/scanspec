# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Concat, Line

spec = Concat(Line("x", 1, 3, 3), Line("x", 4, 5, 5))
plot_spec(spec)