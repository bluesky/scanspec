# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Line, Static

spec = Line("y", 1, 2, 3).zip(Static.duration(0.1))
plot_spec(spec)