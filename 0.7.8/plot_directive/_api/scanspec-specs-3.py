# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Line, Repeat

spec = Repeat(2, gap=False) * ~Line.bounded("x", 3, 4, 1)
plot_spec(spec)