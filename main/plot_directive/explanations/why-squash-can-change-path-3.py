from scanspec.specs import Linspace
from scanspec.plot import plot_spec

spec = Linspace("z", 0, 1, 3) * Linspace("y", 0, 1, 3) * ~Linspace("x", 0, 1, 3)
plot_spec(spec)