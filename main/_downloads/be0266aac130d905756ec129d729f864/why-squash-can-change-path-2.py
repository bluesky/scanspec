from scanspec.specs import Linspace, Squash
from scanspec.plot import plot_spec

spec = Linspace("z", 0, 1, 3) * Squash(
    ~Linspace("y", 0, 1, 3) * Linspace("x", 0, 1, 3), check_path_changes=False
)
plot_spec(spec)