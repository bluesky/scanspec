from scanspec.specs import Line, Squash
from scanspec.plot import plot_spec

spec = Line("z", 0, 1, 3) * Squash(
    ~Line("y", 0, 1, 3) * Line("x", 0, 1, 3), check_path_changes=False
)
plot_spec(spec)