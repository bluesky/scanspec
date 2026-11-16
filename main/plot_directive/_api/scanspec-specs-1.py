# Example Spec

from scanspec.plot import plot_spec
import numpy as np

from scanspec.specs import Array, Fly

spec = Fly(Array("x", np.array([0.0, 1.0, 3.0])))
plot_spec(spec)