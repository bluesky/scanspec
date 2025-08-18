# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Fly, Line, Product

spec = Fly(Product(2, ~Line.bounded("x", 3, 4, 1), gap=False))
plot_spec(spec)