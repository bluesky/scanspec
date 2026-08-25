# Example Spec

from scanspec.plot import plot_spec
from scanspec.specs import Fly, Linspace, Product

spec = Fly(Product(2, ~Linspace.bounded("x", 3, 4, 1), gap=False))
plot_spec(spec)