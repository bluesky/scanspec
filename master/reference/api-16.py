# Example Spec

from scanspec.plot import plot_spec
from scanspec.plot import plot_spec
from scanspec.specs import Line
from scanspec.regions import Ellipse

#Ellipse parameters
x_axis = "x"
y_axis = "y"
x_middle = 5
y_middle = 5
x_radius = 2
y_radius = 3
angle = 75

grid = Line(y_axis, 3, 8, 10) * ~Line(x_axis, 1 ,8, 10)
spec = grid & Ellipse(x_axis, y_axis, x_middle, y_middle, x_radius,
y_radius, angle)
plot_spec(spec)
plot_spec(spec)