from vpython import *

scene2 = canvas(title="P71: Coordinate Axes")

# Ось X (Красная)
box(pos=vector(0,0,0), size=vector(10, 0.2, 0.2), color=color.red)
# Ось Y (Зеленая)
box(pos=vector(0,0,0), size=vector(0.2, 10, 0.2), color=color.green)
# Ось Z (Синяя)
box(pos=vector(0,0,0), size=vector(0.2, 0.2, 10), color=color.blue)